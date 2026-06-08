from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weather_strategy.models import ConsensusValue, ForecastDistribution, TemperatureBucket
from weather_strategy.observations import ObservedHigh


@dataclass(frozen=True)
class ForecastSettings:
    bias_correction_f: float = 0.0
    sigma_floor_f: float = 2.5


@dataclass(frozen=True)
class SameDayPathSettings:
    adjustment_start_hour_local: float = 11.0


class ForecastEngine:
    def __init__(self, settings: Optional[ForecastSettings] = None):
        self.settings = settings or ForecastSettings()

    def probability_by_bucket(
        self,
        distribution: ForecastDistribution,
        buckets: tuple[TemperatureBucket, ...],
    ) -> Mapping[str, float]:
        if not distribution.samples_f:
            raise ValueError("Cannot score buckets without forecast samples")
        raw = {
            bucket.label: self._bucket_probability(distribution.samples_f, bucket)
            for bucket in buckets
        }
        if len(buckets) == 1:
            return raw
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("Bucket probabilities sum to zero; check market bucket coverage")
        return {label: probability / total for label, probability in raw.items()}

    def _bucket_probability(self, samples_f: tuple[float, ...], bucket: TemperatureBucket) -> float:
        sigma = self.settings.sigma_floor_f
        adjusted = tuple(sample + self.settings.bias_correction_f for sample in samples_f)
        return sum(_normal_interval_probability(sample, sigma, bucket.lower_f, bucket.upper_f) for sample in adjusted) / len(adjusted)


class ProbabilityModel:
    name = "base"

    def probability_by_bucket(self, distribution: ForecastDistribution, buckets: tuple[TemperatureBucket, ...]) -> Mapping[str, float]:
        raise NotImplementedError


class EmpiricalEnsembleModel(ProbabilityModel):
    name = "empirical"

    def probability_by_bucket(self, distribution: ForecastDistribution, buckets: tuple[TemperatureBucket, ...]) -> Mapping[str, float]:
        if len(distribution.samples_f) < 5:
            return ForecastEngine(ForecastSettings(sigma_floor_f=2.5)).probability_by_bucket(distribution, buckets)
        probabilities = {}
        for bucket in buckets:
            count = sum(1 for sample in distribution.samples_f if bucket.contains(sample))
            probabilities[bucket.label] = count / len(distribution.samples_f)
        if len(buckets) == 1:
            return probabilities
        total = sum(probabilities.values())
        if total <= 0:
            return {bucket.label: 0.0 for bucket in buckets}
        return {label: probability / total for label, probability in probabilities.items()}


class KernelSmoothingModel(ProbabilityModel):
    def __init__(self, sigma_floor_f: float = 2.5, name: Optional[str] = None):
        self.name = name or f"kernel_sigma_{sigma_floor_f:g}"
        self.engine = ForecastEngine(ForecastSettings(sigma_floor_f=sigma_floor_f))

    def probability_by_bucket(self, distribution: ForecastDistribution, buckets: tuple[TemperatureBucket, ...]) -> Mapping[str, float]:
        return self.engine.probability_by_bucket(distribution, buckets)


class ParametricNormalModel(ProbabilityModel):
    def __init__(self, sigma_floor_f: float = 2.0, sigma_multiplier: float = 1.0, name: Optional[str] = None):
        self.name = name or f"normal_sigma_{sigma_floor_f:g}_x{sigma_multiplier:g}"
        self.sigma_floor_f = sigma_floor_f
        self.sigma_multiplier = sigma_multiplier

    def probability_by_bucket(self, distribution: ForecastDistribution, buckets: tuple[TemperatureBucket, ...]) -> Mapping[str, float]:
        mean = distribution.mean_f
        if len(distribution.samples_f) >= 2:
            sigma = statistics.pstdev(distribution.samples_f)
        else:
            sigma = 0.0
        sigma = max(self.sigma_floor_f, sigma * self.sigma_multiplier)
        raw = {bucket.label: _normal_interval_probability(mean, sigma, bucket.lower_f, bucket.upper_f) for bucket in buckets}
        if len(buckets) == 1:
            return raw
        total = sum(raw.values())
        if total <= 0:
            return {bucket.label: 0.0 for bucket in buckets}
        return {label: probability / total for label, probability in raw.items()}


class FeatureAwareNormalModel(ProbabilityModel):
    name = "feature_aware"

    def __init__(self, sigma_floor_f: float = 3.0):
        self.sigma_floor_f = sigma_floor_f

    def probability_by_bucket(self, distribution: ForecastDistribution, buckets: tuple[TemperatureBucket, ...]) -> Mapping[str, float]:
        features = distribution.model_metadata.get("features") or {}
        mean = distribution.mean_f + self._mean_adjustment(features)
        sigma = self._sigma(distribution, features)
        raw = {bucket.label: _normal_interval_probability(mean, sigma, bucket.lower_f, bucket.upper_f) for bucket in buckets}
        if len(buckets) == 1:
            return raw
        total = sum(raw.values())
        if total <= 0:
            return {bucket.label: 0.0 for bucket in buckets}
        return {label: probability / total for label, probability in raw.items()}

    def _mean_adjustment(self, features: Mapping[str, float]) -> float:
        adjustment = 0.0
        cloud = features.get("cloud_cover_mean")
        radiation = features.get("shortwave_radiation_sum")
        rain_hours = features.get("precipitation_hours")
        precip = features.get("precipitation_sum") or features.get("hourly_precipitation_sum")
        wind = features.get("wind_speed_10m_max") or features.get("hourly_wind_speed_10m_max")
        humidity = features.get("relative_humidity_2m_mean")

        if cloud is not None and cloud > 70:
            adjustment -= 1.0
        if radiation is not None and radiation > 22:
            adjustment += 0.7
        if rain_hours is not None and rain_hours >= 4:
            adjustment -= 1.0
        if precip is not None and precip >= 0.15:
            adjustment -= 0.7
        if wind is not None and wind >= 18:
            adjustment -= 0.3
        if humidity is not None and humidity >= 80:
            adjustment -= 0.3
        return adjustment

    def _sigma(self, distribution: ForecastDistribution, features: Mapping[str, float]) -> float:
        if len(distribution.samples_f) >= 2:
            sigma = statistics.pstdev(distribution.samples_f)
        else:
            sigma = 0.0
        sigma = max(self.sigma_floor_f, sigma)
        pressure_range = features.get("pressure_msl_range")
        precip = features.get("precipitation_sum") or features.get("hourly_precipitation_sum")
        cloud = features.get("cloud_cover_mean")
        wind = features.get("wind_gusts_10m_max")
        if pressure_range is not None and pressure_range >= 4:
            sigma += 0.7
        if precip is not None and precip >= 0.10:
            sigma += 0.5
        if cloud is not None and cloud >= 75:
            sigma += 0.4
        if wind is not None and wind >= 25:
            sigma += 0.5
        return sigma


class ConsensusForecastEngine:
    def __init__(self, models: Optional[tuple[ProbabilityModel, ...]] = None, same_day_path: Optional[SameDayPathSettings] = None):
        self.models = models or (
            EmpiricalEnsembleModel(),
            KernelSmoothingModel(sigma_floor_f=1.5, name="kernel_tight"),
            KernelSmoothingModel(sigma_floor_f=3.0, name="kernel_wide"),
            ParametricNormalModel(sigma_floor_f=2.0, sigma_multiplier=1.0, name="normal_parametric"),
            ParametricNormalModel(sigma_floor_f=4.0, sigma_multiplier=1.75, name="normal_conservative"),
            FeatureAwareNormalModel(sigma_floor_f=3.0),
        )
        self.same_day_path = same_day_path or SameDayPathSettings()
        self.source_weights = {
            "open_meteo_ecmwf": 1.35,
            "open_meteo_best_match": 1.15,
            "open_meteo_gfs_hrrr": 1.10,
            "open_meteo_gfs_global": 0.95,
            "open_meteo_gfs_graphcast": 1.10,
            "open_meteo_ensemble_gfs_seamless": 1.20,
            "open_meteo_ensemble_gfs025": 1.15,
            "open_meteo_ensemble_ecmwf_ifs025": 1.35,
            "fixture": 1.0,
        }

    def consensus_by_bucket(
        self,
        distributions: tuple[ForecastDistribution, ...],
        buckets: tuple[TemperatureBucket, ...],
    ) -> Mapping[str, ConsensusValue]:
        model_values = {bucket.label: {} for bucket in buckets}
        for distribution in distributions:
            for model in self.models:
                probabilities = model.probability_by_bucket(distribution, buckets)
                for bucket in buckets:
                    value = probabilities.get(bucket.label)
                    if value is None:
                        continue
                    key = f"{distribution.source}.{model.name}"
                    model_values[bucket.label][key] = max(0.0, min(1.0, float(value)))

        consensus = {}
        for bucket in buckets:
            probabilities = model_values[bucket.label]
            if not probabilities:
                continue
            source_probabilities = _source_probability_views(probabilities)
            values = list(source_probabilities.values())
            weighted_total = 0.0
            weight_sum = 0.0
            for source_name, probability in source_probabilities.items():
                weight = self.source_weights.get(source_name, 1.0)
                weighted_total += probability * weight
                weight_sum += weight
            consensus[bucket.label] = ConsensusValue(
                bucket_label=bucket.label,
                fair_value=weighted_total / weight_sum if weight_sum else sum(values) / len(values),
                model_probabilities=probabilities,
                model_count=len(source_probabilities),
                probability_stdev=statistics.pstdev(values) if len(values) >= 2 else 0.0,
            )
        return consensus

    def apply_observed_high(
        self,
        consensus_values: Mapping[str, ConsensusValue],
        buckets: tuple[TemperatureBucket, ...],
        observed: Optional[ObservedHigh],
        now: Optional[datetime] = None,
    ) -> Mapping[str, ConsensusValue]:
        if observed is None or observed.max_temperature_f is None:
            return consensus_values
        observed_high = observed.max_temperature_f
        certain_label = _certain_label_from_observed_high(buckets, observed_high)
        impossible_labels = {
            bucket.label
            for bucket in buckets
            if bucket.upper_f is not None and observed_high > bucket.upper_f
        }
        path_probabilities = _same_day_path_probabilities(buckets, observed, now, self.same_day_path)
        if certain_label is None and not impossible_labels and path_probabilities is None:
            return {
                label: _with_observation(value, observed, observation_adjusted=False)
                for label, value in consensus_values.items()
            }

        model_names = sorted({name for value in consensus_values.values() for name in value.model_probabilities})
        adjusted_by_label: dict[str, dict[str, float]] = {bucket.label: {} for bucket in buckets}
        for model_name in model_names:
            raw = {bucket.label: consensus_values.get(bucket.label).model_probabilities.get(model_name, 0.0) if consensus_values.get(bucket.label) else 0.0 for bucket in buckets}
            if certain_label is not None:
                adjusted = {label: 1.0 if label == certain_label else 0.0 for label in raw}
            else:
                adjusted = {label: (0.0 if label in impossible_labels else probability) for label, probability in raw.items()}
                if path_probabilities is not None:
                    adjusted = _apply_same_day_path_caps(adjusted, buckets, observed_high, path_probabilities)
                if len(buckets) > 1:
                    remaining = sum(adjusted.values())
                    if remaining > 0:
                        adjusted = {label: probability / remaining for label, probability in adjusted.items()}
            for label, probability in adjusted.items():
                adjusted_by_label[label][model_name] = probability

        adjusted_consensus = {}
        for bucket in buckets:
            original = consensus_values.get(bucket.label)
            probabilities = adjusted_by_label[bucket.label]
            if not probabilities:
                continue
            source_probabilities = _source_probability_views(probabilities)
            values = list(source_probabilities.values())
            weighted_total = 0.0
            weight_sum = 0.0
            for source_name, probability in source_probabilities.items():
                weight = self.source_weights.get(source_name, 1.0)
                weighted_total += probability * weight
                weight_sum += weight
            adjusted_consensus[bucket.label] = ConsensusValue(
                bucket_label=bucket.label,
                fair_value=weighted_total / weight_sum if weight_sum else sum(values) / len(values),
                model_probabilities=probabilities,
                model_count=original.model_count if original else len(source_probabilities),
                probability_stdev=statistics.pstdev(values) if len(values) >= 2 else 0.0,
                observed_high_f=round(observed_high, 2),
                observation_source=observed.source,
                observation_final=observed.is_final,
                observation_adjusted=True,
            )
        return adjusted_consensus


def _normal_interval_probability(mean: float, sigma: float, lower: Optional[float], upper: Optional[float]) -> float:
    lower_cdf = 0.0 if lower is None else _normal_cdf((lower - mean) / sigma)
    upper_cdf = 1.0 if upper is None else _normal_cdf((upper - mean) / sigma)
    return max(0.0, upper_cdf - lower_cdf)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _certain_label_from_observed_high(buckets: tuple[TemperatureBucket, ...], observed_high_f: float) -> Optional[str]:
    for bucket in buckets:
        if bucket.lower_f is not None and bucket.upper_f is None and observed_high_f >= bucket.lower_f:
            return bucket.label
    return None


def _same_day_path_probabilities(
    buckets: tuple[TemperatureBucket, ...],
    observed: ObservedHigh,
    now: Optional[datetime],
    settings: SameDayPathSettings,
) -> Optional[dict[str, float]]:
    if observed.is_final:
        return None
    try:
        timezone_info = ZoneInfo(observed.city.timezone)
    except ZoneInfoNotFoundError:
        timezone_info = timezone.utc
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local_now = current.astimezone(timezone_info)
    if observed.target_date != local_now.date():
        return None
    local_hour = local_now.hour + local_now.minute / 60
    if local_hour < settings.adjustment_start_hour_local:
        return None

    observed_high = observed.max_temperature_f
    return {
        bucket.label: _same_day_bucket_probability(bucket, observed_high, local_hour)
        for bucket in buckets
    }


def _same_day_bucket_probability(bucket: TemperatureBucket, observed_high_f: float, local_hour: float) -> float:
    if bucket.upper_f is not None and observed_high_f > bucket.upper_f:
        return 0.0
    if bucket.contains(observed_high_f):
        if bucket.upper_f is None:
            return 1.0
        return max(0.0, min(1.0, 1.0 - _remaining_exceedance_probability(bucket.upper_f + 0.01, observed_high_f, local_hour)))
    lower_probability = 1.0 if bucket.lower_f is None or bucket.lower_f <= observed_high_f else _remaining_exceedance_probability(bucket.lower_f, observed_high_f, local_hour)
    upper_probability = 0.0 if bucket.upper_f is None else _remaining_exceedance_probability(bucket.upper_f + 0.01, observed_high_f, local_hour)
    return max(0.0, min(1.0, lower_probability - upper_probability))


def _remaining_exceedance_probability(threshold_f: float, observed_high_f: float, local_hour: float) -> float:
    if threshold_f <= observed_high_f:
        return 1.0
    mean_upside_f, sigma_f = _remaining_upside_parameters(local_hour)
    z = (threshold_f - observed_high_f - mean_upside_f) / sigma_f
    return max(0.0, min(1.0, 1.0 - _normal_cdf(z)))


def _remaining_upside_parameters(local_hour: float) -> tuple[float, float]:
    if local_hour < 12.0:
        return 5.0, 3.0
    if local_hour < 13.5:
        return 3.5, 2.2
    if local_hour < 14.5:
        return 2.0, 1.5
    if local_hour < 15.5:
        return 0.9, 0.9
    if local_hour < 16.5:
        return 0.45, 0.55
    if local_hour < 18.0:
        return 0.18, 0.3
    return 0.08, 0.18


def _apply_same_day_path_caps(
    probabilities: Mapping[str, float],
    buckets: tuple[TemperatureBucket, ...],
    observed_high_f: float,
    path_probabilities: Mapping[str, float],
) -> dict[str, float]:
    adjusted = {}
    for bucket in buckets:
        prior = probabilities.get(bucket.label, 0.0)
        path_probability = path_probabilities.get(bucket.label, prior)
        if bucket.contains(observed_high_f):
            adjusted[bucket.label] = max(prior, path_probability)
        else:
            adjusted[bucket.label] = min(prior, path_probability)
    return adjusted


def _with_observation(value: ConsensusValue, observed: ObservedHigh, observation_adjusted: bool) -> ConsensusValue:
    return ConsensusValue(
        bucket_label=value.bucket_label,
        fair_value=value.fair_value,
        model_probabilities=value.model_probabilities,
        model_count=value.model_count,
        probability_stdev=value.probability_stdev,
        observed_high_f=round(observed.max_temperature_f, 2),
        observation_source=observed.source,
        observation_final=observed.is_final,
        observation_adjusted=observation_adjusted,
    )


def _source_probability_views(model_probabilities: Mapping[str, float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, probability in model_probabilities.items():
        source = key.rsplit(".", 1)[0] if "." in key else key
        grouped.setdefault(source, []).append(float(probability))
    return {
        source: sum(values) / len(values)
        for source, values in grouped.items()
        if values
    }
