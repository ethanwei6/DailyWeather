from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

from weather_strategy.http import HttpClient
from weather_strategy.models import CityConfig, ForecastDistribution


class OpenMeteoClient:
    """Fetch ensemble-style daily high forecasts.

    Open-Meteo's ensemble endpoint returns individual members for some model
    selections and aggregate fields for others. This parser accepts both shapes.
    """

    BASE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    GFS_URL = "https://api.open-meteo.com/v1/gfs"
    ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"

    def __init__(self, http: Optional[HttpClient] = None, models: str = "gfs05_ens"):
        self.http = http or HttpClient()
        self.models = models

    def fetch_daily_high_distribution(self, city: CityConfig, target_date: date) -> ForecastDistribution:
        distributions = self.fetch_daily_high_sources(city, target_date)
        if not distributions:
            raise ValueError(f"Open-Meteo returned no daily high sources for {city.display_name} on {target_date}")
        return distributions[0]

    def fetch_daily_high_sources(self, city: CityConfig, target_date: date) -> tuple[ForecastDistribution, ...]:
        sources = []
        for source_name, url, params in self._source_requests(city):
            try:
                payload = self.http.get_json(url, params=params)
                samples = extract_daily_temperature_samples(payload, target_date)
            except (RuntimeError, ValueError):
                try:
                    fallback_payload = self.http.get_json(url, params=self._high_only_params(city))
                    samples = extract_daily_temperature_samples(fallback_payload, target_date)
                    payload = fallback_payload
                except (RuntimeError, ValueError):
                    continue
            if not samples:
                continue
            sources.append(
                ForecastDistribution(
                    city=city,
                    target_date=target_date,
                    samples_f=tuple(samples),
                    generated_at=datetime.now(timezone.utc),
                    source=source_name,
                    model_metadata={"url": url, "features": extract_weather_features(payload, target_date)},
                )
            )
        ensemble = self._fetch_ensemble_source(city, target_date)
        if ensemble is not None:
            sources.append(ensemble)
        return tuple(_dedupe_temperature_sources(sources))

    def _source_requests(self, city: CityConfig) -> tuple[tuple[str, str, dict[str, object]], ...]:
        base_params = {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "daily": ",".join(
                (
                    "temperature_2m_max",
                    "precipitation_sum",
                    "precipitation_hours",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                    "shortwave_radiation_sum",
                )
            ),
            "hourly": ",".join(
                (
                    "temperature_2m",
                    "relative_humidity_2m",
                    "dew_point_2m",
                    "apparent_temperature",
                    "precipitation",
                    "cloud_cover",
                    "wind_speed_10m",
                    "pressure_msl",
                    "shortwave_radiation",
                )
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": city.timezone,
        }
        return (
            ("open_meteo_best_match", self.FORECAST_URL, dict(base_params)),
            ("open_meteo_gfs_best_match", self.GFS_URL, dict(base_params)),
            ("open_meteo_gfs_global", self.GFS_URL, {**base_params, "models": "gfs_global"}),
            ("open_meteo_gfs_graphcast", self.GFS_URL, {**base_params, "models": "gfs_graphcast025"}),
            ("open_meteo_ecmwf", self.ECMWF_URL, dict(base_params)),
        )

    def _high_only_params(self, city: CityConfig) -> dict[str, object]:
        return {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": city.timezone,
        }

    def _fetch_ensemble_source(self, city: CityConfig, target_date: date) -> Optional[ForecastDistribution]:
        for model_name in ("gfs_seamless", "gfs025", "ecmwf_ifs025"):
            try:
                payload = self.http.get_json(
                    self.BASE_URL,
                    params={
                        "latitude": city.latitude,
                        "longitude": city.longitude,
                        "models": model_name,
                        "daily": "temperature_2m_max",
                        "temperature_unit": "fahrenheit",
                        "timezone": city.timezone,
                    },
                )
                samples = extract_daily_temperature_samples(payload, target_date)
            except (RuntimeError, ValueError):
                continue
            if samples:
                return ForecastDistribution(
                    city=city,
                    target_date=target_date,
                    samples_f=tuple(samples),
                    generated_at=datetime.now(timezone.utc),
                    source=f"open_meteo_ensemble_{model_name}",
                    model_metadata={"models": model_name, "features": extract_weather_features(payload, target_date)},
                )
        return None

    def fetch_daily_high_distribution_legacy(self, city: CityConfig, target_date: date) -> ForecastDistribution:
        source = "open_meteo_ensemble"
        metadata = {"models": self.models}
        try:
            payload = self.http.get_json(
                self.BASE_URL,
                params={
                    "latitude": city.latitude,
                    "longitude": city.longitude,
                    "models": self.models,
                    "daily": "temperature_2m_max",
                    "temperature_unit": "fahrenheit",
                    "timezone": city.timezone,
                },
            )
        except RuntimeError as error:
            payload = self.http.get_json(
                self.FORECAST_URL,
                params={
                    "latitude": city.latitude,
                    "longitude": city.longitude,
                    "daily": "temperature_2m_max",
                    "temperature_unit": "fahrenheit",
                    "timezone": city.timezone,
                },
            )
            source = "open_meteo_forecast_fallback"
            metadata = {"ensemble_error": str(error)[:300]}
        samples = extract_daily_temperature_samples(payload, target_date)
        if not samples:
            raise ValueError(f"Open-Meteo returned no daily high samples for {city.display_name} on {target_date}")
        return ForecastDistribution(
            city=city,
            target_date=target_date,
            samples_f=tuple(samples),
            generated_at=datetime.now(timezone.utc),
            source=source,
            model_metadata=metadata,
        )


class NWSClient:
    BASE_URL = "https://api.weather.gov"

    def __init__(self, http: Optional[HttpClient] = None):
        self.http = http or HttpClient(user_agent="weather-polymarket-strategy/0.1 contact=local")

    def fetch_latest_station_temperature_f(self, station_id: str) -> Optional[float]:
        payload = self.http.get_json(f"{self.BASE_URL}/stations/{station_id}/observations/latest")
        properties = payload.get("properties", {})
        temp_c = properties.get("temperature", {}).get("value")
        if temp_c is None:
            return None
        return temp_c * 9 / 5 + 32


def extract_daily_temperature_samples(payload: dict[str, Any], target_date: date) -> list[float]:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    try:
        index = [date.fromisoformat(str(item)) for item in times].index(target_date)
    except ValueError:
        return []

    samples = []
    for key, values in daily.items():
        if key == "time" or not key.startswith("temperature_2m_max"):
            continue
        samples.extend(_value_at_index(values, index))

    # Some API responses expose a single aggregate. Keep it as one sample; the
    # forecast model will apply a minimum uncertainty floor.
    return [value for value in samples if value is not None]


def _dedupe_temperature_sources(distributions: list[ForecastDistribution]) -> list[ForecastDistribution]:
    deduped = []
    seen = set()
    for distribution in distributions:
        signature = tuple(round(value, 2) for value in distribution.samples_f)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(distribution)
    return deduped


def extract_weather_features(payload: dict[str, Any], target_date: date) -> dict[str, float]:
    features: dict[str, float] = {}
    daily = payload.get("daily") or {}
    daily_times = daily.get("time") or []
    daily_index = _date_index(daily_times, target_date)
    if daily_index is not None:
        for key in (
            "precipitation_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "shortwave_radiation_sum",
        ):
            value = _single_value_at_index(daily.get(key), daily_index)
            if value is not None:
                features[key] = value

    hourly = payload.get("hourly") or {}
    hourly_times = hourly.get("time") or []
    indices = [
        index
        for index, value in enumerate(hourly_times)
        if str(value).startswith(target_date.isoformat())
    ]
    if indices:
        _add_hourly_feature(features, hourly, indices, "temperature_2m", max, "hourly_temperature_2m_max")
        _add_hourly_feature(features, hourly, indices, "apparent_temperature", max, "hourly_apparent_temperature_max")
        _add_hourly_feature(features, hourly, indices, "relative_humidity_2m", _mean, "relative_humidity_2m_mean")
        _add_hourly_feature(features, hourly, indices, "dew_point_2m", max, "dew_point_2m_max")
        _add_hourly_feature(features, hourly, indices, "cloud_cover", _mean, "cloud_cover_mean")
        _add_hourly_feature(features, hourly, indices, "wind_speed_10m", max, "hourly_wind_speed_10m_max")
        _add_hourly_feature(features, hourly, indices, "pressure_msl", _pressure_range, "pressure_msl_range")
        _add_hourly_feature(features, hourly, indices, "shortwave_radiation", _mean, "shortwave_radiation_mean")
        precipitation_values = _hourly_values(hourly, indices, "precipitation")
        if precipitation_values:
            features["hourly_precipitation_sum"] = sum(precipitation_values)

    return features


def _date_index(values: Any, target_date: date) -> Optional[int]:
    if not isinstance(values, list):
        return None
    try:
        return [date.fromisoformat(str(item)) for item in values].index(target_date)
    except ValueError:
        return None


def _single_value_at_index(values: Any, index: int) -> Optional[float]:
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    if isinstance(value, list):
        numeric = [float(item) for item in value if item is not None]
        return _mean(numeric) if numeric else None
    return float(value)


def _hourly_values(hourly: dict[str, Any], indices: list[int], key: str) -> list[float]:
    values = hourly.get(key)
    if not isinstance(values, list):
        return []
    numeric = []
    for index in indices:
        if index < len(values) and values[index] is not None:
            numeric.append(float(values[index]))
    return numeric


def _add_hourly_feature(features: dict[str, float], hourly: dict[str, Any], indices: list[int], key: str, reducer, output_key: str) -> None:
    values = _hourly_values(hourly, indices, key)
    if values:
        features[output_key] = float(reducer(values))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _pressure_range(values: list[float]) -> float:
    return max(values) - min(values)


def _value_at_index(values: Any, index: int) -> Iterable[float]:
    if not isinstance(values, list) or index >= len(values):
        return []
    value = values[index]
    if isinstance(value, list):
        return [float(item) for item in value if item is not None]
    if value is None:
        return []
    return [float(value)]
