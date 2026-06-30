from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from weather_strategy.cities import DEFAULT_CITIES
from weather_strategy.forecast import (
    ConsensusForecastEngine,
    EmpiricalEnsembleModel,
    ForecastEngine,
    ForecastSettings,
    HourlyCurveMaxModel,
    ProbabilityCalibration,
)
from weather_strategy.models import ForecastDistribution, TemperatureBucket
from weather_strategy.observations import ObservedHigh, ObservedHighClient
from weather_strategy.weather import extract_daily_temperature_samples


class ForecastTest(unittest.TestCase):
    def test_extract_daily_temperature_samples_accepts_member_fields(self) -> None:
        payload = {
            "daily": {
                "time": ["2026-06-04", "2026-06-05"],
                "temperature_2m_max_member01": [80, 82],
                "temperature_2m_max_member02": [81, 84],
            }
        }
        self.assertEqual(extract_daily_temperature_samples(payload, date(2026, 6, 5)), [82.0, 84.0])

    def test_probability_by_bucket_normalizes_over_buckets(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(78.0, 80.0, 82.0),
            generated_at=datetime.now(timezone.utc),
            source="test",
        )
        buckets = (
            TemperatureBucket("Under 80", None, 79.99),
            TemperatureBucket("80-84", 80, 84),
            TemperatureBucket("85 or above", 85, None),
        )
        probabilities = ForecastEngine(ForecastSettings(sigma_floor_f=1.0)).probability_by_bucket(distribution, buckets)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["80-84"], probabilities["85 or above"])

    def test_single_binary_bucket_is_not_normalized_to_one(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(70.0, 71.0, 72.0),
            generated_at=datetime.now(timezone.utc),
            source="test",
        )
        bucket = (TemperatureBucket("80 or above", 80, None),)
        probabilities = ForecastEngine(ForecastSettings(sigma_floor_f=1.0)).probability_by_bucket(distribution, bucket)
        self.assertLess(probabilities["80 or above"], 0.01)

    def test_consensus_counts_independent_sources_not_model_transforms(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(80.0, 81.0, 82.0, 83.0),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
        )
        bucket = (TemperatureBucket("80 or above", 80, None),)
        consensus = ConsensusForecastEngine().consensus_by_bucket((distribution,), bucket)
        self.assertIn("80 or above", consensus)
        self.assertEqual(consensus["80 or above"].model_count, 1)
        self.assertGreater(consensus["80 or above"].fair_value, 0.5)

    def test_historical_single_run_source_weights_are_calibrated_for_long_replay(self) -> None:
        weights = ConsensusForecastEngine().source_weights

        self.assertEqual(weights["single_run_best_match"], 1.02)
        self.assertEqual(weights["single_run_ecmwf_ifs025"], 1.08)
        self.assertEqual(weights["single_run_gfs_global"], 0.90)

    def test_probability_calibration_lightly_shrinks_single_binary_bucket(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(90.0, 91.0, 92.0),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
        )
        bucket = (TemperatureBucket("80 or above", 80, None),)
        raw = ConsensusForecastEngine().consensus_by_bucket((distribution,), bucket)["80 or above"]
        calibrated = ConsensusForecastEngine(
            probability_calibration=ProbabilityCalibration(center_shrink_alpha=0.95),
        ).consensus_by_bucket((distribution,), bucket)["80 or above"]

        self.assertLess(calibrated.fair_value, raw.fair_value)
        self.assertGreater(calibrated.fair_value, 0.90)
        self.assertLess(
            calibrated.model_probabilities["fixture.hourly_curve_max"],
            raw.model_probabilities["fixture.hourly_curve_max"],
        )

    def test_probability_calibration_shrinks_only_extreme_tails(self) -> None:
        calibration = ProbabilityCalibration(tail_threshold=0.90, tail_shrink_alpha=0.50)

        self.assertAlmostEqual(calibration.transform(0.98), 0.94)
        self.assertAlmostEqual(calibration.transform(0.02), 0.06)
        self.assertAlmostEqual(calibration.transform(0.75), 0.75)

    def test_consensus_preserves_raw_probabilities_before_calibration(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(90.0, 91.0, 92.0),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
        )
        bucket = (TemperatureBucket("80 or above", 80, None),)
        consensus = ConsensusForecastEngine(
            probability_calibration=ProbabilityCalibration(center_shrink_alpha=0.90),
        ).consensus_by_bucket((distribution,), bucket)["80 or above"]

        self.assertIsNotNone(consensus.raw_model_probabilities)
        self.assertIsNotNone(consensus.raw_fair_value)
        self.assertGreater(consensus.raw_fair_value, consensus.fair_value)
        self.assertGreater(
            consensus.raw_model_probabilities["fixture.hourly_curve_max"],
            consensus.model_probabilities["fixture.hourly_curve_max"],
        )

    def test_probability_calibration_preserves_multi_bucket_normalization_by_default(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(78.0, 80.0, 82.0),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
        )
        buckets = (
            TemperatureBucket("Under 80", None, 79.99),
            TemperatureBucket("80-84", 80, 84),
            TemperatureBucket("85 or above", 85, None),
        )
        consensus = ConsensusForecastEngine(
            probability_calibration=ProbabilityCalibration(center_shrink_alpha=0.95),
        ).consensus_by_bucket((distribution,), buckets)

        self.assertAlmostEqual(sum(value.fair_value for value in consensus.values()), 1.0)

    def test_empirical_singleton_forecast_is_smoothed_not_point_mass(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(80.0,),
            generated_at=datetime.now(timezone.utc),
            source="deterministic",
        )
        bucket = (TemperatureBucket("80-80.9", 80, 80.9),)
        probability = EmpiricalEnsembleModel().probability_by_bucket(distribution, bucket)["80-80.9"]
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)

    def test_hourly_curve_model_uses_intraday_predicted_max(self) -> None:
        city = DEFAULT_CITIES[0]
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(80.0,),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
            model_metadata={"features": {"hourly_temperature_2m_max": 90.0}},
        )
        buckets = (
            TemperatureBucket("80-84", 80, 84.99),
            TemperatureBucket("90 or above", 90, None),
        )
        probabilities = HourlyCurveMaxModel(sigma_floor_f=1.0).probability_by_bucket(distribution, buckets)
        self.assertGreater(probabilities["90 or above"], probabilities["80-84"])

    def test_observed_high_zeroes_buckets_already_exceeded(self) -> None:
        city = DEFAULT_CITIES[0]
        buckets = (
            TemperatureBucket("Under 75", None, 74.99),
            TemperatureBucket("75-79", 75, 79.99),
            TemperatureBucket("80 or above", 80, None),
        )
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(76.0, 77.0, 81.0),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
        )
        engine = ConsensusForecastEngine()
        consensus = engine.consensus_by_bucket((distribution,), buckets)
        observed = ObservedHigh(
            city=city,
            target_date=date(2026, 6, 5),
            max_temperature_f=80.5,
            source="test_station",
            observed_at=datetime.now(timezone.utc),
            sample_count=5,
            is_actual=True,
            is_final=False,
        )
        adjusted = engine.apply_observed_high(consensus, buckets, observed)
        self.assertEqual(adjusted["Under 75"].fair_value, 0.0)
        self.assertEqual(adjusted["75-79"].fair_value, 0.0)
        self.assertEqual(adjusted["80 or above"].fair_value, 1.0)
        self.assertTrue(adjusted["80 or above"].observation_adjusted)

    def test_same_day_afternoon_caps_unreached_higher_buckets(self) -> None:
        city = DEFAULT_CITIES[0]
        buckets = (
            TemperatureBucket("Under 80", None, 79.99),
            TemperatureBucket("80-84", 80, 84.99),
            TemperatureBucket("85 or above", 85, None),
        )
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(86.0, 87.0, 88.0, 89.0),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
        )
        engine = ConsensusForecastEngine()
        consensus = engine.consensus_by_bucket((distribution,), buckets)
        observed = ObservedHigh(
            city=city,
            target_date=date(2026, 6, 5),
            max_temperature_f=80.0,
            source="test_station",
            observed_at=datetime(2026, 6, 5, 18, 0, tzinfo=timezone.utc),
            sample_count=12,
            is_actual=True,
            is_final=False,
        )
        adjusted = engine.apply_observed_high(
            consensus,
            buckets,
            observed,
            now=datetime(2026, 6, 5, 19, 0, tzinfo=timezone.utc),
        )
        self.assertLess(adjusted["85 or above"].fair_value, 0.02)
        self.assertGreater(adjusted["80-84"].fair_value, 0.95)
        self.assertTrue(adjusted["85 or above"].observation_adjusted)

    def test_non_actual_same_day_proxy_does_not_hard_adjust_probabilities(self) -> None:
        city = DEFAULT_CITIES[0]
        buckets = (
            TemperatureBucket("Under 80", None, 79.99),
            TemperatureBucket("80 or above", 80, None),
        )
        distribution = ForecastDistribution(
            city=city,
            target_date=date(2026, 6, 5),
            samples_f=(74.0, 75.0, 76.0),
            generated_at=datetime.now(timezone.utc),
            source="fixture",
        )
        engine = ConsensusForecastEngine()
        consensus = engine.consensus_by_bucket((distribution,), buckets)
        observed = ObservedHigh(
            city=city,
            target_date=date(2026, 6, 5),
            max_temperature_f=82.0,
            source="open_meteo_hourly_proxy",
            observed_at=datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc),
            sample_count=8,
            is_actual=False,
            is_final=False,
        )
        adjusted = engine.apply_observed_high(consensus, buckets, observed, now=datetime(2026, 6, 5, 17, 0, tzinfo=timezone.utc))
        self.assertEqual(adjusted["Under 80"].fair_value, consensus["Under 80"].fair_value)
        self.assertEqual(adjusted["80 or above"].fair_value, consensus["80 or above"].fair_value)
        self.assertFalse(adjusted["80 or above"].observation_adjusted)

    def test_metar_station_high_uses_actual_observations(self) -> None:
        city = DEFAULT_CITIES[0]

        class FakeHttp:
            def get_json(self, url, params=None):
                return [
                    {"obsTime": "2026-06-05T13:00:00Z", "temp": 20.0},
                    {"obsTime": "2026-06-05T16:00:00Z", "temp": 25.0},
                    {"obsTime": "2026-06-05T19:00:00Z", "temp": 23.0},
                ]

        observed = ObservedHighClient(FakeHttp())._fetch_metar_station_high(
            city,
            "KNYC",
            date(2026, 6, 5),
            datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc),
        )
        assert observed is not None
        self.assertTrue(observed.is_actual)
        self.assertAlmostEqual(observed.max_temperature_f, 77.0)
        self.assertEqual(observed.sample_count, 2)

    def test_historical_metar_station_high_uses_station_archive(self) -> None:
        city = DEFAULT_CITIES[17]

        class FakeHttp:
            def __init__(self):
                self.params = None

            def get_text(self, url, params=None, headers=None):
                self.params = params
                return "\n".join(
                    (
                        "station,valid,tmpf",
                        "ZBAA,2026-06-04 00:00,62.60",
                        "ZBAA,2026-06-04 14:00,82.40",
                        "ZBAA,2026-06-05 00:00,90.00",
                    )
                )

        fake_http = FakeHttp()
        observed = ObservedHighClient(fake_http)._fetch_historical_metar_station_high(
            city,
            "ZBAA",
            date(2026, 6, 4),
            datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
        )
        assert observed is not None
        self.assertTrue(observed.is_actual)
        self.assertTrue(observed.is_final)
        self.assertEqual(observed.source, "historical_metar_ZBAA")
        self.assertAlmostEqual(observed.max_temperature_f, 82.4)
        self.assertEqual(observed.sample_count, 2)
        self.assertEqual(fake_http.params["tz"], "Asia/Shanghai")

    def test_fetch_observed_high_prefers_historical_metar_over_archive_for_past_station_day(self) -> None:
        city = DEFAULT_CITIES[17]

        class FakeHttp:
            def get_text(self, url, params=None, headers=None):
                return "\n".join(
                    (
                        "station,valid,tmpf",
                        "ZBAA,2026-06-04 06:00,82.40",
                    )
                )

            def get_json(self, url, params=None, headers=None):
                raise AssertionError("Open-Meteo archive should not be used when historical METAR is available")

        observed = ObservedHighClient(FakeHttp()).fetch_observed_high(
            city,
            date(2026, 6, 4),
            now=datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc),
        )
        assert observed is not None
        self.assertEqual(observed.source, "historical_metar_ZBAA")
        self.assertAlmostEqual(observed.max_temperature_f, 82.4)

    def test_archive_high_resolves_final_historical_day(self) -> None:
        city = DEFAULT_CITIES[0]

        class FakeHttp:
            def get_json(self, url, params=None):
                return {
                    "daily": {
                        "time": ["2026-06-05"],
                        "temperature_2m_max": [84.2],
                    }
                }

        observed = ObservedHighClient(FakeHttp())._fetch_open_meteo_archive_high(
            city,
            date(2026, 6, 5),
            datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc),
        )
        assert observed is not None
        self.assertTrue(observed.is_final)
        self.assertFalse(observed.is_actual)
        self.assertEqual(observed.source, "open_meteo_archive_daily")
        self.assertAlmostEqual(observed.max_temperature_f, 84.2)


if __name__ == "__main__":
    unittest.main()
