from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from weather_strategy.cities import DEFAULT_CITIES
from weather_strategy.forecast import ConsensusForecastEngine, EmpiricalEnsembleModel, ForecastEngine, ForecastSettings, HourlyCurveMaxModel
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
