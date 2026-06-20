from __future__ import annotations

import unittest
from datetime import date

from weather_strategy.parser import parse_market_date, parse_temperature_bucket, parse_weather_market
from weather_strategy.parser import looks_like_temperature_market


class ParserTest(unittest.TestCase):
    def test_parse_market_date_rolls_forward_without_year(self) -> None:
        self.assertEqual(parse_market_date("Highest temperature in New York on June 5?", today=date(2026, 6, 4)), date(2026, 6, 5))
        self.assertEqual(parse_market_date("Highest temperature in New York on January 2?", today=date(2026, 6, 4)), date(2027, 1, 2))
        self.assertEqual(
            parse_market_date("Highest temperature in New York on June 4? highest-temperature-new-york-june-4-2026", today=date(2026, 6, 5)),
            date(2026, 6, 4),
        )

    def test_parse_temperature_buckets_common_labels(self) -> None:
        self.assertEqual(parse_temperature_bucket("80-84").lower_f, 80)
        self.assertEqual(parse_temperature_bucket("80-84").upper_f, 84)
        self.assertEqual(parse_temperature_bucket("Under 75").upper_f, 74.99)
        self.assertEqual(parse_temperature_bucket("90 or above").lower_f, 90)
        self.assertEqual(parse_temperature_bucket("Above 100").lower_f, 100.01)
        self.assertAlmostEqual(parse_temperature_bucket("Will the highest temperature in Seoul be 18°C on June 5?").lower_f, 64.4)
        self.assertAlmostEqual(parse_temperature_bucket("Will the highest temperature in Seoul be 18°C on June 5?").upper_f, 66.1982)

    def test_parse_weather_market_from_gamma_shape(self) -> None:
        raw = {
            "id": "123",
            "question": "What will be the highest temperature in New York on June 5?",
            "slug": "highest-temperature-new-york-june-5",
            "eventSlug": "weather-new-york-june-5",
            "description": "Resolved according to the official station reading.",
            "outcomes": '["Under 75", "75-79", "80 or above"]',
            "clobTokenIds": '["tok1", "tok2", "tok3"]',
            "outcomePrices": '["0.20", "0.55", "0.25"]',
        }
        parsed = parse_weather_market(raw, today=date(2026, 6, 4))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIsNotNone(parsed.city)
        assert parsed.city is not None
        self.assertEqual(parsed.city.name, "New York")
        self.assertEqual(parsed.target_date, date(2026, 6, 5))
        self.assertEqual(len(parsed.buckets), 3)
        self.assertEqual(parsed.buckets[1].token_id, "tok2")
        self.assertEqual(parsed.buckets[1].market_price, 0.55)

    def test_parse_weather_market_uses_event_context(self) -> None:
        raw = {
            "id": "123",
            "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
            "slug": "highest-temperature-new-york-june-5-80-or-above",
            "eventTitle": "What will be the highest temperature in New York on June 5?",
            "description": "Official weather station.",
            "outcomes": '["Yes", "No"]',
        }
        parsed = parse_weather_market(raw, today=date(2026, 6, 4))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.city.name, "New York")
        self.assertEqual(len(parsed.buckets), 1)
        self.assertEqual(parsed.buckets[0].lower_f, 80)

    def test_resolution_precision_rounds_to_market_unit(self) -> None:
        raw = {
            "id": "123",
            "question": "Will the highest temperature in Chicago be between 70-71°F on June 17?",
            "slug": "highest-temperature-in-chicago-on-june-17-2026-70-71f",
            "eventTitle": "Highest temperature in Chicago on June 17?",
            "description": "The resolution source for this market measures temperatures to whole degrees Fahrenheit.",
            "outcomes": '["Yes", "No"]',
        }
        parsed = parse_weather_market(raw, today=date(2026, 6, 4))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        bucket = parsed.buckets[0]
        self.assertEqual(bucket.resolution_unit, "F")
        self.assertEqual(bucket.resolution_precision, 1.0)
        self.assertTrue(bucket.contains(69.98))
        self.assertFalse(bucket.contains(72.49))

    def test_celsius_resolution_precision_rounds_in_celsius(self) -> None:
        raw = {
            "id": "123",
            "question": "Will the highest temperature in Shenzhen be 26°C on June 18?",
            "slug": "highest-temperature-in-shenzhen-on-june-18-2026-26c",
            "eventTitle": "Highest temperature in Shenzhen on June 18?",
            "description": "The resolution source for this market measures temperatures to whole degrees Celsius.",
            "outcomes": '["Yes", "No"]',
        }
        parsed = parse_weather_market(raw, today=date(2026, 6, 4))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        bucket = parsed.buckets[0]
        self.assertEqual(bucket.resolution_unit, "C")
        self.assertTrue(bucket.contains(79.6))
        self.assertFalse(bucket.contains(80.6))

    def test_short_city_alias_does_not_match_inside_words(self) -> None:
        raw = {
            "id": "beijing",
            "question": "Will the highest temperature in Beijing be 28°C on June 5?",
            "slug": "highest-temperature-in-beijing-on-june-5-2026-28c",
            "eventTitle": "Highest temperature in Beijing on June 5?",
            "description": "Resolved according to the latest official temperature source.",
            "outcomes": '["Yes", "No"]',
        }
        parsed = parse_weather_market(raw, today=date(2026, 6, 4))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.city.name, "Beijing")

    def test_lowest_temperature_market_is_not_high_temperature_strategy(self) -> None:
        raw = {
            "question": "Will the lowest temperature in New York City be between 60-61°F on June 4?",
            "slug": "lowest-temperature-new-york-june-4",
        }
        self.assertFalse(looks_like_temperature_market(raw))

    def test_parse_newly_supported_live_cities(self) -> None:
        raw = {
            "id": "shenzhen",
            "question": "Will the highest temperature in Shenzhen be 34°C on June 5?",
            "slug": "highest-temperature-in-shenzhen-on-june-5-2026-34c",
            "eventTitle": "Highest temperature in Shenzhen on June 5?",
            "outcomes": '["Yes", "No"]',
        }
        parsed = parse_weather_market(raw, today=date(2026, 6, 4))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.city.name, "Shenzhen")


if __name__ == "__main__":
    unittest.main()
