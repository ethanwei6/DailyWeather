from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from weather_strategy.models import ConsensusValue, ScoredOutcome
from weather_strategy.models import OrderBookQuote
from weather_strategy.observations import ObservedHigh
from weather_strategy.paper import PaperLedger
from weather_strategy.parser import parse_weather_market
from weather_strategy.polymarket import parse_orderbook_quote
from weather_strategy.signals import SignalSettings, generate_signals, hold_filter_reason, market_entry_timing, score_outcomes, signal_filter_reason, signals_from_scored_outcomes


class TradingLayersTest(unittest.TestCase):
    def test_parse_orderbook_quote_picks_best_levels(self) -> None:
        quote = parse_orderbook_quote(
            "tok1",
            {"bids": [{"price": "0.41", "size": "100"}, {"price": "0.43", "size": "50"}], "asks": [["0.48", "25"], ["0.46", "40"]]},
        )
        self.assertEqual(quote.best_bid, 0.43)
        self.assertEqual(quote.best_ask, 0.46)
        self.assertAlmostEqual(quote.spread, 0.03)

    def test_generate_signal_uses_ask_and_price_adjusted_buffer(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "What will be the highest temperature in New York on June 5?",
                "slug": "highest-temperature-new-york-june-5",
                "description": "Official weather station.",
                "outcomes": '["Under 75", "75-79", "80 or above"]',
                "clobTokenIds": '["tok1", "tok2", "tok3"]',
                "outcomePrices": '["0.10", "0.20", "0.30"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        signals = generate_signals(
            market,
            {"Under 75": 0.05, "75-79": 0.30, "80 or above": 0.70},
            {"tok3": OrderBookQuote("tok3", best_bid=0.28, best_ask=0.40)},
            SignalSettings(min_edge=0.10, uncertainty_buffer=0.03, max_spread=0.20, default_size_usd=5, enforce_entry_timing_filter=False),
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].bucket_label, "80 or above")
        self.assertAlmostEqual(signals[0].edge, 0.282)

    def test_same_day_after_local_cutoff_is_not_entry_eligible(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        eligible, reason = market_entry_timing(
            market,
            SignalSettings(same_day_latest_entry_hour_local=14),
            now=datetime(2026, 6, 4, 18, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(eligible)
        self.assertIn("cutoff", reason or "")

    def test_same_day_before_observation_window_is_not_entry_eligible(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        eligible, reason = market_entry_timing(
            market,
            SignalSettings(same_day_earliest_entry_hour_local=11, same_day_latest_entry_hour_local=17),
            now=datetime(2026, 6, 4, 14, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(eligible)
        self.assertIn("observation-aware", reason or "")

    def test_future_day_market_is_entry_eligible(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        eligible, reason = market_entry_timing(
            market,
            SignalSettings(same_day_latest_entry_hour_local=14),
            now=datetime(2026, 6, 4, 18, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(eligible)
        self.assertIsNone(reason)

    def test_late_same_day_score_does_not_generate_signal(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.90,
                model_probabilities={"a": 0.90, "b": 0.92, "c": 0.88},
                model_count=3,
                probability_stdev=0.02,
            )
        }
        settings = SignalSettings(min_edge=0.05, uncertainty_buffer=0.01)
        scored = score_outcomes(market, consensus, settings=settings)
        late_scored = [outcome.__class__(**{**outcome.__dict__, "entry_eligible": False, "entry_filter_reason": "cutoff"}) for outcome in scored]
        self.assertEqual(signals_from_scored_outcomes(late_scored, settings), [])

    def test_paper_ledger_records_signal(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "What will be the highest temperature in New York on June 5?",
                "slug": "highest-temperature-new-york-june-5",
                "description": "Official weather station.",
                "outcomes": '["80 or above"]',
                "clobTokenIds": '["tok3"]',
                "outcomePrices": '["0.30"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        signals = generate_signals(market, {"80 or above": 0.70}, settings=SignalSettings(min_edge=0.10, uncertainty_buffer=0.03, enforce_entry_timing_filter=False))
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.record_signals(signals, metadata={"test": True}), 1)
            rows = ledger.open_trades()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bucket_label"], "80 or above")

    def test_kelly_rebalance_creates_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        from weather_strategy.models import ConsensusValue
        from weather_strategy.signals import score_outcomes

        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.70,
                model_probabilities={"a": 0.68, "b": 0.72, "c": 0.75},
                model_count=3,
                probability_stdev=0.02,
            )
        }
        scored = score_outcomes(market, consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False))
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50)
            self.assertEqual(executions, 1)
            positions = ledger.positions()
            self.assertEqual(len(positions), 1)
            self.assertGreater(positions[0]["shares"], 0)

    def test_kelly_rebalance_applies_fractional_position_cap(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.50", "0.50"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        from weather_strategy.models import ConsensusValue
        from weather_strategy.signals import score_outcomes

        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=1.0,
                model_probabilities={"a": 1.0, "b": 1.0, "c": 1.0},
                model_count=3,
                probability_stdev=0.0,
            )
        }
        scored = score_outcomes(
            market,
            consensus,
            settings=SignalSettings(min_edge=0.0, uncertainty_buffer=0.0, enforce_entry_timing_filter=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(
                scored,
                bankroll_usd=1000,
                kelly_fraction=1.0,
                max_position_usd=900,
                max_position_fraction=0.05,
                min_edge=0.0,
                min_trade_usd=1,
                settings=SignalSettings(min_edge=0.0, uncertainty_buffer=0.0, enforce_entry_timing_filter=False),
            )
            self.assertEqual(executions, 1)
            positions = ledger.positions()
            self.assertEqual(len(positions), 1)
            self.assertAlmostEqual(positions[0]["cost_basis"], 50.0)

    def test_kelly_rebalance_blends_fair_value_toward_market_for_sizing_only(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.50", "0.50"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.90,
                model_probabilities={"a": 0.90, "b": 0.90, "c": 0.90},
                model_count=3,
                probability_stdev=0.0,
            )
        }
        settings = SignalSettings(min_edge=0.0, uncertainty_buffer=0.0, enforce_entry_timing_filter=False)
        scored = score_outcomes(market, consensus, settings=settings)
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(
                scored,
                bankroll_usd=1000,
                kelly_fraction=1.0,
                max_position_usd=900,
                kelly_market_blend=0.50,
                min_trade_usd=1,
                settings=settings,
            )
            self.assertEqual(executions, 1)
            positions = ledger.positions()
            self.assertEqual(len(positions), 1)
            self.assertAlmostEqual(positions[0]["cost_basis"], 400.0)

    def test_kelly_rebalance_can_scale_position_cap_by_edge(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.50", "0.50"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.90,
                model_probabilities={"a": 0.90, "b": 0.90, "c": 0.90},
                model_count=3,
                probability_stdev=0.0,
            )
        }
        settings = SignalSettings(min_edge=0.0, uncertainty_buffer=0.0, enforce_entry_timing_filter=False)
        scored = score_outcomes(market, consensus, settings=settings)
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(
                scored,
                bankroll_usd=1000,
                kelly_fraction=1.0,
                max_position_usd=100,
                edge_position_full_cap_edge=0.80,
                edge_position_min_multiplier=0.25,
                min_trade_usd=1,
                settings=settings,
            )
            self.assertEqual(executions, 1)
            positions = ledger.positions()
            self.assertEqual(len(positions), 1)
            self.assertAlmostEqual(positions[0]["cost_basis"], 50.0)

    def test_kelly_rebalance_rejects_low_agreement(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        from weather_strategy.models import ConsensusValue
        from weather_strategy.signals import score_outcomes

        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.70,
                model_probabilities={"a": 0.80, "b": 0.20, "c": 0.25},
                model_count=3,
                probability_stdev=0.2,
            )
        }
        scored = score_outcomes(market, consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01))
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(
                scored,
                bankroll_usd=1000,
                kelly_fraction=0.25,
                max_position_usd=50,
                min_edge=0.05,
                min_model_agreement=0.65,
            )
            self.assertEqual(executions, 0)
            self.assertEqual(ledger.positions(), [])

    def test_price_aware_edge_gate_accepts_high_probability_edge(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.90", "0.10"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.95,
                model_probabilities={"source_a.model": 0.95, "source_b.model": 0.94, "source_c.model": 0.96},
                model_count=3,
                probability_stdev=0.01,
            )
        }
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.03,
            low_price_exact_bucket_threshold=0.15,
            low_price_exact_bucket_min_fair_value=0.27,
            low_price_exact_bucket_min_edge=0.16,
            enforce_entry_timing_filter=False,
        )
        scored = score_outcomes(market, consensus, settings=settings)
        signals = signals_from_scored_outcomes(scored, settings)
        self.assertEqual(len(signals), 1)
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_edge=0.08)
            self.assertEqual(executions, 1)
            self.assertEqual(len(ledger.positions()), 1)

    def test_price_aware_edge_gate_rejects_low_probability_longshot_gap(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 100°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-100-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.01", "0.99"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.06,
                model_probabilities={"source_a.model": 0.06, "source_b.model": 0.07, "source_c.model": 0.05},
                model_count=3,
                probability_stdev=0.01,
            )
        }
        settings = SignalSettings(min_edge=0.08, uncertainty_buffer=0.03, min_price=0.0, enforce_entry_timing_filter=False)
        scored = score_outcomes(market, consensus, settings=settings)
        self.assertGreater(scored[0].edge, 0.0)
        self.assertEqual(signals_from_scored_outcomes(scored, settings), [])
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_edge=0.08, min_price=0.0)
            self.assertEqual(executions, 0)
            self.assertEqual(ledger.positions(), [])

    def test_min_signal_fair_value_rejects_mid_confidence_model_edge(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.40", "0.60"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.64,
                model_probabilities={"source_a.model": 0.64, "source_b.model": 0.66, "source_c.model": 0.62},
                model_count=3,
                probability_stdev=0.02,
            )
        }
        settings = SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, min_signal_fair_value=0.70, enforce_entry_timing_filter=False)
        scored = score_outcomes(market, consensus, settings=settings)

        self.assertEqual(signals_from_scored_outcomes(scored, settings), [])
        self.assertEqual("fair value below 0.70", signal_filter_reason(scored[0], settings))
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, settings=settings)
            self.assertEqual(executions, 0)

    def test_default_yes_side_floor_rejects_cheap_open_threshold(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in Seoul be 25°C or higher on June 6?",
                "slug": "highest-temperature-seoul-june-6-25c-or-higher",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.14", "0.86"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.90,
                model_probabilities={"source_a.model": 0.90, "source_b.model": 0.91, "source_c.model": 0.89},
                model_count=3,
                probability_stdev=0.01,
            )
        }
        settings = SignalSettings(enforce_entry_timing_filter=False)
        scored = score_outcomes(market, consensus, settings=settings)
        signals = signals_from_scored_outcomes(scored, settings)

        self.assertEqual(settings.min_price, 0.125)
        self.assertEqual(settings.yes_side_min_price, 0.20)
        self.assertEqual(signals, [])
        self.assertEqual("YES-side market price below 0.2", signal_filter_reason(scored[0], settings))

    def test_yes_side_floor_does_not_block_low_price_no_research_rows(self) -> None:
        outcome = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seoul be 25°C or higher on June 6?",
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.90,
            market_price=0.14,
            edge=0.75,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 6),
            rule_excerpt="rules",
            model_probabilities={"a": 0.93, "b": 0.94, "c": 0.95},
        )

        self.assertIsNone(signal_filter_reason(outcome, SignalSettings(enforce_entry_timing_filter=False)))

    def test_default_price_floor_still_rejects_tiny_longshot_prices(self) -> None:
        outcome = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="Will the highest temperature in Seoul be 25°C or higher on June 6?",
            bucket_label="25°C or higher",
            token_id="yes-token",
            fair_value=0.90,
            market_price=0.10,
            edge=0.77,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 6),
            rule_excerpt="rules",
            model_probabilities={"a": 0.90, "b": 0.91, "c": 0.89},
        )

        self.assertIn("market price below", signal_filter_reason(outcome, SignalSettings(enforce_entry_timing_filter=False)) or "")

    def test_low_price_exact_temperature_bucket_requires_stronger_edge(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in London be 31°C on June 5?",
                "slug": "highest-temperature-london-june-5-31c",
                "description": "Resolved to the nearest whole degree Celsius.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.08", "0.92"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.24,
                model_probabilities={"source_a.model": 0.24, "source_b.model": 0.25, "source_c.model": 0.23},
                model_count=3,
                probability_stdev=0.01,
            )
        }
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.03,
            low_price_exact_bucket_threshold=0.15,
            low_price_exact_bucket_min_fair_value=0.27,
            low_price_exact_bucket_min_edge=0.16,
            enforce_entry_timing_filter=False,
        )
        scored = score_outcomes(market, consensus, settings=settings)
        self.assertGreater(scored[0].edge, 0.0)
        self.assertLess(scored[0].bucket_width_f or 99, settings.exact_bucket_max_width_f)
        self.assertEqual(signals_from_scored_outcomes(scored, settings), [])
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, settings=settings)
            self.assertEqual(executions, 0)
            self.assertEqual(ledger.positions(), [])

    def test_no_side_entry_requires_extra_absolute_edge(self) -> None:
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.03,
            no_side_min_edge=0.05,
            no_side_high_confidence_min_edge=0.05,
            enforce_entry_timing_filter=False,
        )
        base = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="Will the highest temperature in Seoul be 25°C or higher on May 28?",
            bucket_label="25°C or higher",
            token_id="yes-token",
            fair_value=0.935,
            market_price=0.89,
            edge=0.045,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 5, 28),
            rule_excerpt="rules",
            model_probabilities={"a": 0.935, "b": 0.936, "c": 0.934},
        )
        no_side = ScoredOutcome(
            **{
                **base.__dict__,
                "question": f"NO: {base.question}",
                "bucket_label": f"NO: {base.bucket_label}",
                "token_id": "no-token",
            }
        )

        self.assertIsNone(signal_filter_reason(base, settings))
        self.assertEqual(signal_filter_reason(no_side, settings), "NO-side edge below 0.05")
        self.assertEqual(signals_from_scored_outcomes([base, no_side], settings), [signals_from_scored_outcomes([base], settings)[0]])

    def test_high_confidence_no_side_entry_uses_smaller_edge_floor(self) -> None:
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.0,
            no_side_min_edge=0.10,
            no_side_high_confidence_min_edge=0.02,
            no_side_max_counter_event_probability=0.08,
            high_confidence_price_threshold=0.75,
            enforce_entry_timing_filter=False,
        )
        high_confidence_no = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seoul be 25°C or higher on May 28?",
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.95,
            market_price=0.90,
            edge=0.05,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 5, 28),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.95, "b.model": 0.96, "c.model": 0.94},
        )
        mid_price_no = ScoredOutcome(
            **{
                **high_confidence_no.__dict__,
                "market_price": 0.70,
                "fair_value": 0.75,
                "edge": 0.05,
                "model_probabilities": {"a.model": 0.95, "b.model": 0.96, "c.model": 0.94},
            }
        )

        self.assertIsNone(signal_filter_reason(high_confidence_no, settings))
        self.assertEqual(signal_filter_reason(mid_price_no, settings), "NO-side edge below 0.10")

    def test_no_side_entry_rejects_above_no_side_max_price_but_yes_can_pass(self) -> None:
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.0,
            no_side_min_edge=0.10,
            no_side_high_confidence_min_edge=0.02,
            no_side_max_price=0.90,
            no_side_max_counter_event_probability=0.10,
            high_confidence_price_threshold=0.75,
            enforce_entry_timing_filter=False,
        )
        no_side = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seoul be 25°C or higher on May 28?",
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.99,
            market_price=0.91,
            edge=0.05,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 5, 28),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.99, "b.model": 0.99, "c.model": 0.99},
        )
        yes_side = ScoredOutcome(
            **{
                **no_side.__dict__,
                "question": no_side.question.removeprefix("NO: "),
                "bucket_label": "25°C or higher",
                "token_id": "yes-token",
            }
        )

        self.assertEqual(signal_filter_reason(no_side, settings), "NO-side market price above 0.9")
        self.assertIsNone(signal_filter_reason(yes_side, settings))
        self.assertIsNone(signal_filter_reason(no_side, replace(settings, no_side_max_price=0.95)))

    def test_default_no_side_max_price_accepts_ninety_four_but_blocks_above_global_max(self) -> None:
        no_side = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seoul be 25°C or higher on May 28?",
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.99,
            market_price=0.92,
            edge=0.06,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 5, 28),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.99, "b.model": 0.99, "c.model": 0.99},
        )

        self.assertIsNone(signal_filter_reason(no_side, SignalSettings(enforce_entry_timing_filter=False)))
        too_expensive = ScoredOutcome(**{**no_side.__dict__, "market_price": 0.94})
        self.assertIsNone(signal_filter_reason(too_expensive, SignalSettings(enforce_entry_timing_filter=False)))
        above_global_cap = ScoredOutcome(**{**no_side.__dict__, "market_price": 0.96})
        self.assertEqual(
            signal_filter_reason(above_global_cap, SignalSettings(enforce_entry_timing_filter=False)),
            "market price above 0.95",
        )

    def test_default_no_side_counter_event_cap_accepts_ten_percent_but_blocks_more(self) -> None:
        no_side = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seoul be 25°C or higher on May 28?",
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.94,
            market_price=0.80,
            edge=0.13,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 5, 28),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.91, "b.model": 0.92, "c.model": 0.93},
        )

        settings = SignalSettings(enforce_entry_timing_filter=False)
        self.assertIsNone(signal_filter_reason(no_side, settings))
        ten_percent_tail = ScoredOutcome(**{**no_side.__dict__, "model_probabilities": {"a.model": 0.90, "b.model": 0.92, "c.model": 0.93}})
        self.assertIsNone(signal_filter_reason(ten_percent_tail, settings))
        wider_tail = ScoredOutcome(**{**no_side.__dict__, "model_probabilities": {"a.model": 0.899, "b.model": 0.92, "c.model": 0.93}})
        self.assertEqual(signal_filter_reason(wider_tail, settings), "NO-side counter-event probability above 0.1")

    def test_no_side_counter_event_gate_can_relax_at_configured_utc_hour(self) -> None:
        no_side = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Miami be 86°F or higher on January 22?",
            bucket_label="NO: 86°F or higher",
            token_id="no-token",
            fair_value=0.86,
            market_price=0.70,
            edge=0.151,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc),
            city="Miami, FL",
            target_date=date(2026, 1, 22),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.85, "b.model": 0.88, "c.model": 0.90},
            bucket_lower_f=86.0,
            bucket_upper_f=None,
        )
        settings = SignalSettings(
            enforce_entry_timing_filter=False,
            no_side_relaxed_counter_event_probability=0.20,
            no_side_relaxed_counter_event_hours_utc=(12,),
        )

        self.assertIsNone(signal_filter_reason(no_side, settings))
        outside_hour = ScoredOutcome(**{**no_side.__dict__, "generated_at": datetime(2026, 1, 20, 6, 0, tzinfo=timezone.utc)})
        self.assertEqual(signal_filter_reason(outside_hour, settings), "NO-side counter-event probability above 0.1")

    def test_bounded_no_side_entries_can_be_disabled_separately(self) -> None:
        no_side = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seattle be between 40-41°F on January 22?",
            bucket_label="NO: 40-41°F",
            token_id="no-token",
            fair_value=0.98,
            market_price=0.80,
            edge=0.17,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc),
            city="Seattle, WA",
            target_date=date(2026, 1, 22),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.98, "b.model": 0.99, "c.model": 0.97},
            bucket_lower_f=40.0,
            bucket_upper_f=41.0,
            bucket_width_f=1.0,
        )

        self.assertIsNone(signal_filter_reason(no_side, SignalSettings(enforce_entry_timing_filter=False)))
        self.assertEqual(
            signal_filter_reason(
                no_side,
                SignalSettings(enforce_entry_timing_filter=False, allow_bounded_no_side_entries=False),
            ),
            "bounded NO-side exact/range bucket entries disabled",
        )

    def test_no_side_hold_tail_is_wider_than_new_entry_tail(self) -> None:
        no_side = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seoul be 31°C or higher on June 17?",
            bucket_label="NO: 31°C or higher",
            token_id="no-token",
            fair_value=0.96,
            market_price=0.83,
            edge=0.12,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 17),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.88, "b.model": 0.91, "c.model": 0.92},
        )

        settings = SignalSettings(enforce_entry_timing_filter=False)

        self.assertEqual(signal_filter_reason(no_side, settings), "NO-side counter-event probability above 0.1")
        self.assertIsNone(hold_filter_reason(no_side, settings))

    def test_no_side_entry_rejects_opposing_tail_disagreement(self) -> None:
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.03,
            no_side_min_edge=0.05,
            no_side_max_counter_event_probability=0.10,
            enforce_entry_timing_filter=False,
        )
        no_side = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="NO: Will the highest temperature in Seoul be 25°C or higher on May 28?",
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.95,
            market_price=0.60,
            edge=0.338,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.04,
            generated_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 5, 28),
            rule_excerpt="rules",
            model_probabilities={"a.model": 0.97, "b.model": 0.89, "c.model": 0.98},
        )
        yes_side = ScoredOutcome(**{**no_side.__dict__, "question": no_side.question.removeprefix("NO: "), "bucket_label": "25°C or higher"})

        self.assertEqual(signal_filter_reason(no_side, settings), "NO-side counter-event probability above 0.1")
        self.assertEqual(signals_from_scored_outcomes([no_side], settings), [])
        self.assertIsNone(signal_filter_reason(yes_side, settings))

    def test_low_price_exact_temperature_bucket_can_pass_with_exceptional_edge(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in London be 31°C on June 5?",
                "slug": "highest-temperature-london-june-5-31c",
                "description": "Resolved to the nearest whole degree Celsius.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.08", "0.92"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.40,
                model_probabilities={"source_a.model": 0.40, "source_b.model": 0.41, "source_c.model": 0.05},
                model_count=3,
                probability_stdev=0.01,
            )
        }
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.03,
            min_price=0.05,
            yes_side_min_price=0.05,
            min_signal_fair_value=0.0,
            min_model_agreement=0.65,
            allow_bounded_bucket_entries=True,
            bounded_bucket_min_edge=0.0,
            bounded_bucket_min_fair_value=0.0,
            bounded_bucket_min_model_agreement=0.0,
            bounded_bucket_min_price=0.0,
            enforce_entry_timing_filter=False,
        )
        scored = score_outcomes(market, consensus, settings=settings)
        signals = signals_from_scored_outcomes(scored, settings)
        self.assertEqual(len(signals), 1)

    def test_bounded_temperature_bucket_rejected_by_default_quality_gate(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in London be 31°C on June 5?",
                "slug": "highest-temperature-london-june-5-31c",
                "description": "Resolved to the nearest whole degree Celsius.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.90,
                model_probabilities={"source_a.model": 0.90, "source_b.model": 0.91, "source_c.model": 0.89},
                model_count=3,
                probability_stdev=0.01,
            )
        }
        settings = SignalSettings(min_edge=0.08, uncertainty_buffer=0.03, enforce_entry_timing_filter=False)
        scored = score_outcomes(market, consensus, settings=settings)

        self.assertEqual(signals_from_scored_outcomes(scored, settings), [])
        self.assertEqual("bounded exact/range bucket price below 0.5", signal_filter_reason(scored[0], settings))

    def test_correlated_exact_temperature_bucket_rejects_full_source_agreement(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in London be 31°C on June 5?",
                "slug": "highest-temperature-london-june-5-31c",
                "description": "Resolved to the nearest whole degree Celsius.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.08", "0.92"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.40,
                model_probabilities={"source_a.model": 0.40, "source_b.model": 0.41, "source_c.model": 0.39},
                model_count=3,
                probability_stdev=0.01,
            )
        }
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.03,
            min_signal_fair_value=0.0,
            allow_bounded_bucket_entries=True,
            bounded_bucket_min_edge=0.0,
            bounded_bucket_min_fair_value=0.0,
            bounded_bucket_min_model_agreement=0.0,
            bounded_bucket_min_price=0.0,
            enforce_entry_timing_filter=False,
        )
        scored = score_outcomes(market, consensus, settings=settings)
        self.assertEqual(scored[0].model_agreement, 1.0)
        self.assertEqual(signals_from_scored_outcomes(scored, settings), [])

    def test_signals_keep_only_best_entry_per_city_date(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "What will be the highest temperature in New York on June 5?",
                "slug": "highest-temperature-new-york-june-5",
                "description": "Official weather station.",
                "outcomes": '["80 or above", "85 or above"]',
                "clobTokenIds": '["tok80", "tok85"]',
                "outcomePrices": '["0.30", "0.20"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(market.buckets[0].label, 0.70, {"a": 0.70, "b": 0.72, "c": 0.71}, 3, 0.01),
            market.buckets[1].label: ConsensusValue(market.buckets[1].label, 0.55, {"a": 0.55, "b": 0.57, "c": 0.56}, 3, 0.01),
        }
        settings = SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False)
        scored = score_outcomes(market, consensus, settings=settings)
        signals = signals_from_scored_outcomes(scored, settings)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].token_id, "tok80")

    def test_kelly_rebalance_keeps_only_best_entry_per_city_date(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "What will be the highest temperature in New York on June 5?",
                "slug": "highest-temperature-new-york-june-5",
                "description": "Official weather station.",
                "outcomes": '["80 or above", "85 or above"]',
                "clobTokenIds": '["tok80", "tok85"]',
                "outcomePrices": '["0.30", "0.20"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(market.buckets[0].label, 0.70, {"a": 0.70, "b": 0.72, "c": 0.71}, 3, 0.01),
            market.buckets[1].label: ConsensusValue(market.buckets[1].label, 0.55, {"a": 0.55, "b": 0.57, "c": 0.56}, 3, 0.01),
        }
        scored = score_outcomes(market, consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False))
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            executions = ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_edge=0.05)
            self.assertEqual(executions, 1)
            positions = ledger.positions()
            self.assertEqual(len(positions), 1)
            self.assertEqual(positions[0]["token_id"], "tok80")

    def test_kelly_rebalance_holds_existing_position_when_new_entries_time_blocked(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                market.buckets[0].label,
                0.70,
                {"source_a.model": 0.70, "source_b.model": 0.72, "source_c.model": 0.71},
                3,
                0.01,
            )
        }
        scored = score_outcomes(market, consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False))
        blocked_scored = [
            outcome.__class__(
                **{
                    **outcome.__dict__,
                    "entry_eligible": False,
                    "entry_filter_reason": "same-day market before 11:00 local observation-aware entry window",
                }
            )
            for outcome in scored
        ]
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_edge=0.05), 1)
            self.assertEqual(len(ledger.positions()), 1)
            self.assertEqual(ledger.rebalance_kelly(blocked_scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_edge=0.05), 0)
            self.assertEqual(len(ledger.positions()), 1)

    def test_kelly_rebalance_uses_looser_agreement_for_holding_than_entry(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        settings = SignalSettings(
            min_edge=0.05,
            uncertainty_buffer=0.01,
            min_model_agreement=1.0,
            hold_min_model_agreement=0.65,
            enforce_entry_timing_filter=False,
        )
        entry_consensus = {
            market.buckets[0].label: ConsensusValue(
                market.buckets[0].label,
                0.75,
                {"source_a.model": 0.75, "source_b.model": 0.76, "source_c.model": 0.77},
                3,
                0.01,
            )
        }
        hold_consensus = {
            market.buckets[0].label: ConsensusValue(
                market.buckets[0].label,
                0.75,
                {"source_a.model": 0.90, "source_b.model": 0.80, "source_c.model": 0.20},
                3,
                0.30,
            )
        }
        entry_scored = score_outcomes(market, entry_consensus, settings=settings)
        hold_scored = score_outcomes(market, hold_consensus, settings=settings)
        self.assertEqual(entry_scored[0].model_agreement, 1.0)
        self.assertLess(hold_scored[0].model_agreement, settings.min_model_agreement)

        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.rebalance_kelly(entry_scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, settings=settings), 1)
            self.assertEqual(len(ledger.positions()), 1)
            self.assertEqual(ledger.rebalance_kelly(hold_scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, settings=settings), 0)
            self.assertEqual(len(ledger.positions()), 1)

    def test_kelly_rebalance_does_not_trim_valid_existing_hold(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        settings = SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False)
        entry_consensus = {
            market.buckets[0].label: ConsensusValue(
                market.buckets[0].label,
                0.80,
                {"source_a.model": 0.80, "source_b.model": 0.82, "source_c.model": 0.81},
                3,
                0.01,
            )
        }
        entry_scored = score_outcomes(market, entry_consensus, settings=settings)

        repriced_market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.65", "0.35"]',
            },
            today=date(2026, 6, 4),
        )
        assert repriced_market is not None
        hold_consensus = {
            repriced_market.buckets[0].label: ConsensusValue(
                repriced_market.buckets[0].label,
                0.70,
                {"source_a.model": 0.70, "source_b.model": 0.71, "source_c.model": 0.69},
                3,
                0.01,
            )
        }
        hold_scored = score_outcomes(repriced_market, hold_consensus, settings=settings)

        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.rebalance_kelly(entry_scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, settings=settings), 1)
            initial_position = ledger.positions()[0]
            self.assertEqual(ledger.rebalance_kelly(hold_scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, settings=settings), 0)
            held_position = ledger.positions()[0]

        self.assertAlmostEqual(held_position["shares"], initial_position["shares"])
        self.assertGreater(held_position["last_price"], initial_position["last_price"])

    def test_kelly_rebalance_closes_zero_target_dust_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        first_consensus = {
            market.buckets[0].label: ConsensusValue(
                market.buckets[0].label,
                0.70,
                {"a": 0.70, "b": 0.72, "c": 0.71},
                3,
                0.01,
            )
        }
        first_scored = score_outcomes(market, first_consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False))

        dust_market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.001", "0.999"]',
            },
            today=date(2026, 6, 4),
        )
        assert dust_market is not None
        dust_consensus = {
            dust_market.buckets[0].label: ConsensusValue(
                dust_market.buckets[0].label,
                0.0,
                {"a": 0.0, "b": 0.0, "c": 0.0},
                3,
                0.0,
            )
        }
        dust_scored = score_outcomes(dust_market, dust_consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False))

        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.rebalance_kelly(first_scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_edge=0.05), 1)
            self.assertEqual(len(ledger.positions()), 1)
            self.assertEqual(ledger.rebalance_kelly(dust_scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_trade_usd=1, min_edge=0.05), 1)
            self.assertEqual(ledger.positions(), [])

    def test_forecast_scores_record_calibration_rows(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                bucket_label=market.buckets[0].label,
                fair_value=0.70,
                model_probabilities={"a": 0.68, "b": 0.72, "c": 0.75},
                model_count=3,
                probability_stdev=0.02,
                observed_high_f=82.0,
                observation_source="test_station",
                observation_final=True,
                observation_adjusted=True,
            )
        }
        scored = score_outcomes(market, consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01))
        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.record_forecast_scores(scored), 1)
            summary = ledger.calibration_summary()
            self.assertEqual(summary["total_scores"], 1)
            self.assertEqual(summary["resolved_scores"], 1)
            self.assertIsNotNone(summary["brier_score"])

    def test_settle_expired_position_uses_final_observed_high(self) -> None:
        market = parse_weather_market(
            {
                "id": "123",
                "question": "Will the highest temperature in New York City be 80°F or above on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "description": "Official weather station.",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok3", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 4),
        )
        assert market is not None
        consensus = {
            market.buckets[0].label: ConsensusValue(
                market.buckets[0].label,
                0.70,
                {"source_a.model": 0.70, "source_b.model": 0.72, "source_c.model": 0.71},
                3,
                0.01,
            )
        }
        scored = score_outcomes(market, consensus, settings=SignalSettings(min_edge=0.05, uncertainty_buffer=0.01, enforce_entry_timing_filter=False))

        class FakeObservationClient:
            def fetch_observed_high(self, city, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=81.0,
                    source="test_final",
                    observed_at=datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc),
                    sample_count=24,
                    is_actual=True,
                    is_final=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.rebalance_kelly(scored, bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50, min_edge=0.05), 1)
            settled, errors = ledger.settle_expired_positions(FakeObservationClient(), now=datetime(2026, 6, 6, 16, 0, tzinfo=timezone.utc))
            self.assertEqual((settled, errors), (1, 0))
            self.assertEqual(ledger.positions(), [])
            self.assertGreater(ledger.equity_usd(1000), 1000)

    def test_settle_expired_no_position_inverts_final_observed_high(self) -> None:
        no_side = ScoredOutcome(
            market_id="123",
            market_slug="highest-temperature-new-york-june-5-80-or-above",
            question="NO: Will the highest temperature in New York City be 80°F or above on June 5?",
            bucket_label="NO: 80°F or above",
            token_id="tok-no",
            fair_value=0.95,
            market_price=0.30,
            edge=0.64,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.01,
            generated_at=datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc),
            city="New York, NY",
            target_date=date(2026, 6, 5),
            rule_excerpt="rules",
            model_probabilities={"source_a.model": 0.95, "source_b.model": 0.96, "source_c.model": 0.94},
        )

        class FakeObservationClient:
            def fetch_observed_high(self, city, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=79.0,
                    source="test_final",
                    observed_at=datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc),
                    sample_count=24,
                    is_actual=True,
                    is_final=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            ledger = PaperLedger(Path(directory) / "paper.sqlite")
            self.assertEqual(ledger.rebalance_kelly([no_side], bankroll_usd=1000, kelly_fraction=0.25, max_position_usd=50), 1)
            settled, errors = ledger.settle_expired_positions(FakeObservationClient(), now=datetime(2026, 6, 6, 16, 0, tzinfo=timezone.utc))
            self.assertEqual((settled, errors), (1, 0))
            self.assertEqual(ledger.positions(), [])
            self.assertGreater(ledger.equity_usd(1000), 1000)


if __name__ == "__main__":
    unittest.main()
