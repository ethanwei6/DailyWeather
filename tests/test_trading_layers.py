from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from weather_strategy.models import ConsensusValue
from weather_strategy.models import OrderBookQuote
from weather_strategy.observations import ObservedHigh
from weather_strategy.paper import PaperLedger
from weather_strategy.parser import parse_weather_market
from weather_strategy.polymarket import parse_orderbook_quote
from weather_strategy.signals import SignalSettings, generate_signals, market_entry_timing, score_outcomes, signals_from_scored_outcomes


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
        settings = SignalSettings(min_edge=0.08, uncertainty_buffer=0.03, enforce_entry_timing_filter=False)
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


if __name__ == "__main__":
    unittest.main()
