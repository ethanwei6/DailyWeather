from __future__ import annotations

import json
import inspect
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from weather_strategy.backtest import (
    ResolvedForecastRow,
    _effective_max_position_usd as recorded_effective_max_position_usd,
    _single_entry_kelly_replay,
    load_calibration_weights,
)
from weather_strategy.long_backtest import (
    CachedHttpClient,
    HistoricalPosition,
    _binary_no_token,
    _cached_price_history,
    _candidate_replay_times,
    _data_quality_diagnostics,
    _effective_max_position_usd as long_effective_max_position_usd,
    _finalize_positions_for_result,
    _historical_observed_high_for_session,
    _invert_binary_scored_outcome,
    _json_kelly_replay,
    _make_run_log_path,
    _market_parse_today,
    _pnl_concentration,
    _polymarket_yes_payout,
    _price_history_bounds,
    _price_history_bounds_for_replay_times,
    _rebalance_session,
    _resolve_market_outcome,
    _real_data_audit,
    _robustness_diagnostics,
    _score_calibration_diagnostics,
    _scored_to_json,
    _settlement_quality_diagnostics,
    _selected_candidate_weather_validation,
    _signal_opportunity_diagnostics,
    _strategy_recommendation_diagnostics,
    _strategy_sensitivity_diagnostics,
    _telonex_market_record_to_raw,
    _trade_performance_diagnostics,
    forecast_run_time,
    run_long_historical_backtest,
    select_entry_price,
)
from weather_strategy.models import ScoredOutcome
from weather_strategy.observations import ObservedHigh
from weather_strategy.paper import _effective_max_position_usd as paper_effective_max_position_usd
from weather_strategy.parser import parse_weather_market
from weather_strategy.polymarket import PriceHistoryPoint
from weather_strategy.signals import SignalSettings


class BacktestTest(unittest.TestCase):
    def test_replay_times_mark_target_day_as_maintenance_only(self) -> None:
        raw = {
            "id": "m1",
            "question": "Will the highest temperature in Seattle be between 84-85°F on June 24?",
            "slug": "highest-temperature-in-seattle-on-june-24-2026-84-85f",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.35","0.65"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
        market = parse_weather_market(raw, today=date(2026, 6, 1))
        assert market is not None

        replay_times = _candidate_replay_times(market, (0, 12), min_lead_days=1, max_lead_days=2)
        maintenance_times = [timestamp for timestamp, maintenance_only in replay_times if maintenance_only]
        entry_times = [timestamp for timestamp, maintenance_only in replay_times if not maintenance_only]

        self.assertTrue(maintenance_times)
        self.assertTrue(entry_times)
        city_zone = ZoneInfo(market.city.timezone)
        self.assertTrue(all(timestamp.astimezone(city_zone).date() == market.target_date for timestamp in maintenance_times))
        self.assertTrue(all((market.target_date - timestamp.astimezone(city_zone).date()).days >= 1 for timestamp in entry_times))

    def test_historical_observed_high_for_session_uses_target_day_partial_station_high(self) -> None:
        raw = {
            "id": "m2",
            "question": "Will the highest temperature in New York City be between 82-83°F on June 24?",
            "slug": "highest-temperature-in-nyc-on-june-24-2026-82-83f",
            "description": "Resolution source: https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
            "resolutionSource": "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.45","0.55"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
        market = parse_weather_market(raw, today=date(2026, 6, 1))
        assert market is not None
        session_time = datetime(2026, 6, 24, 17, 0, tzinfo=timezone.utc)

        class FakeObservationClient:
            def fetch_partial_historical_high(self, city, station_id, target_date, now):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=82.0,
                    source=f"historical_metar_{station_id}",
                    observed_at=now,
                    sample_count=7,
                    is_actual=True,
                    is_final=False,
                )

        observed = _historical_observed_high_for_session(
            market,
            raw,
            FakeObservationClient(),
            {},
            session_time,
        )

        assert observed is not None
        self.assertEqual(observed.max_temperature_f, 82.0)
        self.assertFalse(observed.is_final)
        self.assertEqual(observed.source, "historical_metar_KLGA")

    def test_maintenance_only_rows_do_not_open_new_positions(self) -> None:
        generated_at = datetime(2026, 6, 24, 17, 0, tzinfo=timezone.utc)
        outcome = ScoredOutcome(
            market_id="market",
            market_slug="market-slug",
            question="Will the highest temperature in Seattle be 90°F or higher on June 24?",
            bucket_label="90°F or higher",
            token_id="token",
            fair_value=0.9,
            market_price=0.5,
            edge=0.39,
            model_count=1,
            model_agreement=1.0,
            probability_stdev=0.0,
            generated_at=generated_at,
            city="Seattle, WA",
            target_date=date(2026, 6, 24),
            rule_excerpt="",
            model_probabilities={"fixture.model": 0.9},
        )
        positions: dict[str, HistoricalPosition] = {}
        executions: list[dict[str, object]] = []
        cash_ref = {"cash": 100.0}

        _rebalance_session(
            [outcome],
            {"token"},
            {"token": {"maintenance_only": True}},
            positions,
            executions,
            cash_ref,
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            compound_kelly_sizing=True,
            max_position_usd=100.0,
            max_position_fraction=0.25,
            kelly_market_blend=0.0,
            edge_position_full_cap_edge=0.25,
            edge_position_min_multiplier=0.35,
            min_trade_usd=1.0,
            settings=SignalSettings(),
        )

        self.assertEqual({}, positions)
        self.assertEqual([], executions)
        self.assertEqual(100.0, cash_ref["cash"])

    def test_load_calibration_weights_aliases_legacy_gfs_source(self) -> None:
        payload = {
            "source_weights": {"open_meteo_gfs_hrrr": 0.62},
            "model_weights": {"open_meteo_gfs_hrrr.kernel_wide": 1.08},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            source_weights, model_weights = load_calibration_weights(path)

        self.assertEqual(source_weights["open_meteo_gfs_hrrr"], 0.62)
        self.assertEqual(source_weights["open_meteo_gfs_best_match"], 0.62)
        self.assertEqual(model_weights["open_meteo_gfs_hrrr.kernel_wide"], 1.08)
        self.assertEqual(model_weights["open_meteo_gfs_best_match.kernel_wide"], 1.08)

    def test_recorded_backtest_applies_no_side_min_edge_floor(self) -> None:
        row = ResolvedForecastRow(
            id=1,
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 20),
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.78,
            market_price=0.70,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.0,
            entry_eligible=True,
            observed_outcome=1,
            model_probabilities={"gfs.kernel": 0.78, "ecmwf.kernel": 0.78, "best.kernel": 0.78},
        )
        settings = SignalSettings(
            min_edge=0.08,
            no_side_min_edge=0.10,
            no_side_max_counter_event_probability=0.30,
            min_signal_fair_value=0.70,
            min_model_agreement=1.0,
            min_price=0.125,
            yes_side_min_price=0.20,
        )

        blocked = _single_entry_kelly_replay([row], {}, {}, 100.0, 0.25, 50.0, 1.0, settings)
        allowed = _single_entry_kelly_replay([row], {}, {}, 100.0, 0.25, 50.0, 1.0, replace(settings, no_side_min_edge=0.05))

        self.assertEqual(blocked["trades"], 0)
        self.assertEqual(allowed["trades"], 1)

    def test_recorded_backtest_relaxes_no_side_edge_for_high_confidence_prices(self) -> None:
        row = ResolvedForecastRow(
            id=1,
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 20),
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.95,
            market_price=0.90,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.0,
            entry_eligible=True,
            observed_outcome=1,
            model_probabilities={"gfs.kernel": 0.95, "ecmwf.kernel": 0.95, "best.kernel": 0.95},
        )
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.0,
            no_side_min_edge=0.10,
            no_side_high_confidence_min_edge=0.02,
            no_side_max_counter_event_probability=0.08,
            min_signal_fair_value=0.70,
            min_model_agreement=1.0,
            min_price=0.125,
            yes_side_min_price=0.20,
        )

        result = _single_entry_kelly_replay([row], {}, {}, 100.0, 0.25, 50.0, 1.0, settings)

        self.assertEqual(result["trades"], 1)

    def test_recorded_backtest_applies_no_side_max_price_only_to_no_tokens(self) -> None:
        row = ResolvedForecastRow(
            id=1,
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 20),
            bucket_label="NO: 25°C or higher",
            token_id="no-token",
            fair_value=0.99,
            market_price=0.91,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.0,
            entry_eligible=True,
            observed_outcome=1,
            model_probabilities={"gfs.kernel": 0.99, "ecmwf.kernel": 0.99, "best.kernel": 0.99},
        )
        settings = SignalSettings(
            min_edge=0.08,
            uncertainty_buffer=0.0,
            no_side_min_edge=0.10,
            no_side_high_confidence_min_edge=0.02,
            no_side_max_price=0.90,
            no_side_max_counter_event_probability=0.10,
            min_signal_fair_value=0.70,
            min_model_agreement=1.0,
            min_price=0.125,
            yes_side_min_price=0.20,
        )

        blocked = _single_entry_kelly_replay([row], {}, {}, 100.0, 0.25, 50.0, 1.0, settings)
        allowed_no = _single_entry_kelly_replay([row], {}, {}, 100.0, 0.25, 50.0, 1.0, replace(settings, no_side_max_price=0.95))
        allowed_yes = _single_entry_kelly_replay(
            [replace(row, bucket_label="25°C or higher", token_id="yes-token")],
            {},
            {},
            100.0,
            0.25,
            50.0,
            1.0,
            settings,
        )

        self.assertEqual(blocked["trades"], 0)
        self.assertEqual(allowed_no["trades"], 1)
        self.assertEqual(allowed_yes["trades"], 1)

    def test_recorded_backtest_applies_fractional_position_cap(self) -> None:
        row = ResolvedForecastRow(
            id=1,
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 20),
            bucket_label="25°C or higher",
            token_id="yes-token",
            fair_value=1.0,
            market_price=0.50,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.0,
            entry_eligible=True,
            observed_outcome=1,
            model_probabilities={"gfs.kernel": 1.0, "ecmwf.kernel": 1.0, "best.kernel": 1.0},
        )
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_signal_fair_value=0.70,
            min_model_agreement=1.0,
            min_price=0.0,
            yes_side_min_price=0.0,
        )

        result = _single_entry_kelly_replay(
            [row],
            {},
            {},
            100.0,
            1.0,
            100.0,
            1.0,
            settings,
            max_position_fraction=0.25,
        )

        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["pnl_usd"], 25.0)

    def test_recorded_backtest_blends_fair_value_toward_market_for_sizing_only(self) -> None:
        row = ResolvedForecastRow(
            id=1,
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 20),
            bucket_label="25°C or higher",
            token_id="yes-token",
            fair_value=0.90,
            market_price=0.50,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.0,
            entry_eligible=True,
            observed_outcome=1,
            model_probabilities={"gfs.kernel": 0.90, "ecmwf.kernel": 0.90, "best.kernel": 0.90},
        )
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_signal_fair_value=0.70,
            min_model_agreement=1.0,
            min_price=0.0,
            yes_side_min_price=0.0,
        )

        result = _single_entry_kelly_replay(
            [row],
            {},
            {},
            100.0,
            1.0,
            100.0,
            1.0,
            settings,
            kelly_market_blend=0.50,
        )

        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["total_notional_usd"], 40.0)
        self.assertEqual(result["pnl_usd"], 40.0)
        self.assertEqual(result["top_trades"][0]["sizing_fair_value"], 0.70)

    def test_recorded_backtest_can_scale_position_cap_by_edge(self) -> None:
        row = ResolvedForecastRow(
            id=1,
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 6, 20),
            bucket_label="25°C or higher",
            token_id="yes-token",
            fair_value=0.90,
            market_price=0.50,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.0,
            entry_eligible=True,
            observed_outcome=1,
            model_probabilities={"gfs.kernel": 0.90, "ecmwf.kernel": 0.90, "best.kernel": 0.90},
        )
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_signal_fair_value=0.70,
            min_model_agreement=1.0,
            min_price=0.0,
            yes_side_min_price=0.0,
        )

        result = _single_entry_kelly_replay(
            [row],
            {},
            {},
            100.0,
            1.0,
            100.0,
            1.0,
            settings,
            edge_position_full_cap_edge=0.80,
            edge_position_min_multiplier=0.25,
        )

        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["total_notional_usd"], 50.0)
        self.assertEqual(result["pnl_usd"], 50.0)

    def test_select_entry_price_uses_only_prior_price_and_staleness(self) -> None:
        history = [
            PriceHistoryPoint(datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc), 0.20),
            PriceHistoryPoint(datetime(2026, 6, 18, 11, 0, tzinfo=timezone.utc), 0.30),
            PriceHistoryPoint(datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc), 0.40),
        ]

        selected = select_entry_price(
            history,
            datetime(2026, 6, 18, 11, 30, tzinfo=timezone.utc),
            max_staleness_minutes=60,
            slippage=0.01,
        )

        assert selected is not None
        self.assertEqual(selected.timestamp, datetime(2026, 6, 18, 11, 0, tzinfo=timezone.utc))
        self.assertAlmostEqual(selected.price, 0.31)
        self.assertIsNone(
            select_entry_price(
                history,
                datetime(2026, 6, 18, 11, 30, tzinfo=timezone.utc),
                max_staleness_minutes=20,
                slippage=0.01,
            )
        )

    def test_forecast_run_time_applies_availability_lag_and_six_hour_cycle(self) -> None:
        run_time = forecast_run_time(datetime(2026, 6, 18, 12, 10, tzinfo=timezone.utc), availability_lag_hours=6)
        self.assertEqual(run_time, datetime(2026, 6, 18, 6, 0, tzinfo=timezone.utc))

    def test_long_backtest_defaults_match_research_replay(self) -> None:
        signature = inspect.signature(run_long_historical_backtest)

        self.assertEqual(signature.parameters["max_markets"].default, 8000)
        self.assertEqual(signature.parameters["max_runtime_seconds"].default, 0.0)
        self.assertEqual(signature.parameters["max_price_staleness_minutes"].default, 90)
        self.assertEqual(signature.parameters["price_source"].default, "telonex")

    def test_long_backtest_run_log_paths_are_unique_for_parameter_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _make_run_log_path(directory, "long-backtest")
            second = _make_run_log_path(directory, "long-backtest")

        self.assertNotEqual(first, second)
        self.assertTrue(first.name.endswith("-long-backtest.json"))
        self.assertTrue(second.name.endswith("-long-backtest.json"))

    def test_scored_json_distinguishes_timing_from_signal_eligibility(self) -> None:
        outcome = ScoredOutcome(
            market_id="market-1",
            market_slug="market-1",
            question="Will New York City be 66F or higher?",
            bucket_label="66F or higher",
            token_id="cheap-yes",
            fair_value=0.85,
            market_price=0.013,
            edge=0.80,
            model_count=3,
            model_agreement=1.0,
            probability_stdev=0.02,
            generated_at=datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc),
            city="New York, NY",
            target_date=date(2026, 4, 12),
            rule_excerpt="",
            model_probabilities={"open_meteo_gfs.kernel": 0.85, "open_meteo_ecmwf.kernel": 0.86, "open_meteo_icon.kernel": 0.84},
            entry_eligible=True,
        )

        row = _scored_to_json(
            outcome,
            {},
            SignalSettings(min_price=0.125, yes_side_min_price=0.20, min_signal_fair_value=0.70),
        )

        self.assertTrue(row["timing_entry_eligible"])
        self.assertTrue(row["entry_eligible"])
        self.assertFalse(row["passes_signal_filter"])
        self.assertFalse(row["signal_eligible"])
        self.assertFalse(row["trade_eligible"])
        self.assertEqual(row["signal_filter_reason"], "market price below 0.125")

    def test_json_kelly_replay_can_compound_from_current_equity(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-04T12:00:00+00:00",
                "token_id": "tok-1",
                "city": "New York, NY",
                "target_date": "2026-06-05",
                "market_price": 0.50,
                "fair_value": 1.0,
                "edge": 0.49,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
            },
            {
                "generated_at": "2026-06-06T12:00:00+00:00",
                "token_id": "tok-2",
                "city": "New York, NY",
                "target_date": "2026-06-07",
                "market_price": 0.50,
                "fair_value": 1.0,
                "edge": 0.49,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
            },
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.5,
            enforce_entry_timing_filter=False,
        )

        fixed = _json_kelly_replay(
            "fixed",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=0.50,
            compound_kelly_sizing=False,
            max_position_usd=100,
            min_trade_usd=1,
        )
        compounded = _json_kelly_replay(
            "compounded",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=0.50,
            compound_kelly_sizing=True,
            max_position_usd=100,
            min_trade_usd=1,
        )

        self.assertEqual(fixed["pnl_usd"], 100.0)
        self.assertEqual(compounded["pnl_usd"], 125.0)
        self.assertGreater(compounded["buy_notional_usd"], fixed["buy_notional_usd"])

    def test_json_kelly_replay_applies_fractional_position_cap(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-04T12:00:00+00:00",
                "token_id": "tok-1",
                "city": "New York, NY",
                "target_date": "2026-06-05",
                "market_price": 0.50,
                "fair_value": 1.0,
                "edge": 0.50,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
            }
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.5,
            enforce_entry_timing_filter=False,
        )

        result = _json_kelly_replay(
            "fraction-capped",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=100,
            max_position_fraction=0.25,
            min_trade_usd=1,
        )

        self.assertEqual(result["buy_notional_usd"], 25.0)
        self.assertEqual(result["pnl_usd"], 25.0)
        self.assertEqual(result["settings"]["max_position_fraction"], 0.25)

    def test_effective_position_cap_scales_after_fractional_cap(self) -> None:
        helpers = (
            recorded_effective_max_position_usd,
            long_effective_max_position_usd,
            paper_effective_max_position_usd,
        )
        for helper in helpers:
            with self.subTest(helper=helper.__module__):
                low_edge_cap = helper(
                    100.0,
                    100.0,
                    0.05,
                    edge=0.05,
                    edge_position_full_cap_edge=0.25,
                    edge_position_min_multiplier=0.35,
                )
                full_edge_cap = helper(
                    100.0,
                    100.0,
                    0.05,
                    edge=0.25,
                    edge_position_full_cap_edge=0.25,
                    edge_position_min_multiplier=0.35,
                )

                self.assertAlmostEqual(low_edge_cap, 1.75)
                self.assertAlmostEqual(full_edge_cap, 5.0)

    def test_json_kelly_replay_blends_fair_value_toward_market_for_sizing_only(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-04T12:00:00+00:00",
                "token_id": "tok-1",
                "city": "New York, NY",
                "target_date": "2026-06-05",
                "market_price": 0.50,
                "fair_value": 0.90,
                "edge": 0.40,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
            }
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.7,
            enforce_entry_timing_filter=False,
        )

        result = _json_kelly_replay(
            "market-blended",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=100,
            kelly_market_blend=0.50,
            min_trade_usd=1,
        )

        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["buy_notional_usd"], 40.0)
        self.assertEqual(result["pnl_usd"], 40.0)
        self.assertEqual(result["settings"]["kelly_market_blend"], 0.50)

    def test_json_kelly_replay_can_scale_position_cap_by_edge(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-04T12:00:00+00:00",
                "token_id": "tok-1",
                "city": "New York, NY",
                "target_date": "2026-06-05",
                "market_price": 0.50,
                "fair_value": 0.90,
                "edge": 0.40,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
            }
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.7,
            enforce_entry_timing_filter=False,
        )

        result = _json_kelly_replay(
            "edge-scaled",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=100,
            edge_position_full_cap_edge=0.80,
            edge_position_min_multiplier=0.25,
            min_trade_usd=1,
        )

        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["buy_notional_usd"], 50.0)
        self.assertEqual(result["pnl_usd"], 50.0)
        self.assertEqual(result["settings"]["edge_position_full_cap_edge"], 0.80)

    def test_json_kelly_replay_separates_profitable_trades_from_event_wins(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-04T12:00:00+00:00",
                "token_id": "tok-1",
                "city": "New York, NY",
                "target_date": "2026-06-05",
                "market_price": 0.50,
                "exit_price": 0.80,
                "fair_value": 0.90,
                "edge": 0.39,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 0,
                "payout": 0,
            },
            {
                "generated_at": "2026-06-04T18:00:00+00:00",
                "token_id": "tok-1",
                "city": "New York, NY",
                "target_date": "2026-06-05",
                "market_price": 0.82,
                "exit_price": 0.80,
                "fair_value": 0.60,
                "edge": -0.23,
                "model_count": 3,
                "model_agreement": 0.0,
                "entry_eligible": True,
                "polymarket_payout": 0,
                "payout": 0,
            },
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.7,
            hold_min_fair_value=0.7,
            enforce_entry_timing_filter=False,
        )

        result = _json_kelly_replay(
            "event-loss-profitable-exit",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
        )

        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["winning_trades"], 1)
        self.assertEqual(result["event_winning_trades"], 0)
        self.assertEqual(result["event_losing_trades"], 1)
        self.assertEqual(result["event_hit_rate"], 0.0)
        self.assertEqual(result["profitable_event_loser_trades"], 1)
        self.assertGreater(result["event_loss_pnl_usd"], 0)
        self.assertEqual(result["exit_management"]["sell_count"], 1)
        self.assertEqual(result["exit_management"]["event_loser_sell_value_usd"], 16.0)
        self.assertEqual(result["exit_management"]["sell_decision_value_vs_settlement_usd"], 16.0)
        self.assertEqual(result["exit_management"]["by_final_payout"][0]["group"], 0)
        self.assertEqual(result["exit_management"]["by_final_payout"][0]["decision_value_vs_settlement_usd"], 16.0)
        self.assertEqual(result["exit_management"]["by_sell_price_bucket"][0]["group"], 0.8)
        forced_hold = _json_kelly_replay(
            "forced-hold",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
            force_hold_existing_positions=True,
        )
        self.assertEqual(forced_hold["sells"], 0)
        self.assertLess(forced_hold["pnl_usd"], result["pnl_usd"])
        self.assertTrue(forced_hold["settings"]["force_hold_existing_positions"])

    def test_selected_candidate_weather_validation_surfaces_ambiguous_and_mismatched_rows(self) -> None:
        base = {
            "market_price": 0.50,
            "fair_value": 0.90,
            "edge": 0.39,
            "model_count": 3,
            "model_agreement": 1.0,
            "entry_eligible": True,
            "bucket_lower_f": 90.0,
            "bucket_upper_f": 91.0,
            "side": "NO",
        }
        rows = [
            {
                **base,
                "generated_at": "2026-06-01T12:00:00+00:00",
                "token_id": "ambiguous",
                "question": "NO: Will the highest temperature in Seoul be 32°C on June 2?",
                "city": "Seoul, KR",
                "target_date": "2026-06-02",
                "polymarket_payout": 0,
                "weather_outcome": None,
                "weather_ambiguous": True,
                "settlement_source": "historical_metar_RKSI_ambiguous_resolution",
            },
            {
                **base,
                "generated_at": "2026-06-01T12:00:00+00:00",
                "token_id": "mismatch",
                "question": "NO: Will the highest temperature in Miami be between 92-93°F on June 2?",
                "city": "Miami, FL",
                "target_date": "2026-06-02",
                "polymarket_payout": 1,
                "weather_outcome": 0,
                "weather_ambiguous": False,
                "settlement_source": "historical_metar_KMIA",
            },
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.70,
            no_side_min_edge=0.0,
            no_side_max_counter_event_probability=1.0,
            bounded_bucket_min_price=0.0,
            bounded_bucket_min_edge=0.0,
            bounded_bucket_min_fair_value=0.0,
            enforce_entry_timing_filter=False,
        )

        diagnostics = _selected_candidate_weather_validation(rows, settings)

        self.assertEqual(diagnostics["selected_candidate_count"], 2)
        self.assertEqual(diagnostics["weather_ambiguous_count"], 1)
        self.assertEqual(diagnostics["weather_mismatch_count"], 1)
        self.assertEqual(diagnostics["quality"]["weather_ambiguous"], 1)
        self.assertEqual(diagnostics["quality"]["weather_mismatches"], 1)
        self.assertEqual(diagnostics["ambiguous_examples"][0]["token_id"], "ambiguous")
        self.assertEqual(diagnostics["mismatch_examples"][0]["token_id"], "mismatch")

    def test_json_kelly_replay_can_partially_exit_high_fair_value_invalid_holds(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-01T12:00:00+00:00",
                "token_id": "event-winner",
                "city": "New York, NY",
                "target_date": "2026-06-02",
                "market_price": 0.50,
                "fair_value": 0.95,
                "edge": 0.44,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
            },
            {
                "generated_at": "2026-06-01T18:00:00+00:00",
                "token_id": "event-winner",
                "city": "New York, NY",
                "target_date": "2026-06-02",
                "market_price": 0.70,
                "exit_price": 0.70,
                "fair_value": 0.95,
                "edge": 0.24,
                "model_count": 3,
                "model_agreement": 0.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
            },
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.70,
            hold_min_model_agreement=0.65,
            enforce_entry_timing_filter=False,
        )

        full_exit = _json_kelly_replay(
            "full-exit",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
        )
        partial_exit = _json_kelly_replay(
            "partial-exit",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
            invalid_hold_partial_exit_fraction=0.5,
            invalid_hold_partial_exit_min_fair_value=0.90,
            invalid_hold_partial_exit_min_price=0.50,
            invalid_hold_partial_exit_max_price=0.80,
        )

        self.assertEqual(full_exit["sells"], 1)
        self.assertEqual(partial_exit["sells"], 1)
        self.assertGreater(partial_exit["pnl_usd"], full_exit["pnl_usd"])
        self.assertEqual(partial_exit["exit_management"]["event_winner_sell_drag_usd"], -3.0)
        self.assertEqual(partial_exit["settings"]["invalid_hold_partial_exit_fraction"], 0.5)

    def test_json_kelly_replay_uses_wider_no_side_hold_tail_without_weakening_entry_tail(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-15T12:00:00+00:00",
                "token_id": "no-token",
                "city": "Seoul, KR",
                "target_date": "2026-06-17",
                "bucket": "NO: Will the highest temperature in Seoul be 31°C or higher on June 17?",
                "side": "NO",
                "market_price": 0.80,
                "exit_price": 0.78,
                "fair_value": 0.95,
                "edge": 0.15,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
                "model_probabilities": {"a.model": 0.93, "b.model": 0.94, "c.model": 0.95},
            },
            {
                "generated_at": "2026-06-16T00:00:00+00:00",
                "token_id": "no-token",
                "city": "Seoul, KR",
                "target_date": "2026-06-17",
                "bucket": "NO: Will the highest temperature in Seoul be 31°C or higher on June 17?",
                "side": "NO",
                "market_price": 0.70,
                "exit_price": 0.70,
                "fair_value": 0.90,
                "edge": 0.20,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
                "model_probabilities": {"a.model": 0.88, "b.model": 0.91, "c.model": 0.92},
            },
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.70,
            no_side_max_counter_event_probability=0.10,
            hold_no_side_max_counter_event_probability=0.15,
            enforce_entry_timing_filter=False,
        )

        held = _json_kelly_replay(
            "wider-hold-tail",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
        )
        sold = _json_kelly_replay(
            "entry-tail-for-holds",
            rows,
            replace(settings, hold_no_side_max_counter_event_probability=0.09),
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
        )

        self.assertEqual(held["sells"], 0)
        self.assertEqual(held["settlements"], 1)
        self.assertGreater(held["pnl_usd"], sold["pnl_usd"])
        self.assertEqual(sold["sells"], 1)

    def test_json_kelly_replay_high_conviction_no_side_holds_can_use_wider_tail(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-15T12:00:00+00:00",
                "token_id": "no-token",
                "city": "Seoul, KR",
                "target_date": "2026-06-17",
                "bucket": "NO: Will the highest temperature in Seoul be 31°C or higher on June 17?",
                "side": "NO",
                "market_price": 0.50,
                "exit_price": 0.50,
                "fair_value": 0.99,
                "edge": 0.48,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
                "model_probabilities": {"a.model": 0.99, "b.model": 0.99, "c.model": 0.99},
            },
            {
                "generated_at": "2026-06-16T12:00:00+00:00",
                "token_id": "no-token",
                "city": "Seoul, KR",
                "target_date": "2026-06-17",
                "bucket": "NO: Will the highest temperature in Seoul be 31°C or higher on June 17?",
                "side": "NO",
                "market_price": 0.62,
                "exit_price": 0.62,
                "fair_value": 0.99,
                "edge": 0.36,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
                "model_probabilities": {"a.model": 0.82, "b.model": 0.91, "c.model": 0.92},
            },
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.70,
            no_side_min_edge=0.0,
            no_side_max_counter_event_probability=0.10,
            hold_no_side_max_counter_event_probability=0.15,
            enforce_entry_timing_filter=False,
        )

        base = _json_kelly_replay(
            "base-hold-tail",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
        )
        high_conviction = _json_kelly_replay(
            "high-conviction-hold-tail",
            rows,
            replace(
                settings,
                hold_no_side_high_conviction_min_fair_value=0.98,
                hold_no_side_high_conviction_min_edge=0.35,
                hold_no_side_high_conviction_counter_event_probability=0.20,
            ),
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=10,
            min_trade_usd=1,
        )

        self.assertEqual(base["sells"], 1)
        self.assertEqual(high_conviction["sells"], 0)
        self.assertEqual(high_conviction["settlements"], 1)
        self.assertGreater(high_conviction["pnl_usd"], base["pnl_usd"])

    def test_score_calibration_diagnostics_include_source_and_signal_eligible_model_accuracy(self) -> None:
        rows = [
            {
                "polymarket_payout": 1,
                "fair_value": 0.82,
                "market_price": 0.60,
                "edge": 0.20,
                "model_agreement": 1.0,
                "signal_filter_reason": None,
                "model_probabilities": {
                    "single_run_gfs_global.kernel_tight": 0.80,
                    "single_run_ecmwf_ifs025.kernel_tight": 0.90,
                    "single_run_ecmwf_ifs025.feature_aware": 0.85,
                },
            },
            {
                "polymarket_payout": 0,
                "fair_value": 0.78,
                "market_price": 0.55,
                "edge": 0.21,
                "model_agreement": 1.0,
                "signal_filter_reason": "bounded exact/range bucket edge below 0.15",
                "model_probabilities": {
                    "single_run_gfs_global.kernel_tight": 0.70,
                    "single_run_ecmwf_ifs025.kernel_tight": 0.20,
                    "single_run_ecmwf_ifs025.feature_aware": 0.30,
                },
            },
        ]

        diagnostics = _score_calibration_diagnostics(rows)

        self.assertEqual(diagnostics["resolved_count"], 2)
        self.assertEqual(diagnostics["signal_eligible_count"], 1)
        by_source = {row["source"]: row for row in diagnostics["source_probability_accuracy"]}
        self.assertEqual(set(by_source), {"single_run_gfs_global", "single_run_ecmwf_ifs025"})
        self.assertGreater(by_source["single_run_gfs_global"]["brier"], by_source["single_run_ecmwf_ifs025"]["brier"])
        by_family = {row["model_family"]: row for row in diagnostics["model_family_probability_accuracy"]}
        self.assertEqual(set(by_family), {"kernel_tight", "feature_aware"})
        eligible_sources = {row["source"]: row for row in diagnostics["signal_eligible_source_probability_accuracy"]}
        self.assertEqual(eligible_sources["single_run_gfs_global"]["actual_rate"], 1.0)
        self.assertEqual(eligible_sources["single_run_ecmwf_ifs025"]["actual_rate"], 1.0)

    def test_robustness_diagnostics_include_time_slices_and_candidate_calibration(self) -> None:
        rows = [
            {
                "generated_at": "2026-02-01T12:00:00+00:00",
                "token_id": "tok-1",
                "city": "New York, NY",
                "target_date": "2026-02-02",
                "market_price": 0.50,
                "fair_value": 0.90,
                "edge": 0.40,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 1,
                "payout": 1,
                "side": "YES",
            },
            {
                "generated_at": "2026-03-01T12:00:00+00:00",
                "token_id": "tok-2",
                "city": "Chicago, IL",
                "target_date": "2026-03-02",
                "market_price": 0.50,
                "fair_value": 0.90,
                "edge": 0.40,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "polymarket_payout": 0,
                "payout": 0,
                "side": "YES",
            },
        ]
        settings = SignalSettings(
            min_edge=0.0,
            uncertainty_buffer=0.0,
            min_price=0.0,
            yes_side_min_price=0.0,
            min_signal_fair_value=0.7,
            enforce_entry_timing_filter=False,
        )

        diagnostics = _robustness_diagnostics(
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=1.0,
            compound_kelly_sizing=False,
            max_position_usd=100,
            max_position_fraction=None,
            kelly_market_blend=0.0,
            edge_position_full_cap_edge=0.0,
            edge_position_min_multiplier=0.35,
            min_trade_usd=1,
        )

        self.assertEqual(diagnostics["selected_candidate_count"], 2)
        self.assertEqual(diagnostics["selected_candidate_calibration"]["n"], 2)
        self.assertEqual(diagnostics["selected_candidate_calibration"]["actual_rate"], 0.5)
        slice_names = {row["slice"] for row in diagnostics["by_chronological_session_slice"]}
        self.assertIn("first_50pct_sessions", slice_names)
        self.assertIn("last_30pct_sessions", slice_names)
        month_names = {row["slice"] for row in diagnostics["by_entry_month_replay"]}
        self.assertEqual(month_names, {"entry_month_2026-02", "entry_month_2026-03"})

    def test_polymarket_yes_payout_from_resolved_outcome_prices(self) -> None:
        self.assertEqual(_polymarket_yes_payout({"outcomes": '["Yes","No"]', "outcomePrices": '["1","0"]'}), 1)
        self.assertEqual(_polymarket_yes_payout({"outcomes": '["Yes","No"]', "outcomePrices": '["0","1"]'}), 0)
        self.assertIsNone(_polymarket_yes_payout({"outcomes": '["Yes","No"]', "outcomePrices": '["0.42","0.58"]'}))

    def test_binary_no_token_extracts_real_no_clob_token(self) -> None:
        token = _binary_no_token(
            {
                "outcomes": '["Yes","No"]',
                "clobTokenIds": '["yes-token","no-token"]',
            }
        )

        self.assertEqual(token, "no-token")
        self.assertIsNone(_binary_no_token({"outcomes": '["Yes"]', "clobTokenIds": '["yes-token"]'}))

    def test_invert_binary_scored_outcome_prices_real_no_probability(self) -> None:
        outcome = ScoredOutcome(
            market_id="market-1",
            market_slug="slug",
            question="Will the highest temperature in Seoul be 25°C or higher on May 28?",
            bucket_label="25°C or higher",
            token_id="no-token",
            fair_value=0.20,
            market_price=0.70,
            edge=-0.509,
            model_count=3,
            model_agreement=0.0,
            probability_stdev=0.05,
            generated_at=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
            city="Seoul, KR",
            target_date=date(2026, 5, 28),
            rule_excerpt="rules",
            model_probabilities={"gfs.kernel": 0.10, "ecmwf.kernel": 0.15, "best.kernel": 0.20},
            observed_outcome=0,
        )

        inverted = _invert_binary_scored_outcome(outcome, SignalSettings())

        self.assertEqual(inverted.question, f"NO: {outcome.question}")
        self.assertEqual(inverted.bucket_label, "NO: 25°C or higher")
        self.assertEqual(inverted.token_id, "no-token")
        self.assertAlmostEqual(inverted.fair_value, 0.80)
        self.assertAlmostEqual(inverted.edge, 0.091)
        self.assertEqual(inverted.model_probabilities["gfs.kernel"], 0.90)
        self.assertEqual(inverted.observed_outcome, 1)
        self.assertEqual(inverted.model_agreement, 1.0)

    def test_price_history_bounds_use_market_lifetime_with_padding(self) -> None:
        start_ts, end_ts = _price_history_bounds(
            {
                "acceptingOrdersTimestamp": "2026-04-09T04:30:39Z",
                "createdAt": "2026-04-09T04:08:45.129113Z",
                "closedTime": "2026-04-13 02:20:55+00",
                "updatedAt": "2026-04-14T02:22:00.931116Z",
            }
        )

        self.assertEqual(start_ts, 1775664525)
        self.assertEqual(end_ts, 1776176520)

    def test_price_history_bounds_use_telonex_quote_availability_without_padding(self) -> None:
        start_ts, end_ts = _price_history_bounds(
            {
                "startDate": "2026-01-01T00:00:00+00:00",
                "endDate": "2026-02-01T00:00:00+00:00",
                "telonex_quotes_from": "2026-01-19",
                "telonex_quotes_to": "2026-01-21",
            }
        )

        self.assertEqual(start_ts, 1768780800)
        self.assertEqual(end_ts, 1769040000)

    def test_price_history_bounds_for_replay_times_only_cover_decision_window(self) -> None:
        replay_times = [
            (datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc), False),
            (datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc), False),
            (datetime(2026, 1, 21, 18, 0, tzinfo=timezone.utc), True),
        ]

        start_ts, end_ts = _price_history_bounds_for_replay_times(
            replay_times,
            max_staleness_minutes=90,
        )

        self.assertEqual(start_ts, int((replay_times[0][0] - timedelta(minutes=90)).timestamp()))
        self.assertEqual(end_ts, int((replay_times[-1][0] + timedelta(seconds=1)).timestamp()))

    def test_cached_price_history_does_not_refetch_existing_token(self) -> None:
        class FakeClob:
            def __init__(self) -> None:
                self.calls = 0

            def fetch_price_history(self, token_id, *, interval, fidelity, start_ts, end_ts):
                self.calls += 1
                return [PriceHistoryPoint(datetime(2026, 6, 18, 12, tzinfo=timezone.utc), 0.42)]

        clob = FakeClob()
        cache = {}

        first = _cached_price_history(cache, clob, "token-1", start_ts=1, end_ts=2)
        second = _cached_price_history(cache, clob, "token-1", start_ts=1, end_ts=2)

        self.assertIs(first, second)
        self.assertEqual(clob.calls, 1)

    def test_cached_price_history_can_use_telonex_quote_source(self) -> None:
        class FakeClob:
            def __init__(self) -> None:
                self.calls = 0

            def fetch_price_history(self, token_id, *, interval, fidelity, start_ts, end_ts):
                self.calls += 1
                return []

        class FakeTelonex:
            def __init__(self) -> None:
                self.calls = []

            def fetch_quote_price_history(self, *, slug, outcome, start_ts, end_ts, token_id=None):
                self.calls.append(
                    {"slug": slug, "outcome": outcome, "start_ts": start_ts, "end_ts": end_ts, "token_id": token_id}
                )
                return [PriceHistoryPoint(datetime(2026, 6, 18, 12, tzinfo=timezone.utc), 0.43)]

        clob = FakeClob()
        telonex = FakeTelonex()
        cache = {}

        first = _cached_price_history(
            cache,
            clob,
            "token-1",
            start_ts=1,
            end_ts=2,
            price_source="telonex",
            telonex=telonex,
            market_slug="market-slug",
            outcome="Yes",
        )
        second = _cached_price_history(
            cache,
            clob,
            "token-1",
            start_ts=1,
            end_ts=2,
            price_source="telonex",
            telonex=telonex,
            market_slug="market-slug",
            outcome="Yes",
        )

        self.assertIs(first, second)
        self.assertEqual(clob.calls, 0)
        self.assertEqual(len(telonex.calls), 1)
        self.assertEqual(telonex.calls[0]["slug"], "market-slug")
        self.assertEqual(telonex.calls[0]["outcome"], "Yes")

    def test_telonex_market_record_converts_to_parseable_weather_raw(self) -> None:
        record = {
            "market_id": "0xabc",
            "slug": "will-the-highest-temperature-in-london-be-68f-or-below-on-august-16",
            "event_slug": "weather",
            "event_title": "Highest temperature in London",
            "question": "Will the highest temperature in London be 68°F or below on August 16?",
            "description": "Resolved by official station.",
            "resolution_source": "station",
            "outcome_0": "Yes",
            "outcome_1": "No",
            "asset_id_0": "yes-token",
            "asset_id_1": "no-token",
            "status": "resolved",
            "result_id": "1",
            "start_date_us": 1755086400000000,
            "end_date_us": 1755345600000000,
            "created_at_us": 1755080000000000,
            "settled_at_us": 1755350000000000,
            "prepared_at_us": 1755086400000000,
            "quotes_from": "2025-08-13",
            "quotes_to": "2025-08-16",
        }

        raw = _telonex_market_record_to_raw(record)
        assert raw is not None
        parsed = parse_weather_market(raw, today=_market_parse_today(raw))

        self.assertEqual(json.loads(raw["outcomePrices"]), ["0", "1"])
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.target_date, date(2025, 8, 16))
        self.assertEqual(parsed.buckets[0].token_id, "yes-token")

    def test_runtime_limited_backtest_marks_open_positions_instead_of_force_settling(self) -> None:
        positions = {
            "token-1": HistoricalPosition(
                token_id="token-1",
                market_id="market-1",
                question="Will the highest temperature in Dallas be 68°F or higher on March 28?",
                bucket_label="68°F or higher",
                city="Dallas, TX",
                target_date=date(2026, 3, 28),
                shares=10.0,
                cost_basis=5.0,
                last_price=0.20,
                payout=1,
                weather_outcome=1,
                observed_high_f=70.0,
                settlement_source="historical_metar_KDAL",
            )
        }
        executions = []

        cash, open_value = _finalize_positions_for_result(positions, executions, 95.0, runtime_limited=True)

        self.assertEqual(cash, 95.0)
        self.assertEqual(open_value, 2.0)
        self.assertEqual(executions, [])
        self.assertIn("token-1", positions)

    def test_completed_backtest_settles_remaining_positions(self) -> None:
        positions = {
            "token-1": HistoricalPosition(
                token_id="token-1",
                market_id="market-1",
                question="Will the highest temperature in Dallas be 68°F or higher on March 28?",
                bucket_label="68°F or higher",
                city="Dallas, TX",
                target_date=date(2026, 3, 28),
                shares=10.0,
                cost_basis=5.0,
                last_price=0.20,
                payout=1,
                weather_outcome=1,
                observed_high_f=70.0,
                settlement_source="historical_metar_KDAL",
            )
        }
        executions = []

        cash, open_value = _finalize_positions_for_result(positions, executions, 95.0, runtime_limited=False)

        self.assertEqual(cash, 105.0)
        self.assertEqual(open_value, 0.0)
        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0]["action"], "SETTLE")
        self.assertNotIn("token-1", positions)

    def test_cached_http_invalidates_bad_json_and_retries(self) -> None:
        class FakeHttp:
            def __init__(self) -> None:
                self.calls = 0

            def get_text(self, url, params=None, headers=None):
                self.calls += 1
                return "" if self.calls == 1 else '{"ok": true}'

        fake = FakeHttp()
        with tempfile.TemporaryDirectory() as directory:
            client = CachedHttpClient(directory)
            client.http = fake

            payload = client.get_json("https://example.test/data", params={"q": "x"})

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(fake.calls, 2)

    def test_cached_http_hard_timeout_does_not_hang(self) -> None:
        class SlowHttp:
            def get_text(self, url, params=None, headers=None):
                time.sleep(2)
                return "{}"

        with tempfile.TemporaryDirectory() as directory:
            client = CachedHttpClient(directory, hard_timeout_seconds=0.05)
            client.http = SlowHttp()

            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "hard-timeout"):
                client.get_text("https://example.test/slow")

        self.assertLess(time.monotonic() - started, 0.5)

    def test_data_quality_diagnostics_flags_lookahead_and_staleness(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-18T12:00:00+00:00",
                "entry_price_timestamp": "2026-06-18T11:30:00+00:00",
                "entry_price_stale_seconds": 1800,
                "forecast_run_time": "2026-06-18T06:00:00+00:00",
            },
            {
                "generated_at": "2026-06-18T12:00:00+00:00",
                "entry_price_timestamp": "2026-06-18T12:01:00+00:00",
                "forecast_run_time": "2026-06-18T10:00:00+00:00",
            },
            {
                "generated_at": "2026-06-18T12:00:00+00:00",
                "entry_price_timestamp": "2026-06-18T09:00:00+00:00",
                "entry_price_stale_seconds": 10800,
                "forecast_run_time": "2026-06-18T06:00:00+00:00",
            },
        ]

        diagnostics = _data_quality_diagnostics(
            [],
            rows,
            max_price_staleness_minutes=120,
            forecast_availability_lag_hours=6,
        )
        scored = diagnostics["scored_rows"]

        self.assertEqual(scored["future_price_violations"], 1)
        self.assertEqual(scored["stale_price_violations"], 1)
        self.assertEqual(scored["future_forecast_violations"], 0)
        self.assertEqual(scored["unavailable_forecast_violations"], 1)
        self.assertEqual(scored["min_forecast_lag_seconds"], 7200.0)

    def test_settlement_quality_diagnostics_separates_weather_checked_from_ambiguous(self) -> None:
        scored = [
            {
                "token_id": "matched",
                "polymarket_payout": 1,
                "weather_outcome": 1,
                "observed_high_f": 81.0,
                "settlement_source": "historical_metar_TEST",
                "signal_filter_reason": None,
            },
            {
                "token_id": "mismatch",
                "polymarket_payout": 1,
                "weather_outcome": 0,
                "observed_high_f": 79.0,
                "settlement_source": "historical_metar_TEST",
                "signal_filter_reason": "below fair-value floor",
            },
            {
                "token_id": "ambiguous",
                "polymarket_payout": 0,
                "weather_ambiguous": True,
                "observed_high_f": 80.6,
                "settlement_source": "historical_metar_TEST_ambiguous_resolution",
                "signal_filter_reason": None,
            },
            {"token_id": "polymarket-only", "polymarket_payout": "1", "signal_filter_reason": None},
            {"token_id": "weather-only", "weather_outcome": 0, "settlement_source": "historical_metar_TEST", "signal_filter_reason": None},
            {"token_id": "unresolved", "signal_filter_reason": None},
        ]
        executions = [
            {
                "action": "BUY",
                "token_id": "matched",
                "polymarket_payout": 1,
                "weather_outcome": 1,
                "observed_high_f": 81.0,
                "settlement_source": "historical_metar_TEST",
            },
            {
                "action": "SETTLE",
                "token_id": "matched",
                "polymarket_payout": 1,
                "weather_outcome": 1,
                "observed_high_f": 81.0,
                "settlement_source": "historical_metar_TEST",
            },
            {
                "action": "BUY",
                "token_id": "ambiguous",
                "polymarket_payout": 0,
                "weather_ambiguous": True,
                "observed_high_f": 80.6,
                "settlement_source": "historical_metar_TEST_ambiguous_resolution",
            },
            {"action": "BUY", "token_id": "polymarket-only", "polymarket_payout": 1},
            {"action": "BUY", "token_id": "unresolved"},
        ]

        diagnostics = _settlement_quality_diagnostics(scored, executions)
        scored_rows = diagnostics["scored_rows"]
        traded_tokens = diagnostics["traded_tokens"]

        self.assertEqual(scored_rows["total_rows"], 6)
        self.assertEqual(scored_rows["polymarket_resolved_rows"], 4)
        self.assertEqual(scored_rows["weather_checked_rows"], 2)
        self.assertEqual(scored_rows["weather_matched_rows"], 1)
        self.assertEqual(scored_rows["weather_mismatch_rows"], 1)
        self.assertEqual(scored_rows["weather_ambiguous_rows"], 1)
        self.assertEqual(scored_rows["polymarket_only_rows"], 1)
        self.assertEqual(scored_rows["weather_only_rows"], 1)
        self.assertEqual(scored_rows["unresolved_rows"], 1)
        self.assertEqual(diagnostics["signal_eligible_rows"]["total_rows"], 5)
        self.assertEqual(traded_tokens["traded_token_count"], 4)
        self.assertEqual(traded_tokens["buy_execution_count"], 4)
        self.assertEqual(traded_tokens["polymarket_resolved_traded_tokens"], 3)
        self.assertEqual(traded_tokens["weather_checked_traded_tokens"], 1)
        self.assertEqual(traded_tokens["weather_matched_traded_tokens"], 1)
        self.assertEqual(traded_tokens["weather_mismatch_traded_tokens"], 0)
        self.assertEqual(traded_tokens["weather_ambiguous_traded_tokens"], 1)
        self.assertEqual(traded_tokens["polymarket_only_traded_tokens"], 1)
        self.assertEqual(traded_tokens["unresolved_traded_tokens"], 1)

    def test_real_data_audit_passes_clean_historical_replay_inputs(self) -> None:
        scored = [
            {
                "token_id": "yes-token",
                "side": "YES",
                "generated_at": "2026-06-02T12:00:00+00:00",
                "entry_price_timestamp": "2026-06-02T11:30:00+00:00",
                "entry_price_stale_seconds": 1800,
                "forecast_run_time": "2026-06-02T06:00:00+00:00",
                "forecast_sources": ["single_run_gfs_global", "single_run_ecmwf_ifs025"],
                "signal_filter_reason": None,
                "polymarket_payout": 1,
                "weather_outcome": 1,
                "observed_high_f": 81.0,
                "settlement_source": "historical_metar_KDAL",
            },
            {
                "token_id": "no-token",
                "yes_token_id": "yes-token-2",
                "side": "NO",
                "generated_at": "2026-06-03T12:00:00+00:00",
                "entry_price_timestamp": "2026-06-03T11:00:00+00:00",
                "entry_price_stale_seconds": 3600,
                "forecast_run_time": "2026-06-03T06:00:00+00:00",
                "forecast_sources": ["single_run_best_match"],
                "signal_filter_reason": None,
                "polymarket_payout": 1,
                "weather_outcome": 1,
                "observed_high_f": 77.0,
                "settlement_source": "historical_metar_KLGA",
            },
        ]
        executions = [
            {**scored[0], "action": "BUY"},
            {**scored[0], "action": "SETTLE"},
            {**scored[1], "action": "BUY"},
            {**scored[1], "action": "SETTLE"},
        ]
        data_quality = _data_quality_diagnostics(
            executions,
            scored,
            max_price_staleness_minutes=90,
            forecast_availability_lag_hours=6,
        )
        settlement_quality = _settlement_quality_diagnostics(scored, executions)

        audit = _real_data_audit(
            scored,
            executions,
            data_quality_diagnostics=data_quality,
            settlement_quality_diagnostics=settlement_quality,
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["failure_reasons"], [])
        self.assertTrue(audit["checks"]["no_side_rows_use_explicit_no_tokens"]["passed"])
        self.assertTrue(audit["checks"]["traded_tokens_weather_matched"]["passed"])

    def test_real_data_audit_flags_fixture_or_unmatched_replay_inputs(self) -> None:
        scored = [
            {
                "token_id": "no-token",
                "side": "NO",
                "generated_at": "2026-06-02T12:00:00+00:00",
                "entry_price_timestamp": "2026-06-02T13:00:00+00:00",
                "entry_price_stale_seconds": 7200,
                "forecast_run_time": "2026-06-02T10:00:00+00:00",
                "forecast_sources": ["fixture"],
                "signal_filter_reason": None,
                "polymarket_payout": 1,
                "weather_outcome": 0,
                "observed_high_f": 81.0,
                "settlement_source": "historical_metar_KDAL",
            }
        ]
        executions = [{**scored[0], "action": "BUY"}]
        data_quality = _data_quality_diagnostics(
            executions,
            scored,
            max_price_staleness_minutes=90,
            forecast_availability_lag_hours=6,
        )
        settlement_quality = _settlement_quality_diagnostics(scored, executions)

        audit = _real_data_audit(
            scored,
            executions,
            data_quality_diagnostics=data_quality,
            settlement_quality_diagnostics=settlement_quality,
        )

        self.assertFalse(audit["passed"])
        self.assertIn("scored_rows_use_open_meteo_single_runs", audit["failure_reasons"])
        self.assertIn("no_future_or_stale_prices", audit["failure_reasons"])
        self.assertIn("forecast_availability_lag_respected", audit["failure_reasons"])
        self.assertIn("no_side_rows_use_explicit_no_tokens", audit["failure_reasons"])
        self.assertIn("traded_tokens_weather_matched", audit["failure_reasons"])

    def test_strategy_sensitivity_diagnostics_show_fair_value_threshold_quality(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-01T00:00:00+00:00",
                "city": "Dallas, TX",
                "target_date": "2026-06-02",
                "token_id": "lower-confidence-loss",
                "market_price": 0.30,
                "fair_value": 0.65,
                "edge": 0.329,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "bucket_width_f": None,
                "polymarket_payout": 0,
                "weather_outcome": 0,
            },
            {
                "generated_at": "2026-06-01T00:00:00+00:00",
                "city": "New York, NY",
                "target_date": "2026-06-02",
                "token_id": "high-confidence-win",
                "market_price": 0.30,
                "fair_value": 0.75,
                "edge": 0.429,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "bucket_width_f": None,
                "polymarket_payout": 1,
                "weather_outcome": 1,
            },
        ]

        diagnostics = _strategy_sensitivity_diagnostics(rows, SignalSettings(min_signal_fair_value=0.70))
        by_fair_value = {row["threshold"]: row for row in diagnostics["by_min_signal_fair_value"]}
        replay_by_variant = {row["variant"]: row for row in diagnostics["counterfactual_kelly_replays"]}

        self.assertEqual(by_fair_value[0.60]["n"], 2)
        self.assertEqual(by_fair_value[0.60]["actual_rate"], 0.5)
        self.assertEqual(by_fair_value[0.70]["n"], 1)
        self.assertEqual(by_fair_value[0.70]["actual_rate"], 1.0)
        self.assertGreater(by_fair_value[0.70]["flat_return_on_notional"], by_fair_value[0.60]["flat_return_on_notional"])
        self.assertGreater(replay_by_variant["current"]["pnl_usd"], replay_by_variant["looser_fair_value_0.60"]["pnl_usd"])
        self.assertIn("aggressive_fractional_kelly_0.50", replay_by_variant)
        self.assertIn("legacy_no_side_counter_event_0.08", replay_by_variant)
        self.assertIn("legacy_no_side_counter_event_0.09", replay_by_variant)
        self.assertIn("selected_no_side_counter_event_0.10", replay_by_variant)
        self.assertIn("looser_hold_no_side_counter_event_0.30", replay_by_variant)
        self.assertIn("disabled_hold_no_side_counter_event_gate", replay_by_variant)
        self.assertIn("force_hold_existing_positions_to_settlement", replay_by_variant)
        self.assertIn("partial_invalid_hold_exit_x0.50_fv0.90_price0.50_0.65", replay_by_variant)
        self.assertIn("looser_no_side_counter_event_0.30", replay_by_variant)
        self.assertIn("max_position_fraction_0.05", replay_by_variant)
        self.assertIn("max_position_fraction_0.10", replay_by_variant)
        self.assertIn("entry_hours_utc_12_only", replay_by_variant)
        self.assertIn("entry_hours_utc_12_no_side_counter_event_0.13", replay_by_variant)
        self.assertIn("entry_hours_utc_12_no_side_counter_event_0.20", replay_by_variant)
        self.assertIn("utc12_relaxed_no_side_counter_event_0.15", replay_by_variant)
        self.assertIn("utc12_relaxed_no_side_counter_event_0.20", replay_by_variant)
        self.assertIn("utc12_relaxed_no_side_counter_event_0.30", replay_by_variant)
        self.assertIn("disabled_no_side_counter_event_gate", replay_by_variant)
        self.assertIn("disable_bounded_no_side_entries", replay_by_variant)
        self.assertIn("high_price_low_edge_damped_cap_0.85_edge_0.12_x0.35_full_0.25", replay_by_variant)
        self.assertIn("by_no_side_max_counter_event_probability", diagnostics)
        self.assertIn("max_drawdown_usd", replay_by_variant["current"])
        self.assertIn("top_1_pnl_share", replay_by_variant["current"])

    def test_json_kelly_replay_respects_time_conditioned_no_side_tail(self) -> None:
        base_row = {
            "city": "Miami, FL",
            "target_date": "2026-01-22",
            "market_price": 0.70,
            "fair_value": 0.86,
            "edge": 0.151,
            "model_count": 3,
            "model_agreement": 1.0,
            "entry_eligible": True,
            "bucket": "NO: 86°F or higher",
            "side": "NO",
            "bucket_width_f": None,
            "model_probabilities": {"a.model": 0.85, "b.model": 0.88, "c.model": 0.90},
            "polymarket_payout": 1,
            "weather_outcome": 1,
        }
        rows = [
            {**base_row, "generated_at": "2026-01-20T06:00:00+00:00", "token_id": "outside-hour"},
            {**base_row, "generated_at": "2026-01-20T12:00:00+00:00", "token_id": "relaxed-hour"},
        ]
        settings = SignalSettings(
            min_signal_fair_value=0.70,
            allow_no_side_entries=True,
            no_side_relaxed_counter_event_probability=0.20,
            no_side_relaxed_counter_event_hours_utc=(12,),
        )

        replay = _json_kelly_replay(
            "time-aware",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=0.75,
            compound_kelly_sizing=True,
            max_position_usd=100,
            max_position_fraction=0.25,
            edge_position_full_cap_edge=0.25,
            edge_position_min_multiplier=0.35,
            min_trade_usd=1,
        )

        self.assertEqual(replay["signals"], 1)
        self.assertEqual(replay["trade_count"], 1)
        self.assertEqual(replay["settings"]["no_side_relaxed_counter_event_probability"], 0.20)
        self.assertEqual(replay["settings"]["no_side_relaxed_counter_event_hours_utc"], [12])

    def test_signal_opportunity_diagnostics_explain_selected_and_rejected_cohorts(self) -> None:
        rows = [
            {
                "generated_at": "2026-01-20T12:00:00+00:00",
                "city": "Miami, FL",
                "target_date": "2026-01-22",
                "question": "Will the highest temperature in Miami be 86°F or higher on January 22?",
                "bucket": "NO: 86°F or higher",
                "side": "NO",
                "token_id": "selected",
                "market_price": 0.70,
                "fair_value": 0.86,
                "edge": 0.151,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "signal_filter_reason": None,
                "model_probabilities": {"a.model": 0.91, "b.model": 0.92, "c.model": 0.93},
                "polymarket_payout": 1,
                "weather_outcome": 1,
            },
            {
                "generated_at": "2026-01-20T12:00:00+00:00",
                "city": "Miami, FL",
                "target_date": "2026-01-22",
                "question": "Will the highest temperature in Miami be 87°F or higher on January 22?",
                "bucket": "87°F or higher",
                "side": "YES",
                "token_id": "rejected",
                "market_price": 0.40,
                "fair_value": 0.65,
                "edge": 0.232,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "signal_filter_reason": "fair value below 0.70",
                "model_probabilities": {"a.model": 0.65, "b.model": 0.66, "c.model": 0.64},
                "polymarket_payout": 0,
                "weather_outcome": 0,
            },
        ]

        diagnostics = _signal_opportunity_diagnostics(rows, SignalSettings(min_signal_fair_value=0.70, allow_no_side_entries=True))

        self.assertEqual(diagnostics["selected_candidate_rows"], 1)
        self.assertEqual(diagnostics["selected_candidate_calibration"]["actual_rate"], 1.0)
        self.assertEqual(diagnostics["selected_candidate_by_entry_hour_utc"][0]["group"], 12)
        self.assertEqual(diagnostics["selected_candidate_by_lead_days"][0]["group"], 2)
        self.assertEqual(diagnostics["selected_candidate_by_no_side_counter_event_probability"][0]["group"], 0.05)
        self.assertEqual(diagnostics["rejected_by_signal_filter_reason"][0]["group"], "fair value below 0.70")
        self.assertEqual(diagnostics["top_rejected_losers_by_edge"][0]["token_id"], "rejected")

    def test_strategy_recommendation_promotes_highest_clean_cap_only(self) -> None:
        sensitivity = {
            "counterfactual_kelly_replays": [
                {
                    "variant": "current",
                    "pnl_usd": 100.0,
                    "return_pct": 1.0,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "hit_rate": 0.95,
                    "max_drawdown_usd": 4.0,
                    "buy_notional_usd": 200.0,
                    "return_on_buy_notional": 0.50,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.20",
                    "pnl_usd": 150.0,
                    "return_pct": 1.5,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "hit_rate": 0.95,
                    "max_drawdown_usd": 7.0,
                    "buy_notional_usd": 300.0,
                    "return_on_buy_notional": 0.50,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.225",
                    "pnl_usd": 175.0,
                    "return_pct": 1.75,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "hit_rate": 0.95,
                    "max_drawdown_usd": 8.0,
                    "buy_notional_usd": 350.0,
                    "return_on_buy_notional": 0.50,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.25",
                    "pnl_usd": 200.0,
                    "return_pct": 2.0,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "hit_rate": 0.95,
                    "max_drawdown_usd": 9.0,
                    "buy_notional_usd": 400.0,
                    "return_on_buy_notional": 0.50,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "looser_no_side_counter_event_0.20",
                    "pnl_usd": 300.0,
                    "event_hit_rate": 0.90,
                    "trade_count": 12,
                    "weather_ambiguous_trades": 1,
                    "weather_mismatch_trades": 0,
                },
            ]
        }
        robustness = {
            "cap_fraction_by_chronological_session_slice": [
                {
                    "variant": "max_position_fraction_0.20",
                    "slice": "first_50pct_sessions",
                    "pnl_usd": 30.0,
                    "event_hit_rate": 0.95,
                    "max_drawdown_usd": 2.0,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.20",
                    "slice": "second_50pct_sessions",
                    "pnl_usd": 20.0,
                    "event_hit_rate": 0.90,
                    "max_drawdown_usd": 3.0,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.225",
                    "slice": "first_50pct_sessions",
                    "pnl_usd": 35.0,
                    "event_hit_rate": 0.95,
                    "max_drawdown_usd": 2.5,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.225",
                    "slice": "second_50pct_sessions",
                    "pnl_usd": 25.0,
                    "event_hit_rate": 0.90,
                    "max_drawdown_usd": 3.5,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.25",
                    "slice": "first_50pct_sessions",
                    "pnl_usd": 40.0,
                    "event_hit_rate": 0.95,
                    "max_drawdown_usd": 3.0,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.25",
                    "slice": "second_50pct_sessions",
                    "pnl_usd": 30.0,
                    "event_hit_rate": 0.90,
                    "max_drawdown_usd": 4.0,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
            ]
        }

        recommendation = _strategy_recommendation_diagnostics(sensitivity, robustness)

        self.assertEqual(recommendation["recommended_profile"], "aggressive_max_position_fraction_0.25")
        self.assertEqual(recommendation["recommendation_type"], "paper_test_aggressive_sizing")
        self.assertTrue(recommendation["cap_20_slice_check"]["all_profitable_and_clean"])
        self.assertTrue(recommendation["cap_25_slice_check"]["all_profitable_and_clean"])
        self.assertTrue(any("not recommended" in reason for reason in recommendation["reasons"]))

    def test_strategy_recommendation_falls_back_when_twenty_five_percent_cap_is_not_clean(self) -> None:
        sensitivity = {
            "counterfactual_kelly_replays": [
                {
                    "variant": "current",
                    "pnl_usd": 100.0,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.20",
                    "pnl_usd": 150.0,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.25",
                    "pnl_usd": 200.0,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
            ]
        }
        robustness = {
            "cap_fraction_by_chronological_session_slice": [
                {
                    "variant": "max_position_fraction_0.20",
                    "slice": "first_50pct_sessions",
                    "pnl_usd": 30.0,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.20",
                    "slice": "second_50pct_sessions",
                    "pnl_usd": 20.0,
                    "event_hit_rate": 0.90,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.25",
                    "slice": "first_50pct_sessions",
                    "pnl_usd": 40.0,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.25",
                    "slice": "second_50pct_sessions",
                    "pnl_usd": -5.0,
                    "event_hit_rate": 0.85,
                    "weather_ambiguous_trades": 1,
                    "weather_mismatch_trades": 0,
                },
            ]
        }

        recommendation = _strategy_recommendation_diagnostics(sensitivity, robustness)

        self.assertEqual(recommendation["recommended_profile"], "aggressive_max_position_fraction_0.20")
        self.assertTrue(recommendation["cap_20_slice_check"]["all_profitable_and_clean"])
        self.assertFalse(recommendation["cap_25_slice_check"]["all_profitable_and_clean"])

    def test_strategy_recommendation_keeps_current_when_twenty_percent_cap_has_ambiguity(self) -> None:
        sensitivity = {
            "counterfactual_kelly_replays": [
                {
                    "variant": "current",
                    "pnl_usd": 100.0,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 0,
                    "weather_mismatch_trades": 0,
                },
                {
                    "variant": "max_position_fraction_0.20",
                    "pnl_usd": 150.0,
                    "trade_count": 10,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 1,
                    "weather_mismatch_trades": 0,
                },
            ]
        }
        robustness = {
            "cap_fraction_by_chronological_session_slice": [
                {
                    "variant": "max_position_fraction_0.20",
                    "slice": "first_50pct_sessions",
                    "pnl_usd": 30.0,
                    "event_hit_rate": 0.95,
                    "weather_ambiguous_trades": 1,
                    "weather_mismatch_trades": 0,
                }
            ]
        }

        recommendation = _strategy_recommendation_diagnostics(sensitivity, robustness)

        self.assertEqual(recommendation["recommended_profile"], "current")
        self.assertEqual(recommendation["recommendation_type"], "keep_current")

    def test_robustness_diagnostics_include_cap_fraction_slices(self) -> None:
        rows = [
            {
                "generated_at": f"2026-06-0{index + 1}T00:00:00+00:00",
                "city": "Dallas, TX",
                "target_date": f"2026-06-0{index + 2}",
                "token_id": f"win-{index}",
                "market_price": 0.50,
                "fair_value": 0.90,
                "edge": 0.37,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "bucket_width_f": None,
                "polymarket_payout": 1,
                "weather_outcome": 1,
            }
            for index in range(4)
        ]

        diagnostics = _robustness_diagnostics(
            rows,
            SignalSettings(min_signal_fair_value=0.70),
            bankroll_usd=100,
            kelly_fraction=0.75,
            compound_kelly_sizing=True,
            max_position_usd=100,
            max_position_fraction=0.10,
            kelly_market_blend=0.0,
            edge_position_full_cap_edge=0.25,
            edge_position_min_multiplier=0.35,
            min_trade_usd=1,
        )
        cap_slices = diagnostics["cap_fraction_by_chronological_session_slice"]
        by_variant_slice = {(row["variant"], row["slice"]): row for row in cap_slices}

        self.assertEqual(len(cap_slices), 24)
        self.assertIn(("max_position_fraction_0.05", "first_50pct_sessions"), by_variant_slice)
        self.assertIn(("max_position_fraction_0.20", "first_50pct_sessions"), by_variant_slice)
        self.assertIn(("max_position_fraction_0.25", "first_50pct_sessions"), by_variant_slice)
        self.assertGreater(
            by_variant_slice[("max_position_fraction_0.20", "first_50pct_sessions")]["pnl_usd"],
            by_variant_slice[("max_position_fraction_0.05", "first_50pct_sessions")]["pnl_usd"],
        )
        self.assertGreater(
            by_variant_slice[("max_position_fraction_0.25", "first_50pct_sessions")]["pnl_usd"],
            by_variant_slice[("max_position_fraction_0.20", "first_50pct_sessions")]["pnl_usd"],
        )

    def test_json_kelly_replay_can_damp_high_price_low_edge_position_size(self) -> None:
        rows = [
            {
                "generated_at": "2026-06-01T00:00:00+00:00",
                "city": "Dallas, TX",
                "target_date": "2026-06-02",
                "token_id": "high-price-low-edge",
                "market_price": 0.86,
                "fair_value": 0.95,
                "edge": 0.0858,
                "model_count": 3,
                "model_agreement": 1.0,
                "entry_eligible": True,
                "bucket_width_f": None,
                "polymarket_payout": 1,
                "weather_outcome": 1,
            }
        ]
        settings = SignalSettings(min_signal_fair_value=0.70)

        undamped = _json_kelly_replay(
            "undamped",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=0.75,
            compound_kelly_sizing=True,
            max_position_usd=175,
            max_position_fraction=0.75,
            edge_position_full_cap_edge=0.25,
            edge_position_min_multiplier=0.35,
            min_trade_usd=1,
        )
        damped = _json_kelly_replay(
            "damped",
            rows,
            settings,
            bankroll_usd=100,
            kelly_fraction=0.75,
            compound_kelly_sizing=True,
            max_position_usd=175,
            max_position_fraction=0.75,
            edge_position_full_cap_edge=0.25,
            edge_position_min_multiplier=0.35,
            min_trade_usd=1,
            high_price_damping_threshold=0.85,
            high_price_damping_edge=0.12,
            high_price_damping_multiplier=0.35,
        )

        self.assertEqual(undamped["trade_count"], 1)
        self.assertEqual(damped["trade_count"], 1)
        self.assertLess(damped["buy_notional_usd"], undamped["buy_notional_usd"])

    def test_trade_performance_diagnostics_include_concentration_and_time_cohorts(self) -> None:
        executions = [
            {
                "action": "BUY",
                "token_id": "winner",
                "executed_at": "2026-06-01T12:00:00+00:00",
                "question": "Will the highest temperature in Dallas be 80°F or higher on June 2?",
                "city": "Dallas, TX",
                "target_date": "2026-06-02",
                "bucket": "80°F or higher",
                "side": "YES",
                "notional_usd": 10.0,
                "realized_pnl_usd": 0.0,
                "price": 0.5,
                "fair_value": 0.75,
                "edge": 0.235,
                "model_agreement": 1.0,
                "bucket_lower_f": 80.0,
                "bucket_upper_f": None,
                "bucket_shape": "upper_tail",
            },
            {
                "action": "SETTLE",
                "token_id": "winner",
                "executed_at": "2026-06-03T12:00:00+00:00",
                "question": "Will the highest temperature in Dallas be 80°F or higher on June 2?",
                "city": "Dallas, TX",
                "target_date": "2026-06-02",
                "bucket": "80°F or higher",
                "side": "YES",
                "notional_usd": 20.0,
                "realized_pnl_usd": 10.0,
                "polymarket_payout": 1,
                "weather_outcome": 1,
                "bucket_lower_f": 80.0,
                "bucket_upper_f": None,
                "bucket_shape": "upper_tail",
            },
            {
                "action": "BUY",
                "token_id": "loser",
                "executed_at": "2026-06-01T00:00:00+00:00",
                "question": "Will the highest temperature in New York City be 50°F or higher on June 2?",
                "city": "New York, NY",
                "target_date": "2026-06-02",
                "bucket": "NO: 50°F or higher",
                "side": "NO",
                "notional_usd": 5.0,
                "realized_pnl_usd": 0.0,
                "price": 0.25,
                "fair_value": 0.80,
                "edge": 0.5275,
                "model_agreement": 1.0,
                "bucket_lower_f": 50.0,
                "bucket_upper_f": None,
                "bucket_shape": "upper_tail",
            },
            {
                "action": "SETTLE",
                "token_id": "loser",
                "executed_at": "2026-06-03T12:00:00+00:00",
                "question": "Will the highest temperature in New York City be 50°F or higher on June 2?",
                "city": "New York, NY",
                "target_date": "2026-06-02",
                "bucket": "NO: 50°F or higher",
                "side": "NO",
                "notional_usd": 0.0,
                "realized_pnl_usd": -5.0,
                "polymarket_payout": 0,
                "weather_outcome": 0,
                "bucket_lower_f": 50.0,
                "bucket_upper_f": None,
                "bucket_shape": "upper_tail",
            },
        ]

        diagnostics = _trade_performance_diagnostics(executions)

        self.assertEqual(diagnostics["pnl_concentration"]["loss_trade_count"], 1)
        self.assertEqual(diagnostics["pnl_concentration"]["loss_pnl_usd"], -5.0)
        self.assertEqual(diagnostics["event_winning_trades"], 1)
        self.assertEqual(diagnostics["event_losing_trades"], 1)
        self.assertEqual(diagnostics["event_hit_rate"], 0.5)
        self.assertEqual(diagnostics["event_loss_pnl_usd"], -5.0)
        self.assertEqual(diagnostics["profitable_event_loser_trades"], 0)
        self.assertEqual({row["group"] for row in diagnostics["by_event_outcome"]}, {"event_win", "event_loss"})
        self.assertEqual({row["group"] for row in diagnostics["by_bucket_shape"]}, {"upper_tail"})
        self.assertEqual({row["group"] for row in diagnostics["by_entry_hour_utc"]}, {0, 12})
        self.assertEqual({row["group"] for row in diagnostics["by_entry_month"]}, {"2026-06"})
        self.assertEqual({row["group"] for row in diagnostics["by_city"]}, {"Dallas, TX", "New York, NY"})

    def test_trade_performance_diagnostics_flag_event_winners_that_lost_money(self) -> None:
        executions = [
            {
                "action": "BUY",
                "token_id": "sold-winner",
                "executed_at": "2026-06-01T12:00:00+00:00",
                "question": "NO: Will the highest temperature in New York City be between 44-45°F on June 2?",
                "city": "New York, NY",
                "target_date": "2026-06-02",
                "bucket": "NO: between 44-45°F",
                "side": "NO",
                "shares": 100.0,
                "notional_usd": 60.0,
                "realized_pnl_usd": 0.0,
                "price": 0.60,
                "fair_value": 0.98,
                "edge": 0.36,
                "model_agreement": 1.0,
                "polymarket_payout": 1,
                "weather_outcome": 1,
            },
            {
                "action": "SELL",
                "token_id": "sold-winner",
                "executed_at": "2026-06-01T18:00:00+00:00",
                "question": "NO: Will the highest temperature in New York City be between 44-45°F on June 2?",
                "city": "New York, NY",
                "target_date": "2026-06-02",
                "bucket": "NO: between 44-45°F",
                "side": "NO",
                "shares": 100.0,
                "notional_usd": 57.0,
                "realized_pnl_usd": -3.0,
                "price": 0.57,
                "fair_value": 0.98,
                "edge": 0.38,
                "model_agreement": 1.0,
                "polymarket_payout": 1,
                "weather_outcome": 1,
            },
            {
                "action": "BUY",
                "token_id": "sold-loser",
                "executed_at": "2026-06-01T12:00:00+00:00",
                "question": "Will the highest temperature in Dallas be 90°F or higher on June 2?",
                "city": "Dallas, TX",
                "target_date": "2026-06-02",
                "bucket": "90°F or higher",
                "side": "YES",
                "shares": 50.0,
                "notional_usd": 25.0,
                "realized_pnl_usd": 0.0,
                "price": 0.50,
                "fair_value": 0.80,
                "edge": 0.29,
                "model_agreement": 1.0,
                "polymarket_payout": 0,
                "weather_outcome": 0,
            },
            {
                "action": "SELL",
                "token_id": "sold-loser",
                "executed_at": "2026-06-01T18:00:00+00:00",
                "question": "Will the highest temperature in Dallas be 90°F or higher on June 2?",
                "city": "Dallas, TX",
                "target_date": "2026-06-02",
                "bucket": "90°F or higher",
                "side": "YES",
                "shares": 50.0,
                "notional_usd": 20.0,
                "realized_pnl_usd": -5.0,
                "price": 0.40,
                "fair_value": 0.40,
                "edge": -0.01,
                "model_agreement": 0.5,
                "polymarket_payout": 0,
                "weather_outcome": 0,
            },
        ]

        diagnostics = _trade_performance_diagnostics(executions)

        self.assertEqual(diagnostics["event_winning_trades"], 1)
        self.assertEqual(diagnostics["unprofitable_event_winner_trades"], 1)
        self.assertEqual(diagnostics["unprofitable_event_winner_pnl_usd"], -3.0)
        self.assertEqual(diagnostics["worst_unprofitable_event_winners"][0]["token_id"], "sold-winner")
        exit_management = diagnostics["exit_management"]
        self.assertEqual(exit_management["sell_count"], 2)
        self.assertEqual(exit_management["trades_with_sells"], 2)
        self.assertEqual(exit_management["event_winner_sell_drag_usd"], -43.0)
        self.assertEqual(exit_management["event_loser_sell_value_usd"], 20.0)
        self.assertEqual(exit_management["sell_decision_value_vs_settlement_usd"], -23.0)
        self.assertEqual(exit_management["worst_sells_vs_settlement"][0]["token_id"], "sold-winner")
        self.assertEqual(exit_management["best_sells_vs_settlement"][0]["token_id"], "sold-loser")

    def test_trade_concentration_omits_top_share_when_total_pnl_is_negative(self) -> None:
        concentration = _pnl_concentration(
            [
                {
                    "token_id": "loser",
                    "realized_pnl_usd": -5.0,
                },
                {
                    "token_id": "small-winner",
                    "realized_pnl_usd": 1.0,
                },
            ]
        )

        self.assertEqual(concentration["total_pnl_usd"], -4.0)
        self.assertIsNone(concentration["top_1_pnl_share"])

    def test_station_market_does_not_use_gridded_archive_as_weather_crosscheck(self) -> None:
        raw = {
            "id": "2416222",
            "question": "Will the highest temperature in Beijing be 25°C or below on June 4?",
            "slug": "highest-temperature-in-beijing-on-june-4-2026-25corbelow",
            "description": "Resolution source: https://www.wunderground.com/history/daily/cn/beijing/ZBAA",
            "resolutionSource": "https://www.wunderground.com/history/daily/cn/beijing/ZBAA",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0","1"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
        market = parse_weather_market(raw, today=date(2026, 6, 1))
        assert market is not None

        class FakeObservationClient:
            def fetch_observed_high(self, city, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=74.8,
                    source="open_meteo_archive_daily",
                    observed_at=datetime(2026, 6, 4, 23, 59, tzinfo=timezone.utc),
                    sample_count=1,
                    is_actual=False,
                    is_final=True,
                )

        result = _resolve_market_outcome(market, raw, market.buckets[0], FakeObservationClient())
        self.assertEqual(result["payout"], 0)
        self.assertEqual(result["polymarket_payout"], 0)
        self.assertIsNone(result["weather_outcome"])
        self.assertIsNone(result["settlement_source"])

    def test_station_market_uses_actual_historical_station_crosscheck(self) -> None:
        raw = {
            "id": "2416222",
            "question": "Will the highest temperature in Beijing be 25°C or below on June 4?",
            "slug": "highest-temperature-in-beijing-on-june-4-2026-25corbelow",
            "description": "Resolution source: https://www.wunderground.com/history/daily/cn/beijing/ZBAA",
            "resolutionSource": "https://www.wunderground.com/history/daily/cn/beijing/ZBAA",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0","1"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
        market = parse_weather_market(raw, today=date(2026, 6, 1))
        assert market is not None

        class FakeObservationClient:
            def fetch_observed_high(self, city, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=82.4,
                    source="historical_metar_ZBAA",
                    observed_at=datetime(2026, 6, 4, 6, 0, tzinfo=timezone.utc),
                    sample_count=51,
                    is_actual=True,
                    is_final=True,
                )

        result = _resolve_market_outcome(market, raw, market.buckets[0], FakeObservationClient())
        self.assertEqual(result["payout"], 0)
        self.assertEqual(result["polymarket_payout"], 0)
        self.assertEqual(result["weather_outcome"], 0)
        self.assertEqual(result["settlement_source"], "historical_metar_ZBAA")

    def test_adjacent_exact_celsius_station_mismatch_is_marked_ambiguous(self) -> None:
        raw = {
            "id": "2416223",
            "question": "Will the highest temperature in Shenzhen be 26°C on June 18?",
            "slug": "highest-temperature-in-shenzhen-on-june-18-2026-26c",
            "description": "Resolution source: https://www.wunderground.com/history/daily/cn/shenzhen/ZGSZ. Resolved to the nearest whole degree Celsius.",
            "resolutionSource": "https://www.wunderground.com/history/daily/cn/shenzhen/ZGSZ",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["1","0"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
        market = parse_weather_market(raw, today=date(2026, 6, 1))
        assert market is not None

        class FakeObservationClient:
            def fetch_observed_high(self, city, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=80.6,
                    source="historical_metar_ZGSZ",
                    observed_at=datetime(2026, 6, 18, 6, 0, tzinfo=timezone.utc),
                    sample_count=45,
                    is_actual=True,
                    is_final=True,
                )

        result = _resolve_market_outcome(market, raw, market.buckets[0], FakeObservationClient())
        self.assertEqual(result["payout"], 1)
        self.assertEqual(result["polymarket_payout"], 1)
        self.assertIsNone(result["weather_outcome"])
        self.assertTrue(result["weather_ambiguous"])
        self.assertEqual(result["settlement_source"], "historical_metar_ZGSZ_ambiguous_resolution")

    def test_clear_open_ended_celsius_station_mismatch_remains_crosscheck(self) -> None:
        raw = {
            "id": "2416224",
            "question": "Will the highest temperature in Shenzhen be 26°C or higher on June 18?",
            "slug": "highest-temperature-in-shenzhen-on-june-18-2026-26c-or-higher",
            "description": "Resolution source: https://www.wunderground.com/history/daily/cn/shenzhen/ZGSZ. Resolved to the nearest whole degree Celsius.",
            "resolutionSource": "https://www.wunderground.com/history/daily/cn/shenzhen/ZGSZ",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["1","0"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
        market = parse_weather_market(raw, today=date(2026, 6, 1))
        assert market is not None

        class FakeObservationClient:
            def fetch_observed_high(self, city, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=68.0,
                    source="historical_metar_ZGSZ",
                    observed_at=datetime(2026, 6, 18, 6, 0, tzinfo=timezone.utc),
                    sample_count=45,
                    is_actual=True,
                    is_final=True,
                )

        result = _resolve_market_outcome(market, raw, market.buckets[0], FakeObservationClient())
        self.assertEqual(result["payout"], 1)
        self.assertEqual(result["polymarket_payout"], 1)
        self.assertEqual(result["weather_outcome"], 0)
        self.assertFalse(result["weather_ambiguous"])
        self.assertEqual(result["settlement_source"], "historical_metar_ZGSZ")

    def test_station_resolution_prefers_historical_station_over_generic_nws(self) -> None:
        raw = {
            "id": "2416225",
            "question": "Will the highest temperature in New York City be between 76-77°F on June 17?",
            "slug": "highest-temperature-in-new-york-city-on-june-17-2026-between-76-77f",
            "description": "Resolution source: https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA. Resolved to the nearest whole degree Fahrenheit.",
            "resolutionSource": "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["1","0"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
        market = parse_weather_market(raw, today=date(2026, 6, 1))
        assert market is not None

        class FakeObservationClient:
            def fetch_historical_station_high(self, city, station_id, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=77.0,
                    source=f"historical_metar_{station_id}",
                    observed_at=datetime(2026, 6, 17, 18, 0, tzinfo=timezone.utc),
                    sample_count=24,
                    is_actual=True,
                    is_final=True,
                )

            def fetch_observed_high(self, city, target_date, now=None):
                return ObservedHigh(
                    city=city,
                    target_date=target_date,
                    max_temperature_f=78.8,
                    source="nws_station_KLGA",
                    observed_at=datetime(2026, 6, 17, 18, 0, tzinfo=timezone.utc),
                    sample_count=24,
                    is_actual=True,
                    is_final=True,
                )

        result = _resolve_market_outcome(market, raw, market.buckets[0], FakeObservationClient())
        self.assertEqual(result["payout"], 1)
        self.assertEqual(result["weather_outcome"], 1)
        self.assertEqual(result["settlement_source"], "historical_metar_KLGA")


if __name__ == "__main__":
    unittest.main()
