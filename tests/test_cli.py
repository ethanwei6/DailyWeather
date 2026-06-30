from __future__ import annotations

import json
import argparse
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from weather_strategy.cli import _apply_strategy_profile, _live_collateral_balance_after_executions, _long_backtest_summary, _make_run_log_path, _parse_markets, main
from weather_strategy.cli import _should_process_market
from weather_strategy.paper import PaperLedger
from weather_strategy.parser import parse_weather_market
from weather_strategy.signals import SignalSettings


class CliTest(unittest.TestCase):
    def test_strategy_profile_applies_utc12_relaxed_live_forward_settings(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-utc12-relaxed-no-tail-0.20",
            allow_no_side_entries=False,
            no_side_max_counter_event_probability=0.05,
            no_side_relaxed_counter_event_probability=None,
            no_side_relaxed_counter_event_hours_utc="",
            bankroll_usd=1000.0,
            kelly_fraction=0.25,
            compound_kelly_sizing=False,
            max_position_usd=1000.0,
            max_position_fraction=0.15,
            trim_valid_holds_to_kelly_target=True,
            min_lead_days=0,
            max_lead_days=7,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.10)
        self.assertEqual(args.no_side_relaxed_counter_event_probability, 0.20)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "12")
        self.assertEqual(args.bankroll_usd, 100.0)
        self.assertEqual(args.kelly_fraction, 0.75)
        self.assertTrue(args.compound_kelly_sizing)
        self.assertEqual(args.max_position_usd, 175.0)
        self.assertEqual(args.max_position_fraction, 0.25)
        self.assertFalse(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.min_lead_days, 1)
        self.assertEqual(args.max_lead_days, 2)

    def test_strategy_profile_can_trim_valid_holds_to_kelly_target(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-utc12-relaxed-no-tail-0.20-trim-holds",
            allow_no_side_entries=False,
            no_side_relaxed_counter_event_probability=None,
            no_side_relaxed_counter_event_hours_utc="",
            trim_valid_holds_to_kelly_target=False,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.no_side_relaxed_counter_event_probability, 0.20)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "12")
        self.assertTrue(args.trim_valid_holds_to_kelly_target)

    def test_strategy_profile_can_hold_high_conviction_no_side_tails(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-holds",
            allow_no_side_entries=False,
            no_side_relaxed_counter_event_probability=None,
            no_side_relaxed_counter_event_hours_utc="",
            trim_valid_holds_to_kelly_target=False,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.no_side_relaxed_counter_event_probability, 0.20)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "12")
        self.assertTrue(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)

    def test_strategy_profile_can_require_stronger_bounded_bucket_edge(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-bounded-edge-0.15",
            allow_no_side_entries=False,
            no_side_relaxed_counter_event_probability=None,
            no_side_relaxed_counter_event_hours_utc="",
            trim_valid_holds_to_kelly_target=False,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            invalid_hold_partial_exit_fraction=None,
            invalid_hold_partial_exit_min_fair_value=0.90,
            invalid_hold_partial_exit_min_price=0.50,
            invalid_hold_partial_exit_max_price=0.80,
            bounded_bucket_min_edge=0.10,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.no_side_relaxed_counter_event_probability, 0.20)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "12")
        self.assertTrue(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.invalid_hold_partial_exit_fraction, 0.50)
        self.assertEqual(args.invalid_hold_partial_exit_min_fair_value, 0.90)
        self.assertEqual(args.invalid_hold_partial_exit_min_price, 0.50)
        self.assertEqual(args.invalid_hold_partial_exit_max_price, 0.65)
        self.assertEqual(args.bounded_bucket_min_edge, 0.15)

    def test_strategy_profile_can_use_strict_no_tail_trim_highconv_settings(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15",
            allow_no_side_entries=False,
            no_side_relaxed_counter_event_probability=0.20,
            no_side_relaxed_counter_event_hours_utc="12",
            trim_valid_holds_to_kelly_target=False,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.10,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertIsNone(args.no_side_relaxed_counter_event_probability)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "")
        self.assertTrue(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.15)

    def test_strategy_profile_can_use_strict_no_tail_preserve_highconv_settings(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.15",
            allow_no_side_entries=False,
            no_side_relaxed_counter_event_probability=0.20,
            no_side_relaxed_counter_event_hours_utc="12",
            trim_valid_holds_to_kelly_target=True,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.10,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertIsNone(args.no_side_relaxed_counter_event_probability)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "")
        self.assertFalse(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.15)

    def test_strategy_profile_can_use_strict_no_tail_preserve_highconv_bounded_edge_010(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.10",
            allow_no_side_entries=False,
            no_side_max_counter_event_probability=0.05,
            no_side_relaxed_counter_event_probability=0.20,
            no_side_relaxed_counter_event_hours_utc="12",
            trim_valid_holds_to_kelly_target=True,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.10)
        self.assertIsNone(args.no_side_relaxed_counter_event_probability)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "")
        self.assertFalse(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)

    def test_strategy_profile_can_use_strict_no_tail_011_preserve_highconv_bounded_edge_010(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-strict-no-tail-0.11-preserve-highconv-bounded-edge-0.10",
            allow_no_side_entries=False,
            no_side_max_counter_event_probability=0.05,
            no_side_relaxed_counter_event_probability=0.20,
            no_side_relaxed_counter_event_hours_utc="12",
            trim_valid_holds_to_kelly_target=True,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.11)
        self.assertIsNone(args.no_side_relaxed_counter_event_probability)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "")
        self.assertFalse(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)

    def test_strategy_profile_can_use_50_highwin_no_side_max_094_settings(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94",
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            max_position_usd=175.0,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            no_side_relaxed_counter_event_probability=0.20,
            no_side_relaxed_counter_event_hours_utc="12",
            trim_valid_holds_to_kelly_target=True,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.max_position_usd, 50.0)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.11)
        self.assertEqual(args.no_side_max_price, 0.94)
        self.assertEqual(args.no_side_high_confidence_min_edge, 0.05)
        self.assertIsNone(args.no_side_relaxed_counter_event_probability)
        self.assertEqual(args.no_side_relaxed_counter_event_hours_utc, "")
        self.assertFalse(args.trim_valid_holds_to_kelly_target)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)

    def test_strategy_profile_can_require_confirmed_bounded_buckets_for_50_highwin(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94-bounded-confirmed",
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            max_position_usd=175.0,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.max_position_usd, 50.0)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.11)
        self.assertEqual(args.no_side_max_price, 0.94)
        self.assertEqual(args.no_side_high_confidence_min_edge, 0.05)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.98)
        self.assertEqual(args.bounded_bucket_min_price, 0.75)

    def test_strategy_profile_can_use_50_highwin_012_tail_cap_030_position_cap(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-highwin-strict-no-tail-0.12-no-side-max-0.94-bounded-confirmed-cap-0.30",
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            max_position_usd=175.0,
            max_position_fraction=0.25,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.max_position_usd, 50.0)
        self.assertEqual(args.max_position_fraction, 0.30)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.12)
        self.assertEqual(args.no_side_max_price, 0.94)
        self.assertEqual(args.no_side_high_confidence_min_edge, 0.05)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.98)
        self.assertEqual(args.bounded_bucket_min_price, 0.75)

    def test_strategy_profile_can_use_50_highwin_014_tail_cap_035_bprice_070(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-highwin-strict-no-tail-0.14-no-side-max-0.94-bounded-confirmed-cap-0.35-bprice-0.70",
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.0,
            max_position_usd=175.0,
            max_position_fraction=0.25,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.max_position_usd, 50.0)
        self.assertEqual(args.max_position_fraction, 0.35)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.14)
        self.assertEqual(args.no_side_max_price, 0.94)
        self.assertEqual(args.no_side_high_confidence_min_edge, 0.05)
        self.assertEqual(args.hold_no_side_high_conviction_min_fair_value, 0.98)
        self.assertEqual(args.hold_no_side_high_conviction_min_edge, 0.35)
        self.assertEqual(args.hold_no_side_high_conviction_counter_event_probability, 0.20)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.98)
        self.assertEqual(args.bounded_bucket_min_price, 0.70)

    def test_strategy_profile_can_use_50_reserved_live_money_sizing(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.0,
            max_position_usd=175.0,
            max_position_fraction=0.35,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.25)
        self.assertEqual(args.max_position_usd, 50.0)
        self.assertEqual(args.max_position_fraction, 0.20)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.14)
        self.assertEqual(args.no_side_max_price, 0.94)
        self.assertEqual(args.no_side_high_confidence_min_edge, 0.05)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.98)
        self.assertEqual(args.bounded_bucket_min_price, 0.70)

    def test_strategy_profile_can_pace_new_exposure_across_global_runs(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-paced-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.25,
            max_new_exposure_fraction_per_run=None,
            max_new_exposure_usd_per_run=None,
            new_exposure_target_positions_per_run=None,
            max_position_usd=175.0,
            max_position_fraction=0.35,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.0)
        self.assertEqual(args.max_new_exposure_fraction_per_run, 0.25)
        self.assertIsNone(args.max_new_exposure_usd_per_run)
        self.assertIsNone(args.new_exposure_target_positions_per_run)
        self.assertEqual(args.max_position_usd, 50.0)
        self.assertEqual(args.max_position_fraction, 0.20)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.14)
        self.assertEqual(args.no_side_max_price, 0.94)
        self.assertEqual(args.no_side_high_confidence_min_edge, 0.05)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.98)
        self.assertEqual(args.bounded_bucket_min_price, 0.70)

    def test_strategy_profile_can_pace_new_exposure_into_position_slots(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-paced-0.25-slots-4-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.25,
            max_new_exposure_fraction_per_run=None,
            max_new_exposure_usd_per_run=None,
            new_exposure_target_positions_per_run=None,
            max_position_usd=175.0,
            max_position_fraction=0.35,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.0)
        self.assertEqual(args.max_new_exposure_fraction_per_run, 0.25)
        self.assertEqual(args.new_exposure_target_positions_per_run, 4.0)
        self.assertEqual(args.max_position_usd, 50.0)
        self.assertEqual(args.max_position_fraction, 0.20)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.14)
        self.assertEqual(args.no_side_max_price, 0.94)
        self.assertEqual(args.no_side_high_confidence_min_edge, 0.05)
        self.assertEqual(args.bounded_bucket_min_edge, 0.10)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.98)
        self.assertEqual(args.bounded_bucket_min_price, 0.70)

    def test_strategy_profile_can_use_two_position_slot_live_pacing(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-paced-0.25-slots-2-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.25,
            max_new_exposure_fraction_per_run=None,
            max_new_exposure_usd_per_run=None,
            new_exposure_target_positions_per_run=None,
            max_position_usd=175.0,
            max_position_fraction=0.35,
        )

        _apply_strategy_profile(args)

        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.0)
        self.assertEqual(args.max_new_exposure_fraction_per_run, 0.25)
        self.assertEqual(args.new_exposure_target_positions_per_run, 2.0)
        self.assertEqual(args.max_position_fraction, 0.20)

    def test_strategy_profile_can_use_window_bankroll_kelly_sizing(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.50-strict-no-tail-0.14-bprice-0.70",
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.25,
            kelly_sizing_bankroll_fraction_per_run=None,
            max_new_exposure_fraction_per_run=None,
            max_new_exposure_usd_per_run=None,
            new_exposure_target_positions_per_run=2.0,
            max_position_usd=175.0,
            max_position_fraction=0.20,
        )

        _apply_strategy_profile(args)

        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.0)
        self.assertEqual(args.kelly_sizing_bankroll_fraction_per_run, 0.25)
        self.assertEqual(args.max_new_exposure_fraction_per_run, 0.25)
        self.assertIsNone(args.new_exposure_target_positions_per_run)
        self.assertEqual(args.max_position_fraction, 0.50)

    def test_strategy_profile_can_use_smaller_window_bankroll_position_cap(self) -> None:
        args = argparse.Namespace(
            strategy_profile="live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70",
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.25,
            kelly_sizing_bankroll_fraction_per_run=None,
            max_new_exposure_fraction_per_run=None,
            max_new_exposure_usd_per_run=None,
            new_exposure_target_positions_per_run=2.0,
            max_position_usd=175.0,
            max_position_fraction=0.20,
        )

        _apply_strategy_profile(args)

        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.0)
        self.assertEqual(args.kelly_sizing_bankroll_fraction_per_run, 0.25)
        self.assertEqual(args.max_new_exposure_fraction_per_run, 0.25)
        self.assertIsNone(args.new_exposure_target_positions_per_run)
        self.assertEqual(args.max_position_fraction, 0.25)

    def test_strategy_profile_can_use_bounded_low_dispersion_candidate(self) -> None:
        args = argparse.Namespace(
            strategy_profile=(
                "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-"
                "bounded-fv94-stdev03"
            ),
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.0,
            max_position_usd=175.0,
            max_position_fraction=0.35,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
            bounded_bucket_max_probability_stdev=None,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.25)
        self.assertEqual(args.max_position_fraction, 0.20)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.14)
        self.assertEqual(args.bounded_bucket_min_edge, 0.04)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.94)
        self.assertEqual(args.bounded_bucket_min_price, 0.70)
        self.assertEqual(args.bounded_bucket_max_probability_stdev, 0.03)

    def test_strategy_profile_can_use_strict_bounded_low_dispersion_candidate(self) -> None:
        args = argparse.Namespace(
            strategy_profile=(
                "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-"
                "bounded-fv95-edge08-stdev03"
            ),
            allow_no_side_entries=False,
            bankroll_usd=100.0,
            kelly_fraction=0.75,
            cash_reserve_fraction=0.0,
            max_position_usd=175.0,
            max_position_fraction=0.35,
            no_side_max_counter_event_probability=0.05,
            no_side_max_price=0.95,
            no_side_high_confidence_min_edge=0.02,
            hold_no_side_high_conviction_min_fair_value=None,
            hold_no_side_high_conviction_min_edge=None,
            hold_no_side_high_conviction_counter_event_probability=None,
            bounded_bucket_min_edge=0.15,
            bounded_bucket_min_fair_value=0.90,
            bounded_bucket_min_price=0.50,
            bounded_bucket_max_probability_stdev=None,
        )

        _apply_strategy_profile(args)

        self.assertTrue(args.allow_no_side_entries)
        self.assertEqual(args.bankroll_usd, 50.0)
        self.assertEqual(args.kelly_fraction, 0.50)
        self.assertEqual(args.cash_reserve_fraction, 0.25)
        self.assertEqual(args.max_position_fraction, 0.20)
        self.assertEqual(args.no_side_max_counter_event_probability, 0.14)
        self.assertEqual(args.bounded_bucket_min_edge, 0.08)
        self.assertEqual(args.bounded_bucket_min_fair_value, 0.95)
        self.assertEqual(args.bounded_bucket_min_price, 0.70)
        self.assertEqual(args.bounded_bucket_max_probability_stdev, 0.03)

    def test_manual_strategy_profile_leaves_explicit_settings_unchanged(self) -> None:
        args = argparse.Namespace(
            strategy_profile="manual",
            allow_no_side_entries=False,
            no_side_relaxed_counter_event_probability=None,
            bankroll_usd=250.0,
        )

        _apply_strategy_profile(args)

        self.assertFalse(args.allow_no_side_entries)
        self.assertIsNone(args.no_side_relaxed_counter_event_probability)
        self.assertEqual(args.bankroll_usd, 250.0)

    def test_long_backtest_summary_includes_strategy_profile(self) -> None:
        summary = _long_backtest_summary(
            {
                "strategy_profile": "live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94-bounded-confirmed",
                "bankroll_usd": 50.0,
                "ending_equity_usd": 50.0,
            }
        )

        self.assertEqual(
            summary["strategy_profile"],
            "live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94-bounded-confirmed",
        )

    def test_build_live_like_backtest_uses_year_window_live_cadence_and_polymarket_settlement(self) -> None:
        with patch("weather_strategy.cli.run_long_historical_backtest") as run_backtest:
            run_backtest.return_value = {
                "strategy_profile": "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70",
                "bankroll_usd": 50.0,
                "ending_equity_usd": 50.0,
                "run_log_path": "run.json",
            }
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "build-live-like-backtest",
                        "--strategy-profile",
                        "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70",
                        "--lookback-days",
                        "365",
                        "--max-end-date",
                        "2026-06-29",
                        "--summary-only",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = run_backtest.call_args.kwargs
        self.assertEqual(kwargs["min_end_date"], date(2025, 6, 30))
        self.assertEqual(kwargs["max_end_date"], date(2026, 6, 29))
        self.assertEqual(kwargs["entry_hours_utc"], (0, 6, 12, 18))
        self.assertEqual(kwargs["settlement_audit"], "polymarket_only")
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["run_log_path"], "run.json")

    def test_replay_scored_outcomes_cli_sweeps_profiles_from_saved_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "long-backtest.json"
            log_dir = Path(directory) / "replays"
            source_path.write_text(
                json.dumps(
                    {
                        "strategy_profile": "source-profile",
                        "run_log_path": str(source_path),
                        "session_count": 1,
                        "markets_scored": 1,
                        "outcomes_scored": 1,
                        "settings": {
                            "entry_hours_utc": [0, 6, 12, 18],
                            "entry_hours_match_live_forward": True,
                            "min_lead_days": 1,
                            "max_lead_days": 2,
                            "price_source": "telonex",
                            "market_source": "telonex",
                        },
                        "real_data_audit": {"passed": True},
                        "scored_outcomes_detail": [
                            {
                                "generated_at": "2026-01-01T00:00:00+00:00",
                                "token_id": "tok-yes-1",
                                "question": "Will the highest temperature in Test City be 80°F or above on January 2?",
                                "bucket": "80°F or higher",
                                "city": "Test City",
                                "target_date": "2026-01-02",
                                "market_price": 0.40,
                                "exit_price": 0.39,
                                "fair_value": 0.86,
                                "edge": 0.43,
                                "model_count": 3,
                                "model_agreement": 1.0,
                                "entry_eligible": True,
                                "bucket_width_f": None,
                                "polymarket_payout": 1,
                                "payout": 1,
                                "weather_outcome": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "replay-scored-outcomes",
                        "--source-run-log",
                        str(source_path),
                        "--strategy-profiles",
                        "manual,live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
                        "--bankroll-usd",
                        "100",
                        "--kelly-fraction",
                        "0.50",
                        "--compound-kelly-sizing",
                        "--max-position-usd",
                        "50",
                        "--max-position-fraction",
                        "0.25",
                        "--run-log-dir",
                        str(log_dir),
                        "--summary-only",
                    ]
                )

            self.assertEqual(exit_code, 0)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["replay_count"], 2)
            self.assertEqual({profile["strategy_profile"] for profile in output["profiles"]}, {"manual", "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70"})
            self.assertTrue(all(profile["cache_replay"]["uses_cached_scored_outcomes_only"] for profile in output["profiles"]))
            self.assertTrue(all(Path(profile["run_log_path"]).exists() for profile in output["profiles"]))

    def test_fixture_paper_run_records_trade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "paper.sqlite"
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "paper-run",
                        "--fixture",
                        "tests/fixtures/weather_markets.json",
                        "--ledger",
                        str(ledger_path),
                        "--min-model-count",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(PaperLedger(ledger_path).open_trades()), 1)

    def test_fixture_paper_run_writes_detailed_run_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "paper.sqlite"
            log_dir = Path(directory) / "logs"
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "paper-run",
                        "--fixture",
                        "tests/fixtures/weather_markets.json",
                        "--ledger",
                        str(ledger_path),
                        "--min-model-count",
                        "1",
                        "--run-log-dir",
                        str(log_dir),
                    ]
                )
            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            run_log_path = Path(summary["run_log_path"])
            self.assertTrue(run_log_path.exists())
            self.assertEqual(run_log_path.parent, log_dir)
            detail = json.loads(run_log_path.read_text(encoding="utf-8"))
            self.assertEqual(detail["forecast_score_rows_inserted"], summary["forecast_score_rows_inserted"])
            self.assertIn("signal_settings", detail)
            self.assertEqual(detail["run_log_schema_version"], 2)
            self.assertIn("data_provenance", detail)
            self.assertEqual(detail["data_provenance"]["execution_mode"], "paper-only Kelly ledger; no real orders are sent")
            self.assertIn("coverage_diagnostics", detail)
            self.assertIn("signal_filter_counts", detail)
            self.assertEqual(detail["edge_position_full_cap_edge"], 0.25)
            self.assertIn("scored_outcomes_detail", detail)
            self.assertGreaterEqual(len(detail["scored_outcomes_detail"]), 1)
            first_score = detail["scored_outcomes_detail"][0]
            self.assertIn("token_id", first_score)
            self.assertIn("city", first_score)
            self.assertIn("target_date", first_score)
            self.assertIn("passes_signal_filter", first_score)

    def test_run_log_paths_are_unique_for_quick_repeated_cli_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _make_run_log_path(directory, "paper-run")
            second = _make_run_log_path(directory, "paper-run")

        self.assertNotEqual(first, second)
        self.assertTrue(first.name.endswith("-paper-run.json"))
        self.assertTrue(second.name.endswith("-paper-run.json"))

    def test_live_collateral_balance_after_executions_polls_stale_balance(self) -> None:
        class FakeLiveExecutor:
            def __init__(self) -> None:
                self.values = [17.0, 17.0, 2.0]

            def collateral_balance_usd(self) -> float:
                return self.values.pop(0)

        balance = _live_collateral_balance_after_executions(
            FakeLiveExecutor(),
            executions=1,
            starting_balance=17.0,
            poll_delay_seconds=0.0,
        )
        self.assertEqual(balance, 2.0)

    def test_fixture_paper_run_can_record_explicit_no_token_position(self) -> None:
        fixture = [
            {
                "id": "binary-no",
                "question": "Will the highest temperature in New York City be 80°F or higher on June 5?",
                "slug": "highest-temperature-new-york-june-5-80-or-above",
                "eventTitle": "Highest temperature in New York City on June 5?",
                "description": "Official weather station.",
                "outcomes": "[\"Yes\", \"No\"]",
                "clobTokenIds": "[\"tok-yes\", \"tok-no\"]",
                "outcomePrices": "[\"0.80\", \"0.20\"]",
                "forecastSamplesF": [65.0, 66.0, 67.0, 68.0, 69.0],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "binary.json"
            ledger_path = Path(directory) / "paper.sqlite"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "paper-run",
                        "--fixture",
                        str(fixture_path),
                        "--ledger",
                        str(ledger_path),
                        "--disable-observations",
                        "--allow-no-side-entries",
                        "--min-model-count",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["signals"], 1)
            positions = PaperLedger(ledger_path).positions()
            self.assertEqual(len(positions), 1)
            self.assertEqual(positions[0]["token_id"], "tok-no")
            self.assertTrue(positions[0]["bucket_label"].startswith("NO: "))

    def test_already_parsed_markets_preserve_raw_payload_for_no_side_scoring(self) -> None:
        raw = {
            "id": "binary-no",
            "question": "Will the highest temperature in New York City be 80°F or higher on June 5?",
            "slug": "highest-temperature-new-york-june-5-80-or-above",
            "eventTitle": "Highest temperature in New York City on June 5?",
            "description": "Official weather station.",
            "outcomes": "[\"Yes\", \"No\"]",
            "clobTokenIds": "[\"tok-yes\", \"tok-no\"]",
            "outcomePrices": "[\"0.80\", \"0.20\"]",
        }
        parsed = parse_weather_market(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None

        [(market, preserved_raw)] = list(_parse_markets([parsed], already_parsed=True))

        self.assertEqual(market.id, "binary-no")
        self.assertEqual(preserved_raw["clobTokenIds"], raw["clobTokenIds"])
        self.assertEqual(preserved_raw["outcomes"], raw["outcomes"])

    def test_should_skip_late_same_day_market_without_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "late",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_latest_entry_hour_local=14),
            max_lead_days=2,
            open_token_ids=set(),
            now=datetime(2026, 6, 4, 20, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(should_process)
        self.assertIn("cutoff", reason or "")

    def test_should_process_late_same_day_market_with_open_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "late",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_latest_entry_hour_local=14),
            max_lead_days=2,
            open_token_ids={"tok-yes"},
            now=datetime(2026, 6, 4, 20, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(should_process)
        self.assertIsNone(reason)

    def test_min_lead_days_skips_same_day_without_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "same-day",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_earliest_entry_hour_local=11, same_day_latest_entry_hour_local=17),
            max_lead_days=2,
            open_token_ids=set(),
            min_lead_days=1,
            now=datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(should_process)
        self.assertIn("lead window", reason or "")

    def test_min_lead_days_still_processes_open_same_day_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "same-day",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_earliest_entry_hour_local=11, same_day_latest_entry_hour_local=17),
            max_lead_days=2,
            open_token_ids={"tok-yes"},
            min_lead_days=1,
            now=datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(should_process)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
