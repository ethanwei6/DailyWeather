from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weather_strategy.backtest import load_calibration_weights, load_probability_calibration, run_backtest
from weather_strategy.backtest_engine import (
    CURRENT_LIVE_STRATEGY_PROFILE,
    DEFAULT_STRATEGY_COMPARISON_PROFILES,
    compare_strategy_replays,
    live_like_backtest_dates,
)
from weather_strategy.cities import city_with_station_coordinates, find_city
from weather_strategy.execution import PolymarketLiveExecutionAdapter
from weather_strategy.forecast import ConsensusForecastEngine
from weather_strategy.live_setup import (
    LiveSetupError,
    check_geoblock,
    clob_readonly_smoke,
    create_bot_wallet,
    derive_clob_credentials,
    live_status,
)
from weather_strategy.long_backtest import LIVE_FORWARD_ENTRY_HOURS_UTC, run_cached_scored_outcome_replay, run_long_historical_backtest
from weather_strategy.model_validation import run_weather_model_validation
from weather_strategy.models import ForecastDistribution, ScoredOutcome, TradeSignal, WeatherMarket
from weather_strategy.observations import ObservedHigh, ObservedHighClient
from weather_strategy.paper import PaperLedger
from weather_strategy.parser import parse_jsonish_list, parse_weather_market
from weather_strategy.polymarket import PolymarketClobClient, PolymarketGammaClient
from weather_strategy.signals import (
    SignalSettings,
    invert_binary_scored_outcome,
    market_entry_timing,
    score_outcomes,
    signal_filter_reason,
    signals_from_scored_outcomes,
)
from weather_strategy.weather import OpenMeteoClient


STRATEGY_PROFILE_CHOICES = (
    "manual",
    "live-forward-strict-100",
    "live-forward-utc12-relaxed-no-tail-0.20",
    "live-forward-utc12-relaxed-no-tail-0.20-trim-holds",
    "live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-holds",
    "live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-bounded-edge-0.15",
    "live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15",
    "live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.15",
    "live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.10",
    "live-forward-strict-no-tail-0.11-preserve-highconv-bounded-edge-0.10",
    "live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94",
    "live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94-bounded-confirmed",
    "live-forward-50-highwin-strict-no-tail-0.12-no-side-max-0.94-bounded-confirmed-cap-0.30",
    "live-forward-50-highwin-strict-no-tail-0.14-no-side-max-0.94-bounded-confirmed-cap-0.35-bprice-0.70",
    "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-bounded-fv94-stdev03",
    "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-bounded-fv95-edge08-stdev03",
    "live-forward-50-paced-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-paced-0.25-slots-2-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-paced-0.25-slots-4-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.50-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.20-strict-no-tail-0.14-bprice-0.70",
)

LIVE_FORWARD_PROFILE_SETTINGS: dict[str, Any] = {
    "min_edge": 0.08,
    "min_model_agreement": 1.0,
    "hold_min_model_agreement": 0.65,
    "hold_min_fair_value": 0.60,
    "hold_market_confirmation_price": 0.80,
    "hold_market_confirmation_min_fair_value": 0.50,
    "min_signal_fair_value": 0.70,
    "min_price": 0.125,
    "yes_side_min_price": 0.20,
    "allow_no_side_entries": True,
    "no_side_min_edge": 0.10,
    "no_side_high_confidence_min_edge": 0.02,
    "no_side_max_price": 0.95,
    "no_side_max_counter_event_probability": 0.10,
    "no_side_relaxed_counter_event_probability": None,
    "no_side_relaxed_counter_event_hours_utc": "",
    "hold_no_side_max_counter_event_probability": 0.15,
    "hold_no_side_high_conviction_min_fair_value": None,
    "hold_no_side_high_conviction_min_edge": None,
    "hold_no_side_high_conviction_counter_event_probability": None,
    "invalid_hold_partial_exit_fraction": 0.50,
    "invalid_hold_partial_exit_min_fair_value": 0.90,
    "invalid_hold_partial_exit_min_price": 0.50,
    "invalid_hold_partial_exit_max_price": 0.65,
    "high_confidence_price_threshold": 0.75,
    "high_confidence_min_kelly_edge": 0.02,
    "bankroll_usd": 100.0,
    "kelly_fraction": 0.75,
    "compound_kelly_sizing": True,
    "cash_reserve_fraction": 0.0,
    "max_new_exposure_fraction_per_run": None,
    "max_new_exposure_usd_per_run": None,
    "new_exposure_target_positions_per_run": None,
    "kelly_sizing_bankroll_fraction_per_run": None,
    "max_position_usd": 175.0,
    "max_position_fraction": 0.25,
    "kelly_market_blend": 0.0,
    "edge_position_full_cap_edge": 0.25,
    "edge_position_min_multiplier": 0.35,
    "min_trade_usd": 1.0,
    "trim_valid_holds_to_kelly_target": False,
    "min_lead_days": 1,
    "max_lead_days": 2,
}

STRATEGY_PROFILE_SETTINGS: dict[str, dict[str, Any]] = {
    "manual": {},
    "live-forward-strict-100": LIVE_FORWARD_PROFILE_SETTINGS,
    "live-forward-utc12-relaxed-no-tail-0.20": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "no_side_relaxed_counter_event_probability": 0.20,
        "no_side_relaxed_counter_event_hours_utc": "12",
    },
    "live-forward-utc12-relaxed-no-tail-0.20-trim-holds": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "no_side_relaxed_counter_event_probability": 0.20,
        "no_side_relaxed_counter_event_hours_utc": "12",
        "trim_valid_holds_to_kelly_target": True,
    },
    "live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-holds": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "no_side_relaxed_counter_event_probability": 0.20,
        "no_side_relaxed_counter_event_hours_utc": "12",
        "trim_valid_holds_to_kelly_target": True,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
    },
    "live-forward-utc12-relaxed-no-tail-0.20-trim-highconv-bounded-edge-0.15": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "no_side_relaxed_counter_event_probability": 0.20,
        "no_side_relaxed_counter_event_hours_utc": "12",
        "trim_valid_holds_to_kelly_target": True,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.15,
    },
    "live-forward-strict-no-tail-trim-highconv-bounded-edge-0.15": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "trim_valid_holds_to_kelly_target": True,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.15,
    },
    "live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.15": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.15,
    },
    "live-forward-strict-no-tail-preserve-highconv-bounded-edge-0.10": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
    },
    "live-forward-strict-no-tail-0.11-preserve-highconv-bounded-edge-0.10": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "no_side_max_counter_event_probability": 0.11,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
    },
    "live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "max_position_usd": 50.0,
        "no_side_max_counter_event_probability": 0.11,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
    },
    "live-forward-50-highwin-strict-no-tail-0.11-no-side-max-0.94-bounded-confirmed": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "max_position_usd": 50.0,
        "no_side_max_counter_event_probability": 0.11,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.75,
    },
    "live-forward-50-highwin-strict-no-tail-0.12-no-side-max-0.94-bounded-confirmed-cap-0.30": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.30,
        "no_side_max_counter_event_probability": 0.12,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.75,
    },
    "live-forward-50-highwin-strict-no-tail-0.14-no-side-max-0.94-bounded-confirmed-cap-0.35-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.35,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
    "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.25,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.20,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
    "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-bounded-fv94-stdev03": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.25,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.20,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.04,
        "bounded_bucket_min_fair_value": 0.94,
        "bounded_bucket_min_price": 0.70,
        "bounded_bucket_max_probability_stdev": 0.03,
    },
    "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70-bounded-fv95-edge08-stdev03": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.25,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.20,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.08,
        "bounded_bucket_min_fair_value": 0.95,
        "bounded_bucket_min_price": 0.70,
        "bounded_bucket_max_probability_stdev": 0.03,
    },
    "live-forward-50-paced-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.0,
        "max_new_exposure_fraction_per_run": 0.25,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.20,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
    "live-forward-50-paced-0.25-slots-2-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.0,
        "max_new_exposure_fraction_per_run": 0.25,
        "new_exposure_target_positions_per_run": 2.0,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.20,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
    "live-forward-50-paced-0.25-slots-4-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.0,
        "max_new_exposure_fraction_per_run": 0.25,
        "new_exposure_target_positions_per_run": 4.0,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.20,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.50-strict-no-tail-0.14-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.0,
        "kelly_sizing_bankroll_fraction_per_run": 0.25,
        "max_new_exposure_fraction_per_run": 0.25,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.50,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.0,
        "kelly_sizing_bankroll_fraction_per_run": 0.25,
        "max_new_exposure_fraction_per_run": 0.25,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.25,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.20-strict-no-tail-0.14-bprice-0.70": {
        **LIVE_FORWARD_PROFILE_SETTINGS,
        "bankroll_usd": 50.0,
        "kelly_fraction": 0.50,
        "cash_reserve_fraction": 0.0,
        "kelly_sizing_bankroll_fraction_per_run": 0.25,
        "max_new_exposure_fraction_per_run": 0.25,
        "max_position_usd": 50.0,
        "max_position_fraction": 0.20,
        "no_side_max_counter_event_probability": 0.14,
        "no_side_max_price": 0.94,
        "no_side_high_confidence_min_edge": 0.05,
        "hold_no_side_high_conviction_min_fair_value": 0.98,
        "hold_no_side_high_conviction_min_edge": 0.35,
        "hold_no_side_high_conviction_counter_event_probability": 0.20,
        "bounded_bucket_min_edge": 0.10,
        "bounded_bucket_min_fair_value": 0.98,
        "bounded_bucket_min_price": 0.70,
    },
}


def _add_strategy_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy-profile",
        choices=STRATEGY_PROFILE_CHOICES,
        default="manual",
        help=(
            "Optional named strategy preset. 'manual' leaves explicit flags unchanged; "
            "live-forward-* presets apply the backtest/live-aligned paper settings."
        ),
    )


def _apply_strategy_profile(args: argparse.Namespace) -> None:
    profile = getattr(args, "strategy_profile", "manual") or "manual"
    values = STRATEGY_PROFILE_SETTINGS.get(profile)
    if values is None:
        raise ValueError(f"Unknown strategy profile: {profile}")
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)


def _add_cached_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bankroll-usd", type=float, default=100.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--compound-kelly-sizing", action="store_true", help="Size Kelly targets from current replay equity instead of the starting bankroll")
    parser.add_argument("--cash-reserve-fraction", type=float, default=0.0, help="Accepted for strategy-profile compatibility; cached replay uses the already-scored rows")
    parser.add_argument("--max-new-exposure-usd-per-run", type=float, default=None, help="Cap total new BUY notional per replay session")
    parser.add_argument("--max-new-exposure-fraction-per-run", type=float, default=None, help="Cap total new BUY notional per replay session as a fraction of sizing bankroll/equity")
    parser.add_argument("--new-exposure-target-positions-per-run", type=float, default=None, help="Divide the per-run new-exposure budget across at least this many new position slots")
    parser.add_argument("--kelly-sizing-bankroll-fraction-per-run", type=float, default=None, help="Use this fraction of sizing bankroll for Kelly target calculations in each replay session")
    parser.add_argument("--max-position-usd", type=float, default=100.0)
    parser.add_argument("--max-position-fraction", type=float, default=0.15, help="Optional cap as a fraction of sizing bankroll/equity; the stricter of this and --max-position-usd is used")
    parser.add_argument("--kelly-market-blend", type=float, default=0.0, help="Blend model FV toward market price for Kelly sizing only; 0 keeps raw FV, 1 sizes to zero model edge")
    parser.add_argument("--edge-position-full-cap-edge", type=float, default=0.25, help="If >0, scale max position by buffered edge; full max-position applies at this edge")
    parser.add_argument("--edge-position-min-multiplier", type=float, default=0.35, help="Minimum max-position multiplier when edge-scaled sizing is enabled")
    parser.add_argument("--min-trade-usd", type=float, default=1.0)
    parser.add_argument("--min-edge", type=float, default=0.08)
    parser.add_argument("--uncertainty-buffer", type=float, default=0.03)
    parser.add_argument("--min-model-count", type=int, default=3)
    parser.add_argument("--min-model-agreement", type=float, default=1.0)
    parser.add_argument("--hold-min-model-agreement", type=float, default=0.65)
    parser.add_argument("--hold-min-fair-value", type=float, default=0.60)
    parser.add_argument("--hold-market-confirmation-price", type=float, default=0.80)
    parser.add_argument("--hold-market-confirmation-min-fair-value", type=float, default=0.50)
    parser.add_argument("--trim-valid-holds-to-kelly-target", action="store_true")
    parser.add_argument("--high-confidence-price-threshold", type=float, default=0.75)
    parser.add_argument("--high-confidence-min-kelly-edge", type=float, default=0.02)
    parser.add_argument("--low-price-exact-bucket-threshold", type=float, default=0.20)
    parser.add_argument("--low-price-exact-bucket-min-fair-value", type=float, default=0.22)
    parser.add_argument("--low-price-exact-bucket-min-edge", type=float, default=0.08)
    parser.add_argument("--correlated-exact-bucket-max-price", type=float, default=0.15)
    parser.add_argument("--correlated-exact-bucket-min-agreement", type=float, default=0.95)
    parser.add_argument("--exact-bucket-max-width-f", type=float, default=2.25)
    parser.add_argument("--min-price", type=float, default=0.125)
    parser.add_argument("--yes-side-min-price", type=float, default=0.20)
    parser.add_argument("--no-side-max-price", type=float, default=0.95, help="Maximum entry price for NO-token entries; set at or above --max-price to disable the side-specific cap")
    parser.add_argument("--no-side-max-counter-event-probability", type=float, default=0.10, help="For NO-token entries, reject rows if any model view gives the opposite YES event more than this probability; set >=1 to disable")
    parser.add_argument("--no-side-relaxed-counter-event-probability", type=float, default=None, help="Optional relaxed NO counter-event cap used only during --no-side-relaxed-counter-event-hours-utc")
    parser.add_argument("--no-side-relaxed-counter-event-hours-utc", default="", help="Comma-separated UTC hours that may use --no-side-relaxed-counter-event-probability")
    parser.add_argument("--hold-no-side-max-counter-event-probability", type=float, default=0.15, help="For existing NO-token positions, allow holding while the opposite YES tail is below this probability; set >=1 to disable")
    parser.add_argument("--hold-no-side-high-conviction-min-fair-value", type=float, default=None, help="Optional FV floor for using the wider high-conviction NO hold counter-event cap")
    parser.add_argument("--hold-no-side-high-conviction-min-edge", type=float, default=None, help="Optional edge floor for using the wider high-conviction NO hold counter-event cap")
    parser.add_argument("--hold-no-side-high-conviction-counter-event-probability", type=float, default=None, help="Optional wider NO hold counter-event cap used only for high-conviction existing positions")
    parser.add_argument("--invalid-hold-partial-exit-fraction", type=float, default=None, help="If set, sell only this fraction of an invalid existing hold when the FV/price partial-exit gate passes")
    parser.add_argument("--invalid-hold-partial-exit-min-fair-value", type=float, default=0.90)
    parser.add_argument("--invalid-hold-partial-exit-min-price", type=float, default=0.50)
    parser.add_argument("--invalid-hold-partial-exit-max-price", type=float, default=0.65)
    parser.add_argument("--min-signal-fair-value", type=float, default=0.70)
    parser.add_argument("--allow-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_true", default=True)
    parser.add_argument("--disable-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_false")
    parser.add_argument("--allow-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_true", default=True)
    parser.add_argument("--disable-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_false")
    parser.add_argument("--bounded-bucket-min-edge", type=float, default=0.10)
    parser.add_argument("--bounded-bucket-min-fair-value", type=float, default=0.90)
    parser.add_argument("--bounded-bucket-min-model-agreement", type=float, default=1.0)
    parser.add_argument("--bounded-bucket-min-price", type=float, default=0.50)
    parser.add_argument("--bounded-bucket-max-probability-stdev", type=float, default=None)
    parser.add_argument("--max-price", type=float, default=0.95)
    parser.add_argument("--allow-no-side-entries", action="store_true", help="Evaluate buying real NO-token rows from the cached scored artifact")
    parser.add_argument("--no-side-min-edge", type=float, default=0.10, help="Minimum absolute buffered edge required for NO-token entries")
    parser.add_argument("--no-side-high-confidence-min-edge", type=float, default=0.02, help="Minimum absolute buffered edge for NO entries when the NO price is at or above --high-confidence-price-threshold")
    parser.add_argument("--same-day-entry-start-hour", type=int, default=11)
    parser.add_argument("--same-day-entry-cutoff-hour", type=int, default=17)
    parser.add_argument("--min-lead-days", type=int, default=1, help="Recorded in settings for profile compatibility; source artifact controls replay timing")
    parser.add_argument("--max-lead-days", type=int, default=2, help="Recorded in settings for profile compatibility; source artifact controls replay timing")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Weather Polymarket strategy tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper = subparsers.add_parser("paper-run", help="Generate and record paper-trade signals")
    _add_strategy_profile_argument(paper)
    paper.add_argument("--fixture", help="Explicit fixture JSON file for deterministic local runs")
    paper.add_argument("--ledger", default="work/data/paper_trades.sqlite", help="SQLite paper-trade ledger path")
    paper.add_argument("--limit", type=int, default=50, help="Maximum eligible live markets to score when no fixture is supplied")
    paper.add_argument("--discovery-request-limit", type=int, default=50, help="Maximum Gamma results requested per network call")
    paper.add_argument("--discovery-pages", type=int, default=1, help="Gamma pages to scan per discovery source")
    paper.add_argument("--max-runtime-seconds", type=float, default=150.0, help="Stop scoring new markets after this many seconds; <=0 disables")
    paper.add_argument("--progress-every", type=int, default=10, help="Emit progress to stderr every N scored markets; <=0 disables")
    paper.add_argument("--min-edge", type=float, default=0.08)
    paper.add_argument("--uncertainty-buffer", type=float, default=0.03)
    paper.add_argument("--max-spread", type=float, default=0.10)
    paper.add_argument("--size-usd", type=float, default=10.0)
    paper.add_argument("--bankroll-usd", type=float, default=1000.0)
    paper.add_argument("--kelly-fraction", type=float, default=0.25)
    paper.add_argument("--compound-kelly-sizing", action="store_true", help="Size Kelly targets from current paper equity instead of the starting bankroll")
    paper.add_argument("--cash-reserve-fraction", type=float, default=0.0, help="Reserve this fraction of starting bankroll as untradeable cash for sizing and live buy caps")
    paper.add_argument("--max-new-exposure-usd-per-run", type=float, default=None, help="Cap total new BUY notional in one automation run")
    paper.add_argument("--max-new-exposure-fraction-per-run", type=float, default=None, help="Cap total new BUY notional in one automation run as a fraction of sizing bankroll/equity")
    paper.add_argument("--new-exposure-target-positions-per-run", type=float, default=None, help="Divide the per-run new-exposure budget across at least this many new position slots")
    paper.add_argument("--kelly-sizing-bankroll-fraction-per-run", type=float, default=None, help="Use this fraction of sizing bankroll for Kelly target calculations in this automation run")
    paper.add_argument("--max-position-usd", type=float, default=1000.0)
    paper.add_argument("--max-position-fraction", type=float, default=0.15, help="Optional cap as a fraction of sizing bankroll/equity; the stricter of this and --max-position-usd is used")
    paper.add_argument("--kelly-market-blend", type=float, default=0.0, help="Blend model FV toward market price for Kelly sizing only; 0 keeps raw FV, 1 sizes to zero model edge")
    paper.add_argument("--edge-position-full-cap-edge", type=float, default=0.25, help="If >0, scale max position by buffered edge; full max-position applies at this edge")
    paper.add_argument("--edge-position-min-multiplier", type=float, default=0.35, help="Minimum max-position multiplier when edge-scaled sizing is enabled")
    paper.add_argument("--min-trade-usd", type=float, default=1.0)
    paper.add_argument("--min-model-count", type=int, default=3)
    paper.add_argument("--min-model-agreement", type=float, default=1.0)
    paper.add_argument("--hold-min-model-agreement", type=float, default=0.65)
    paper.add_argument("--hold-min-fair-value", type=float, default=0.60)
    paper.add_argument("--hold-market-confirmation-price", type=float, default=0.80)
    paper.add_argument("--hold-market-confirmation-min-fair-value", type=float, default=0.50)
    paper.add_argument("--trim-valid-holds-to-kelly-target", action="store_true")
    paper.add_argument("--high-confidence-price-threshold", type=float, default=0.75)
    paper.add_argument("--high-confidence-min-kelly-edge", type=float, default=0.02)
    paper.add_argument("--low-price-exact-bucket-threshold", type=float, default=0.20)
    paper.add_argument("--low-price-exact-bucket-min-fair-value", type=float, default=0.22)
    paper.add_argument("--low-price-exact-bucket-min-edge", type=float, default=0.08)
    paper.add_argument("--correlated-exact-bucket-max-price", type=float, default=0.15)
    paper.add_argument("--correlated-exact-bucket-min-agreement", type=float, default=0.95)
    paper.add_argument("--exact-bucket-max-width-f", type=float, default=2.25)
    paper.add_argument("--min-price", type=float, default=0.125)
    paper.add_argument("--yes-side-min-price", type=float, default=0.20, help="Minimum entry price for YES-token entries; NO research entries use --min-price")
    paper.add_argument("--allow-no-side-entries", action="store_true", help="Paper-trade explicit NO tokens for binary markets using real NO quotes")
    paper.add_argument("--no-side-min-edge", type=float, default=0.10, help="Minimum absolute buffered edge required for NO-token paper entries")
    paper.add_argument("--no-side-high-confidence-min-edge", type=float, default=0.02, help="Minimum absolute buffered edge for NO-token entries when the NO price is at or above --high-confidence-price-threshold")
    paper.add_argument("--no-side-max-price", type=float, default=0.95, help="Maximum entry price for NO-token entries; set at or above --max-price to disable the side-specific cap")
    paper.add_argument("--no-side-max-counter-event-probability", type=float, default=0.10, help="For NO-token research rows, reject entries if any model view gives the opposite YES event more than this probability")
    paper.add_argument("--no-side-relaxed-counter-event-probability", type=float, default=None, help="Optional relaxed NO counter-event cap used only during --no-side-relaxed-counter-event-hours-utc")
    paper.add_argument("--no-side-relaxed-counter-event-hours-utc", default="", help="Comma-separated UTC hours that may use --no-side-relaxed-counter-event-probability")
    paper.add_argument("--hold-no-side-max-counter-event-probability", type=float, default=0.15, help="For existing NO-token positions, allow holding while the opposite YES tail is below this probability")
    paper.add_argument("--hold-no-side-high-conviction-min-fair-value", type=float, default=None, help="Optional FV floor for using the wider high-conviction NO hold counter-event cap")
    paper.add_argument("--hold-no-side-high-conviction-min-edge", type=float, default=None, help="Optional edge floor for using the wider high-conviction NO hold counter-event cap")
    paper.add_argument("--hold-no-side-high-conviction-counter-event-probability", type=float, default=None, help="Optional wider NO hold counter-event cap used only for high-conviction existing positions")
    paper.add_argument("--invalid-hold-partial-exit-fraction", type=float, default=None, help="If set, sell only this fraction of an invalid existing hold when the FV/price partial-exit gate passes")
    paper.add_argument("--invalid-hold-partial-exit-min-fair-value", type=float, default=0.90)
    paper.add_argument("--invalid-hold-partial-exit-min-price", type=float, default=0.50)
    paper.add_argument("--invalid-hold-partial-exit-max-price", type=float, default=0.65)
    paper.add_argument("--min-signal-fair-value", type=float, default=0.70)
    paper.add_argument("--allow-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_true", default=True)
    paper.add_argument("--disable-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_false")
    paper.add_argument("--allow-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_true", default=True)
    paper.add_argument("--disable-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_false")
    paper.add_argument("--bounded-bucket-min-edge", type=float, default=0.10)
    paper.add_argument("--bounded-bucket-min-fair-value", type=float, default=0.90)
    paper.add_argument("--bounded-bucket-min-model-agreement", type=float, default=1.0)
    paper.add_argument("--bounded-bucket-min-price", type=float, default=0.50)
    paper.add_argument("--bounded-bucket-max-probability-stdev", type=float, default=None)
    paper.add_argument("--max-price", type=float, default=0.95)
    paper.add_argument("--same-day-entry-start-hour", type=int, default=11)
    paper.add_argument("--same-day-entry-cutoff-hour", type=int, default=17)
    paper.add_argument("--allow-late-same-day", action="store_true")
    paper.add_argument("--disable-observations", action="store_true")
    paper.add_argument("--min-lead-days", type=int, default=0)
    paper.add_argument("--max-lead-days", type=int, default=2)
    paper.add_argument("--weights-file", default="work/data/model_weights.json")
    paper.add_argument("--run-log-dir", default="work/logs/paper_runs", help="Directory for detailed JSON paper-run logs")
    paper.add_argument("--no-run-log", action="store_true", help="Disable detailed JSON paper-run log output")
    paper.add_argument("--execution-mode", choices=("paper", "live"), default="paper", help="Default is paper. Live mode sends real CLOB market orders and requires --confirm-live plus DAILYWEATHER_LIVE_TRADING=1")
    paper.add_argument("--confirm-live", action="store_true", help="Required with --execution-mode live to acknowledge real-money order placement")
    paper.add_argument("--live-env-file", default=".env.local", help="Gitignored env file containing live Polymarket credentials and risk caps")

    live = subparsers.add_parser("scan-live", help="Discover live active weather markets")
    live.add_argument("--limit", type=int, default=50)
    live.add_argument("--pages", type=int, default=1)
    live.add_argument("--json-out", help="Optional output path for parsed live markets")

    debug = subparsers.add_parser("debug-search", help="Print raw Gamma public-search event and market titles")
    debug.add_argument("--query", default="temperature")
    debug.add_argument("--limit", type=int, default=10)
    debug.add_argument("--page", type=int, default=1)

    live_setup = subparsers.add_parser(
        "live-setup",
        help="Prepare and smoke-test local international Polymarket API credentials without placing orders",
    )
    live_setup.add_argument("--env-file", default=".env.local")
    live_setup.add_argument("--create-wallet", action="store_true", help="Create a fresh local bot wallet if PRIVATE_KEY is absent")
    live_setup.add_argument("--derive-clob-creds", action="store_true", help="Create or derive CLOB API credentials from the local bot key")
    live_setup.add_argument("--check-geoblock", action="store_true", help="Check Polymarket international geoblock status for this machine")
    live_setup.add_argument("--clob-readonly-smoke", action="store_true", help="Fetch open orders/trades using CLOB credentials; no orders are sent")
    live_setup.add_argument("--overwrite", action="store_true", help="Overwrite existing .env.local keys for selected setup steps")

    report = subparsers.add_parser("report", help="Summarize paper-trading equity, PnL, and open positions")
    report.add_argument("--ledger", default="work/data/paper_trades.sqlite")
    report.add_argument("--bankroll-usd", type=float, default=1000.0)

    calibration = subparsers.add_parser("calibration", help="Summarize recorded forecast calibration data")
    calibration.add_argument("--ledger", default="work/data/paper_trades.sqlite")

    backtest = subparsers.add_parser("backtest", help="Backtest recorded forecast snapshots and fit model/source weights")
    _add_strategy_profile_argument(backtest)
    backtest.add_argument("--ledger", default="work/data/weather_kelly_paper.sqlite")
    backtest.add_argument("--bankroll-usd", type=float, default=1000.0)
    backtest.add_argument("--kelly-fraction", type=float, default=0.25)
    backtest.add_argument("--max-position-usd", type=float, default=1000.0)
    backtest.add_argument("--max-position-fraction", type=float, default=0.15, help="Optional cap as a fraction of sizing bankroll; the stricter of this and --max-position-usd is used")
    backtest.add_argument("--kelly-market-blend", type=float, default=0.0, help="Blend model FV toward market price for Kelly sizing only; 0 keeps raw FV, 1 sizes to zero model edge")
    backtest.add_argument("--edge-position-full-cap-edge", type=float, default=0.25, help="If >0, scale max position by buffered edge; full max-position applies at this edge")
    backtest.add_argument("--edge-position-min-multiplier", type=float, default=0.35, help="Minimum max-position multiplier when edge-scaled sizing is enabled")
    backtest.add_argument("--min-trade-usd", type=float, default=1.0)
    backtest.add_argument("--min-edge", type=float, default=0.08)
    backtest.add_argument("--uncertainty-buffer", type=float, default=0.03)
    backtest.add_argument("--min-model-count", type=int, default=3)
    backtest.add_argument("--min-model-agreement", type=float, default=1.0)
    backtest.add_argument("--hold-min-model-agreement", type=float, default=0.65)
    backtest.add_argument("--hold-min-fair-value", type=float, default=0.60)
    backtest.add_argument("--hold-market-confirmation-price", type=float, default=0.80)
    backtest.add_argument("--hold-market-confirmation-min-fair-value", type=float, default=0.50)
    backtest.add_argument("--trim-valid-holds-to-kelly-target", action="store_true")
    backtest.add_argument("--high-confidence-price-threshold", type=float, default=0.75)
    backtest.add_argument("--high-confidence-min-kelly-edge", type=float, default=0.02)
    backtest.add_argument("--low-price-exact-bucket-threshold", type=float, default=0.20)
    backtest.add_argument("--low-price-exact-bucket-min-fair-value", type=float, default=0.22)
    backtest.add_argument("--low-price-exact-bucket-min-edge", type=float, default=0.08)
    backtest.add_argument("--correlated-exact-bucket-max-price", type=float, default=0.15)
    backtest.add_argument("--correlated-exact-bucket-min-agreement", type=float, default=0.95)
    backtest.add_argument("--exact-bucket-max-width-f", type=float, default=2.25)
    backtest.add_argument("--min-price", type=float, default=0.125)
    backtest.add_argument("--yes-side-min-price", type=float, default=0.20)
    backtest.add_argument("--allow-no-side-entries", action="store_true")
    backtest.add_argument("--no-side-min-edge", type=float, default=0.10)
    backtest.add_argument("--no-side-high-confidence-min-edge", type=float, default=0.02)
    backtest.add_argument("--no-side-max-price", type=float, default=0.95)
    backtest.add_argument("--no-side-max-counter-event-probability", type=float, default=0.10)
    backtest.add_argument("--no-side-relaxed-counter-event-probability", type=float, default=None)
    backtest.add_argument("--no-side-relaxed-counter-event-hours-utc", default="")
    backtest.add_argument("--hold-no-side-max-counter-event-probability", type=float, default=0.15)
    backtest.add_argument("--hold-no-side-high-conviction-min-fair-value", type=float, default=None)
    backtest.add_argument("--hold-no-side-high-conviction-min-edge", type=float, default=None)
    backtest.add_argument("--hold-no-side-high-conviction-counter-event-probability", type=float, default=None)
    backtest.add_argument("--invalid-hold-partial-exit-fraction", type=float, default=None)
    backtest.add_argument("--invalid-hold-partial-exit-min-fair-value", type=float, default=0.90)
    backtest.add_argument("--invalid-hold-partial-exit-min-price", type=float, default=0.50)
    backtest.add_argument("--invalid-hold-partial-exit-max-price", type=float, default=0.65)
    backtest.add_argument("--min-signal-fair-value", type=float, default=0.70)
    backtest.add_argument("--allow-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_true", default=True)
    backtest.add_argument("--disable-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_false")
    backtest.add_argument("--allow-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_true", default=True)
    backtest.add_argument("--disable-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_false")
    backtest.add_argument("--bounded-bucket-min-edge", type=float, default=0.10)
    backtest.add_argument("--bounded-bucket-min-fair-value", type=float, default=0.90)
    backtest.add_argument("--bounded-bucket-min-model-agreement", type=float, default=1.0)
    backtest.add_argument("--bounded-bucket-min-price", type=float, default=0.50)
    backtest.add_argument("--bounded-bucket-max-probability-stdev", type=float, default=None)
    backtest.add_argument("--max-price", type=float, default=0.95)
    backtest.add_argument("--train-fraction", type=float, default=0.70)
    backtest.add_argument("--output-weights", default="work/data/model_weights.json")
    backtest.add_argument("--no-fetch-observations", action="store_true")
    backtest.add_argument("--max-observation-lookups", type=int, default=200)
    backtest.add_argument("--min-weight-samples", type=int, default=20)
    backtest.add_argument("--weight-prior-samples", type=int, default=50)
    backtest.add_argument("--run-log-dir", default="work/logs/backtests", help="Directory for detailed JSON backtest logs")
    backtest.add_argument("--no-run-log", action="store_true", help="Disable detailed JSON backtest log output")

    long_backtest = subparsers.add_parser("long-backtest", help="Run a historical live-compatible backtest using real Polymarket price history and historical forecasts")
    _add_strategy_profile_argument(long_backtest)
    long_backtest.add_argument("--bankroll-usd", type=float, default=100.0)
    long_backtest.add_argument("--pages", type=int, default=10, help="Gamma public-search pages to scan")
    long_backtest.add_argument("--limit-per-page", type=int, default=50)
    long_backtest.add_argument("--max-markets", type=int, default=8000)
    long_backtest.add_argument("--query", default="highest temperature")
    long_backtest.add_argument("--min-end-date", type=_parse_iso_date, default=None, help="Earliest market end date to include, YYYY-MM-DD")
    long_backtest.add_argument("--max-end-date", type=_parse_iso_date, default=None, help="Latest market end date to include, YYYY-MM-DD")
    long_backtest.add_argument(
        "--market-selection",
        choices=("end_date_asc", "end_date_desc", "month_balanced"),
        default="end_date_asc",
        help="How to choose Telonex markets after filtering and before --max-markets is applied",
    )
    long_backtest.add_argument(
        "--entry-hours-utc",
        default=",".join(str(hour) for hour in LIVE_FORWARD_ENTRY_HOURS_UTC),
        help="Comma-separated simulated run hours in UTC; defaults to the live automation cadence",
    )
    long_backtest.add_argument("--min-lead-days", type=int, default=1)
    long_backtest.add_argument("--max-lead-days", type=int, default=2)
    long_backtest.add_argument("--max-runtime-seconds", type=float, default=0.0)
    long_backtest.add_argument(
        "--price-source",
        choices=("telonex", "clob", "auto"),
        default="telonex",
        help="Historical Polymarket pricing source. Telonex uses tick-level quote Parquet; clob is the legacy prices-history fallback.",
    )
    long_backtest.add_argument(
        "--market-source",
        choices=("telonex", "gamma"),
        default="telonex",
        help="Historical market universe source. Telonex uses the markets dataset with actual quote availability; gamma uses public-search.",
    )
    long_backtest.add_argument(
        "--settlement-audit",
        choices=("weather_crosscheck", "polymarket_only"),
        default="weather_crosscheck",
        help=(
            "weather_crosscheck verifies station observations when available; "
            "polymarket_only settles from resolved Polymarket payouts and skips slow weather QA."
        ),
    )
    long_backtest.add_argument("--max-price-staleness-minutes", type=int, default=90)
    long_backtest.add_argument("--historical-price-slippage", type=float, default=0.01)
    long_backtest.add_argument("--forecast-availability-lag-hours", type=int, default=6)
    long_backtest.add_argument("--kelly-fraction", type=float, default=0.25)
    long_backtest.add_argument("--compound-kelly-sizing", action="store_true", help="Size Kelly targets from current replay equity instead of the starting bankroll")
    long_backtest.add_argument("--max-new-exposure-usd-per-run", type=float, default=None, help="Cap total new BUY notional per simulated run")
    long_backtest.add_argument("--max-new-exposure-fraction-per-run", type=float, default=None, help="Cap total new BUY notional per simulated run as a fraction of sizing bankroll/equity")
    long_backtest.add_argument("--new-exposure-target-positions-per-run", type=float, default=None, help="Divide the per-run new-exposure budget across at least this many new position slots")
    long_backtest.add_argument("--kelly-sizing-bankroll-fraction-per-run", type=float, default=None, help="Use this fraction of sizing bankroll for Kelly target calculations in each simulated run")
    long_backtest.add_argument("--max-position-usd", type=float, default=100.0)
    long_backtest.add_argument("--max-position-fraction", type=float, default=0.15, help="Optional cap as a fraction of sizing bankroll/equity; the stricter of this and --max-position-usd is used")
    long_backtest.add_argument("--kelly-market-blend", type=float, default=0.0, help="Blend model FV toward market price for Kelly sizing only; 0 keeps raw FV, 1 sizes to zero model edge")
    long_backtest.add_argument("--edge-position-full-cap-edge", type=float, default=0.25, help="If >0, scale max position by buffered edge; full max-position applies at this edge")
    long_backtest.add_argument("--edge-position-min-multiplier", type=float, default=0.35, help="Minimum max-position multiplier when edge-scaled sizing is enabled")
    long_backtest.add_argument("--min-trade-usd", type=float, default=1.0)
    long_backtest.add_argument("--min-edge", type=float, default=0.08)
    long_backtest.add_argument("--uncertainty-buffer", type=float, default=0.03)
    long_backtest.add_argument("--min-model-count", type=int, default=3)
    long_backtest.add_argument("--min-model-agreement", type=float, default=1.0)
    long_backtest.add_argument("--hold-min-model-agreement", type=float, default=0.65)
    long_backtest.add_argument("--hold-min-fair-value", type=float, default=0.60)
    long_backtest.add_argument("--hold-market-confirmation-price", type=float, default=0.80)
    long_backtest.add_argument("--hold-market-confirmation-min-fair-value", type=float, default=0.50)
    long_backtest.add_argument("--trim-valid-holds-to-kelly-target", action="store_true")
    long_backtest.add_argument("--high-confidence-price-threshold", type=float, default=0.75)
    long_backtest.add_argument("--high-confidence-min-kelly-edge", type=float, default=0.02)
    long_backtest.add_argument("--low-price-exact-bucket-threshold", type=float, default=0.20)
    long_backtest.add_argument("--low-price-exact-bucket-min-fair-value", type=float, default=0.22)
    long_backtest.add_argument("--low-price-exact-bucket-min-edge", type=float, default=0.08)
    long_backtest.add_argument("--correlated-exact-bucket-max-price", type=float, default=0.15)
    long_backtest.add_argument("--correlated-exact-bucket-min-agreement", type=float, default=0.95)
    long_backtest.add_argument("--exact-bucket-max-width-f", type=float, default=2.25)
    long_backtest.add_argument("--min-price", type=float, default=0.125)
    long_backtest.add_argument("--yes-side-min-price", type=float, default=0.20)
    long_backtest.add_argument("--no-side-max-price", type=float, default=0.95, help="Maximum entry price for experimental NO-token entries; set at or above --max-price to disable the side-specific cap")
    long_backtest.add_argument("--no-side-max-counter-event-probability", type=float, default=0.10, help="For experimental NO-token entries, reject rows if any model view gives the opposite YES event more than this probability; set >=1 to disable")
    long_backtest.add_argument("--no-side-relaxed-counter-event-probability", type=float, default=None, help="Optional relaxed NO counter-event cap used only during --no-side-relaxed-counter-event-hours-utc")
    long_backtest.add_argument("--no-side-relaxed-counter-event-hours-utc", default="", help="Comma-separated UTC hours that may use --no-side-relaxed-counter-event-probability")
    long_backtest.add_argument("--hold-no-side-max-counter-event-probability", type=float, default=0.15, help="For existing NO-token positions, allow holding while the opposite YES tail is below this probability; set >=1 to disable")
    long_backtest.add_argument("--hold-no-side-high-conviction-min-fair-value", type=float, default=None, help="Optional FV floor for using the wider high-conviction NO hold counter-event cap")
    long_backtest.add_argument("--hold-no-side-high-conviction-min-edge", type=float, default=None, help="Optional edge floor for using the wider high-conviction NO hold counter-event cap")
    long_backtest.add_argument("--hold-no-side-high-conviction-counter-event-probability", type=float, default=None, help="Optional wider NO hold counter-event cap used only for high-conviction existing positions")
    long_backtest.add_argument("--invalid-hold-partial-exit-fraction", type=float, default=None, help="If set, sell only this fraction of an invalid existing hold when the FV/price partial-exit gate passes")
    long_backtest.add_argument("--invalid-hold-partial-exit-min-fair-value", type=float, default=0.90)
    long_backtest.add_argument("--invalid-hold-partial-exit-min-price", type=float, default=0.50)
    long_backtest.add_argument("--invalid-hold-partial-exit-max-price", type=float, default=0.65)
    long_backtest.add_argument("--min-signal-fair-value", type=float, default=0.70)
    long_backtest.add_argument("--allow-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_true", default=True)
    long_backtest.add_argument("--disable-bounded-bucket-entries", dest="allow_bounded_bucket_entries", action="store_false")
    long_backtest.add_argument("--allow-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_true", default=True)
    long_backtest.add_argument("--disable-bounded-no-side-entries", dest="allow_bounded_no_side_entries", action="store_false")
    long_backtest.add_argument("--bounded-bucket-min-edge", type=float, default=0.10)
    long_backtest.add_argument("--bounded-bucket-min-fair-value", type=float, default=0.90)
    long_backtest.add_argument("--bounded-bucket-min-model-agreement", type=float, default=1.0)
    long_backtest.add_argument("--bounded-bucket-min-price", type=float, default=0.50)
    long_backtest.add_argument("--bounded-bucket-max-probability-stdev", type=float, default=None)
    long_backtest.add_argument("--max-price", type=float, default=0.95)
    long_backtest.add_argument(
        "--allow-no-side-entries",
        action="store_true",
        help="Experimentally evaluate buying real NO tokens for binary markets using historical NO CLOB prices",
    )
    long_backtest.add_argument(
        "--no-side-min-edge",
        type=float,
        default=0.10,
        help="Minimum absolute buffered edge required for experimental NO-token entries",
    )
    long_backtest.add_argument(
        "--no-side-high-confidence-min-edge",
        type=float,
        default=0.02,
        help="Minimum absolute buffered edge for experimental NO entries when the NO price is at or above --high-confidence-price-threshold",
    )
    long_backtest.add_argument("--same-day-entry-start-hour", type=int, default=11)
    long_backtest.add_argument("--same-day-entry-cutoff-hour", type=int, default=17)
    long_backtest.add_argument("--min-volume-usd", type=float, default=0.0)
    long_backtest.add_argument("--weights-file", default="work/data/model_weights.json")
    long_backtest.add_argument("--cache-dir", default="work/cache/long_backtest")
    long_backtest.add_argument("--run-log-dir", default="work/logs/long_backtests")
    long_backtest.add_argument("--progress-every", type=int, default=50, help="Emit long-backtest progress every N prepared markets; <=0 disables")
    long_backtest.add_argument("--http-hard-timeout-seconds", type=int, default=30, help="Abort one stalled live API request after N seconds; <=0 disables")
    long_backtest.add_argument("--summary-only", action="store_true", help="Print a compact summary while preserving the full JSON run log")

    replay = subparsers.add_parser(
        "replay-scored-outcomes",
        help="Replay one or more strategies from a saved long-backtest scored_outcomes_detail artifact without refetching data",
    )
    _add_strategy_profile_argument(replay)
    replay.add_argument("--source-run-log", required=True, help="Path to a long-backtest JSON artifact containing scored_outcomes_detail")
    replay.add_argument("--weights-file", default="work/data/model_weights.json", help="Weights/calibration file used when recomputing cached rows from raw model probabilities")
    replay.add_argument("--recompute-from-raw-model-probabilities", action="store_true", help="If raw_model_probabilities are present, recompute FV/edge/agreement from raw probabilities before replaying")
    replay.add_argument(
        "--strategy-profiles",
        default="",
        help="Comma-separated profile names to sweep. Use 'all' for every named profile except manual. Defaults to --strategy-profile.",
    )
    _add_cached_replay_arguments(replay)
    replay.add_argument("--run-log-dir", default="work/logs/scored_replays")
    replay.add_argument("--summary-only", action="store_true", help="Print compact replay summaries while preserving detailed JSON run logs")

    live_like = subparsers.add_parser(
        "build-live-like-backtest",
        help="Build a broad Telonex/Open-Meteo historical artifact using the same timing semantics as the live trader",
    )
    _add_strategy_profile_argument(live_like)
    _add_cached_replay_arguments(live_like)
    live_like.set_defaults(bankroll_usd=50.0, compound_kelly_sizing=True)
    live_like.add_argument("--lookback-days", type=int, default=365)
    live_like.add_argument("--min-end-date", type=_parse_iso_date, default=None)
    live_like.add_argument("--max-end-date", type=_parse_iso_date, default=None)
    live_like.add_argument("--pages", type=int, default=20)
    live_like.add_argument("--limit-per-page", type=int, default=50)
    live_like.add_argument("--max-markets", type=int, default=50000)
    live_like.add_argument("--query", default="highest temperature")
    live_like.add_argument(
        "--market-selection",
        choices=("end_date_asc", "end_date_desc", "month_balanced"),
        default="month_balanced",
    )
    live_like.add_argument(
        "--entry-hours-utc",
        default=",".join(str(hour) for hour in LIVE_FORWARD_ENTRY_HOURS_UTC),
        help="Comma-separated simulated run hours in UTC; defaults to the live automation cadence",
    )
    live_like.add_argument("--max-runtime-seconds", type=float, default=0.0)
    live_like.add_argument("--max-price-staleness-minutes", type=int, default=90)
    live_like.add_argument("--historical-price-slippage", type=float, default=0.01)
    live_like.add_argument("--forecast-availability-lag-hours", type=int, default=6)
    live_like.add_argument("--min-volume-usd", type=float, default=0.0)
    live_like.add_argument("--weights-file", default="work/data/model_weights.json")
    live_like.add_argument("--cache-dir", default="work/cache/live_like_backtest")
    live_like.add_argument("--run-log-dir", default="work/logs/live_like_backtests")
    live_like.add_argument("--progress-every", type=int, default=250)
    live_like.add_argument("--http-hard-timeout-seconds", type=int, default=30)
    live_like.add_argument("--price-source", choices=("telonex", "clob", "auto"), default="telonex")
    live_like.add_argument("--market-source", choices=("telonex", "gamma"), default="telonex")
    live_like.add_argument(
        "--settlement-audit",
        choices=("weather_crosscheck", "polymarket_only"),
        default="polymarket_only",
        help="Default is polymarket_only so year-scale replay builds are not blocked on station-weather QA.",
    )
    live_like.add_argument("--summary-only", action="store_true")

    compare_live_like = subparsers.add_parser(
        "compare-live-like-strategies",
        help="Replay current and candidate strategies from a saved live-like scored-outcome artifact and write comparison artifacts",
    )
    _add_strategy_profile_argument(compare_live_like)
    compare_live_like.add_argument("--source-run-log", required=True)
    compare_live_like.add_argument("--weights-file", default="work/data/model_weights.json")
    compare_live_like.add_argument("--recompute-from-raw-model-probabilities", action="store_true")
    compare_live_like.add_argument(
        "--strategy-profiles",
        default=",".join(DEFAULT_STRATEGY_COMPARISON_PROFILES),
        help="Comma-separated profile names. Defaults to current live plus important candidates.",
    )
    _add_cached_replay_arguments(compare_live_like)
    compare_live_like.set_defaults(bankroll_usd=50.0, compound_kelly_sizing=True)
    compare_live_like.add_argument("--run-log-dir", default="work/logs/live_like_strategy_replays")
    compare_live_like.add_argument("--output-dir", default="work/reports/live_like_strategy_comparison")
    compare_live_like.add_argument("--summary-only", action="store_true")

    model_validate = subparsers.add_parser(
        "model-validate",
        help="Validate weather-only probabilities against market Brier on a saved scored_outcomes_detail artifact",
    )
    model_validate.add_argument("--source-run-log", required=True, help="Path to a long-backtest JSON artifact containing scored_outcomes_detail")
    model_validate.add_argument("--train-fraction", type=float, default=0.70)
    model_validate.add_argument(
        "--probability-key",
        choices=("auto", "raw_model_probabilities", "model_probabilities"),
        default="auto",
        help="Probability map to use. auto prefers raw_model_probabilities when available, then model_probabilities.",
    )
    model_validate.add_argument("--run-log-dir", default="work/logs/model_validation")
    model_validate.add_argument("--max-coordinate-epochs", type=int, default=4)
    model_validate.add_argument("--summary-only", action="store_true")

    args = parser.parse_args(argv)
    if hasattr(args, "strategy_profile") and args.command not in ("replay-scored-outcomes", "compare-live-like-strategies"):
        _apply_strategy_profile(args)
    if args.command == "paper-run":
        return run_paper(args)
    if args.command == "scan-live":
        return run_scan_live(args)
    if args.command == "debug-search":
        return run_debug_search(args)
    if args.command == "live-setup":
        return run_live_setup(args)
    if args.command == "report":
        return run_report(args)
    if args.command == "calibration":
        return run_calibration(args)
    if args.command == "backtest":
        return run_backtest_command(args)
    if args.command == "long-backtest":
        return run_long_backtest_command(args)
    if args.command == "replay-scored-outcomes":
        return run_replay_scored_outcomes_command(args)
    if args.command == "build-live-like-backtest":
        return run_build_live_like_backtest_command(args)
    if args.command == "compare-live-like-strategies":
        return run_compare_live_like_strategies_command(args)
    if args.command == "model-validate":
        return run_model_validate_command(args)
    raise ValueError(args.command)


def run_scan_live(args: argparse.Namespace) -> int:
    markets = PolymarketGammaClient().discover_temperature_markets(limit=args.limit, pages=args.pages)
    payload = [_market_to_json(market) for market in markets]
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"markets_found": len(markets), "markets": payload[:10]}, indent=2, sort_keys=True))
    return 0


def run_live_setup(args: argparse.Namespace) -> int:
    result: dict[str, Any] = {}
    try:
        if args.create_wallet:
            result["wallet"] = create_bot_wallet(args.env_file, overwrite=args.overwrite)
        if args.derive_clob_creds:
            result["clob_credentials"] = derive_clob_credentials(args.env_file, overwrite=args.overwrite)
        if args.check_geoblock:
            result["geoblock"] = check_geoblock()
        if args.clob_readonly_smoke:
            result["clob_readonly_smoke"] = clob_readonly_smoke(args.env_file)
        result["status"] = live_status(args.env_file)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except LiveSetupError as error:
        result["error"] = str(error)
        try:
            result["status"] = live_status(args.env_file)
        except Exception:
            pass
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1


def run_debug_search(args: argparse.Namespace) -> int:
    payload = PolymarketGammaClient().public_search(query=args.query, limit=args.limit, page=args.page)
    rows = []
    for event in payload.get("events") or []:
        rows.append({"type": "event", "title": event.get("title"), "slug": event.get("slug"), "active": event.get("active"), "closed": event.get("closed")})
        for market in event.get("markets") or []:
            rows.append({"type": "market", "question": market.get("question"), "slug": market.get("slug"), "active": market.get("active"), "closed": market.get("closed")})
    print(json.dumps({"query": args.query, "rows": rows[: args.limit * 5], "pagination": payload.get("pagination")}, indent=2, sort_keys=True))
    return 0


def run_report(args: argparse.Namespace) -> int:
    ledger = PaperLedger(args.ledger)
    equity = ledger.equity_usd(args.bankroll_usd)
    positions = ledger.positions()
    print(
        json.dumps(
            {
                "ledger": args.ledger,
                "bankroll_usd": args.bankroll_usd,
                "equity_usd": round(equity, 2),
                "pnl_usd": round(equity - args.bankroll_usd, 2),
                "open_positions": len(positions),
                "positions": positions[:20],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_calibration(args: argparse.Namespace) -> int:
    ledger = PaperLedger(args.ledger)
    print(json.dumps(ledger.calibration_summary(), indent=2, sort_keys=True))
    return 0


def run_backtest_command(args: argparse.Namespace) -> int:
    settings = SignalSettings(
        min_edge=args.min_edge,
        uncertainty_buffer=args.uncertainty_buffer,
        min_model_count=args.min_model_count,
        min_model_agreement=args.min_model_agreement,
        high_confidence_price_threshold=args.high_confidence_price_threshold,
        high_confidence_min_kelly_edge=args.high_confidence_min_kelly_edge,
        low_price_exact_bucket_threshold=args.low_price_exact_bucket_threshold,
        low_price_exact_bucket_min_fair_value=args.low_price_exact_bucket_min_fair_value,
        low_price_exact_bucket_min_edge=args.low_price_exact_bucket_min_edge,
        correlated_exact_bucket_max_price=args.correlated_exact_bucket_max_price,
        correlated_exact_bucket_min_agreement=args.correlated_exact_bucket_min_agreement,
        exact_bucket_max_width_f=args.exact_bucket_max_width_f,
        min_price=args.min_price,
        yes_side_min_price=args.yes_side_min_price,
        allow_no_side_entries=args.allow_no_side_entries,
        no_side_min_edge=args.no_side_min_edge,
        no_side_high_confidence_min_edge=args.no_side_high_confidence_min_edge,
        no_side_max_price=args.no_side_max_price,
        no_side_max_counter_event_probability=args.no_side_max_counter_event_probability,
        no_side_relaxed_counter_event_probability=args.no_side_relaxed_counter_event_probability,
        no_side_relaxed_counter_event_hours_utc=_parse_optional_entry_hours(args.no_side_relaxed_counter_event_hours_utc),
        hold_no_side_max_counter_event_probability=args.hold_no_side_max_counter_event_probability,
        hold_no_side_high_conviction_min_fair_value=args.hold_no_side_high_conviction_min_fair_value,
        hold_no_side_high_conviction_min_edge=args.hold_no_side_high_conviction_min_edge,
        hold_no_side_high_conviction_counter_event_probability=args.hold_no_side_high_conviction_counter_event_probability,
        invalid_hold_partial_exit_fraction=args.invalid_hold_partial_exit_fraction,
        invalid_hold_partial_exit_min_fair_value=args.invalid_hold_partial_exit_min_fair_value,
        invalid_hold_partial_exit_min_price=args.invalid_hold_partial_exit_min_price,
        invalid_hold_partial_exit_max_price=args.invalid_hold_partial_exit_max_price,
        min_signal_fair_value=args.min_signal_fair_value,
        allow_bounded_bucket_entries=args.allow_bounded_bucket_entries,
        allow_bounded_no_side_entries=args.allow_bounded_no_side_entries,
        bounded_bucket_min_edge=args.bounded_bucket_min_edge,
        bounded_bucket_min_fair_value=args.bounded_bucket_min_fair_value,
        bounded_bucket_min_model_agreement=args.bounded_bucket_min_model_agreement,
        bounded_bucket_min_price=args.bounded_bucket_min_price,
        bounded_bucket_max_probability_stdev=args.bounded_bucket_max_probability_stdev,
        max_price=args.max_price,
        hold_min_model_agreement=args.hold_min_model_agreement,
        hold_min_fair_value=args.hold_min_fair_value,
        hold_market_confirmation_price=args.hold_market_confirmation_price,
        hold_market_confirmation_min_fair_value=args.hold_market_confirmation_min_fair_value,
        preserve_valid_holds=not args.trim_valid_holds_to_kelly_target,
        enforce_entry_timing_filter=False,
    )
    result = run_backtest(
        args.ledger,
        bankroll_usd=args.bankroll_usd,
        kelly_fraction=args.kelly_fraction,
        max_position_usd=args.max_position_usd,
        max_position_fraction=args.max_position_fraction,
        kelly_market_blend=args.kelly_market_blend,
        edge_position_full_cap_edge=args.edge_position_full_cap_edge,
        edge_position_min_multiplier=args.edge_position_min_multiplier,
        min_trade_usd=args.min_trade_usd,
        settings=settings,
        train_fraction=args.train_fraction,
        output_weights_path=args.output_weights,
        fetch_observations=not args.no_fetch_observations,
        max_observation_lookups=args.max_observation_lookups,
        min_weight_samples=args.min_weight_samples,
        weight_prior_samples=args.weight_prior_samples,
    )
    result["strategy_profile"] = args.strategy_profile
    if not args.no_run_log:
        run_log_path = _make_run_log_path(args.run_log_dir, "backtest")
        result["run_log_path"] = str(run_log_path)
        _write_json_log(run_log_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _long_backtest_signal_settings_from_args(args: argparse.Namespace) -> SignalSettings:
    return SignalSettings(
        min_edge=args.min_edge,
        uncertainty_buffer=args.uncertainty_buffer,
        min_model_count=args.min_model_count,
        min_model_agreement=args.min_model_agreement,
        high_confidence_price_threshold=args.high_confidence_price_threshold,
        high_confidence_min_kelly_edge=args.high_confidence_min_kelly_edge,
        low_price_exact_bucket_threshold=args.low_price_exact_bucket_threshold,
        low_price_exact_bucket_min_fair_value=args.low_price_exact_bucket_min_fair_value,
        low_price_exact_bucket_min_edge=args.low_price_exact_bucket_min_edge,
        correlated_exact_bucket_max_price=args.correlated_exact_bucket_max_price,
        correlated_exact_bucket_min_agreement=args.correlated_exact_bucket_min_agreement,
        exact_bucket_max_width_f=args.exact_bucket_max_width_f,
        min_price=args.min_price,
        yes_side_min_price=args.yes_side_min_price,
        min_signal_fair_value=args.min_signal_fair_value,
        allow_bounded_bucket_entries=args.allow_bounded_bucket_entries,
        allow_bounded_no_side_entries=args.allow_bounded_no_side_entries,
        bounded_bucket_min_edge=args.bounded_bucket_min_edge,
        bounded_bucket_min_fair_value=args.bounded_bucket_min_fair_value,
        bounded_bucket_min_model_agreement=args.bounded_bucket_min_model_agreement,
        bounded_bucket_min_price=args.bounded_bucket_min_price,
        bounded_bucket_max_probability_stdev=args.bounded_bucket_max_probability_stdev,
        allow_no_side_entries=args.allow_no_side_entries,
        no_side_min_edge=args.no_side_min_edge,
        no_side_high_confidence_min_edge=args.no_side_high_confidence_min_edge,
        no_side_max_price=args.no_side_max_price,
        no_side_max_counter_event_probability=args.no_side_max_counter_event_probability,
        no_side_relaxed_counter_event_probability=args.no_side_relaxed_counter_event_probability,
        no_side_relaxed_counter_event_hours_utc=_parse_optional_entry_hours(args.no_side_relaxed_counter_event_hours_utc),
        hold_no_side_max_counter_event_probability=args.hold_no_side_max_counter_event_probability,
        hold_no_side_high_conviction_min_fair_value=args.hold_no_side_high_conviction_min_fair_value,
        hold_no_side_high_conviction_min_edge=args.hold_no_side_high_conviction_min_edge,
        hold_no_side_high_conviction_counter_event_probability=args.hold_no_side_high_conviction_counter_event_probability,
        invalid_hold_partial_exit_fraction=args.invalid_hold_partial_exit_fraction,
        invalid_hold_partial_exit_min_fair_value=args.invalid_hold_partial_exit_min_fair_value,
        invalid_hold_partial_exit_min_price=args.invalid_hold_partial_exit_min_price,
        invalid_hold_partial_exit_max_price=args.invalid_hold_partial_exit_max_price,
        max_price=args.max_price,
        hold_min_model_agreement=args.hold_min_model_agreement,
        hold_min_fair_value=args.hold_min_fair_value,
        hold_market_confirmation_price=args.hold_market_confirmation_price,
        hold_market_confirmation_min_fair_value=args.hold_market_confirmation_min_fair_value,
        preserve_valid_holds=not args.trim_valid_holds_to_kelly_target,
        same_day_earliest_entry_hour_local=args.same_day_entry_start_hour,
        same_day_latest_entry_hour_local=args.same_day_entry_cutoff_hour,
        enforce_entry_timing_filter=True,
    )


def run_long_backtest_command(args: argparse.Namespace) -> int:
    settings = _long_backtest_signal_settings_from_args(args)
    result = run_long_historical_backtest(
        bankroll_usd=args.bankroll_usd,
        pages=args.pages,
        limit_per_page=args.limit_per_page,
        max_markets=args.max_markets,
        query=args.query,
        min_end_date=args.min_end_date,
        max_end_date=args.max_end_date,
        market_selection=args.market_selection,
        entry_hours_utc=_parse_entry_hours(args.entry_hours_utc),
        min_lead_days=args.min_lead_days,
        max_lead_days=args.max_lead_days,
        max_runtime_seconds=args.max_runtime_seconds,
        max_price_staleness_minutes=args.max_price_staleness_minutes,
        historical_price_slippage=args.historical_price_slippage,
        forecast_availability_lag_hours=args.forecast_availability_lag_hours,
        kelly_fraction=args.kelly_fraction,
        compound_kelly_sizing=args.compound_kelly_sizing,
        max_new_exposure_usd_per_run=args.max_new_exposure_usd_per_run,
        max_new_exposure_fraction_per_run=args.max_new_exposure_fraction_per_run,
        new_exposure_target_positions_per_run=args.new_exposure_target_positions_per_run,
        kelly_sizing_bankroll_fraction_per_run=args.kelly_sizing_bankroll_fraction_per_run,
        max_position_usd=args.max_position_usd,
        max_position_fraction=args.max_position_fraction,
        kelly_market_blend=args.kelly_market_blend,
        edge_position_full_cap_edge=args.edge_position_full_cap_edge,
        edge_position_min_multiplier=args.edge_position_min_multiplier,
        min_trade_usd=args.min_trade_usd,
        settings=settings,
        min_volume_usd=args.min_volume_usd,
        weights_file=args.weights_file,
        cache_dir=args.cache_dir,
        run_log_dir=args.run_log_dir,
        progress_every=args.progress_every,
        http_hard_timeout_seconds=args.http_hard_timeout_seconds,
        price_source=args.price_source,
        market_source=args.market_source,
        settlement_audit=args.settlement_audit,
        strategy_profile=args.strategy_profile,
    )
    print(json.dumps(_long_backtest_summary(result) if args.summary_only else result, indent=2, sort_keys=True))
    return 0


def run_build_live_like_backtest_command(args: argparse.Namespace) -> int:
    min_end_date, max_end_date = live_like_backtest_dates(
        lookback_days=args.lookback_days,
        min_end_date=args.min_end_date,
        max_end_date=args.max_end_date,
    )
    args.min_end_date = min_end_date
    args.max_end_date = max_end_date
    result = run_long_historical_backtest(
        bankroll_usd=args.bankroll_usd,
        pages=args.pages,
        limit_per_page=args.limit_per_page,
        max_markets=args.max_markets,
        query=args.query,
        min_end_date=min_end_date,
        max_end_date=max_end_date,
        market_selection=args.market_selection,
        entry_hours_utc=_parse_entry_hours(args.entry_hours_utc),
        min_lead_days=args.min_lead_days,
        max_lead_days=args.max_lead_days,
        max_runtime_seconds=args.max_runtime_seconds,
        max_price_staleness_minutes=args.max_price_staleness_minutes,
        historical_price_slippage=args.historical_price_slippage,
        forecast_availability_lag_hours=args.forecast_availability_lag_hours,
        kelly_fraction=args.kelly_fraction,
        compound_kelly_sizing=args.compound_kelly_sizing,
        max_new_exposure_usd_per_run=args.max_new_exposure_usd_per_run,
        max_new_exposure_fraction_per_run=args.max_new_exposure_fraction_per_run,
        new_exposure_target_positions_per_run=args.new_exposure_target_positions_per_run,
        kelly_sizing_bankroll_fraction_per_run=args.kelly_sizing_bankroll_fraction_per_run,
        max_position_usd=args.max_position_usd,
        max_position_fraction=args.max_position_fraction,
        kelly_market_blend=args.kelly_market_blend,
        edge_position_full_cap_edge=args.edge_position_full_cap_edge,
        edge_position_min_multiplier=args.edge_position_min_multiplier,
        min_trade_usd=args.min_trade_usd,
        settings=_long_backtest_signal_settings_from_args(args),
        min_volume_usd=args.min_volume_usd,
        weights_file=args.weights_file,
        cache_dir=args.cache_dir,
        run_log_dir=args.run_log_dir,
        progress_every=args.progress_every,
        http_hard_timeout_seconds=args.http_hard_timeout_seconds,
        price_source=args.price_source,
        market_source=args.market_source,
        settlement_audit=args.settlement_audit,
        strategy_profile=args.strategy_profile,
    )
    print(json.dumps(_long_backtest_summary(result) if args.summary_only else result, indent=2, sort_keys=True))
    return 0


def run_replay_scored_outcomes_command(args: argparse.Namespace) -> int:
    profiles = _replay_strategy_profiles(args.strategy_profiles, args.strategy_profile)
    results = []
    for profile in profiles:
        profile_args = argparse.Namespace(**vars(args))
        profile_args.strategy_profile = profile
        _apply_strategy_profile(profile_args)
        settings = _long_backtest_signal_settings_from_args(profile_args)
        results.append(
            run_cached_scored_outcome_replay(
                snapshot_path=profile_args.source_run_log,
                settings=settings,
                strategy_profile=profile,
                bankroll_usd=profile_args.bankroll_usd,
                kelly_fraction=profile_args.kelly_fraction,
                compound_kelly_sizing=profile_args.compound_kelly_sizing,
                max_new_exposure_usd_per_run=profile_args.max_new_exposure_usd_per_run,
                max_new_exposure_fraction_per_run=profile_args.max_new_exposure_fraction_per_run,
                new_exposure_target_positions_per_run=profile_args.new_exposure_target_positions_per_run,
                kelly_sizing_bankroll_fraction_per_run=profile_args.kelly_sizing_bankroll_fraction_per_run,
                max_position_usd=profile_args.max_position_usd,
                max_position_fraction=profile_args.max_position_fraction,
                kelly_market_blend=profile_args.kelly_market_blend,
                edge_position_full_cap_edge=profile_args.edge_position_full_cap_edge,
                edge_position_min_multiplier=profile_args.edge_position_min_multiplier,
                min_trade_usd=profile_args.min_trade_usd,
                run_log_dir=profile_args.run_log_dir,
                recompute_from_raw_model_probabilities=profile_args.recompute_from_raw_model_probabilities,
                weights_file=profile_args.weights_file,
            )
        )
    if len(results) == 1 and not args.summary_only:
        print(json.dumps(results[0], indent=2, sort_keys=True))
        return 0
    summaries = [_scored_replay_summary(result) for result in results]
    ranked = sorted(
        summaries,
        key=lambda item: (
            float(item.get("pnl_usd") or 0.0),
            float(item.get("event_hit_rate") or 0.0),
            -float(item.get("max_drawdown_usd") or 0.0),
        ),
        reverse=True,
    )
    output = {
        "source_run_log": args.source_run_log,
        "replay_count": len(results),
        "profiles": summaries,
        "ranked_profiles": [_scored_replay_rank_row(row) for row in ranked],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_compare_live_like_strategies_command(args: argparse.Namespace) -> int:
    profiles = _replay_strategy_profiles(args.strategy_profiles, args.strategy_profile)
    results = []
    for profile in profiles:
        profile_args = argparse.Namespace(**vars(args))
        profile_args.strategy_profile = profile
        _apply_strategy_profile(profile_args)
        settings = _long_backtest_signal_settings_from_args(profile_args)
        results.append(
            run_cached_scored_outcome_replay(
                snapshot_path=profile_args.source_run_log,
                settings=settings,
                strategy_profile=profile,
                bankroll_usd=profile_args.bankroll_usd,
                kelly_fraction=profile_args.kelly_fraction,
                compound_kelly_sizing=profile_args.compound_kelly_sizing,
                max_new_exposure_usd_per_run=profile_args.max_new_exposure_usd_per_run,
                max_new_exposure_fraction_per_run=profile_args.max_new_exposure_fraction_per_run,
                new_exposure_target_positions_per_run=profile_args.new_exposure_target_positions_per_run,
                kelly_sizing_bankroll_fraction_per_run=profile_args.kelly_sizing_bankroll_fraction_per_run,
                max_position_usd=profile_args.max_position_usd,
                max_position_fraction=profile_args.max_position_fraction,
                kelly_market_blend=profile_args.kelly_market_blend,
                edge_position_full_cap_edge=profile_args.edge_position_full_cap_edge,
                edge_position_min_multiplier=profile_args.edge_position_min_multiplier,
                min_trade_usd=profile_args.min_trade_usd,
                run_log_dir=profile_args.run_log_dir,
                recompute_from_raw_model_probabilities=profile_args.recompute_from_raw_model_probabilities,
                weights_file=profile_args.weights_file,
            )
        )
    report = compare_strategy_replays(
        results,
        output_dir=args.output_dir,
        current_live_strategy_profile=CURRENT_LIVE_STRATEGY_PROFILE,
        source_run_log=args.source_run_log,
    )
    print(json.dumps(report["summary"] if args.summary_only else report, indent=2, sort_keys=True))
    return 0


def run_model_validate_command(args: argparse.Namespace) -> int:
    result = run_weather_model_validation(
        source_run_log=args.source_run_log,
        train_fraction=args.train_fraction,
        probability_key=args.probability_key,
        run_log_dir=args.run_log_dir,
        max_coordinate_epochs=args.max_coordinate_epochs,
    )
    print(json.dumps(_model_validation_summary(result) if args.summary_only else result, indent=2, sort_keys=True))
    return 0


def _replay_strategy_profiles(value: str, fallback: str) -> list[str]:
    raw_profiles = [item.strip() for item in value.split(",") if item.strip()] if value else [fallback]
    if any(item.lower() == "all" for item in raw_profiles):
        return [profile for profile in STRATEGY_PROFILE_CHOICES if profile != "manual"]
    invalid = [profile for profile in raw_profiles if profile not in STRATEGY_PROFILE_CHOICES]
    if invalid:
        raise ValueError(f"Unknown strategy profile(s): {', '.join(invalid)}")
    profiles: list[str] = []
    for profile in raw_profiles:
        if profile not in profiles:
            profiles.append(profile)
    return profiles


def _parse_entry_hours(value: str) -> tuple[int, ...]:
    hours = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        hour = int(stripped)
        if hour < 0 or hour > 23:
            raise ValueError(f"Entry hour must be in [0, 23], got {hour}")
        hours.append(hour)
    if not hours:
        raise ValueError("At least one entry hour is required")
    return tuple(sorted(set(hours)))


def _parse_optional_entry_hours(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return _parse_entry_hours(value)


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from error


def _long_backtest_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "strategy_profile",
        "bankroll_usd",
        "ending_equity_usd",
        "pnl_usd",
        "return_pct",
        "cash_usd",
        "open_positions",
        "markets_discovered_raw",
        "markets_parsed",
        "price_history_requests",
        "markets_with_price_history",
        "session_count",
        "markets_scored",
        "outcomes_scored",
        "signals",
        "executions",
        "buys",
        "sells",
        "settlements",
        "forecast_error_count",
        "forecast_error_examples",
        "price_error_count",
        "no_side_price_error_count",
        "no_side_price_history_count",
        "no_side_session_count",
        "resolution_error_count",
        "weather_crosscheck_mismatches",
        "weather_crosscheck_ambiguous_count",
        "weather_crosscheck_checked_executions",
        "weather_crosscheck_mismatch_executions",
        "realized_pnl_crosscheck_mismatch_usd",
        "realized_pnl_crosscheck_matched_or_unchecked_usd",
        "runtime_limited",
        "elapsed_seconds",
        "scored_outcomes_detail_count",
        "skipped_reason_counts",
        "run_log_path",
        "run_log_write_error",
    )
    return {
        **{key: result.get(key) for key in keys},
        "data_provenance": result.get("data_provenance"),
        "settings": result.get("settings"),
        "trade_diagnostics": result.get("trade_diagnostics"),
        "score_calibration_diagnostics": result.get("score_calibration_diagnostics"),
        "signal_filter_diagnostics": result.get("signal_filter_diagnostics"),
        "signal_opportunity_diagnostics": result.get("signal_opportunity_diagnostics"),
        "strategy_sensitivity_diagnostics": _compact_dict(
            result.get("strategy_sensitivity_diagnostics"),
            ("method", "baseline", "recommended_profile", "selected_profile"),
        ),
        "robustness_diagnostics": _compact_dict(
            result.get("robustness_diagnostics"),
            ("method", "summary", "passed", "warnings"),
        ),
        "strategy_recommendation_diagnostics": _compact_dict(
            result.get("strategy_recommendation_diagnostics"),
            ("method", "recommended_profile", "recommendation", "warnings"),
        ),
        "real_data_audit": result.get("real_data_audit"),
        "top_trades": _compact_replay_trade_rows(result.get("top_trades", [])[:10]),
        "top_scored_outcomes": _compact_scored_rows(result.get("scored_outcomes_detail", [])[:10]),
        "skipped_examples": result.get("skipped_examples", [])[:10],
    }


def _scored_replay_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "strategy_profile",
        "variant",
        "bankroll_usd",
        "ending_equity_usd",
        "pnl_usd",
        "return_pct",
        "signals",
        "trade_count",
        "executions",
        "buys",
        "sells",
        "settlements",
        "hit_rate",
        "event_hit_rate",
        "event_winning_trades",
        "event_losing_trades",
        "buy_notional_usd",
        "return_on_buy_notional",
        "min_cash_usd",
        "min_cash_pct",
        "max_drawdown_usd",
        "max_drawdown_pct",
        "weather_mismatch_trades",
        "weather_ambiguous_trades",
        "source_scored_outcomes_detail_count",
        "source_strategy_profile",
        "source_session_count",
        "source_markets_scored",
        "source_outcomes_scored",
        "elapsed_seconds",
        "run_log_path",
        "run_log_write_error",
    )
    audit = result.get("real_data_audit") if isinstance(result.get("real_data_audit"), dict) else {}
    return {
        **{key: result.get(key) for key in keys},
        "source_run_log_path": result.get("source_run_log_path"),
        "source_settings": _compact_source_replay_settings(result.get("source_settings")),
        "cache_replay": result.get("cache_replay"),
        "raw_recalibration": _compact_dict(
            result.get("raw_recalibration"),
            ("enabled", "rows_with_raw_model_probabilities", "rows_recomputed", "weights_file"),
        ),
        "settings": _compact_replay_settings(result.get("settings")),
        "performance_diagnostics": _compact_dict(
            result.get("performance_diagnostics"),
            (
                "period_start",
                "period_end",
                "period_days",
                "total_return_pct",
                "average_monthly_return_pct",
                "annualized_from_average_monthly_pct",
                "calendar_daily_sharpe_365",
                "max_drawdown_usd",
                "max_drawdown_pct",
            ),
        ),
        "real_data_audit": {
            "passed": audit.get("passed"),
            "source_real_data_audit_passed": audit.get("source_real_data_audit_passed"),
            "replay_weather_mismatch_trades": audit.get("replay_weather_mismatch_trades"),
            "replay_weather_ambiguous_trades": audit.get("replay_weather_ambiguous_trades"),
        },
        "selected_candidate_weather_validation": _compact_dict(
            result.get("selected_candidate_weather_validation"),
            ("selected_candidate_count", "weather_mismatch_count", "weather_ambiguous_count", "passed"),
        ),
        "trade_diagnostics": _compact_dict(
            result.get("trade_diagnostics"),
            ("trade_count", "realized_trade_count", "winning_trades", "hit_rate", "event_hit_rate", "pnl_concentration"),
        ),
        "score_calibration_diagnostics": _compact_calibration(result.get("score_calibration_diagnostics")),
        "top_trades": _compact_replay_trade_rows(result.get("top_trades", [])[:3]),
        "top_scored_outcomes": _compact_scored_rows(result.get("top_scored_outcomes", [])[:3]),
    }


def _model_validation_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}

    def brier_row(name: str) -> dict[str, Any]:
        predictor_metrics = metrics.get(name) if isinstance(metrics.get(name), dict) else {}
        return {
            "train_all": _compact_dict(predictor_metrics.get("train_all"), ("n", "avg_prediction", "actual_rate", "brier", "log_loss")),
            "test_all": _compact_dict(predictor_metrics.get("test_all"), ("n", "avg_prediction", "actual_rate", "brier", "log_loss")),
            "test_signal": _compact_dict(predictor_metrics.get("test_signal"), ("n", "avg_prediction", "actual_rate", "brier", "log_loss")),
            "test_high_recorded_fv": _compact_dict(
                predictor_metrics.get("test_high_recorded_fv"),
                ("n", "avg_prediction", "actual_rate", "brier", "log_loss"),
            ),
        }

    fitted = result.get("fitted_weather_only_weights") if isinstance(result.get("fitted_weather_only_weights"), dict) else {}
    return {
        "source_run_log": result.get("source_run_log"),
        "run_log_path": result.get("run_log_path"),
        "rows": result.get("rows"),
        "train_rows": result.get("train_rows"),
        "test_rows": result.get("test_rows"),
        "range": result.get("range"),
        "probability_key": result.get("probability_key"),
        "selected_weather_only_candidate": result.get("selected_weather_only_candidate"),
        "best_weather_by_slice": result.get("best_weather_by_slice"),
        "beats_market_brier": result.get("beats_market_brier"),
        "recorded_fair_value": brier_row("recorded_fair_value"),
        "weighted_weather_ensemble": brier_row("weighted_weather_ensemble"),
        "weighted_tail_calibrated_weather_ensemble": brier_row("weighted_tail_calibrated_weather_ensemble"),
        "market_price": brier_row("market_price"),
        "fitted_weather_only_weights": {
            "source_weights": fitted.get("source_weights"),
            "model_family_weights": fitted.get("model_family_weights"),
            "tail_calibration": fitted.get("tail_calibration"),
        },
        "interpretation": result.get("interpretation"),
    }


def _scored_replay_rank_row(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "strategy_profile",
        "ending_equity_usd",
        "pnl_usd",
        "return_pct",
        "trade_count",
        "event_hit_rate",
        "hit_rate",
        "max_drawdown_usd",
        "min_cash_pct",
        "buy_notional_usd",
        "return_on_buy_notional",
        "weather_mismatch_trades",
        "weather_ambiguous_trades",
        "run_log_path",
    )
    return {key: result.get(key) for key in keys}


def _compact_source_replay_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "entry_hours_utc",
            "entry_hours_match_live_forward",
            "min_lead_days",
            "max_lead_days",
            "price_source",
            "market_source",
        )
        if key in value
    }


def _compact_replay_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "compound_kelly_sizing",
        "max_new_exposure_usd_per_run",
        "max_new_exposure_fraction_per_run",
        "new_exposure_target_positions_per_run",
        "kelly_sizing_bankroll_fraction_per_run",
        "max_position_usd",
        "max_position_fraction",
        "kelly_market_blend",
        "edge_position_full_cap_edge",
        "edge_position_min_multiplier",
        "min_price",
        "yes_side_min_price",
        "min_signal_fair_value",
        "min_model_agreement",
        "allow_no_side_entries",
        "no_side_min_edge",
        "no_side_high_confidence_min_edge",
        "no_side_max_price",
        "no_side_max_counter_event_probability",
        "bounded_bucket_min_edge",
        "bounded_bucket_min_fair_value",
        "bounded_bucket_min_price",
        "bounded_bucket_max_probability_stdev",
        "invalid_hold_partial_exit_fraction",
    )
    return {key: value.get(key) for key in keys if key in value}


def _compact_dict(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value.get(key) for key in keys if key in value}


def _compact_calibration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "resolved_count": value.get("resolved_count"),
        "signal_eligible_count": value.get("signal_eligible_count"),
        "overall": _compact_dict(
            value.get("overall"),
            ("n", "avg_market_price", "avg_fair_value", "actual_rate", "brier_fair_value", "brier_market"),
        ),
        "signal_eligible": _compact_dict(
            value.get("signal_eligible"),
            ("n", "avg_market_price", "avg_fair_value", "actual_rate", "brier_fair_value", "brier_market"),
        ),
    }


def _compact_replay_trade_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    keys = (
        "token_id",
        "question",
        "city",
        "target_date",
        "side",
        "price",
        "notional_usd",
        "realized_pnl_usd",
        "fair_value",
        "edge",
        "polymarket_payout",
        "weather_outcome",
    )
    return [{key: row.get(key) for key in keys if isinstance(row, dict) and key in row} for row in rows]


def _compact_scored_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    keys = (
        "generated_at",
        "token_id",
        "question",
        "city",
        "target_date",
        "side",
        "market_price",
        "fair_value",
        "edge",
        "model_agreement",
        "signal_filter_reason",
        "polymarket_payout",
        "weather_outcome",
    )
    return [{key: row.get(key) for key in keys if isinstance(row, dict) and key in row} for row in rows]


def run_paper(args: argparse.Namespace) -> int:
    started = time.monotonic()
    run_started_at = datetime.now(timezone.utc)
    deadline = None if args.max_runtime_seconds <= 0 else started + args.max_runtime_seconds
    fixture_mode = bool(args.fixture)
    execution_mode = getattr(args, "execution_mode", "paper")
    if execution_mode == "live":
        if fixture_mode:
            raise RuntimeError("Live execution cannot run from fixtures")
        if not getattr(args, "confirm_live", False):
            raise RuntimeError("--confirm-live is required for --execution-mode live")
    ledger = PaperLedger(args.ledger)
    live_executor = None
    live_collateral_start = None
    cash_reserve_fraction = max(0.0, min(1.0, float(getattr(args, "cash_reserve_fraction", 0.0) or 0.0)))
    cash_reserve_usd = round(float(args.bankroll_usd) * cash_reserve_fraction, 2)
    if execution_mode == "live":
        live_executor = PolymarketLiveExecutionAdapter(
            getattr(args, "live_env_file", ".env.local"),
            min_collateral_reserve_usd=cash_reserve_usd,
        )
        live_collateral_start = live_executor.collateral_balance_usd()
    weights_file = args.weights_file if not fixture_mode else None
    source_weights, model_weights = load_calibration_weights(weights_file)
    probability_calibration = load_probability_calibration(weights_file)
    run_log_prefix = "live-run" if execution_mode == "live" else "paper-run"
    run_log_path = None if args.no_run_log else _make_run_log_path(args.run_log_dir, run_log_prefix)
    forecast_engine = ConsensusForecastEngine(
        source_weights=source_weights,
        model_weights=model_weights,
        probability_calibration=probability_calibration,
    )
    weather_client = OpenMeteoClient()
    observation_client = ObservedHighClient()
    clob_client = PolymarketClobClient()
    settlement_count = 0
    settlement_error_count = 0
    if not fixture_mode and not args.disable_observations:
        settlement_count, settlement_error_count = ledger.settle_expired_positions(observation_client, now=datetime.now(timezone.utc))
        _progress(f"settled_expired_positions={settlement_count} settlement_errors={settlement_error_count}")

    raw_markets = (
        load_fixture(args.fixture)
        if fixture_mode
        else PolymarketGammaClient().discover_temperature_markets(
            limit=args.limit,
            pages=args.discovery_pages,
            request_limit=args.discovery_request_limit,
        )
    )
    markets = list(_parse_markets(raw_markets, already_parsed=not fixture_mode))
    _progress(f"markets_discovered={len(markets)}")
    open_token_ids = {str(position.get("token_id")) for position in ledger.positions() if position.get("token_id")}
    signal_settings = SignalSettings(
        min_edge=args.min_edge,
        uncertainty_buffer=args.uncertainty_buffer,
        max_spread=args.max_spread,
        default_size_usd=args.size_usd,
        min_model_count=args.min_model_count,
        min_model_agreement=args.min_model_agreement,
        high_confidence_price_threshold=args.high_confidence_price_threshold,
        high_confidence_min_kelly_edge=args.high_confidence_min_kelly_edge,
        low_price_exact_bucket_threshold=args.low_price_exact_bucket_threshold,
        low_price_exact_bucket_min_fair_value=args.low_price_exact_bucket_min_fair_value,
        low_price_exact_bucket_min_edge=args.low_price_exact_bucket_min_edge,
        correlated_exact_bucket_max_price=args.correlated_exact_bucket_max_price,
        correlated_exact_bucket_min_agreement=args.correlated_exact_bucket_min_agreement,
        exact_bucket_max_width_f=args.exact_bucket_max_width_f,
        min_price=args.min_price,
        yes_side_min_price=args.yes_side_min_price,
        allow_no_side_entries=args.allow_no_side_entries,
        no_side_min_edge=args.no_side_min_edge,
        no_side_high_confidence_min_edge=args.no_side_high_confidence_min_edge,
        no_side_max_price=args.no_side_max_price,
        no_side_max_counter_event_probability=args.no_side_max_counter_event_probability,
        no_side_relaxed_counter_event_probability=args.no_side_relaxed_counter_event_probability,
        no_side_relaxed_counter_event_hours_utc=_parse_optional_entry_hours(args.no_side_relaxed_counter_event_hours_utc),
        hold_no_side_max_counter_event_probability=args.hold_no_side_max_counter_event_probability,
        hold_no_side_high_conviction_min_fair_value=args.hold_no_side_high_conviction_min_fair_value,
        hold_no_side_high_conviction_min_edge=args.hold_no_side_high_conviction_min_edge,
        hold_no_side_high_conviction_counter_event_probability=args.hold_no_side_high_conviction_counter_event_probability,
        invalid_hold_partial_exit_fraction=args.invalid_hold_partial_exit_fraction,
        invalid_hold_partial_exit_min_fair_value=args.invalid_hold_partial_exit_min_fair_value,
        invalid_hold_partial_exit_min_price=args.invalid_hold_partial_exit_min_price,
        invalid_hold_partial_exit_max_price=args.invalid_hold_partial_exit_max_price,
        min_signal_fair_value=args.min_signal_fair_value,
        allow_bounded_bucket_entries=args.allow_bounded_bucket_entries,
        allow_bounded_no_side_entries=args.allow_bounded_no_side_entries,
        bounded_bucket_min_edge=args.bounded_bucket_min_edge,
        bounded_bucket_min_fair_value=args.bounded_bucket_min_fair_value,
        bounded_bucket_min_model_agreement=args.bounded_bucket_min_model_agreement,
        bounded_bucket_min_price=args.bounded_bucket_min_price,
        bounded_bucket_max_probability_stdev=args.bounded_bucket_max_probability_stdev,
        max_price=args.max_price,
        hold_min_model_agreement=args.hold_min_model_agreement,
        hold_min_fair_value=args.hold_min_fair_value,
        hold_market_confirmation_price=args.hold_market_confirmation_price,
        hold_market_confirmation_min_fair_value=args.hold_market_confirmation_min_fair_value,
        preserve_valid_holds=not args.trim_valid_holds_to_kelly_target,
        same_day_earliest_entry_hour_local=args.same_day_entry_start_hour,
        same_day_latest_entry_hour_local=args.same_day_entry_cutoff_hour,
        enforce_entry_timing_filter=not args.allow_late_same_day and not fixture_mode,
    )

    all_scored = []
    processed_markets = 0
    skipped_markets: list[dict[str, str]] = []
    quote_error_count = 0
    weather_error_count = 0
    observation_error_count = 0
    runtime_limited = False
    distribution_cache: dict[tuple[str, str], tuple[ForecastDistribution, ...]] = {}
    observation_cache: dict[tuple[str, str], Optional[ObservedHigh]] = {}
    markets_to_process = []
    eligible_markets_selected = 0
    for market, raw in markets:
        if not fixture_mode:
            process_market, skip_reason = _should_process_market(market, signal_settings, args.max_lead_days, open_token_ids, min_lead_days=args.min_lead_days)
            if not process_market:
                skipped_markets.append({"question": market.question, "reason": skip_reason or "filtered"})
                continue
            if _market_has_open_position(market, open_token_ids):
                markets_to_process.append((market, raw))
                continue
            if eligible_markets_selected >= args.limit:
                skipped_markets.append({"question": market.question, "reason": f"processing limit {args.limit} reached"})
                continue
            eligible_markets_selected += 1
        markets_to_process.append((market, raw))

    for market, raw in markets_to_process:
        if _deadline_exceeded(deadline):
            runtime_limited = True
            skipped_markets.append({"question": market.question, "reason": "max runtime reached"})
            break
        cache_key = (
            market.city.display_name if market.city else market.slug,
            market.target_date.isoformat() if market.target_date else market.slug,
        )
        if cache_key not in distribution_cache:
            try:
                distribution_cache[cache_key] = _distributions_for_market(market, raw, weather_client)
            except (RuntimeError, ValueError) as error:
                weather_error_count += 1
                skipped_markets.append({"question": market.question, "reason": f"weather: {str(error)[:300]}"})
                continue
        distributions = distribution_cache[cache_key]
        consensus = forecast_engine.consensus_by_bucket(distributions, market.buckets)
        if not args.disable_observations:
            if cache_key not in observation_cache:
                try:
                    observation_cache[cache_key] = _observed_high_for_market(market, raw, observation_client)
                except (RuntimeError, ValueError) as error:
                    observation_error_count += 1
                    observation_cache[cache_key] = None
                    skipped_markets.append({"question": market.question, "reason": f"observation: {str(error)[:300]}"})
            consensus = forecast_engine.apply_observed_high(consensus, market.buckets, observation_cache[cache_key], now=datetime.now(timezone.utc))
        quotes = {}
        if not fixture_mode:
            for bucket in market.buckets:
                if bucket.token_id:
                    try:
                        quotes[bucket.token_id] = clob_client.fetch_order_book_quote(bucket.token_id)
                    except RuntimeError as error:
                        quote_error_count += 1
                        skipped_markets.append({"question": market.question, "reason": f"quote: {str(error)[:300]}"})
        scored = score_outcomes(market, consensus, quotes_by_token=quotes, settings=signal_settings)
        if signal_settings.allow_no_side_entries:
            no_scored, no_quote_errors = _no_side_scored_outcomes(
                market,
                raw,
                scored,
                clob_client=clob_client,
                settings=signal_settings,
                fixture_mode=fixture_mode,
            )
            quote_error_count += no_quote_errors
            scored.extend(no_scored)
        all_scored.extend(scored)
        processed_markets += 1
        if args.progress_every > 0 and processed_markets % args.progress_every == 0:
            _progress(f"markets_scored={processed_markets} outcomes_scored={len(all_scored)} elapsed_seconds={time.monotonic() - started:.1f}")
    all_signals: list[TradeSignal] = signals_from_scored_outcomes(all_scored, settings=signal_settings)
    scored_detail = [_scored_to_json(outcome, signal_settings) for outcome in sorted(all_scored, key=lambda item: item.edge, reverse=True)]
    signal_filter_counts = _signal_filter_counts(all_scored, signal_settings)
    coverage_diagnostics = _paper_coverage_diagnostics(all_scored, markets_to_process, skipped_markets)

    score_rows = ledger.record_forecast_scores(all_scored)
    gross_sizing_bankroll_usd = round(ledger.equity_usd(args.bankroll_usd), 2) if args.compound_kelly_sizing else args.bankroll_usd
    sizing_bankroll_usd = max(0.0, round(gross_sizing_bankroll_usd - cash_reserve_usd, 2))
    executions = ledger.rebalance_kelly(
        all_scored,
        bankroll_usd=sizing_bankroll_usd,
        kelly_fraction=args.kelly_fraction,
        max_position_usd=args.max_position_usd,
        max_position_fraction=args.max_position_fraction,
        max_new_exposure_usd_per_run=args.max_new_exposure_usd_per_run,
        max_new_exposure_fraction_per_run=args.max_new_exposure_fraction_per_run,
        new_exposure_target_positions_per_run=args.new_exposure_target_positions_per_run,
        kelly_sizing_bankroll_fraction_per_run=args.kelly_sizing_bankroll_fraction_per_run,
        kelly_market_blend=args.kelly_market_blend,
        edge_position_full_cap_edge=args.edge_position_full_cap_edge,
        edge_position_min_multiplier=args.edge_position_min_multiplier,
        min_trade_usd=args.min_trade_usd,
        min_edge=args.min_edge,
        min_model_count=args.min_model_count,
        min_model_agreement=args.min_model_agreement,
        high_confidence_price_threshold=args.high_confidence_price_threshold,
        high_confidence_min_kelly_edge=args.high_confidence_min_kelly_edge,
        min_price=signal_settings.min_price,
        max_price=signal_settings.max_price,
        settings=signal_settings,
        execution_callback=_live_execution_callback(live_executor) if live_executor is not None else None,
    )
    max_new_exposure_budget_usd = _new_exposure_budget_usd(
        sizing_bankroll_usd,
        args.max_new_exposure_usd_per_run,
        args.max_new_exposure_fraction_per_run,
    )
    max_new_exposure_per_position_budget_usd = _new_exposure_per_position_budget_usd(
        max_new_exposure_budget_usd,
        args.new_exposure_target_positions_per_run,
    )
    kelly_sizing_bankroll_usd = _kelly_sizing_bankroll_usd(
        sizing_bankroll_usd,
        args.kelly_sizing_bankroll_fraction_per_run,
    )
    live_collateral_end = _live_collateral_balance_after_executions(
        live_executor,
        executions=executions,
        starting_balance=live_collateral_start,
    )
    rows = ledger.record_signals(
        all_signals,
        metadata={
            "fixture_mode": fixture_mode,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "strategy_profile": args.strategy_profile,
            "market_count": len(markets),
            "processed_market_count": processed_markets,
            "skipped_market_count": len(skipped_markets),
            "model_consensus": True,
            "weights_file": args.weights_file if source_weights or model_weights or probability_calibration.active else None,
        },
    )
    equity = ledger.record_run(
        bankroll_usd=args.bankroll_usd,
        markets_scored=processed_markets,
        outcomes_scored=len(all_scored),
        signals=len(all_signals),
        executions=executions,
        metadata={
            "fixture_mode": fixture_mode,
            "markets_discovered": len(markets),
            "markets_skipped": len(skipped_markets),
            "weather_error_count": weather_error_count,
            "observation_error_count": observation_error_count,
            "quote_error_count": quote_error_count,
            "settled_expired_positions": settlement_count,
            "settlement_error_count": settlement_error_count,
            "runtime_limited": runtime_limited,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "weights_file": args.weights_file if source_weights or model_weights or probability_calibration.active else None,
            "strategy_profile": args.strategy_profile,
            "compound_kelly_sizing": args.compound_kelly_sizing,
            "gross_sizing_bankroll_usd": gross_sizing_bankroll_usd,
            "sizing_bankroll_usd": sizing_bankroll_usd,
            "cash_reserve_fraction": cash_reserve_fraction,
            "cash_reserve_usd": cash_reserve_usd,
            "max_new_exposure_usd_per_run": args.max_new_exposure_usd_per_run,
            "max_new_exposure_fraction_per_run": args.max_new_exposure_fraction_per_run,
            "max_new_exposure_budget_usd": max_new_exposure_budget_usd,
            "new_exposure_target_positions_per_run": args.new_exposure_target_positions_per_run,
            "max_new_exposure_per_position_budget_usd": max_new_exposure_per_position_budget_usd,
            "kelly_sizing_bankroll_fraction_per_run": args.kelly_sizing_bankroll_fraction_per_run,
            "kelly_sizing_bankroll_usd": kelly_sizing_bankroll_usd,
            "max_position_fraction": args.max_position_fraction,
            "kelly_market_blend": args.kelly_market_blend,
            "edge_position_full_cap_edge": args.edge_position_full_cap_edge,
            "edge_position_min_multiplier": args.edge_position_min_multiplier,
            "run_log_path": str(run_log_path) if run_log_path is not None else None,
            "execution_mode": execution_mode,
            "live_collateral_start_usd": round(live_collateral_start, 6) if live_collateral_start is not None else None,
            "live_collateral_end_usd": round(live_collateral_end, 6) if live_collateral_end is not None else None,
        },
    )
    summary = {
        "fixture_mode": fixture_mode,
        "execution_mode": execution_mode,
        "run_started_at": run_started_at.isoformat(),
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "markets_discovered": len(markets),
        "markets_scored": processed_markets,
        "markets_skipped": len(skipped_markets),
        "outcomes_scored": len(all_scored),
        "signals": len(all_signals),
        "paper_rows_inserted": rows,
        "forecast_score_rows_inserted": score_rows,
        "kelly_executions": executions,
        "weather_error_count": weather_error_count,
        "observation_error_count": observation_error_count,
        "quote_error_count": quote_error_count,
        "settled_expired_positions": settlement_count,
        "settlement_error_count": settlement_error_count,
        "runtime_limited": runtime_limited,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "bankroll_usd": args.bankroll_usd,
        "sizing_bankroll_usd": sizing_bankroll_usd,
        "equity_usd": round(equity, 2),
        "pnl_usd": round(equity - args.bankroll_usd, 2),
        "open_positions": len(ledger.positions()),
        "ledger": args.ledger,
        "strategy_profile": args.strategy_profile,
        "weights_file": args.weights_file if source_weights or model_weights or probability_calibration.active else None,
        "source_weights_loaded": len(source_weights),
        "model_weights_loaded": len(model_weights),
        "probability_calibration": _probability_calibration_to_json(probability_calibration),
        "compound_kelly_sizing": args.compound_kelly_sizing,
        "gross_sizing_bankroll_usd": gross_sizing_bankroll_usd,
        "max_position_fraction": args.max_position_fraction,
        "cash_reserve_fraction": cash_reserve_fraction,
        "cash_reserve_usd": cash_reserve_usd,
        "max_new_exposure_usd_per_run": args.max_new_exposure_usd_per_run,
        "max_new_exposure_fraction_per_run": args.max_new_exposure_fraction_per_run,
        "max_new_exposure_budget_usd": max_new_exposure_budget_usd,
        "new_exposure_target_positions_per_run": args.new_exposure_target_positions_per_run,
        "max_new_exposure_per_position_budget_usd": max_new_exposure_per_position_budget_usd,
        "kelly_sizing_bankroll_fraction_per_run": args.kelly_sizing_bankroll_fraction_per_run,
        "kelly_sizing_bankroll_usd": kelly_sizing_bankroll_usd,
        "kelly_market_blend": args.kelly_market_blend,
        "edge_position_full_cap_edge": args.edge_position_full_cap_edge,
        "edge_position_min_multiplier": args.edge_position_min_multiplier,
        "run_log_path": str(run_log_path) if run_log_path is not None else None,
        "live_collateral_start_usd": round(live_collateral_start, 6) if live_collateral_start is not None else None,
        "live_collateral_end_usd": round(live_collateral_end, 6) if live_collateral_end is not None else None,
        "top_signals": [_signal_to_json(signal) for signal in all_signals[:10]],
        "top_scored_outcomes": scored_detail[:10],
        "signal_filter_counts": signal_filter_counts,
        "coverage_diagnostics": coverage_diagnostics,
        "skipped_market_examples": skipped_markets[:10],
    }
    if run_log_path is not None:
        _write_json_log(
            run_log_path,
            {
                **summary,
                "signal_settings": _signal_settings_to_json(signal_settings),
                "cli_args": _paper_args_to_json(args),
                "run_log_schema_version": 2,
                "data_provenance": {
                    "polymarket_market_source": "Gamma active events plus public-search",
                    "polymarket_quote_source": "CLOB live order book midpoint for YES and explicit NO tokens",
                    "forecast_source": "Open-Meteo live forecast APIs via weather_strategy.weather.OpenMeteoClient",
                    "observation_source": "ObservedHighClient current/final observations when enabled",
                    "execution_mode": "live CLOB market orders" if execution_mode == "live" else "paper-only Kelly ledger; no real orders are sent",
                },
                "source_weights": {key: round(value, 6) for key, value in sorted(source_weights.items())},
                "model_weights": {key: round(value, 6) for key, value in sorted(model_weights.items())},
                "probability_calibration": _probability_calibration_to_json(probability_calibration),
                "signals_detail": [_signal_to_json(signal) for signal in all_signals],
                "scored_outcomes_detail": scored_detail,
                "skipped_markets_detail": skipped_markets,
                "signal_filter_counts": signal_filter_counts,
                "coverage_diagnostics": coverage_diagnostics,
                "positions_after_run": ledger.positions(),
            },
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _live_execution_callback(live_executor: PolymarketLiveExecutionAdapter):
    def execute(order: dict[str, Any]) -> dict[str, Any]:
        result = live_executor.execute_rebalance_order(order)
        return {
            "filled_shares": result.filled_shares,
            "filled_notional_usd": result.filled_notional_usd,
            "average_price": result.average_price,
            "metadata": result.to_metadata(),
        }

    return execute


def _live_collateral_balance_after_executions(
    live_executor: Optional[PolymarketLiveExecutionAdapter],
    *,
    executions: int,
    starting_balance: Optional[float],
    poll_delay_seconds: float = 2.0,
) -> Optional[float]:
    if live_executor is None:
        return None
    balance = live_executor.collateral_balance_usd()
    if executions <= 0 or starting_balance is None:
        return balance
    if abs(balance - starting_balance) > 1e-6:
        return balance
    for _ in range(3):
        if poll_delay_seconds > 0:
            time.sleep(poll_delay_seconds)
        balance = live_executor.collateral_balance_usd()
        if abs(balance - starting_balance) > 1e-6:
            break
    return balance


def _make_run_log_path(log_dir: str, prefix: str) -> Path:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{timestamp}-{time.time_ns()}-{prefix}.json"


def _signal_filter_counts(scored_outcomes: Iterable[ScoredOutcome], settings: SignalSettings) -> dict[str, int]:
    counts = Counter()
    for outcome in scored_outcomes:
        counts[signal_filter_reason(outcome, settings) or "eligible"] += 1
    return dict(sorted(counts.items()))


def _paper_coverage_diagnostics(
    scored_outcomes: Iterable[ScoredOutcome],
    markets_to_process: Iterable[tuple[WeatherMarket, dict[str, Any]]],
    skipped_markets: Iterable[dict[str, str]],
) -> dict[str, Any]:
    scored = list(scored_outcomes)
    skipped = list(skipped_markets)
    return {
        "scored_by_city": _counter_dict(outcome.city for outcome in scored),
        "scored_by_target_date": _counter_dict(outcome.target_date.isoformat() if outcome.target_date else "unknown" for outcome in scored),
        "scored_by_local_lead_days": _counter_dict(_local_lead_days_for_outcome(outcome) for outcome in scored),
        "skipped_reason_counts": _counter_dict(item.get("reason") or "unknown" for item in skipped),
        "processed_market_contexts": [_market_context(market) for market, _ in markets_to_process],
    }


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values if value is not None)
    return dict(sorted(counts.items()))


def _market_context(market: WeatherMarket, now: Optional[datetime] = None) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    context = _market_to_json(market)
    context["run_time_utc"] = current.isoformat()
    if market.city is None or market.target_date is None:
        context["local_time"] = None
        context["local_date"] = None
        context["local_lead_days"] = None
        return context
    try:
        timezone_info = ZoneInfo(market.city.timezone)
    except ZoneInfoNotFoundError:
        context["local_time"] = None
        context["local_date"] = None
        context["local_lead_days"] = None
        return context
    local_now = current.astimezone(timezone_info)
    context["timezone"] = market.city.timezone
    context["local_time"] = local_now.isoformat()
    context["local_date"] = local_now.date().isoformat()
    context["local_hour"] = round(local_now.hour + local_now.minute / 60 + local_now.second / 3600, 4)
    context["local_lead_days"] = (market.target_date - local_now.date()).days
    return context


def _local_lead_days_for_outcome(outcome: ScoredOutcome, now: Optional[datetime] = None) -> Optional[int]:
    if outcome.target_date is None:
        return None
    city = find_city(outcome.city)
    if city is None:
        return None
    current = now or outcome.generated_at or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    try:
        local_date = current.astimezone(ZoneInfo(city.timezone)).date()
    except ZoneInfoNotFoundError:
        return None
    return (outcome.target_date - local_date).days


def _write_json_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _signal_settings_to_json(settings: SignalSettings) -> dict[str, Any]:
    return {
        "min_edge": settings.min_edge,
        "uncertainty_buffer": settings.uncertainty_buffer,
        "max_spread": settings.max_spread,
        "default_size_usd": settings.default_size_usd,
        "max_price": settings.max_price,
        "min_price": settings.min_price,
        "yes_side_min_price": settings.yes_side_min_price,
        "min_signal_fair_value": settings.min_signal_fair_value,
        "allow_bounded_bucket_entries": settings.allow_bounded_bucket_entries,
        "allow_bounded_no_side_entries": settings.allow_bounded_no_side_entries,
        "bounded_bucket_min_edge": settings.bounded_bucket_min_edge,
        "bounded_bucket_min_fair_value": settings.bounded_bucket_min_fair_value,
        "bounded_bucket_min_model_agreement": settings.bounded_bucket_min_model_agreement,
        "bounded_bucket_min_price": settings.bounded_bucket_min_price,
        "bounded_bucket_max_probability_stdev": settings.bounded_bucket_max_probability_stdev,
        "allow_no_side_entries": settings.allow_no_side_entries,
        "no_side_min_edge": settings.no_side_min_edge,
        "no_side_high_confidence_min_edge": settings.no_side_high_confidence_min_edge,
        "no_side_max_price": settings.no_side_max_price,
        "no_side_max_counter_event_probability": settings.no_side_max_counter_event_probability,
        "no_side_relaxed_counter_event_probability": settings.no_side_relaxed_counter_event_probability,
        "no_side_relaxed_counter_event_hours_utc": list(settings.no_side_relaxed_counter_event_hours_utc),
        "hold_no_side_max_counter_event_probability": settings.hold_no_side_max_counter_event_probability,
        "hold_no_side_high_conviction_min_fair_value": settings.hold_no_side_high_conviction_min_fair_value,
        "hold_no_side_high_conviction_min_edge": settings.hold_no_side_high_conviction_min_edge,
        "hold_no_side_high_conviction_counter_event_probability": settings.hold_no_side_high_conviction_counter_event_probability,
        "invalid_hold_partial_exit_fraction": settings.invalid_hold_partial_exit_fraction,
        "invalid_hold_partial_exit_min_fair_value": settings.invalid_hold_partial_exit_min_fair_value,
        "invalid_hold_partial_exit_min_price": settings.invalid_hold_partial_exit_min_price,
        "invalid_hold_partial_exit_max_price": settings.invalid_hold_partial_exit_max_price,
        "min_model_count": settings.min_model_count,
        "min_model_agreement": settings.min_model_agreement,
        "hold_min_model_agreement": settings.hold_min_model_agreement,
        "hold_min_fair_value": settings.hold_min_fair_value,
        "hold_market_confirmation_price": settings.hold_market_confirmation_price,
        "hold_market_confirmation_min_fair_value": settings.hold_market_confirmation_min_fair_value,
        "preserve_valid_holds": settings.preserve_valid_holds,
        "high_confidence_price_threshold": settings.high_confidence_price_threshold,
        "high_confidence_min_kelly_edge": settings.high_confidence_min_kelly_edge,
        "low_price_exact_bucket_threshold": settings.low_price_exact_bucket_threshold,
        "low_price_exact_bucket_min_fair_value": settings.low_price_exact_bucket_min_fair_value,
        "low_price_exact_bucket_min_edge": settings.low_price_exact_bucket_min_edge,
        "correlated_exact_bucket_max_price": settings.correlated_exact_bucket_max_price,
        "correlated_exact_bucket_min_agreement": settings.correlated_exact_bucket_min_agreement,
        "exact_bucket_max_width_f": settings.exact_bucket_max_width_f,
        "enforce_entry_timing_filter": settings.enforce_entry_timing_filter,
        "same_day_earliest_entry_hour_local": settings.same_day_earliest_entry_hour_local,
        "same_day_latest_entry_hour_local": settings.same_day_latest_entry_hour_local,
    }


def _probability_calibration_to_json(calibration: Any) -> dict[str, Any]:
    return {
        "active": bool(getattr(calibration, "active", False)),
        "center_shrink_alpha": getattr(calibration, "center_shrink_alpha", 1.0),
        "high_cap": getattr(calibration, "high_cap", None),
        "low_floor": getattr(calibration, "low_floor", None),
        "single_bucket_only": getattr(calibration, "single_bucket_only", True),
        "tail_threshold": getattr(calibration, "tail_threshold", None),
        "tail_shrink_alpha": getattr(calibration, "tail_shrink_alpha", 1.0),
    }


def _paper_args_to_json(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: _json_default(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key != "command"
    }


def _new_exposure_budget_usd(
    sizing_bankroll_usd: float,
    max_new_exposure_usd_per_run: Optional[float],
    max_new_exposure_fraction_per_run: Optional[float],
) -> Optional[float]:
    caps: list[float] = []
    if max_new_exposure_usd_per_run is not None and max_new_exposure_usd_per_run > 0:
        caps.append(float(max_new_exposure_usd_per_run))
    if max_new_exposure_fraction_per_run is not None and max_new_exposure_fraction_per_run > 0:
        caps.append(max(0.0, float(sizing_bankroll_usd) * float(max_new_exposure_fraction_per_run)))
    if not caps:
        return None
    return round(max(0.0, min(caps)), 4)


def _new_exposure_per_position_budget_usd(
    max_new_exposure_budget_usd: Optional[float],
    target_positions_per_run: Optional[float],
) -> Optional[float]:
    if max_new_exposure_budget_usd is None:
        return None
    if target_positions_per_run is None or target_positions_per_run <= 0:
        return None
    return round(max_new_exposure_budget_usd / float(target_positions_per_run), 4)


def _kelly_sizing_bankroll_usd(
    sizing_bankroll_usd: float,
    fraction_per_run: Optional[float],
) -> float:
    if fraction_per_run is None or fraction_per_run <= 0:
        return round(max(0.0, sizing_bankroll_usd), 4)
    return round(max(0.0, sizing_bankroll_usd * float(fraction_per_run)), 4)


def _deadline_exceeded(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _progress(message: str) -> None:
    print(f"[weather-paper-run] {message}", file=sys.stderr, flush=True)


def _no_side_scored_outcomes(
    market: WeatherMarket,
    raw: dict[str, Any],
    scored: list[ScoredOutcome],
    *,
    clob_client: PolymarketClobClient,
    settings: SignalSettings,
    fixture_mode: bool,
) -> tuple[list[ScoredOutcome], int]:
    if len(scored) != 1:
        return [], 0
    no_token_id, no_raw_price = _binary_no_token_and_price(raw)
    if no_token_id is None:
        return [], 0
    no_price = no_raw_price
    quote_errors = 0
    if not fixture_mode:
        try:
            quote = clob_client.fetch_order_book_quote(no_token_id)
            no_price = quote.mid if quote.mid is not None else no_price
        except RuntimeError:
            quote_errors += 1
    if no_price is None:
        return [], quote_errors
    return [invert_binary_scored_outcome(scored[0], settings, token_id=no_token_id, market_price=no_price)], quote_errors


def _binary_no_token_and_price(raw: dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    try:
        outcomes = [str(item).lower() for item in parse_jsonish_list(raw.get("outcomes"))]
        token_ids = [str(item) if item is not None else None for item in parse_jsonish_list(raw.get("clobTokenIds") or raw.get("clob_token_ids"))]
        prices = parse_jsonish_list(raw.get("outcomePrices") or raw.get("outcome_prices"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, None
    if "yes" not in outcomes or "no" not in outcomes:
        return None, None
    no_index = outcomes.index("no")
    if no_index >= len(token_ids):
        return None, None
    price = None
    if no_index < len(prices):
        try:
            price = float(prices[no_index])
        except (TypeError, ValueError):
            price = None
    return token_ids[no_index], price


def _should_process_market(
    market: WeatherMarket,
    settings: SignalSettings,
    max_lead_days: int,
    open_token_ids: set[str],
    min_lead_days: int = 0,
    now: Optional[datetime] = None,
) -> tuple[bool, Optional[str]]:
    if _market_has_open_position(market, open_token_ids):
        return True, None
    if market.city is None or market.target_date is None:
        return False, "missing city or target date"
    try:
        city_timezone = ZoneInfo(market.city.timezone)
    except ZoneInfoNotFoundError:
        return False, f"unknown city timezone {market.city.timezone}"
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local_today = current.astimezone(city_timezone).date()
    if market.target_date < local_today:
        return False, "target date has passed in local market time"
    if market.target_date < local_today + timedelta(days=min_lead_days):
        return False, f"target date before {min_lead_days}-day lead window"
    if market.target_date > local_today + timedelta(days=max_lead_days):
        return False, f"target date beyond {max_lead_days}-day lead window"
    entry_eligible, reason = market_entry_timing(market, settings=settings, now=current)
    if not entry_eligible:
        return False, reason
    return True, None


def _market_has_open_position(market: WeatherMarket, open_token_ids: set[str]) -> bool:
    return any(bucket.token_id in open_token_ids for bucket in market.buckets if bucket.token_id)


def load_fixture(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Fixture must be a JSON array of raw market objects")
    return payload


def _parse_markets(raw_markets: Iterable[Any], already_parsed: bool) -> Iterable[tuple[WeatherMarket, dict[str, Any]]]:
    for raw in raw_markets:
        if already_parsed:
            yield raw, raw.raw if isinstance(raw, WeatherMarket) else {}
            continue
        market = parse_weather_market(raw)
        if market is not None:
            market = _market_with_resolution_station(market, raw)
            yield market, raw


def _market_with_resolution_station(market: WeatherMarket, raw: dict[str, Any]) -> WeatherMarket:
    if market.city is None:
        return market
    station = _resolution_station(raw)
    if station is None:
        return market
    return replace(market, city=city_with_station_coordinates(market.city, station))


def _resolution_station(raw: dict[str, Any]) -> Optional[str]:
    text = " ".join(
        str(raw.get(key) or "")
        for key in ("resolutionSource", "description", "rules", "eventDescription")
    )
    patterns = (
        r"[?&]site=([A-Z0-9]{4})\b",
        r"/history/daily/[^/\s]+/[^/\s]+/([A-Z0-9]{4})\b",
        r"\b([A-Z]{4})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def _distributions_for_market(market: WeatherMarket, raw: dict[str, Any], weather_client: OpenMeteoClient) -> tuple[ForecastDistribution, ...]:
    samples = raw.get("forecastSamplesF")
    if samples is not None:
        sources = raw.get("forecastSourcesF")
        if isinstance(sources, dict):
            return tuple(
                ForecastDistribution(
                    city=market.city,
                    target_date=market.target_date,
                    samples_f=tuple(float(sample) for sample in source_samples),
                    generated_at=datetime.now(timezone.utc),
                    source=str(source_name),
                )
                for source_name, source_samples in sources.items()
            )
        return (
            ForecastDistribution(
                city=market.city,
                target_date=market.target_date,
                samples_f=tuple(float(sample) for sample in samples),
                generated_at=datetime.now(timezone.utc),
                source="fixture",
            ),
        )
    if market.city is None or market.target_date is None:
        raise ValueError(f"Cannot fetch live weather without city/date for market {market.slug}")
    distributions = weather_client.fetch_daily_high_sources(market.city, market.target_date)
    if not distributions:
        raise ValueError(f"No weather distributions available for {market.city.display_name} on {market.target_date}")
    return distributions


def _observed_high_for_market(market: WeatherMarket, raw: dict[str, Any], observation_client: ObservedHighClient) -> Optional[ObservedHigh]:
    observed_high = raw.get("observedHighF")
    if observed_high is not None and market.city is not None and market.target_date is not None:
        return ObservedHigh(
            city=market.city,
            target_date=market.target_date,
            max_temperature_f=float(observed_high),
            source=str(raw.get("observedHighSource") or "fixture"),
            observed_at=datetime.now(timezone.utc),
            sample_count=int(raw.get("observedHighSampleCount") or 1),
            is_actual=bool(raw.get("observedHighIsActual", True)),
            is_final=bool(raw.get("observedHighIsFinal", False)),
        )
    if market.city is None or market.target_date is None:
        return None
    return observation_client.fetch_observed_high(market.city, market.target_date)


def _market_to_json(market: WeatherMarket) -> dict[str, Any]:
    return {
        "id": market.id,
        "question": market.question,
        "slug": market.slug,
        "event_slug": market.event_slug,
        "city": market.city.display_name if market.city else None,
        "target_date": market.target_date.isoformat() if market.target_date else None,
        "bucket_count": len(market.buckets),
        "buckets": [
            {"label": bucket.label, "lower_f": bucket.lower_f, "upper_f": bucket.upper_f, "token_id": bucket.token_id, "market_price": bucket.market_price}
            for bucket in market.buckets
        ],
    }


def _signal_to_json(signal: TradeSignal) -> dict[str, Any]:
    return {
        "question": signal.question,
        "bucket": signal.bucket_label,
        "side": signal.side.value,
        "fair_value": signal.fair_value,
        "market_price": signal.market_price,
        "edge": signal.edge,
        "size_usd": signal.size_usd,
        "reason": signal.reason,
    }


def _scored_to_json(outcome, settings: Optional[SignalSettings] = None) -> dict[str, Any]:
    filter_reason = signal_filter_reason(outcome, settings)
    return {
        "market_id": outcome.market_id,
        "market_slug": outcome.market_slug,
        "token_id": outcome.token_id,
        "question": outcome.question,
        "bucket": outcome.bucket_label,
        "city": outcome.city,
        "target_date": outcome.target_date.isoformat() if outcome.target_date else None,
        "generated_at": outcome.generated_at.isoformat() if outcome.generated_at else None,
        "side": "NO" if str(outcome.bucket_label).startswith("NO: ") else "YES",
        "bucket_lower_f": outcome.bucket_lower_f,
        "bucket_upper_f": outcome.bucket_upper_f,
        "bucket_width_f": outcome.bucket_width_f,
        "resolution_unit": outcome.resolution_unit,
        "resolution_precision": outcome.resolution_precision,
        "fair_value_probability": outcome.fair_value,
        "raw_fair_value_probability": outcome.raw_fair_value,
        "market_probability": outcome.market_price,
        "edge_after_buffer": outcome.edge,
        "model_agreement": outcome.model_agreement,
        "model_count": outcome.model_count,
        "probability_stdev": outcome.probability_stdev,
        "raw_probability_stdev": outcome.raw_probability_stdev,
        "timing_entry_eligible": outcome.entry_eligible,
        "entry_eligible": outcome.entry_eligible,
        "entry_filter_reason": outcome.entry_filter_reason,
        "passes_signal_filter": filter_reason is None,
        "signal_eligible": filter_reason is None,
        "trade_eligible": filter_reason is None,
        "signal_filter_reason": filter_reason,
        "observed_high_f": outcome.observed_high_f,
        "observation_source": outcome.observation_source,
        "observation_final": outcome.observation_final,
        "observation_adjusted": outcome.observation_adjusted,
        "observed_outcome": outcome.observed_outcome,
        "model_probabilities": {
            key: round(value, 4)
            for key, value in sorted(outcome.model_probabilities.items())
        },
        "raw_model_probabilities": {
            key: round(value, 4)
            for key, value in sorted((outcome.raw_model_probabilities or {}).items())
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
