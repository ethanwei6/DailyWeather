from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weather_strategy.backtest import load_calibration_weights, load_probability_calibration
from weather_strategy.cities import city_with_station_coordinates
from weather_strategy.forecast import ConsensusForecastEngine, _calibrated_probabilities, _source_probability_views
from weather_strategy.http import HttpClient
from weather_strategy.models import CityConfig, ConsensusValue, ForecastDistribution, ScoredOutcome, TemperatureBucket, WeatherMarket
from weather_strategy.observations import ObservedHigh, ObservedHighClient, observed_outcome_for_bucket
from weather_strategy.parser import looks_like_temperature_market, parse_jsonish_list, parse_weather_market
from weather_strategy.polymarket import PolymarketClobClient, PolymarketGammaClient, PriceHistoryPoint
from weather_strategy.signals import (
    SignalSettings,
    _fails_no_side_max_price_gate,
    _price_adjusted_uncertainty_buffer,
    _required_no_side_min_edge,
    hold_filter_reason,
    no_side_counter_event_probability,
    score_outcomes,
    signal_filter_reason,
    signals_from_scored_outcomes,
)
from weather_strategy.telonex import TelonexClient, TelonexConfigurationError, _load_parquet_records
from weather_strategy.weather import extract_daily_temperature_samples, extract_weather_features


LIVE_FORWARD_ENTRY_HOURS_UTC = (0, 6, 12, 18)


@dataclass(frozen=True)
class EntryPrice:
    price: float
    timestamp: datetime
    stale_seconds: float
    raw_price: float


@dataclass
class HistoricalPosition:
    token_id: str
    market_id: str
    question: str
    bucket_label: str
    city: str
    target_date: Optional[date]
    shares: float
    cost_basis: float
    last_price: float
    payout: Optional[int]
    weather_outcome: Optional[int]
    observed_high_f: Optional[float]
    settlement_source: Optional[str]
    side: Optional[str] = None


class CachedHttpClient:
    def __init__(self, cache_dir: str | Path, *, timeout_seconds: int = 12, hard_timeout_seconds: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = HttpClient(timeout_seconds=timeout_seconds)
        self.hard_timeout_seconds = hard_timeout_seconds
        self.hits = 0
        self.misses = 0

    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> Any:
        path, legacy_json_path = self._cache_paths(url, params=params, headers=headers)
        last_error: Optional[json.JSONDecodeError] = None
        for _ in range(2):
            payload = self.get_text(url, params=params, headers=headers)
            try:
                return json.loads(payload)
            except json.JSONDecodeError as error:
                last_error = error
                for candidate in (path, legacy_json_path):
                    if candidate.exists():
                        candidate.unlink()
        raise RuntimeError(f"Invalid JSON response for {url}: {last_error}") from last_error

    def get_text(self, url: str, params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, str]] = None) -> str:
        path, legacy_json_path = self._cache_paths(url, params=params, headers=headers)
        if path.exists():
            self.hits += 1
            return path.read_text(encoding="utf-8")
        if legacy_json_path.exists():
            self.hits += 1
            return legacy_json_path.read_text(encoding="utf-8")
        self.misses += 1
        try:
            payload = self._fetch_text_with_deadline(url, params=params, headers=headers)
        except TimeoutError as error:
            raise RuntimeError(f"Request hard-timeout after {self.hard_timeout_seconds}s for {url}") from error
        path.write_text(payload, encoding="utf-8")
        return payload

    def _fetch_text_with_deadline(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> str:
        if self.hard_timeout_seconds <= 0:
            return self.http.get_text(url, params=params, headers=headers)

        result_queue: queue.Queue[tuple[bool, str | BaseException]] = queue.Queue(maxsize=1)

        def fetch() -> None:
            try:
                result_queue.put((True, self.http.get_text(url, params=params, headers=headers)))
            except BaseException as error:
                result_queue.put((False, error))

        worker = threading.Thread(target=fetch, daemon=True)
        worker.start()
        worker.join(self.hard_timeout_seconds)
        if worker.is_alive():
            raise TimeoutError("HTTP request exceeded hard timeout")

        ok, value = result_queue.get_nowait()
        if ok:
            return str(value)
        raise value

    def _cache_paths(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> tuple[Path, Path]:
        key_payload = {"url": url, "params": params or {}, "headers": headers or {}}
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.txt", self.cache_dir / f"{key}.json"


class SingleRunForecastClient:
    SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

    def __init__(self, http: CachedHttpClient):
        self.http = http

    def fetch_sources(
        self,
        city: CityConfig,
        target_date: date,
        decision_time: datetime,
        *,
        availability_lag_hours: int,
    ) -> tuple[tuple[ForecastDistribution, ...], datetime, list[str]]:
        run_time = forecast_run_time(decision_time, availability_lag_hours=availability_lag_hours)
        forecast_days = max(2, min(10, (target_date - run_time.date()).days + 2))
        sources: list[ForecastDistribution] = []
        errors: list[str] = []
        for source_name, model_name in (
            ("single_run_best_match", None),
            ("single_run_gfs_global", "gfs_global"),
            ("single_run_ecmwf_ifs025", "ecmwf_ifs025"),
        ):
            params: dict[str, Any] = {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "run": run_time.strftime("%Y-%m-%dT%H:%M"),
                "forecast_days": forecast_days,
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
            if model_name:
                params["models"] = model_name
            try:
                payload = self.http.get_json(self.SINGLE_RUN_URL, params=params)
                samples = extract_daily_temperature_samples(payload, target_date)
            except (RuntimeError, ValueError, KeyError) as error:
                errors.append(f"{source_name}: {str(error)[:900]}")
                continue
            if not samples:
                errors.append(f"{source_name}: no samples for {target_date}")
                continue
            sources.append(
                ForecastDistribution(
                    city=city,
                    target_date=target_date,
                    samples_f=tuple(samples),
                    generated_at=run_time,
                    source=source_name,
                    model_metadata={
                        "historical_single_run": True,
                        "run_time": run_time.isoformat(),
                        "features": extract_weather_features(payload, target_date),
                    },
                )
            )
        return tuple(sources), run_time, errors


def forecast_run_time(decision_time: datetime, *, availability_lag_hours: int = 6) -> datetime:
    current = decision_time if decision_time.tzinfo else decision_time.replace(tzinfo=timezone.utc)
    eligible = current.astimezone(timezone.utc) - timedelta(hours=availability_lag_hours)
    run_hour = (eligible.hour // 6) * 6
    return eligible.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def select_entry_price(
    history: Iterable[PriceHistoryPoint],
    decision_time: datetime,
    *,
    max_staleness_minutes: int,
    slippage: float,
) -> Optional[EntryPrice]:
    current = decision_time if decision_time.tzinfo else decision_time.replace(tzinfo=timezone.utc)
    selected: Optional[PriceHistoryPoint] = None
    for point in history:
        if point.timestamp <= current:
            selected = point
        else:
            break
    if selected is None:
        return None
    stale_seconds = (current - selected.timestamp).total_seconds()
    if stale_seconds < 0 or stale_seconds > max_staleness_minutes * 60:
        return None
    price = max(0.001, min(0.999, selected.price + slippage))
    return EntryPrice(price=price, timestamp=selected.timestamp, stale_seconds=stale_seconds, raw_price=selected.price)


def run_long_historical_backtest(
    *,
    bankroll_usd: float = 100.0,
    pages: int = 10,
    limit_per_page: int = 50,
    max_markets: int = 8000,
    query: str = "highest temperature",
    min_end_date: Optional[date] = None,
    max_end_date: Optional[date] = None,
    market_selection: str = "end_date_asc",
    entry_hours_utc: tuple[int, ...] = LIVE_FORWARD_ENTRY_HOURS_UTC,
    min_lead_days: int = 1,
    max_lead_days: int = 2,
    max_runtime_seconds: float = 0.0,
    max_price_staleness_minutes: int = 90,
    historical_price_slippage: float = 0.01,
    forecast_availability_lag_hours: int = 6,
    kelly_fraction: float = 0.25,
    compound_kelly_sizing: bool = False,
    max_new_exposure_usd_per_run: Optional[float] = None,
    max_new_exposure_fraction_per_run: Optional[float] = None,
    new_exposure_target_positions_per_run: Optional[float] = None,
    kelly_sizing_bankroll_fraction_per_run: Optional[float] = None,
    max_position_usd: float = 50.0,
    max_position_fraction: Optional[float] = None,
    kelly_market_blend: float = 0.0,
    edge_position_full_cap_edge: float = 0.0,
    edge_position_min_multiplier: float = 0.35,
    min_trade_usd: float = 1.0,
    settings: Optional[SignalSettings] = None,
    min_volume_usd: float = 0.0,
    weights_file: str | Path = "work/data/model_weights.json",
    cache_dir: str | Path = "work/cache/long_backtest",
    run_log_dir: str | Path = "work/logs/long_backtests",
    progress_every: int = 50,
    http_hard_timeout_seconds: int = 30,
    price_source: str = "telonex",
    market_source: str = "telonex",
    settlement_audit: str = "weather_crosscheck",
    strategy_profile: str = "manual",
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = None if max_runtime_seconds <= 0 else started + max_runtime_seconds
    preparation_deadline = None if max_runtime_seconds <= 0 else started + (max_runtime_seconds * 0.65)
    settings = settings or SignalSettings(enforce_entry_timing_filter=True)
    http = CachedHttpClient(cache_dir, hard_timeout_seconds=http_hard_timeout_seconds)
    gamma = PolymarketGammaClient(http=http)
    clob = PolymarketClobClient(http=http)
    price_source_name, telonex = _historical_price_source(price_source, cache_dir, hard_timeout_seconds=http_hard_timeout_seconds)
    market_source_name, telonex = _historical_market_source(market_source, cache_dir, telonex, hard_timeout_seconds=http_hard_timeout_seconds)
    forecast_client = SingleRunForecastClient(http)
    observation_client = ObservedHighClient(http=http)
    source_weights, model_weights = load_calibration_weights(weights_file)
    probability_calibration = load_probability_calibration(weights_file)
    forecast_engine = ConsensusForecastEngine(
        source_weights=source_weights,
        model_weights=model_weights,
        probability_calibration=probability_calibration,
    )

    raw_markets = (
        _discover_telonex_raw_temperature_markets(
            telonex,
            query=query,
            max_candidates=max(max_markets * 3, pages * limit_per_page, 500),
            min_end_date=min_end_date,
            max_end_date=max_end_date,
            market_selection=market_selection,
        )
        if market_source_name == "telonex"
        else _discover_raw_temperature_markets(gamma, query=query, pages=pages, limit_per_page=limit_per_page)
    )
    parsed = _parse_historical_markets(raw_markets, max_markets=max_markets, min_volume_usd=min_volume_usd)
    _progress(
        progress_every,
        f"discovered_raw={len(raw_markets)} parsed={len(parsed)} cache_hits={http.hits} cache_misses={http.misses}",
    )

    price_error_count = 0
    no_side_price_error_count = 0
    no_side_price_history_count = 0
    no_side_session_count = 0
    resolution_error_count = 0
    weather_crosscheck_mismatches = 0
    weather_crosscheck_ambiguous_count = 0
    weather_crosscheck_mismatch_examples: list[dict[str, Any]] = []
    weather_crosscheck_ambiguous_examples: list[dict[str, Any]] = []
    session_markets: dict[datetime, list[tuple[WeatherMarket, dict[str, Any], EntryPrice, dict[str, Any]]]] = {}
    price_history_cache: dict[str, list[PriceHistoryPoint]] = {}
    observation_cache: dict[tuple[str, ...], Optional[ObservedHigh]] = {}
    nonempty_price_history_count = 0
    skipped: list[dict[str, str]] = []

    for market_index, (market, raw) in enumerate(parsed, start=1):
        if _deadline_exceeded(preparation_deadline):
            break
        bucket = _single_bucket(market)
        if bucket is None or not bucket.token_id or market.city is None or market.target_date is None:
            skipped.append({"question": market.question, "reason": "missing token/city/target/bucket"})
            continue
        resolution_city = _market_resolution_city(market.city, raw)
        if resolution_city != market.city:
            market = replace(market, city=resolution_city)
        resolution = _resolve_market_outcome(
            market,
            raw,
            bucket,
            observation_client,
            observation_cache,
            fetch_weather_crosscheck=settlement_audit == "weather_crosscheck",
        )
        if resolution["payout"] is None:
            resolution_error_count += 1
            skipped.append({"question": market.question, "reason": "no final Polymarket/weather outcome"})
            continue
        if resolution["weather_outcome"] is not None and resolution["polymarket_payout"] is not None and resolution["weather_outcome"] != resolution["polymarket_payout"]:
            weather_crosscheck_mismatches += 1
            if len(weather_crosscheck_mismatch_examples) < 50:
                weather_crosscheck_mismatch_examples.append(_resolution_example(market, bucket, raw, resolution))
        if resolution.get("weather_ambiguous"):
            weather_crosscheck_ambiguous_count += 1
            if len(weather_crosscheck_ambiguous_examples) < 50:
                weather_crosscheck_ambiguous_examples.append(_resolution_example(market, bucket, raw, resolution))
        replay_times = [
            (decision_time, maintenance_only)
            for decision_time, maintenance_only in _candidate_replay_times(market, entry_hours_utc, min_lead_days, max_lead_days)
            if _market_active_at(raw, decision_time)
        ]
        if not replay_times:
            skipped.append({"question": market.question, "reason": "no configured replay sessions while market active"})
            continue
        price_start_ts, price_end_ts = _price_history_bounds_for_replay_times(
            replay_times,
            max_staleness_minutes=max_price_staleness_minutes,
        )
        no_token_id = _binary_no_token(raw) if settings.allow_no_side_entries else None
        no_history: list[PriceHistoryPoint] = []
        if no_token_id:
            with ThreadPoolExecutor(max_workers=2) as executor:
                yes_future = executor.submit(
                    _cached_price_history,
                    price_history_cache,
                    clob,
                    bucket.token_id,
                    start_ts=price_start_ts,
                    end_ts=price_end_ts,
                    price_source=price_source_name,
                    telonex=telonex,
                    market_slug=raw.get("slug") or market.slug,
                    outcome="Yes",
                )
                no_future = executor.submit(
                    _cached_price_history,
                    price_history_cache,
                    clob,
                    no_token_id,
                    start_ts=price_start_ts,
                    end_ts=price_end_ts,
                    price_source=price_source_name,
                    telonex=telonex,
                    market_slug=raw.get("slug") or market.slug,
                    outcome="No",
                )
                try:
                    history = yes_future.result()
                except (RuntimeError, ValueError) as error:
                    price_error_count += 1
                    skipped.append({"question": market.question, "reason": f"price_history: {str(error)[:240]}"})
                    continue
                try:
                    no_history = no_future.result()
                    if no_history:
                        no_side_price_history_count += 1
                except (RuntimeError, ValueError) as error:
                    no_side_price_error_count += 1
                    if len(skipped) < 50:
                        skipped.append({"question": market.question, "reason": f"no_side_price_history: {str(error)[:240]}"})
        else:
            try:
                history = _cached_price_history(
                    price_history_cache,
                    clob,
                    bucket.token_id,
                    start_ts=price_start_ts,
                    end_ts=price_end_ts,
                    price_source=price_source_name,
                    telonex=telonex,
                    market_slug=raw.get("slug") or market.slug,
                    outcome="Yes",
                )
            except (RuntimeError, ValueError) as error:
                price_error_count += 1
                skipped.append({"question": market.question, "reason": f"price_history: {str(error)[:240]}"})
                continue
        if not history:
            skipped.append({"question": market.question, "reason": "empty price history"})
            continue
        nonempty_price_history_count += 1
        added_entry_session = False
        for decision_time, maintenance_only in replay_times:
            if maintenance_only and not added_entry_session:
                continue
            entry = select_entry_price(
                history,
                decision_time,
                max_staleness_minutes=max_price_staleness_minutes,
                slippage=historical_price_slippage,
            )
            if entry is None:
                continue
            priced_bucket = replace(bucket, market_price=entry.price)
            priced_market = replace(market, buckets=(priced_bucket,))
            session_markets.setdefault(decision_time, []).append(
                (
                    priced_market,
                    raw,
                    entry,
                    {
                        "polymarket_payout": resolution["polymarket_payout"],
                        "payout": resolution["payout"],
                        "side": "YES",
                        "weather_outcome": resolution["weather_outcome"],
                        "weather_ambiguous": resolution.get("weather_ambiguous", False),
                        "observed_high_f": resolution["observed_high_f"],
                        "settlement_source": resolution["settlement_source"],
                        "maintenance_only": maintenance_only,
                    },
                )
            )
            if not maintenance_only:
                added_entry_session = True
            if no_token_id and no_history:
                no_entry = select_entry_price(
                    no_history,
                    decision_time,
                    max_staleness_minutes=max_price_staleness_minutes,
                    slippage=historical_price_slippage,
                )
                if no_entry is None:
                    continue
                no_bucket = replace(bucket, token_id=no_token_id, market_price=no_entry.price)
                no_market = replace(market, buckets=(no_bucket,))
                session_markets.setdefault(decision_time, []).append(
                    (
                        no_market,
                        raw,
                        no_entry,
                        {
                            "polymarket_payout": _invert_binary_outcome(resolution["polymarket_payout"]),
                            "payout": _invert_binary_outcome(resolution["payout"]),
                            "side": "NO",
                            "yes_token_id": bucket.token_id,
                            "weather_outcome": _invert_binary_outcome(resolution["weather_outcome"]),
                            "weather_ambiguous": resolution.get("weather_ambiguous", False),
                            "observed_high_f": resolution["observed_high_f"],
                            "settlement_source": resolution["settlement_source"],
                            "maintenance_only": maintenance_only,
                        },
                    )
                )
                no_side_session_count += 1
        if not added_entry_session:
            skipped.append({"question": market.question, "reason": "no usable historical price at configured entry sessions"})
        if progress_every > 0 and market_index % progress_every == 0:
            _progress(
                progress_every,
                (
                    f"prepared_markets={market_index}/{len(parsed)} "
                    f"price_history_requests={len(price_history_cache)} usable_price_histories={nonempty_price_history_count} "
                    f"sessions={sum(len(items) for items in session_markets.values())} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}"
                ),
            )

    cash = bankroll_usd
    positions: dict[str, HistoricalPosition] = {}
    executions: list[dict[str, Any]] = []
    scored_detail: list[dict[str, Any]] = []
    forecast_error_count = 0
    forecast_error_count_by_city: dict[str, int] = {}
    forecast_error_count_by_source: dict[str, int] = {}
    forecast_error_examples: list[dict[str, Any]] = []
    markets_scored = 0
    outcomes_scored = 0
    signal_count = 0
    session_count = 0
    equity_curve: list[dict[str, Any]] = []

    sorted_session_times = sorted(session_markets)
    if sorted_session_times:
        equity_curve.append(_historical_equity_snapshot(sorted_session_times[0], cash, positions))

    for session_time in sorted_session_times:
        if _deadline_exceeded(deadline):
            break
        session_count += 1
        _settle_due_positions(positions, executions, session_time, cash_ref := {"cash": cash})
        cash = cash_ref["cash"]
        scored: list[ScoredOutcome] = []
        score_metadata: dict[str, dict[str, Any]] = {}
        forecast_cache, forecast_errors = _fetch_session_forecasts(
            forecast_client,
            session_markets[session_time],
            session_time,
            availability_lag_hours=forecast_availability_lag_hours,
            max_workers=_historical_forecast_workers(),
        )
        for forecast_key, error_record in forecast_errors.items():
            errors = tuple(error_record.get("errors", ()))
            if not errors:
                continue
            city_name, target_date_text = forecast_key
            forecast_error_count += len(errors)
            for error in errors:
                forecast_error_count_by_city[city_name] = forecast_error_count_by_city.get(city_name, 0) + 1
                source_name = str(error).split(":", 1)[0]
                forecast_error_count_by_source[source_name] = forecast_error_count_by_source.get(source_name, 0) + 1
            for error in errors[:3]:
                if len(forecast_error_examples) < 50:
                    forecast_error_examples.append(
                        {
                            "session_time": session_time.isoformat(),
                            "city": city_name,
                            "target_date": target_date_text,
                            "question": str(error_record.get("example_question") or ""),
                            "error": error,
                        }
                    )
        for market, raw, entry, settlement in session_markets[session_time]:
            if _deadline_exceeded(deadline):
                break
            if market.city is None or market.target_date is None:
                continue
            forecast_key = (market.city.display_name, market.target_date.isoformat())
            if forecast_key not in forecast_cache:
                skipped.append({"question": market.question, "reason": "no historical forecast distributions"})
                continue
            distributions, run_time = forecast_cache[forecast_key]
            consensus = forecast_engine.consensus_by_bucket(distributions, market.buckets)
            observed = _historical_observed_high_for_session(
                market,
                raw,
                observation_client,
                observation_cache,
                session_time,
            )
            if observed is not None:
                consensus = forecast_engine.apply_observed_high(consensus, market.buckets, observed, now=session_time)
            scored_rows = score_outcomes(market, consensus, settings=settings, now=session_time)
            if settlement.get("side") == "NO":
                scored_rows = [_invert_binary_scored_outcome(outcome, settings) for outcome in scored_rows]
            if settlement.get("maintenance_only"):
                scored_rows = [
                    replace(
                        outcome,
                        entry_eligible=False,
                        entry_filter_reason="target-day maintenance only; new entries disabled by lead window",
                    )
                    for outcome in scored_rows
                ]
            markets_scored += 1
            outcomes_scored += len(scored_rows)
            for outcome in scored_rows:
                scored.append(outcome)
                score_metadata[outcome.token_id or outcome.bucket_label] = {
                    **settlement,
                    "entry_price_timestamp": entry.timestamp.isoformat(),
                    "entry_price_stale_seconds": round(entry.stale_seconds, 1),
                    "raw_price": round(entry.raw_price, 4),
                    "entry_price": round(entry.price, 4),
                    "exit_price": round(max(0.001, min(0.999, entry.raw_price - historical_price_slippage)), 4),
                    "forecast_run_time": run_time.isoformat(),
                    "forecast_sources": sorted({distribution.source for distribution in distributions}),
                    "market_created_at": raw.get("createdAt") or raw.get("creationDate"),
                    "market_start_date": raw.get("startDate"),
                }
        signals = signals_from_scored_outcomes(scored, settings=settings)
        signal_count += len(signals)
        selected_tokens = {signal.token_id for signal in signals if signal.token_id}
        _rebalance_session(
            scored,
            selected_tokens,
            score_metadata,
            positions,
            executions,
            cash_ref := {"cash": cash},
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_new_exposure_usd_per_run=max_new_exposure_usd_per_run,
            max_new_exposure_fraction_per_run=max_new_exposure_fraction_per_run,
            new_exposure_target_positions_per_run=new_exposure_target_positions_per_run,
            kelly_sizing_bankroll_fraction_per_run=kelly_sizing_bankroll_fraction_per_run,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
            settings=settings,
        )
        cash = cash_ref["cash"]
        scored_detail.extend(_scored_to_json(outcome, score_metadata.get(outcome.token_id or outcome.bucket_label, {}), settings) for outcome in scored)
        equity_curve.append(_historical_equity_snapshot(session_time, cash, positions))
        _progress(
            progress_every,
            (
                f"session={session_time.isoformat()} session_count={session_count} "
                f"markets_scored={markets_scored} outcomes_scored={outcomes_scored} signals={signal_count} "
                f"cash={cash:.2f} elapsed_seconds={time.monotonic() - started:.1f}"
            ),
        )

    runtime_limited = _deadline_exceeded(deadline)
    cash, open_value = _finalize_positions_for_result(positions, executions, cash, runtime_limited=runtime_limited)
    equity = cash + open_value
    equity_curve.append(_historical_final_equity_snapshot(cash, open_value, positions))
    crosscheck_summary = _crosscheck_pnl_summary(executions)
    run_log_path = _make_run_log_path(run_log_dir, "long-backtest")
    trade_diagnostics = _trade_performance_diagnostics(executions)
    performance_diagnostics = _performance_diagnostics_from_equity_curve(equity_curve, bankroll_usd)
    score_calibration_diagnostics = _score_calibration_diagnostics(scored_detail)
    signal_filter_diagnostics = _signal_filter_diagnostics(scored_detail)
    signal_opportunity_diagnostics = _signal_opportunity_diagnostics(scored_detail, settings)
    data_quality_diagnostics = _data_quality_diagnostics(
        executions,
        scored_detail,
        max_price_staleness_minutes=max_price_staleness_minutes,
        forecast_availability_lag_hours=forecast_availability_lag_hours,
    )
    settlement_quality_diagnostics = _settlement_quality_diagnostics(scored_detail, executions)
    selected_candidate_weather_validation = _selected_candidate_weather_validation(scored_detail, settings)
    strategy_sensitivity_diagnostics = _strategy_sensitivity_diagnostics(
        scored_detail,
        settings,
        bankroll_usd=bankroll_usd,
        kelly_fraction=kelly_fraction,
        compound_kelly_sizing=compound_kelly_sizing,
        max_position_usd=max_position_usd,
        max_position_fraction=max_position_fraction,
        kelly_market_blend=kelly_market_blend,
        edge_position_full_cap_edge=edge_position_full_cap_edge,
        edge_position_min_multiplier=edge_position_min_multiplier,
        min_trade_usd=min_trade_usd,
    )
    robustness_diagnostics = _robustness_diagnostics(
        scored_detail,
        settings,
        bankroll_usd=bankroll_usd,
        kelly_fraction=kelly_fraction,
        compound_kelly_sizing=compound_kelly_sizing,
        max_position_usd=max_position_usd,
        max_position_fraction=max_position_fraction,
        kelly_market_blend=kelly_market_blend,
        edge_position_full_cap_edge=edge_position_full_cap_edge,
        edge_position_min_multiplier=edge_position_min_multiplier,
        min_trade_usd=min_trade_usd,
    )
    strategy_recommendation_diagnostics = _strategy_recommendation_diagnostics(
        strategy_sensitivity_diagnostics,
        robustness_diagnostics,
    )
    result = {
        "strategy_profile": strategy_profile,
        "bankroll_usd": bankroll_usd,
        "ending_equity_usd": round(equity, 2),
        "pnl_usd": round(equity - bankroll_usd, 2),
        "return_pct": round((equity - bankroll_usd) / bankroll_usd, 4) if bankroll_usd else None,
        "cash_usd": round(cash, 2),
        "open_position_value_usd": round(open_value, 2),
        "open_positions": len(positions),
        "markets_discovered_raw": len(raw_markets),
        "markets_parsed": len(parsed),
        "price_history_requests": len(price_history_cache),
        "markets_with_price_history": nonempty_price_history_count,
        "session_count": session_count,
        "markets_scored": markets_scored,
        "outcomes_scored": outcomes_scored,
        "signals": signal_count,
        "executions": len(executions),
        "buys": sum(1 for item in executions if item["action"] == "BUY"),
        "sells": sum(1 for item in executions if item["action"] == "SELL"),
        "settlements": sum(1 for item in executions if item["action"] == "SETTLE"),
        "realized_pnl_usd": round(sum(float(item.get("realized_pnl_usd") or 0.0) for item in executions), 2),
        **crosscheck_summary,
        "performance_diagnostics": performance_diagnostics,
        "equity_curve": equity_curve,
        "trade_diagnostics": trade_diagnostics,
        "score_calibration_diagnostics": score_calibration_diagnostics,
        "signal_filter_diagnostics": signal_filter_diagnostics,
        "signal_opportunity_diagnostics": signal_opportunity_diagnostics,
        "strategy_sensitivity_diagnostics": strategy_sensitivity_diagnostics,
        "robustness_diagnostics": robustness_diagnostics,
        "strategy_recommendation_diagnostics": strategy_recommendation_diagnostics,
        "real_data_audit": _real_data_audit(
            scored_detail,
            executions,
            data_quality_diagnostics=data_quality_diagnostics,
            settlement_quality_diagnostics=settlement_quality_diagnostics,
            require_weather_crosscheck=settlement_audit == "weather_crosscheck",
        ),
        "forecast_error_count": forecast_error_count,
        "forecast_error_count_by_city": dict(sorted(forecast_error_count_by_city.items(), key=lambda item: (-item[1], item[0]))),
        "forecast_error_count_by_source": dict(sorted(forecast_error_count_by_source.items(), key=lambda item: (-item[1], item[0]))),
        "forecast_error_examples": forecast_error_examples,
        "price_error_count": price_error_count,
        "no_side_price_error_count": no_side_price_error_count,
        "no_side_price_history_count": no_side_price_history_count,
        "no_side_session_count": no_side_session_count,
        "resolution_error_count": resolution_error_count,
        "weather_crosscheck_mismatches": weather_crosscheck_mismatches,
        "weather_crosscheck_ambiguous_count": weather_crosscheck_ambiguous_count,
        "weather_crosscheck_mismatch_examples": weather_crosscheck_mismatch_examples,
        "weather_crosscheck_ambiguous_examples": weather_crosscheck_ambiguous_examples,
        "skipped_reason_counts": _reason_counts(skipped),
        "runtime_limited": runtime_limited,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "settings": {
            "query": query,
            "min_end_date": min_end_date.isoformat() if min_end_date else None,
            "max_end_date": max_end_date.isoformat() if max_end_date else None,
            "market_selection": market_selection,
            "pages": pages,
            "limit_per_page": limit_per_page,
            "max_markets": max_markets,
            "max_runtime_seconds": max_runtime_seconds,
            "entry_hours_utc": list(entry_hours_utc),
            "live_forward_entry_hours_utc": list(LIVE_FORWARD_ENTRY_HOURS_UTC),
            "entry_hours_match_live_forward": tuple(entry_hours_utc) == LIVE_FORWARD_ENTRY_HOURS_UTC,
            "min_lead_days": min_lead_days,
            "max_lead_days": max_lead_days,
            "max_price_staleness_minutes": max_price_staleness_minutes,
            "historical_price_slippage": historical_price_slippage,
            "forecast_availability_lag_hours": forecast_availability_lag_hours,
            "kelly_fraction": kelly_fraction,
            "compound_kelly_sizing": compound_kelly_sizing,
            "max_new_exposure_usd_per_run": max_new_exposure_usd_per_run,
            "max_new_exposure_fraction_per_run": max_new_exposure_fraction_per_run,
            "new_exposure_target_positions_per_run": new_exposure_target_positions_per_run,
            "kelly_sizing_bankroll_fraction_per_run": kelly_sizing_bankroll_fraction_per_run,
            "max_position_usd": max_position_usd,
            "max_position_fraction": max_position_fraction,
            "kelly_market_blend": kelly_market_blend,
            "edge_position_full_cap_edge": edge_position_full_cap_edge,
            "edge_position_min_multiplier": edge_position_min_multiplier,
            "min_trade_usd": min_trade_usd,
            "min_volume_usd": min_volume_usd,
            "weights_file": str(weights_file),
            "progress_every": progress_every,
            "http_hard_timeout_seconds": http_hard_timeout_seconds,
            "price_source": price_source_name,
            "market_source": market_source_name,
            "settlement_audit": settlement_audit,
            "strategy_profile": strategy_profile,
            "source_weights_loaded": len(source_weights),
            "model_weights_loaded": len(model_weights),
            "probability_calibration": {
                "active": probability_calibration.active,
                "center_shrink_alpha": probability_calibration.center_shrink_alpha,
                "high_cap": probability_calibration.high_cap,
                "low_floor": probability_calibration.low_floor,
                "single_bucket_only": probability_calibration.single_bucket_only,
                "tail_threshold": probability_calibration.tail_threshold,
                "tail_shrink_alpha": probability_calibration.tail_shrink_alpha,
            },
            "signal_settings": _signal_settings_to_json(settings),
        },
        "data_provenance": {
            "polymarket_market_source": "Telonex Polymarket markets dataset filtered to resolved markets with quote availability" if market_source_name == "telonex" else "Gamma public-search",
            "polymarket_price_source": _price_source_description(price_source_name, "YES"),
            "polymarket_no_price_source": _price_source_description(price_source_name, "NO"),
            "forecast_source": "Open-Meteo Single Runs API",
            "settlement_preference": (
                "Polymarket resolved YES payout, station METAR/ASOS weather cross-check when the market names a station"
                if settlement_audit == "weather_crosscheck"
                else "Polymarket resolved YES payout; station weather cross-check skipped for broad replay speed"
            ),
            "cache_dir": str(cache_dir),
            "cache_hits": http.hits,
            "cache_misses": http.misses,
            "telonex_download_hits": telonex.download_hits if telonex is not None else 0,
            "telonex_download_misses": telonex.download_misses if telonex is not None else 0,
        },
        "data_quality_diagnostics": data_quality_diagnostics,
        "settlement_quality_diagnostics": settlement_quality_diagnostics,
        "selected_candidate_weather_validation": selected_candidate_weather_validation,
        "top_trades": sorted(executions, key=lambda item: abs(float(item.get("realized_pnl_usd") or 0.0)), reverse=True)[:20],
        "top_weather_crosscheck_mismatch_trades": _top_weather_crosscheck_mismatch_trades(executions),
        "executions_detail": executions,
        "scored_outcomes_detail_count": len(scored_detail),
        "scored_outcomes_detail": sorted(scored_detail, key=lambda item: item["edge"], reverse=True),
        "skipped_examples": skipped[:50],
        "run_log_path": str(run_log_path),
    }
    try:
        _write_json_log(run_log_path, result)
    except OSError as exc:
        result["run_log_write_error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_scored_outcome_snapshot(snapshot_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load scored rows from a long-backtest JSON artifact without refetching data."""
    path = Path(snapshot_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        source: dict[str, Any] = {"source_path": str(path)}
        raw_rows = payload
    elif isinstance(payload, Mapping):
        source = dict(payload)
        source.setdefault("source_path", str(path))
        raw_rows = source.get("scored_outcomes_detail") or source.get("scored_outcomes") or []
    else:
        raise ValueError(f"Unsupported scored-outcome snapshot payload: {type(payload).__name__}")
    if not isinstance(raw_rows, list):
        raise ValueError("Scored-outcome snapshot must contain a list of scored rows")

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        if "fair_value" not in row and "fair_value_probability" in row:
            row["fair_value"] = row["fair_value_probability"]
        if "market_price" not in row and "market_probability" in row:
            row["market_price"] = row["market_probability"]
        if "edge" not in row and "edge_after_buffer" in row:
            row["edge"] = row["edge_after_buffer"]
        rows.append(row)
    if not rows:
        raise ValueError(f"No scored_outcomes_detail rows found in {path}")
    return source, rows


def run_cached_scored_outcome_replay(
    *,
    snapshot_path: str | Path,
    settings: SignalSettings,
    strategy_profile: str = "manual",
    bankroll_usd: float = 100.0,
    kelly_fraction: float = 0.25,
    compound_kelly_sizing: bool = False,
    max_new_exposure_usd_per_run: Optional[float] = None,
    max_new_exposure_fraction_per_run: Optional[float] = None,
    new_exposure_target_positions_per_run: Optional[float] = None,
    kelly_sizing_bankroll_fraction_per_run: Optional[float] = None,
    max_position_usd: float = 50.0,
    max_position_fraction: Optional[float] = None,
    kelly_market_blend: float = 0.0,
    edge_position_full_cap_edge: float = 0.0,
    edge_position_min_multiplier: float = 0.35,
    min_trade_usd: float = 1.0,
    run_log_dir: str | Path = "work/logs/scored_replays",
    recompute_from_raw_model_probabilities: bool = False,
    weights_file: str | Path | None = None,
) -> dict[str, Any]:
    """Replay a strategy from saved long-backtest scored rows only.

    This intentionally performs no market, forecast, price-history, or observation
    fetches. It is meant for fast strategy sweeps over one expensive real-data
    scoring artifact.
    """
    started = time.monotonic()
    source, scored = load_scored_outcome_snapshot(snapshot_path)
    raw_recalibration = {
        "enabled": False,
        "weights_file": str(weights_file) if weights_file else None,
        "rows_with_raw_model_probabilities": sum(
            1
            for row in scored
            if isinstance(row.get("raw_model_probabilities"), Mapping) and row.get("raw_model_probabilities")
        ),
        "rows_recomputed": 0,
    }
    if recompute_from_raw_model_probabilities:
        if weights_file is None:
            raise ValueError("weights_file is required when recompute_from_raw_model_probabilities=True")
        scored, raw_recalibration = _recompute_scored_from_raw_model_probabilities(scored, weights_file)
    replay = _json_kelly_replay(
        strategy_profile,
        scored,
        settings,
        bankroll_usd=bankroll_usd,
        kelly_fraction=kelly_fraction,
        compound_kelly_sizing=compound_kelly_sizing,
        max_new_exposure_usd_per_run=max_new_exposure_usd_per_run,
        max_new_exposure_fraction_per_run=max_new_exposure_fraction_per_run,
        new_exposure_target_positions_per_run=new_exposure_target_positions_per_run,
        kelly_sizing_bankroll_fraction_per_run=kelly_sizing_bankroll_fraction_per_run,
        max_position_usd=max_position_usd,
        max_position_fraction=max_position_fraction,
        kelly_market_blend=kelly_market_blend,
        edge_position_full_cap_edge=edge_position_full_cap_edge,
        edge_position_min_multiplier=edge_position_min_multiplier,
        min_trade_usd=min_trade_usd,
        include_executions=True,
    )
    run_log_path = _make_run_log_path(run_log_dir, "scored-replay")
    source_settings = source.get("settings") if isinstance(source.get("settings"), Mapping) else {}
    source_signal_settings = (
        source_settings.get("signal_settings")
        if isinstance(source_settings, Mapping) and isinstance(source_settings.get("signal_settings"), Mapping)
        else {}
    )
    source_real_data_audit = source.get("real_data_audit") if isinstance(source.get("real_data_audit"), Mapping) else None
    selected_candidate_weather_validation = _selected_candidate_weather_validation(scored, settings)
    replay_real_data_audit = {
        "method": (
            "Replay inherits the source artifact's real-data audit and rechecks traded-token "
            "weather mismatch/ambiguity counts from cached scored rows. No live data was refetched."
        ),
        "source_real_data_audit_passed": source_real_data_audit.get("passed") if source_real_data_audit else None,
        "replay_weather_mismatch_trades": replay.get("weather_mismatch_trades"),
        "replay_weather_ambiguous_trades": replay.get("weather_ambiguous_trades"),
        "passed": (
            (source_real_data_audit.get("passed") is not False if source_real_data_audit else True)
            and int(replay.get("weather_mismatch_trades") or 0) == 0
        ),
        "source_real_data_audit": source_real_data_audit,
    }
    executions_detail = replay.get("executions_detail") or []
    result = {
        **replay,
        "strategy_profile": strategy_profile,
        "bankroll_usd": bankroll_usd,
        "cash_usd": replay.get("ending_equity_usd"),
        "open_positions": 0,
        "source_snapshot_path": str(snapshot_path),
        "source_run_log_path": source.get("run_log_path") or str(snapshot_path),
        "source_strategy_profile": source.get("strategy_profile"),
        "source_scored_outcomes_detail_count": len(scored),
        "raw_recalibration": raw_recalibration,
        "source_runtime_limited": source.get("runtime_limited"),
        "source_session_count": source.get("session_count"),
        "source_markets_scored": source.get("markets_scored"),
        "source_outcomes_scored": source.get("outcomes_scored"),
        "source_settings": {
            "entry_hours_utc": source_settings.get("entry_hours_utc") if isinstance(source_settings, Mapping) else None,
            "entry_hours_match_live_forward": (
                source_settings.get("entry_hours_match_live_forward")
                if isinstance(source_settings, Mapping)
                else None
            ),
            "min_lead_days": source_settings.get("min_lead_days") if isinstance(source_settings, Mapping) else None,
            "max_lead_days": source_settings.get("max_lead_days") if isinstance(source_settings, Mapping) else None,
            "price_source": source_settings.get("price_source") if isinstance(source_settings, Mapping) else None,
            "market_source": source_settings.get("market_source") if isinstance(source_settings, Mapping) else None,
            "signal_settings": dict(source_signal_settings),
        },
        "score_calibration_diagnostics": _score_calibration_diagnostics(scored),
        "signal_filter_diagnostics": _signal_filter_diagnostics(scored),
        "signal_opportunity_diagnostics": _signal_opportunity_diagnostics(scored, settings),
        "selected_candidate_weather_validation": selected_candidate_weather_validation,
        "real_data_audit": replay_real_data_audit,
        "top_scored_outcomes": sorted(scored, key=lambda item: float(item.get("edge") or 0.0), reverse=True)[:20],
        "run_log_path": str(run_log_path),
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "cache_replay": {
            "uses_cached_scored_outcomes_only": True,
            "refetched_forecasts": False,
            "refetched_prices": False,
            "refetched_observations": False,
            "source_rows": len(scored),
        },
    }
    result["executions_detail"] = executions_detail
    result["top_trades"] = sorted(
        [_json_replay_execution_for_trade_diagnostics(execution) for execution in executions_detail],
        key=lambda item: abs(float(item.get("realized_pnl_usd") or 0.0)),
        reverse=True,
    )[:20]
    try:
        _write_json_log(run_log_path, result)
    except OSError as exc:
        result["run_log_write_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _recompute_scored_from_raw_model_probabilities(
    scored: list[dict[str, Any]],
    weights_file: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_weights, model_weights = load_calibration_weights(weights_file)
    probability_calibration = load_probability_calibration(weights_file)
    recomputed: list[dict[str, Any]] = []
    rows_with_raw = 0
    rows_recomputed = 0
    for row in scored:
        raw_probabilities = row.get("raw_model_probabilities") or {}
        if not isinstance(raw_probabilities, Mapping) or not raw_probabilities:
            recomputed.append(dict(row))
            continue
        rows_with_raw += 1
        calibrated_probabilities = _calibrated_probabilities(
            raw_probabilities,
            probability_calibration,
            bucket_count=1,
        )
        source_probabilities = _source_probability_views(calibrated_probabilities, model_weights)
        if not source_probabilities:
            recomputed.append(dict(row))
            continue
        values = list(source_probabilities.values())
        fair_value = _weighted_source_probability(source_probabilities, source_weights)
        raw_source_probabilities = _source_probability_views(raw_probabilities, model_weights)
        raw_values = list(raw_source_probabilities.values())
        raw_fair_value = _weighted_source_probability(raw_source_probabilities, source_weights) if raw_source_probabilities else None
        market_price = float(row.get("market_price") or row.get("market_probability") or 0.0)
        old_fair_value = _optional_float(row.get("fair_value")) or _optional_float(row.get("fair_value_probability")) or fair_value
        old_edge = _optional_float(row.get("edge")) or _optional_float(row.get("edge_after_buffer")) or (old_fair_value - market_price)
        preserved_buffer = max(0.0, old_fair_value - market_price - old_edge)
        edge = fair_value - market_price - preserved_buffer
        model_agreement = _source_agreement_above(source_probabilities, market_price + preserved_buffer)
        updated = dict(row)
        updated.update(
            {
                "fair_value": round(fair_value, 4),
                "fair_value_probability": round(fair_value, 4),
                "raw_fair_value": round(raw_fair_value, 4) if raw_fair_value is not None else None,
                "raw_fair_value_probability": round(raw_fair_value, 4) if raw_fair_value is not None else None,
                "edge": round(edge, 4),
                "edge_after_buffer": round(edge, 4),
                "model_probabilities": {
                    key: round(float(value), 4)
                    for key, value in sorted(calibrated_probabilities.items())
                },
                "model_count": len(source_probabilities),
                "model_agreement": round(model_agreement, 4),
                "probability_stdev": round(statistics.pstdev(values), 4) if len(values) >= 2 else 0.0,
                "raw_probability_stdev": round(statistics.pstdev(raw_values), 4) if len(raw_values) >= 2 else None,
                "recomputed_from_raw_model_probabilities": True,
            }
        )
        rows_recomputed += 1
        recomputed.append(updated)
    return recomputed, {
        "enabled": True,
        "weights_file": str(weights_file),
        "probability_calibration": {
            "active": probability_calibration.active,
            "center_shrink_alpha": probability_calibration.center_shrink_alpha,
            "high_cap": probability_calibration.high_cap,
            "low_floor": probability_calibration.low_floor,
            "single_bucket_only": probability_calibration.single_bucket_only,
            "tail_threshold": probability_calibration.tail_threshold,
            "tail_shrink_alpha": probability_calibration.tail_shrink_alpha,
        },
        "source_weights_loaded": len(source_weights),
        "model_weights_loaded": len(model_weights),
        "rows_with_raw_model_probabilities": rows_with_raw,
        "rows_recomputed": rows_recomputed,
    }


def _weighted_source_probability(source_probabilities: Mapping[str, float], source_weights: Mapping[str, float]) -> float:
    weighted_total = 0.0
    weight_sum = 0.0
    for source_name, probability in source_probabilities.items():
        weight = max(0.0, float(source_weights.get(source_name, 1.0)))
        weighted_total += float(probability) * weight
        weight_sum += weight
    return weighted_total / weight_sum if weight_sum else sum(source_probabilities.values()) / len(source_probabilities)


def _source_agreement_above(source_probabilities: Mapping[str, float], threshold: float) -> float:
    if not source_probabilities:
        return 0.0
    return sum(1 for probability in source_probabilities.values() if probability > threshold) / len(source_probabilities)


def _fetch_session_forecasts(
    forecast_client: SingleRunForecastClient,
    session_items: list[tuple[WeatherMarket, dict[str, Any], EntryPrice, dict[str, Any]]],
    session_time: datetime,
    *,
    availability_lag_hours: int,
    max_workers: int,
) -> tuple[
    dict[tuple[str, str], tuple[tuple[ForecastDistribution, ...], datetime]],
    dict[tuple[str, str], dict[str, Any]],
]:
    jobs: dict[tuple[str, str], tuple[CityConfig, date, str]] = {}
    for market, _raw, _entry, _settlement in session_items:
        if market.city is None or market.target_date is None:
            continue
        key = (market.city.display_name, market.target_date.isoformat())
        jobs.setdefault(key, (market.city, market.target_date, market.question))

    cache: dict[tuple[str, str], tuple[tuple[ForecastDistribution, ...], datetime]] = {}
    errors_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def fetch(key: tuple[str, str], city: CityConfig, target_date: date) -> tuple[tuple[str, str], tuple[ForecastDistribution, ...], datetime, list[str]]:
        distributions, run_time, errors = forecast_client.fetch_sources(
            city,
            target_date,
            session_time,
            availability_lag_hours=availability_lag_hours,
        )
        return key, distributions, run_time, errors

    if max_workers <= 1 or len(jobs) <= 1:
        results = [fetch(key, city, target_date) for key, (city, target_date, _question) in jobs.items()]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
            futures = [
                executor.submit(fetch, key, city, target_date)
                for key, (city, target_date, _question) in jobs.items()
            ]
            results = [future.result() for future in futures]

    for key, distributions, run_time, errors in results:
        if distributions:
            cache[key] = (distributions, run_time)
        if errors:
            errors_by_key[key] = {"errors": tuple(errors), "example_question": jobs[key][2]}
    return cache, errors_by_key


def _historical_forecast_workers() -> int:
    try:
        configured = int(os.environ.get("HISTORICAL_FORECAST_WORKERS", "2"))
    except ValueError:
        configured = 2
    return max(1, configured)


def _discover_raw_temperature_markets(gamma: PolymarketGammaClient, *, query: str, pages: int, limit_per_page: int) -> list[dict[str, Any]]:
    raw_markets: list[dict[str, Any]] = []
    seen = set()
    for page in range(1, pages + 1):
        payload = gamma.public_search(query=query, limit=limit_per_page, page=page)
        for raw in _iter_event_markets(payload.get("events") or []):
            key = raw.get("id") or raw.get("conditionId") or raw.get("slug")
            if key in seen:
                continue
            seen.add(key)
            if looks_like_temperature_market(raw):
                raw_markets.append(raw)
    return raw_markets


def _discover_telonex_raw_temperature_markets(
    telonex: Optional[TelonexClient],
    *,
    query: str,
    max_candidates: int,
    min_end_date: Optional[date] = None,
    max_end_date: Optional[date] = None,
    market_selection: str = "end_date_asc",
) -> list[dict[str, Any]]:
    if telonex is None:
        raise TelonexConfigurationError("Telonex market source selected but no Telonex client is configured")
    dataset_path = telonex.download_dataset_parquet(exchange="polymarket", dataset="markets")
    records = _load_telonex_market_records(
        dataset_path,
        query=query,
        max_candidates=max_candidates,
        min_end_date=min_end_date,
        max_end_date=max_end_date,
        market_selection=market_selection,
    )
    raw_markets = []
    seen = set()
    for record in records:
        raw = _telonex_market_record_to_raw(record)
        if raw is None or not looks_like_temperature_market(raw):
            continue
        key = raw.get("id") or raw.get("conditionId") or raw.get("slug")
        if key in seen:
            continue
        seen.add(key)
        raw_markets.append(raw)
    return raw_markets


def _load_telonex_market_records(
    path: Path,
    *,
    query: str,
    max_candidates: int,
    min_end_date: Optional[date] = None,
    max_end_date: Optional[date] = None,
    market_selection: str = "end_date_asc",
) -> list[Mapping[str, Any]]:
    limit = max(1, max_candidates)
    if market_selection not in {"end_date_asc", "end_date_desc", "month_balanced"}:
        raise ValueError(f"Unsupported market_selection={market_selection!r}")
    min_end_us = _date_to_epoch_us(min_end_date) if min_end_date else None
    max_end_us = _date_to_epoch_us(max_end_date + timedelta(days=1)) - 1 if max_end_date else None
    try:
        import polars as pl  # type: ignore

        text_columns = ("question", "description", "slug", "event_slug", "event_title")
        text_filter = pl.any_horizontal(
            [
                pl.col(column).fill_null("").cast(pl.String).str.to_lowercase().str.contains("highest temperature")
                for column in text_columns
            ]
        )
        query_filter = pl.lit(True)
        cleaned_query = query.strip().lower()
        if cleaned_query and cleaned_query != "highest temperature":
            query_filter = pl.any_horizontal(
                [
                    pl.col(column).fill_null("").cast(pl.String).str.to_lowercase().str.contains(cleaned_query, literal=True)
                    for column in text_columns
                ]
            )
        scan = (
            pl.scan_parquet(path)
            .filter(text_filter & query_filter)
            .filter(pl.col("status").fill_null("") == "resolved")
            .filter(pl.col("quotes_from").fill_null("") >= "2026-01-19")
            .filter(pl.col("quotes_to").fill_null("") != "")
        )
        if min_end_us is not None:
            scan = scan.filter(pl.col("end_date_us") >= min_end_us)
        if max_end_us is not None:
            scan = scan.filter(pl.col("end_date_us") <= max_end_us)
        rows = [dict(row) for row in scan.sort("end_date_us").collect().to_dicts()]
        return _select_telonex_market_records(rows, limit=limit, market_selection=market_selection)
    except ModuleNotFoundError:
        records = []
        for record in _load_parquet_records(path):
            raw = _telonex_market_record_to_raw(record)
            if raw is None or not looks_like_temperature_market(raw):
                continue
            if str(record.get("status") or "") != "resolved":
                continue
            if not record.get("quotes_from") or str(record.get("quotes_from")) < "2026-01-19" or not record.get("quotes_to"):
                continue
            end_date_us = _to_int(record.get("end_date_us"))
            if min_end_us is not None and (end_date_us is None or end_date_us < min_end_us):
                continue
            if max_end_us is not None and (end_date_us is None or end_date_us > max_end_us):
                continue
            records.append(record)
        records.sort(key=lambda record: _to_int(record.get("end_date_us")) or 0)
        return _select_telonex_market_records(records, limit=limit, market_selection=market_selection)


def _select_telonex_market_records(
    records: list[Mapping[str, Any]],
    *,
    limit: int,
    market_selection: str,
) -> list[Mapping[str, Any]]:
    if market_selection == "end_date_asc":
        return records[:limit]
    if market_selection == "end_date_desc":
        return list(reversed(records))[:limit]
    by_month: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        month = _end_month(record) or "unknown"
        by_month.setdefault(month, []).append(record)
    selected: list[Mapping[str, Any]] = []
    months = sorted(by_month)
    while len(selected) < limit and months:
        remaining_months = []
        for month in months:
            bucket = by_month[month]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) >= limit:
                    break
            if bucket:
                remaining_months.append(month)
        months = remaining_months
    return selected


def _end_month(record: Mapping[str, Any]) -> Optional[str]:
    end_date_us = _to_int(record.get("end_date_us"))
    if end_date_us is None:
        return None
    return datetime.fromtimestamp(end_date_us / 1_000_000, tz=timezone.utc).strftime("%Y-%m")


def _date_to_epoch_us(value: date) -> int:
    return int(datetime.combine(value, datetime_time(0, 0), tzinfo=timezone.utc).timestamp() * 1_000_000)


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _telonex_market_record_to_raw(record: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    outcome_0 = record.get("outcome_0")
    outcome_1 = record.get("outcome_1")
    asset_id_0 = record.get("asset_id_0")
    asset_id_1 = record.get("asset_id_1")
    if outcome_0 is None or outcome_1 is None or asset_id_0 is None or asset_id_1 is None:
        return None
    outcomes = [str(outcome_0), str(outcome_1)]
    token_ids = [str(asset_id_0), str(asset_id_1)]
    return {
        "id": str(record.get("market_id") or record.get("slug") or ""),
        "conditionId": str(record.get("market_id") or ""),
        "slug": str(record.get("slug") or ""),
        "eventSlug": str(record.get("event_slug") or ""),
        "eventTitle": str(record.get("event_title") or ""),
        "question": str(record.get("question") or ""),
        "description": str(record.get("description") or ""),
        "rules": str(record.get("description") or ""),
        "resolutionSource": str(record.get("resolution_source") or ""),
        "outcomes": json.dumps(outcomes),
        "clobTokenIds": json.dumps(token_ids),
        "outcomePrices": json.dumps(_telonex_settled_prices(record.get("result_id"))),
        "status": str(record.get("status") or ""),
        "startDate": _iso_from_epoch_us(record.get("start_date_us")),
        "endDate": _iso_from_epoch_us(record.get("end_date_us")),
        "createdAt": _iso_from_epoch_us(record.get("created_at_us")),
        "closedTime": _iso_from_epoch_us(record.get("settled_at_us")),
        "acceptingOrdersTimestamp": _iso_from_epoch_us(record.get("prepared_at_us")),
        "telonex_quotes_from": record.get("quotes_from"),
        "telonex_quotes_to": record.get("quotes_to"),
    }


def _telonex_settled_prices(result_id: Any) -> list[str]:
    try:
        winner = int(str(result_id))
    except (TypeError, ValueError):
        return ["0.5", "0.5"]
    if winner == 0:
        return ["1", "0"]
    if winner == 1:
        return ["0", "1"]
    return ["0.5", "0.5"]


def _iso_from_epoch_us(value: Any) -> Optional[str]:
    numeric = _numeric(value)
    if numeric <= 0:
        return None
    try:
        return datetime.fromtimestamp(numeric / 1_000_000, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _cached_price_history(
    cache: dict[str, list[PriceHistoryPoint]],
    clob: PolymarketClobClient,
    token_id: str,
    *,
    start_ts: Optional[int],
    end_ts: Optional[int],
    price_source: str = "clob",
    telonex: Optional[TelonexClient] = None,
    market_slug: Optional[str] = None,
    outcome: str = "Yes",
) -> list[PriceHistoryPoint]:
    cache_key = f"{price_source}:{token_id}:{market_slug or ''}:{outcome}:{start_ts}:{end_ts}"
    if cache_key in cache:
        return cache[cache_key]
    if price_source == "telonex":
        if telonex is None:
            raise TelonexConfigurationError("Telonex price source selected but no Telonex client is configured")
        history = telonex.fetch_quote_price_history(
            slug=market_slug or "",
            outcome=outcome,
            start_ts=start_ts,
            end_ts=end_ts,
            token_id=token_id,
        )
    else:
        history = clob.fetch_price_history(
            token_id,
            interval="max",
            fidelity=60,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    cache[cache_key] = history
    return history


def _historical_price_source(price_source: str, cache_dir: str | Path, *, hard_timeout_seconds: float = 30) -> tuple[str, Optional[TelonexClient]]:
    normalized = price_source.strip().lower()
    if normalized not in {"telonex", "clob", "auto"}:
        raise ValueError(f"Unsupported historical price source: {price_source}")
    if normalized == "clob":
        return "clob", None
    try:
        client = TelonexClient(cache_dir=Path(cache_dir) / "telonex", hard_timeout_seconds=hard_timeout_seconds)
    except TelonexConfigurationError:
        if normalized == "auto":
            return "clob", None
        raise
    return "telonex", client


def _historical_market_source(
    market_source: str,
    cache_dir: str | Path,
    telonex: Optional[TelonexClient],
    *,
    hard_timeout_seconds: float = 30,
) -> tuple[str, Optional[TelonexClient]]:
    normalized = market_source.strip().lower()
    if normalized not in {"telonex", "gamma"}:
        raise ValueError(f"Unsupported historical market source: {market_source}")
    if normalized == "gamma":
        return "gamma", telonex
    if telonex is not None:
        return "telonex", telonex
    return "telonex", TelonexClient(cache_dir=Path(cache_dir) / "telonex", hard_timeout_seconds=hard_timeout_seconds)


def _price_source_description(price_source: str, side: str) -> str:
    if price_source == "telonex":
        return f"Telonex Polymarket quotes daily Parquet filtered by {side} asset_id/token when available, otherwise market slug and outcome"
    if side == "NO":
        return "CLOB prices-history for explicit NO clobTokenIds when allow_no_side_entries is enabled"
    return "CLOB prices-history with market-lifetime startTs/endTs bounds when available"


def _iter_event_markets(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for event in events:
        markets = event.get("markets")
        if not isinstance(markets, list):
            if isinstance(event, dict):
                yield event
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            merged = dict(market)
            merged.setdefault("eventSlug", event.get("slug"))
            merged.setdefault("eventTitle", event.get("title"))
            merged.setdefault("eventSubtitle", event.get("subtitle"))
            merged.setdefault("eventDescription", event.get("description"))
            yield merged


def _parse_historical_markets(raw_markets: Iterable[dict[str, Any]], *, max_markets: int, min_volume_usd: float) -> list[tuple[WeatherMarket, dict[str, Any]]]:
    parsed: list[tuple[WeatherMarket, dict[str, Any]]] = []
    current = datetime.now(timezone.utc).date()
    for raw in raw_markets:
        if len(parsed) >= max_markets:
            break
        if _numeric(raw.get("volumeNum") or raw.get("volume") or 0.0) < min_volume_usd:
            continue
        market = parse_weather_market(raw, today=_market_parse_today(raw))
        if market is None or market.target_date is None:
            continue
        if market.target_date >= current:
            continue
        parsed.append((market, raw))
    return parsed


def _market_parse_today(raw: Mapping[str, Any]) -> Optional[date]:
    for key in ("endDate", "closedTime", "startDate", "createdAt"):
        timestamp = _parse_dt(raw.get(key))
        if timestamp is not None:
            return date(timestamp.year, 1, 1)
    return None


def _single_bucket(market: WeatherMarket) -> Optional[TemperatureBucket]:
    return market.buckets[0] if len(market.buckets) == 1 else None


def _binary_no_token(raw: Mapping[str, Any]) -> Optional[str]:
    try:
        outcomes = [str(item).lower() for item in parse_jsonish_list(raw.get("outcomes"))]
        token_ids = [str(item) for item in parse_jsonish_list(raw.get("clobTokenIds") or raw.get("clob_token_ids"))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if "no" not in outcomes:
        return None
    no_index = outcomes.index("no")
    if no_index >= len(token_ids):
        return None
    return token_ids[no_index]


def _invert_binary_outcome(value: object) -> Optional[int]:
    if value not in (0, 1):
        return None
    return 1 - int(value)


def _invert_binary_scored_outcome(outcome: ScoredOutcome, settings: SignalSettings) -> ScoredOutcome:
    model_probabilities = {
        model_name: 1.0 - max(0.0, min(1.0, float(probability)))
        for model_name, probability in outcome.model_probabilities.items()
    }
    raw_model_probabilities = (
        {
            model_name: 1.0 - max(0.0, min(1.0, float(probability)))
            for model_name, probability in outcome.raw_model_probabilities.items()
        }
        if outcome.raw_model_probabilities
        else None
    )
    fair_value = 1.0 - outcome.fair_value
    raw_fair_value = 1.0 - outcome.raw_fair_value if outcome.raw_fair_value is not None else None
    buffer = _price_adjusted_uncertainty_buffer(outcome.market_price, settings)
    agreement = ConsensusValue(
        bucket_label=outcome.bucket_label,
        fair_value=fair_value,
        model_probabilities=model_probabilities,
        model_count=outcome.model_count,
        probability_stdev=outcome.probability_stdev,
    ).agreement_above(outcome.market_price, buffer)
    return replace(
        outcome,
        question=f"NO: {outcome.question}",
        bucket_label=f"NO: {outcome.bucket_label}",
        fair_value=round(fair_value, 4),
        edge=round(fair_value - outcome.market_price - buffer, 4),
        model_agreement=round(agreement, 4),
        model_probabilities=model_probabilities,
        raw_fair_value=round(raw_fair_value, 4) if raw_fair_value is not None else None,
        raw_model_probabilities=raw_model_probabilities,
        observed_outcome=_invert_binary_outcome(outcome.observed_outcome),
    )


def _resolve_market_outcome(
    market: WeatherMarket,
    raw: dict[str, Any],
    bucket: TemperatureBucket,
    observation_client: ObservedHighClient,
    observation_cache: Optional[dict[tuple[str, ...], Optional[ObservedHigh]]] = None,
    *,
    fetch_weather_crosscheck: bool = True,
) -> dict[str, Any]:
    polymarket_payout = _polymarket_yes_payout(raw)
    observed_high_f = None
    weather_outcome = None
    weather_ambiguous = False
    settlement_source = None
    if fetch_weather_crosscheck and market.city is not None and market.target_date is not None:
        observation_city = _market_resolution_city(market.city, raw)
        station = _resolution_station(raw)
        cache_key = (
            observation_city.display_name,
            station or observation_city.metar_station or observation_city.nws_station or "",
            market.target_date.isoformat(),
        )
        cache_hit = observation_cache is not None and cache_key in observation_cache
        if cache_hit:
            observed = observation_cache[cache_key]
        else:
            try:
                now = datetime.combine(market.target_date + timedelta(days=3), datetime_time(12, 0), tzinfo=timezone.utc)
                observed = None
                if station is not None and hasattr(observation_client, "fetch_historical_station_high"):
                    observed = observation_client.fetch_historical_station_high(observation_city, station, market.target_date, now=now)
                if observed is None:
                    observed = observation_client.fetch_observed_high(observation_city, market.target_date, now=now)
            except (RuntimeError, ValueError):
                observed = None
            if station is not None and observed is not None and not _is_station_actual_observation(observed.source):
                observed = None
            if observation_cache is not None and (observed is not None or station is None):
                observation_cache[cache_key] = observed
        if observed is not None:
            observed_high_f = round(observed.max_temperature_f, 2)
            settlement_source = observed.source
            candidate_outcome = observed_outcome_for_bucket(bucket, observed.max_temperature_f, True)
            if _is_ambiguous_resolution_crosscheck(bucket, observed) and (
                polymarket_payout is None or candidate_outcome != polymarket_payout
            ):
                weather_ambiguous = True
                settlement_source = f"{observed.source}_ambiguous_resolution"
            else:
                weather_outcome = candidate_outcome
    payout = polymarket_payout if polymarket_payout is not None else weather_outcome
    return {
        "payout": payout,
        "polymarket_payout": polymarket_payout,
        "weather_outcome": weather_outcome,
        "weather_ambiguous": weather_ambiguous,
        "observed_high_f": observed_high_f,
        "settlement_source": settlement_source,
    }


def _is_station_actual_observation(source: str) -> bool:
    return source.startswith(("historical_metar_", "metar_", "nws_station_"))


def _is_ambiguous_resolution_crosscheck(bucket: TemperatureBucket, observed: ObservedHigh) -> bool:
    if bucket.resolution_precision is None or bucket.resolution_precision <= 0:
        return False
    if not _is_station_actual_observation(observed.source):
        return False
    if bucket.lower_f is not None and bucket.upper_f is not None:
        return True
    unit = (bucket.resolution_unit or "F").upper()
    observed_value = _temperature_to_unit(observed.max_temperature_f, unit)
    precision = float(bucket.resolution_precision)
    boundaries = []
    if bucket.lower_f is not None:
        boundaries.append(_temperature_to_unit(bucket.lower_f, unit))
    if bucket.upper_f is not None:
        boundaries.append(_temperature_to_unit(bucket.upper_f, unit))
    return any(abs(observed_value - boundary) <= precision + 1e-6 for boundary in boundaries)


def _temperature_to_unit(value_f: float, unit: str) -> float:
    if unit.upper() == "C":
        return (value_f - 32.0) * 5.0 / 9.0
    return value_f


def _resolution_example(
    market: WeatherMarket,
    bucket: TemperatureBucket,
    raw: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "question": market.question,
        "city": market.city.display_name if market.city else None,
        "target_date": market.target_date.isoformat() if market.target_date else None,
        "bucket": bucket.label,
        "bucket_lower_f": bucket.lower_f,
        "bucket_upper_f": bucket.upper_f,
        "resolution_unit": bucket.resolution_unit,
        "resolution_precision": bucket.resolution_precision,
        "polymarket_payout": resolution.get("polymarket_payout"),
        "weather_outcome": resolution.get("weather_outcome"),
        "weather_ambiguous": resolution.get("weather_ambiguous"),
        "observed_high_f": resolution.get("observed_high_f"),
        "settlement_source": resolution.get("settlement_source"),
        "resolution_source": raw.get("resolutionSource"),
    }


def _market_resolution_city(city: CityConfig, raw: dict[str, Any]) -> CityConfig:
    station = _resolution_station(raw)
    if station is None:
        return city
    return city_with_station_coordinates(city, station)


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
            return match.group(1)
    return None


def _polymarket_yes_payout(raw: dict[str, Any]) -> Optional[int]:
    try:
        outcomes = [str(item).lower() for item in parse_jsonish_list(raw.get("outcomes"))]
        prices = parse_jsonish_list(raw.get("outcomePrices") or raw.get("outcome_prices"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if "yes" not in outcomes:
        return None
    index = outcomes.index("yes")
    if index >= len(prices):
        return None
    try:
        value = float(prices[index])
    except (TypeError, ValueError):
        return None
    if value <= 0.001:
        return 0
    if value >= 0.999:
        return 1
    return None


def _candidate_entry_times(market: WeatherMarket, entry_hours_utc: tuple[int, ...], min_lead_days: int, max_lead_days: int) -> list[datetime]:
    if market.city is None or market.target_date is None:
        return []
    candidates = []
    start_date = market.target_date - timedelta(days=max_lead_days + 1)
    end_date = market.target_date
    current = start_date
    while current <= end_date:
        for hour in entry_hours_utc:
            candidate = datetime.combine(current, datetime_time(hour, 0), tzinfo=timezone.utc)
            local_date = candidate.astimezone(_zoneinfo_or_utc(market.city)).date()
            lead_days = (market.target_date - local_date).days
            if min_lead_days <= lead_days <= max_lead_days:
                candidates.append(candidate)
        current += timedelta(days=1)
    return sorted(set(candidates))


def _candidate_replay_times(
    market: WeatherMarket,
    entry_hours_utc: tuple[int, ...],
    min_lead_days: int,
    max_lead_days: int,
) -> list[tuple[datetime, bool]]:
    entry_times = _candidate_entry_times(market, entry_hours_utc, min_lead_days, max_lead_days)
    replay_times = {entry_time: False for entry_time in entry_times}
    for maintenance_time in _candidate_target_day_maintenance_times(market, entry_hours_utc):
        replay_times.setdefault(maintenance_time, True)
    return sorted(replay_times.items())


def _candidate_target_day_maintenance_times(market: WeatherMarket, entry_hours_utc: tuple[int, ...]) -> list[datetime]:
    if market.city is None or market.target_date is None:
        return []
    candidates = []
    timezone_info = _zoneinfo_or_utc(market.city)
    current = market.target_date - timedelta(days=1)
    end_date = market.target_date + timedelta(days=1)
    while current <= end_date:
        for hour in entry_hours_utc:
            candidate = datetime.combine(current, datetime_time(hour, 0), tzinfo=timezone.utc)
            if candidate.astimezone(timezone_info).date() == market.target_date:
                candidates.append(candidate)
        current += timedelta(days=1)
    return sorted(set(candidates))


def _historical_observed_high_for_session(
    market: WeatherMarket,
    raw: dict[str, Any],
    observation_client: ObservedHighClient,
    observation_cache: dict[tuple[str, str, str, str], Optional[ObservedHigh]],
    session_time: datetime,
) -> Optional[ObservedHigh]:
    if market.city is None or market.target_date is None:
        return None
    local_session_date = session_time.astimezone(_zoneinfo_or_utc(market.city)).date()
    if market.target_date != local_session_date:
        return None
    station = _resolution_station(raw) or market.city.metar_station or market.city.nws_station
    if station is None:
        return None
    cache_key = (
        market.city.display_name,
        station,
        market.target_date.isoformat(),
        session_time.isoformat(),
    )
    if cache_key in observation_cache:
        return observation_cache[cache_key]
    try:
        observed = observation_client.fetch_partial_historical_high(
            market.city,
            station,
            market.target_date,
            now=session_time,
        )
    except (RuntimeError, ValueError):
        observed = None
    observation_cache[cache_key] = observed
    return observed


def _market_active_at(raw: dict[str, Any], decision_time: datetime) -> bool:
    created = _parse_dt(raw.get("acceptingOrdersTimestamp") or raw.get("startDate") or raw.get("createdAt") or raw.get("creationDate"))
    closed = _parse_dt(raw.get("closedTime") or raw.get("endDate"))
    if created is not None and decision_time < created:
        return False
    if closed is not None and decision_time >= closed:
        return False
    return True


def _price_history_bounds(raw: dict[str, Any], *, padding_hours: int = 12) -> tuple[Optional[int], Optional[int]]:
    telonex_from = _parse_date(raw.get("telonex_quotes_from"))
    telonex_to = _parse_date(raw.get("telonex_quotes_to"))
    if telonex_from is not None and telonex_to is not None:
        opened = datetime.combine(telonex_from, datetime_time(0, 0), tzinfo=timezone.utc)
        closed = datetime.combine(telonex_to + timedelta(days=1), datetime_time(0, 0), tzinfo=timezone.utc)
        return int(opened.timestamp()), int(closed.timestamp())
    opened_candidates = (
        _parse_dt(raw.get("acceptingOrdersTimestamp")),
        _parse_dt(raw.get("startDate")),
        _parse_dt(raw.get("createdAt")),
        _parse_dt(raw.get("creationDate")),
    )
    closed_candidates = (
        _parse_dt(raw.get("closedTime")),
        _parse_dt(raw.get("endDate")),
        _parse_dt(raw.get("umaEndDate")),
        _parse_dt(raw.get("updatedAt")),
    )
    opened = min((value for value in opened_candidates if value is not None), default=None)
    closed = max((value for value in closed_candidates if value is not None), default=None)
    if opened is not None:
        opened = opened.astimezone(timezone.utc) - timedelta(hours=padding_hours)
    if closed is not None:
        closed = closed.astimezone(timezone.utc) + timedelta(hours=padding_hours)
    if opened is not None and closed is not None and closed <= opened:
        closed = opened + timedelta(days=14)
    start_ts = int(opened.timestamp()) if opened is not None else None
    end_ts = int(closed.timestamp()) if closed is not None else None
    return start_ts, end_ts


def _price_history_bounds_for_replay_times(
    replay_times: Iterable[tuple[datetime, bool]],
    *,
    max_staleness_minutes: int,
) -> tuple[Optional[int], Optional[int]]:
    timestamps = sorted(
        (timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
        for timestamp, _maintenance_only in replay_times
    )
    if not timestamps:
        return None, None
    start = timestamps[0] - timedelta(minutes=max(0, max_staleness_minutes))
    end = timestamps[-1] + timedelta(seconds=1)
    return int(start.timestamp()), int(end.timestamp())


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _rebalance_session(
    scored: list[ScoredOutcome],
    selected_tokens: set[str],
    metadata: Mapping[str, Mapping[str, Any]],
    positions: dict[str, HistoricalPosition],
    executions: list[dict[str, Any]],
    cash_ref: dict[str, float],
    *,
    bankroll_usd: float,
    kelly_fraction: float,
    compound_kelly_sizing: bool,
    max_position_usd: float,
    max_position_fraction: Optional[float],
    kelly_market_blend: float,
    edge_position_full_cap_edge: float,
    edge_position_min_multiplier: float,
    min_trade_usd: float,
    settings: SignalSettings,
    max_new_exposure_usd_per_run: Optional[float] = None,
    max_new_exposure_fraction_per_run: Optional[float] = None,
    new_exposure_target_positions_per_run: Optional[float] = None,
    kelly_sizing_bankroll_fraction_per_run: Optional[float] = None,
) -> None:
    sizing_base = _historical_portfolio_equity(cash_ref["cash"], positions) if compound_kelly_sizing else bankroll_usd
    buy_budget_remaining = _new_exposure_budget_usd(
        sizing_base,
        max_new_exposure_usd_per_run,
        max_new_exposure_fraction_per_run,
    )
    per_buy_budget = _new_exposure_per_buy_budget_usd(
        buy_budget_remaining,
        new_exposure_target_positions_per_run,
    )
    for outcome in sorted(scored, key=lambda item: item.edge, reverse=True):
        if not outcome.token_id:
            continue
        token = outcome.token_id
        current = positions.get(token)
        token_metadata = metadata.get(token, {})
        if current is None and token_metadata.get("maintenance_only"):
            continue
        current_shares = current.shares if current else 0.0
        current_notional = current_shares * outcome.market_price
        hold_eligible = current is not None and hold_filter_reason(outcome, settings) is None
        target_notional = 0.0
        if token in selected_tokens:
            sizing_bankroll = _historical_portfolio_equity(cash_ref["cash"], positions) if compound_kelly_sizing else bankroll_usd
            sizing_bankroll = _kelly_sizing_bankroll_usd(sizing_bankroll, kelly_sizing_bankroll_fraction_per_run)
            target_notional = _kelly_target_notional(
                outcome,
                sizing_bankroll,
                kelly_fraction,
                max_position_usd,
                max_position_fraction,
                kelly_market_blend,
                edge_position_full_cap_edge,
                edge_position_min_multiplier,
            )
            if hold_eligible and settings.preserve_valid_holds:
                target_notional = max(target_notional, current_notional)
        elif hold_eligible:
            target_notional = current_notional
        delta = target_notional - current_notional
        if delta > 0 and delta >= min_trade_usd:
            if buy_budget_remaining is not None:
                delta = min(delta, buy_budget_remaining)
            if per_buy_budget is not None:
                delta = min(delta, per_buy_budget)
            if buy_budget_remaining is not None or per_buy_budget is not None:
                if delta < min_trade_usd:
                    continue
            notional = min(delta, cash_ref["cash"])
            if notional < min_trade_usd:
                continue
            shares = notional / outcome.market_price
            cash_ref["cash"] -= notional
            if buy_budget_remaining is not None:
                buy_budget_remaining = max(0.0, buy_budget_remaining - notional)
            _upsert_historical_position(positions, outcome, shares, notional, token_metadata)
            executions.append(_execution_json("BUY", outcome, shares, outcome.market_price, notional, 0.0, token_metadata))
        elif delta < 0 and current is not None and abs(delta) >= min_trade_usd:
            shares = min(current.shares, abs(delta) / outcome.market_price)
            _sell_historical_position(positions, executions, cash_ref, outcome, shares, token_metadata)
        elif current is not None:
            current.last_price = _exit_price(outcome, token_metadata)


def _historical_portfolio_equity(cash: float, positions: Mapping[str, HistoricalPosition]) -> float:
    marked = sum(position.shares * position.last_price for position in positions.values())
    return max(0.0, cash + marked)


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
    return max(0.0, min(caps))


def _new_exposure_per_buy_budget_usd(
    run_budget_usd: Optional[float],
    target_positions_per_run: Optional[float],
) -> Optional[float]:
    if run_budget_usd is None:
        return None
    if target_positions_per_run is None or target_positions_per_run <= 0:
        return None
    return max(0.0, run_budget_usd / float(target_positions_per_run))


def _kelly_sizing_bankroll_usd(
    bankroll_usd: float,
    fraction_per_run: Optional[float],
) -> float:
    if fraction_per_run is None or fraction_per_run <= 0:
        return bankroll_usd
    return max(0.0, bankroll_usd * float(fraction_per_run))


def _upsert_historical_position(positions: dict[str, HistoricalPosition], outcome: ScoredOutcome, shares: float, cost: float, metadata: Mapping[str, Any]) -> None:
    assert outcome.token_id is not None
    current = positions.get(outcome.token_id)
    if current is None:
        positions[outcome.token_id] = HistoricalPosition(
            token_id=outcome.token_id,
            market_id=outcome.market_id,
            question=outcome.question,
            bucket_label=outcome.bucket_label,
            city=outcome.city,
            target_date=outcome.target_date,
            shares=shares,
            cost_basis=cost,
            last_price=outcome.market_price,
            payout=metadata.get("payout"),
            weather_outcome=metadata.get("weather_outcome"),
            observed_high_f=metadata.get("observed_high_f"),
            settlement_source=metadata.get("settlement_source"),
            side=metadata.get("side"),
        )
        return
    current.shares += shares
    current.cost_basis += cost
    current.last_price = outcome.market_price


def _sell_historical_position(
    positions: dict[str, HistoricalPosition],
    executions: list[dict[str, Any]],
    cash_ref: dict[str, float],
    outcome: ScoredOutcome,
    shares: float,
    metadata: Mapping[str, Any],
) -> None:
    assert outcome.token_id is not None
    current = positions[outcome.token_id]
    shares = min(shares, current.shares)
    cost_reduction = current.cost_basis * (shares / current.shares) if current.shares else 0.0
    price = _exit_price(outcome, metadata)
    notional = shares * price
    realized = notional - cost_reduction
    current.shares -= shares
    current.cost_basis -= cost_reduction
    current.last_price = price
    cash_ref["cash"] += notional
    executions.append(_execution_json("SELL", outcome, shares, price, notional, realized, metadata))
    if current.shares <= 1e-9:
        del positions[outcome.token_id]


def _exit_price(outcome: ScoredOutcome, metadata: Mapping[str, Any]) -> float:
    try:
        return max(0.001, min(0.999, float(metadata.get("exit_price"))))
    except (TypeError, ValueError):
        return outcome.market_price


def _settle_due_positions(positions: dict[str, HistoricalPosition], executions: list[dict[str, Any]], session_time: datetime, cash_ref: dict[str, float]) -> None:
    for token, position in list(positions.items()):
        if position.target_date is None:
            continue
        city = _city_from_display_name(position.city)
        if city is None:
            continue
        if position.target_date >= session_time.astimezone(_zoneinfo_or_utc(city)).date():
            continue
        _settle_position(positions, executions, cash_ref, token, position, executed_at=session_time)


def _settle_all_positions(positions: dict[str, HistoricalPosition], executions: list[dict[str, Any]], cash_ref: dict[str, float]) -> None:
    for token, position in list(positions.items()):
        _settle_position(positions, executions, cash_ref, token, position, executed_at=_historical_settlement_time(position))


def _finalize_positions_for_result(
    positions: dict[str, HistoricalPosition],
    executions: list[dict[str, Any]],
    cash: float,
    *,
    runtime_limited: bool,
) -> tuple[float, float]:
    if not runtime_limited:
        cash_ref = {"cash": cash}
        _settle_all_positions(positions, executions, cash_ref)
        cash = cash_ref["cash"]
    open_value = sum(position.shares * position.last_price for position in positions.values())
    return cash, open_value


def _settle_position(
    positions: dict[str, HistoricalPosition],
    executions: list[dict[str, Any]],
    cash_ref: dict[str, float],
    token: str,
    position: HistoricalPosition,
    *,
    executed_at: datetime,
) -> None:
    if position.payout is None:
        return
    notional = position.shares * float(position.payout)
    realized = notional - position.cost_basis
    cash_ref["cash"] += notional
    executions.append(
        {
            "action": "SETTLE",
            "executed_at": executed_at.isoformat(),
            "token_id": token,
            "market_id": position.market_id,
            "question": position.question,
            "bucket": position.bucket_label,
            "city": position.city,
            "target_date": position.target_date.isoformat() if position.target_date else None,
            "shares": round(position.shares, 6),
            "price": float(position.payout),
            "notional_usd": round(notional, 4),
            "realized_pnl_usd": round(realized, 4),
            "polymarket_payout": position.payout,
            "weather_outcome": position.weather_outcome,
            "observed_high_f": position.observed_high_f,
            "settlement_source": position.settlement_source,
            "side": position.side,
        }
    )
    del positions[token]


def _historical_settlement_time(position: HistoricalPosition) -> datetime:
    if position.target_date is None:
        return datetime.now(timezone.utc)
    city = _city_from_display_name(position.city)
    if city is None:
        return datetime.combine(position.target_date + timedelta(days=1), datetime_time(12, 0), tzinfo=timezone.utc)
    local_zone = _zoneinfo_or_utc(city)
    local_noon = datetime.combine(position.target_date + timedelta(days=1), datetime_time(12, 0), tzinfo=local_zone)
    return local_noon.astimezone(timezone.utc)


def _execution_json(action: str, outcome: ScoredOutcome, shares: float, price: float, notional: float, realized: float, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": action,
        "executed_at": outcome.generated_at.isoformat(),
        "token_id": outcome.token_id,
        "market_id": outcome.market_id,
        "question": outcome.question,
        "bucket": outcome.bucket_label,
        "city": outcome.city,
        "target_date": outcome.target_date.isoformat() if outcome.target_date else None,
        "shares": round(shares, 6),
        "price": round(price, 4),
        "notional_usd": round(notional, 4),
        "realized_pnl_usd": round(realized, 4),
        "fair_value": outcome.fair_value,
        "edge": outcome.edge,
        "model_agreement": outcome.model_agreement,
        "bucket_lower_f": outcome.bucket_lower_f,
        "bucket_upper_f": outcome.bucket_upper_f,
        "bucket_width_f": outcome.bucket_width_f,
        "bucket_shape": _bucket_shape(outcome.bucket_lower_f, outcome.bucket_upper_f),
        **dict(metadata),
    }


def _kelly_target_notional(
    outcome: ScoredOutcome,
    bankroll_usd: float,
    kelly_fraction: float,
    max_position_usd: float,
    max_position_fraction: Optional[float] = None,
    kelly_market_blend: float = 0.0,
    edge_position_full_cap_edge: float = 0.0,
    edge_position_min_multiplier: float = 0.35,
) -> float:
    price = max(0.0001, min(0.9999, outcome.market_price))
    sizing_fair_value = _blend_probability_with_market(outcome.fair_value, price, kelly_market_blend)
    raw_fraction = max(0.0, (sizing_fair_value - price) / max(0.0001, 1.0 - price))
    agreement_scaled = raw_fraction * max(0.0, min(1.0, outcome.model_agreement))
    return min(
        _effective_max_position_usd(
            bankroll_usd,
            max_position_usd,
            max_position_fraction,
            edge=outcome.edge,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
        ),
        bankroll_usd * kelly_fraction * agreement_scaled,
    )


def _scored_to_json(outcome: ScoredOutcome, metadata: Mapping[str, Any], settings: SignalSettings) -> dict[str, Any]:
    filter_reason = signal_filter_reason(outcome, settings)
    return {
        "generated_at": outcome.generated_at.isoformat(),
        "question": outcome.question,
        "city": outcome.city,
        "target_date": outcome.target_date.isoformat() if outcome.target_date else None,
        "bucket": outcome.bucket_label,
        "token_id": outcome.token_id,
        "fair_value": outcome.fair_value,
        "market_price": outcome.market_price,
        "edge": outcome.edge,
        "model_count": outcome.model_count,
        "model_agreement": outcome.model_agreement,
        "probability_stdev": outcome.probability_stdev,
        "raw_fair_value": outcome.raw_fair_value,
        "raw_probability_stdev": outcome.raw_probability_stdev,
        "bucket_lower_f": outcome.bucket_lower_f,
        "bucket_upper_f": outcome.bucket_upper_f,
        "bucket_width_f": outcome.bucket_width_f,
        "bucket_shape": _bucket_shape(outcome.bucket_lower_f, outcome.bucket_upper_f),
        "resolution_unit": outcome.resolution_unit,
        "resolution_precision": outcome.resolution_precision,
        "timing_entry_eligible": outcome.entry_eligible,
        "entry_eligible": outcome.entry_eligible,
        "entry_filter_reason": outcome.entry_filter_reason,
        "model_observed_high_f": outcome.observed_high_f,
        "model_observation_source": outcome.observation_source,
        "model_observation_final": outcome.observation_final,
        "model_observation_adjusted": outcome.observation_adjusted,
        "model_observed_outcome": outcome.observed_outcome,
        "passes_signal_filter": filter_reason is None,
        "signal_eligible": filter_reason is None,
        "trade_eligible": filter_reason is None,
        "signal_filter_reason": filter_reason,
        "model_probabilities": outcome.model_probabilities,
        "raw_model_probabilities": outcome.raw_model_probabilities or {},
        **dict(metadata),
    }


def _crosscheck_pnl_summary(executions: list[dict[str, Any]]) -> dict[str, Any]:
    matched_pnl = 0.0
    mismatch_pnl = 0.0
    mismatch_count = 0
    checked_count = 0
    for execution in executions:
        realized = float(execution.get("realized_pnl_usd") or 0.0)
        if _has_weather_crosscheck(execution):
            checked_count += 1
        if _is_weather_crosscheck_mismatch(execution):
            mismatch_count += 1
            mismatch_pnl += realized
        else:
            matched_pnl += realized
    return {
        "weather_crosscheck_checked_executions": checked_count,
        "weather_crosscheck_mismatch_executions": mismatch_count,
        "realized_pnl_crosscheck_mismatch_usd": round(mismatch_pnl, 2),
        "realized_pnl_crosscheck_matched_or_unchecked_usd": round(matched_pnl, 2),
    }


def _trade_performance_diagnostics(executions: list[dict[str, Any]]) -> dict[str, Any]:
    by_token: dict[str, dict[str, Any]] = {}
    for execution in executions:
        token = str(execution.get("token_id") or "")
        if not token:
            continue
        row = by_token.setdefault(
            token,
            {
                "token_id": token,
                "question": execution.get("question"),
                "city": execution.get("city"),
                "target_date": execution.get("target_date"),
                "bucket": execution.get("bucket"),
                "first_buy_at": None,
                "first_buy_price": None,
                "first_buy_fair_value": None,
                "first_buy_edge": None,
                "first_buy_model_agreement": None,
                "first_buy_no_side_counter_event_probability": None,
                "side": execution.get("side"),
                "buy_notional_usd": 0.0,
                "sell_notional_usd": 0.0,
                "sell_realized_pnl_usd": 0.0,
                "settlement_realized_pnl_usd": 0.0,
                "sell_count": 0,
                "settlement_count": 0,
                "sell_decision_value_vs_settlement_usd": 0.0,
                "realized_pnl_usd": 0.0,
                "polymarket_payout": execution.get("polymarket_payout"),
                "weather_outcome": execution.get("weather_outcome"),
                "bucket_lower_f": execution.get("bucket_lower_f"),
                "bucket_upper_f": execution.get("bucket_upper_f"),
                "bucket_shape": execution.get("bucket_shape"),
            },
        )
        if execution.get("side") is not None:
            row["side"] = execution.get("side")
        row["realized_pnl_usd"] += float(execution.get("realized_pnl_usd") or 0.0)
        if execution.get("polymarket_payout") is not None:
            row["polymarket_payout"] = execution.get("polymarket_payout")
        if execution.get("weather_outcome") is not None:
            row["weather_outcome"] = execution.get("weather_outcome")
        if execution.get("action") == "SELL":
            row["sell_count"] += 1
            row["sell_notional_usd"] += float(execution.get("notional_usd") or 0.0)
            row["sell_realized_pnl_usd"] += float(execution.get("realized_pnl_usd") or 0.0)
            payout = _coerce_binary_outcome(execution.get("polymarket_payout"))
            if payout is not None:
                try:
                    row["sell_decision_value_vs_settlement_usd"] += (
                        float(execution.get("price")) - float(payout)
                    ) * float(execution.get("shares") or 0.0)
                except (TypeError, ValueError):
                    pass
        elif execution.get("action") == "SETTLE":
            row["settlement_count"] += 1
            row["settlement_realized_pnl_usd"] += float(execution.get("realized_pnl_usd") or 0.0)
        if execution.get("action") != "BUY":
            continue
        row["buy_notional_usd"] += float(execution.get("notional_usd") or 0.0)
        if row["first_buy_at"] is None:
            row["first_buy_at"] = execution.get("executed_at")
            row["first_buy_price"] = execution.get("price")
            row["first_buy_fair_value"] = execution.get("fair_value")
            row["first_buy_edge"] = execution.get("edge")
            row["first_buy_model_agreement"] = execution.get("model_agreement")
            row["first_buy_no_side_counter_event_probability"] = execution.get("no_side_counter_event_probability")

    trades = [row for row in by_token.values() if row["buy_notional_usd"] > 0]
    for trade in trades:
        trade["realized_pnl_usd"] = round(float(trade["realized_pnl_usd"]), 4)
        trade["buy_notional_usd"] = round(float(trade["buy_notional_usd"]), 4)
        trade["sell_notional_usd"] = round(float(trade["sell_notional_usd"]), 4)
        trade["sell_realized_pnl_usd"] = round(float(trade["sell_realized_pnl_usd"]), 4)
        trade["settlement_realized_pnl_usd"] = round(float(trade["settlement_realized_pnl_usd"]), 4)
        trade["sell_decision_value_vs_settlement_usd"] = round(float(trade["sell_decision_value_vs_settlement_usd"]), 4)
        trade["return_on_buy_notional"] = (
            round(float(trade["realized_pnl_usd"]) / float(trade["buy_notional_usd"]), 4)
            if trade["buy_notional_usd"]
            else None
        )
        trade["weather_crosscheck"] = _crosscheck_label(trade)
        trade["lead_days"] = _trade_lead_days(trade)
        trade["entry_hour_utc"] = _trade_entry_hour_utc(trade)
        trade["entry_month"] = _month_key(trade.get("first_buy_at"))
        trade["target_month"] = _month_key(trade.get("target_date"))
        trade["entry_price_bucket"] = _bucket_float(trade.get("first_buy_price"), 0.05)
        trade["fair_value_bucket"] = _bucket_float(trade.get("first_buy_fair_value"), 0.05)
        trade["edge_bucket"] = _bucket_float(trade.get("first_buy_edge"), 0.05)
        trade["agreement_bucket"] = _bucket_float(trade.get("first_buy_model_agreement"), 0.25)
        trade["no_side_counter_event_probability_bucket"] = _bucket_float(
            trade.get("first_buy_no_side_counter_event_probability"),
            0.025,
        )
        trade["event_outcome"] = _event_outcome_label(trade.get("polymarket_payout"))
        if trade.get("bucket_shape") is None:
            trade["bucket_shape"] = _bucket_shape(trade.get("bucket_lower_f"), trade.get("bucket_upper_f"))
    unprofitable_event_winners = [
        trade
        for trade in trades
        if trade.get("polymarket_payout") == 1 and float(trade.get("realized_pnl_usd") or 0.0) < 0.0
    ]

    return {
        "trade_count": len(trades),
        **_event_outcome_trade_summary(trades),
        "unprofitable_event_winner_trades": len(unprofitable_event_winners),
        "unprofitable_event_winner_pnl_usd": round(
            sum(float(trade.get("realized_pnl_usd") or 0.0) for trade in unprofitable_event_winners),
            2,
        ),
        "exit_management": _exit_management_diagnostics(trades),
        "worst_unprofitable_event_winners": sorted(
            unprofitable_event_winners,
            key=lambda item: float(item["realized_pnl_usd"]),
        )[:10],
        "pnl_concentration": _pnl_concentration(trades),
        "by_city": _aggregate_trade_groups(trades, "city"),
        "by_entry_price_bucket": _aggregate_trade_groups(trades, "entry_price_bucket"),
        "by_fair_value_bucket": _aggregate_trade_groups(trades, "fair_value_bucket"),
        "by_entry_hour_utc": _aggregate_trade_groups(trades, "entry_hour_utc"),
        "by_entry_month": _aggregate_trade_groups(trades, "entry_month"),
        "by_target_month": _aggregate_trade_groups(trades, "target_month"),
        "by_side": _aggregate_trade_groups(trades, "side"),
        "by_bucket_shape": _aggregate_trade_groups(trades, "bucket_shape"),
        "by_event_outcome": _aggregate_trade_groups(trades, "event_outcome"),
        "by_edge_bucket": _aggregate_trade_groups(trades, "edge_bucket"),
        "by_model_agreement_bucket": _aggregate_trade_groups(trades, "agreement_bucket"),
        "by_no_side_counter_event_probability_bucket": _aggregate_trade_groups(
            [trade for trade in trades if trade.get("side") == "NO"],
            "no_side_counter_event_probability_bucket",
        ),
        "by_lead_days": _aggregate_trade_groups(trades, "lead_days"),
        "by_weather_crosscheck": _aggregate_trade_groups(trades, "weather_crosscheck"),
        "best_trades": sorted(trades, key=lambda item: float(item["realized_pnl_usd"]), reverse=True)[:10],
        "worst_trades": sorted(trades, key=lambda item: float(item["realized_pnl_usd"]))[:10],
    }


def _score_calibration_diagnostics(scored: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in scored if row.get("polymarket_payout") in (0, 1)]
    signal_eligible = [row for row in resolved if row.get("signal_filter_reason") is None]
    raw_resolved = [row for row in resolved if row.get("raw_fair_value") is not None]
    raw_signal_eligible = [row for row in signal_eligible if row.get("raw_fair_value") is not None]
    return {
        "resolved_count": len(resolved),
        "signal_eligible_count": len(signal_eligible),
        "raw_probability_rows": len(raw_resolved),
        "raw_signal_eligible_rows": len(raw_signal_eligible),
        "overall": _calibration_metrics(resolved),
        "signal_eligible": _calibration_metrics(signal_eligible),
        "raw_overall": _calibration_metrics(raw_resolved, probability_key="raw_fair_value"),
        "raw_signal_eligible": _calibration_metrics(raw_signal_eligible, probability_key="raw_fair_value"),
        "by_market_price_bucket": _aggregate_score_groups(resolved, "market_price", 0.10),
        "by_fair_value_bucket": _aggregate_score_groups(resolved, "fair_value", 0.10),
        "by_edge_bucket": _aggregate_score_groups(resolved, "edge", 0.10),
        "by_model_agreement_bucket": _aggregate_score_groups(resolved, "model_agreement", 0.25),
        "by_signal_filter_reason": _aggregate_score_groups(resolved, "signal_filter_reason", None),
        "model_probability_accuracy": _model_probability_accuracy(resolved),
        "raw_model_probability_accuracy": _model_probability_accuracy(resolved, probability_key="raw_model_probabilities"),
        "source_probability_accuracy": _model_probability_accuracy(resolved, group_by="source"),
        "raw_source_probability_accuracy": _model_probability_accuracy(resolved, group_by="source", probability_key="raw_model_probabilities"),
        "model_family_probability_accuracy": _model_probability_accuracy(resolved, group_by="model_family"),
        "raw_model_family_probability_accuracy": _model_probability_accuracy(resolved, group_by="model_family", probability_key="raw_model_probabilities"),
        "signal_eligible_model_probability_accuracy": _model_probability_accuracy(signal_eligible),
        "signal_eligible_raw_model_probability_accuracy": _model_probability_accuracy(signal_eligible, probability_key="raw_model_probabilities"),
        "signal_eligible_source_probability_accuracy": _model_probability_accuracy(signal_eligible, group_by="source"),
        "signal_eligible_raw_source_probability_accuracy": _model_probability_accuracy(signal_eligible, group_by="source", probability_key="raw_model_probabilities"),
        "signal_eligible_model_family_probability_accuracy": _model_probability_accuracy(signal_eligible, group_by="model_family"),
        "signal_eligible_raw_model_family_probability_accuracy": _model_probability_accuracy(signal_eligible, group_by="model_family", probability_key="raw_model_probabilities"),
    }


def _data_quality_diagnostics(
    executions: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    *,
    max_price_staleness_minutes: int,
    forecast_availability_lag_hours: int,
) -> dict[str, Any]:
    execution_checks = _timestamp_quality_counts(
        executions,
        generated_key="executed_at",
        max_price_staleness_seconds=max_price_staleness_minutes * 60,
        forecast_availability_lag_seconds=forecast_availability_lag_hours * 3600,
    )
    scored_checks = _timestamp_quality_counts(
        scored,
        generated_key="generated_at",
        max_price_staleness_seconds=max_price_staleness_minutes * 60,
        forecast_availability_lag_seconds=forecast_availability_lag_hours * 3600,
    )
    return {
        "max_allowed_price_staleness_seconds": max_price_staleness_minutes * 60,
        "required_forecast_availability_lag_seconds": forecast_availability_lag_hours * 3600,
        "executions": execution_checks,
        "scored_rows": scored_checks,
    }


def _settlement_quality_diagnostics(scored: list[dict[str, Any]], executions: list[dict[str, Any]]) -> dict[str, Any]:
    signal_eligible = [row for row in scored if row.get("signal_filter_reason") is None]
    return {
        "scored_rows": _settlement_quality_row_counts(scored),
        "signal_eligible_rows": _settlement_quality_row_counts(signal_eligible),
        "traded_tokens": _settlement_quality_trade_counts(executions),
    }


def _real_data_audit(
    scored: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    *,
    data_quality_diagnostics: Mapping[str, Any],
    settlement_quality_diagnostics: Mapping[str, Any],
    require_weather_crosscheck: bool = True,
) -> dict[str, Any]:
    scored_quality = data_quality_diagnostics.get("scored_rows") or {}
    execution_quality = data_quality_diagnostics.get("executions") or {}
    traded_quality = settlement_quality_diagnostics.get("traded_tokens") or {}
    signal_quality = settlement_quality_diagnostics.get("signal_eligible_rows") or {}

    scored_count = len(scored)
    execution_count = len(executions)
    no_side_rows = [row for row in scored if row.get("side") == "NO"]
    explicit_no_rows = [
        row
        for row in no_side_rows
        if row.get("yes_token_id") and str(row.get("yes_token_id")) != str(row.get("token_id"))
    ]
    no_side_buy_executions = [
        row
        for row in executions
        if row.get("action") == "BUY" and row.get("side") == "NO"
    ]
    explicit_no_buy_executions = [
        row
        for row in no_side_buy_executions
        if row.get("yes_token_id") and str(row.get("yes_token_id")) != str(row.get("token_id"))
    ]
    forecast_source_rows = [row for row in scored if row.get("forecast_sources")]
    historical_forecast_rows = [
        row
        for row in forecast_source_rows
        if all(str(source).startswith("single_run_") for source in row.get("forecast_sources") or [])
    ]
    fixture_forecast_rows = [
        _audit_row_example(row)
        for row in forecast_source_rows
        if any(str(source).lower() == "fixture" for source in row.get("forecast_sources") or [])
    ][:5]

    signal_weather_check = {
        "required": require_weather_crosscheck,
        "passed": int(signal_quality.get("total_rows") or 0) == 0
        or (
            int(signal_quality.get("weather_checked_rows") or 0) == int(signal_quality.get("total_rows") or 0)
            and int(signal_quality.get("weather_matched_rows") or 0) == int(signal_quality.get("total_rows") or 0)
            and int(signal_quality.get("weather_mismatch_rows") or 0) == 0
            and int(signal_quality.get("weather_ambiguous_rows") or 0) == 0
            and int(signal_quality.get("unresolved_rows") or 0) == 0
        ),
        "total_rows": int(signal_quality.get("total_rows") or 0),
        "weather_checked_rows": int(signal_quality.get("weather_checked_rows") or 0),
        "weather_matched_rows": int(signal_quality.get("weather_matched_rows") or 0),
        "weather_mismatch_rows": int(signal_quality.get("weather_mismatch_rows") or 0),
        "weather_ambiguous_rows": int(signal_quality.get("weather_ambiguous_rows") or 0),
        "unresolved_rows": int(signal_quality.get("unresolved_rows") or 0),
    }
    if not require_weather_crosscheck:
        signal_weather_check["passed"] = True
        signal_weather_check["note"] = "Weather cross-check skipped by settlement_audit=polymarket_only; Polymarket resolved payout is the settlement source."

    traded_weather_check = {
        "required": require_weather_crosscheck,
        "passed": int(traded_quality.get("traded_token_count") or 0) == 0
        or (
            int(traded_quality.get("weather_checked_traded_tokens") or 0) == int(traded_quality.get("traded_token_count") or 0)
            and int(traded_quality.get("weather_matched_traded_tokens") or 0) == int(traded_quality.get("traded_token_count") or 0)
            and int(traded_quality.get("weather_mismatch_traded_tokens") or 0) == 0
            and int(traded_quality.get("weather_ambiguous_traded_tokens") or 0) == 0
            and int(traded_quality.get("unresolved_traded_tokens") or 0) == 0
            and int(traded_quality.get("polymarket_only_traded_tokens") or 0) == 0
        ),
        "traded_token_count": int(traded_quality.get("traded_token_count") or 0),
        "weather_checked_traded_tokens": int(traded_quality.get("weather_checked_traded_tokens") or 0),
        "weather_matched_traded_tokens": int(traded_quality.get("weather_matched_traded_tokens") or 0),
        "weather_mismatch_traded_tokens": int(traded_quality.get("weather_mismatch_traded_tokens") or 0),
        "weather_ambiguous_traded_tokens": int(traded_quality.get("weather_ambiguous_traded_tokens") or 0),
        "unresolved_traded_tokens": int(traded_quality.get("unresolved_traded_tokens") or 0),
        "polymarket_only_traded_tokens": int(traded_quality.get("polymarket_only_traded_tokens") or 0),
    }
    if not require_weather_crosscheck:
        traded_weather_check["passed"] = True
        traded_weather_check["note"] = "Weather cross-check skipped by settlement_audit=polymarket_only; Polymarket resolved payout is the settlement source."

    checks = {
        "scored_rows_have_historical_price_timestamps": {
            "passed": scored_count > 0 and int(scored_quality.get("price_timestamp_checked") or 0) == scored_count,
            "checked": int(scored_quality.get("price_timestamp_checked") or 0),
            "expected": scored_count,
        },
        "scored_rows_have_historical_forecast_run_times": {
            "passed": scored_count > 0 and int(scored_quality.get("forecast_timestamp_checked") or 0) == scored_count,
            "checked": int(scored_quality.get("forecast_timestamp_checked") or 0),
            "expected": scored_count,
        },
        "scored_rows_use_open_meteo_single_runs": {
            "passed": scored_count > 0 and len(historical_forecast_rows) == scored_count and not fixture_forecast_rows,
            "checked": len(historical_forecast_rows),
            "expected": scored_count,
            "fixture_examples": fixture_forecast_rows,
        },
        "no_future_or_stale_prices": {
            "passed": int(scored_quality.get("future_price_violations") or 0) == 0
            and int(scored_quality.get("stale_price_violations") or 0) == 0
            and int(execution_quality.get("future_price_violations") or 0) == 0
            and int(execution_quality.get("stale_price_violations") or 0) == 0,
            "scored_future_price_violations": int(scored_quality.get("future_price_violations") or 0),
            "scored_stale_price_violations": int(scored_quality.get("stale_price_violations") or 0),
            "execution_future_price_violations": int(execution_quality.get("future_price_violations") or 0),
            "execution_stale_price_violations": int(execution_quality.get("stale_price_violations") or 0),
        },
        "forecast_availability_lag_respected": {
            "passed": int(scored_quality.get("future_forecast_violations") or 0) == 0
            and int(scored_quality.get("unavailable_forecast_violations") or 0) == 0
            and int(execution_quality.get("future_forecast_violations") or 0) == 0
            and int(execution_quality.get("unavailable_forecast_violations") or 0) == 0,
            "scored_future_forecast_violations": int(scored_quality.get("future_forecast_violations") or 0),
            "scored_unavailable_forecast_violations": int(scored_quality.get("unavailable_forecast_violations") or 0),
            "execution_future_forecast_violations": int(execution_quality.get("future_forecast_violations") or 0),
            "execution_unavailable_forecast_violations": int(execution_quality.get("unavailable_forecast_violations") or 0),
            "min_forecast_lag_seconds": scored_quality.get("min_forecast_lag_seconds"),
        },
        "no_side_rows_use_explicit_no_tokens": {
            "passed": len(explicit_no_rows) == len(no_side_rows) and len(explicit_no_buy_executions) == len(no_side_buy_executions),
            "no_side_rows": len(no_side_rows),
            "explicit_no_side_rows": len(explicit_no_rows),
            "no_side_buy_executions": len(no_side_buy_executions),
            "explicit_no_side_buy_executions": len(explicit_no_buy_executions),
        },
        "signal_eligible_rows_weather_matched": signal_weather_check,
        "traded_tokens_weather_matched": traded_weather_check,
    }
    failures = [name for name, check in checks.items() if not check.get("passed")]
    return {
        "passed": not failures,
        "method": (
            "Strict real-data audit for the historical replay. It verifies historical price timestamps, "
            "forecast run-time lag, Open-Meteo Single Runs forecast sources, explicit NO-token attribution, "
            "and weather-matched settlement for signal-eligible and traded rows when settlement_audit=weather_crosscheck."
        ),
        "require_weather_crosscheck": require_weather_crosscheck,
        "failure_reasons": failures,
        "checks": checks,
    }


def _audit_row_example(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question": row.get("question"),
        "city": row.get("city"),
        "target_date": row.get("target_date"),
        "generated_at": row.get("generated_at"),
        "forecast_sources": row.get("forecast_sources"),
        "token_id": row.get("token_id"),
    }


def _settlement_quality_row_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    counts = {
        "total_rows": 0,
        "polymarket_resolved_rows": 0,
        "weather_observed_rows": 0,
        "weather_checked_rows": 0,
        "weather_matched_rows": 0,
        "weather_mismatch_rows": 0,
        "weather_ambiguous_rows": 0,
        "polymarket_only_rows": 0,
        "weather_only_rows": 0,
        "unresolved_rows": 0,
        "observed_high_rows": 0,
    }
    for row in rows:
        counts["total_rows"] += 1
        polymarket_payout = _coerce_binary_outcome(row.get("polymarket_payout"))
        weather_outcome = _coerce_binary_outcome(row.get("weather_outcome"))
        weather_ambiguous = bool(row.get("weather_ambiguous"))
        if polymarket_payout is not None:
            counts["polymarket_resolved_rows"] += 1
        if weather_outcome is not None:
            counts["weather_observed_rows"] += 1
        if row.get("observed_high_f") is not None:
            counts["observed_high_rows"] += 1
        if weather_ambiguous:
            counts["weather_ambiguous_rows"] += 1
        if polymarket_payout is not None and weather_outcome is not None:
            counts["weather_checked_rows"] += 1
            if polymarket_payout == weather_outcome:
                counts["weather_matched_rows"] += 1
            else:
                counts["weather_mismatch_rows"] += 1
        elif polymarket_payout is not None:
            if not weather_ambiguous:
                counts["polymarket_only_rows"] += 1
        elif weather_outcome is not None:
            counts["weather_only_rows"] += 1
        else:
            counts["unresolved_rows"] += 1
        source = str(row.get("settlement_source") or "none")
        sources[source] = sources.get(source, 0) + 1
    return {**counts, "settlement_source_counts": _sorted_count_dict(sources)}


def _settlement_quality_trade_counts(executions: list[dict[str, Any]]) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    buy_execution_count = 0
    for execution in executions:
        token = str(execution.get("token_id") or "")
        if not token:
            continue
        state = states.setdefault(
            token,
            {
                "bought": False,
                "polymarket_payout": None,
                "weather_outcome": None,
                "weather_ambiguous": False,
                "observed_high": False,
                "settlement_source": None,
            },
        )
        if execution.get("action") == "BUY":
            state["bought"] = True
            buy_execution_count += 1
        polymarket_payout = _coerce_binary_outcome(execution.get("polymarket_payout"))
        weather_outcome = _coerce_binary_outcome(execution.get("weather_outcome"))
        if polymarket_payout is not None:
            state["polymarket_payout"] = polymarket_payout
        if weather_outcome is not None:
            state["weather_outcome"] = weather_outcome
        if execution.get("weather_ambiguous"):
            state["weather_ambiguous"] = True
        if execution.get("observed_high_f") is not None:
            state["observed_high"] = True
        if execution.get("settlement_source") and state["settlement_source"] is None:
            state["settlement_source"] = str(execution.get("settlement_source"))

    sources: dict[str, int] = {}
    counts = {
        "buy_execution_count": buy_execution_count,
        "traded_token_count": 0,
        "polymarket_resolved_traded_tokens": 0,
        "weather_observed_traded_tokens": 0,
        "weather_checked_traded_tokens": 0,
        "weather_matched_traded_tokens": 0,
        "weather_mismatch_traded_tokens": 0,
        "weather_ambiguous_traded_tokens": 0,
        "polymarket_only_traded_tokens": 0,
        "weather_only_traded_tokens": 0,
        "unresolved_traded_tokens": 0,
        "observed_high_traded_tokens": 0,
    }
    for state in states.values():
        if not state["bought"]:
            continue
        counts["traded_token_count"] += 1
        polymarket_payout = state["polymarket_payout"]
        weather_outcome = state["weather_outcome"]
        weather_ambiguous = bool(state["weather_ambiguous"])
        if polymarket_payout is not None:
            counts["polymarket_resolved_traded_tokens"] += 1
        if weather_outcome is not None:
            counts["weather_observed_traded_tokens"] += 1
        if state["observed_high"]:
            counts["observed_high_traded_tokens"] += 1
        if weather_ambiguous:
            counts["weather_ambiguous_traded_tokens"] += 1
        if polymarket_payout is not None and weather_outcome is not None:
            counts["weather_checked_traded_tokens"] += 1
            if polymarket_payout == weather_outcome:
                counts["weather_matched_traded_tokens"] += 1
            else:
                counts["weather_mismatch_traded_tokens"] += 1
        elif polymarket_payout is not None:
            if not weather_ambiguous:
                counts["polymarket_only_traded_tokens"] += 1
        elif weather_outcome is not None:
            counts["weather_only_traded_tokens"] += 1
        else:
            counts["unresolved_traded_tokens"] += 1
        source = str(state["settlement_source"] or "none")
        sources[source] = sources.get(source, 0) + 1
    return {**counts, "settlement_source_counts": _sorted_count_dict(sources)}


def _coerce_binary_outcome(value: Any) -> Optional[int]:
    if value in (0, 1):
        return int(value)
    if isinstance(value, str) and value in {"0", "1"}:
        return int(value)
    return None


def _sorted_count_dict(counts: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _timestamp_quality_counts(
    rows: list[dict[str, Any]],
    *,
    generated_key: str,
    max_price_staleness_seconds: int,
    forecast_availability_lag_seconds: int,
) -> dict[str, Any]:
    price_checked = 0
    forecast_checked = 0
    future_price_violations = 0
    stale_price_violations = 0
    future_forecast_violations = 0
    unavailable_forecast_violations = 0
    max_price_stale_seconds = 0.0
    min_forecast_lag_seconds: Optional[float] = None
    forecast_lag_seconds: set[float] = set()
    for row in rows:
        generated_at = _parse_dt(row.get(generated_key))
        if generated_at is None:
            continue
        price_time = _parse_dt(row.get("entry_price_timestamp"))
        if price_time is not None:
            price_checked += 1
            stale_seconds = max(0.0, (generated_at - price_time).total_seconds())
            stale_value = _optional_float(row.get("entry_price_stale_seconds"))
            if stale_value is not None:
                stale_seconds = max(stale_seconds, stale_value)
            max_price_stale_seconds = max(max_price_stale_seconds, stale_seconds)
            if price_time > generated_at:
                future_price_violations += 1
            if stale_seconds > max_price_staleness_seconds + 1e-6:
                stale_price_violations += 1
        forecast_time = _parse_dt(row.get("forecast_run_time"))
        if forecast_time is not None:
            forecast_checked += 1
            lag_seconds = (generated_at - forecast_time).total_seconds()
            min_forecast_lag_seconds = lag_seconds if min_forecast_lag_seconds is None else min(min_forecast_lag_seconds, lag_seconds)
            forecast_lag_seconds.add(round(lag_seconds, 3))
            if forecast_time > generated_at:
                future_forecast_violations += 1
            if lag_seconds + 1e-6 < forecast_availability_lag_seconds:
                unavailable_forecast_violations += 1
    return {
        "rows_checked": len(rows),
        "price_timestamp_checked": price_checked,
        "forecast_timestamp_checked": forecast_checked,
        "future_price_violations": future_price_violations,
        "stale_price_violations": stale_price_violations,
        "future_forecast_violations": future_forecast_violations,
        "unavailable_forecast_violations": unavailable_forecast_violations,
        "max_price_stale_seconds": round(max_price_stale_seconds, 3),
        "min_forecast_lag_seconds": round(min_forecast_lag_seconds, 3) if min_forecast_lag_seconds is not None else None,
        "unique_forecast_lag_seconds": sorted(forecast_lag_seconds)[:20],
    }


def _signal_filter_diagnostics(scored: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in scored:
        reason = str(row.get("signal_filter_reason") or "eligible")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _signal_opportunity_diagnostics(scored: list[dict[str, Any]], settings: SignalSettings) -> dict[str, Any]:
    resolved = [row for row in scored if row.get("polymarket_payout") in (0, 1)]
    entry_eligible = [row for row in resolved if row.get("entry_eligible", True)]
    signal_eligible = [row for row in resolved if row.get("signal_filter_reason") is None]
    selected_candidates = _selected_candidate_rows(scored, settings)
    rejected_entry_rows = [
        row
        for row in entry_eligible
        if row.get("signal_filter_reason") is not None
    ]
    selected_rows = _score_cohort_rows(selected_candidates)
    signal_rows = _score_cohort_rows(signal_eligible)
    rejected_rows = _score_cohort_rows(rejected_entry_rows)
    return {
        "method": (
            "Resolved scored-row diagnostics. Selected candidates use the same one-token-per-city/date/session "
            "selection as the replay. Rejected rows are not trade recommendations; they show which gates filtered "
            "rows that later resolved true or false."
        ),
        "resolved_rows": len(resolved),
        "entry_eligible_rows": len(entry_eligible),
        "signal_eligible_rows": len(signal_eligible),
        "selected_candidate_rows": len(selected_candidates),
        "selected_candidate_calibration": _calibration_metrics(selected_rows),
        "selected_candidate_by_side": _aggregate_opportunity_groups(selected_rows, "side", None),
        "selected_candidate_by_entry_hour_utc": _aggregate_opportunity_groups(selected_rows, "entry_hour_utc", None),
        "selected_candidate_by_lead_days": _aggregate_opportunity_groups(selected_rows, "lead_days", None),
        "selected_candidate_by_bucket_shape": _aggregate_opportunity_groups(selected_rows, "bucket_shape", None),
        "selected_candidate_by_market_price_bucket": _aggregate_opportunity_groups(selected_rows, "market_price", 0.10),
        "selected_candidate_by_edge_bucket": _aggregate_opportunity_groups(selected_rows, "edge", 0.10),
        "selected_candidate_by_no_side_counter_event_probability": _aggregate_opportunity_groups(
            [row for row in selected_rows if row.get("side") == "NO"],
            "no_side_counter_event_probability",
            0.05,
        ),
        "signal_eligible_by_side": _aggregate_opportunity_groups(signal_rows, "side", None),
        "signal_eligible_by_entry_hour_utc": _aggregate_opportunity_groups(signal_rows, "entry_hour_utc", None),
        "rejected_by_signal_filter_reason": _rejected_reason_quality(rejected_rows),
        "top_rejected_winners_by_edge": _top_rejected_examples(rejected_rows, payout=1),
        "top_rejected_losers_by_edge": _top_rejected_examples(rejected_rows, payout=0),
    }


def _score_cohort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, **_score_cohort_fields(row)} for row in rows]


def _score_cohort_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_hour_utc": _score_entry_hour_utc(row),
        "lead_days": _score_lead_days(row),
        "target_month": _month_key(row.get("target_date")),
        "entry_month": _month_key(row.get("generated_at")),
        "no_side_counter_event_probability": (
            no_side_counter_event_probability(row.get("model_probabilities") or {})
            if _json_is_no_side_row(row)
            else None
        ),
    }


def _rejected_reason_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("signal_filter_reason") or "eligible", []).append(row)
    return [
        _opportunity_quality_row(reason, items)
        for reason, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0])))
    ][:30]


def _aggregate_opportunity_groups(rows: list[dict[str, Any]], key: str, width: Optional[float]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = row.get(key)
        if width is not None:
            group_key = _bucket_float(group_key, width)
        grouped.setdefault(group_key, []).append(row)
    return [
        _opportunity_quality_row(group, items)
        for group, items in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _opportunity_quality_row(group: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = _candidate_quality_row("cohort", group, rows)
    row.pop("setting", None)
    row["group"] = row.pop("threshold")
    return row


def _top_rejected_examples(rows: list[dict[str, Any]], *, payout: int) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row.get("polymarket_payout") == payout]
    return [
        {
            "generated_at": row.get("generated_at"),
            "token_id": row.get("token_id"),
            "question": row.get("question"),
            "city": row.get("city"),
            "target_date": row.get("target_date"),
            "side": row.get("side"),
            "bucket": row.get("bucket"),
            "signal_filter_reason": row.get("signal_filter_reason"),
            "market_price": row.get("market_price"),
            "fair_value": row.get("fair_value"),
            "edge": row.get("edge"),
            "model_agreement": row.get("model_agreement"),
            "entry_hour_utc": row.get("entry_hour_utc"),
            "lead_days": row.get("lead_days"),
            "no_side_counter_event_probability": row.get("no_side_counter_event_probability"),
            "polymarket_payout": row.get("polymarket_payout"),
            "weather_outcome": row.get("weather_outcome"),
        }
        for row in sorted(filtered, key=lambda item: float(item.get("edge") or 0.0), reverse=True)[:10]
    ]


def _strategy_sensitivity_diagnostics(
    scored: list[dict[str, Any]],
    settings: SignalSettings,
    *,
    bankroll_usd: float = 100.0,
    kelly_fraction: float = 0.25,
    compound_kelly_sizing: bool = False,
    max_position_usd: float = 50.0,
    max_position_fraction: Optional[float] = None,
    kelly_market_blend: float = 0.0,
    edge_position_full_cap_edge: float = 0.0,
    edge_position_min_multiplier: float = 0.35,
    min_trade_usd: float = 1.0,
) -> dict[str, Any]:
    return {
        "method": "Candidate threshold tables use first qualifying selected token per city/date/session with equal $1 buy-to-settlement diagnostics; counterfactual_kelly_replays use JSON-row Kelly buy/sell/hold/settlement replay.",
        "by_min_signal_fair_value": [
            _candidate_quality_row(
                "min_signal_fair_value",
                threshold,
                _selected_candidate_rows(scored, settings, min_signal_fair_value=threshold),
            )
            for threshold in (0.0, 0.50, 0.60, 0.70, 0.80, 0.90)
        ],
        "by_min_price": [
            _candidate_quality_row("min_price", threshold, _selected_candidate_rows(scored, settings, min_price=threshold))
            for threshold in (0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.70)
        ],
        "by_yes_side_min_price": [
            _candidate_quality_row(
                "yes_side_min_price",
                threshold,
                _selected_candidate_rows(scored, settings, yes_side_min_price=threshold),
            )
            for threshold in (0.125, 0.15, 0.20, 0.25, 0.30, 0.35)
        ],
        "by_no_side_max_counter_event_probability": [
            _candidate_quality_row(
                "no_side_max_counter_event_probability",
                threshold,
                _selected_candidate_rows(scored, settings, no_side_max_counter_event_probability=threshold),
            )
            for threshold in (0.05, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.30, 1.0)
        ],
        "by_no_side_max_price": [
            _candidate_quality_row(
                "no_side_max_price",
                threshold,
                _selected_candidate_rows(scored, replace(settings, no_side_max_price=threshold)),
            )
            for threshold in (0.90, 0.925, 0.93, 0.935, 0.94, 0.95)
        ],
        "by_no_side_min_edge": [
            _candidate_quality_row(
                "no_side_min_edge",
                threshold,
                _selected_candidate_rows(scored, settings, no_side_min_edge=threshold),
            )
            for threshold in (0.00, 0.03, 0.05, 0.075, 0.10, 0.12, 0.15, 0.20)
        ],
        "by_no_side_high_confidence_min_edge": [
            _candidate_quality_row(
                "no_side_high_confidence_min_edge",
                threshold,
                _selected_candidate_rows(scored, replace(settings, no_side_high_confidence_min_edge=threshold)),
            )
            for threshold in (0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10)
        ],
        "by_min_model_agreement": [
            _candidate_quality_row(
                "min_model_agreement",
                threshold,
                _selected_candidate_rows(scored, settings, min_model_agreement=threshold),
            )
            for threshold in (0.50, 0.65, 0.75, 0.90, 1.0)
        ],
        "counterfactual_kelly_replays": _counterfactual_kelly_replays(
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        ),
    }


def _strategy_recommendation_diagnostics(
    sensitivity: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> dict[str, Any]:
    replays = {
        str(row.get("variant")): row
        for row in sensitivity.get("counterfactual_kelly_replays", [])
        if isinstance(row, Mapping) and row.get("variant")
    }
    current = replays.get("current")
    conservative_cap = replays.get("max_position_fraction_0.10")
    cap_upgrade_variants = (
        ("aggressive_max_position_fraction_0.20", "max_position_fraction_0.20", "20%"),
        ("aggressive_max_position_fraction_0.225", "max_position_fraction_0.225", "22.5%"),
        ("aggressive_max_position_fraction_0.25", "max_position_fraction_0.25", "25%"),
    )
    looser_tail = replays.get("looser_no_side_counter_event_0.20")
    very_loose_tail = replays.get("looser_no_side_counter_event_0.30")
    time_conditioned_tail = replays.get("utc12_relaxed_no_side_counter_event_0.20")
    very_loose_time_conditioned_tail = replays.get("utc12_relaxed_no_side_counter_event_0.30")
    candidates = [
        _profile_summary("current", current),
        _profile_summary("conservative_max_position_fraction_0.10", conservative_cap),
        *(
            _profile_summary(display_name, replays.get(variant_name))
            for display_name, variant_name, _label in cap_upgrade_variants
        ),
        _profile_summary("looser_no_side_counter_event_0.20", looser_tail),
        _profile_summary("looser_no_side_counter_event_0.30", very_loose_tail),
        _profile_summary("utc12_relaxed_no_side_counter_event_0.20", time_conditioned_tail),
        _profile_summary("utc12_relaxed_no_side_counter_event_0.30", very_loose_time_conditioned_tail),
    ]
    candidates = [candidate for candidate in candidates if candidate]

    cap_slice_checks = {
        variant_name: _cap_slice_check(robustness, variant_name)
        for _display_name, variant_name, _label in cap_upgrade_variants
    }

    recommended = "current"
    recommendation_type = "keep_current"
    reasons = [
        "Current gates remain the accuracy baseline because traded tokens have weather-matched settlement checks and no ambiguous traded tokens.",
    ]
    if current and _safe_float((current.get("pnl_concentration") or {}).get("top_1_pnl_share")) >= 0.75:
        reasons.append(
            "Current replay PnL is highly concentrated in its top trade, so headline return should be treated as strategy research rather than production proof."
        )
    for display_name, variant_name, label in cap_upgrade_variants:
        replay = replays.get(variant_name)
        slice_check = cap_slice_checks[variant_name]
        if current and replay and _clean_cap_upgrade(current, replay) and slice_check["all_profitable_and_clean"]:
            recommended = display_name
            recommendation_type = "paper_test_aggressive_sizing"
            reasons.append(
                f"The {label} current-equity cap increased replay PnL without changing trade count, event hit rate, weather ambiguity, or mismatch count."
            )
            reasons.append(f"All {label} cap chronological robustness slices stayed profitable with event hit rate at or above 90%.")
    if looser_tail:
        if (
            _safe_float(looser_tail.get("pnl_usd")) > _safe_float(current.get("pnl_usd") if current else None)
            and (
                _safe_float(looser_tail.get("event_hit_rate")) < _safe_float(current.get("event_hit_rate") if current else None)
                or int(looser_tail.get("weather_ambiguous_trades") or 0) > 0
                or int(looser_tail.get("weather_mismatch_trades") or 0) > 0
            )
        ):
            reasons.append(
                "The looser 20% NO counter-event tail is not recommended despite higher in-sample PnL because it lowered event hit rate or introduced ambiguous weather validation."
            )
        elif _safe_float(looser_tail.get("pnl_usd")) > _safe_float(current.get("pnl_usd") if current else None):
            reasons.append(
                "The looser 20% NO counter-event tail is a high-risk candidate rather than a default promotion because it increases trade count, gross exposure, and drawdown; it needs live paper validation before replacing the strict 10% entry gate."
            )
    if time_conditioned_tail and _safe_float(time_conditioned_tail.get("pnl_usd")) > _safe_float(current.get("pnl_usd") if current else None):
        reasons.append(
            "The UTC-12 relaxed 20% NO counter-event tail is a narrower candidate for live-forward paper testing because it keeps the strict 10% tail outside its configured entry hour."
        )
    if very_loose_tail and _safe_float(very_loose_tail.get("max_drawdown_usd")) > _safe_float(looser_tail.get("max_drawdown_usd") if looser_tail else None):
        reasons.append(
            "The 30% NO counter-event tail is tracked as an exploratory diagnostic only; it is expected to add more trades but should not be promoted without cleaner drawdown and forward-paper evidence."
        )

    return {
        "method": (
            "Recommendation compares precomputed real-data Kelly replays and robustness slices. "
            "A higher-PnL profile is promoted only when it preserves current trade count, event hit rate, "
            "and weather-validation cleanliness, and when its chronological cap slices remain positive."
        ),
        "recommended_profile": recommended,
        "recommendation_type": recommendation_type,
        "candidate_profiles": candidates,
        "cap_slice_checks": cap_slice_checks,
        "cap_20_slice_check": cap_slice_checks["max_position_fraction_0.20"],
        "cap_25_slice_check": cap_slice_checks["max_position_fraction_0.25"],
        "reasons": reasons,
    }


def _cap_slice_check(robustness: Mapping[str, Any], variant_name: str) -> dict[str, Any]:
    slices = [
        row
        for row in robustness.get("cap_fraction_by_chronological_session_slice", [])
        if isinstance(row, Mapping) and row.get("variant") == variant_name
    ]
    clean_slices = bool(slices) and all(
        _safe_float(row.get("pnl_usd")) > 0
        and (_safe_float(row.get("event_hit_rate")) is None or _safe_float(row.get("event_hit_rate")) >= 0.90)
        and int(row.get("weather_mismatch_trades") or 0) == 0
        and int(row.get("weather_ambiguous_trades") or 0) == 0
        for row in slices
    )
    return {
        "variant": variant_name,
        "slice_count": len(slices),
        "all_profitable_and_clean": clean_slices,
        "slices": [
            {
                "slice": row.get("slice"),
                "pnl_usd": row.get("pnl_usd"),
                "event_hit_rate": row.get("event_hit_rate"),
                "max_drawdown_usd": row.get("max_drawdown_usd"),
                "weather_ambiguous_trades": row.get("weather_ambiguous_trades"),
                "weather_mismatch_trades": row.get("weather_mismatch_trades"),
            }
            for row in slices
        ],
    }


def _profile_summary(name: str, row: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if not row:
        return None
    keys = (
        "pnl_usd",
        "return_pct",
        "trade_count",
        "event_hit_rate",
        "hit_rate",
        "max_drawdown_usd",
        "buy_notional_usd",
        "return_on_buy_notional",
        "weather_ambiguous_trades",
        "weather_mismatch_trades",
    )
    return {"profile": name, **{key: row.get(key) for key in keys}}


def _clean_cap_upgrade(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    return (
        _safe_float(candidate.get("pnl_usd")) > _safe_float(current.get("pnl_usd"))
        and int(candidate.get("trade_count") or 0) == int(current.get("trade_count") or 0)
        and _safe_float(candidate.get("event_hit_rate")) >= _safe_float(current.get("event_hit_rate"))
        and int(candidate.get("weather_ambiguous_trades") or 0) == int(current.get("weather_ambiguous_trades") or 0)
        and int(candidate.get("weather_mismatch_trades") or 0) == int(current.get("weather_mismatch_trades") or 0)
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _robustness_diagnostics(
    scored: list[dict[str, Any]],
    settings: SignalSettings,
    *,
    bankroll_usd: float,
    kelly_fraction: float,
    compound_kelly_sizing: bool,
    max_position_usd: float,
    max_position_fraction: Optional[float],
    kelly_market_blend: float,
    edge_position_full_cap_edge: float,
    edge_position_min_multiplier: float,
    min_trade_usd: float,
) -> dict[str, Any]:
    selected_candidates = _selected_candidate_rows(scored, settings)
    session_keys = _sorted_session_keys(scored)
    chronological_slices = [
        _session_fraction_slice("first_50pct_sessions", scored, session_keys, 0.0, 0.50),
        _session_fraction_slice("second_50pct_sessions", scored, session_keys, 0.50, 1.0),
        _session_fraction_slice("first_70pct_sessions", scored, session_keys, 0.0, 0.70),
        _session_fraction_slice("last_30pct_sessions", scored, session_keys, 0.70, 1.0),
    ]
    month_slices = [
        (f"entry_month_{month}", _rows_for_entry_month(scored, month))
        for month in _entry_months(scored)
    ]
    return {
        "method": "Fresh-bankroll slice replays use current gates and sizing on subsets of the same real scored rows. They are robustness diagnostics, not separately optimized walk-forward parameters.",
        "selected_candidate_count": len(selected_candidates),
        "selected_candidate_calibration": _calibration_metrics(selected_candidates),
        "selected_candidate_by_side": _aggregate_score_groups(selected_candidates, "side", None),
        "by_chronological_session_slice": [
            _replay_slice(
                label,
                rows,
                settings,
                bankroll_usd=bankroll_usd,
                kelly_fraction=kelly_fraction,
                compound_kelly_sizing=compound_kelly_sizing,
                max_position_usd=max_position_usd,
                max_position_fraction=max_position_fraction,
                kelly_market_blend=kelly_market_blend,
                edge_position_full_cap_edge=edge_position_full_cap_edge,
                edge_position_min_multiplier=edge_position_min_multiplier,
                min_trade_usd=min_trade_usd,
            )
            for label, rows in chronological_slices
        ],
        "cap_fraction_by_chronological_session_slice": _cap_fraction_slice_replays(
            chronological_slices,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        ),
        "by_entry_month_replay": [
            _replay_slice(
                label,
                rows,
                settings,
                bankroll_usd=bankroll_usd,
                kelly_fraction=kelly_fraction,
                compound_kelly_sizing=compound_kelly_sizing,
                max_position_usd=max_position_usd,
                max_position_fraction=max_position_fraction,
                kelly_market_blend=kelly_market_blend,
                edge_position_full_cap_edge=edge_position_full_cap_edge,
                edge_position_min_multiplier=edge_position_min_multiplier,
                min_trade_usd=min_trade_usd,
            )
            for label, rows in month_slices
        ],
    }


def _cap_fraction_slice_replays(
    slices: list[tuple[str, list[dict[str, Any]]]],
    settings: SignalSettings,
    *,
    bankroll_usd: float,
    kelly_fraction: float,
    compound_kelly_sizing: bool,
    max_position_usd: float,
    kelly_market_blend: float,
    edge_position_full_cap_edge: float,
    edge_position_min_multiplier: float,
    min_trade_usd: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cap_fraction in (0.05, 0.10, 0.15, 0.20, 0.225, 0.25):
        for label, slice_rows in slices:
            replay = _replay_slice(
                label,
                slice_rows,
                settings,
                bankroll_usd=bankroll_usd,
                kelly_fraction=kelly_fraction,
                compound_kelly_sizing=compound_kelly_sizing,
                max_position_usd=max_position_usd,
                max_position_fraction=cap_fraction,
                kelly_market_blend=kelly_market_blend,
                edge_position_full_cap_edge=edge_position_full_cap_edge,
                edge_position_min_multiplier=edge_position_min_multiplier,
                min_trade_usd=min_trade_usd,
            )
            replay["variant"] = f"max_position_fraction_{cap_fraction:.2f}"
            replay["cap_fraction"] = cap_fraction
            rows.append(replay)
    return rows


def _replay_slice(
    label: str,
    rows: list[dict[str, Any]],
    settings: SignalSettings,
    *,
    bankroll_usd: float,
    kelly_fraction: float,
    compound_kelly_sizing: bool,
    max_position_usd: float,
    max_position_fraction: Optional[float],
    kelly_market_blend: float,
    edge_position_full_cap_edge: float,
    edge_position_min_multiplier: float,
    min_trade_usd: float,
) -> dict[str, Any]:
    replay = _json_kelly_replay(
        label,
        rows,
        settings,
        bankroll_usd=bankroll_usd,
        kelly_fraction=kelly_fraction,
        compound_kelly_sizing=compound_kelly_sizing,
        max_position_usd=max_position_usd,
        max_position_fraction=max_position_fraction,
        kelly_market_blend=kelly_market_blend,
        edge_position_full_cap_edge=edge_position_full_cap_edge,
        edge_position_min_multiplier=edge_position_min_multiplier,
        min_trade_usd=min_trade_usd,
    )
    session_keys = _sorted_session_keys(rows)
    return {
        "slice": label,
        "scored_rows": len(rows),
        "session_count": len(session_keys),
        "first_session": session_keys[0] if session_keys else None,
        "last_session": session_keys[-1] if session_keys else None,
        **replay,
    }


def _sorted_session_keys(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("generated_at")) for row in rows if row.get("generated_at")})


def _session_fraction_slice(
    label: str,
    rows: list[dict[str, Any]],
    session_keys: list[str],
    start_fraction: float,
    end_fraction: float,
) -> tuple[str, list[dict[str, Any]]]:
    if not session_keys:
        return label, []
    start = max(0, min(len(session_keys), math.floor(len(session_keys) * start_fraction)))
    end = max(start, min(len(session_keys), math.ceil(len(session_keys) * end_fraction)))
    if end == start and start < len(session_keys):
        end += 1
    selected_sessions = set(session_keys[start:end])
    return label, [row for row in rows if row.get("generated_at") in selected_sessions]


def _entry_months(rows: list[dict[str, Any]]) -> list[str]:
    months = {
        month
        for row in rows
        if (month := _month_key(row.get("generated_at"))) is not None
    }
    return sorted(months)


def _rows_for_entry_month(rows: list[dict[str, Any]], month: str) -> list[dict[str, Any]]:
    return [row for row in rows if _month_key(row.get("generated_at")) == month]


def _counterfactual_kelly_replays(
    scored: list[dict[str, Any]],
    settings: SignalSettings,
    *,
    bankroll_usd: float,
    kelly_fraction: float,
    compound_kelly_sizing: bool,
    max_position_usd: float,
    max_position_fraction: Optional[float],
    kelly_market_blend: float,
    edge_position_full_cap_edge: float,
    edge_position_min_multiplier: float,
    min_trade_usd: float,
) -> list[dict[str, Any]]:
    variants = (
        ("current", settings),
        ("looser_no_side_min_edge_0.00", replace(settings, no_side_min_edge=0.0)),
        ("looser_no_side_min_edge_0.05", replace(settings, no_side_min_edge=0.05)),
        ("stricter_no_side_min_edge_0.12", replace(settings, no_side_min_edge=0.12)),
        ("stricter_no_side_min_edge_0.15", replace(settings, no_side_min_edge=0.15)),
        ("looser_no_side_high_confidence_edge_0.00", replace(settings, no_side_high_confidence_min_edge=0.0)),
        ("stricter_no_side_high_confidence_edge_0.05", replace(settings, no_side_high_confidence_min_edge=0.05)),
        ("old_absolute_no_side_edge_0.10", replace(settings, no_side_high_confidence_min_edge=settings.no_side_min_edge)),
        ("looser_entry_agreement_0.65", replace(settings, min_model_agreement=0.65)),
        ("looser_fair_value_0.60", replace(settings, min_signal_fair_value=0.60)),
        ("stricter_fair_value_0.80", replace(settings, min_signal_fair_value=0.80)),
        ("allow_bounded_bucket_entries", replace(settings, allow_bounded_bucket_entries=True)),
        ("disable_bounded_no_side_entries", replace(settings, allow_bounded_no_side_entries=False)),
        (
            "allow_bounded_strict_fv_0.80",
            replace(settings, allow_bounded_bucket_entries=True, min_signal_fair_value=0.80),
        ),
        (
            "allow_bounded_strict_price_0.20",
            replace(settings, allow_bounded_bucket_entries=True, min_price=0.20, yes_side_min_price=0.20),
        ),
        (
            "allow_bounded_strict_fv_0.80_price_0.20",
            replace(settings, allow_bounded_bucket_entries=True, min_signal_fair_value=0.80, min_price=0.20, yes_side_min_price=0.20),
        ),
        (
            "allow_bounded_strict_edge_0.15",
            replace(settings, allow_bounded_bucket_entries=True, min_edge=0.15),
        ),
        ("looser_min_price_0.10", replace(settings, min_price=0.10)),
        ("stricter_min_price_0.20", replace(settings, min_price=0.20)),
        ("stricter_min_price_0.35", replace(settings, min_price=0.35)),
        ("looser_yes_side_min_price_0.125", replace(settings, yes_side_min_price=0.125)),
        ("stricter_yes_side_min_price_0.25", replace(settings, yes_side_min_price=0.25)),
        ("legacy_no_side_counter_event_0.08", replace(settings, no_side_max_counter_event_probability=0.08)),
        ("legacy_no_side_counter_event_0.09", replace(settings, no_side_max_counter_event_probability=0.09)),
        ("selected_no_side_counter_event_0.10", replace(settings, no_side_max_counter_event_probability=0.10)),
        ("candidate_no_side_counter_event_0.11", replace(settings, no_side_max_counter_event_probability=0.11)),
        ("candidate_no_side_counter_event_0.12", replace(settings, no_side_max_counter_event_probability=0.12)),
        ("looser_no_side_counter_event_0.20", replace(settings, no_side_max_counter_event_probability=0.20)),
        ("looser_no_side_counter_event_0.30", replace(settings, no_side_max_counter_event_probability=0.30)),
        (
            "utc12_relaxed_no_side_counter_event_0.15",
            replace(settings, no_side_relaxed_counter_event_probability=0.15, no_side_relaxed_counter_event_hours_utc=(12,)),
        ),
        (
            "utc12_relaxed_no_side_counter_event_0.20",
            replace(settings, no_side_relaxed_counter_event_probability=0.20, no_side_relaxed_counter_event_hours_utc=(12,)),
        ),
        (
            "utc12_relaxed_no_side_counter_event_0.30",
            replace(settings, no_side_relaxed_counter_event_probability=0.30, no_side_relaxed_counter_event_hours_utc=(12,)),
        ),
        ("disabled_no_side_counter_event_gate", replace(settings, no_side_max_counter_event_probability=1.0)),
        ("legacy_no_side_max_price_0.90", replace(settings, no_side_max_price=0.90)),
        ("prior_no_side_max_price_0.93", replace(settings, no_side_max_price=0.93)),
        ("candidate_no_side_max_price_0.94", replace(settings, no_side_max_price=0.94)),
        ("selected_no_side_max_price_0.95", replace(settings, no_side_max_price=0.95)),
        ("strict_hold_agreement_1.00", replace(settings, hold_min_model_agreement=1.0)),
        ("legacy_hold_no_side_counter_event_0.09", replace(settings, hold_no_side_max_counter_event_probability=0.09)),
        ("selected_hold_no_side_counter_event_0.15", replace(settings, hold_no_side_max_counter_event_probability=0.15)),
        ("looser_hold_no_side_counter_event_0.20", replace(settings, hold_no_side_max_counter_event_probability=0.20)),
        ("looser_hold_no_side_counter_event_0.30", replace(settings, hold_no_side_max_counter_event_probability=0.30)),
        ("disabled_hold_no_side_counter_event_gate", replace(settings, hold_no_side_max_counter_event_probability=1.0)),
        ("trim_valid_holds_to_kelly_target", replace(settings, preserve_valid_holds=False)),
    )
    setting_replays = [
        _json_kelly_replay(
            name,
            scored,
            variant,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, variant in variants
    ]
    sizing_replays = [
        _json_kelly_replay(
            name,
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=variant_kelly,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=variant_max_position,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, variant_kelly, variant_max_position in (
            ("conservative_fractional_kelly_0.10", 0.10, max_position_usd),
            ("conservative_fractional_kelly_0.25", 0.25, max_position_usd),
            ("aggressive_fractional_kelly_0.50", 0.50, max_position_usd),
            ("aggressive_fractional_kelly_0.75", 0.75, max_position_usd),
            ("conservative_max_position_20", kelly_fraction, min(max_position_usd, 20.0)),
            ("aggressive_max_position_100", kelly_fraction, max(max_position_usd, 100.0)),
        )
    ]
    cap_fraction_replays = [
        _json_kelly_replay(
            name,
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=variant_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, variant_fraction in (
            ("max_position_fraction_0.025", 0.025),
            ("max_position_fraction_0.05", 0.05),
            ("max_position_fraction_0.075", 0.075),
            ("max_position_fraction_0.10", 0.10),
            ("max_position_fraction_0.15", 0.15),
            ("max_position_fraction_0.20", 0.20),
            ("max_position_fraction_0.225", 0.225),
            ("max_position_fraction_0.25", 0.25),
        )
    ]
    strict_no_side_sizing_replays = [
        _json_kelly_replay(
            name,
            scored,
            replace(settings, no_side_min_edge=0.10),
            bankroll_usd=bankroll_usd,
            kelly_fraction=variant_kelly,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, variant_kelly in (
            ("strict_no_side_edge_0.10_fractional_kelly_0.50", 0.50),
            ("strict_no_side_edge_0.10_fractional_kelly_0.75", 0.75),
        )
    ]
    hour_replays = [
        _json_kelly_replay(
            name,
            _rows_for_entry_hours(scored, hours),
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, hours in (
            ("live_forward_entry_hours_utc_0_6_12_18", set(LIVE_FORWARD_ENTRY_HOURS_UTC)),
            *(
                (f"entry_hours_utc_{hour}_only", {hour})
                for hour in LIVE_FORWARD_ENTRY_HOURS_UTC
            ),
        )
    ]
    hour_tail_replays = [
        _json_kelly_replay(
            name,
            _rows_for_entry_hours(scored, hours),
            replace(settings, no_side_max_counter_event_probability=tail_threshold),
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, hours, tail_threshold in (
            ("entry_hours_utc_12_no_side_counter_event_0.13", {12}, 0.13),
            ("entry_hours_utc_12_no_side_counter_event_0.20", {12}, 0.20),
        )
    ]
    compounding_replays = [
        _json_kelly_replay(
            name,
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=variant_compounding,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, variant_compounding in (
            ("fixed_starting_bankroll_kelly_sizing", False),
            ("compound_current_equity_kelly_sizing", True),
        )
    ]
    calibration_sizing_replays = [
        _json_kelly_replay(
            name,
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, blend in (
            ("kelly_market_blend_0.10", 0.10),
            ("kelly_market_blend_0.25", 0.25),
            ("kelly_market_blend_0.50", 0.50),
        )
    ]
    edge_scaled_cap_replays = [
        _json_kelly_replay(
            name,
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=variant_max_position,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=full_cap_edge,
            edge_position_min_multiplier=min_multiplier,
            min_trade_usd=min_trade_usd,
        )
        for name, variant_max_position, full_cap_edge, min_multiplier in (
            ("edge_scaled_cap_175_full_0.25_floor_0.35", max(max_position_usd, 175.0), 0.25, 0.35),
            ("edge_scaled_cap_100_full_0.35_floor_0.35", max_position_usd, 0.35, 0.35),
            ("edge_scaled_cap_175_full_0.35_floor_0.35", max(max_position_usd, 175.0), 0.35, 0.35),
            ("edge_scaled_cap_175_full_0.30_floor_0.35", max(max_position_usd, 175.0), 0.30, 0.35),
        )
    ]
    high_price_damping_replays = [
        _json_kelly_replay(
            "high_price_low_edge_damped_cap_0.85_edge_0.12_x0.35_full_0.25",
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max(max_position_usd, 175.0),
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=0.25,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
            high_price_damping_threshold=0.85,
            high_price_damping_edge=0.12,
            high_price_damping_multiplier=0.35,
        )
    ]
    exit_management_replays = [
        _json_kelly_replay(
            "force_hold_existing_positions_to_settlement",
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
            force_hold_existing_positions=True,
        )
    ]
    partial_exit_replays = [
        _json_kelly_replay(
            name,
            scored,
            settings,
            bankroll_usd=bankroll_usd,
            kelly_fraction=kelly_fraction,
            compound_kelly_sizing=compound_kelly_sizing,
            max_position_usd=max_position_usd,
            max_position_fraction=max_position_fraction,
            kelly_market_blend=kelly_market_blend,
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
            min_trade_usd=min_trade_usd,
            invalid_hold_partial_exit_fraction=fraction,
            invalid_hold_partial_exit_min_fair_value=min_fair_value,
            invalid_hold_partial_exit_min_price=min_price,
            invalid_hold_partial_exit_max_price=max_price,
        )
        for name, fraction, min_fair_value, min_price, max_price in (
            ("partial_invalid_hold_exit_x0.50_fv0.90_price0.50_0.80", 0.50, 0.90, 0.50, 0.80),
            ("partial_invalid_hold_exit_x0.50_fv0.95_price0.50_0.80", 0.50, 0.95, 0.50, 0.80),
            ("partial_invalid_hold_exit_x0.25_fv0.90_price0.50_0.80", 0.25, 0.90, 0.50, 0.80),
            ("partial_invalid_hold_exit_x0.50_fv0.90_price0.50_0.70", 0.50, 0.90, 0.50, 0.70),
            ("partial_invalid_hold_exit_x0.50_fv0.90_price0.50_0.65", 0.50, 0.90, 0.50, 0.65),
        )
    ]
    return (
        setting_replays
        + sizing_replays
        + cap_fraction_replays
        + strict_no_side_sizing_replays
        + hour_replays
        + hour_tail_replays
        + compounding_replays
        + calibration_sizing_replays
        + edge_scaled_cap_replays
        + high_price_damping_replays
        + exit_management_replays
        + partial_exit_replays
    )


def _selected_candidate_rows(
    scored: list[dict[str, Any]],
    settings: SignalSettings,
    *,
    min_signal_fair_value: Optional[float] = None,
    min_price: Optional[float] = None,
    yes_side_min_price: Optional[float] = None,
    no_side_max_counter_event_probability: Optional[float] = None,
    no_side_min_edge: Optional[float] = None,
    min_model_agreement: Optional[float] = None,
) -> list[dict[str, Any]]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        if _json_signal_candidate(
            row,
            settings,
            min_signal_fair_value=min_signal_fair_value,
            min_price=min_price,
            yes_side_min_price=yes_side_min_price,
            no_side_max_counter_event_probability=no_side_max_counter_event_probability,
            no_side_min_edge=no_side_min_edge,
            min_model_agreement=min_model_agreement,
        ):
            sessions.setdefault(str(row.get("generated_at") or ""), []).append(row)
    selected_by_token: dict[str, dict[str, Any]] = {}
    for session_time in sorted(sessions):
        by_city_day: dict[tuple[Any, Any], dict[str, Any]] = {}
        for row in sorted(sessions[session_time], key=lambda item: float(item.get("edge") or 0.0), reverse=True):
            by_city_day.setdefault((row.get("city"), row.get("target_date")), row)
        for row in by_city_day.values():
            token = str(row.get("token_id") or "")
            if token:
                selected_by_token.setdefault(token, row)
    return list(selected_by_token.values())


def _json_kelly_replay(
    name: str,
    scored: list[dict[str, Any]],
    settings: SignalSettings,
    *,
    bankroll_usd: float,
    kelly_fraction: float,
    compound_kelly_sizing: bool = False,
    max_new_exposure_usd_per_run: Optional[float] = None,
    max_new_exposure_fraction_per_run: Optional[float] = None,
    new_exposure_target_positions_per_run: Optional[float] = None,
    kelly_sizing_bankroll_fraction_per_run: Optional[float] = None,
    max_position_usd: float,
    max_position_fraction: Optional[float] = None,
    kelly_market_blend: float = 0.0,
    edge_position_full_cap_edge: float = 0.0,
    edge_position_min_multiplier: float = 0.35,
    min_trade_usd: float,
    high_price_damping_threshold: Optional[float] = None,
    high_price_damping_edge: float = 0.0,
    high_price_damping_multiplier: float = 1.0,
    force_hold_existing_positions: bool = False,
    invalid_hold_partial_exit_fraction: Optional[float] = None,
    invalid_hold_partial_exit_min_fair_value: float = 0.90,
    invalid_hold_partial_exit_min_price: float = 0.50,
    invalid_hold_partial_exit_max_price: float = 0.80,
    include_executions: bool = False,
) -> dict[str, Any]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        generated_at = row.get("generated_at")
        if generated_at:
            sessions.setdefault(str(generated_at), []).append(row)

    cash = bankroll_usd
    positions: dict[str, dict[str, Any]] = {}
    executions: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    signal_count = 0
    active_partial_exit_fraction = (
        settings.invalid_hold_partial_exit_fraction
        if invalid_hold_partial_exit_fraction is None
        else invalid_hold_partial_exit_fraction
    )
    active_partial_exit_min_fair_value = (
        settings.invalid_hold_partial_exit_min_fair_value
        if invalid_hold_partial_exit_fraction is None
        else invalid_hold_partial_exit_min_fair_value
    )
    active_partial_exit_min_price = (
        settings.invalid_hold_partial_exit_min_price
        if invalid_hold_partial_exit_fraction is None
        else invalid_hold_partial_exit_min_price
    )
    active_partial_exit_max_price = (
        settings.invalid_hold_partial_exit_max_price
        if invalid_hold_partial_exit_fraction is None
        else invalid_hold_partial_exit_max_price
    )
    for session_key in sorted(sessions):
        session_time = _parse_dt(session_key)
        if session_time is None:
            continue
        cash = _json_settle_due_positions(positions, executions, cash, session_time)
        rows = sessions[session_key]
        selected_tokens = _json_selected_tokens(rows, settings)
        signal_count += len(selected_tokens)
        sizing_base = _json_portfolio_equity(cash, positions) if compound_kelly_sizing else bankroll_usd
        buy_budget_remaining = _new_exposure_budget_usd(
            sizing_base,
            max_new_exposure_usd_per_run,
            max_new_exposure_fraction_per_run,
        )
        per_buy_budget = _new_exposure_per_buy_budget_usd(
            buy_budget_remaining,
            new_exposure_target_positions_per_run,
        )
        for row in sorted(rows, key=lambda item: float(item.get("edge") or 0.0), reverse=True):
            token = str(row.get("token_id") or "")
            if not token:
                continue
            price = _optional_float(row.get("market_price"))
            if price is None or price <= 0:
                continue
            current = positions.get(token)
            current_shares = float(current["shares"]) if current else 0.0
            current_notional = current_shares * price
            hold_eligible = current is not None and (force_hold_existing_positions or _json_hold_candidate(row, settings))
            target_notional = 0.0
            if token in selected_tokens:
                sizing_bankroll = _json_portfolio_equity(cash, positions) if compound_kelly_sizing else bankroll_usd
                sizing_bankroll = _kelly_sizing_bankroll_usd(sizing_bankroll, kelly_sizing_bankroll_fraction_per_run)
                row_max_position_usd = _damped_max_position_usd(
                    max_position_usd,
                    row,
                    high_price_damping_threshold=high_price_damping_threshold,
                    high_price_damping_edge=high_price_damping_edge,
                    high_price_damping_multiplier=high_price_damping_multiplier,
                )
                target_notional = _json_kelly_target_notional(
                    row,
                    sizing_bankroll,
                    kelly_fraction,
                    row_max_position_usd,
                    max_position_fraction,
                    kelly_market_blend,
                    edge_position_full_cap_edge,
                    edge_position_min_multiplier,
                )
                if hold_eligible and settings.preserve_valid_holds:
                    target_notional = max(target_notional, current_notional)
            elif hold_eligible:
                target_notional = current_notional
            delta = target_notional - current_notional
            delta = _json_partial_exit_delta(
                delta,
                row,
                invalid_hold_partial_exit_fraction=active_partial_exit_fraction,
                invalid_hold_partial_exit_min_fair_value=active_partial_exit_min_fair_value,
                invalid_hold_partial_exit_min_price=active_partial_exit_min_price,
                invalid_hold_partial_exit_max_price=active_partial_exit_max_price,
            )
            if delta > 0 and delta >= min_trade_usd:
                if buy_budget_remaining is not None:
                    delta = min(delta, buy_budget_remaining)
                if per_buy_budget is not None:
                    delta = min(delta, per_buy_budget)
                if buy_budget_remaining is not None or per_buy_budget is not None:
                    if delta < min_trade_usd:
                        continue
                notional = min(delta, cash)
                if notional < min_trade_usd:
                    continue
                shares = notional / price
                cash -= notional
                if buy_budget_remaining is not None:
                    buy_budget_remaining = max(0.0, buy_budget_remaining - notional)
                if current is None:
                    positions[token] = {
                        "shares": shares,
                        "cost_basis": notional,
                        "last_price": price,
                        "row": row,
                    }
                else:
                    current["shares"] += shares
                    current["cost_basis"] += notional
                    current["last_price"] = price
                    current["row"] = row
                executions.append(
                    {
                        "action": "BUY",
                        "token_id": token,
                        "shares": shares,
                        "price": price,
                        "notional_usd": notional,
                        "realized_pnl_usd": 0.0,
                        "row": row,
                    }
                )
            elif delta < 0 and current is not None and abs(delta) >= min_trade_usd:
                shares = min(current_shares, abs(delta) / price)
                cash = _json_sell_position(positions, executions, cash, token, shares, row)
            elif current is not None:
                current["last_price"] = _json_exit_price(row)
                current["row"] = row
        equity_curve.append(_json_equity_snapshot(session_key, cash, positions))

    for token in list(positions):
        cash = _json_settle_position(positions, executions, cash, token)
    equity_curve.append(_json_equity_snapshot("final", cash, positions))

    trade_tokens = {execution["token_id"] for execution in executions if execution["action"] == "BUY"}
    trade_pnl: dict[str, float] = {token: 0.0 for token in trade_tokens}
    trade_buy_notional: dict[str, float] = {token: 0.0 for token in trade_tokens}
    trade_payout: dict[str, Optional[int]] = {token: None for token in trade_tokens}
    weather_mismatch_trades = set()
    weather_ambiguous_trades = set()
    for execution in executions:
        token = str(execution.get("token_id") or "")
        if not token:
            continue
        if token in trade_pnl:
            trade_pnl[token] += float(execution.get("realized_pnl_usd") or 0.0)
        if execution.get("action") == "BUY":
            trade_buy_notional[token] = trade_buy_notional.get(token, 0.0) + float(execution.get("notional_usd") or 0.0)
        row = execution.get("row") or {}
        if token in trade_payout and row.get("polymarket_payout") in (0, 1):
            trade_payout[token] = int(row["polymarket_payout"])
        if row.get("weather_ambiguous"):
            weather_ambiguous_trades.add(token)
        elif row.get("weather_outcome") in (0, 1) and row.get("weather_outcome") != row.get("polymarket_payout"):
            weather_mismatch_trades.add(token)

    pnl = cash - bankroll_usd
    winners = sum(1 for token in trade_tokens if trade_pnl.get(token, 0.0) > 0)
    event_winners = sum(1 for token in trade_tokens if trade_payout.get(token) == 1)
    event_losers = sum(1 for token in trade_tokens if trade_payout.get(token) == 0)
    event_loss_pnl = sum(trade_pnl.get(token, 0.0) for token in trade_tokens if trade_payout.get(token) == 0)
    event_win_pnl = sum(trade_pnl.get(token, 0.0) for token in trade_tokens if trade_payout.get(token) == 1)
    profitable_event_losers = sum(1 for token in trade_tokens if trade_payout.get(token) == 0 and trade_pnl.get(token, 0.0) > 0)
    unprofitable_event_winners = [
        token
        for token in trade_tokens
        if trade_payout.get(token) == 1 and trade_pnl.get(token, 0.0) < 0
    ]
    unprofitable_event_winner_pnl = sum(trade_pnl.get(token, 0.0) for token in unprofitable_event_winners)
    buy_notional = sum(trade_buy_notional.values())
    replay_concentration = _pnl_concentration_from_values(list(trade_pnl.values()))
    trade_diagnostics = _trade_performance_diagnostics(
        [_json_replay_execution_for_trade_diagnostics(execution) for execution in executions]
    )
    performance_diagnostics = _performance_diagnostics_from_equity_curve(equity_curve, bankroll_usd)
    max_drawdown_usd, max_drawdown_pct = _equity_curve_drawdown(equity_curve)
    cash_points = [
        float(snapshot.get("cash") or 0.0)
        for snapshot in equity_curve
        if isinstance(snapshot, Mapping) and snapshot.get("cash") is not None
    ]
    min_cash_usd = min(cash_points) if cash_points else None
    result = {
        "variant": name,
        "ending_equity_usd": round(cash, 2),
        "pnl_usd": round(pnl, 2),
        "return_pct": round(pnl / bankroll_usd, 4) if bankroll_usd else None,
        "performance_diagnostics": performance_diagnostics,
        "signals": signal_count,
        "trade_count": len(trade_tokens),
        "winning_trades": winners,
        "hit_rate": round(winners / len(trade_tokens), 4) if trade_tokens else None,
        "event_winning_trades": event_winners,
        "event_losing_trades": event_losers,
        "event_hit_rate": round(event_winners / (event_winners + event_losers), 4) if event_winners + event_losers else None,
        "event_win_pnl_usd": round(event_win_pnl, 2),
        "event_loss_pnl_usd": round(event_loss_pnl, 2),
        "profitable_event_loser_trades": profitable_event_losers,
        "unprofitable_event_winner_trades": len(unprofitable_event_winners),
        "unprofitable_event_winner_pnl_usd": round(unprofitable_event_winner_pnl, 2),
        "pnl_concentration": replay_concentration,
        "top_1_pnl_share": replay_concentration["top_1_pnl_share"],
        "buy_notional_usd": round(buy_notional, 4),
        "return_on_buy_notional": round(pnl / buy_notional, 4) if buy_notional else None,
        "executions": len(executions),
        "buys": sum(1 for execution in executions if execution["action"] == "BUY"),
        "sells": sum(1 for execution in executions if execution["action"] == "SELL"),
        "settlements": sum(1 for execution in executions if execution["action"] == "SETTLE"),
        "min_equity_usd": _equity_curve_min(equity_curve),
        "min_cash_usd": round(min_cash_usd, 4) if min_cash_usd is not None else None,
        "min_cash_pct": round(min_cash_usd / bankroll_usd, 4) if min_cash_usd is not None and bankroll_usd else None,
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "weather_mismatch_trades": len(weather_mismatch_trades),
        "weather_ambiguous_trades": len(weather_ambiguous_trades),
        "equity_curve": equity_curve,
        "exit_management": _json_replay_exit_management(executions),
        "trade_diagnostics": trade_diagnostics,
        "settings": {
            "min_price": settings.min_price,
            "yes_side_min_price": settings.yes_side_min_price,
            "min_signal_fair_value": settings.min_signal_fair_value,
            "min_model_agreement": settings.min_model_agreement,
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
            "hold_min_model_agreement": settings.hold_min_model_agreement,
            "hold_min_fair_value": settings.hold_min_fair_value,
            "hold_market_confirmation_price": settings.hold_market_confirmation_price,
            "hold_market_confirmation_min_fair_value": settings.hold_market_confirmation_min_fair_value,
            "preserve_valid_holds": settings.preserve_valid_holds,
            "allow_bounded_bucket_entries": settings.allow_bounded_bucket_entries,
            "allow_bounded_no_side_entries": settings.allow_bounded_no_side_entries,
            "bounded_bucket_min_edge": settings.bounded_bucket_min_edge,
            "bounded_bucket_min_fair_value": settings.bounded_bucket_min_fair_value,
            "bounded_bucket_min_model_agreement": settings.bounded_bucket_min_model_agreement,
            "bounded_bucket_min_price": settings.bounded_bucket_min_price,
            "compound_kelly_sizing": compound_kelly_sizing,
            "max_new_exposure_usd_per_run": max_new_exposure_usd_per_run,
            "max_new_exposure_fraction_per_run": max_new_exposure_fraction_per_run,
            "new_exposure_target_positions_per_run": new_exposure_target_positions_per_run,
            "kelly_sizing_bankroll_fraction_per_run": kelly_sizing_bankroll_fraction_per_run,
            "max_position_usd": max_position_usd,
            "max_position_fraction": max_position_fraction,
            "kelly_market_blend": kelly_market_blend,
            "edge_position_full_cap_edge": edge_position_full_cap_edge,
            "edge_position_min_multiplier": edge_position_min_multiplier,
            "high_price_damping_threshold": high_price_damping_threshold,
            "high_price_damping_edge": high_price_damping_edge,
            "high_price_damping_multiplier": high_price_damping_multiplier,
            "force_hold_existing_positions": force_hold_existing_positions,
            "invalid_hold_partial_exit_fraction": active_partial_exit_fraction,
            "invalid_hold_partial_exit_min_fair_value": active_partial_exit_min_fair_value,
            "invalid_hold_partial_exit_min_price": active_partial_exit_min_price,
            "invalid_hold_partial_exit_max_price": active_partial_exit_max_price,
        },
    }
    if include_executions:
        result["executions_detail"] = executions
    return result


def _json_replay_execution_for_trade_diagnostics(execution: Mapping[str, Any]) -> dict[str, Any]:
    row = execution.get("row") or {}
    payout = row.get("polymarket_payout")
    if payout not in (0, 1):
        payout = row.get("payout")
    model_probabilities = row.get("model_probabilities") or {}
    return {
        "action": execution.get("action"),
        "executed_at": row.get("generated_at"),
        "token_id": execution.get("token_id"),
        "question": row.get("question"),
        "bucket": row.get("bucket") or row.get("bucket_label"),
        "city": row.get("city"),
        "target_date": row.get("target_date"),
        "shares": execution.get("shares"),
        "price": execution.get("price"),
        "notional_usd": execution.get("notional_usd"),
        "realized_pnl_usd": execution.get("realized_pnl_usd"),
        "fair_value": row.get("fair_value"),
        "edge": row.get("edge"),
        "model_agreement": row.get("model_agreement"),
        "no_side_counter_event_probability": (
            no_side_counter_event_probability(model_probabilities)
            if _json_is_no_side_row(row) and isinstance(model_probabilities, Mapping)
            else None
        ),
        "side": row.get("side"),
        "bucket_lower_f": row.get("bucket_lower_f"),
        "bucket_upper_f": row.get("bucket_upper_f"),
        "bucket_shape": _bucket_shape(row.get("bucket_lower_f"), row.get("bucket_upper_f")),
        "polymarket_payout": payout,
        "weather_outcome": row.get("weather_outcome"),
        "weather_ambiguous": row.get("weather_ambiguous"),
    }


def _damped_max_position_usd(
    max_position_usd: float,
    row: Mapping[str, Any],
    *,
    high_price_damping_threshold: Optional[float],
    high_price_damping_edge: float,
    high_price_damping_multiplier: float,
) -> float:
    if high_price_damping_threshold is None or high_price_damping_multiplier >= 1.0:
        return max_position_usd
    price = _optional_float(row.get("market_price"))
    edge = _optional_float(row.get("edge"))
    if price is None or edge is None:
        return max_position_usd
    if price >= high_price_damping_threshold and edge < high_price_damping_edge:
        return max(0.0, max_position_usd * max(0.0, high_price_damping_multiplier))
    return max_position_usd


def _json_partial_exit_delta(
    delta: float,
    row: Mapping[str, Any],
    *,
    invalid_hold_partial_exit_fraction: Optional[float],
    invalid_hold_partial_exit_min_fair_value: float,
    invalid_hold_partial_exit_min_price: float,
    invalid_hold_partial_exit_max_price: float,
) -> float:
    if delta >= 0.0 or invalid_hold_partial_exit_fraction is None:
        return delta
    fraction = max(0.0, min(1.0, invalid_hold_partial_exit_fraction))
    if fraction >= 1.0:
        return delta
    price = _optional_float(row.get("market_price"))
    fair_value = _optional_float(row.get("fair_value"))
    if price is None or fair_value is None:
        return delta
    if price < invalid_hold_partial_exit_min_price or price > invalid_hold_partial_exit_max_price:
        return delta
    if fair_value < invalid_hold_partial_exit_min_fair_value:
        return delta
    return delta * fraction


def _json_selected_tokens(rows: list[dict[str, Any]], settings: SignalSettings) -> set[str]:
    by_city_day: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: float(item.get("edge") or 0.0), reverse=True):
        if not _json_signal_candidate(row, settings):
            continue
        by_city_day.setdefault((row.get("city"), row.get("target_date")), row)
    return {str(row.get("token_id")) for row in by_city_day.values() if row.get("token_id")}


def _json_hold_candidate(row: Mapping[str, Any], settings: SignalSettings) -> bool:
    try:
        price = float(row.get("market_price"))
        fair_value = float(row.get("fair_value"))
        model_count = int(row.get("model_count") or 0)
        agreement = float(row.get("model_agreement") or 0.0)
    except (TypeError, ValueError):
        return False
    if price < settings.min_price:
        return False
    no_side_hold_counter_threshold = _json_active_no_side_hold_counter_event_threshold(row, settings)
    if _json_is_no_side_row(row) and _json_fails_no_side_counter_event_gate(
        row,
        settings,
        no_side_max_counter_event_probability=no_side_hold_counter_threshold,
    ):
        return False
    if (
        not settings.allow_bounded_bucket_entries
        and row.get("bucket_lower_f") is not None
        and row.get("bucket_upper_f") is not None
    ):
        return False
    if (
        _json_is_no_side_row(row)
        and not settings.allow_bounded_no_side_entries
        and row.get("bucket_lower_f") is not None
        and row.get("bucket_upper_f") is not None
    ):
        return False
    if model_count < settings.min_model_count:
        return False
    if agreement < settings.hold_min_model_agreement:
        return False
    market_confirmed = (
        price >= settings.hold_market_confirmation_price
        and fair_value >= settings.hold_market_confirmation_min_fair_value
    )
    if fair_value < settings.hold_min_fair_value and not market_confirmed:
        return False
    return True


def _rows_for_entry_hours(scored: list[dict[str, Any]], hours: set[int]) -> list[dict[str, Any]]:
    rows = []
    for row in scored:
        generated_at = _parse_dt(row.get("generated_at"))
        if generated_at is not None and generated_at.astimezone(timezone.utc).hour in hours:
            rows.append(row)
    return rows


def _json_kelly_target_notional(
    row: Mapping[str, Any],
    bankroll_usd: float,
    kelly_fraction: float,
    max_position_usd: float,
    max_position_fraction: Optional[float] = None,
    kelly_market_blend: float = 0.0,
    edge_position_full_cap_edge: float = 0.0,
    edge_position_min_multiplier: float = 0.35,
) -> float:
    price = max(0.0001, min(0.9999, float(row.get("market_price") or 0.0)))
    fair_value = max(0.0, min(1.0, float(row.get("fair_value") or 0.0)))
    sizing_fair_value = _blend_probability_with_market(fair_value, price, kelly_market_blend)
    agreement = max(0.0, min(1.0, float(row.get("model_agreement") or 0.0)))
    raw_fraction = max(0.0, (sizing_fair_value - price) / max(0.0001, 1.0 - price))
    return min(
        _effective_max_position_usd(
            bankroll_usd,
            max_position_usd,
            max_position_fraction,
            edge=_optional_float(row.get("edge")),
            edge_position_full_cap_edge=edge_position_full_cap_edge,
            edge_position_min_multiplier=edge_position_min_multiplier,
        ),
        bankroll_usd * kelly_fraction * raw_fraction * agreement,
    )


def _json_portfolio_equity(cash: float, positions: Mapping[str, Mapping[str, Any]]) -> float:
    marked = 0.0
    for position in positions.values():
        try:
            marked += float(position.get("shares") or 0.0) * float(position.get("last_price") or 0.0)
        except (TypeError, ValueError):
            continue
    return max(0.0, cash + marked)


def _json_settle_due_positions(
    positions: dict[str, dict[str, Any]],
    executions: list[dict[str, Any]],
    cash: float,
    session_time: datetime,
) -> float:
    for token, position in list(positions.items()):
        row = position.get("row") or {}
        target_date = _parse_json_date(row.get("target_date"))
        if target_date is None:
            continue
        city = _city_from_display_name(str(row.get("city") or ""))
        local_date = session_time.date() if city is None else session_time.astimezone(_zoneinfo_or_utc(city)).date()
        if target_date >= local_date:
            continue
        cash = _json_settle_position(positions, executions, cash, token)
    return cash


def _json_settle_position(
    positions: dict[str, dict[str, Any]],
    executions: list[dict[str, Any]],
    cash: float,
    token: str,
) -> float:
    position = positions.get(token)
    if position is None:
        return cash
    row = position.get("row") or {}
    payout = int(row.get("payout") if row.get("payout") in (0, 1) else row.get("polymarket_payout") or 0)
    notional = float(position["shares"]) * payout
    realized = notional - float(position["cost_basis"])
    cash += notional
    executions.append(
        {
            "action": "SETTLE",
            "token_id": token,
            "shares": float(position["shares"]),
            "price": float(payout),
            "notional_usd": notional,
            "realized_pnl_usd": realized,
            "row": row,
        }
    )
    del positions[token]
    return cash


def _json_sell_position(
    positions: dict[str, dict[str, Any]],
    executions: list[dict[str, Any]],
    cash: float,
    token: str,
    shares: float,
    row: Mapping[str, Any],
) -> float:
    position = positions[token]
    shares = min(float(position["shares"]), shares)
    cost_reduction = float(position["cost_basis"]) * (shares / float(position["shares"])) if position["shares"] else 0.0
    price = _json_exit_price(row)
    notional = shares * price
    realized = notional - cost_reduction
    position["shares"] -= shares
    position["cost_basis"] -= cost_reduction
    position["last_price"] = price
    position["row"] = dict(row)
    cash += notional
    executions.append(
        {
            "action": "SELL",
            "token_id": token,
            "shares": shares,
            "price": price,
            "notional_usd": notional,
            "realized_pnl_usd": realized,
            "row": dict(row),
        }
    )
    if position["shares"] <= 1e-9:
        del positions[token]
    return cash


def _json_exit_price(row: Mapping[str, Any]) -> float:
    price = _optional_float(row.get("exit_price"))
    if price is None:
        price = _optional_float(row.get("market_price")) or 0.0
    return max(0.001, min(0.999, price))


def _parse_json_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _json_signal_candidate(
    row: Mapping[str, Any],
    settings: SignalSettings,
    *,
    min_signal_fair_value: Optional[float] = None,
    min_price: Optional[float] = None,
    yes_side_min_price: Optional[float] = None,
    no_side_max_counter_event_probability: Optional[float] = None,
    no_side_min_edge: Optional[float] = None,
    min_model_agreement: Optional[float] = None,
    ignore_entry_timing: bool = False,
) -> bool:
    if row.get("polymarket_payout") not in (0, 1):
        return False
    if not ignore_entry_timing and not row.get("entry_eligible", True):
        return False
    try:
        price = float(row.get("market_price"))
        fair_value = float(row.get("fair_value"))
        edge = float(row.get("edge"))
        model_count = int(row.get("model_count") or 0)
        agreement = float(row.get("model_agreement") or 0.0)
    except (TypeError, ValueError):
        return False
    active_min_price = settings.min_price if min_price is None else min_price
    active_yes_side_min_price = settings.yes_side_min_price if yes_side_min_price is None else yes_side_min_price
    active_min_fair_value = settings.min_signal_fair_value if min_signal_fair_value is None else min_signal_fair_value
    active_min_agreement = settings.min_model_agreement if min_model_agreement is None else min_model_agreement
    active_no_side_min_edge = settings.no_side_min_edge if no_side_min_edge is None else no_side_min_edge
    if price < active_min_price or price > settings.max_price:
        return False
    if not _json_is_no_side_row(row) and price < active_yes_side_min_price:
        return False
    if _json_is_no_side_row(row) and _fails_no_side_max_price_gate(price, settings):
        return False
    if (
        not settings.allow_bounded_bucket_entries
        and row.get("bucket_lower_f") is not None
        and row.get("bucket_upper_f") is not None
    ):
        return False
    if (
        _json_is_no_side_row(row)
        and not settings.allow_bounded_no_side_entries
        and row.get("bucket_lower_f") is not None
        and row.get("bucket_upper_f") is not None
    ):
        return False
    if fair_value < active_min_fair_value:
        return False
    if model_count < settings.min_model_count:
        return False
    if agreement < active_min_agreement:
        return False
    if edge < _json_required_buffered_edge(price, settings):
        return False
    if _json_fails_bounded_bucket_quality_gate(row, settings):
        return False
    if _json_is_no_side_row(row):
        no_side_edge_floor = min(active_no_side_min_edge, _required_no_side_min_edge(price, settings))
        if edge < no_side_edge_floor:
            return False
    if _json_is_no_side_row(row) and _json_fails_no_side_counter_event_gate(
        row,
        settings,
        no_side_max_counter_event_probability=no_side_max_counter_event_probability,
    ):
        return False
    bucket_width = _optional_float(row.get("bucket_width_f"))
    if (
        price < settings.low_price_exact_bucket_threshold
        and bucket_width is not None
        and 0.0 < bucket_width <= settings.exact_bucket_max_width_f
        and (fair_value < settings.low_price_exact_bucket_min_fair_value or edge < settings.low_price_exact_bucket_min_edge)
    ):
        return False
    if (
        price < settings.correlated_exact_bucket_max_price
        and bucket_width is not None
        and 0.0 < bucket_width <= settings.exact_bucket_max_width_f
        and agreement >= settings.correlated_exact_bucket_min_agreement
    ):
        return False
    return True


def _json_fails_bounded_bucket_quality_gate(row: Mapping[str, Any], settings: SignalSettings) -> bool:
    if row.get("bucket_lower_f") is None or row.get("bucket_upper_f") is None:
        return False
    try:
        price = float(row.get("market_price"))
        fair_value = float(row.get("fair_value"))
        edge = float(row.get("edge"))
        agreement = float(row.get("model_agreement") or 0.0)
        probability_stdev = float(row.get("probability_stdev") or 0.0)
    except (TypeError, ValueError):
        return True
    if (
        settings.bounded_bucket_max_probability_stdev is not None
        and probability_stdev > settings.bounded_bucket_max_probability_stdev
    ):
        return True
    return (
        price < settings.bounded_bucket_min_price
        or fair_value < settings.bounded_bucket_min_fair_value
        or edge < settings.bounded_bucket_min_edge
        or agreement < settings.bounded_bucket_min_model_agreement
    )


def _json_is_no_side_row(row: Mapping[str, Any]) -> bool:
    return row.get("side") == "NO" or str(row.get("bucket") or "").startswith("NO: ") or str(row.get("question") or "").startswith("NO: ")


def _bucket_shape(lower_f: Any, upper_f: Any) -> str:
    if lower_f is None and upper_f is None:
        return "unknown"
    if lower_f is None:
        return "lower_tail"
    if upper_f is None:
        return "upper_tail"
    return "bounded"


def _json_fails_no_side_counter_event_gate(
    row: Mapping[str, Any],
    settings: SignalSettings,
    *,
    no_side_max_counter_event_probability: Optional[float] = None,
) -> bool:
    threshold = _json_active_no_side_counter_event_threshold(
        row,
        settings,
        override=no_side_max_counter_event_probability,
    )
    if threshold is None or threshold >= 1.0:
        return False
    probabilities = row.get("model_probabilities") or {}
    if not isinstance(probabilities, Mapping):
        return True
    counter_probability = no_side_counter_event_probability(probabilities)
    return counter_probability is not None and counter_probability > threshold


def _json_active_no_side_counter_event_threshold(
    row: Mapping[str, Any],
    settings: SignalSettings,
    *,
    override: Optional[float] = None,
) -> Optional[float]:
    if override is not None:
        return override
    threshold = settings.no_side_max_counter_event_probability
    relaxed_threshold = settings.no_side_relaxed_counter_event_probability
    relaxed_hours = settings.no_side_relaxed_counter_event_hours_utc
    if relaxed_threshold is None or not relaxed_hours:
        return threshold
    generated_at = _parse_dt(row.get("generated_at"))
    if generated_at is None:
        return threshold
    return relaxed_threshold if generated_at.astimezone(timezone.utc).hour in set(relaxed_hours) else threshold


def _json_active_no_side_hold_counter_event_threshold(row: Mapping[str, Any], settings: SignalSettings) -> Optional[float]:
    threshold = settings.hold_no_side_max_counter_event_probability
    high_conviction_threshold = settings.hold_no_side_high_conviction_counter_event_probability
    if high_conviction_threshold is None:
        return threshold
    min_fair_value = settings.hold_no_side_high_conviction_min_fair_value
    min_edge = settings.hold_no_side_high_conviction_min_edge
    if min_fair_value is None or min_edge is None:
        return threshold
    if not _json_is_no_side_row(row):
        return threshold
    try:
        fair_value = float(row.get("fair_value"))
        edge = float(row.get("edge"))
    except (TypeError, ValueError):
        return threshold
    if fair_value < min_fair_value or edge < min_edge:
        return threshold
    return high_conviction_threshold


def _candidate_quality_row(name: str, threshold: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "setting": name,
            "threshold": threshold,
            "n": 0,
            "actual_rate": None,
            "avg_market_price": None,
            "avg_fair_value": None,
            "avg_edge": None,
            "flat_pnl_per_1usd": None,
            "flat_return_on_notional": None,
            "weather_mismatches": 0,
            "weather_ambiguous": 0,
        }
    pnl = 0.0
    notional = 0.0
    payouts = []
    prices = []
    fair_values = []
    edges = []
    weather_mismatches = 0
    weather_ambiguous = 0
    for row in rows:
        payout = int(row["polymarket_payout"])
        price = float(row["market_price"])
        fair_value = float(row["fair_value"])
        edge = float(row["edge"])
        payouts.append(payout)
        prices.append(price)
        fair_values.append(fair_value)
        edges.append(edge)
        pnl += payout / max(0.0001, price) - 1.0
        notional += 1.0
        if row.get("weather_ambiguous"):
            weather_ambiguous += 1
        elif row.get("weather_outcome") in (0, 1) and row.get("weather_outcome") != row.get("polymarket_payout"):
            weather_mismatches += 1
    return {
        "setting": name,
        "threshold": threshold,
        "n": len(rows),
        "actual_rate": round(sum(payouts) / len(payouts), 4),
        "avg_market_price": round(sum(prices) / len(prices), 4),
        "avg_fair_value": round(sum(fair_values) / len(fair_values), 4),
        "avg_edge": round(sum(edges) / len(edges), 4),
        "flat_pnl_per_1usd": round(pnl, 4),
        "flat_return_on_notional": round(pnl / notional, 4) if notional else None,
        "weather_mismatches": weather_mismatches,
        "weather_ambiguous": weather_ambiguous,
    }


def _selected_candidate_weather_validation(scored: list[dict[str, Any]], settings: SignalSettings) -> dict[str, Any]:
    selected = _selected_candidate_rows(scored, settings)
    ambiguous = [row for row in selected if row.get("weather_ambiguous")]
    mismatches = [
        row
        for row in selected
        if not row.get("weather_ambiguous")
        and row.get("weather_outcome") in (0, 1)
        and row.get("weather_outcome") != row.get("polymarket_payout")
    ]
    return {
        "method": (
            "Uses the same first-token-per-city-date-session selection rule as the JSON Kelly replay, "
            "then checks whether the selected candidates are independently weather-matched."
        ),
        "selected_candidate_count": len(selected),
        "weather_ambiguous_count": len(ambiguous),
        "weather_mismatch_count": len(mismatches),
        "quality": _candidate_quality_row("selected_candidates", 0.0, selected),
        "by_city": _selected_candidate_weather_groups(selected, "city"),
        "by_bucket_shape": _selected_candidate_weather_groups(selected, "bucket_shape"),
        "by_settlement_source": _selected_candidate_weather_groups(selected, "settlement_source"),
        "ambiguous_examples": [_selected_weather_example(row) for row in ambiguous[:10]],
        "mismatch_examples": [_selected_weather_example(row) for row in mismatches[:10]],
    }


def _selected_candidate_weather_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if key == "bucket_shape":
            group = _bucket_shape(row.get("bucket_lower_f"), row.get("bucket_upper_f"))
        else:
            group = str(row.get(key) or "none")
        grouped.setdefault(group, []).append(row)
    groups = []
    for group, items in grouped.items():
        quality = _candidate_quality_row(key, 0.0, items)
        groups.append(
            {
                "group": group,
                "n": quality["n"],
                "actual_rate": quality["actual_rate"],
                "avg_market_price": quality["avg_market_price"],
                "avg_fair_value": quality["avg_fair_value"],
                "avg_edge": quality["avg_edge"],
                "flat_pnl_per_1usd": quality["flat_pnl_per_1usd"],
                "weather_mismatches": quality["weather_mismatches"],
                "weather_ambiguous": quality["weather_ambiguous"],
            }
        )
    return sorted(groups, key=lambda item: (-int(item["n"]), str(item["group"])))


def _selected_weather_example(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question": row.get("question"),
        "city": row.get("city"),
        "target_date": row.get("target_date"),
        "generated_at": row.get("generated_at"),
        "market_price": row.get("market_price"),
        "fair_value": row.get("fair_value"),
        "edge": row.get("edge"),
        "side": row.get("side"),
        "polymarket_payout": row.get("polymarket_payout"),
        "weather_outcome": row.get("weather_outcome"),
        "observed_high_f": row.get("observed_high_f"),
        "settlement_source": row.get("settlement_source"),
        "token_id": row.get("token_id"),
    }


def _json_replay_exit_management(executions: list[dict[str, Any]]) -> dict[str, Any]:
    sells = [execution for execution in executions if execution.get("action") == "SELL"]
    rows = []
    unchecked_sells = 0
    for execution in sells:
        row = execution.get("row") or {}
        payout = _coerce_binary_outcome(row.get("polymarket_payout"))
        if payout is None:
            payout = _coerce_binary_outcome(row.get("payout"))
        try:
            price = float(execution.get("price"))
            shares = float(execution.get("shares"))
        except (TypeError, ValueError):
            unchecked_sells += 1
            continue
        if payout is None:
            unchecked_sells += 1
            continue
        rows.append(
            {
                "token_id": execution.get("token_id"),
                "question": row.get("question"),
                "city": row.get("city"),
                "target_date": row.get("target_date"),
                "side": row.get("side"),
                "shares": round(shares, 6),
                "sell_price": round(price, 4),
                "sell_fair_value": _optional_float(row.get("fair_value")),
                "sell_edge": _optional_float(row.get("edge")),
                "final_payout": payout,
                "notional_usd": round(float(execution.get("notional_usd") or 0.0), 4),
                "realized_pnl_usd": round(float(execution.get("realized_pnl_usd") or 0.0), 4),
                "decision_value_vs_settlement_usd": round((price - payout) * shares, 4),
                "bucket_shape": _bucket_shape(row.get("bucket_lower_f"), row.get("bucket_upper_f")),
            }
        )
    decision_value = sum(float(row["decision_value_vs_settlement_usd"]) for row in rows)
    event_winner_drag = sum(
        float(row["decision_value_vs_settlement_usd"])
        for row in rows
        if row.get("final_payout") == 1
    )
    event_loser_value = sum(
        float(row["decision_value_vs_settlement_usd"])
        for row in rows
        if row.get("final_payout") == 0
    )
    return {
        "method": (
            "For each replay sell, decision value is shares * (sell_price - final_payout). "
            "Positive means selling beat holding to settlement; negative means the sell reduced final PnL."
        ),
        "sell_count": len(sells),
        "checked_sell_count": len(rows),
        "unchecked_sell_count": unchecked_sells,
        "sell_notional_usd": round(sum(float(execution.get("notional_usd") or 0.0) for execution in sells), 4),
        "sell_realized_pnl_usd": round(sum(float(execution.get("realized_pnl_usd") or 0.0) for execution in sells), 4),
        "sell_decision_value_vs_settlement_usd": round(decision_value, 4),
        "event_winner_sell_drag_usd": round(event_winner_drag, 4),
        "event_loser_sell_value_usd": round(event_loser_value, 4),
        "by_final_payout": _aggregate_replay_sell_rows(rows, "final_payout"),
        "by_side": _aggregate_replay_sell_rows(rows, "side"),
        "by_bucket_shape": _aggregate_replay_sell_rows(rows, "bucket_shape"),
        "by_city": _aggregate_replay_sell_rows(rows, "city"),
        "by_sell_price_bucket": _aggregate_replay_sell_rows(rows, "sell_price", width=0.10),
        "by_sell_fair_value_bucket": _aggregate_replay_sell_rows(rows, "sell_fair_value", width=0.10),
        "worst_sells_vs_settlement": sorted(rows, key=lambda item: float(item["decision_value_vs_settlement_usd"]))[:10],
        "best_sells_vs_settlement": sorted(rows, key=lambda item: float(item["decision_value_vs_settlement_usd"]), reverse=True)[:10],
    }


def _aggregate_replay_sell_rows(rows: list[dict[str, Any]], key: str, *, width: Optional[float] = None) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        group = row.get(key)
        if width is not None:
            group = _bucket_float(group, width)
        grouped.setdefault(group, []).append(row)
    aggregates = []
    for group, items in grouped.items():
        decision_value = sum(float(item.get("decision_value_vs_settlement_usd") or 0.0) for item in items)
        sell_notional = sum(float(item.get("notional_usd") or 0.0) for item in items)
        realized_pnl = sum(float(item.get("realized_pnl_usd") or 0.0) for item in items)
        prices = [float(item["sell_price"]) for item in items if item.get("sell_price") is not None]
        fair_values = [float(item["sell_fair_value"]) for item in items if item.get("sell_fair_value") is not None]
        aggregates.append(
            {
                "group": group,
                "sell_count": len(items),
                "sell_notional_usd": round(sell_notional, 4),
                "sell_realized_pnl_usd": round(realized_pnl, 4),
                "decision_value_vs_settlement_usd": round(decision_value, 4),
                "avg_sell_price": round(sum(prices) / len(prices), 4) if prices else None,
                "avg_sell_fair_value": round(sum(fair_values) / len(fair_values), 4) if fair_values else None,
            }
        )
    return sorted(aggregates, key=lambda item: (-abs(float(item["decision_value_vs_settlement_usd"])), str(item["group"])))


def _json_required_buffered_edge(market_price: float, settings: SignalSettings) -> float:
    min_kelly_edge = settings.min_edge
    if market_price >= settings.high_confidence_price_threshold:
        min_kelly_edge = min(min_kelly_edge, settings.high_confidence_min_kelly_edge)
    return min_kelly_edge * max(0.0001, 1.0 - market_price)


def _optional_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _calibration_metrics(rows: list[dict[str, Any]], *, probability_key: str = "fair_value") -> dict[str, Any]:
    if not rows:
        return {"n": 0, "avg_market_price": None, "avg_fair_value": None, "actual_rate": None, "brier_fair_value": None, "brier_market": None}
    actual = [int(row["polymarket_payout"]) for row in rows]
    fair_values = [float(row[probability_key]) for row in rows]
    prices = [float(row["market_price"]) for row in rows]
    return {
        "n": len(rows),
        "avg_market_price": round(sum(prices) / len(prices), 4),
        "avg_fair_value": round(sum(fair_values) / len(fair_values), 4),
        "actual_rate": round(sum(actual) / len(actual), 4),
        "brier_fair_value": round(sum((p - y) ** 2 for p, y in zip(fair_values, actual)) / len(rows), 6),
        "brier_market": round(sum((p - y) ** 2 for p, y in zip(prices, actual)) / len(rows), 6),
    }


def _aggregate_score_groups(rows: list[dict[str, Any]], key: str, width: Optional[float]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = row.get(key)
        if width is not None:
            group_key = _bucket_float(group_key, width)
        grouped.setdefault(group_key, []).append(row)
    return [
        {"group": group, **_calibration_metrics(items)}
        for group, items in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _model_probability_accuracy(
    rows: list[dict[str, Any]],
    *,
    group_by: str = "model",
    probability_key: str = "model_probabilities",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[float, int]]] = {}
    for row in rows:
        outcome = int(row["polymarket_payout"])
        probabilities = row.get(probability_key) or {}
        if not isinstance(probabilities, dict):
            continue
        for name, probability in probabilities.items():
            try:
                grouped.setdefault(_model_accuracy_group_key(str(name), group_by), []).append((float(probability), outcome))
            except (TypeError, ValueError):
                continue
    diagnostics = []
    for name, values in grouped.items():
        if not values:
            continue
        avg_probability = sum(prob for prob, _ in values) / len(values)
        actual_rate = sum(outcome for _, outcome in values) / len(values)
        diagnostics.append(
            {
                group_by: name,
                "n": len(values),
                "avg_probability": round(avg_probability, 4),
                "actual_rate": round(actual_rate, 4),
                "bias": round(avg_probability - actual_rate, 4),
                "brier": round(sum((prob - outcome) ** 2 for prob, outcome in values) / len(values), 6),
            }
        )
    return sorted(diagnostics, key=lambda item: (item["brier"], -item["n"]))[:30]


def _model_accuracy_group_key(model_key: str, group_by: str) -> str:
    if group_by == "model":
        return model_key
    source, separator, model_family = model_key.rpartition(".")
    if group_by == "source":
        return source if separator else "unknown"
    if group_by == "model_family":
        return model_family if separator else model_key
    raise ValueError(f"Unknown model accuracy group: {group_by}")


def _exit_management_diagnostics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    sold_trades = [trade for trade in trades if int(trade.get("sell_count") or 0) > 0]
    sell_decision_value = sum(float(trade.get("sell_decision_value_vs_settlement_usd") or 0.0) for trade in sold_trades)
    event_winner_sell_drag = sum(
        float(trade.get("sell_decision_value_vs_settlement_usd") or 0.0)
        for trade in sold_trades
        if trade.get("event_outcome") == "event_win"
    )
    event_loser_sell_value = sum(
        float(trade.get("sell_decision_value_vs_settlement_usd") or 0.0)
        for trade in sold_trades
        if trade.get("event_outcome") == "event_loss"
    )
    return {
        "method": (
            "For each sell, decision value is shares * (sell_price - final_payout). "
            "Positive means selling beat holding to settlement; negative means the sell reduced final PnL."
        ),
        "trades_with_sells": len(sold_trades),
        "sell_count": sum(int(trade.get("sell_count") or 0) for trade in sold_trades),
        "sell_notional_usd": round(sum(float(trade.get("sell_notional_usd") or 0.0) for trade in sold_trades), 4),
        "sell_realized_pnl_usd": round(sum(float(trade.get("sell_realized_pnl_usd") or 0.0) for trade in sold_trades), 4),
        "settlement_realized_pnl_usd": round(sum(float(trade.get("settlement_realized_pnl_usd") or 0.0) for trade in trades), 4),
        "sell_decision_value_vs_settlement_usd": round(sell_decision_value, 4),
        "event_winner_sell_drag_usd": round(event_winner_sell_drag, 4),
        "event_loser_sell_value_usd": round(event_loser_sell_value, 4),
        "worst_sells_vs_settlement": sorted(
            sold_trades,
            key=lambda item: float(item.get("sell_decision_value_vs_settlement_usd") or 0.0),
        )[:10],
        "best_sells_vs_settlement": sorted(
            sold_trades,
            key=lambda item: float(item.get("sell_decision_value_vs_settlement_usd") or 0.0),
            reverse=True,
        )[:10],
    }


def _aggregate_trade_groups(trades: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for trade in trades:
        grouped.setdefault(trade.get(key), []).append(trade)
    rows = []
    for group, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        buy_notional = sum(float(item.get("buy_notional_usd") or 0.0) for item in items)
        pnl = sum(float(item.get("realized_pnl_usd") or 0.0) for item in items)
        rows.append(
            {
                "group": group,
                "n": len(items),
                "buy_notional_usd": round(buy_notional, 4),
                "realized_pnl_usd": round(pnl, 4),
                "return_on_buy_notional": round(pnl / buy_notional, 4) if buy_notional else None,
            }
        )
    return rows


def _event_outcome_label(value: Any) -> str:
    outcome = _coerce_binary_outcome(value)
    if outcome == 1:
        return "event_win"
    if outcome == 0:
        return "event_loss"
    return "event_unknown"


def _event_outcome_trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [trade for trade in trades if trade.get("event_outcome") in {"event_win", "event_loss"}]
    event_winners = [trade for trade in resolved if trade.get("event_outcome") == "event_win"]
    event_losers = [trade for trade in resolved if trade.get("event_outcome") == "event_loss"]
    event_loss_pnl = sum(float(trade.get("realized_pnl_usd") or 0.0) for trade in event_losers)
    event_win_pnl = sum(float(trade.get("realized_pnl_usd") or 0.0) for trade in event_winners)
    return {
        "event_winning_trades": len(event_winners),
        "event_losing_trades": len(event_losers),
        "event_unknown_trades": len(trades) - len(resolved),
        "event_hit_rate": round(len(event_winners) / len(resolved), 4) if resolved else None,
        "event_win_pnl_usd": round(event_win_pnl, 4),
        "event_loss_pnl_usd": round(event_loss_pnl, 4),
        "profitable_event_loser_trades": sum(1 for trade in event_losers if float(trade.get("realized_pnl_usd") or 0.0) > 0),
    }


def _pnl_concentration(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return _pnl_concentration_from_values([float(item.get("realized_pnl_usd") or 0.0) for item in trades])


def _pnl_concentration_from_values(values: list[float]) -> dict[str, Any]:
    total_pnl = sum(values)
    if not values:
        return {
            "total_pnl_usd": 0.0,
            "loss_trade_count": 0,
            "loss_pnl_usd": 0.0,
            "top_1_pnl_share": None,
            "top_3_pnl_share": None,
            "top_5_pnl_share": None,
            "top_10_pnl_share": None,
        }
    winners = sorted(values, reverse=True)
    losses = [value for value in values if value < 0]

    def share(count: int) -> Optional[float]:
        if total_pnl <= 1e-9:
            return None
        pnl = sum(winners[:count])
        return round(pnl / total_pnl, 4)

    return {
        "total_pnl_usd": round(total_pnl, 4),
        "loss_trade_count": len(losses),
        "loss_pnl_usd": round(sum(losses), 4),
        "top_1_pnl_share": share(1),
        "top_3_pnl_share": share(3),
        "top_5_pnl_share": share(5),
        "top_10_pnl_share": share(10),
    }


def _json_equity_snapshot(session_key: str, cash: float, positions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    open_value = 0.0
    for position in positions.values():
        open_value += float(position.get("shares") or 0.0) * float(position.get("last_price") or 0.0)
    return {
        "session": session_key,
        "cash": cash,
        "equity": cash + open_value,
        "open_positions": len(positions),
    }


def _historical_equity_snapshot(
    session_time: datetime,
    cash: float,
    positions: Mapping[str, HistoricalPosition],
) -> dict[str, Any]:
    open_value = sum(position.shares * position.last_price for position in positions.values())
    return {
        "session": session_time.isoformat(),
        "cash": round(cash, 4),
        "open_value": round(open_value, 4),
        "equity": round(cash + open_value, 4),
        "open_positions": len(positions),
    }


def _historical_final_equity_snapshot(
    cash: float,
    open_value: float,
    positions: Mapping[str, HistoricalPosition],
) -> dict[str, Any]:
    return {
        "session": "final",
        "cash": round(cash, 4),
        "open_value": round(open_value, 4),
        "equity": round(cash + open_value, 4),
        "open_positions": len(positions),
    }


def _performance_diagnostics_from_equity_curve(
    equity_curve: list[dict[str, Any]],
    starting_equity: float,
) -> dict[str, Any]:
    parseable_points: list[tuple[datetime, float]] = []
    final_equity: Optional[float] = None
    for point in equity_curve:
        equity = _optional_float(point.get("equity"))
        if equity is None:
            continue
        if point.get("session") == "final":
            final_equity = equity
            continue
        session_time = _parse_dt(point.get("session"))
        if session_time is not None:
            parseable_points.append((session_time, equity))
    parseable_points.sort(key=lambda item: item[0])
    if not parseable_points or starting_equity <= 0:
        return {
            "method": "Calendar-day equity diagnostics from configured replay decision sessions; Sharpe is annualized with 365 trading days.",
            "period_start": None,
            "period_end": None,
            "period_days": None,
            "total_return_pct": None,
            "annualized_return_pct": None,
            "average_monthly_return_pct": None,
            "annualized_from_average_monthly_pct": None,
            "calendar_daily_sharpe_365": None,
            "daily_return_count": 0,
            "max_drawdown_usd": _equity_curve_drawdown(equity_curve)[0],
            "max_drawdown_pct": _equity_curve_drawdown(equity_curve)[1],
        }

    ending_equity = final_equity if final_equity is not None else parseable_points[-1][1]
    period_start = parseable_points[0][0]
    period_end = parseable_points[-1][0]
    period_days = max((period_end - period_start).total_seconds() / 86400.0, 1e-9)
    total_return = ending_equity / starting_equity - 1.0
    annualized_return = _annualized_return(ending_equity, starting_equity, period_days)

    daily_equity: dict[date, float] = {}
    for session_time, equity in parseable_points:
        daily_equity[session_time.date()] = equity
    if final_equity is not None:
        daily_equity[period_end.date()] = final_equity
    daily_returns = _period_returns_from_end_values(daily_equity, starting_equity)

    monthly_equity: dict[str, float] = {}
    for session_time, equity in parseable_points:
        monthly_equity[session_time.strftime("%Y-%m")] = equity
    if final_equity is not None:
        monthly_equity[period_end.strftime("%Y-%m")] = final_equity
    monthly_returns = _period_returns_from_end_values(monthly_equity, starting_equity)
    average_monthly_return = sum(monthly_returns) / len(monthly_returns) if monthly_returns else None
    annualized_from_average_monthly = (
        (1.0 + average_monthly_return) ** 12 - 1.0
        if average_monthly_return is not None and average_monthly_return > -1.0
        else None
    )
    average_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else None
    daily_volatility = _sample_standard_deviation(daily_returns)
    sharpe = (
        average_daily_return / daily_volatility * math.sqrt(365.0)
        if average_daily_return is not None and daily_volatility and daily_volatility > 0
        else None
    )
    max_drawdown_usd, max_drawdown_pct = _equity_curve_drawdown(equity_curve)
    return {
        "method": "Calendar-day equity diagnostics from configured replay decision sessions; Sharpe is annualized with 365 trading days.",
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "period_days": round(period_days, 2),
        "total_return_pct": round(total_return, 4),
        "annualized_return_pct": round(annualized_return, 4) if annualized_return is not None else None,
        "average_monthly_return_pct": round(average_monthly_return, 4) if average_monthly_return is not None else None,
        "annualized_from_average_monthly_pct": round(annualized_from_average_monthly, 4) if annualized_from_average_monthly is not None else None,
        "average_daily_return_pct": round(average_daily_return, 5) if average_daily_return is not None else None,
        "daily_volatility_pct": round(daily_volatility, 5) if daily_volatility is not None else None,
        "calendar_daily_sharpe_365": round(sharpe, 4) if sharpe is not None else None,
        "daily_return_count": len(daily_returns),
        "monthly_return_count": len(monthly_returns),
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
    }


def _period_returns_from_end_values(period_end_values: Mapping[Any, float], starting_equity: float) -> list[float]:
    returns: list[float] = []
    previous = starting_equity
    for key in sorted(period_end_values):
        current = period_end_values[key]
        if previous > 0:
            returns.append(current / previous - 1.0)
        previous = current
    return returns


def _sample_standard_deviation(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _annualized_return(ending_equity: float, starting_equity: float, period_days: float) -> Optional[float]:
    if starting_equity <= 0 or ending_equity <= 0 or period_days < 7:
        return None
    try:
        return (ending_equity / starting_equity) ** (365.0 / period_days) - 1.0
    except OverflowError:
        return None


def _equity_curve_min(equity_curve: list[dict[str, Any]]) -> Optional[float]:
    if not equity_curve:
        return None
    return round(min(float(point["equity"]) for point in equity_curve), 2)


def _equity_curve_drawdown(equity_curve: list[dict[str, Any]]) -> tuple[Optional[float], Optional[float]]:
    peak: Optional[float] = None
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        peak = equity if peak is None else max(peak, equity)
        if peak <= 0:
            continue
        drawdown = peak - equity
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max(max_drawdown_pct, drawdown / peak)
    return round(max_drawdown, 2), round(max_drawdown_pct, 4)


def _month_key(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value)
    return text[:7] if len(text) >= 7 else None


def _crosscheck_label(row: Mapping[str, Any]) -> str:
    if row.get("weather_outcome") is None or row.get("polymarket_payout") is None:
        return "unchecked"
    return "mismatch" if row.get("weather_outcome") != row.get("polymarket_payout") else "matched"


def _trade_lead_days(trade: Mapping[str, Any]) -> Optional[int]:
    first_buy_at = trade.get("first_buy_at")
    target_date = trade.get("target_date")
    city_name = str(trade.get("city") or "")
    if not first_buy_at or not target_date:
        return None
    city = _city_from_display_name(city_name)
    try:
        executed = datetime.fromisoformat(str(first_buy_at).replace("Z", "+00:00"))
        target = date.fromisoformat(str(target_date))
    except ValueError:
        return None
    if city is None:
        return None
    return (target - executed.astimezone(_zoneinfo_or_utc(city)).date()).days


def _score_lead_days(row: Mapping[str, Any]) -> Optional[int]:
    generated_at = _parse_dt(row.get("generated_at"))
    target_date = _parse_json_date(row.get("target_date"))
    city = _city_from_display_name(str(row.get("city") or ""))
    if generated_at is None or target_date is None or city is None:
        return None
    return (target_date - generated_at.astimezone(_zoneinfo_or_utc(city)).date()).days


def _trade_entry_hour_utc(trade: Mapping[str, Any]) -> Optional[int]:
    first_buy_at = trade.get("first_buy_at")
    if not first_buy_at:
        return None
    try:
        return datetime.fromisoformat(str(first_buy_at).replace("Z", "+00:00")).astimezone(timezone.utc).hour
    except ValueError:
        return None


def _score_entry_hour_utc(row: Mapping[str, Any]) -> Optional[int]:
    generated_at = _parse_dt(row.get("generated_at"))
    return generated_at.astimezone(timezone.utc).hour if generated_at is not None else None


def _bucket_float(value: Any, width: float) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(math.floor(numeric / width) * width, 4)


def _top_weather_crosscheck_mismatch_trades(executions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches = [execution for execution in executions if _is_weather_crosscheck_mismatch(execution)]
    return sorted(mismatches, key=lambda item: abs(float(item.get("realized_pnl_usd") or 0.0)), reverse=True)[:20]


def _has_weather_crosscheck(execution: Mapping[str, Any]) -> bool:
    return execution.get("weather_outcome") is not None and execution.get("polymarket_payout") is not None


def _is_weather_crosscheck_mismatch(execution: Mapping[str, Any]) -> bool:
    return _has_weather_crosscheck(execution) and execution.get("weather_outcome") != execution.get("polymarket_payout")


def _signal_settings_to_json(settings: SignalSettings) -> dict[str, Any]:
    return {
        "min_edge": settings.min_edge,
        "uncertainty_buffer": settings.uncertainty_buffer,
        "max_spread": settings.max_spread,
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
        "same_day_earliest_entry_hour_local": settings.same_day_earliest_entry_hour_local,
        "same_day_latest_entry_hour_local": settings.same_day_latest_entry_hour_local,
        "enforce_entry_timing_filter": settings.enforce_entry_timing_filter,
    }


def _reason_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        reason_key = reason.split(":", 1)[0]
        counts[reason_key] = counts.get(reason_key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _zoneinfo_or_utc(city: CityConfig) -> ZoneInfo | timezone:
    try:
        return ZoneInfo(city.timezone)
    except ZoneInfoNotFoundError:
        return timezone.utc


def _city_from_display_name(display_name: str) -> Optional[CityConfig]:
    from weather_strategy.cities import DEFAULT_CITIES

    for city in DEFAULT_CITIES:
        if city.display_name == display_name:
            return city
    return None


def _deadline_exceeded(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _effective_max_position_usd(
    bankroll_usd: float,
    max_position_usd: float,
    max_position_fraction: Optional[float],
    *,
    edge: Optional[float] = None,
    edge_position_full_cap_edge: float = 0.0,
    edge_position_min_multiplier: float = 0.35,
) -> float:
    caps = [max(0.0, max_position_usd)]
    if max_position_fraction is not None and max_position_fraction > 0:
        caps.append(max(0.0, bankroll_usd * max_position_fraction))
    base_cap = min(caps)
    return _edge_scaled_position_cap(base_cap, edge, edge_position_full_cap_edge, edge_position_min_multiplier)


def _edge_scaled_position_cap(
    max_position_usd: float,
    edge: Optional[float],
    edge_position_full_cap_edge: float,
    edge_position_min_multiplier: float,
) -> float:
    cap = max(0.0, max_position_usd)
    if edge is None or edge_position_full_cap_edge <= 0:
        return cap
    edge_ratio = max(0.0, float(edge)) / max(0.0001, edge_position_full_cap_edge)
    floor = max(0.0, min(1.0, edge_position_min_multiplier))
    multiplier = max(floor, min(1.0, edge_ratio))
    return cap * multiplier


def _blend_probability_with_market(fair_value: float, market_price: float, market_blend: float) -> float:
    blend = max(0.0, min(1.0, market_blend))
    probability = fair_value * (1.0 - blend) + market_price * blend
    return max(0.0, min(1.0, probability))


def _make_run_log_path(log_dir: str | Path, prefix: str) -> Path:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{timestamp}-{time.time_ns()}-{prefix}.json"


def _write_json_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _progress(progress_every: int, message: str) -> None:
    if progress_every <= 0:
        return
    print(f"[weather-long-backtest] {message}", file=sys.stderr, flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)
