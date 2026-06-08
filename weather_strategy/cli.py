from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weather_strategy.forecast import ConsensusForecastEngine
from weather_strategy.models import ForecastDistribution, TradeSignal, WeatherMarket
from weather_strategy.observations import ObservedHigh, ObservedHighClient
from weather_strategy.paper import PaperLedger
from weather_strategy.parser import parse_weather_market
from weather_strategy.polymarket import PolymarketClobClient, PolymarketGammaClient
from weather_strategy.signals import SignalSettings, market_entry_timing, score_outcomes, signals_from_scored_outcomes
from weather_strategy.weather import OpenMeteoClient


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Weather Polymarket strategy tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper = subparsers.add_parser("paper-run", help="Generate and record paper-trade signals")
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
    paper.add_argument("--max-position-usd", type=float, default=50.0)
    paper.add_argument("--min-trade-usd", type=float, default=1.0)
    paper.add_argument("--min-model-count", type=int, default=3)
    paper.add_argument("--min-model-agreement", type=float, default=0.65)
    paper.add_argument("--same-day-entry-start-hour", type=int, default=11)
    paper.add_argument("--same-day-entry-cutoff-hour", type=int, default=17)
    paper.add_argument("--allow-late-same-day", action="store_true")
    paper.add_argument("--disable-observations", action="store_true")
    paper.add_argument("--min-lead-days", type=int, default=0)
    paper.add_argument("--max-lead-days", type=int, default=2)

    live = subparsers.add_parser("scan-live", help="Discover live active weather markets")
    live.add_argument("--limit", type=int, default=50)
    live.add_argument("--pages", type=int, default=1)
    live.add_argument("--json-out", help="Optional output path for parsed live markets")

    debug = subparsers.add_parser("debug-search", help="Print raw Gamma public-search event and market titles")
    debug.add_argument("--query", default="temperature")
    debug.add_argument("--limit", type=int, default=10)
    debug.add_argument("--page", type=int, default=1)

    report = subparsers.add_parser("report", help="Summarize paper-trading equity, PnL, and open positions")
    report.add_argument("--ledger", default="work/data/paper_trades.sqlite")
    report.add_argument("--bankroll-usd", type=float, default=1000.0)

    calibration = subparsers.add_parser("calibration", help="Summarize recorded forecast calibration data")
    calibration.add_argument("--ledger", default="work/data/paper_trades.sqlite")

    args = parser.parse_args(argv)
    if args.command == "paper-run":
        return run_paper(args)
    if args.command == "scan-live":
        return run_scan_live(args)
    if args.command == "debug-search":
        return run_debug_search(args)
    if args.command == "report":
        return run_report(args)
    if args.command == "calibration":
        return run_calibration(args)
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


def run_paper(args: argparse.Namespace) -> int:
    started = time.monotonic()
    deadline = None if args.max_runtime_seconds <= 0 else started + args.max_runtime_seconds
    fixture_mode = bool(args.fixture)
    ledger = PaperLedger(args.ledger)
    forecast_engine = ConsensusForecastEngine()
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
        all_scored.extend(scored)
        processed_markets += 1
        if args.progress_every > 0 and processed_markets % args.progress_every == 0:
            _progress(f"markets_scored={processed_markets} outcomes_scored={len(all_scored)} elapsed_seconds={time.monotonic() - started:.1f}")
    all_signals: list[TradeSignal] = signals_from_scored_outcomes(all_scored, settings=signal_settings)

    score_rows = ledger.record_forecast_scores(all_scored)
    executions = ledger.rebalance_kelly(
        all_scored,
        bankroll_usd=args.bankroll_usd,
        kelly_fraction=args.kelly_fraction,
        max_position_usd=args.max_position_usd,
        min_trade_usd=args.min_trade_usd,
        min_edge=args.min_edge,
        min_model_count=args.min_model_count,
        min_model_agreement=args.min_model_agreement,
        min_price=signal_settings.min_price,
        max_price=signal_settings.max_price,
    )
    rows = ledger.record_signals(
        all_signals,
        metadata={
            "fixture_mode": fixture_mode,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "market_count": len(markets),
            "processed_market_count": processed_markets,
            "skipped_market_count": len(skipped_markets),
            "model_consensus": True,
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
        },
    )
    print(
        json.dumps(
            {
                "fixture_mode": fixture_mode,
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
                "equity_usd": round(equity, 2),
                "pnl_usd": round(equity - args.bankroll_usd, 2),
                "open_positions": len(ledger.positions()),
                "ledger": args.ledger,
                "top_signals": [_signal_to_json(signal) for signal in all_signals[:10]],
                "top_scored_outcomes": [_scored_to_json(outcome) for outcome in sorted(all_scored, key=lambda item: item.edge, reverse=True)[:10]],
                "skipped_market_examples": skipped_markets[:10],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _deadline_exceeded(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _progress(message: str) -> None:
    print(f"[weather-paper-run] {message}", file=sys.stderr, flush=True)


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
            yield raw, {}
            continue
        market = parse_weather_market(raw)
        if market is not None:
            yield market, raw


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


def _scored_to_json(outcome) -> dict[str, Any]:
    return {
        "question": outcome.question,
        "bucket": outcome.bucket_label,
        "fair_value_probability": outcome.fair_value,
        "market_probability": outcome.market_price,
        "edge_after_buffer": outcome.edge,
        "model_agreement": outcome.model_agreement,
        "model_count": outcome.model_count,
        "probability_stdev": outcome.probability_stdev,
        "entry_eligible": outcome.entry_eligible,
        "entry_filter_reason": outcome.entry_filter_reason,
        "observed_high_f": outcome.observed_high_f,
        "observation_source": outcome.observation_source,
        "observation_final": outcome.observation_final,
        "observation_adjusted": outcome.observation_adjusted,
        "observed_outcome": outcome.observed_outcome,
        "model_probabilities": {
            key: round(value, 4)
            for key, value in sorted(outcome.model_probabilities.items())
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
