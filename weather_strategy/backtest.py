from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from weather_strategy.cities import find_city
from weather_strategy.http import HttpClient
from weather_strategy.models import CityConfig
from weather_strategy.observations import ObservedHighClient, observed_outcome_for_bucket
from weather_strategy.parser import parse_temperature_bucket
from weather_strategy.signals import SignalSettings, _passes_edge_gate, _price_adjusted_uncertainty_buffer


@dataclass(frozen=True)
class ResolvedForecastRow:
    id: int
    generated_at: datetime
    city: str
    target_date: Optional[date]
    bucket_label: str
    token_id: Optional[str]
    fair_value: float
    market_price: float
    model_count: int
    model_agreement: float
    probability_stdev: float
    entry_eligible: bool
    observed_outcome: int
    model_probabilities: dict[str, float]


@dataclass
class AccuracyMetric:
    n: int = 0
    brier_sum: float = 0.0
    log_loss_sum: float = 0.0

    def add(self, probability: float, outcome: int) -> None:
        probability = max(1e-6, min(1 - 1e-6, probability))
        self.n += 1
        self.brier_sum += (probability - outcome) ** 2
        self.log_loss_sum += -math.log(probability if outcome else 1 - probability)

    @property
    def brier(self) -> Optional[float]:
        return self.brier_sum / self.n if self.n else None

    @property
    def log_loss(self) -> Optional[float]:
        return self.log_loss_sum / self.n if self.n else None


def load_calibration_weights(path: str | Path | None) -> tuple[dict[str, float], dict[str, float]]:
    if not path:
        return {}, {}
    weights_path = Path(path)
    if not weights_path.exists():
        return {}, {}
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    source_weights = {str(key): float(value) for key, value in (payload.get("source_weights") or {}).items()}
    model_weights = {str(key): float(value) for key, value in (payload.get("model_weights") or {}).items()}
    _apply_weight_alias(source_weights, "open_meteo_gfs_hrrr", "open_meteo_gfs_best_match")
    _apply_model_weight_alias(model_weights, "open_meteo_gfs_hrrr", "open_meteo_gfs_best_match")
    return source_weights, model_weights


def _apply_weight_alias(weights: dict[str, float], old_key: str, new_key: str) -> None:
    if old_key in weights and new_key not in weights:
        weights[new_key] = weights[old_key]


def _apply_model_weight_alias(weights: dict[str, float], old_source: str, new_source: str) -> None:
    prefix = f"{old_source}."
    for key, value in list(weights.items()):
        if not key.startswith(prefix):
            continue
        alias = f"{new_source}.{key[len(prefix):]}"
        if alias not in weights:
            weights[alias] = value


def run_backtest(
    ledger_path: str | Path,
    *,
    bankroll_usd: float,
    kelly_fraction: float,
    max_position_usd: float,
    min_trade_usd: float,
    settings: SignalSettings,
    train_fraction: float,
    output_weights_path: Optional[str | Path] = None,
    fetch_observations: bool = True,
    max_observation_lookups: int = 200,
    min_weight_samples: int = 20,
    weight_prior_samples: int = 50,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    rows = load_resolved_forecast_rows(
        ledger_path,
        fetch_observations=fetch_observations,
        max_observation_lookups=max_observation_lookups,
        now=now,
    )
    if not rows:
        return {"resolved_rows": 0, "error": "No resolved forecast rows available for backtest"}

    split_index = max(1, min(len(rows) - 1, int(len(rows) * train_fraction))) if len(rows) >= 2 else len(rows)
    train_rows = rows[:split_index]
    test_rows = rows[split_index:] if split_index < len(rows) else rows

    source_weights, model_weights, weight_diagnostics = fit_accuracy_weights(
        train_rows,
        min_samples=min_weight_samples,
        prior_samples=weight_prior_samples,
    )
    if output_weights_path:
        _write_weights(output_weights_path, source_weights, model_weights, weight_diagnostics, train_rows)

    report = {
        "resolved_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "date_range": {
            "first_generated_at": rows[0].generated_at.isoformat(),
            "last_generated_at": rows[-1].generated_at.isoformat(),
        },
        "prediction_accuracy": {
            "train": _prediction_report(train_rows, source_weights, model_weights),
            "test": _prediction_report(test_rows, source_weights, model_weights),
        },
        "kelly_replay": {
            "default_weights": _single_entry_kelly_replay(test_rows, {}, {}, bankroll_usd, kelly_fraction, max_position_usd, min_trade_usd, settings),
            "calibrated_weights": _single_entry_kelly_replay(test_rows, source_weights, model_weights, bankroll_usd, kelly_fraction, max_position_usd, min_trade_usd, settings),
        },
        "learned_weights": {
            "source_weights": _top_weights(source_weights),
            "model_weights": _top_weights(model_weights),
            "diagnostics": weight_diagnostics,
        },
        "weights_output": str(output_weights_path) if output_weights_path else None,
        "notes": [
            "Backtest uses recorded forecast snapshots and recorded market prices from the ledger.",
            "Kelly replay is a simple first-qualifying-entry simulation on the held-out rows, not a full historical order-book replay.",
            "Weights are shrunk toward 1.0 to reduce overfitting on small samples.",
        ],
    }
    return report


def load_resolved_forecast_rows(
    ledger_path: str | Path,
    *,
    fetch_observations: bool,
    max_observation_lookups: int,
    now: Optional[datetime] = None,
) -> list[ResolvedForecastRow]:
    current = now or datetime.now(timezone.utc)
    observation_client = ObservedHighClient(HttpClient(timeout_seconds=8))
    observation_cache: dict[tuple[str, str], Optional[float]] = {}
    observation_lookups = 0
    resolved = []
    with sqlite3.connect(str(ledger_path)) as conn:
        conn.row_factory = sqlite3.Row
        db_rows = conn.execute("SELECT * FROM forecast_scores ORDER BY generated_at, id").fetchall()

    if fetch_observations and max_observation_lookups > 0:
        observation_lookups = _prefetch_archive_final_highs(
            db_rows,
            observation_client,
            observation_cache,
            current,
            max_observation_lookups,
        )

    for row in db_rows:
        observed_outcome = row["observed_outcome"]
        if observed_outcome is None:
            if not fetch_observations or observation_lookups >= max_observation_lookups:
                continue
            observed_outcome = _resolve_row_outcome(row, observation_client, observation_cache, current)
            if observed_outcome is None:
                if _needs_observation_lookup(row, current):
                    observation_lookups += 1
                continue
        model_probabilities = _parse_model_probabilities(row["model_probabilities_json"])
        if not model_probabilities:
            continue
        resolved.append(
            ResolvedForecastRow(
                id=int(row["id"]),
                generated_at=_parse_datetime(row["generated_at"]),
                city=str(row["city"]),
                target_date=_parse_date(row["target_date"]),
                bucket_label=str(row["bucket_label"]),
                token_id=row["token_id"],
                fair_value=float(row["fair_value"]),
                market_price=float(row["market_price"]),
                model_count=int(row["model_count"]),
                model_agreement=float(row["model_agreement"]),
                probability_stdev=float(row["probability_stdev"]),
                entry_eligible=bool(row["entry_eligible"]),
                observed_outcome=int(observed_outcome),
                model_probabilities=model_probabilities,
            )
        )
    return resolved


def _prefetch_archive_final_highs(
    rows: list[sqlite3.Row],
    observation_client: ObservedHighClient,
    observation_cache: dict[tuple[str, str], Optional[float]],
    current: datetime,
    max_city_day_lookups: int,
) -> int:
    city_dates: dict[str, set[date]] = {}
    cities: dict[str, CityConfig] = {}
    for row in rows:
        if row["observed_outcome"] is not None:
            continue
        city = find_city(str(row["city"]))
        target = _parse_date(row["target_date"])
        if city is None or target is None or not _is_final_target(city, target, current):
            continue
        key = (city.display_name, target.isoformat())
        if key in observation_cache:
            continue
        city_dates.setdefault(city.display_name, set()).add(target)
        cities[city.display_name] = city

    selected = 0
    for city_name in sorted(city_dates):
        if selected >= max_city_day_lookups:
            break
        city = cities[city_name]
        dates = sorted(city_dates[city_name])
        remaining = max_city_day_lookups - selected
        dates = dates[:remaining]
        if not dates:
            continue
        selected += len(dates)
        for target in dates:
            observation_cache[(city.display_name, target.isoformat())] = None
        highs = _fetch_archive_high_range(observation_client, city, dates[0], dates[-1])
        for target, high in highs.items():
            if target in dates:
                observation_cache[(city.display_name, target.isoformat())] = high
    return selected


def _fetch_archive_high_range(
    observation_client: ObservedHighClient,
    city: CityConfig,
    start_date: date,
    end_date: date,
) -> dict[date, float]:
    try:
        payload = observation_client.http.get_json(
            observation_client.OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": city.timezone,
            },
        )
    except RuntimeError:
        return {}
    daily = payload.get("daily") if isinstance(payload, dict) else {}
    times = (daily or {}).get("time") or []
    values = (daily or {}).get("temperature_2m_max") or []
    highs: dict[date, float] = {}
    for index, value in enumerate(values):
        if index >= len(times) or value is None:
            continue
        try:
            target = date.fromisoformat(str(times[index]))
            highs[target] = float(value)
        except (TypeError, ValueError):
            continue
    return highs


def fit_accuracy_weights(
    rows: list[ResolvedForecastRow],
    *,
    min_samples: int,
    prior_samples: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    source_metrics: dict[str, AccuracyMetric] = {}
    model_metrics: dict[str, AccuracyMetric] = {}
    for row in rows:
        for source, probability in _source_probability_views(row.model_probabilities).items():
            source_metrics.setdefault(source, AccuracyMetric()).add(probability, row.observed_outcome)
        for model_key, probability in row.model_probabilities.items():
            model_metrics.setdefault(model_key, AccuracyMetric()).add(probability, row.observed_outcome)

    source_baseline = _pooled_brier(source_metrics)
    model_baseline = _pooled_brier(model_metrics)
    source_weights = {
        key: _metric_to_weight(metric, source_baseline, min_samples=min_samples, prior_samples=prior_samples)
        for key, metric in source_metrics.items()
    }
    model_weights = {
        key: _metric_to_weight(metric, model_baseline, min_samples=min_samples, prior_samples=prior_samples)
        for key, metric in model_metrics.items()
    }
    diagnostics = {
        "source_baseline_brier": round(source_baseline, 6) if source_baseline is not None else None,
        "model_baseline_brier": round(model_baseline, 6) if model_baseline is not None else None,
        "min_weight_samples": min_samples,
        "weight_prior_samples": prior_samples,
        "source_metrics": _metric_summary(source_metrics),
        "model_metrics": _metric_summary(model_metrics),
    }
    return source_weights, model_weights, diagnostics


def _prediction_report(rows: list[ResolvedForecastRow], source_weights: Mapping[str, float], model_weights: Mapping[str, float]) -> dict[str, Any]:
    market = AccuracyMetric()
    recorded = AccuracyMetric()
    default_recomputed = AccuracyMetric()
    calibrated = AccuracyMetric()
    for row in rows:
        market.add(row.market_price, row.observed_outcome)
        recorded.add(row.fair_value, row.observed_outcome)
        default_recomputed.add(_weighted_consensus_probability(row.model_probabilities, {}, {}), row.observed_outcome)
        calibrated.add(_weighted_consensus_probability(row.model_probabilities, source_weights, model_weights), row.observed_outcome)
    return {
        "rows": len(rows),
        "market": _metric_to_json(market),
        "recorded_fair_value": _metric_to_json(recorded),
        "default_recomputed_fair_value": _metric_to_json(default_recomputed),
        "calibrated_fair_value": _metric_to_json(calibrated),
    }


def _single_entry_kelly_replay(
    rows: list[ResolvedForecastRow],
    source_weights: Mapping[str, float],
    model_weights: Mapping[str, float],
    bankroll_usd: float,
    kelly_fraction: float,
    max_position_usd: float,
    min_trade_usd: float,
    settings: SignalSettings,
) -> dict[str, Any]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fair_value = _weighted_consensus_probability(row.model_probabilities, source_weights, model_weights)
        buffer = _price_adjusted_uncertainty_buffer(row.market_price, settings)
        edge = fair_value - row.market_price - buffer
        source_views = _source_probability_views(row.model_probabilities, model_weights)
        model_agreement = _agreement_above(source_views, row.market_price, buffer)
        probability_stdev = statistics.pstdev(source_views.values()) if len(source_views) >= 2 else 0.0
        if not row.entry_eligible:
            continue
        if row.market_price < settings.min_price or row.market_price > settings.max_price:
            continue
        if len(source_views) < settings.min_model_count:
            continue
        if model_agreement < settings.min_model_agreement:
            continue
        if not _passes_edge_gate(edge, row.market_price, settings):
            continue
        sessions.setdefault(_session_key(row.generated_at), []).append(
            {
                "row": row,
                "fair_value": fair_value,
                "edge": edge,
                "agreement": model_agreement,
                "probability_stdev": probability_stdev,
            }
        )

    positions: dict[tuple[str, Optional[date]], dict[str, Any]] = {}
    trades = []
    for session_key in sorted(sessions):
        by_city_day: dict[tuple[str, Optional[date]], dict[str, Any]] = {}
        for candidate in sorted(sessions[session_key], key=lambda item: item["edge"], reverse=True):
            row = candidate["row"]
            group_key = (row.city, row.target_date)
            if group_key not in by_city_day:
                by_city_day[group_key] = candidate
        for group_key, candidate in by_city_day.items():
            if group_key in positions:
                continue
            row = candidate["row"]
            raw_fraction = max(0.0, (candidate["fair_value"] - row.market_price) / max(0.0001, 1.0 - row.market_price))
            target_notional = min(max_position_usd, bankroll_usd * kelly_fraction * raw_fraction * candidate["agreement"])
            if target_notional < min_trade_usd:
                continue
            shares = target_notional / row.market_price
            pnl = shares * (row.observed_outcome - row.market_price)
            trade = {
                "generated_at": row.generated_at.isoformat(),
                "city": row.city,
                "target_date": row.target_date.isoformat() if row.target_date else None,
                "bucket": row.bucket_label,
                "market_price": round(row.market_price, 4),
                "fair_value": round(candidate["fair_value"], 4),
                "edge": round(candidate["edge"], 4),
                "agreement": round(candidate["agreement"], 4),
                "notional_usd": round(target_notional, 2),
                "observed_outcome": row.observed_outcome,
                "pnl_usd": pnl,
            }
            positions[group_key] = trade
            trades.append(trade)

    total_pnl = sum(float(trade["pnl_usd"]) for trade in trades)
    total_notional = sum(float(trade["notional_usd"]) for trade in trades)
    wins = sum(1 for trade in trades if float(trade["pnl_usd"]) > 0)
    return {
        "trades": len(trades),
        "wins": wins,
        "hit_rate": round(wins / len(trades), 4) if trades else None,
        "total_notional_usd": round(total_notional, 2),
        "pnl_usd": round(total_pnl, 2),
        "roi_on_staked": round(total_pnl / total_notional, 4) if total_notional else None,
        "ending_equity_usd": round(bankroll_usd + total_pnl, 2),
        "top_trades": [
            {**trade, "pnl_usd": round(float(trade["pnl_usd"]), 2)}
            for trade in sorted(trades, key=lambda item: abs(float(item["pnl_usd"])), reverse=True)[:10]
        ],
    }


def _resolve_row_outcome(
    row: sqlite3.Row,
    observation_client: ObservedHighClient,
    observation_cache: dict[tuple[str, str], Optional[float]],
    current: datetime,
) -> Optional[int]:
    city = find_city(str(row["city"]))
    target = _parse_date(row["target_date"])
    bucket = parse_temperature_bucket(str(row["bucket_label"]))
    if city is None or target is None or bucket is None:
        return None
    cache_key = (city.display_name, target.isoformat())
    if cache_key not in observation_cache:
        if not _is_final_target(city, target, current):
            observation_cache[cache_key] = None
        else:
            try:
                observed = observation_client.fetch_observed_high(city, target, now=current)
            except (RuntimeError, ValueError):
                observed = None
            observation_cache[cache_key] = observed.max_temperature_f if observed is not None and observed.is_final else None
    observed_high = observation_cache[cache_key]
    if observed_high is None:
        return None
    return observed_outcome_for_bucket(bucket, observed_high, True)


def _needs_observation_lookup(row: sqlite3.Row, current: datetime) -> bool:
    city = find_city(str(row["city"]))
    target = _parse_date(row["target_date"])
    return city is not None and target is not None and _is_final_target(city, target, current)


def _is_final_target(city: CityConfig, target: date, current: datetime) -> bool:
    try:
        from zoneinfo import ZoneInfo

        local_date = current.astimezone(ZoneInfo(city.timezone)).date()
    except Exception:
        local_date = current.date()
    return target < local_date


def _weighted_consensus_probability(
    model_probabilities: Mapping[str, float],
    source_weights: Mapping[str, float],
    model_weights: Mapping[str, float],
) -> float:
    source_views = _source_probability_views(model_probabilities, model_weights)
    if not source_views:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    for source, probability in source_views.items():
        weight = max(0.0, float(source_weights.get(source, 1.0)))
        weighted_sum += probability * weight
        weight_sum += weight
    return weighted_sum / weight_sum if weight_sum > 0 else sum(source_views.values()) / len(source_views)


def _source_probability_views(model_probabilities: Mapping[str, float], model_weights: Optional[Mapping[str, float]] = None) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    grouped_weights: dict[str, list[float]] = {}
    for key, probability in model_probabilities.items():
        source = key.rsplit(".", 1)[0] if "." in key else key
        grouped.setdefault(source, []).append(float(probability))
        grouped_weights.setdefault(source, []).append(float((model_weights or {}).get(key, 1.0)))
    return {
        source: _weighted_mean(values, grouped_weights.get(source) or [])
        for source, values in grouped.items()
        if values
    }


def _agreement_above(source_views: Mapping[str, float], market_price: float, buffer: float) -> float:
    if not source_views:
        return 0.0
    agreeing = sum(1 for probability in source_views.values() if probability > market_price + buffer)
    return agreeing / len(source_views)


def _metric_to_weight(
    metric: AccuracyMetric,
    baseline_brier: Optional[float],
    *,
    min_samples: int,
    prior_samples: int,
    min_weight: float = 0.25,
    max_weight: float = 3.0,
) -> float:
    if metric.n < min_samples or baseline_brier is None or not metric.brier:
        return 1.0
    skill = baseline_brier / max(1e-6, metric.brier)
    shrink = metric.n / (metric.n + prior_samples)
    return round(max(min_weight, min(max_weight, 1.0 + shrink * (skill - 1.0))), 4)


def _pooled_brier(metrics: Mapping[str, AccuracyMetric]) -> Optional[float]:
    n = sum(metric.n for metric in metrics.values())
    if n <= 0:
        return None
    return sum(metric.brier_sum for metric in metrics.values()) / n


def _metric_summary(metrics: Mapping[str, AccuracyMetric]) -> dict[str, dict[str, Any]]:
    return {
        key: _metric_to_json(metric)
        for key, metric in sorted(metrics.items())
    }


def _metric_to_json(metric: AccuracyMetric) -> dict[str, Any]:
    return {
        "n": metric.n,
        "brier": round(metric.brier, 6) if metric.brier is not None else None,
        "log_loss": round(metric.log_loss, 6) if metric.log_loss is not None else None,
    }


def _top_weights(weights: Mapping[str, float], limit: int = 12) -> dict[str, float]:
    ranked = sorted(weights.items(), key=lambda item: abs(item[1] - 1.0), reverse=True)
    return {key: value for key, value in ranked[:limit]}


def _write_weights(
    output_path: str | Path,
    source_weights: Mapping[str, float],
    model_weights: Mapping[str, float],
    diagnostics: Mapping[str, Any],
    train_rows: list[ResolvedForecastRow],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train_rows),
        "train_range": {
            "first_generated_at": train_rows[0].generated_at.isoformat() if train_rows else None,
            "last_generated_at": train_rows[-1].generated_at.isoformat() if train_rows else None,
        },
        "source_weights": dict(sorted(source_weights.items())),
        "model_weights": dict(sorted(model_weights.items())),
        "diagnostics": diagnostics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_model_probabilities(value: str) -> dict[str, float]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    probabilities = {}
    for key, probability in parsed.items():
        try:
            probabilities[str(key)] = max(0.0, min(1.0, float(probability)))
        except (TypeError, ValueError):
            continue
    return probabilities


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _parse_date(value: object) -> Optional[date]:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _session_key(value: datetime, minutes: int = 30) -> str:
    minute = (value.minute // minutes) * minutes
    return value.replace(minute=minute, second=0, microsecond=0).isoformat()


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    if not weights or len(weights) != len(values):
        return sum(values) / len(values)
    weight_sum = sum(max(0.0, weight) for weight in weights)
    if weight_sum <= 0:
        return sum(values) / len(values)
    return sum(value * max(0.0, weight) for value, weight in zip(values, weights)) / weight_sum
