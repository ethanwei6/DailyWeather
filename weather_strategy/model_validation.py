from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class WeatherModelRow:
    generated_at: str
    actual: float
    market_price: float
    fair_value: float
    model_probabilities: dict[str, float]
    is_signal: bool


def run_weather_model_validation(
    *,
    source_run_log: str | Path,
    train_fraction: float = 0.70,
    probability_key: str = "auto",
    run_log_dir: str | Path = "work/logs/model_validation",
    max_coordinate_epochs: int = 4,
) -> dict[str, Any]:
    rows = _load_rows(source_run_log, probability_key=probability_key)
    rows.sort(key=lambda row: row.generated_at)
    if len(rows) < 50:
        raise ValueError("Need at least 50 resolved rows for model validation")
    split = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
    train_rows = rows[:split]
    test_rows = rows[split:]

    source_names, family_names = _source_and_family_names(rows)
    source_weights, family_weights = _fit_coordinate_weights(
        train_rows,
        source_names=source_names,
        family_names=family_names,
        max_epochs=max_coordinate_epochs,
    )
    tail_calibration = _fit_tail_calibration(
        train_rows,
        lambda row: _weighted_ensemble_probability(row, source_weights, family_weights),
    )

    predictors: dict[str, Callable[[WeatherModelRow], float]] = {
        "recorded_fair_value": lambda row: row.fair_value,
        "market_price": lambda row: row.market_price,
        "weighted_weather_ensemble": lambda row: _weighted_ensemble_probability(row, source_weights, family_weights),
        "weighted_tail_calibrated_weather_ensemble": lambda row: _apply_tail_calibration(
            _weighted_ensemble_probability(row, source_weights, family_weights),
            tail_calibration,
        ),
    }
    slices = {
        "train_all": train_rows,
        "test_all": test_rows,
        "test_signal": [row for row in test_rows if row.is_signal],
        "test_high_recorded_fv": [row for row in test_rows if row.fair_value >= 0.90],
    }
    metrics = {
        predictor_name: {
            slice_name: _metrics(slice_rows, predictor)
            for slice_name, slice_rows in slices.items()
        }
        for predictor_name, predictor in predictors.items()
    }
    selected_name = min(
        ("recorded_fair_value", "weighted_weather_ensemble", "weighted_tail_calibrated_weather_ensemble"),
        key=lambda name: metrics[name]["test_all"]["brier"] if metrics[name]["test_all"]["n"] else float("inf"),
    )
    best_weather_by_slice = _best_weather_by_slice(metrics)
    market_test_brier = metrics["market_price"]["test_all"]["brier"]
    selected_test_brier = metrics[selected_name]["test_all"]["brier"]
    signal_market_brier = metrics["market_price"]["test_signal"]["brier"]
    signal_selected_brier = metrics[selected_name]["test_signal"]["brier"]
    high_market_brier = metrics["market_price"]["test_high_recorded_fv"]["brier"]
    high_selected_brier = metrics[selected_name]["test_high_recorded_fv"]["brier"]

    output = {
        "source_run_log": str(source_run_log),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probability_key": probability_key,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "range": {
            "first_generated_at": rows[0].generated_at,
            "last_generated_at": rows[-1].generated_at,
        },
        "sources": source_names,
        "model_families": family_names,
        "fitted_weather_only_weights": {
            "source_weights": source_weights,
            "model_family_weights": family_weights,
            "tail_calibration": tail_calibration,
        },
        "metrics": metrics,
        "selected_weather_only_candidate": selected_name,
        "best_weather_by_slice": best_weather_by_slice,
        "beats_market_brier": {
            "test_all": bool(selected_test_brier is not None and market_test_brier is not None and selected_test_brier < market_test_brier),
            "test_signal": bool(signal_selected_brier is not None and signal_market_brier is not None and signal_selected_brier < signal_market_brier),
            "test_high_recorded_fv": bool(high_selected_brier is not None and high_market_brier is not None and high_selected_brier < high_market_brier),
        },
        "interpretation": _interpretation(metrics, selected_name),
    }
    log_path = _write_run_log(output, run_log_dir)
    output["run_log_path"] = str(log_path)
    log_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _load_rows(source_run_log: str | Path, *, probability_key: str) -> list[WeatherModelRow]:
    payload = json.loads(Path(source_run_log).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Model-validation source run log must be a JSON object")
    scored = payload.get("scored_outcomes_detail") or []
    rows: list[WeatherModelRow] = []
    for item in scored:
        if not isinstance(item, Mapping):
            continue
        actual = item.get("polymarket_payout")
        market_price = item.get("market_price")
        fair_value = item.get("fair_value")
        generated_at = item.get("generated_at")
        if actual is None or market_price is None or fair_value is None or generated_at is None:
            continue
        probabilities = _probabilities_for_item(item, probability_key)
        if not probabilities:
            continue
        rows.append(
            WeatherModelRow(
                generated_at=str(generated_at),
                actual=1.0 if float(actual) >= 0.5 else 0.0,
                market_price=_clip_probability(float(market_price)),
                fair_value=_clip_probability(float(fair_value)),
                model_probabilities=probabilities,
                is_signal=bool(item.get("passes_signal_filter") or item.get("signal_eligible")),
            )
        )
    return rows


def _probabilities_for_item(item: Mapping[str, Any], probability_key: str) -> dict[str, float]:
    if probability_key == "raw_model_probabilities":
        candidates = [item.get("raw_model_probabilities")]
    elif probability_key == "model_probabilities":
        candidates = [item.get("model_probabilities")]
    elif probability_key == "auto":
        candidates = [item.get("raw_model_probabilities"), item.get("model_probabilities")]
    else:
        raise ValueError("probability_key must be auto, raw_model_probabilities, or model_probabilities")
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate:
            probabilities = {
                str(key): _clip_probability(float(value))
                for key, value in candidate.items()
                if value is not None
            }
            if probabilities:
                return probabilities
    return {}


def _source_and_family_names(rows: list[WeatherModelRow]) -> tuple[list[str], list[str]]:
    sources: set[str] = set()
    families: set[str] = set()
    for row in rows:
        for key in row.model_probabilities:
            source, family = _split_model_key(key)
            sources.add(source)
            families.add(family)
    return sorted(sources), sorted(families)


def _fit_coordinate_weights(
    rows: list[WeatherModelRow],
    *,
    source_names: list[str],
    family_names: list[str],
    max_epochs: int,
) -> tuple[dict[str, float], dict[str, float]]:
    source_weights = {source: 1.0 for source in source_names}
    family_weights = {family: 1.0 for family in family_names}
    choices = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
    for _ in range(max(1, max_epochs)):
        for family in family_names:
            original = family_weights[family]
            best_weight = original
            best_brier = _brier(rows, lambda row: _weighted_ensemble_probability(row, source_weights, family_weights))
            for candidate in choices:
                family_weights[family] = candidate
                candidate_brier = _brier(rows, lambda row: _weighted_ensemble_probability(row, source_weights, family_weights))
                if candidate_brier < best_brier:
                    best_brier = candidate_brier
                    best_weight = candidate
            family_weights[family] = best_weight
        for source in source_names:
            original = source_weights[source]
            best_weight = original
            best_brier = _brier(rows, lambda row: _weighted_ensemble_probability(row, source_weights, family_weights))
            for candidate in choices:
                source_weights[source] = candidate
                candidate_brier = _brier(rows, lambda row: _weighted_ensemble_probability(row, source_weights, family_weights))
                if candidate_brier < best_brier:
                    best_brier = candidate_brier
                    best_weight = candidate
            source_weights[source] = best_weight
    return source_weights, family_weights


def _weighted_ensemble_probability(
    row: WeatherModelRow,
    source_weights: Mapping[str, float],
    family_weights: Mapping[str, float],
) -> float:
    source_totals: dict[str, float] = {}
    source_weight_sums: dict[str, float] = {}
    for key, probability in row.model_probabilities.items():
        source, family = _split_model_key(key)
        weight = max(0.0, float(family_weights.get(family, 1.0)))
        source_totals[source] = source_totals.get(source, 0.0) + probability * weight
        source_weight_sums[source] = source_weight_sums.get(source, 0.0) + weight
    total = 0.0
    weight_sum = 0.0
    for source, source_total in source_totals.items():
        family_weight_sum = source_weight_sums.get(source, 0.0)
        if family_weight_sum <= 0:
            continue
        source_probability = source_total / family_weight_sum
        source_weight = max(0.0, float(source_weights.get(source, 1.0)))
        total += source_probability * source_weight
        weight_sum += source_weight
    if weight_sum <= 0:
        return row.fair_value
    return _clip_probability(total / weight_sum)


def _fit_tail_calibration(
    rows: list[WeatherModelRow],
    predictor: Callable[[WeatherModelRow], float],
) -> dict[str, float]:
    best: Optional[tuple[float, float, float, float]] = None
    for center_alpha in (0.60, 0.70, 0.80, 0.90, 1.00, 1.10):
        for tail_threshold in (0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00):
            for tail_alpha in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
                calibration = {
                    "center_alpha": center_alpha,
                    "tail_threshold": tail_threshold,
                    "tail_alpha": tail_alpha,
                }
                brier = _brier(rows, lambda row: _apply_tail_calibration(predictor(row), calibration))
                if best is None or brier < best[0]:
                    best = (brier, center_alpha, tail_threshold, tail_alpha)
    assert best is not None
    return {
        "center_alpha": best[1],
        "tail_threshold": best[2],
        "tail_alpha": best[3],
        "train_brier": round(best[0], 6),
    }


def _apply_tail_calibration(probability: float, calibration: Mapping[str, float]) -> float:
    center_alpha = float(calibration.get("center_alpha", 1.0))
    tail_threshold = float(calibration.get("tail_threshold", 1.0))
    tail_alpha = float(calibration.get("tail_alpha", 1.0))
    probability = 0.5 + center_alpha * (_clip_probability(probability) - 0.5)
    low_threshold = 1.0 - tail_threshold
    if probability >= tail_threshold:
        probability = tail_threshold + tail_alpha * (probability - tail_threshold)
    elif probability <= low_threshold:
        probability = low_threshold + tail_alpha * (probability - low_threshold)
    return _clip_probability(probability)


def _metrics(rows: list[WeatherModelRow], predictor: Callable[[WeatherModelRow], float]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "avg_prediction": None, "actual_rate": None, "brier": None, "log_loss": None}
    predictions = [_clip_probability(predictor(row)) for row in rows]
    actual = [row.actual for row in rows]
    return {
        "n": len(rows),
        "avg_prediction": round(sum(predictions) / len(predictions), 6),
        "actual_rate": round(sum(actual) / len(actual), 6),
        "brier": round(sum((prediction - outcome) ** 2 for prediction, outcome in zip(predictions, actual)) / len(rows), 6),
        "log_loss": round(
            -sum(
                outcome * math.log(prediction) + (1.0 - outcome) * math.log(1.0 - prediction)
                for prediction, outcome in zip(predictions, actual)
            )
            / len(rows),
            6,
        ),
    }


def _brier(rows: list[WeatherModelRow], predictor: Callable[[WeatherModelRow], float]) -> float:
    if not rows:
        return float("inf")
    return sum((_clip_probability(predictor(row)) - row.actual) ** 2 for row in rows) / len(rows)


def _interpretation(metrics: Mapping[str, Mapping[str, Mapping[str, Any]]], selected_name: str) -> str:
    selected_test = metrics[selected_name]["test_all"]["brier"]
    market_test = metrics["market_price"]["test_all"]["brier"]
    selected_signal = metrics[selected_name]["test_signal"]["brier"]
    market_signal = metrics["market_price"]["test_signal"]["brier"]
    if selected_test is not None and market_test is not None and selected_test < market_test:
        return "Weather-only candidate beats market Brier on the full held-out slice."
    if selected_signal is not None and market_signal is not None and selected_signal < market_signal:
        return "Weather-only candidate does not beat market globally, but beats market Brier on the held-out signal slice."
    return "Weather-only candidate does not beat market Brier on held-out slices; use this as a calibration diagnostic, not a promotion signal."


def _best_weather_by_slice(metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    weather_names = ("recorded_fair_value", "weighted_weather_ensemble", "weighted_tail_calibrated_weather_ensemble")
    market_metrics = metrics.get("market_price") or {}
    output: dict[str, dict[str, Any]] = {}
    for slice_name in ("train_all", "test_all", "test_signal", "test_high_recorded_fv"):
        candidates = []
        for weather_name in weather_names:
            slice_metrics = (metrics.get(weather_name) or {}).get(slice_name) or {}
            brier = slice_metrics.get("brier")
            if brier is not None:
                candidates.append((float(brier), weather_name, slice_metrics))
        market_brier = (market_metrics.get(slice_name) or {}).get("brier")
        if not candidates:
            output[slice_name] = {"candidate": None, "brier": None, "market_brier": market_brier, "beats_market": False}
            continue
        best_brier, best_name, best_metrics = min(candidates, key=lambda item: item[0])
        output[slice_name] = {
            "candidate": best_name,
            "n": best_metrics.get("n"),
            "brier": round(best_brier, 6),
            "market_brier": market_brier,
            "beats_market": bool(market_brier is not None and best_brier < float(market_brier)),
        }
    return output


def _split_model_key(key: str) -> tuple[str, str]:
    if "." not in key:
        return key, "unknown"
    source, family = key.rsplit(".", 1)
    return source, family


def _clip_probability(value: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _write_run_log(output: Mapping[str, Any], run_log_dir: str | Path) -> Path:
    directory = Path(run_log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{timestamp}-weather-model-validation.json"
