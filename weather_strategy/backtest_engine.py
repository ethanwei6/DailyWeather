from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


CURRENT_LIVE_STRATEGY_PROFILE = "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.25-strict-no-tail-0.14-bprice-0.70"

DEFAULT_STRATEGY_COMPARISON_PROFILES = (
    CURRENT_LIVE_STRATEGY_PROFILE,
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.50-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-windowbank-0.25-kelly-0.50-poscap-0.20-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-paced-0.25-slots-4-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-paced-0.25-slots-2-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
    "live-forward-50-highwin-strict-no-tail-0.14-no-side-max-0.94-bounded-confirmed-cap-0.35-bprice-0.70",
    "live-forward-50-reserve-0.25-kelly-0.50-cap-0.20-strict-no-tail-0.14-bprice-0.70",
)


_REGION_ALIASES = {
    "atlanta": "North America",
    "austin": "North America",
    "boston": "North America",
    "chicago": "North America",
    "dallas": "North America",
    "denver": "North America",
    "houston": "North America",
    "las vegas": "North America",
    "los angeles": "North America",
    "miami": "North America",
    "new york": "North America",
    "philadelphia": "North America",
    "phoenix": "North America",
    "san francisco": "North America",
    "seattle": "North America",
    "toronto": "North America",
    "washington": "North America",
    "berlin": "Europe",
    "london": "Europe",
    "madrid": "Europe",
    "moscow": "Europe",
    "paris": "Europe",
    "rome": "Europe",
    "amsterdam": "Europe",
    "beijing": "Asia-Pacific",
    "guangzhou": "Asia-Pacific",
    "hong kong": "Asia-Pacific",
    "seoul": "Asia-Pacific",
    "shanghai": "Asia-Pacific",
    "shenzhen": "Asia-Pacific",
    "singapore": "Asia-Pacific",
    "sydney": "Asia-Pacific",
    "taipei": "Asia-Pacific",
    "tokyo": "Asia-Pacific",
    "dubai": "Middle East",
}


def live_like_backtest_dates(
    *,
    lookback_days: int,
    min_end_date: Optional[date],
    max_end_date: Optional[date],
    today: Optional[date] = None,
) -> tuple[date, date]:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    effective_today = today or date.today()
    end = max_end_date or (effective_today - timedelta(days=1))
    start = min_end_date or (end - timedelta(days=lookback_days - 1))
    if start > end:
        raise ValueError(f"min_end_date {start.isoformat()} is after max_end_date {end.isoformat()}")
    return start, end


def compare_strategy_replays(
    replay_results: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    current_live_strategy_profile: str = CURRENT_LIVE_STRATEGY_PROFILE,
    source_run_log: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = [dict(result) for result in replay_results]
    summary_rows = [strategy_summary_row(result, current_live_strategy_profile) for result in results]
    summary_rows.sort(
        key=lambda row: (
            bool(row.get("current_live_strategy")),
            float(row.get("pnl_usd") or 0.0),
            float(row.get("sharpe_365") or -999.0),
        ),
        reverse=True,
    )
    breakdowns = {str(result.get("strategy_profile") or result.get("variant")): strategy_breakdowns(result) for result in results}
    trades = {str(result.get("strategy_profile") or result.get("variant")): trade_rows(result) for result in results}

    payload = {
        "source_run_log": str(source_run_log) if source_run_log is not None else None,
        "current_live_strategy_profile": current_live_strategy_profile,
        "strategy_count": len(results),
        "summary_rows": summary_rows,
        "breakdowns": breakdowns,
        "trade_rows": trades,
    }
    summary_json = output / "summary.json"
    summary_md = output / "summary.md"
    equity_svg = output / "equity_curves.svg"
    trades_json = output / "trades.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    trades_json.write_text(json.dumps(trades, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(_summary_markdown(payload), encoding="utf-8")
    equity_svg.write_text(_equity_svg(results), encoding="utf-8")
    return {
        "summary": {
            "source_run_log": payload["source_run_log"],
            "current_live_strategy_profile": current_live_strategy_profile,
            "strategy_count": len(results),
            "summary_rows": summary_rows,
            "artifact_paths": {
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "equity_curves_svg": str(equity_svg),
                "trades_json": str(trades_json),
            },
        },
        "artifact_paths": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "equity_curves_svg": str(equity_svg),
            "trades_json": str(trades_json),
        },
        "breakdowns": breakdowns,
        "trade_rows": trades,
    }


def strategy_summary_row(result: Mapping[str, Any], current_live_strategy_profile: str) -> dict[str, Any]:
    profile = str(result.get("strategy_profile") or result.get("variant") or "unknown")
    diagnostics = result.get("performance_diagnostics") if isinstance(result.get("performance_diagnostics"), Mapping) else {}
    trade_diagnostics = result.get("trade_diagnostics") if isinstance(result.get("trade_diagnostics"), Mapping) else {}
    return {
        "strategy_profile": profile,
        "current_live_strategy": profile == current_live_strategy_profile,
        "bankroll_usd": _round(result.get("bankroll_usd")),
        "ending_equity_usd": _round(result.get("ending_equity_usd")),
        "pnl_usd": _round(result.get("pnl_usd")),
        "return_pct": _round(result.get("return_pct"), 4),
        "average_monthly_return_pct": _round(diagnostics.get("average_monthly_return_pct"), 4),
        "annualized_return_pct": _round(diagnostics.get("annualized_return_pct"), 4),
        "annualized_from_average_monthly_pct": _round(diagnostics.get("annualized_from_average_monthly_pct"), 4),
        "sharpe_365": _round(diagnostics.get("calendar_daily_sharpe_365"), 4),
        "max_drawdown_usd": _round(result.get("max_drawdown_usd")),
        "max_drawdown_pct": _round(result.get("max_drawdown_pct"), 4),
        "min_cash_pct": _round(result.get("min_cash_pct"), 4),
        "signals": _int(result.get("signals")),
        "trade_count": _int(result.get("trade_count")),
        "executions": _int(result.get("executions")),
        "buys": _int(result.get("buys")),
        "sells": _int(result.get("sells")),
        "settlements": _int(result.get("settlements")),
        "buy_notional_usd": _round(result.get("buy_notional_usd")),
        "return_on_buy_notional": _round(result.get("return_on_buy_notional"), 4),
        "hit_rate": _round(result.get("hit_rate"), 4),
        "event_hit_rate": _round(result.get("event_hit_rate"), 4),
        "event_winning_trades": _int(result.get("event_winning_trades")),
        "event_losing_trades": _int(result.get("event_losing_trades")),
        "top_1_pnl_share": _round(result.get("top_1_pnl_share"), 4),
        "period_start": diagnostics.get("period_start"),
        "period_end": diagnostics.get("period_end"),
        "period_days": _round(diagnostics.get("period_days"), 2),
        "trade_diagnostics_hit_rate": _round(trade_diagnostics.get("hit_rate"), 4),
        "run_log_path": result.get("run_log_path"),
    }


def strategy_breakdowns(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = trade_rows(result)
    return {
        "by_region": _group_trade_rows(rows, "region"),
        "by_side": _group_trade_rows(rows, "side"),
        "by_city": _group_trade_rows(rows, "city", limit=30),
        "by_entry_hour_utc": _group_trade_rows(rows, "entry_hour_utc"),
        "by_target_month": _group_trade_rows(rows, "target_month"),
        "by_action": _group_trade_rows(rows, "action"),
    }


def trade_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for execution in result.get("executions_detail") or []:
        if not isinstance(execution, Mapping):
            continue
        row = execution.get("row") if isinstance(execution.get("row"), Mapping) else execution
        city = str(row.get("city") or execution.get("city") or "Unknown")
        generated_at = str(row.get("generated_at") or execution.get("executed_at") or "")
        target_date = str(row.get("target_date") or execution.get("target_date") or "")
        rows.append(
            {
                "action": execution.get("action"),
                "token_id": execution.get("token_id") or row.get("token_id"),
                "question": row.get("question") or execution.get("question"),
                "city": city,
                "region": world_region(city),
                "target_date": target_date,
                "target_month": target_date[:7] if len(target_date) >= 7 else None,
                "entry_time_utc": generated_at or None,
                "entry_hour_utc": _hour_from_iso(generated_at),
                "bucket": row.get("bucket") or row.get("bucket_label") or execution.get("bucket"),
                "side": row.get("side") or execution.get("side"),
                "shares": _round(execution.get("shares"), 6),
                "price": _round(execution.get("price"), 4),
                "notional_usd": _round(execution.get("notional_usd")),
                "realized_pnl_usd": _round(execution.get("realized_pnl_usd")),
                "fair_value": _round(row.get("fair_value"), 4),
                "edge": _round(row.get("edge"), 4),
                "model_agreement": _round(row.get("model_agreement"), 4),
                "polymarket_payout": row.get("polymarket_payout") if row.get("polymarket_payout") in (0, 1) else execution.get("polymarket_payout"),
                "signal_filter_reason": row.get("signal_filter_reason"),
            }
        )
    return rows


def world_region(city: str) -> str:
    city_key = city.lower().replace(",", " ")
    for alias, region in _REGION_ALIASES.items():
        if alias in city_key:
            return region
    return "Other"


def _group_trade_rows(rows: list[dict[str, Any]], key: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"executions": 0, "buys": 0, "sells": 0, "settlements": 0, "buy_notional_usd": 0.0, "realized_pnl_usd": 0.0})
    for row in rows:
        value = str(row.get(key) if row.get(key) is not None else "Unknown")
        bucket = grouped[value]
        action = str(row.get("action") or "")
        bucket["executions"] += 1
        if action == "BUY":
            bucket["buys"] += 1
            bucket["buy_notional_usd"] += float(row.get("notional_usd") or 0.0)
        elif action == "SELL":
            bucket["sells"] += 1
        elif action == "SETTLE":
            bucket["settlements"] += 1
        bucket["realized_pnl_usd"] += float(row.get("realized_pnl_usd") or 0.0)
    items = [
        {
            key: value,
            **{
                name: (_round(amount) if isinstance(amount, float) else amount)
                for name, amount in metrics.items()
            },
        }
        for value, metrics in grouped.items()
    ]
    items.sort(key=lambda item: (float(item.get("buy_notional_usd") or 0.0), abs(float(item.get("realized_pnl_usd") or 0.0))), reverse=True)
    return items[:limit] if limit is not None else items


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    rows = payload.get("summary_rows") if isinstance(payload.get("summary_rows"), list) else []
    lines = [
        "# DailyWeather Live-Like Strategy Comparison",
        "",
        f"Source run log: `{payload.get('source_run_log')}`",
        f"Current live profile: `{payload.get('current_live_strategy_profile')}`",
        "",
        "| Strategy | Live | PnL | Return | Max DD | Sharpe | Trades | Event hit | Min cash |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {live} | {pnl} | {ret} | {dd} | {sharpe} | {trades} | {hit} | {cash} |".format(
                profile=row.get("strategy_profile"),
                live="yes" if row.get("current_live_strategy") else "",
                pnl=_money(row.get("pnl_usd")),
                ret=_pct(row.get("return_pct")),
                dd=_pct(row.get("max_drawdown_pct")),
                sharpe=_display(row.get("sharpe_365")),
                trades=_display(row.get("trade_count")),
                hit=_pct(row.get("event_hit_rate")),
                cash=_pct(row.get("min_cash_pct")),
            )
        )
    lines.extend(
        [
            "",
            "The comparison uses cached scored outcomes from the source artifact. Strategy replays do not refetch forecasts, prices, or observations.",
            "",
        ]
    )
    return "\n".join(lines)


def _equity_svg(results: list[Mapping[str, Any]]) -> str:
    series = []
    for result in results:
        points = _equity_points(result.get("equity_curve") if isinstance(result.get("equity_curve"), list) else [])
        if points:
            series.append((str(result.get("strategy_profile") or result.get("variant") or "strategy"), points))
    width, height = 1200, 640
    margin = 70
    if not series:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="40" y="80">No equity curve data</text></svg>\n'
    min_time = min(point[0] for _, points in series for point in points)
    max_time = max(point[0] for _, points in series for point in points)
    min_equity = min(point[1] for _, points in series for point in points)
    max_equity = max(point[1] for _, points in series for point in points)
    if max_time <= min_time:
        max_time = min_time + 1.0
    if math.isclose(max_equity, min_equity):
        max_equity += 1.0
        min_equity -= 1.0
    palette = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c")
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="70" y="38" font-family="Arial" font-size="24" font-weight="700">DailyWeather Live-Like Backtest Equity Curves</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#94a3b8" stroke-width="1"/>',
    ]
    for index, (name, points) in enumerate(series):
        color = palette[index % len(palette)]
        coords = []
        for timestamp, equity in points:
            x = margin + (timestamp - min_time) / (max_time - min_time) * (width - 2 * margin)
            y = height - margin - (equity - min_equity) / (max_equity - min_equity) * (height - 2 * margin)
            coords.append(f"{x:.2f},{y:.2f}")
        pieces.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}"/>')
        legend_y = 70 + index * 22
        pieces.append(f'<rect x="{width - 430}" y="{legend_y - 12}" width="12" height="12" fill="{color}"/>')
        pieces.append(f'<text x="{width - 410}" y="{legend_y}" font-family="Arial" font-size="12">{_escape_xml(name[:58])}</text>')
    pieces.append(f'<text x="{margin}" y="{height - 24}" font-family="Arial" font-size="12" fill="#475569">Equity range: {_money(min_equity)} to {_money(max_equity)}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def _equity_points(equity_curve: list[Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    synthetic_index = 0
    last_timestamp: Optional[float] = None
    for point in equity_curve:
        if not isinstance(point, Mapping):
            continue
        equity = _to_float(point.get("equity"))
        if equity is None:
            continue
        session = point.get("session")
        parsed = _parse_timestamp(session)
        if parsed is None:
            synthetic_index += 1
            timestamp = (last_timestamp + 3600.0) if last_timestamp is not None else float(synthetic_index)
        else:
            timestamp = parsed
        last_timestamp = timestamp
        points.append((timestamp, equity))
    return points


def _parse_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or value == "final":
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _hour_from_iso(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).hour
    except ValueError:
        return None


def _round(value: Any, digits: int = 2) -> Optional[float]:
    number = _to_float(value)
    if number is None:
        return None
    return round(number, digits)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _money(value: Any) -> str:
    number = _to_float(value)
    return "n/a" if number is None else f"${number:,.2f}"


def _pct(value: Any) -> str:
    number = _to_float(value)
    return "n/a" if number is None else f"{number * 100:.2f}%"


def _display(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
