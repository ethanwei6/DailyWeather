from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from datetime import date, datetime, timezone
from typing import Iterable, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weather_strategy.cities import find_city
from weather_strategy.models import ScoredOutcome, TradeSignal
from weather_strategy.observations import ObservedHighClient
from weather_strategy.parser import parse_temperature_bucket


SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    question TEXT NOT NULL,
    city TEXT NOT NULL,
    target_date TEXT,
    bucket_label TEXT NOT NULL,
    token_id TEXT,
    side TEXT NOT NULL,
    fair_value REAL NOT NULL,
    market_price REAL NOT NULL,
    edge REAL NOT NULL,
    size_usd REAL NOT NULL,
    reason TEXT NOT NULL,
    rule_excerpt TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_market ON paper_trades(market_id, bucket_label);
CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);

CREATE TABLE IF NOT EXISTS paper_positions (
    token_id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    question TEXT NOT NULL,
    bucket_label TEXT NOT NULL,
    city TEXT NOT NULL,
    target_date TEXT,
    shares REAL NOT NULL,
    cost_basis REAL NOT NULL,
    last_price REAL NOT NULL,
    last_fair_value REAL NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS paper_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    executed_at TEXT NOT NULL,
    token_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    bucket_label TEXT NOT NULL,
    action TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    notional_usd REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_paper_executions_token ON paper_executions(token_id);

CREATE TABLE IF NOT EXISTS paper_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    bankroll_usd REAL NOT NULL,
    equity_usd REAL NOT NULL,
    markets_scored INTEGER NOT NULL,
    outcomes_scored INTEGER NOT NULL,
    signals INTEGER NOT NULL,
    executions INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS forecast_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    question TEXT NOT NULL,
    city TEXT NOT NULL,
    target_date TEXT,
    bucket_label TEXT NOT NULL,
    token_id TEXT,
    fair_value REAL NOT NULL,
    market_price REAL NOT NULL,
    edge REAL NOT NULL,
    model_count INTEGER NOT NULL,
    model_agreement REAL NOT NULL,
    probability_stdev REAL NOT NULL,
    entry_eligible INTEGER NOT NULL,
    entry_filter_reason TEXT,
    observed_high_f REAL,
    observation_source TEXT,
    observation_final INTEGER NOT NULL DEFAULT 0,
    observation_adjusted INTEGER NOT NULL DEFAULT 0,
    observed_outcome INTEGER,
    model_probabilities_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_forecast_scores_market ON forecast_scores(market_id, bucket_label);
CREATE INDEX IF NOT EXISTS idx_forecast_scores_target ON forecast_scores(city, target_date);
CREATE INDEX IF NOT EXISTS idx_forecast_scores_observed ON forecast_scores(observed_outcome);
"""


class PaperLedger:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def record_signals(self, signals: Iterable[TradeSignal], metadata: Optional[dict] = None) -> int:
        rows = 0
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with sqlite3.connect(str(self.path)) as conn:
            for signal in signals:
                conn.execute(
                    """
                    INSERT INTO paper_trades (
                        generated_at, market_id, market_slug, question, city, target_date,
                        bucket_label, token_id, side, fair_value, market_price, edge,
                        size_usd, reason, rule_excerpt, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.generated_at.isoformat(),
                        signal.market_id,
                        signal.market_slug,
                        signal.question,
                        signal.city,
                        signal.target_date.isoformat() if signal.target_date else None,
                        signal.bucket_label,
                        signal.token_id,
                        signal.side.value,
                        signal.fair_value,
                        signal.market_price,
                        signal.edge,
                        signal.size_usd,
                        signal.reason,
                        signal.rule_excerpt,
                        metadata_json,
                    ),
                )
                rows += 1
        return rows

    def open_trades(self) -> list[dict]:
        with sqlite3.connect(str(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute("SELECT * FROM paper_trades WHERE status = 'OPEN' ORDER BY generated_at DESC")]

    def record_forecast_scores(self, scored_outcomes: Iterable[ScoredOutcome]) -> int:
        rows = 0
        with sqlite3.connect(str(self.path)) as conn:
            for outcome in scored_outcomes:
                conn.execute(
                    """
                    INSERT INTO forecast_scores (
                        generated_at, market_id, market_slug, question, city, target_date,
                        bucket_label, token_id, fair_value, market_price, edge,
                        model_count, model_agreement, probability_stdev, entry_eligible,
                        entry_filter_reason, observed_high_f, observation_source,
                        observation_final, observation_adjusted, observed_outcome,
                        model_probabilities_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome.generated_at.isoformat(),
                        outcome.market_id,
                        outcome.market_slug,
                        outcome.question,
                        outcome.city,
                        outcome.target_date.isoformat() if outcome.target_date else None,
                        outcome.bucket_label,
                        outcome.token_id,
                        outcome.fair_value,
                        outcome.market_price,
                        outcome.edge,
                        outcome.model_count,
                        outcome.model_agreement,
                        outcome.probability_stdev,
                        1 if outcome.entry_eligible else 0,
                        outcome.entry_filter_reason,
                        outcome.observed_high_f,
                        outcome.observation_source,
                        1 if outcome.observation_final else 0,
                        1 if outcome.observation_adjusted else 0,
                        outcome.observed_outcome,
                        json.dumps(outcome.model_probabilities, sort_keys=True),
                    ),
                )
                rows += 1
        return rows

    def calibration_summary(self) -> dict:
        with sqlite3.connect(str(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_scores,
                    SUM(CASE WHEN observed_high_f IS NOT NULL THEN 1 ELSE 0 END) AS scores_with_observation,
                    SUM(CASE WHEN observation_adjusted = 1 THEN 1 ELSE 0 END) AS observation_adjusted_scores,
                    SUM(CASE WHEN observed_outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved_scores
                FROM forecast_scores
                """
            ).fetchone()
            resolved = conn.execute(
                """
                SELECT fair_value, observed_outcome
                FROM forecast_scores
                WHERE observed_outcome IS NOT NULL
                """
            ).fetchall()
        brier = None
        log_loss = None
        if resolved:
            brier = sum((float(item["fair_value"]) - float(item["observed_outcome"])) ** 2 for item in resolved) / len(resolved)
            log_loss = sum(_log_loss(float(item["fair_value"]), int(item["observed_outcome"])) for item in resolved) / len(resolved)
        return {
            "total_scores": int(row["total_scores"] or 0),
            "scores_with_observation": int(row["scores_with_observation"] or 0),
            "observation_adjusted_scores": int(row["observation_adjusted_scores"] or 0),
            "resolved_scores": int(row["resolved_scores"] or 0),
            "brier_score": round(brier, 6) if brier is not None else None,
            "log_loss": round(log_loss, 6) if log_loss is not None else None,
        }

    def rebalance_kelly(
        self,
        scored_outcomes: Iterable[ScoredOutcome],
        bankroll_usd: float,
        kelly_fraction: float = 0.25,
        max_position_usd: float = 50.0,
        min_trade_usd: float = 1.0,
        min_edge: float = 0.08,
        min_model_count: int = 3,
        min_model_agreement: float = 0.65,
        min_price: float = 0.05,
        max_price: float = 0.95,
    ) -> int:
        scored = [outcome for outcome in scored_outcomes if outcome.token_id]
        selected_entry_tokens = self._selected_entry_tokens(scored, min_edge, min_model_count, min_model_agreement, min_price, max_price)
        executions = 0
        with sqlite3.connect(str(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            for outcome in scored:
                assert outcome.token_id is not None
                if outcome.token_id in selected_entry_tokens:
                    target_notional = self._kelly_target_notional(outcome, bankroll_usd, kelly_fraction, max_position_usd)
                else:
                    target_notional = 0.0
                current = conn.execute("SELECT * FROM paper_positions WHERE token_id = ?", (outcome.token_id,)).fetchone()
                current_shares = float(current["shares"]) if current else 0.0
                current_notional = current_shares * outcome.market_price
                delta_notional = target_notional - current_notional
                if target_notional <= 0 and current_shares > 0:
                    self._sell(conn, outcome, current, current_shares)
                    executions += 1
                    continue
                if abs(delta_notional) < min_trade_usd:
                    self._mark_position(conn, outcome, current)
                    continue
                if delta_notional > 0:
                    shares = delta_notional / outcome.market_price
                    self._buy(conn, outcome, shares)
                    executions += 1
                elif current_shares > 0:
                    shares = min(current_shares, abs(delta_notional) / outcome.market_price)
                    self._sell(conn, outcome, current, shares)
                    executions += 1
        return executions

    def record_run(self, bankroll_usd: float, markets_scored: int, outcomes_scored: int, signals: int, executions: int, metadata: Optional[dict] = None) -> float:
        equity = self.equity_usd(bankroll_usd)
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute(
                """
                INSERT INTO paper_runs (
                    run_at, bankroll_usd, equity_usd, markets_scored, outcomes_scored,
                    signals, executions, metadata_json
                ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)
                """,
                (bankroll_usd, equity, markets_scored, outcomes_scored, signals, executions, json.dumps(metadata or {}, sort_keys=True)),
            )
        return equity

    def equity_usd(self, bankroll_usd: float) -> float:
        with sqlite3.connect(str(self.path)) as conn:
            invested = conn.execute("SELECT COALESCE(SUM(cost_basis), 0) FROM paper_positions").fetchone()[0]
            marked = conn.execute("SELECT COALESCE(SUM(shares * last_price), 0) FROM paper_positions").fetchone()[0]
            realized = conn.execute("SELECT COALESCE(SUM(realized_pnl), 0) FROM paper_executions").fetchone()[0]
        return bankroll_usd - float(invested) + float(marked) + float(realized)

    def positions(self) -> list[dict]:
        with sqlite3.connect(str(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute("SELECT * FROM paper_positions ORDER BY cost_basis DESC")]

    def settle_expired_positions(self, observation_client: ObservedHighClient, now: Optional[datetime] = None) -> tuple[int, int]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        settled = 0
        errors = 0
        with sqlite3.connect(str(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            positions = conn.execute("SELECT * FROM paper_positions").fetchall()
            for position in positions:
                city = find_city(str(position["city"]))
                target = _parse_position_date(position["target_date"])
                bucket = parse_temperature_bucket(str(position["bucket_label"]))
                if city is None or target is None or bucket is None:
                    errors += 1
                    continue
                if target >= _local_date(city.timezone, current):
                    continue
                try:
                    observed = observation_client.fetch_observed_high(city, target, now=current)
                except (RuntimeError, ValueError):
                    observed = None
                if observed is None or observed.max_temperature_f is None or not observed.is_final:
                    errors += 1
                    continue
                payout_price = 1.0 if bucket.contains(observed.max_temperature_f) else 0.0
                shares = float(position["shares"])
                cost_basis = float(position["cost_basis"])
                notional = shares * payout_price
                realized_pnl = notional - cost_basis
                conn.execute("DELETE FROM paper_positions WHERE token_id = ?", (position["token_id"],))
                conn.execute(
                    """
                    INSERT INTO paper_executions (
                        executed_at, token_id, market_id, market_slug, bucket_label, action,
                        shares, price, notional_usd, realized_pnl, reason, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        current.isoformat(),
                        position["token_id"],
                        position["market_id"],
                        position["market_slug"],
                        position["bucket_label"],
                        "SETTLE",
                        shares,
                        payout_price,
                        notional,
                        realized_pnl,
                        "expired market final observation settlement",
                        json.dumps(
                            {
                                "observed_high_f": round(observed.max_temperature_f, 2),
                                "observation_source": observed.source,
                                "target_date": target.isoformat(),
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                settled += 1
        return settled, errors

    @staticmethod
    def _kelly_target_notional(outcome: ScoredOutcome, bankroll_usd: float, kelly_fraction: float, max_position_usd: float) -> float:
        price = max(0.0001, min(0.9999, outcome.market_price))
        raw_fraction = max(0.0, (outcome.fair_value - price) / (1.0 - price))
        agreement_scaled = raw_fraction * max(0.0, min(1.0, outcome.model_agreement))
        return min(max_position_usd, bankroll_usd * kelly_fraction * agreement_scaled)

    @staticmethod
    def _is_trade_eligible(
        outcome: ScoredOutcome,
        min_edge: float,
        min_model_count: int,
        min_model_agreement: float,
        min_price: float,
        max_price: float,
    ) -> bool:
        return (
            outcome.entry_eligible
            and outcome.edge >= min_edge
            and outcome.model_count >= min_model_count
            and outcome.model_agreement >= min_model_agreement
            and min_price <= outcome.market_price <= max_price
        )

    @classmethod
    def _selected_entry_tokens(
        cls,
        scored: list[ScoredOutcome],
        min_edge: float,
        min_model_count: int,
        min_model_agreement: float,
        min_price: float,
        max_price: float,
    ) -> set[str]:
        selected: dict[tuple[str, Optional[str]], ScoredOutcome] = {}
        for outcome in scored:
            if outcome.token_id is None:
                continue
            if not cls._is_trade_eligible(outcome, min_edge, min_model_count, min_model_agreement, min_price, max_price):
                continue
            group_key = (outcome.city, outcome.target_date.isoformat() if outcome.target_date else None)
            current = selected.get(group_key)
            if current is None or (outcome.edge, outcome.model_agreement) > (current.edge, current.model_agreement):
                selected[group_key] = outcome
        return {outcome.token_id for outcome in selected.values() if outcome.token_id}

    def _buy(self, conn: sqlite3.Connection, outcome: ScoredOutcome, shares: float) -> None:
        current = conn.execute("SELECT * FROM paper_positions WHERE token_id = ?", (outcome.token_id,)).fetchone()
        notional = shares * outcome.market_price
        if current:
            new_shares = float(current["shares"]) + shares
            new_cost = float(current["cost_basis"]) + notional
        else:
            new_shares = shares
            new_cost = notional
        self._upsert_position(conn, outcome, new_shares, new_cost)
        self._record_execution(conn, outcome, "BUY", shares, outcome.market_price, notional, 0.0, "kelly target increase")

    def _sell(self, conn: sqlite3.Connection, outcome: ScoredOutcome, current: sqlite3.Row, shares: float) -> None:
        current_shares = float(current["shares"])
        current_cost = float(current["cost_basis"])
        cost_reduction = current_cost * (shares / current_shares) if current_shares else 0.0
        notional = shares * outcome.market_price
        realized_pnl = notional - cost_reduction
        remaining_shares = current_shares - shares
        remaining_cost = current_cost - cost_reduction
        if remaining_shares <= 1e-9:
            conn.execute("DELETE FROM paper_positions WHERE token_id = ?", (outcome.token_id,))
        else:
            self._upsert_position(conn, outcome, remaining_shares, remaining_cost)
        self._record_execution(conn, outcome, "SELL", shares, outcome.market_price, notional, realized_pnl, "kelly target reduction")

    def _mark_position(self, conn: sqlite3.Connection, outcome: ScoredOutcome, current: Optional[sqlite3.Row]) -> None:
        if current:
            self._upsert_position(conn, outcome, float(current["shares"]), float(current["cost_basis"]))

    def _upsert_position(self, conn: sqlite3.Connection, outcome: ScoredOutcome, shares: float, cost_basis: float) -> None:
        conn.execute(
            """
            INSERT INTO paper_positions (
                token_id, market_id, market_slug, question, bucket_label, city, target_date,
                shares, cost_basis, last_price, last_fair_value, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(token_id) DO UPDATE SET
                shares=excluded.shares,
                cost_basis=excluded.cost_basis,
                last_price=excluded.last_price,
                last_fair_value=excluded.last_fair_value,
                updated_at=excluded.updated_at,
                metadata_json=excluded.metadata_json
            """,
            (
                outcome.token_id,
                outcome.market_id,
                outcome.market_slug,
                outcome.question,
                outcome.bucket_label,
                outcome.city,
                outcome.target_date.isoformat() if outcome.target_date else None,
                shares,
                cost_basis,
                outcome.market_price,
                outcome.fair_value,
                outcome.generated_at.isoformat(),
                json.dumps(
                    {
                        "edge": outcome.edge,
                        "model_count": outcome.model_count,
                        "model_agreement": outcome.model_agreement,
                        "probability_stdev": outcome.probability_stdev,
                        "entry_eligible": outcome.entry_eligible,
                        "entry_filter_reason": outcome.entry_filter_reason,
                        "observed_high_f": outcome.observed_high_f,
                        "observation_source": outcome.observation_source,
                        "observation_final": outcome.observation_final,
                        "observation_adjusted": outcome.observation_adjusted,
                        "observed_outcome": outcome.observed_outcome,
                        "model_probabilities": outcome.model_probabilities,
                    },
                    sort_keys=True,
                ),
            ),
        )

    def _record_execution(
        self,
        conn: sqlite3.Connection,
        outcome: ScoredOutcome,
        action: str,
        shares: float,
        price: float,
        notional: float,
        realized_pnl: float,
        reason: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO paper_executions (
                executed_at, token_id, market_id, market_slug, bucket_label, action,
                shares, price, notional_usd, realized_pnl, reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.generated_at.isoformat(),
                outcome.token_id,
                outcome.market_id,
                outcome.market_slug,
                outcome.bucket_label,
                action,
                shares,
                price,
                notional,
                realized_pnl,
                reason,
                json.dumps({"fair_value": outcome.fair_value, "edge": outcome.edge}, sort_keys=True),
            ),
        )

    def _initialize(self) -> None:
        with sqlite3.connect(str(self.path)) as conn:
            conn.executescript(SCHEMA)


def _log_loss(probability: float, outcome: int) -> float:
    probability = max(1e-6, min(1 - 1e-6, probability))
    if outcome:
        return -math.log(probability)
    return -math.log(1 - probability)


def _parse_position_date(value: object) -> Optional[date]:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _local_date(timezone_name: str, current: datetime) -> date:
    try:
        timezone_info = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        timezone_info = timezone.utc
    return current.astimezone(timezone_info).date()
