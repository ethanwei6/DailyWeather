from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weather_strategy.models import ConsensusValue, OrderBookQuote, ScoredOutcome, Side, TradeSignal, WeatherMarket
from weather_strategy.observations import observed_outcome_for_bucket


@dataclass(frozen=True)
class SignalSettings:
    min_edge: float = 0.08
    uncertainty_buffer: float = 0.03
    max_spread: float = 0.10
    default_size_usd: float = 10.0
    max_price: float = 0.95
    min_price: float = 0.05
    min_model_count: int = 3
    min_model_agreement: float = 0.65
    high_confidence_price_threshold: float = 0.75
    high_confidence_min_kelly_edge: float = 0.02
    enforce_entry_timing_filter: bool = True
    same_day_earliest_entry_hour_local: int = 11
    same_day_latest_entry_hour_local: int = 17


def generate_signals(
    market: WeatherMarket,
    fair_values: Mapping[str, float],
    quotes_by_token: Optional[Mapping[str, OrderBookQuote]] = None,
    settings: Optional[SignalSettings] = None,
) -> list[TradeSignal]:
    settings = settings or SignalSettings()
    entry_eligible, _ = market_entry_timing(market, settings=settings)
    if not entry_eligible:
        return []
    signals = []
    for bucket in market.buckets:
        fair_value = fair_values.get(bucket.label)
        if fair_value is None:
            continue
        quote = quotes_by_token.get(bucket.token_id) if quotes_by_token and bucket.token_id else None
        market_price = _entry_price(bucket.market_price, quote)
        if market_price is None:
            continue
        spread = quote.spread if quote else None
        if spread is not None and spread > settings.max_spread:
            continue
        if market_price < settings.min_price or market_price > settings.max_price:
            continue
        edge = fair_value - market_price - _price_adjusted_uncertainty_buffer(market_price, settings)
        if not _passes_edge_gate(edge, market_price, settings):
            continue
        signals.append(
            TradeSignal(
                market_id=market.id,
                market_slug=market.slug,
                question=market.question,
                bucket_label=bucket.label,
                token_id=bucket.token_id,
                side=Side.BUY,
                fair_value=round(fair_value, 4),
                market_price=round(market_price, 4),
                edge=round(edge, 4),
                size_usd=settings.default_size_usd,
                reason=f"FV {fair_value:.2%} exceeds entry {market_price:.2%}; buffered Kelly edge gate passed",
                generated_at=datetime.now(timezone.utc),
                city=market.city.display_name if market.city else "unknown",
                target_date=market.target_date,
                rule_excerpt=market.resolution_rules[:500],
            )
        )
    return sorted(signals, key=lambda signal: signal.edge, reverse=True)


def score_outcomes(
    market: WeatherMarket,
    consensus_values: Mapping[str, ConsensusValue],
    quotes_by_token: Optional[Mapping[str, OrderBookQuote]] = None,
    settings: Optional[SignalSettings] = None,
) -> list[ScoredOutcome]:
    settings = settings or SignalSettings()
    entry_eligible, entry_filter_reason = market_entry_timing(market, settings=settings)
    scored = []
    for bucket in market.buckets:
        consensus = consensus_values.get(bucket.label)
        if consensus is None:
            continue
        quote = quotes_by_token.get(bucket.token_id) if quotes_by_token and bucket.token_id else None
        market_price = _entry_price(bucket.market_price, quote)
        if market_price is None:
            continue
        spread = quote.spread if quote else None
        if spread is not None and spread > settings.max_spread:
            continue
        buffer = _price_adjusted_uncertainty_buffer(market_price, settings)
        model_agreement = consensus.agreement_above(market_price, buffer)
        edge = consensus.fair_value - market_price - buffer
        scored.append(
            ScoredOutcome(
                market_id=market.id,
                market_slug=market.slug,
                question=market.question,
                bucket_label=bucket.label,
                token_id=bucket.token_id,
                fair_value=round(consensus.fair_value, 4),
                market_price=round(market_price, 4),
                edge=round(edge, 4),
                model_count=consensus.model_count,
                model_agreement=round(model_agreement, 4),
                probability_stdev=round(consensus.probability_stdev, 4),
                generated_at=datetime.now(timezone.utc),
                city=market.city.display_name if market.city else "unknown",
                target_date=market.target_date,
                rule_excerpt=market.resolution_rules[:500],
                model_probabilities=consensus.model_probabilities,
                entry_eligible=entry_eligible,
                entry_filter_reason=entry_filter_reason,
                observed_high_f=consensus.observed_high_f,
                observation_source=consensus.observation_source,
                observation_final=consensus.observation_final,
                observation_adjusted=consensus.observation_adjusted,
                observed_outcome=observed_outcome_for_bucket(bucket, consensus.observed_high_f, consensus.observation_final),
            )
        )
    return scored


def signals_from_scored_outcomes(scored: list[ScoredOutcome], settings: Optional[SignalSettings] = None) -> list[TradeSignal]:
    settings = settings or SignalSettings()
    candidates = []
    for outcome in scored:
        if not outcome.entry_eligible:
            continue
        if outcome.market_price < settings.min_price or outcome.market_price > settings.max_price:
            continue
        if not _passes_edge_gate(outcome.edge, outcome.market_price, settings):
            continue
        if outcome.model_count < settings.min_model_count:
            continue
        if outcome.model_agreement < settings.min_model_agreement:
            continue
        candidates.append(
            TradeSignal(
                market_id=outcome.market_id,
                market_slug=outcome.market_slug,
                question=outcome.question,
                bucket_label=outcome.bucket_label,
                token_id=outcome.token_id,
                side=Side.BUY,
                fair_value=outcome.fair_value,
                market_price=outcome.market_price,
                edge=outcome.edge,
                size_usd=settings.default_size_usd,
                reason=(
                    f"FV {outcome.fair_value:.2%} exceeds entry {outcome.market_price:.2%}; "
                    f"{outcome.model_agreement:.0%} model agreement across {outcome.model_count} model views; "
                    f"buffered Kelly edge gate passed"
                ),
                generated_at=outcome.generated_at,
                city=outcome.city,
                target_date=outcome.target_date,
                rule_excerpt=outcome.rule_excerpt,
            )
        )
    by_group: dict[tuple[str, Optional[object]], TradeSignal] = {}
    for signal in sorted(candidates, key=lambda item: item.edge, reverse=True):
        group_key = (signal.city, signal.target_date)
        if group_key not in by_group:
            by_group[group_key] = signal
    return sorted(by_group.values(), key=lambda signal: signal.edge, reverse=True)


def _price_adjusted_uncertainty_buffer(market_price: float, settings: SignalSettings) -> float:
    return settings.uncertainty_buffer * max(0.05, 1.0 - market_price)


def _required_buffered_edge(market_price: float, settings: SignalSettings) -> float:
    min_kelly_edge = settings.min_edge
    if market_price >= settings.high_confidence_price_threshold:
        min_kelly_edge = min(min_kelly_edge, settings.high_confidence_min_kelly_edge)
    return min_kelly_edge * max(0.0001, 1.0 - market_price)


def _passes_edge_gate(edge: float, market_price: float, settings: SignalSettings) -> bool:
    return edge >= _required_buffered_edge(market_price, settings)


def market_entry_timing(
    market: WeatherMarket,
    settings: Optional[SignalSettings] = None,
    now: Optional[datetime] = None,
) -> tuple[bool, Optional[str]]:
    settings = settings or SignalSettings()
    if not settings.enforce_entry_timing_filter:
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
    local_now = current.astimezone(city_timezone)
    local_date = local_now.date()
    if market.target_date < local_date:
        return False, "target date has passed in local market time"
    if market.target_date == local_date and local_now.hour < settings.same_day_earliest_entry_hour_local:
        return False, f"same-day market before {settings.same_day_earliest_entry_hour_local}:00 local observation-aware entry window"
    if market.target_date == local_date and local_now.hour >= settings.same_day_latest_entry_hour_local:
        return False, f"same-day market after {settings.same_day_latest_entry_hour_local}:00 local entry cutoff"
    return True, None


def _entry_price(gamma_price: Optional[float], quote: Optional[OrderBookQuote]) -> Optional[float]:
    if quote and quote.best_ask is not None:
        return quote.best_ask
    if gamma_price is not None:
        return gamma_price
    if quote:
        return quote.mid
    return None
