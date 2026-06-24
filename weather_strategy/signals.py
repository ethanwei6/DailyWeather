from __future__ import annotations

from dataclasses import dataclass, replace
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
    min_price: float = 0.125
    yes_side_min_price: float = 0.20
    min_signal_fair_value: float = 0.70
    allow_bounded_bucket_entries: bool = True
    bounded_bucket_min_edge: float = 0.10
    bounded_bucket_min_fair_value: float = 0.90
    bounded_bucket_min_model_agreement: float = 1.0
    bounded_bucket_min_price: float = 0.50
    min_model_count: int = 3
    min_model_agreement: float = 1.0
    hold_min_model_agreement: float = 0.65
    hold_min_fair_value: float = 0.60
    hold_market_confirmation_price: float = 0.80
    hold_market_confirmation_min_fair_value: float = 0.50
    preserve_valid_holds: bool = True
    allow_no_side_entries: bool = False
    no_side_min_edge: float = 0.10
    no_side_high_confidence_min_edge: float = 0.02
    no_side_max_price: Optional[float] = 0.95
    no_side_max_counter_event_probability: Optional[float] = 0.10
    hold_no_side_max_counter_event_probability: Optional[float] = 0.15
    high_confidence_price_threshold: float = 0.75
    high_confidence_min_kelly_edge: float = 0.02
    low_price_exact_bucket_threshold: float = 0.20
    low_price_exact_bucket_min_fair_value: float = 0.22
    low_price_exact_bucket_min_edge: float = 0.08
    correlated_exact_bucket_max_price: float = 0.15
    correlated_exact_bucket_min_agreement: float = 0.95
    exact_bucket_max_width_f: float = 2.25
    enforce_entry_timing_filter: bool = True
    same_day_earliest_entry_hour_local: int = 11
    same_day_latest_entry_hour_local: int = 17


def generate_signals(
    market: WeatherMarket,
    fair_values: Mapping[str, float],
    quotes_by_token: Optional[Mapping[str, OrderBookQuote]] = None,
    settings: Optional[SignalSettings] = None,
    now: Optional[datetime] = None,
) -> list[TradeSignal]:
    settings = settings or SignalSettings()
    current = now or datetime.now(timezone.utc)
    entry_eligible, _ = market_entry_timing(market, settings=settings, now=current)
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
        if market_price < settings.yes_side_min_price:
            continue
        if _fails_bounded_bucket_entry_gate(bucket.lower_f, bucket.upper_f, settings):
            continue
        if fair_value < settings.min_signal_fair_value:
            continue
        edge = fair_value - market_price - _price_adjusted_uncertainty_buffer(market_price, settings)
        if _fails_bounded_bucket_quality_gate(
            fair_value=fair_value,
            edge=edge,
            market_price=market_price,
            model_agreement=1.0,
            lower_f=bucket.lower_f,
            upper_f=bucket.upper_f,
            settings=settings,
        ):
            continue
        if not _passes_edge_gate(edge, market_price, settings):
            continue
        if _fails_low_price_exact_bucket_gate(
            fair_value=fair_value,
            edge=edge,
            market_price=market_price,
            bucket_width_f=_bucket_width_f(bucket.lower_f, bucket.upper_f),
            settings=settings,
        ):
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
                generated_at=current,
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
    now: Optional[datetime] = None,
) -> list[ScoredOutcome]:
    settings = settings or SignalSettings()
    current = now or datetime.now(timezone.utc)
    entry_eligible, entry_filter_reason = market_entry_timing(market, settings=settings, now=current)
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
                generated_at=current,
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
                bucket_lower_f=bucket.lower_f,
                bucket_upper_f=bucket.upper_f,
                bucket_width_f=_bucket_width_f(bucket.lower_f, bucket.upper_f),
                resolution_unit=bucket.resolution_unit,
                resolution_precision=bucket.resolution_precision,
            )
        )
    return scored


def signals_from_scored_outcomes(scored: list[ScoredOutcome], settings: Optional[SignalSettings] = None) -> list[TradeSignal]:
    settings = settings or SignalSettings()
    candidates = []
    for outcome in scored:
        if signal_filter_reason(outcome, settings) is not None:
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


def signal_filter_reason(outcome: ScoredOutcome, settings: Optional[SignalSettings] = None) -> Optional[str]:
    settings = settings or SignalSettings()
    if not outcome.entry_eligible:
        return outcome.entry_filter_reason or "entry timing filter"
    if outcome.market_price < settings.min_price:
        return f"market price below {_format_threshold(settings.min_price)}"
    if not _is_no_side_outcome(outcome) and outcome.market_price < settings.yes_side_min_price:
        return f"YES-side market price below {_format_threshold(settings.yes_side_min_price)}"
    if outcome.market_price > settings.max_price:
        return f"market price above {_format_threshold(settings.max_price)}"
    if _is_no_side_outcome(outcome) and _fails_no_side_max_price_gate(outcome.market_price, settings):
        return f"NO-side market price above {_format_threshold(settings.no_side_max_price or 0.0)}"
    if _fails_bounded_bucket_entry_gate(outcome.bucket_lower_f, outcome.bucket_upper_f, settings):
        return "bounded exact/range bucket entries disabled"
    if outcome.fair_value < settings.min_signal_fair_value:
        return f"fair value below {settings.min_signal_fair_value:.2f}"
    if not _passes_edge_gate(outcome.edge, outcome.market_price, settings):
        return "buffered edge below price-aware Kelly threshold"
    bounded_reason = _bounded_bucket_quality_filter_reason(outcome, settings)
    if bounded_reason is not None:
        return bounded_reason
    if _is_no_side_outcome(outcome):
        no_side_edge_floor = _required_no_side_min_edge(outcome.market_price, settings)
        if outcome.edge < no_side_edge_floor:
            return f"NO-side edge below {no_side_edge_floor:.2f}"
    if _is_no_side_outcome(outcome) and _fails_no_side_counter_event_gate(outcome.model_probabilities, settings):
        return f"NO-side counter-event probability above {_format_threshold(settings.no_side_max_counter_event_probability or 0.0)}"
    if outcome.model_count < settings.min_model_count:
        return f"model count below {settings.min_model_count}"
    if outcome.model_agreement < settings.min_model_agreement:
        return f"model agreement below {settings.min_model_agreement:.2f}"
    if _fails_low_price_exact_bucket_gate(
        fair_value=outcome.fair_value,
        edge=outcome.edge,
        market_price=outcome.market_price,
        bucket_width_f=outcome.bucket_width_f,
        settings=settings,
    ):
        return (
            "cheap exact-temperature bucket requires stronger historical edge "
            f"(FV>={settings.low_price_exact_bucket_min_fair_value:.2f}, "
            f"edge>={settings.low_price_exact_bucket_min_edge:.2f})"
        )
    if _fails_correlated_exact_bucket_gate(outcome, settings):
        return (
            "cheap exact-temperature bucket has over-correlated source agreement "
            f"(agreement>={settings.correlated_exact_bucket_min_agreement:.2f})"
        )
    return None


def hold_filter_reason(outcome: ScoredOutcome, settings: Optional[SignalSettings] = None) -> Optional[str]:
    settings = settings or SignalSettings()
    hold_outcome = replace(outcome, entry_eligible=True, entry_filter_reason=None)
    if hold_outcome.observation_final and hold_outcome.observed_outcome == 0:
        return "final observation contradicts bucket"
    if hold_outcome.market_price < settings.min_price:
        return f"market price below {_format_threshold(settings.min_price)}"
    if _is_no_side_outcome(hold_outcome) and _fails_no_side_counter_event_gate(
        hold_outcome.model_probabilities,
        settings,
        threshold=settings.hold_no_side_max_counter_event_probability,
    ):
        return f"NO-side hold counter-event probability above {_format_threshold(settings.hold_no_side_max_counter_event_probability or 0.0)}"
    if _fails_bounded_bucket_entry_gate(hold_outcome.bucket_lower_f, hold_outcome.bucket_upper_f, settings):
        return "bounded exact/range bucket entries disabled"
    if hold_outcome.model_count < settings.min_model_count:
        return f"model count below {settings.min_model_count}"
    if hold_outcome.model_agreement < settings.hold_min_model_agreement:
        return f"model agreement below {settings.hold_min_model_agreement:.2f}"
    market_confirmed = (
        hold_outcome.market_price >= settings.hold_market_confirmation_price
        and hold_outcome.fair_value >= settings.hold_market_confirmation_min_fair_value
    )
    if hold_outcome.fair_value < settings.hold_min_fair_value and not market_confirmed:
        return f"hold fair value below {settings.hold_min_fair_value:.2f}"
    return None


def _price_adjusted_uncertainty_buffer(market_price: float, settings: SignalSettings) -> float:
    return settings.uncertainty_buffer * max(0.05, 1.0 - market_price)


def _fails_bounded_bucket_entry_gate(lower_f: Optional[float], upper_f: Optional[float], settings: SignalSettings) -> bool:
    return not settings.allow_bounded_bucket_entries and lower_f is not None and upper_f is not None


def _bounded_bucket_quality_filter_reason(outcome: ScoredOutcome, settings: SignalSettings) -> Optional[str]:
    if not _is_bounded_bucket(outcome.bucket_lower_f, outcome.bucket_upper_f):
        return None
    if outcome.market_price < settings.bounded_bucket_min_price:
        return f"bounded exact/range bucket price below {_format_threshold(settings.bounded_bucket_min_price)}"
    if outcome.fair_value < settings.bounded_bucket_min_fair_value:
        return f"bounded exact/range bucket fair value below {settings.bounded_bucket_min_fair_value:.2f}"
    if outcome.edge < settings.bounded_bucket_min_edge:
        return f"bounded exact/range bucket edge below {settings.bounded_bucket_min_edge:.2f}"
    if outcome.model_agreement < settings.bounded_bucket_min_model_agreement:
        return f"bounded exact/range bucket agreement below {settings.bounded_bucket_min_model_agreement:.2f}"
    return None


def _fails_bounded_bucket_quality_gate(
    *,
    fair_value: float,
    edge: float,
    market_price: float,
    model_agreement: float,
    lower_f: Optional[float],
    upper_f: Optional[float],
    settings: SignalSettings,
) -> bool:
    if not _is_bounded_bucket(lower_f, upper_f):
        return False
    return (
        market_price < settings.bounded_bucket_min_price
        or fair_value < settings.bounded_bucket_min_fair_value
        or edge < settings.bounded_bucket_min_edge
        or model_agreement < settings.bounded_bucket_min_model_agreement
    )


def _is_bounded_bucket(lower_f: Optional[float], upper_f: Optional[float]) -> bool:
    return lower_f is not None and upper_f is not None


def _required_buffered_edge(market_price: float, settings: SignalSettings) -> float:
    min_kelly_edge = settings.min_edge
    if market_price >= settings.high_confidence_price_threshold:
        min_kelly_edge = min(min_kelly_edge, settings.high_confidence_min_kelly_edge)
    return min_kelly_edge * max(0.0001, 1.0 - market_price)


def _passes_edge_gate(edge: float, market_price: float, settings: SignalSettings) -> bool:
    return edge >= _required_buffered_edge(market_price, settings)


def _required_no_side_min_edge(market_price: float, settings: SignalSettings) -> float:
    if market_price >= settings.high_confidence_price_threshold:
        return min(settings.no_side_min_edge, settings.no_side_high_confidence_min_edge)
    return settings.no_side_min_edge


def _fails_no_side_max_price_gate(market_price: float, settings: SignalSettings) -> bool:
    threshold = settings.no_side_max_price
    if threshold is None or threshold >= settings.max_price:
        return False
    return market_price > threshold


def _is_no_side_outcome(outcome: ScoredOutcome) -> bool:
    return outcome.bucket_label.startswith("NO: ") or outcome.question.startswith("NO: ")


def no_side_counter_event_probability(model_probabilities: Mapping[str, float]) -> Optional[float]:
    values = [max(0.0, min(1.0, float(value))) for value in model_probabilities.values()]
    if not values:
        return None
    return 1.0 - min(values)


def invert_binary_scored_outcome(
    outcome: ScoredOutcome,
    settings: SignalSettings,
    *,
    token_id: Optional[str] = None,
    market_price: Optional[float] = None,
) -> ScoredOutcome:
    price = outcome.market_price if market_price is None else float(market_price)
    model_probabilities = {
        model_name: 1.0 - max(0.0, min(1.0, float(probability)))
        for model_name, probability in outcome.model_probabilities.items()
    }
    fair_value = 1.0 - outcome.fair_value
    buffer = _price_adjusted_uncertainty_buffer(price, settings)
    agreement = ConsensusValue(
        bucket_label=outcome.bucket_label,
        fair_value=fair_value,
        model_probabilities=model_probabilities,
        model_count=outcome.model_count,
        probability_stdev=outcome.probability_stdev,
    ).agreement_above(price, buffer)
    return replace(
        outcome,
        question=outcome.question if outcome.question.startswith("NO: ") else f"NO: {outcome.question}",
        bucket_label=outcome.bucket_label if outcome.bucket_label.startswith("NO: ") else f"NO: {outcome.bucket_label}",
        token_id=token_id or outcome.token_id,
        fair_value=round(fair_value, 4),
        market_price=round(price, 4),
        edge=round(fair_value - price - buffer, 4),
        model_agreement=round(agreement, 4),
        model_probabilities=model_probabilities,
        observed_outcome=_invert_binary_outcome(outcome.observed_outcome),
    )


def _fails_no_side_counter_event_gate(
    model_probabilities: Mapping[str, float],
    settings: SignalSettings,
    *,
    threshold: Optional[float] = None,
) -> bool:
    threshold = settings.no_side_max_counter_event_probability if threshold is None else threshold
    if threshold is None or threshold >= 1.0:
        return False
    counter_probability = no_side_counter_event_probability(model_probabilities)
    return counter_probability is not None and counter_probability > threshold


def _fails_low_price_exact_bucket_gate(
    *,
    fair_value: float,
    edge: float,
    market_price: float,
    bucket_width_f: Optional[float],
    settings: SignalSettings,
) -> bool:
    if market_price >= settings.low_price_exact_bucket_threshold:
        return False
    if bucket_width_f is None or bucket_width_f <= 0 or bucket_width_f > settings.exact_bucket_max_width_f:
        return False
    return fair_value < settings.low_price_exact_bucket_min_fair_value or edge < settings.low_price_exact_bucket_min_edge


def _fails_correlated_exact_bucket_gate(outcome: ScoredOutcome, settings: SignalSettings) -> bool:
    if outcome.market_price >= settings.correlated_exact_bucket_max_price:
        return False
    bucket_width_f = outcome.bucket_width_f
    if bucket_width_f is None or bucket_width_f <= 0 or bucket_width_f > settings.exact_bucket_max_width_f:
        return False
    return outcome.model_agreement >= settings.correlated_exact_bucket_min_agreement


def _bucket_width_f(lower_f: Optional[float], upper_f: Optional[float]) -> Optional[float]:
    if lower_f is None or upper_f is None:
        return None
    return max(0.0, upper_f - lower_f)


def _format_threshold(value: float) -> str:
    return f"{value:.3g}"


def _invert_binary_outcome(value: object) -> Optional[int]:
    if value not in (0, 1):
        return None
    return 1 - int(value)


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
