from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class CityConfig:
    name: str
    state: str
    latitude: float
    longitude: float
    timezone: str
    nws_station: Optional[str] = None
    aliases: tuple[str, ...] = ()
    metar_station: Optional[str] = None

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.state}"


@dataclass(frozen=True)
class TemperatureBucket:
    label: str
    lower_f: Optional[float]
    upper_f: Optional[float]
    token_id: Optional[str] = None
    market_price: Optional[float] = None

    def contains(self, value_f: float) -> bool:
        if self.lower_f is not None and value_f < self.lower_f:
            return False
        if self.upper_f is not None and value_f > self.upper_f:
            return False
        return True


@dataclass(frozen=True)
class WeatherMarket:
    id: str
    question: str
    slug: str
    event_slug: Optional[str]
    close_time: Optional[datetime]
    target_date: Optional[date]
    city: Optional[CityConfig]
    resolution_rules: str
    buckets: tuple[TemperatureBucket, ...]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastDistribution:
    city: CityConfig
    target_date: date
    samples_f: tuple[float, ...]
    generated_at: datetime
    source: str
    model_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.samples_f)

    @property
    def mean_f(self) -> float:
        if not self.samples_f:
            raise ValueError("ForecastDistribution has no samples")
        return sum(self.samples_f) / len(self.samples_f)


@dataclass(frozen=True)
class OrderBookQuote:
    token_id: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None and self.best_ask is None:
            return None
        if self.best_bid is None:
            return self.best_ask
        if self.best_ask is None:
            return self.best_bid
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


@dataclass(frozen=True)
class TradeSignal:
    market_id: str
    market_slug: str
    question: str
    bucket_label: str
    token_id: Optional[str]
    side: Side
    fair_value: float
    market_price: float
    edge: float
    size_usd: float
    reason: str
    generated_at: datetime
    city: str
    target_date: Optional[date]
    rule_excerpt: str


@dataclass(frozen=True)
class ConsensusValue:
    bucket_label: str
    fair_value: float
    model_probabilities: dict[str, float]
    model_count: int
    probability_stdev: float
    observed_high_f: Optional[float] = None
    observation_source: Optional[str] = None
    observation_final: bool = False
    observation_adjusted: bool = False

    def agreement_above(self, market_price: float, buffer: float = 0.0) -> float:
        if not self.model_probabilities:
            return 0.0
        source_probabilities = _source_probability_views(self.model_probabilities)
        agreeing = sum(1 for probability in source_probabilities.values() if probability > market_price + buffer)
        return agreeing / len(source_probabilities)


@dataclass(frozen=True)
class ScoredOutcome:
    market_id: str
    market_slug: str
    question: str
    bucket_label: str
    token_id: Optional[str]
    fair_value: float
    market_price: float
    edge: float
    model_count: int
    model_agreement: float
    probability_stdev: float
    generated_at: datetime
    city: str
    target_date: Optional[date]
    rule_excerpt: str
    model_probabilities: dict[str, float]
    entry_eligible: bool = True
    entry_filter_reason: Optional[str] = None
    observed_high_f: Optional[float] = None
    observation_source: Optional[str] = None
    observation_final: bool = False
    observation_adjusted: bool = False
    observed_outcome: Optional[int] = None


def _source_probability_views(model_probabilities: dict[str, float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, probability in model_probabilities.items():
        source = key.rsplit(".", 1)[0] if "." in key else key
        grouped.setdefault(source, []).append(float(probability))
    return {
        source: sum(values) / len(values)
        for source, values in grouped.items()
        if values
    }
