from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

from weather_strategy.cities import DEFAULT_CITIES, find_city
from weather_strategy.models import CityConfig, TemperatureBucket, WeatherMarket


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
_DATE_RE = re.compile(rf"\b(?P<month>{_MONTH_PATTERN})\.?(?:\s+|-)(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:[,\s-]+(?P<year>\d{{4}}))?\b", re.I)


def parse_jsonish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected list-like value, got {type(value).__name__}")


def parse_close_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_market_date(text: str, today: Optional[date] = None) -> Optional[date]:
    today = today or datetime.now(timezone.utc).date()
    matches = list(_DATE_RE.finditer(text))
    if not matches:
        return None
    match = next((item for item in matches if item.group("year")), matches[0])
    month = MONTHS[match.group("month").lower().rstrip(".")]
    day = int(match.group("day"))
    year = int(match.group("year")) if match.group("year") else today.year
    parsed = date(year, month, day)
    if match.group("year") is None and parsed < today:
        parsed = date(year + 1, month, day)
    return parsed


def parse_temperature_bucket(label: str, token_id: Optional[str] = None, market_price: Optional[float] = None) -> Optional[TemperatureBucket]:
    unit = _detect_unit(label)
    clean = " ".join(label.replace("°", "").replace("F", "").replace("f", "").replace("C", "").replace("c", "").split())
    lower = clean.lower()

    range_match = re.search(r"(?P<low>-?\d+(?:\.\d+)?)\s*(?:-|to|through|thru)\s*(?P<high>-?\d+(?:\.\d+)?)", lower)
    if range_match:
        return TemperatureBucket(
            label=label,
            lower_f=_to_fahrenheit(float(range_match.group("low")), unit),
            upper_f=_to_fahrenheit(float(range_match.group("high")), unit),
            token_id=token_id,
            market_price=market_price,
        )

    under_match = re.search(r"(?:under|below|less than|lower than|<)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if under_match:
        return TemperatureBucket(label, None, _to_fahrenheit(float(under_match.group("value")) - 0.01, unit), token_id, market_price)

    at_most_match = re.search(r"(?:at most|no more than|<=)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if at_most_match:
        return TemperatureBucket(label, None, _to_fahrenheit(float(at_most_match.group("value")), unit), token_id, market_price)

    above_match = re.search(r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:or above|or higher|and above|or more|\+)", lower)
    if above_match:
        return TemperatureBucket(label, _to_fahrenheit(float(above_match.group("value")), unit), None, token_id, market_price)

    over_match = re.search(r"(?:over|above|greater than|higher than|>)\s*(?P<value>-?\d+(?:\.\d+)?)", lower)
    if over_match:
        return TemperatureBucket(label, _to_fahrenheit(float(over_match.group("value")) + 0.01, unit), None, token_id, market_price)

    predicate_match = re.search(
        r"\bbe\s+(?P<value>-?\d+(?:\.\d+)?)\s*(?:°\s*)?(?P<unit>[cf])?\s*(?P<qualifier>or below|or lower|or less|or above|or higher|or more)?\b",
        label,
        re.I,
    )
    if predicate_match:
        predicate_unit = (predicate_match.group("unit") or unit).upper()
        value = float(predicate_match.group("value"))
        qualifier = (predicate_match.group("qualifier") or "").lower()
        if "below" in qualifier or "lower" in qualifier or "less" in qualifier:
            return TemperatureBucket(label, None, _to_fahrenheit(value, predicate_unit), token_id, market_price)
        if "above" in qualifier or "higher" in qualifier or "more" in qualifier:
            return TemperatureBucket(label, _to_fahrenheit(value, predicate_unit), None, token_id, market_price)
        return TemperatureBucket(label, _to_fahrenheit(value, predicate_unit), _to_fahrenheit(value + 0.999, predicate_unit), token_id, market_price)

    exact_match = re.fullmatch(r"\s*(?P<value>-?\d+(?:\.\d+)?)\s*", clean)
    if exact_match:
        value = float(exact_match.group("value"))
        return TemperatureBucket(label, _to_fahrenheit(value, unit), _to_fahrenheit(value + 0.999, unit), token_id, market_price)

    return None


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_buckets(outcomes: Iterable[Any], token_ids: Iterable[Any] = (), prices: Iterable[Any] = ()) -> tuple[TemperatureBucket, ...]:
    token_list = [str(token) if token is not None else None for token in token_ids]
    price_list = [_safe_float(price) for price in prices]
    buckets: list[TemperatureBucket] = []
    for index, outcome in enumerate(outcomes):
        label = str(outcome)
        token_id = token_list[index] if index < len(token_list) else None
        price = price_list[index] if index < len(price_list) else None
        bucket = parse_temperature_bucket(label, token_id=token_id, market_price=price)
        if bucket is not None:
            buckets.append(bucket)
    return tuple(buckets)


def parse_binary_temperature_bucket(question: str, outcomes: Iterable[Any], token_ids: Iterable[Any] = (), prices: Iterable[Any] = ()) -> tuple[TemperatureBucket, ...]:
    outcome_list = [str(outcome).lower() for outcome in outcomes]
    if "yes" not in outcome_list or "no" not in outcome_list:
        return ()
    yes_index = outcome_list.index("yes")
    token_list = [str(token) if token is not None else None for token in token_ids]
    price_list = [_safe_float(price) for price in prices]
    token_id = token_list[yes_index] if yes_index < len(token_list) else None
    price = price_list[yes_index] if yes_index < len(price_list) else None
    bucket = parse_temperature_bucket(question, token_id=token_id, market_price=price)
    return (bucket,) if bucket is not None else ()


def parse_weather_market(raw: dict[str, Any], today: Optional[date] = None, cities: tuple[CityConfig, ...] = DEFAULT_CITIES) -> Optional[WeatherMarket]:
    question = str(raw.get("question") or raw.get("title") or "")
    description = str(raw.get("description") or raw.get("rules") or raw.get("resolutionSource") or "")
    event_context = " ".join(
        str(raw.get(key) or "")
        for key in ("slug", "eventTitle", "eventSubtitle", "eventDescription", "eventSlug")
    )
    combined = f"{question}\n{description}\n{event_context}"
    city = find_city(combined, cities)
    target_date = parse_market_date(combined, today=today)

    outcomes = parse_jsonish_list(raw.get("outcomes"))
    token_ids = parse_jsonish_list(raw.get("clobTokenIds") or raw.get("clob_token_ids"))
    prices = parse_jsonish_list(raw.get("outcomePrices") or raw.get("outcome_prices"))
    buckets = parse_buckets(outcomes, token_ids=token_ids, prices=prices)
    if not buckets:
        buckets = parse_binary_temperature_bucket(question, outcomes, token_ids=token_ids, prices=prices)
    if city is None or target_date is None or not buckets:
        return None

    return WeatherMarket(
        id=str(raw.get("id") or raw.get("conditionId") or raw.get("condition_id") or raw.get("slug") or ""),
        question=question,
        slug=str(raw.get("slug") or ""),
        event_slug=raw.get("eventSlug") or raw.get("event_slug"),
        close_time=parse_close_time(raw.get("endDate") or raw.get("end_date") or raw.get("closeTime")),
        target_date=target_date,
        city=city,
        resolution_rules=description,
        buckets=buckets,
        raw=raw,
    )


def looks_like_temperature_market(raw: dict[str, Any]) -> bool:
    text = " ".join(str(raw.get(key) or "") for key in ("question", "title", "description", "slug", "eventSlug", "eventTitle", "eventDescription"))
    lowered = text.lower()
    if "lowest temperature" in lowered or "lowest temp" in lowered or "low temperature" in lowered:
        return False
    return "highest temperature" in lowered or "highest temp" in lowered or "high temp" in lowered


def _detect_unit(label: str) -> str:
    if re.search(r"°?\s*c\b", label, re.I):
        return "C"
    return "F"


def _to_fahrenheit(value: float, unit: str) -> float:
    if unit.upper() == "C":
        return value * 9 / 5 + 32
    return value
