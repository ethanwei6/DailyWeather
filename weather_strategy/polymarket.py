from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from weather_strategy.http import HttpClient
from weather_strategy.models import OrderBookQuote
from weather_strategy.parser import looks_like_temperature_market, parse_weather_market


class PolymarketGammaClient:
    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, http: Optional[HttpClient] = None):
        self.http = http or HttpClient()

    def fetch_active_events(self, limit: int = 100, offset: int = 0, order: str = "volume_24hr") -> list[dict[str, Any]]:
        payload = self.http.get_json(
            f"{self.BASE_URL}/events",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
                "order": order,
                "ascending": "false",
            },
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            return payload["events"]
        raise ValueError("Unexpected Gamma events response shape")

    def discover_temperature_markets(self, limit: int = 100, pages: int = 1, request_limit: int = 50) -> list[Any]:
        parsed = []
        request_limit = max(1, min(request_limit, 50))
        for page in range(pages):
            try:
                events = self.fetch_active_events(limit=request_limit, offset=page * request_limit)
            except RuntimeError:
                events = []
            for raw_market in _iter_event_markets(events):
                if raw_market.get("closed") is True:
                    continue
                if not looks_like_temperature_market(raw_market):
                    continue
                market = parse_weather_market(raw_market)
                if market is not None:
                    parsed.append(market)
        for query in _temperature_search_queries():
            try:
                parsed.extend(self.search_temperature_markets(query=query, limit=request_limit, pages=pages))
            except RuntimeError:
                continue
        return _dedupe_markets(parsed)

    def public_search(self, query: str, limit: int = 50, page: int = 1) -> dict[str, Any]:
        payload = self.http.get_json(
            f"{self.BASE_URL}/public-search",
            params={
                "q": query,
                "limit_per_type": limit,
                "page": page,
                "search_profiles": "false",
                "search_tags": "false",
            },
        )
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Gamma public-search response shape")
        return payload

    def search_temperature_markets(self, query: str = "highest temperature", limit: int = 50, pages: int = 1) -> list[Any]:
        parsed = []
        for page in range(1, pages + 1):
            payload = self.public_search(query=query, limit=limit, page=page)
            for raw_market in _iter_event_markets(payload.get("events") or []):
                if raw_market.get("closed") is True:
                    continue
                if not looks_like_temperature_market(raw_market):
                    continue
                market = parse_weather_market(raw_market)
                if market is not None:
                    parsed.append(market)
        return _dedupe_markets(parsed)


class PolymarketClobClient:
    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, http: Optional[HttpClient] = None):
        self.http = http or HttpClient()

    def fetch_order_book_quote(self, token_id: str) -> OrderBookQuote:
        payload = self.http.get_json(f"{self.BASE_URL}/book", params={"token_id": token_id})
        return parse_orderbook_quote(token_id, payload)


def parse_orderbook_quote(token_id: str, payload: dict[str, Any]) -> OrderBookQuote:
    bids = _parse_levels(payload.get("bids") or [])
    asks = _parse_levels(payload.get("asks") or [])
    best_bid = max(bids, key=lambda level: level[0]) if bids else None
    best_ask = min(asks, key=lambda level: level[0]) if asks else None
    return OrderBookQuote(
        token_id=token_id,
        best_bid=best_bid[0] if best_bid else None,
        best_ask=best_ask[0] if best_ask else None,
        bid_size=best_bid[1] if best_bid else None,
        ask_size=best_ask[1] if best_ask else None,
    )


def _parse_levels(levels: Iterable[Any]) -> list[tuple[float, float]]:
    parsed = []
    for level in levels:
        if isinstance(level, dict):
            price = level.get("price")
            size = level.get("size")
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price, size = level[0], level[1]
        else:
            continue
        try:
            parsed.append((float(price), float(size)))
        except (TypeError, ValueError):
            continue
    return parsed


def _iter_event_markets(events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for event in events:
        markets = event.get("markets")
        if isinstance(markets, list):
            for market in markets:
                if isinstance(market, dict):
                    merged = dict(market)
                    merged.setdefault("eventSlug", event.get("slug"))
                    merged.setdefault("eventTitle", event.get("title"))
                    merged.setdefault("eventSubtitle", event.get("subtitle"))
                    merged.setdefault("eventDescription", event.get("description"))
                    yield merged
        elif isinstance(event, dict):
            yield event


def _dedupe_markets(markets: Iterable[Any]) -> list[Any]:
    seen = set()
    deduped = []
    for market in markets:
        key = market.id or market.slug or market.question
        if key in seen:
            continue
        seen.add(key)
        deduped.append(market)
    return deduped


def _temperature_search_queries() -> tuple[str, ...]:
    today = datetime.now(timezone.utc).date()
    dates = (today, today + timedelta(days=1), today + timedelta(days=2))
    date_queries = tuple(f"{_month_day(value)} highest temperature" for value in dates)
    return (*date_queries, "highest temperature", "temperature weather")


def _month_day(value) -> str:
    return f"{value.strftime('%B')} {value.day}"
