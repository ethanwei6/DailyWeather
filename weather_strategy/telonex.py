from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from weather_strategy.http import HttpClient
from weather_strategy.polymarket import PriceHistoryPoint


DEFAULT_BASE_URL = "https://api.telonex.io/v1"


class TelonexConfigurationError(RuntimeError):
    pass


class TelonexDataError(RuntimeError):
    pass


class TelonexClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: str | Path = "work/cache/telonex",
        http: Optional[HttpClient] = None,
        env_path: str | Path = ".env",
    ) -> None:
        load_local_env(env_path)
        self.api_key = api_key or os.environ.get("TELONEX_API_KEY")
        if not self.api_key:
            raise TelonexConfigurationError(
                "TELONEX_API_KEY is required for Telonex historical pricing. "
                "Put it in a local ignored .env file or export it in the shell."
            )
        self.base_url = (base_url or os.environ.get("TELONEX_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = http or HttpClient(timeout_seconds=30)
        self.download_hits = 0
        self.download_misses = 0

    def availability(self, exchange: str = "polymarket", params: Optional[Mapping[str, Any]] = None) -> Any:
        return self.http.get_json(
            f"{self.base_url}/availability/{exchange}",
            params=params,
        )

    def dataset(self, exchange: str, dataset: str) -> Any:
        return self.http.get_json(f"{self.base_url}/datasets/{exchange}/{dataset}")

    def download_dataset_parquet(self, *, exchange: str, dataset: str) -> Path:
        cache_path = self.cache_dir / exchange / "datasets" / f"{dataset}.parquet"
        if cache_path.exists():
            self.download_hits += 1
            return cache_path
        self.download_misses += 1
        endpoint = f"{self.base_url}/datasets/{exchange}/{dataset}"
        redirect_url = self.http.get_redirect_location(endpoint, headers=self._headers(accept="*/*"))
        payload = self.http.get_bytes(redirect_url or endpoint, headers={"Accept": "*/*"})
        if not _looks_like_parquet(payload):
            preview = payload[:250].decode("utf-8", errors="replace")
            raise TelonexDataError(f"Telonex dataset did not return Parquet for {exchange}/{dataset}: {preview}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
        return cache_path

    def download_parquet(
        self,
        *,
        exchange: str,
        channel: str,
        day: date,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        cache_path = self._cache_path(exchange, channel, day, params or {})
        missing_path = _missing_cache_path(cache_path)
        if cache_path.exists():
            self.download_hits += 1
            return cache_path
        if missing_path.exists():
            self.download_hits += 1
            raise RuntimeError(f"Telonex cached missing file for {exchange}/{channel}/{day.isoformat()}")
        self.download_misses += 1
        endpoint = f"{self.base_url}/downloads/{exchange}/{channel}/{day.isoformat()}"
        try:
            redirect_url = self.http.get_redirect_location(
                endpoint,
                params=params,
                headers=self._headers(accept="*/*"),
            )
            payload = self.http.get_bytes(redirect_url or endpoint, params=None if redirect_url else params, headers={"Accept": "*/*"})
        except RuntimeError as error:
            if _is_missing_telonex_file(error):
                missing_path.parent.mkdir(parents=True, exist_ok=True)
                missing_path.write_text(
                    json.dumps(
                        {
                            "exchange": exchange,
                            "channel": channel,
                            "day": day.isoformat(),
                            "params": dict(params or {}),
                            "error": str(error),
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            raise
        if not _looks_like_parquet(payload):
            preview = payload[:250].decode("utf-8", errors="replace")
            raise TelonexDataError(f"Telonex download did not return Parquet for {exchange}/{channel}/{day}: {preview}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
        return cache_path

    def fetch_quote_price_history(
        self,
        *,
        slug: str,
        outcome: str,
        start_ts: Optional[int],
        end_ts: Optional[int],
        token_id: Optional[str] = None,
    ) -> list[PriceHistoryPoint]:
        if not slug:
            raise TelonexDataError("Telonex quote history requires a Polymarket market slug")
        if start_ts is None or end_ts is None:
            raise TelonexDataError("Telonex quote history requires start_ts and end_ts bounds")
        start = datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
        end = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
        if end < start:
            raise TelonexDataError("Telonex quote history end_ts is before start_ts")

        points: list[PriceHistoryPoint] = []
        identifier_params = {"asset_id": token_id} if token_id else {"slug": slug, "outcome": outcome}
        local_slug = None if token_id else slug
        local_outcome = None if token_id else outcome
        for day in _date_range(start.date(), end.date()):
            try:
                path = self.download_parquet(
                    exchange="polymarket",
                    channel="quotes",
                    day=day,
                    params=identifier_params,
                )
            except RuntimeError as error:
                if _is_missing_telonex_file(error):
                    continue
                raise
            records = _load_parquet_records(path)
            points.extend(
                _quote_records_to_price_history(
                    records,
                    start=start,
                    end=end,
                    slug=local_slug,
                    outcome=local_outcome,
                    token_id=token_id,
                )
            )
        return _dedupe_price_history(points)

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": accept}

    def _cache_path(self, exchange: str, channel: str, day: date, params: Mapping[str, Any]) -> Path:
        payload = {"exchange": exchange, "channel": channel, "day": day.isoformat(), "params": dict(params)}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / exchange / channel / f"{day.isoformat()}-{key}.parquet"


def load_local_env(path: str | Path = ".env") -> None:
    env_file = Path(path)
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _clean_env_value(value)


def _clean_env_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ("'", '"'):
        return stripped[1:-1]
    return stripped


def _load_parquet_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        import polars as pl  # type: ignore

        return [dict(row) for row in pl.read_parquet(path).to_dicts()]
    except ModuleNotFoundError:
        pass
    try:
        import pandas as pd  # type: ignore

        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    except ModuleNotFoundError:
        pass
    try:
        import pyarrow.parquet as pq  # type: ignore

        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except ModuleNotFoundError:
        pass
    raise TelonexConfigurationError(
        "Reading Telonex Parquet requires one of polars, pandas+pyarrow, or pyarrow. "
        "Install the optional telonex dependencies before running Telonex backtests."
    )


def _quote_records_to_price_history(
    records: Iterable[Mapping[str, Any]],
    *,
    start: datetime,
    end: datetime,
    slug: Optional[str] = None,
    outcome: Optional[str] = None,
    token_id: Optional[str] = None,
) -> list[PriceHistoryPoint]:
    start_utc = _as_utc(start)
    end_utc = _as_utc(end)
    points: list[PriceHistoryPoint] = []
    for record in records:
        if not _matches_if_field_exists(record, slug, ("slug", "market_slug", "marketSlug")):
            continue
        if not _matches_if_field_exists(record, outcome, ("outcome", "outcome_name", "outcomeName")):
            continue
        if not _matches_if_field_exists(record, token_id, ("token_id", "tokenId", "clobTokenId", "asset_id", "assetId")):
            continue
        timestamp = _record_timestamp(record)
        if timestamp is None:
            continue
        timestamp = _as_utc(timestamp)
        if timestamp < start_utc or timestamp > end_utc:
            continue
        price = _record_buy_price(record)
        if price is None or not 0.0 <= price <= 1.0:
            continue
        points.append(PriceHistoryPoint(timestamp=timestamp, price=price))
    return _dedupe_price_history(points)


def _matches_if_field_exists(record: Mapping[str, Any], expected: Optional[str], keys: tuple[str, ...]) -> bool:
    if expected is None:
        return True
    value = _first_present(record, keys)
    if value is None:
        return True
    return str(value).strip().lower() == str(expected).strip().lower()


def _record_timestamp(record: Mapping[str, Any]) -> Optional[datetime]:
    for key in ("timestamp", "ts", "time", "datetime", "created_at", "createdAt", "event_time", "eventTime"):
        value = _field(record, key)
        parsed = _parse_timestamp(value, key)
        if parsed is not None:
            return parsed
    for key, value in record.items():
        lower = str(key).lower()
        if "timestamp" in lower or lower.endswith("_time"):
            parsed = _parse_timestamp(value, str(key))
            if parsed is not None:
                return parsed
    return None


def _record_buy_price(record: Mapping[str, Any]) -> Optional[float]:
    ask = _first_numeric_by_name(record, ("best_ask_price", "ask_price", "bestAskPrice", "best_ask", "bestAsk", "ask"))
    if ask is not None:
        return ask
    bid = _first_numeric_by_name(record, ("best_bid_price", "bid_price", "bestBidPrice", "best_bid", "bestBid", "bid"))
    mid = _first_numeric_by_name(record, ("mid_price", "midPrice", "midpoint", "mid"))
    if mid is not None:
        return mid
    if bid is not None:
        return bid
    return _first_numeric_by_name(record, ("price", "last_price", "lastPrice"))


def _first_numeric_by_name(record: Mapping[str, Any], names: tuple[str, ...]) -> Optional[float]:
    for name in names:
        value = _field(record, name)
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    dynamic = _dynamic_numeric_field(record, names)
    return dynamic


def _dynamic_numeric_field(record: Mapping[str, Any], names: tuple[str, ...]) -> Optional[float]:
    wants_ask = any("ask" in name.lower() for name in names)
    wants_bid = any("bid" in name.lower() for name in names)
    wants_price = any("price" in name.lower() for name in names)
    for key, value in record.items():
        lower = str(key).lower()
        if wants_ask and "ask" in lower and ("price" in lower or lower.endswith("ask")) and "size" not in lower:
            numeric = _to_float(value)
            if numeric is not None:
                return numeric
        if wants_bid and "bid" in lower and ("price" in lower or lower.endswith("bid")) and "size" not in lower:
            numeric = _to_float(value)
            if numeric is not None:
                return numeric
        if wants_price and lower in ("price", "last_price", "lastprice"):
            numeric = _to_float(value)
            if numeric is not None:
                return numeric
    return None


def _field(record: Mapping[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    lower = key.lower()
    for candidate, value in record.items():
        if str(candidate).lower() == lower:
            return value
    return None


def _first_present(record: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _field(record, key)
        if value is not None:
            return value
    return None


def _parse_timestamp(value: Any, key: str = "") -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        return converted if converted.tzinfo else converted.replace(tzinfo=timezone.utc)
    numeric = _to_float(value)
    if numeric is not None:
        lower = key.lower()
        try:
            if lower.endswith("_ns") or numeric > 1e17:
                return datetime.fromtimestamp(numeric / 1_000_000_000, tz=timezone.utc)
            if lower.endswith("_us") or numeric > 1e14:
                return datetime.fromtimestamp(numeric / 1_000_000, tz=timezone.utc)
            if lower.endswith("_ms") or numeric > 1e11:
                return datetime.fromtimestamp(numeric / 1_000, tz=timezone.utc)
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _dedupe_price_history(points: Iterable[PriceHistoryPoint]) -> list[PriceHistoryPoint]:
    by_timestamp: dict[datetime, PriceHistoryPoint] = {}
    for point in points:
        by_timestamp[_as_utc(point.timestamp)] = PriceHistoryPoint(timestamp=_as_utc(point.timestamp), price=point.price)
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def _looks_like_parquet(payload: bytes) -> bool:
    return len(payload) >= 4 and payload[:4] == b"PAR1"


def _missing_cache_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".missing.json")


def _is_missing_telonex_file(error: BaseException) -> bool:
    text = str(error)
    return ("HTTP 404" in text and "File not found" in text) or "Telonex cached missing file" in text
