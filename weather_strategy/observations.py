from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weather_strategy.http import HttpClient
from weather_strategy.models import CityConfig, TemperatureBucket


@dataclass(frozen=True)
class ObservedHigh:
    city: CityConfig
    target_date: date
    max_temperature_f: float
    source: str
    observed_at: datetime
    sample_count: int
    is_actual: bool
    is_final: bool


class ObservedHighClient:
    NWS_URL = "https://api.weather.gov"
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, http: Optional[HttpClient] = None):
        self.http = http or HttpClient(user_agent="weather-polymarket-strategy/0.1 contact=local")

    def fetch_observed_high(self, city: CityConfig, target_date: date, now: Optional[datetime] = None) -> Optional[ObservedHigh]:
        local_now = _local_now(city, now)
        if target_date > local_now.date():
            return None
        if city.nws_station:
            try:
                nws = self._fetch_nws_station_high(city, target_date, local_now)
            except RuntimeError:
                nws = None
            if nws is not None:
                return nws
        try:
            return self._fetch_open_meteo_proxy_high(city, target_date, local_now)
        except RuntimeError:
            return None

    def _fetch_nws_station_high(self, city: CityConfig, target_date: date, local_now: datetime) -> Optional[ObservedHigh]:
        start, end = _target_window(city, target_date, local_now)
        payload = self.http.get_json(
            f"{self.NWS_URL}/stations/{city.nws_station}/observations",
            params={"start": start.astimezone(timezone.utc).isoformat(), "end": end.astimezone(timezone.utc).isoformat(), "limit": 500},
        )
        temperatures: list[tuple[float, datetime]] = []
        for feature in payload.get("features") or []:
            properties = feature.get("properties") or {}
            temperature = (properties.get("temperature") or {}).get("value")
            timestamp = properties.get("timestamp")
            if temperature is None or timestamp is None:
                continue
            try:
                observed_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                temperatures.append((float(temperature) * 9 / 5 + 32, observed_at))
            except ValueError:
                continue
        if not temperatures:
            return None
        max_temperature, observed_at = max(temperatures, key=lambda item: item[0])
        return ObservedHigh(
            city=city,
            target_date=target_date,
            max_temperature_f=max_temperature,
            source=f"nws_station_{city.nws_station}",
            observed_at=observed_at,
            sample_count=len(temperatures),
            is_actual=True,
            is_final=target_date < local_now.date(),
        )

    def _fetch_open_meteo_proxy_high(self, city: CityConfig, target_date: date, local_now: datetime) -> Optional[ObservedHigh]:
        payload = self.http.get_json(
            self.OPEN_METEO_URL,
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "hourly": "temperature_2m",
                "temperature_unit": "fahrenheit",
                "timezone": city.timezone,
                "past_days": 2,
                "forecast_days": 2,
            },
        )
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        values = hourly.get("temperature_2m") or []
        temperatures: list[tuple[float, datetime]] = []
        for index, value in enumerate(values):
            if index >= len(times) or value is None:
                continue
            try:
                observed_local = datetime.fromisoformat(str(times[index])).replace(tzinfo=local_now.tzinfo)
            except ValueError:
                continue
            if observed_local.date() != target_date or observed_local > local_now:
                continue
            temperatures.append((float(value), observed_local))
        if not temperatures:
            return None
        max_temperature, observed_at = max(temperatures, key=lambda item: item[0])
        return ObservedHigh(
            city=city,
            target_date=target_date,
            max_temperature_f=max_temperature,
            source="open_meteo_hourly_proxy",
            observed_at=observed_at,
            sample_count=len(temperatures),
            is_actual=False,
            is_final=target_date < local_now.date(),
        )


def observed_outcome_for_bucket(bucket: TemperatureBucket, observed_high_f: Optional[float], is_final: bool) -> Optional[int]:
    if observed_high_f is None or not is_final:
        return None
    return 1 if bucket.contains(observed_high_f) else 0


def _target_window(city: CityConfig, target_date: date, local_now: datetime) -> tuple[datetime, datetime]:
    timezone_info = _zoneinfo(city)
    start = datetime.combine(target_date, time.min, tzinfo=timezone_info)
    if target_date < local_now.date():
        end = datetime.combine(target_date, time.max, tzinfo=timezone_info)
    else:
        end = local_now
    return start, end


def _local_now(city: CityConfig, now: Optional[datetime]) -> datetime:
    timezone_info = _zoneinfo(city)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone_info)


def _zoneinfo(city: CityConfig) -> ZoneInfo:
    try:
        return ZoneInfo(city.timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone for {city.display_name}: {city.timezone}") from error
