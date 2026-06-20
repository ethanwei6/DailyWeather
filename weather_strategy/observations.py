from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
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
    AVIATION_WEATHER_URL = "https://aviationweather.gov/api/data/metar"
    IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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
        metar_station = city.metar_station or city.nws_station
        if metar_station:
            if target_date >= local_now.date() - timedelta(days=2):
                try:
                    metar = self._fetch_metar_station_high(city, metar_station, target_date, local_now)
                except RuntimeError:
                    metar = None
                if metar is not None:
                    return metar
            if target_date < local_now.date():
                try:
                    historical_metar = self._fetch_historical_metar_station_high(city, metar_station, target_date, local_now)
                except RuntimeError:
                    historical_metar = None
                if historical_metar is not None:
                    return historical_metar
        if target_date < local_now.date():
            try:
                archive = self._fetch_open_meteo_archive_high(city, target_date, local_now)
            except RuntimeError:
                archive = None
            if archive is not None:
                return archive
        try:
            return self._fetch_open_meteo_proxy_high(city, target_date, local_now)
        except RuntimeError:
            return None

    def fetch_historical_station_high(
        self,
        city: CityConfig,
        station_id: str,
        target_date: date,
        now: Optional[datetime] = None,
    ) -> Optional[ObservedHigh]:
        local_now = _local_now(city, now)
        if target_date >= local_now.date():
            try:
                return self._fetch_metar_station_high(city, station_id, target_date, local_now)
            except RuntimeError:
                return None
        try:
            return self._fetch_historical_metar_station_high(city, station_id, target_date, local_now)
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

    def _fetch_metar_station_high(self, city: CityConfig, station_id: str, target_date: date, local_now: datetime) -> Optional[ObservedHigh]:
        payload = self.http.get_json(
            self.AVIATION_WEATHER_URL,
            params={"ids": station_id, "format": "json", "hours": 48},
        )
        records = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
        temperatures: list[tuple[float, datetime]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            observed_at = _parse_metar_time(record)
            temperature_f = _parse_metar_temperature_f(record)
            if observed_at is None or temperature_f is None:
                continue
            observed_local = observed_at.astimezone(local_now.tzinfo)
            if observed_local.date() != target_date or observed_local > local_now:
                continue
            temperatures.append((temperature_f, observed_at))
        if not temperatures:
            return None
        max_temperature, observed_at = max(temperatures, key=lambda item: item[0])
        return ObservedHigh(
            city=city,
            target_date=target_date,
            max_temperature_f=max_temperature,
            source=f"metar_{station_id}",
            observed_at=observed_at,
            sample_count=len(temperatures),
            is_actual=True,
            is_final=target_date < local_now.date(),
        )

    def _fetch_historical_metar_station_high(self, city: CityConfig, station_id: str, target_date: date, local_now: datetime) -> Optional[ObservedHigh]:
        end_date = target_date + timedelta(days=1)
        payload = self.http.get_text(
            self.IEM_ASOS_URL,
            params={
                "station": station_id,
                "data": "tmpf",
                "year1": target_date.year,
                "month1": target_date.month,
                "day1": target_date.day,
                "year2": end_date.year,
                "month2": end_date.month,
                "day2": end_date.day,
                "tz": city.timezone,
                "format": "onlycomma",
                "latlon": "no",
                "elev": "no",
                "missing": "M",
                "trace": "T",
                "direct": "no",
                "report_type": ("1", "2"),
            },
            headers={"Accept": "text/csv,text/plain,*/*"},
        )
        timezone_info = _zoneinfo(city)
        temperatures: list[tuple[float, datetime]] = []
        for row in csv.DictReader(io.StringIO(payload)):
            valid = row.get("valid")
            value = row.get("tmpf")
            if not valid or value in (None, "", "M"):
                continue
            try:
                observed_local = datetime.fromisoformat(valid).replace(tzinfo=timezone_info)
                temperature_f = float(value)
            except ValueError:
                continue
            if observed_local.date() != target_date or observed_local > local_now:
                continue
            temperatures.append((temperature_f, observed_local.astimezone(timezone.utc)))
        if not temperatures:
            return None
        max_temperature, observed_at = max(temperatures, key=lambda item: item[0])
        return ObservedHigh(
            city=city,
            target_date=target_date,
            max_temperature_f=max_temperature,
            source=f"historical_metar_{station_id}",
            observed_at=observed_at,
            sample_count=len(temperatures),
            is_actual=True,
            is_final=target_date < local_now.date(),
        )

    def _fetch_open_meteo_archive_high(self, city: CityConfig, target_date: date, local_now: datetime) -> Optional[ObservedHigh]:
        payload = self.http.get_json(
            self.OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": city.timezone,
            },
        )
        daily = payload.get("daily") or {}
        times = daily.get("time") or []
        values = daily.get("temperature_2m_max") or []
        for index, value in enumerate(values):
            if index >= len(times) or value is None:
                continue
            try:
                observed_date = date.fromisoformat(str(times[index]))
            except ValueError:
                continue
            if observed_date != target_date:
                continue
            return ObservedHigh(
                city=city,
                target_date=target_date,
                max_temperature_f=float(value),
                source="open_meteo_archive_daily",
                observed_at=datetime.combine(target_date, time.max, tzinfo=local_now.tzinfo),
                sample_count=1,
                is_actual=False,
                is_final=True,
            )
        return None

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


def _parse_metar_time(record: dict[str, Any]) -> Optional[datetime]:
    for key in ("obsTime", "reportTime", "receiptTime"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _parse_metar_temperature_f(record: dict[str, Any]) -> Optional[float]:
    for key in ("temp", "temp_c", "temperature"):
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value) * 9 / 5 + 32
        except (TypeError, ValueError):
            continue
    return None


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
