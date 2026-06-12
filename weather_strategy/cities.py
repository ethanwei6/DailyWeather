from __future__ import annotations

import re

from weather_strategy.models import CityConfig


DEFAULT_CITIES: tuple[CityConfig, ...] = (
    CityConfig("New York", "NY", 40.7128, -74.0060, "America/New_York", "KNYC", ("NYC", "New York City"), metar_station="KNYC"),
    CityConfig("Los Angeles", "CA", 34.0522, -118.2437, "America/Los_Angeles", "KCQT", ("LA",), metar_station="KCQT"),
    CityConfig("Chicago", "IL", 41.8781, -87.6298, "America/Chicago", "KMDW", (), metar_station="KMDW"),
    CityConfig("Miami", "FL", 25.7617, -80.1918, "America/New_York", "KMIA", (), metar_station="KMIA"),
    CityConfig("Austin", "TX", 30.2672, -97.7431, "America/Chicago", "KAUS", (), metar_station="KAUS"),
    CityConfig("Dallas", "TX", 32.7767, -96.7970, "America/Chicago", "KDAL", (), metar_station="KDAL"),
    CityConfig("Houston", "TX", 29.7604, -95.3698, "America/Chicago", "KHOU", (), metar_station="KHOU"),
    CityConfig("Phoenix", "AZ", 33.4484, -112.0740, "America/Phoenix", "KPHX", (), metar_station="KPHX"),
    CityConfig("Philadelphia", "PA", 39.9526, -75.1652, "America/New_York", "KPHL", (), metar_station="KPHL"),
    CityConfig("Washington", "DC", 38.9072, -77.0369, "America/New_York", "KDCA", ("Washington DC", "D.C."), metar_station="KDCA"),
    CityConfig("Boston", "MA", 42.3601, -71.0589, "America/New_York", "KBOS", (), metar_station="KBOS"),
    CityConfig("San Francisco", "CA", 37.7749, -122.4194, "America/Los_Angeles", "KSFO", ("SF",), metar_station="KSFO"),
    CityConfig("Seattle", "WA", 47.6062, -122.3321, "America/Los_Angeles", "KSEA", (), metar_station="KSEA"),
    CityConfig("Denver", "CO", 39.7392, -104.9903, "America/Denver", "KDEN", (), metar_station="KDEN"),
    CityConfig("Las Vegas", "NV", 36.1699, -115.1398, "America/Los_Angeles", "KLAS", ("Vegas",), metar_station="KLAS"),
    CityConfig("Seoul", "KR", 37.5665, 126.9780, "Asia/Seoul", None, (), metar_station="RKSS"),
    CityConfig("Tokyo", "JP", 35.6762, 139.6503, "Asia/Tokyo", None, (), metar_station="RJTT"),
    CityConfig("Beijing", "CN", 39.9042, 116.4074, "Asia/Shanghai", None, (), metar_station="ZBAA"),
    CityConfig("Shanghai", "CN", 31.2304, 121.4737, "Asia/Shanghai", None, (), metar_station="ZSSS"),
    CityConfig("Guangzhou", "CN", 23.1291, 113.2644, "Asia/Shanghai", None, (), metar_station="ZGGG"),
    CityConfig("Shenzhen", "CN", 22.5431, 114.0579, "Asia/Shanghai", None, (), metar_station="ZGSZ"),
    CityConfig("Taipei", "TW", 25.0330, 121.5654, "Asia/Taipei", None, (), metar_station="RCSS"),
    CityConfig("Singapore", "SG", 1.3521, 103.8198, "Asia/Singapore", None, (), metar_station="WSSS"),
    CityConfig("Moscow", "RU", 55.7558, 37.6173, "Europe/Moscow", None, (), metar_station="UUEE"),
    CityConfig("Paris", "FR", 48.8566, 2.3522, "Europe/Paris", None, (), metar_station="LFPG"),
    CityConfig("London", "GB", 51.5072, -0.1276, "Europe/London", None, (), metar_station="EGLL"),
    CityConfig("Berlin", "DE", 52.5200, 13.4050, "Europe/Berlin", None, (), metar_station="EDDB"),
    CityConfig("Rome", "IT", 41.9028, 12.4964, "Europe/Rome", None, (), metar_station="LIRF"),
    CityConfig("Madrid", "ES", 40.4168, -3.7038, "Europe/Madrid", None, (), metar_station="LEMD"),
    CityConfig("Sydney", "AU", -33.8688, 151.2093, "Australia/Sydney", None, (), metar_station="YSSY"),
)


def find_city(text: str, cities: tuple[CityConfig, ...] = DEFAULT_CITIES):
    lowered = text.lower()
    for city in cities:
        candidates = (city.name, city.display_name, *city.aliases)
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if len(candidate_lower) <= 3:
                if re.search(rf"\b{re.escape(candidate_lower)}\b", lowered):
                    return city
            elif candidate_lower in lowered:
                return city
    return None
