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


STATION_COORDINATES: dict[str, tuple[float, float]] = {
    "KNYC": (40.7794, -73.9692),
    "KLGA": (40.7769, -73.8740),
    "KCQT": (34.0239, -118.2912),
    "KMDW": (41.7868, -87.7522),
    "KMIA": (25.7959, -80.2870),
    "KAUS": (30.1945, -97.6699),
    "KDAL": (32.8471, -96.8518),
    "KHOU": (29.6454, -95.2789),
    "KPHX": (33.4342, -112.0116),
    "KPHL": (39.8744, -75.2424),
    "KDCA": (38.8512, -77.0402),
    "KBOS": (42.3656, -71.0096),
    "KSFO": (37.6190, -122.3749),
    "KSEA": (47.4502, -122.3088),
    "KDEN": (39.8561, -104.6737),
    "KLAS": (36.0840, -115.1537),
    "RKSS": (37.5583, 126.7906),
    "RKSI": (37.4602, 126.4407),
    "RJTT": (35.5494, 139.7798),
    "ZBAA": (40.0801, 116.5846),
    "ZSSS": (31.1979, 121.3363),
    "ZGGG": (23.3924, 113.2990),
    "ZGSZ": (22.6393, 113.8107),
    "RCSS": (25.0697, 121.5525),
    "WSSS": (1.3644, 103.9915),
    "UUEE": (55.9726, 37.4146),
    "LFPG": (49.0097, 2.5479),
    "EGLL": (51.4700, -0.4543),
    "EDDB": (52.3667, 13.5033),
    "LIRF": (41.8003, 12.2389),
    "LEMD": (40.4983, -3.5676),
    "YSSY": (-33.9399, 151.1753),
}


def city_with_station_coordinates(city: CityConfig, station: str) -> CityConfig:
    station = station.upper()
    latitude, longitude = STATION_COORDINATES.get(station, (city.latitude, city.longitude))
    return CityConfig(
        name=city.name,
        state=city.state,
        latitude=latitude,
        longitude=longitude,
        timezone=city.timezone,
        nws_station=station if station.startswith("K") else city.nws_station,
        aliases=city.aliases,
        metar_station=station,
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
