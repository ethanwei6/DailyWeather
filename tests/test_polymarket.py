from __future__ import annotations

import unittest

from weather_strategy.polymarket import PolymarketGammaClient, _temperature_search_queries


class PolymarketDiscoveryTest(unittest.TestCase):
    def test_temperature_search_queries_include_date_specific_queries(self) -> None:
        queries = _temperature_search_queries()
        self.assertGreaterEqual(len(queries), 5)
        self.assertTrue(any("highest temperature" in query for query in queries))

    def test_discovery_caps_request_size_and_survives_search_failure(self) -> None:
        client = PolymarketGammaClient()
        event_calls = []
        search_calls = []

        def fake_fetch_active_events(limit: int = 100, offset: int = 0, order: str = "volume_24hr"):
            event_calls.append((limit, offset))
            return []

        def fake_search_temperature_markets(query: str, limit: int = 50, pages: int = 1):
            search_calls.append((query, limit, pages))
            raise RuntimeError("timeout")

        client.fetch_active_events = fake_fetch_active_events  # type: ignore[method-assign]
        client.search_temperature_markets = fake_search_temperature_markets  # type: ignore[method-assign]

        markets = client.discover_temperature_markets(limit=150, pages=1, request_limit=200)
        self.assertEqual(markets, [])
        self.assertEqual(event_calls[0][0], 50)
        self.assertTrue(search_calls)
        self.assertTrue(all(call[1] == 50 for call in search_calls))


if __name__ == "__main__":
    unittest.main()
