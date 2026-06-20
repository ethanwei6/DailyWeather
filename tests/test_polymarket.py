from __future__ import annotations

import unittest

from weather_strategy.polymarket import PolymarketClobClient, PolymarketGammaClient, _temperature_search_queries


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

    def test_discovery_raises_when_every_gamma_request_fails(self) -> None:
        client = PolymarketGammaClient()

        def fake_fetch_active_events(limit: int = 100, offset: int = 0, order: str = "volume_24hr"):
            raise RuntimeError("dns failure")

        def fake_search_temperature_markets(query: str, limit: int = 50, pages: int = 1):
            raise RuntimeError("dns failure")

        client.fetch_active_events = fake_fetch_active_events  # type: ignore[method-assign]
        client.search_temperature_markets = fake_search_temperature_markets  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "discovery failed for all requests"):
            client.discover_temperature_markets(limit=150, pages=1, request_limit=50)

    def test_price_history_passes_timestamp_bounds(self) -> None:
        calls = []

        class FakeHttp:
            def get_json(self, url, params=None, headers=None):
                calls.append((url, params))
                return {"history": [{"t": 1775710856, "p": 0.26}]}

        history = PolymarketClobClient(http=FakeHttp()).fetch_price_history(
            "token-1",
            interval="max",
            fidelity=60,
            start_ts=1775692800,
            end_ts=1776124800,
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(calls[0][1]["market"], "token-1")
        self.assertEqual(calls[0][1]["startTs"], 1775692800)
        self.assertEqual(calls[0][1]["endTs"], 1776124800)


if __name__ == "__main__":
    unittest.main()
