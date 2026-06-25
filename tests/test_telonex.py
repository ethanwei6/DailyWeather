from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from weather_strategy.telonex import (
    TelonexClient,
    TelonexConfigurationError,
    _is_missing_telonex_file,
    _quote_records_to_price_history,
    load_local_env,
)


class TelonexTest(unittest.TestCase):
    def test_load_local_env_sets_missing_values_without_overriding_existing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("TELONEX_API_KEY=local-secret\nexport TELONEX_BASE_URL='https://example.test/v1'\n", encoding="utf-8")
            old_key = os.environ.pop("TELONEX_API_KEY", None)
            old_base = os.environ.get("TELONEX_BASE_URL")
            os.environ["TELONEX_BASE_URL"] = "already-set"
            try:
                load_local_env(env_path)

                self.assertEqual(os.environ["TELONEX_API_KEY"], "local-secret")
                self.assertEqual(os.environ["TELONEX_BASE_URL"], "already-set")
            finally:
                if old_key is None:
                    os.environ.pop("TELONEX_API_KEY", None)
                else:
                    os.environ["TELONEX_API_KEY"] = old_key
                if old_base is None:
                    os.environ.pop("TELONEX_BASE_URL", None)
                else:
                    os.environ["TELONEX_BASE_URL"] = old_base

    def test_client_requires_key(self) -> None:
        old_key = os.environ.pop("TELONEX_API_KEY", None)
        try:
            with self.assertRaises(TelonexConfigurationError):
                TelonexClient(env_path="/tmp/does-not-exist")
        finally:
            if old_key is not None:
                os.environ["TELONEX_API_KEY"] = old_key

    def test_download_parquet_uses_bearer_auth_and_local_cache(self) -> None:
        class FakeHttp:
            def __init__(self) -> None:
                self.calls = []

            def get_redirect_location(self, url, params=None, headers=None):
                self.calls.append({"kind": "redirect", "url": url, "params": params, "headers": headers})
                return "https://s3.example.test/file.parquet?signature=abc"

            def get_bytes(self, url, params=None, headers=None):
                self.calls.append({"kind": "bytes", "url": url, "params": params, "headers": headers})
                return b"PAR1fake-parquet"

        fake_http = FakeHttp()
        with tempfile.TemporaryDirectory() as directory:
            client = TelonexClient(
                api_key="secret-key",
                base_url="https://api.telonex.test/v1",
                cache_dir=directory,
                http=fake_http,
                env_path=Path(directory) / ".env-missing",
            )

            first = client.download_parquet(
                exchange="polymarket",
                channel="quotes",
                day=date(2026, 1, 20),
                params={"slug": "market-slug", "outcome": "Yes"},
            )
            second = client.download_parquet(
                exchange="polymarket",
                channel="quotes",
                day=date(2026, 1, 20),
                params={"slug": "market-slug", "outcome": "Yes"},
            )

        self.assertEqual(first, second)
        self.assertEqual(len(fake_http.calls), 2)
        self.assertEqual(fake_http.calls[0]["kind"], "redirect")
        self.assertEqual(fake_http.calls[0]["url"], "https://api.telonex.test/v1/downloads/polymarket/quotes/2026-01-20")
        self.assertEqual(fake_http.calls[0]["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(fake_http.calls[1]["kind"], "bytes")
        self.assertEqual(fake_http.calls[1]["url"], "https://s3.example.test/file.parquet?signature=abc")
        self.assertNotIn("Authorization", fake_http.calls[1]["headers"])
        self.assertNotIn("secret-key", str(first))

    def test_missing_download_is_negative_cached(self) -> None:
        class FakeHttp:
            def __init__(self) -> None:
                self.calls = 0

            def get_redirect_location(self, url, params=None, headers=None):
                self.calls += 1
                raise RuntimeError('HTTP 404 for https://api.telonex.test: {"detail":"File not found: polymarket/quotes"}')

        fake_http = FakeHttp()
        with tempfile.TemporaryDirectory() as directory:
            client = TelonexClient(
                api_key="secret-key",
                base_url="https://api.telonex.test/v1",
                cache_dir=directory,
                http=fake_http,
                env_path=Path(directory) / ".env-missing",
            )
            kwargs = {
                "exchange": "polymarket",
                "channel": "quotes",
                "day": date(2026, 1, 20),
                "params": {"asset_id": "123"},
            }

            with self.assertRaises(RuntimeError):
                client.download_parquet(**kwargs)
            with self.assertRaises(RuntimeError) as second_error:
                client.download_parquet(**kwargs)

        self.assertEqual(fake_http.calls, 1)
        self.assertIn("cached missing file", str(second_error.exception))

    def test_download_parquet_hard_timeout_interrupts_stalled_http(self) -> None:
        class SlowHttp:
            def get_redirect_location(self, url, params=None, headers=None):
                time.sleep(0.2)
                return "https://s3.example.test/file.parquet?signature=abc"

            def get_bytes(self, url, params=None, headers=None):
                return b"PAR1fake-parquet"

        with tempfile.TemporaryDirectory() as directory:
            client = TelonexClient(
                api_key="secret-key",
                base_url="https://api.telonex.test/v1",
                cache_dir=directory,
                http=SlowHttp(),
                env_path=Path(directory) / ".env-missing",
                hard_timeout_seconds=0.01,
            )

            with self.assertRaises(RuntimeError) as error:
                client.download_parquet(
                    exchange="polymarket",
                    channel="quotes",
                    day=date(2026, 1, 20),
                    params={"asset_id": "123"},
                )

        self.assertIn("hard-timeout", str(error.exception))

    def test_quote_records_to_price_history_uses_buy_ask_and_filters(self) -> None:
        records = [
            {"timestamp": "2026-01-20T00:00:00Z", "slug": "target", "outcome": "Yes", "best_ask_price": 0.44},
            {"timestamp_ms": 1768869000000, "slug": "target", "outcome": "Yes", "bid_price": 0.40, "ask_price": 0.42},
            {"timestamp": "2026-01-20T00:10:00Z", "slug": "target", "outcome": "No", "best_ask_price": 0.10},
            {"timestamp": "2026-01-19T23:59:00Z", "slug": "target", "outcome": "Yes", "best_ask_price": 0.30},
            {"timestamp": "2026-01-20T00:15:00Z", "slug": "other", "outcome": "Yes", "best_ask_price": 0.31},
            {"timestamp": "2026-01-20T00:20:00Z", "slug": "target", "outcome": "Yes", "ask_size": 10, "best_bid": 0.51},
        ]

        points = _quote_records_to_price_history(
            records,
            start=datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 1, 20, 0, 30, tzinfo=timezone.utc),
            slug="target",
            outcome="Yes",
        )

        self.assertEqual([point.timestamp for point in points], sorted(point.timestamp for point in points))
        self.assertEqual([round(point.price, 2) for point in points], [0.44, 0.51, 0.42])

    def test_missing_daily_file_detection_is_narrow(self) -> None:
        self.assertTrue(_is_missing_telonex_file(RuntimeError('HTTP 404 for url: {"detail":"File not found: polymarket/quotes"}')))
        self.assertTrue(_is_missing_telonex_file(RuntimeError("Telonex cached missing file for polymarket/quotes/2026-01-20")))
        self.assertFalse(_is_missing_telonex_file(RuntimeError("HTTP 401 for url: Unauthorized")))


if __name__ == "__main__":
    unittest.main()
