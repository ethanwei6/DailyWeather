from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from weather_strategy.live_setup import live_status, load_env_file, write_env_updates


class LiveSetupTest(unittest.TestCase):
    def test_write_env_updates_preserves_existing_values_unless_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env.local"
            env_path.write_text("PRIVATE_KEY=old\nBOT_OWNER_ADDRESS=0xabc\n")

            write_env_updates(
                env_path,
                {
                    "PRIVATE_KEY": "new",
                    "CLOB_API_URL": "https://clob.polymarket.com",
                },
                overwrite=False,
            )

            values = load_env_file(env_path)
            self.assertEqual(values["PRIVATE_KEY"], "old")
            self.assertEqual(values["BOT_OWNER_ADDRESS"], "0xabc")
            self.assertEqual(values["CLOB_API_URL"], "https://clob.polymarket.com")
            self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o600)

            write_env_updates(env_path, {"PRIVATE_KEY": "new"}, overwrite=True)
            self.assertEqual(load_env_file(env_path)["PRIVATE_KEY"], "new")

    def test_live_status_reports_secret_presence_without_exposing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env.local"
            write_env_updates(
                env_path,
                {
                    "POLYMARKET_VENUE": "international",
                    "CHAIN_ID": "137",
                    "CLOB_API_URL": "https://clob.polymarket.com",
                    "PRIVATE_KEY": "0xsecret",
                    "CLOB_API_KEY": "key",
                    "CLOB_SECRET": "secret",
                    "CLOB_PASS_PHRASE": "passphrase",
                    "DAILYWEATHER_LIVE_TRADING": "0",
                    "DAILYWEATHER_MAX_BANKROLL_USD": "100",
                },
            )

            status = live_status(str(env_path))

            self.assertTrue(status["has_private_key"])
            self.assertTrue(status["has_clob_credentials"])
            self.assertFalse(status["live_trading_enabled"])
            self.assertNotIn("0xsecret", repr(status))
            self.assertEqual(status["max_bankroll_usd"], "100")
