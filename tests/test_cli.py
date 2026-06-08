from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

from weather_strategy.cli import main
from weather_strategy.cli import _should_process_market
from weather_strategy.paper import PaperLedger
from weather_strategy.parser import parse_weather_market
from weather_strategy.signals import SignalSettings


class CliTest(unittest.TestCase):
    def test_fixture_paper_run_records_trade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "paper.sqlite"
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "paper-run",
                        "--fixture",
                        "tests/fixtures/weather_markets.json",
                        "--ledger",
                        str(ledger_path),
                        "--min-model-count",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(PaperLedger(ledger_path).open_trades()), 1)

    def test_should_skip_late_same_day_market_without_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "late",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_latest_entry_hour_local=14),
            max_lead_days=2,
            open_token_ids=set(),
            now=datetime(2026, 6, 4, 20, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(should_process)
        self.assertIn("cutoff", reason or "")

    def test_should_process_late_same_day_market_with_open_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "late",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_latest_entry_hour_local=14),
            max_lead_days=2,
            open_token_ids={"tok-yes"},
            now=datetime(2026, 6, 4, 20, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(should_process)
        self.assertIsNone(reason)

    def test_min_lead_days_skips_same_day_without_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "same-day",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_earliest_entry_hour_local=11, same_day_latest_entry_hour_local=17),
            max_lead_days=2,
            open_token_ids=set(),
            min_lead_days=1,
            now=datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(should_process)
        self.assertIn("lead window", reason or "")

    def test_min_lead_days_still_processes_open_same_day_position(self) -> None:
        market = parse_weather_market(
            {
                "id": "same-day",
                "question": "Will the highest temperature in New York City be 80°F or above on June 4?",
                "slug": "highest-temperature-new-york-june-4-2026-80-or-above",
                "outcomes": '["Yes", "No"]',
                "clobTokenIds": '["tok-yes", "tok-no"]',
                "outcomePrices": '["0.30", "0.70"]',
            },
            today=date(2026, 6, 5),
        )
        assert market is not None
        should_process, reason = _should_process_market(
            market,
            SignalSettings(same_day_earliest_entry_hour_local=11, same_day_latest_entry_hour_local=17),
            max_lead_days=2,
            open_token_ids={"tok-yes"},
            min_lead_days=1,
            now=datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(should_process)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
