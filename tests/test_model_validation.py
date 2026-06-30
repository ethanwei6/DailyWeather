from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from weather_strategy.model_validation import run_weather_model_validation


class WeatherModelValidationTests(unittest.TestCase):
    def test_validation_reports_weather_vs_market_brier_and_writes_log(self) -> None:
        rows = []
        for index in range(80):
            outcome = 1 if index % 4 else 0
            weather_probability = 0.82 if outcome else 0.28
            market_probability = 0.62 if outcome else 0.38
            rows.append(
                {
                    "generated_at": f"2026-01-{1 + index // 4:02d}T00:00:00+00:00",
                    "polymarket_payout": outcome,
                    "market_price": market_probability,
                    "fair_value": weather_probability,
                    "signal_eligible": index >= 60,
                    "model_probabilities": {
                        "single_run_ecmwf_ifs025.hourly_curve_max": weather_probability,
                        "single_run_best_match.hourly_curve_max": weather_probability,
                        "single_run_gfs_global.normal_conservative": 0.50,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.json"
            source.write_text(json.dumps({"scored_outcomes_detail": rows}), encoding="utf-8")
            result = run_weather_model_validation(
                source_run_log=source,
                train_fraction=0.70,
                run_log_dir=Path(directory) / "logs",
                max_coordinate_epochs=1,
            )

            self.assertEqual(result["rows"], 80)
            self.assertIn("recorded_fair_value", result["metrics"])
            self.assertIn("market_price", result["metrics"])
            self.assertTrue(result["best_weather_by_slice"]["test_all"]["beats_market"])
            self.assertLess(
                result["metrics"]["recorded_fair_value"]["test_all"]["brier"],
                result["metrics"]["market_price"]["test_all"]["brier"],
            )
            self.assertTrue(Path(result["run_log_path"]).exists())


if __name__ == "__main__":
    unittest.main()
