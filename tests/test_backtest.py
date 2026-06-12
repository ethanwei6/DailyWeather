from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from weather_strategy.backtest import load_calibration_weights


class BacktestTest(unittest.TestCase):
    def test_load_calibration_weights_aliases_legacy_gfs_source(self) -> None:
        payload = {
            "source_weights": {"open_meteo_gfs_hrrr": 0.62},
            "model_weights": {"open_meteo_gfs_hrrr.kernel_wide": 1.08},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            source_weights, model_weights = load_calibration_weights(path)

        self.assertEqual(source_weights["open_meteo_gfs_hrrr"], 0.62)
        self.assertEqual(source_weights["open_meteo_gfs_best_match"], 0.62)
        self.assertEqual(model_weights["open_meteo_gfs_hrrr.kernel_wide"], 1.08)
        self.assertEqual(model_weights["open_meteo_gfs_best_match.kernel_wide"], 1.08)


if __name__ == "__main__":
    unittest.main()
