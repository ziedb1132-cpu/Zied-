import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import evolution_ai_bridge


class _FakeZied:
    @staticmethod
    def classify_market_regime(row):
        return str(row["regime"])


class EvolutionAIBridgeTests(unittest.TestCase):
    def _run(self, profile):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strategy_meta_selector_report.json"
            path.write_text(json.dumps({
                "updated_at": time.time(),
                "selected_by_symbol_regime": {"SOL-USDT|BULL": [profile]}
            }), encoding="utf-8")
            frame = pd.DataFrame(
                {"regime": ["BULL", "BEAR"]},
                index=["SOL-USDT", "SOL-USDT"],
            )
            return evolution_ai_bridge.apply_strategy_feedback(
                _FakeZied, frame, np.asarray([0.60, 0.60]), str(path)
            )

    def test_approved_forward_shadow_profile_softly_blends(self):
        result = self._run({
            "paper_approved": True, "trades": 100, "win_rate": 0.75,
            "avg_net": 0.004, "profit_factor": 1.5, "max_drawdown": 0.10,
        })
        self.assertGreater(result[0], 0.60)
        self.assertEqual(result[1], 0.60)
        self.assertLessEqual(result[0], 0.615)

    def test_unapproved_or_weak_profile_does_not_change_primary(self):
        result = self._run({
            "paper_approved": False, "trades": 100, "win_rate": 0.90,
            "avg_net": 0.01, "profit_factor": 2.0, "max_drawdown": 0.05,
        })
        np.testing.assert_allclose(result, [0.60, 0.60])

    def test_missing_report_keeps_primary_predictions(self):
        frame = pd.DataFrame({"regime": ["BULL"]}, index=["SOL-USDT"])
        result = evolution_ai_bridge.apply_strategy_feedback(
            _FakeZied, frame, np.asarray([0.60]), "/missing/strategy-report.json"
        )
        np.testing.assert_allclose(result, [0.60])

    def test_invalid_primary_probabilities_are_rejected(self):
        frame = pd.DataFrame({"regime": ["BULL"]}, index=["SOL-USDT"])
        with self.assertRaises(ValueError):
            evolution_ai_bridge.apply_strategy_feedback(
                _FakeZied, frame, np.asarray([np.nan]), "/missing/report.json"
            )


if __name__ == "__main__":
    unittest.main()
