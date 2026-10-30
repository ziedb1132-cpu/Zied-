import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import zied_quant as zq
import advanced_intelligence as ai


class AdvancedIntelligenceTests(unittest.TestCase):
    def test_probability_calibration_metrics_are_bounded(self):
        y = np.array([0, 1] * 40, dtype=float)
        p = np.where(y == 1, 0.8, 0.2)
        report = ai.calibration_metrics(y, p)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["n"], 80)
        self.assertGreaterEqual(report["brier"], 0.0)
        self.assertLessEqual(report["brier"], 1.0)
        self.assertLess(report["expected_calibration_error"], 0.21)

    def test_calibration_requires_enough_independent_gate_rows(self):
        report = ai.calibration_metrics([0, 1, 0], [0.2, 0.8, 0.3])
        self.assertEqual(report["status"], "insufficient_gate_samples")

    def test_feature_drift_monitor_does_not_change_features(self):
        class Brain:
            ready = True
            mu = np.zeros(zq.N_FEATURES)
            sd = np.ones(zq.N_FEATURES)

        frame = pd.DataFrame(
            np.full((64, zq.N_FEATURES), 3.0), columns=zq.FEATURE_COLS
        )
        old_dir = zq.CFG["DIR"]
        with tempfile.TemporaryDirectory() as tmp:
            zq.CFG["DIR"] = tmp
            try:
                before = frame.to_numpy(copy=True)
                report = ai.feature_drift_report(zq, Brain(), frame)
                self.assertEqual(report["status"], "ok")
                self.assertEqual(report["feature_count"], zq.N_FEATURES)
                self.assertEqual(report["drifted_feature_count"], zq.N_FEATURES)
                self.assertTrue(os.path.exists(os.path.join(tmp, "feature_drift.json")))
                np.testing.assert_array_equal(frame.to_numpy(), before)
            finally:
                zq.CFG["DIR"] = old_dir

    def test_correlation_filter_blocks_strong_positive_or_negative_overlap(self):
        class Paper:
            def open_tracked_signals(self):
                return [{"inst": "OPEN-USDT"}]

        old_returns = zq._hourly_returns
        x = np.arange(80, dtype=float)
        zq._hourly_returns = lambda inst: pd.Series(x if inst == "NEW-USDT" else -x,
                                                    index=np.arange(80))
        try:
            result = ai.portfolio_correlation_filter(zq, Paper(), "NEW-USDT", 0.85, {})
            self.assertFalse(result["allowed"])
            self.assertAlmostEqual(result["max_abs_correlation"], 1.0, places=6)
        finally:
            zq._hourly_returns = old_returns


if __name__ == "__main__":
    unittest.main()
