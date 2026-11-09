import json
import os
import sqlite3
import tempfile
import threading
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

    def test_feature_quality_audit_finds_duplicates_constants_and_missingness(self):
        class SmallZiedQuant:
            N_FEATURES = 3
            FEATURE_COLS = ["signal", "duplicate", "constant"]

        y = np.array([0.0, 1.0] * 50)
        signal = y + np.linspace(0.0, 0.01, len(y))
        X = np.column_stack([signal, signal.copy(), np.ones(len(y))])
        X[:10, 0] = np.nan
        X[:10, 1] = np.nan
        report = ai.feature_quality_report(
            SmallZiedQuant(), {"X_gate": X, "gate_y": y}
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["feature_count"], 3)
        self.assertAlmostEqual(report["finite_feature_fraction"], 0.9333333333, places=6)
        self.assertAlmostEqual(report["features_per_sample"], 2.8, places=6)
        self.assertEqual(report["constant_feature_count"], 1)
        self.assertGreaterEqual(report["duplicate_feature_count"], 1)
        self.assertGreater(report["high_missing_feature_count"], 0)
        self.assertIn("constant_or_near_constant_features", report["risk_flags"])

    def test_feature_audit_runs_from_matured_db_rows_before_model_gate(self):
        class SmallZiedQuant:
            N_FEATURES = 3
            DATA_VER = 1
            FEATURE_COLS = ["signal", "duplicate", "constant"]

        class Paper:
            def __init__(self):
                self.lock = threading.RLock()
                self.db = sqlite3.connect(":memory:")
                self.db.row_factory = sqlite3.Row

        paper = Paper()
        paper.db.execute(
            "CREATE TABLE signals(feats TEXT,label INTEGER,labeled INTEGER,"
            "feature_count INTEGER,feature_schema_version INTEGER,ts REAL)"
        )
        for i in range(100):
            value = float(i % 2)
            vector = [value, value, 1.0]
            paper.db.execute(
                "INSERT INTO signals VALUES(?, ?, 1, 3, 1, ?)",
                (json.dumps(vector), int(value), float(i))
            )
        paper.db.commit()
        report = ai.feature_quality_report_from_db(SmallZiedQuant(), paper)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["sample_rows"], 100)
        self.assertEqual(report["source"], "recent_matured_signal_feature_snapshots")

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

    def test_heavy_backtest_guard_allows_only_one_run_per_24_hours(self):
        class MetaPaper:
            def __init__(self):
                self.meta = {}
            def _get(self, key):
                return self.meta.get(key)
            def _set(self, key, value):
                self.meta[key] = str(value)

        paper = MetaPaper()
        self.assertTrue(zq.daily_backtest_due(paper, now=1000))
        zq.mark_daily_backtest_attempt(paper, now=1000)
        self.assertFalse(zq.daily_backtest_due(paper, now=1000 + 86399))
        self.assertTrue(zq.daily_backtest_due(paper, now=1000 + 86400))

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
