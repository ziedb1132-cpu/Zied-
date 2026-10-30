import os
import sqlite3
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exit_evolution import (
    BASELINE_NAME,
    POLICIES,
    _base_sl,
    _ensure,
    _regime,
    _canonical_reason,
    _holdout_metrics,
    _holdout_passes,
    _simulate,
)


class ExitEvolutionTests(unittest.TestCase):
    COST = 0.0035

    def test_volatility_regimes_are_stable(self):
        self.assertEqual(_regime(0.008), "LOW")
        self.assertEqual(_regime(0.018), "MID")
        self.assertEqual(_regime(0.035), "HIGH")

    def test_stop_wins_when_tp_and_sl_touch_same_candle(self):
        result = _simulate(100, 0.01, [104], [98], [101],
                           self.COST, 1.5, 3.0, 1.5, 1.8, 10)
        self.assertIsNotNone(result)
        net_ret, reason, hold = result
        self.assertEqual(reason, "SL")
        self.assertEqual(hold, 1)
        self.assertAlmostEqual(net_ret, 0.985 - 1.0 - self.COST)

    def test_take_profit_and_time_exit(self):
        tp = _simulate(100, 0.01, [104], [102], [103],
                       self.COST, 1.5, 3.0, 1.5, 1.8, 10)
        self.assertEqual(tp[1], "TP")
        self.assertAlmostEqual(tp[0], 0.03 - self.COST)

        timed = _simulate(100, 0.01, [101], [100], [102],
                          self.COST, 1.5, 3.0, 1.5, 1.8, 1)
        self.assertEqual(timed[1], "TIME")
        self.assertEqual(timed[2], 1)
        self.assertAlmostEqual(timed[0], 0.02 - self.COST)

    def test_trailing_stop_is_at_least_one_percent_above_entry(self):
        result = _simulate(100, 0.01,
            [102, 101.5], [101.2, 100.8], [101.8, 101.0],
            self.COST, 1.8, 4.0, 1.2, 1.8, 10)
        self.assertIsNotNone(result)
        net_ret, reason, _ = result
        self.assertEqual(reason, "TRAIL")
        self.assertGreaterEqual(net_ret + self.COST, 0.01 - 1e-12)

    def test_existing_shadow_database_is_upgraded_without_data_loss(self):
        class Paper:
            def __init__(self):
                self.lock = threading.RLock()
                self.db = sqlite3.connect(":memory:")
                self.db.row_factory = sqlite3.Row

        paper = Paper()
        paper.db.execute(
            "CREATE TABLE exit_policy_shadow_v2("
            "tracked_id INTEGER NOT NULL, regime TEXT NOT NULL, policy TEXT NOT NULL,"
            "net_ret REAL NOT NULL, reason TEXT NOT NULL, hold_bars INTEGER NOT NULL,"
            "opened_ts REAL NOT NULL, PRIMARY KEY(tracked_id,policy))"
        )
        paper.db.execute(
            "INSERT INTO exit_policy_shadow_v2 VALUES(1,'LOW','current',0.01,'TP',3,1.0)"
        )
        _ensure(paper)
        columns = {r["name"] for r in paper.db.execute(
            "PRAGMA table_info(exit_policy_shadow_v2)"
        )}
        self.assertTrue({"actual_reason", "actual_net_ret", "actual_hold_bars"}.issubset(columns))
        old = paper.db.execute(
            "SELECT net_ret,reason FROM exit_policy_shadow_v2 WHERE tracked_id=1"
        ).fetchone()
        self.assertAlmostEqual(old["net_ret"], 0.01)
        self.assertEqual(old["reason"], "TP")

    def test_actual_exit_reasons_are_normalized_for_learning(self):
        self.assertEqual(_canonical_reason("TP"), "TP")
        self.assertEqual(_canonical_reason("SL"), "SL")
        self.assertEqual(_canonical_reason("TRAILING_STOP"), "TRAIL")
        self.assertEqual(_canonical_reason("TIME_EXIT"), "TIME")
        self.assertEqual(_canonical_reason(None), "UNKNOWN")

    def test_holdout_requires_beating_actual_tp_sl_trailing_results(self):
        candidate = [0.025, 0.030, 0.018, 0.022, 0.027] * 4
        baseline = [0.010, 0.012, 0.009, 0.011, 0.010] * 4
        actual = [0.008, 0.009, 0.007, 0.008, 0.009] * 4
        metrics = _holdout_metrics(candidate, baseline, actual)
        self.assertTrue(_holdout_passes(metrics))
        self.assertEqual(metrics["n"], 20)
        self.assertGreater(metrics["paired_win_rate"], 0.99)

    def test_holdout_rejects_candidate_that_only_beats_shadow_baseline(self):
        candidate = [0.012] * 20
        baseline = [0.005] * 20
        actual = [0.020] * 20
        metrics = _holdout_metrics(candidate, baseline, actual)
        self.assertFalse(_holdout_passes(metrics))

    def test_holdout_rejects_nonpositive_or_small_samples(self):
        self.assertFalse(_holdout_passes(_holdout_metrics(
            [-0.01] * 20, [-0.02] * 20, [-0.03] * 20)))
        self.assertFalse(_holdout_passes(_holdout_metrics(
            [0.03] * 10, [0.01] * 10, [0.005] * 10)))

    def test_selected_exit_policy_is_migrated_without_dropping_tracked_rows(self):
        class Paper:
            def __init__(self):
                self.lock = threading.RLock()
                self.db = sqlite3.connect(":memory:")
                self.db.row_factory = sqlite3.Row

        paper = Paper()
        paper.db.execute(
            "CREATE TABLE tracked_signals(id INTEGER PRIMARY KEY, status TEXT, "
            "regime TEXT, close_reason TEXT, outcome TEXT, ret REAL, hold REAL, "
            "opened_ts REAL, closed_ts REAL)"
        )
        paper.db.execute(
            "INSERT INTO tracked_signals VALUES(1,'CLOSED','LOW','TRAIL','WIN',0.01,4,1,2)"
        )
        _ensure(paper)
        columns = {r["name"] for r in paper.db.execute(
            "PRAGMA table_info(tracked_signals)"
        )}
        self.assertIn("exit_policy", columns)
        row = paper.db.execute(
            "SELECT exit_policy FROM tracked_signals WHERE id=1"
        ).fetchone()
        self.assertEqual(row["exit_policy"], "legacy_unknown")

    def test_policy_grid_contains_baseline_and_is_bounded(self):
        policies = {p[0]: p for p in POLICIES}
        self.assertIn(BASELINE_NAME, policies)
        self.assertEqual(len(policies), len(POLICIES))
        for name, sl_delta, tp, activation, distance, max_bars in POLICIES:
            self.assertTrue(1.4 <= _base_sl(0.008) + sl_delta <= 2.4 + 1e-9, name)
            self.assertTrue(1.4 <= _base_sl(0.018) + sl_delta <= 2.4 + 1e-9, name)
            self.assertTrue(1.4 <= _base_sl(0.035) + sl_delta <= 2.4 + 1e-9, name)
            self.assertTrue(2.5 <= tp <= 4.0, name)
            self.assertGreater(activation, 0, name)
            self.assertGreater(distance, 0, name)
            self.assertGreater(max_bars, 0, name)


if __name__ == "__main__":
    unittest.main()
