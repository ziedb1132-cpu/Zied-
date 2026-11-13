import math
import unittest
import numpy as np
from evolution_engine import trade_metrics, chronological_train_tune_holdout_masks, holdout_edge_passes


class TradeMetricsTests(unittest.TestCase):
    def test_verified_671_samples_clear_the_obsolete_1000_row_floor(self):
        from evolution_engine import fit_partition_requirements
        block, min_train, min_tune, min_test = fit_partition_requirements(671)
        self.assertEqual(block, 100)
        self.assertEqual((min_train, min_tune, min_test), (40, 15, 15))

    def test_adaptive_partition_preserves_embargo_and_finds_usable_tuning_block(self):
        from evolution_engine import adaptive_chronological_partition
        ts = np.arange(672, dtype=float) * 3600
        block, masks, actual = adaptive_chronological_partition(
            ts, initial_block=100, embargo=24 * 3600
        )
        self.assertGreaterEqual(block, 100)
        self.assertGreaterEqual(actual[0], 40)
        self.assertGreaterEqual(actual[1], 15)
        self.assertGreaterEqual(actual[2], 15)
        train, tune, test = masks
        self.assertLess(np.max(ts[train]), np.min(ts[tune]) - 24 * 3600)
        self.assertLess(np.max(ts[tune]), np.min(ts[test]) - 24 * 3600)

    def test_supervised_fit_uses_the_shared_100_sample_minimum(self):
        from evolution_engine import fit_partition_requirements
        self.assertEqual(fit_partition_requirements(100)[1:], (40, 15, 15))
        with self.assertRaisesRegex(ValueError, "need 100"):
            fit_partition_requirements(99)

    def test_chronological_split_keeps_equal_timestamps_together(self):
        # Each timestamp represents multiple coins observed on the same candle.
        ts = np.repeat(np.arange(1600, dtype=float) * 3600, 3)
        train, tune, test = chronological_train_tune_holdout_masks(ts, 240, 24 * 3600)
        for timestamp in np.unique(ts):
            mask = ts == timestamp
            memberships = [bool(train[mask].any()), bool(tune[mask].any()), bool(test[mask].any())]
            self.assertLessEqual(sum(memberships), 1)
        self.assertGreater(train.sum(), 400)
        self.assertGreater(tune.sum(), 100)
        self.assertGreater(test.sum(), 100)

    def test_chronological_split_rejects_unsorted_or_invalid_timestamps(self):
        with self.assertRaisesRegex(ValueError, "finite and sorted"):
            chronological_train_tune_holdout_masks(np.array([1.0, 3.0, 2.0]), 1, 0)
        with self.assertRaisesRegex(ValueError, "finite and sorted"):
            chronological_train_tune_holdout_masks(np.array([1.0, np.nan]), 1, 0)

    def test_metrics_ignore_missing_returns(self):
        m = trade_metrics(np.array([0.02, -0.01, np.nan, 0.03]), np.array([1, 1, 1, 0]))
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["avg"], 0.005)
        self.assertAlmostEqual(m["pf"], 2.0)
        self.assertAlmostEqual(m["win"], 0.5)

    def test_empty_selection_fails_closed(self):
        m = trade_metrics(np.array([0.01, -0.01]), np.array([0, 0]))
        self.assertEqual(m["n"], 0)
        self.assertTrue(math.isnan(m["avg"]))
        self.assertEqual(m["pf"], 0.0)

    def test_all_winners_have_infinite_profit_factor(self):
        m = trade_metrics(np.array([0.01, 0.02]), np.array([1, 1]))
        self.assertEqual(m["pf"], 999.0)



    def test_holdout_edge_requires_positive_edge_over_baseline(self):
        selected = {"n": 35, "avg": 0.012, "pf": 1.2}
        baseline = {"avg": 0.010}
        self.assertTrue(holdout_edge_passes(selected, baseline))
        self.assertFalse(holdout_edge_passes(
            {"n": 35, "avg": 0.0105, "pf": 1.5}, baseline))
        self.assertFalse(holdout_edge_passes(
            {"n": 7, "avg": 0.020, "pf": 2.0}, baseline))
        self.assertFalse(holdout_edge_passes(
            {"n": 35, "avg": -0.001, "pf": 0.8}, baseline))

    def test_auc_uses_average_ranks_for_tied_scores(self):
        from zied_quant import auc_score
        self.assertAlmostEqual(auc_score(np.array([1, 0]), np.array([0.5, 0.5])), 0.5)
        # Positive beats one negative and ties the other: (1 + 0.5) / 2 = 0.75.
        self.assertAlmostEqual(auc_score(np.array([1, 0, 0]), np.array([0.9, 0.5, 0.9])), 0.75)

    def test_auc_rejects_nonfinite_scores(self):
        from zied_quant import auc_score
        with self.assertRaises(ValueError):
            auc_score(np.array([0, 1]), np.array([0.2, np.nan]))

if __name__ == "__main__":
    unittest.main()
