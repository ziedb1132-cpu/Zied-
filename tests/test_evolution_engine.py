import math
import unittest
import numpy as np
from evolution_engine import trade_metrics


class TradeMetricsTests(unittest.TestCase):
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
        self.assertTrue(math.isinf(m["pf"]))


if __name__ == "__main__":
    unittest.main()
