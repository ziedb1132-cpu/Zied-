import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exit_evolution import (
    BASELINE_NAME,
    POLICIES,
    _base_sl,
    _regime,
    _simulate,
)


class ExitEvolutionTests(unittest.TestCase):
    COST = 0.0035

    def test_volatility_regimes_are_stable(self):
        self.assertEqual(_regime(0.008), "LOW")
        self.assertEqual(_regime(0.018), "MID")
        self.assertEqual(_regime(0.035), "HIGH")

    def test_stop_wins_when_tp_and_sl_touch_same_candle(self):
        result = _simulate(
            100, 0.01, [104], [98], [101],
            self.COST, 1.5, 3.0, 1.5, 1.8, 10,
        )
        self.assertIsNotNone(result)
        net_ret, reason, hold = result
        self.assertEqual(reason, "SL")
        self.assertEqual(hold, 1)
        self.assertAlmostEqual(net_ret, 0.985 - 1.0 - self.COST)

    def test_take_profit_and_time_exit(self):
        tp = _simulate(
            100, 0.01, [104], [102], [103],
            self.COST, 1.5, 3.0, 1.5, 1.8, 10,
        )
        self.assertEqual(tp[1], "TP")
        self.assertAlmostEqual(tp[0], 0.03 - self.COST)

        timed = _simulate(
            100, 0.01, [101], [100], [102],
            self.COST, 1.5, 3.0, 1.5, 1.8, 1,
        )
        self.assertEqual(timed[1], "TIME")
        self.assertEqual(timed[2], 1)
        self.assertAlmostEqual(timed[0], 0.02 - self.COST)

    def test_trailing_stop_is_at_least_one_percent_above_entry(self):
        result = _simulate(
            100, 0.01,
            [102, 101.5], [101.2, 100.8], [101.8, 101.0],
            self.COST, 1.8, 4.0, 1.2, 1.8, 10,
        )
        self.assertIsNotNone(result)
        net_ret, reason, _ = result
        self.assertEqual(reason, "TRAIL")
        # The trailing stop must protect at least +1% from the entry price.
        self.assertGreaterEqual(net_ret + self.COST, 0.01 - 1e-12)

    def test_policy_grid_contains_baseline_and_is_bounded(self):
        names = {policy[0] for policy in POLICIES}
        self.assertIn(BASELINE_NAME, names)
        self.assertTrue(all(
            1.4 <= _base_sl(atr) + delta <= 2.4
            and 2.5 <= tp <= 4.0
            and max_bars > 0
            for _, delta, tp, _, _, max_bars in POLICIES
            for atr in (0.008, 0.018, 0.035)
        ))


if __name__ == "__main__":
    unittest.main()
