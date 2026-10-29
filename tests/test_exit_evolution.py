import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exit_evolution import simulate_exit, volatility_regime, BASELINE, CANDIDATES


class ExitEvolutionTests(unittest.TestCase):
    def test_volatility_regimes_are_stable(self):
        self.assertEqual(volatility_regime(0.008), "LOW")
        self.assertEqual(volatility_regime(0.018), "MID")
        self.assertEqual(volatility_regime(0.035), "HIGH")

    def test_stop_wins_when_tp_and_sl_touch_same_candle(self):
        result = simulate_exit(100, 0.01, [104], [98], [101], 1.5, 3.0, 0.0035)
        self.assertEqual(result["reason"], "SL")
        self.assertAlmostEqual(result["net_ret"], 0.985 / 1.0 - 1.0 - 0.0035)

    def test_take_profit_and_time_exit(self):
        tp = simulate_exit(100, 0.01, [104], [102], [103], 1.5, 3.0, 0.0035)
        self.assertEqual(tp["reason"], "TP")
        self.assertAlmostEqual(tp["net_ret"], 0.03 - 0.0035)
        timed = simulate_exit(100, 0.01, [101], [100], [102], 1.5, 3.0, 0.0035)
        self.assertEqual(timed["reason"], "TIME")
        self.assertAlmostEqual(timed["net_ret"], 0.02 - 0.0035)

    def test_candidate_grid_contains_baseline_and_is_bounded(self):
        self.assertIn(BASELINE, CANDIDATES)
        self.assertTrue(all(0 < sl <= 2.5 and tp > sl for sl, tp in CANDIDATES))


if __name__ == "__main__":
    unittest.main()
