import unittest
import numpy as np
import pandas as pd
from strategy_evolver import candidate_specs, signal_for, evaluate, evaluate_candidate


def sample_candles(n=1200, seed=4):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * 1.004
    lo = np.minimum(op, close) * 0.996
    return pd.DataFrame({"ts": np.arange(n) * 3600_000, "o": op, "h": hi, "l": lo,
                         "c": close, "vol": rng.uniform(100, 1000, n)})


class StrategyEvolutionTests(unittest.TestCase):
    def test_candidate_generator_has_multiple_rule_families(self):
        specs = candidate_specs()
        self.assertGreater(len(specs), 20)
        self.assertEqual({s["kind"] for s in specs}, {"trend_pullback", "breakout", "rsi_reversal"})

    def test_signals_are_boolean_and_correct_length(self):
        df = sample_candles()
        for spec in candidate_specs()[:5] + candidate_specs()[-3:]:
            sig = signal_for(df, spec)
            self.assertEqual(sig.dtype, np.bool_)
            self.assertEqual(len(sig), len(df))

    def test_evaluation_returns_cost_adjusted_metrics(self):
        df = sample_candles()
        sig = np.zeros(len(df), dtype=bool)
        sig[[100, 200, 300, 400]] = True
        m = evaluate(df, sig, 80, 500)
        self.assertLessEqual(m["trades"], 4)
        self.assertTrue(m["avg_net"] is None or np.isfinite(m["avg_net"]))
        self.assertGreaterEqual(m["profit_factor"], 0)

    def test_chronological_splits_produce_complete_report(self):
        df = sample_candles()
        result = evaluate_candidate([df, df.assign(c=df["c"] * 1.001)], candidate_specs()[0])
        self.assertEqual(set(result), {"train", "tune", "test"})
        self.assertIn("trades", result["test"])


if __name__ == "__main__":
    unittest.main()
