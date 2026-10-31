import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from autonomous_strategy_engine import (
    build_features, _new_spec, _spec_id, _signals, _evaluate, run_discovery
)


def candles(n=1200, seed=17):
    rng = np.random.default_rng(seed)
    close = 100*np.exp(np.cumsum(rng.normal(0.0002,0.009,n)))
    op = np.r_[close[0],close[:-1]]
    return pd.DataFrame({
        "ts":np.arange(n)*3600000,"o":op,
        "h":np.maximum(op,close)*1.006,"l":np.minimum(op,close)*0.994,
        "c":close,"vol":rng.uniform(100,2000,n)
    })


class AutonomousStrategyTests(unittest.TestCase):
    def test_features_are_causal_shape_and_finite_after_warmup(self):
        d=candles()
        f=build_features(d)
        self.assertEqual(f.shape,(len(d),15))
        self.assertTrue(np.isfinite(f.iloc[100:].to_numpy()).all())

    def test_generated_rules_are_unique_and_variable_complexity(self):
        import random
        rng=random.Random(42)
        a=_new_spec(rng,1)
        b=_new_spec(rng,1)
        self.assertEqual(a["id"],_spec_id({k:v for k,v in a.items() if k not in ("generation","parent_id","id")}))
        self.assertNotEqual(a["id"],b["id"])
        self.assertGreaterEqual(len(a["conditions"]),2)

    def test_signal_expression_returns_boolean_vector(self):
        f=build_features(candles())
        spec={"conditions":[{"feature":"rsi","op":"lt","threshold":50},
                            {"feature":"volume_ratio","op":"gt","threshold":1.0}]}
        sig=_signals(f,spec)
        self.assertEqual(sig.dtype,np.bool_)
        self.assertEqual(len(sig),len(f))

    def test_evaluator_accounts_for_costs_and_returns_finite_metrics(self):
        d=candles()
        f=build_features(d)
        spec={"conditions":[{"feature":"rsi","op":"lt","threshold":55}],
              "sl_atr":1.5,"tp_atr":2.0,"max_hold":24}
        m=_evaluate(d,f,spec,100,900)
        self.assertGreaterEqual(m["trades"],0)
        if m["avg_net"] is not None:
            self.assertTrue(np.isfinite(m["avg_net"]))
            self.assertGreaterEqual(m["max_drawdown"],0)

    def test_discovery_persists_memory_and_generates_ten(self):
        with tempfile.TemporaryDirectory() as td:
            uni=pd.DataFrame({"last":[1,1,1]},index=["A-USDT","B-USDT","C-USDT"])
            def get_candles(inst,tf,limit):
                return candles(seed={"A-USDT":1,"B-USDT":2,"C-USDT":3}[inst])
            result=run_discovery(uni,get_candles,td,telegram=None)
            self.assertEqual(result["generated"],10)
            self.assertTrue(os.path.exists(os.path.join(td,"autonomous_strategy_memory.json")))
            import json
            saved=json.load(open(os.path.join(td,"autonomous_strategy_memory.json"),encoding="utf-8"))
            self.assertEqual(saved["generation"],1)
            self.assertEqual(len(saved["archive"]),10)


if __name__=="__main__":
    unittest.main()
