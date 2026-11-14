import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from autonomous_strategy_engine import (
    build_features, FEATURES, _LIBRARY_FEATURES, _new_spec, _spec_id, _signals, _evaluate, run_discovery, _generate_candidates
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
        self.assertEqual(len(_LIBRARY_FEATURES),546)
        self.assertEqual(len(FEATURES),561)
        self.assertEqual(f.shape,(len(d),561))
        self.assertTrue(np.isfinite(f.iloc[250:].to_numpy()).all())

    def test_feature_library_has_no_future_data_leakage(self):
        d=candles(n=700)
        full=build_features(d)
        truncated=build_features(d.iloc[:600].copy())
        pd.testing.assert_frame_equal(
            full.iloc[:600].reset_index(drop=True),
            truncated.reset_index(drop=True),
            check_exact=False, rtol=1e-10, atol=1e-10,
        )

    def test_generated_rules_are_unique_and_variable_complexity(self):
        import random
        rng=random.Random(42)
        a=_new_spec(rng,1)
        b=_new_spec(rng,1)
        self.assertEqual(a["id"],_spec_id({k:v for k,v in a.items() if k not in ("generation","parent_id","id")}))
        self.assertNotEqual(a["id"],b["id"])
        self.assertGreaterEqual(len(a["conditions"]),2)

    def test_crossover_combines_parent_inputs_and_mutates_exit_genes(self):
        import random
        from autonomous_strategy_engine import _crossover_specs, _genome_summary
        a = {"id":"parent-a","spec":{"conditions":[
            {"feature":"ret1","op":"gt","threshold":0.0},
            {"feature":"rsi","op":"lt","threshold":50}],
            "sl_atr":1.5,"tp_atr":2.0,"max_hold":12}}
        b = {"id":"parent-b","spec":{"conditions":[
            {"feature":"ema_gap","op":"gt","threshold":0.01},
            {"feature":"volume_ratio","op":"gt","threshold":1.25}],
            "sl_atr":2.0,"tp_atr":3.0,"max_hold":24}}
        child = _crossover_specs(random.Random(7), a, b, 3)
        self.assertEqual(child["generation"], 3)
        self.assertTrue(child["crossover"])
        self.assertGreaterEqual(len(child["conditions"]), 2)
        self.assertLessEqual(len(child["conditions"]), 12)
        self.assertTrue(set(_genome_summary(child)["input_features"]).issubset(set(FEATURES)))
        self.assertIn(child["sl_atr"], [1.0,1.25,1.5,1.75,2.0,2.5])
        self.assertIn(child["tp_atr"], [1.25,1.5,2.0,2.5,3.0,4.0])
        self.assertIn(child["max_hold"], [6,12,18,24,36,48])

    def test_generation_uses_exact_quotas_when_two_validated_parents_exist(self):
        import random
        parents = [
            {"id":"parent-a","status":"validation_pass","spec":{"conditions":[
                {"feature":"ret1","op":"gt","threshold":0.0},
                {"feature":"rsi","op":"lt","threshold":50}],"sl_atr":1.5,"tp_atr":2.0,"max_hold":12}},
            {"id":"parent-b","status":"validation_pass","spec":{"conditions":[
                {"feature":"ema_gap","op":"gt","threshold":0.01},
                {"feature":"volume_ratio","op":"gt","threshold":1.25}],"sl_atr":2.0,"tp_atr":3.0,"max_hold":24}},
        ]
        candidates, mutations, crossovers = _generate_candidates(
            random.Random(123), 2, parents, set(), 30, 20)
        self.assertEqual(len(candidates), 50)
        self.assertEqual(mutations, 30)
        self.assertEqual(crossovers, 20)
        self.assertEqual(len({x["id"] for x in candidates}), 50)

    def test_first_generation_fills_crossover_slots_with_mutations(self):
        import random
        candidates, mutations, crossovers = _generate_candidates(
            random.Random(321), 1, [], set(), 30, 20)
        self.assertEqual(len(candidates), 50)
        self.assertEqual(mutations, 50)
        self.assertEqual(crossovers, 0)

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

    def test_discovery_persists_memory_and_generates_fifty(self):
        with tempfile.TemporaryDirectory() as td:
            uni=pd.DataFrame({"last":[1,1,1]},index=["A-USDT","B-USDT","C-USDT"])
            def get_candles(inst,tf,limit):
                return candles(seed={"A-USDT":1,"B-USDT":2,"C-USDT":3}[inst])
            result=run_discovery(uni,get_candles,td,telegram=None)
            self.assertEqual(result["generated"],50)
            self.assertEqual(result["mutation_candidates"],50)
            self.assertEqual(result["crossover_candidates"],0)
            self.assertTrue(os.path.exists(os.path.join(td,"autonomous_strategy_memory.json")))
            import json
            saved=json.load(open(os.path.join(td,"autonomous_strategy_memory.json"),encoding="utf-8"))
            self.assertEqual(saved["generation"],1)
            self.assertEqual(len(saved["archive"]),50)


if __name__=="__main__":
    unittest.main()
