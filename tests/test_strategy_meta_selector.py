import tempfile
import unittest
import numpy as np
import pandas as pd

from strategy_meta_selector import market_regime, _metrics, run_meta_selector


class MetaSelectorTests(unittest.TestCase):
    def test_regime_classifier(self):
        bullish = pd.DataFrame([{"atr_pct":0.01,"ema_gap":0.02,"ema_slope":0.003}])
        bearish = pd.DataFrame([{"atr_pct":0.01,"ema_gap":-0.02,"ema_slope":-0.003}])
        volatile = pd.DataFrame([{"atr_pct":0.04,"ema_gap":0.02,"ema_slope":0.003}])
        self.assertEqual(market_regime(bullish), "BULL")
        self.assertEqual(market_regime(bearish), "BEAR")
        self.assertEqual(market_regime(volatile), "HIGH_VOL")

    def test_metrics_net_and_drawdown(self):
        m = _metrics([{"net_return":0.04},{"net_return":-0.02},{"net_return":0.01}])
        self.assertEqual(m["trades"], 3)
        self.assertAlmostEqual(m["avg_net"], 0.01)
        self.assertGreater(m["profit_factor"], 1.0)
        self.assertGreaterEqual(m["max_drawdown"], 0.0)

    def test_shadow_state_persists_without_live_integration(self):
        with tempfile.TemporaryDirectory() as td:
            universe = pd.DataFrame({"last":[100,100,100]},index=["A-USDT","B-USDT","C-USDT"])
            def get_candles(symbol, tf, limit):
                n=160
                close=100*np.exp(np.linspace(0,0.15,n))
                op=np.r_[close[0],close[:-1]]
                return pd.DataFrame({"ts":np.arange(n),"o":op,
                    "h":np.maximum(op,close)*1.002,"l":np.minimum(op,close)*0.998,
                    "c":close,"vol":np.full(n,1000.0)})
            report=run_meta_selector(universe,get_candles,td,
                discovery_report={"latest_experiments":[],"champion":None})
            self.assertEqual(report["cycle"],1)
            self.assertFalse(report["live_integration"])
            import json, os
            saved=json.load(open(os.path.join(td,"strategy_meta_selector_memory.json"),encoding="utf-8"))
            self.assertEqual(saved["cycle"],1)
            self.assertTrue(os.path.exists(os.path.join(td,"strategy_meta_selector_report.json")))


if __name__=="__main__":
    unittest.main()
