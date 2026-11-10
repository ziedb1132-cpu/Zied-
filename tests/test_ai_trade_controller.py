"""Regression tests for pattern-aware AI trading integration."""
import json
import os
import subprocess
import sys
import tempfile
import unittest


class PatternControllerTests(unittest.TestCase):
    def test_patterns_influence_scores_only_after_discovery_threshold(self):
        script = r"""
import json, os, tempfile
import numpy as np
import zied_quant as zq
import deep_learning_evolution as ext
import ai_trade_controller as ctl
import candle_news_intelligence as cni
ext.install(zq)
cni.fetch_news = lambda: []
zq.get_candles = lambda *args, **kwargs: __import__('pandas').DataFrame()
cni.install(zq)
with tempfile.TemporaryDirectory() as d:
    zq.CFG["DIR"] = d
    feature_count = zq.N_FEATURES
    idx = zq.FEATURE_COLS.index("1H_ret1")
    payload = {
        "feature_count": feature_count, "feature_names": list(zq.FEATURE_COLS),
        "winning_samples": 500, "losing_samples": 500,
        "top_recurring_features": [{
            "feature": "1H_ret1", "winner_median": 2.0, "loser_median": -2.0,
            "winner_q25_q75": [1.0, 3.0], "loser_q25_q75": [-3.0, -1.0],
            "robust_effect": 2.0
        }]
    }
    with open(os.path.join(d, "winning_patterns.json"), "w") as f:
        json.dump(payload, f)
    winner_like = np.zeros(feature_count); winner_like[idx] = 2.0
    loser_like = np.zeros(feature_count); loser_like[idx] = -2.0
    scores = ctl.pattern_scores(zq, np.vstack([winner_like, loser_like]))
    assert scores[0] > scores[1], scores
    assert scores[0] > 0.5 and scores[1] < 0.5, scores
    payload["winning_samples"] = 499
    with open(os.path.join(d, "winning_patterns.json"), "w") as f:
        json.dump(payload, f)
    inactive = ctl.pattern_scores(zq, np.vstack([winner_like, loser_like]))
    assert np.allclose(inactive, [0.5, 0.5]), inactive
ctl.install(zq)
ctl.install(zq)
assert zq._pattern_controller_installed

# Regression: both overlays must preserve rank_universe(paper=..., regimes=...).
import pandas as pd
X = pd.DataFrame(0.0, index=["AAA-USDT", "BBB-USDT"], columns=zq.FEATURE_COLS)
vol = pd.Series([1_000_000.0, 2_000_000.0], index=X.index)
regimes = pd.Series(["BULL", "RANGE"], index=X.index)
ranked = zq.rank_universe(X, vol, paper=None, regimes=regimes)
assert len(ranked) == 2 and set(ranked.index) == set(X.index), ranked
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)


if __name__ == "__main__":
    unittest.main()
