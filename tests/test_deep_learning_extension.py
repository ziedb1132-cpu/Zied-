"""Isolated regression tests for the runtime 300-feature/30-layer extension."""
import subprocess
import sys
import unittest


class DeepLearningExtensionTests(unittest.TestCase):
    def test_runtime_schema_features_and_network_depth(self):
        script = r"""
import numpy as np
import pandas as pd
import os, json, tempfile
import zied_quant as zq
import deep_learning_evolution as ext
ext.install(zq)
assert zq.N_FEATURES == 300, zq.N_FEATURES
assert len(zq.FEATURE_COLS) == 300
assert zq.CFG["EXPLORATION_PROB_MIN"] == 0.65
assert ext.WIN_THRESHOLD == 500 and ext.LOSS_THRESHOLD == 500  # closed-outcome pattern gate
n = 360
c = pd.Series(100 * np.exp(np.linspace(0, .2, n)))
df = pd.DataFrame({
    "o": c * .999, "h": c * 1.01, "l": c * .99, "c": c,
    "vol": np.linspace(1000, 2000, n),
    "avail": pd.date_range("2025-01-01", periods=n, freq="h"),
})
out = zq.tf_features(df)
assert out.shape == (n, 58), out.shape
assert set(ext.EXTRA_NAMES).issubset(out.columns)
assert np.isfinite(out.iloc[-1].to_numpy(dtype=float)).all()
if zq.torch is not None:
    model = zq.ZiedNet(zq.N_FEATURES)
    assert sum(isinstance(layer, zq.nn.Linear) for layer in model.modules()) == 30

# Every newly created tracked signal must persist an immutable, named 300-value
# feature snapshot in both the signal log and its tracked-signal record.
with tempfile.TemporaryDirectory() as tmp:
    old_db = zq.CFG["DB"]
    zq.CFG["DB"] = os.path.join(tmp, "features.db")
    try:
        paper = zq.Paper()
        vector = np.linspace(-1.0, 1.0, zq.N_FEATURES, dtype=np.float64)
        created = paper.create_tracked_signal({
            "inst": "FEATURE-TEST-USDT", "price": 100.0, "atr": 0.02,
            "feats": vector, "prob": None, "rank": 1,
            "ts": 1_800_000_000.0, "feature_ts": 1_800_000_000.0,
            "prob_source": "ranking_exploration", "regime": "TEST",
            "entry_score": 75.0,
        })
        assert created and created["feature_count"] == 300, created
        signal = paper.db.execute(
            "SELECT feats,feature_count,feature_names,feature_schema_version "
            "FROM signals WHERE id=?", (created["id"],)
        ).fetchone()
        tracked = paper.db.execute(
            "SELECT feats,feature_count,feature_names,feature_schema_version "
            "FROM tracked_signals WHERE signal_id=?", (created["id"],)
        ).fetchone()
        for row in (signal, tracked):
            saved = np.asarray(json.loads(row["feats"]), dtype=np.float64)
            names = json.loads(row["feature_names"])
            assert row["feature_count"] == 300
            assert row["feature_schema_version"] == zq.DATA_VER
            assert len(names) == 300 and names == zq.FEATURE_COLS
            assert saved.shape == (300,) and np.array_equal(saved, vector)
        paper.close_tracked_signal(created["id"], 1_800_000_300.0,
                                   101.0, "TP", 0.009, 1.0, "TP")
        closed = paper.db.execute(
            "SELECT s.feats,s.feature_count,t.status,s.labeled,s.ret "
            "FROM signals s JOIN tracked_signals t ON t.signal_id=s.id "
            "WHERE s.id=?", (created["id"],)
        ).fetchone()
        assert closed["feature_count"] == 300
        assert closed["status"] == "CLOSED" and closed["labeled"] == 1
        assert np.array_equal(np.asarray(json.loads(closed["feats"])), vector)
        paper.db.close()
    finally:
        zq.CFG["DB"] = old_db
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)


if __name__ == "__main__":
    unittest.main()
