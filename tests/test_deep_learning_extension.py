"""Isolated regression tests for the expanded runtime feature schema and 30-layer extension."""
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
assert zq.N_FEATURES >= 300, zq.N_FEATURES
assert len(zq.FEATURE_COLS) == zq.N_FEATURES
assert zq.CFG["EXPLORATION_PROB_MIN"] == 0.65
assert ext.WIN_THRESHOLD == 500 and ext.LOSS_THRESHOLD == 500  # closed-outcome pattern gate
n = 800
t = np.arange(n, dtype=float)
c = pd.Series(100 * np.exp(
    np.linspace(0, .2, n) + .015*np.sin(t/7) + .007*np.sin(t/17)
))
df = pd.DataFrame({
    "o": c * (0.999 + .0005*np.sin(t/5)),
    "h": c * (1.006 + .001*np.sin(t/11)),
    "l": c * (0.994 - .001*np.sin(t/13)),
    "c": c,
    "vol": np.linspace(1000, 2000, n) * (1 + .2*np.sin(t/9)),
    "avail": pd.date_range("2025-01-01", periods=n, freq="h"),
})
out = zq.tf_features(df)
assert out.shape == (n, len(zq.NAMES)), out.shape
assert set(ext.EXTRA_NAMES).issubset(out.columns)
last_row = out.iloc[-1].to_numpy(dtype=float)
finite_count = int(np.isfinite(last_row).sum())
assert finite_count >= int(len(last_row) * 0.90), (
    f"Too few finite runtime features: {finite_count}/{len(last_row)}"
)
assert not np.isinf(last_row).any(), "Runtime feature matrix contains infinite values"
if zq.torch is not None:
    model = zq.ZiedNet(zq.N_FEATURES)
    assert sum(isinstance(layer, zq.nn.Linear) for layer in model.modules()) == 30

# Every newly created tracked signal must persist an immutable, named feature snapshot
# matching the complete runtime schema in both the signal log and tracked-signal record.
with tempfile.TemporaryDirectory() as tmp:
    old_db = zq.CFG["DB"]
    zq.CFG["DB"] = os.path.join(tmp, "features.db")
    try:
        paper = zq.Paper()
        feature_count = zq.N_FEATURES
        vector = np.linspace(-1.0, 1.0, feature_count, dtype=np.float64)
        created = paper.create_tracked_signal({
            "inst": "FEATURE-TEST-USDT", "price": 100.0, "atr": 0.02,
            "feats": vector, "prob": None, "rank": 1,
            "ts": 1_800_000_000.0, "feature_ts": 1_800_000_000.0,
            "prob_source": "ranking_exploration", "regime": "TEST",
            "entry_score": 75.0,
        })
        assert created and created["feature_count"] == feature_count, created
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
            assert row["feature_count"] == feature_count
            assert row["feature_schema_version"] == zq.DATA_VER
            assert len(names) == feature_count and names == zq.FEATURE_COLS
            assert saved.shape == (feature_count,) and np.array_equal(saved, vector)
        paper.close_tracked_signal(created["id"], 1_800_000_300.0,
                                   101.0, "TP", 0.009, 1.0, "TP")
        closed = paper.db.execute(
            "SELECT s.feats,s.feature_count,t.status,s.labeled,s.ret "
            "FROM signals s JOIN tracked_signals t ON t.signal_id=s.id "
            "WHERE s.id=?", (created["id"],)
        ).fetchone()
        assert closed["feature_count"] == feature_count
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

    def test_deep_network_training_produces_finite_checkpoint(self):
        script = r"""
import numpy as np
import zied_quant as zq
import deep_learning_evolution as ext
ext.install(zq)
if zq.torch is None:
    raise SystemExit(0)
rng = np.random.default_rng(7)
Xf = rng.normal(0, 0.5, size=(64, 32)).astype(np.float32)
yf = (rng.random(64) > 0.5).astype(np.float32)
Xe = rng.normal(0, 0.5, size=(24, 32)).astype(np.float32)
ye = (rng.random(24) > 0.5).astype(np.float32)
net = zq._train_net(Xf, yf, Xe, ye, epochs=3)
net.eval()
with zq.torch.no_grad():
    p = zq.torch.sigmoid(net(zq.torch.from_numpy(Xe))).numpy()
assert p.shape == (24, 1)
assert np.isfinite(p).all()
assert ((p >= 0) & (p <= 1)).all()
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\\n" + proc.stderr)

    def test_walk_forward_keeps_gbm_when_deep_network_training_fails(self):
        script = r"""
import numpy as np
import zied_quant as zq
import deep_learning_evolution as ext
ext.install(zq)

class StableFakeGBM:
    kind = "test-fallback"
    def fit(self, Xf, yf, Xe, ye):
        return True
    def predict(self, X):
        return np.linspace(0.2, 0.8, len(X), dtype=np.float64)

zq.GBM = StableFakeGBM
def fail_nn(*args, **kwargs):
    raise RuntimeError("simulated non-finite neural checkpoint")
zq._train_net = fail_nn

n = 900
rng = np.random.default_rng(11)
X = rng.normal(size=(n, zq.N_FEATURES)).astype(np.float64)
y = (np.arange(n) % 2).astype(np.float32)
ts = np.arange(n, dtype=np.float64) * 3600.0
ret = np.where(y > 0, 0.012, -0.009).astype(np.float64)
ds = {
    "X": X, "y": y, "ts": ts, "ret": ret,
    "hold": np.full(n, 4.0),
    "coin": np.asarray([f"COIN{i % 40}" for i in range(n)]),
}
rep = zq.walk_forward(ds, folds=3, epochs=2)
assert "gbm" in rep, rep
assert rep["gbm"]["n_oos"] >= 30, rep["gbm"]
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\\n" + proc.stderr)


if __name__ == "__main__":
    unittest.main()
