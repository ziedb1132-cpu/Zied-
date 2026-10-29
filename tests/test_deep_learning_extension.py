"""Isolated regression tests for the runtime 300-feature/30-layer extension."""
import subprocess
import sys
import unittest


class DeepLearningExtensionTests(unittest.TestCase):
    def test_runtime_schema_features_and_network_depth(self):
        script = r"""
import numpy as np
import pandas as pd
import zied_quant as zq
import deep_learning_evolution as ext
ext.install(zq)
assert zq.N_FEATURES == 300, zq.N_FEATURES
assert len(zq.FEATURE_COLS) == 300
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
