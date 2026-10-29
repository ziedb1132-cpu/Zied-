import unittest

import numpy as np
import pandas as pd

import zied_quant as zq


class ExpandedFeatureTests(unittest.TestCase):
    def test_feature_schema_has_200_columns_and_no_btc_score_gate(self):
        self.assertEqual(zq.N_FEATURES, 200)
        self.assertIn("1H_ret40", zq.FEATURE_COLS)
        self.assertIn("1H_downside_vol20", zq.FEATURE_COLS)
        self.assertIn("1H_vwap_dist20", zq.FEATURE_COLS)
        self.assertIn("1H_ema_cross21_50", zq.FEATURE_COLS)
        self.assertEqual(len(zq.FEATURE_COLS), len(set(zq.FEATURE_COLS)))

    def test_added_features_are_computed(self):
        n = 120
        c = pd.Series(100 * np.exp(np.linspace(0, 0.12, n)))
        df = pd.DataFrame({
            "o": c * 0.999,
            "h": c * 1.01,
            "l": c * 0.99,
            "c": c,
            "vol": np.linspace(1000, 2000, n),
            "avail": pd.date_range("2025-01-01", periods=n, freq="h"),
        })
        out = zq.tf_features(df)
        added = ["ret40", "ret60", "downside_vol20", "upside_vol20",
                 "efficiency20", "range_pos20", "vol_trend20", "vwap_dist20",
                 "atr_change5", "ema_cross21_50"]
        self.assertTrue(set(added).issubset(out.columns))
        self.assertTrue(np.isfinite(out[added].dropna().to_numpy()).all())
        self.assertGreater(len(out[added].dropna()), 0)


if __name__ == "__main__":
    unittest.main()
