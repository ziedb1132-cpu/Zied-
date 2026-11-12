"""Regression tests for dynamically extended historical feature schemas."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import zied_quant as zq


class DynamicFeatureAlignmentTests(unittest.TestCase):
    def test_dynamic_news_and_event_columns_are_preserved_as_unknown(self):
        original_tfs, original_base = zq.TFS, zq.BASE
        original_cols = zq.FEATURE_COLS
        original_n = zq.N_FEATURES
        try:
            zq.TFS = ["1H"]
            zq.BASE = "1H"
            zq.FEATURE_COLS = [
                "1H_ret1",
                "ctx_hour_sin",
                "global_news_sentiment",
                "upcoming_event_risk",
            ]
            zq.N_FEATURES = len(zq.FEATURE_COLS)

            index = pd.date_range("2026-01-01", periods=3, freq="h")
            technical = pd.DataFrame({"1H_ret1": [0.1, 0.2, 0.3]}, index=index)
            context = pd.DataFrame({"ctx_hour_sin": [0.4, 0.5, 0.6]}, index=index)

            with patch.object(zq, "all_tf_features", return_value={"1H": technical}), \
                 patch.object(zq, "ctx_features", return_value=context):
                result = zq.aligned_matrix({"1H": pd.DataFrame()}, {})

            self.assertEqual(result.columns.tolist(), zq.FEATURE_COLS)
            np.testing.assert_allclose(result["1H_ret1"].to_numpy(), [0.1, 0.2, 0.3])
            np.testing.assert_allclose(result["ctx_hour_sin"].to_numpy(), [0.4, 0.5, 0.6])
            self.assertTrue(result["global_news_sentiment"].isna().all())
            self.assertTrue(result["upcoming_event_risk"].isna().all())
        finally:
            zq.TFS = original_tfs
            zq.BASE = original_base
            zq.FEATURE_COLS = original_cols
            zq.N_FEATURES = original_n


if __name__ == "__main__":
    unittest.main()
