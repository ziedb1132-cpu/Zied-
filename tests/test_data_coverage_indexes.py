"""Regression tests for resilient market-data coverage and SQLite query paths."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import zied_quant as zq


def candles(n=80):
    close = 100.0 + np.arange(n, dtype=float) * 0.1
    frame = pd.DataFrame({
        "ts": np.arange(n, dtype=np.int64) * 3_600_000,
        "o": close - 0.03,
        "h": close + 0.10,
        "l": close - 0.10,
        "c": close,
        "vol": np.full(n, 1000.0),
        "avail": pd.date_range("2026-01-01", periods=n, freq="h"),
    })
    return frame


class DataCoverageAndIndexTests(unittest.TestCase):
    def test_one_missing_non_base_timeframe_is_retained_as_partial_data(self):
        frame = candles()

        def fake_get(inst, tf, depth):
            return None if tf == "1D" else frame

        with patch.object(zq, "get_candles", side_effect=fake_get):
            result = zq.fetch_frames("TEST-USDT", 80)

        self.assertIsNotNone(result)
        self.assertIn(zq.BASE, result)
        self.assertNotIn("1D", result)

    def test_two_missing_timeframes_reject_symbol_instead_of_fabricating_data(self):
        frame = candles()

        def fake_get(inst, tf, depth):
            return None if tf in ("1D", "4H") else frame

        with patch.object(zq, "get_candles", side_effect=fake_get):
            result = zq.fetch_frames("TEST-USDT", 80)

        self.assertIsNone(result)

    def test_missing_timeframe_features_are_nan_not_copied_from_another_frame(self):
        frame = candles()
        features = zq.all_tf_features({"1H": frame})

        missing_col = f"5m_{zq.NAMES[0]}"
        self.assertTrue(features["5m"][missing_col].isna().all())
        self.assertEqual(len(features["5m"]), len(frame))
        self.assertTrue(features["1H"].notna().any().any())

    def test_live_vector_keeps_appended_news_and_event_columns_as_unknown(self):
        original_features = list(zq.FEATURE_COLS)
        extra = ["global_news_sentiment", "upcoming_event_impact"]
        expanded_features = original_features + extra
        frames = {"1H": candles()}
        frame_features = {
            tf: pd.DataFrame([{name: 0.25 for name in zq.TF_COLS
                               if name.startswith(f"{tf}_")}])
            for tf in zq.TFS
        }
        context = pd.DataFrame([{name: 0.0 for name in zq.CTX_COLS}])
        with patch.object(zq, "FEATURE_COLS", expanded_features):
            with patch.object(zq, "all_tf_features", return_value=frame_features):
                with patch.object(zq, "ctx_features", return_value=context):
                    vector = zq.live_vector(frames, {"1H": candles(), "4H": candles()})
        self.assertEqual(list(vector.index), expanded_features)
        self.assertTrue(np.isnan(vector[extra].to_numpy(dtype=float)).all())

    def test_required_sqlite_indexes_are_created_on_existing_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "paper.db")
            with patch.dict(zq.CFG, {"DB": db_path}):
                paper = zq.Paper()
                indexes = {
                    row["name"] for row in paper.db.execute(
                        "SELECT name FROM sqlite_master WHERE type='index'"
                    )
                }
                self.assertTrue({
                    "idx_signals_label_time",
                    "idx_signals_inst_time",
                    "idx_tracked_closed_time",
                    "idx_tracked_open_inst",
                }.issubset(indexes))
                paper.db.close()


if __name__ == "__main__":
    unittest.main()
