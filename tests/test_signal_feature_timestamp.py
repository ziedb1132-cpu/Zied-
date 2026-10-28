import json
import os
import tempfile
import unittest

import numpy as np
import zied_quant as zq


class SignalFeatureTimestampTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = zq.CFG["DB"]
        zq.CFG["DB"] = os.path.join(self.tmp.name, "paper.db")
        self.paper = zq.Paper()

    def tearDown(self):
        self.paper.db.close()
        zq.CFG["DB"] = self.old_db
        self.tmp.cleanup()

    def test_signal_dataset_uses_feature_timestamp_not_scan_time(self):
        self.paper.log_signals([{
            "inst": "TEST-USDT",
            "price": 10.0,
            "atr": 0.01,
            "feats": np.zeros(zq.N_FEATURES, dtype=np.float32),
            "prob": None,
            "rank": 1,
            "feature_ts": 1700000000.0,
        }])
        row = self.paper.db.execute("SELECT id, ts, feature_ts FROM signals").fetchone()
        self.assertGreater(row["ts"], 1700000000.0)
        self.assertEqual(row["feature_ts"], 1700000000.0)
        self.paper.set_label(row["id"], 1, 0.02, 3.0)
        ds = self.paper.signal_dataset()
        self.assertEqual(float(ds["ts"][0]), 1700000000.0)

    def test_pending_signal_keeps_creation_time_for_maturity_but_exposes_feature_time(self):
        self.paper.log_signals([{
            "inst": "TEST-USDT",
            "price": 10.0,
            "atr": 0.01,
            "feats": np.zeros(zq.N_FEATURES, dtype=np.float32),
            "prob": None,
            "rank": 1,
            "feature_ts": 1700000000.0,
        }])
        row = self.paper.db.execute("SELECT ts FROM signals").fetchone()
        pending = self.paper.pending_signals(float(row["ts"]) + 1)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["ts"], row["ts"])
        self.assertEqual(pending[0]["feature_ts"], 1700000000.0)


if __name__ == "__main__":
    unittest.main()
