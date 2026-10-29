import os
import tempfile
import unittest

import numpy as np
import zied_quant as zq


class SignalOnlyTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = zq.CFG["DB"]
        zq.CFG["DB"] = os.path.join(self.tmp.name, "signals.db")
        self.paper = zq.Paper()

    def tearDown(self):
        self.paper.db.close()
        zq.CFG["DB"] = self.old_db
        self.tmp.cleanup()

    def test_logging_signal_does_not_create_virtual_wallet_or_position(self):
        self.paper.log_signals([{
            "inst": "TEST-USDT", "price": 100.0, "atr": 0.02,
            "feats": np.zeros(zq.N_FEATURES), "prob": 0.75,
            "rank": 1, "ts": 1000.0, "feature_ts": 1000.0,
        }])
        self.assertEqual(self.paper.count("signals"), 1)
        self.assertEqual(self.paper.count("signals", "labeled=0"), 1)
        self.assertFalse(hasattr(self.paper, "cash"))
        self.assertFalse(hasattr(self.paper, "open_trades"))
        self.assertFalse(hasattr(self.paper, "pending_orders"))
        self.assertTrue(hasattr(self.paper, "open_tracked_signals"))
        self.assertTrue(hasattr(self.paper, "close_tracked_signal"))

    def test_mature_signal_records_net_return_and_learning_label(self):
        self.paper.log_signals([{
            "inst": "TEST-USDT", "price": 100.0, "atr": 0.02,
            "feats": np.zeros(zq.N_FEATURES), "prob": 0.75,
            "rank": 1, "ts": 1000.0, "feature_ts": 1000.0,
        }])
        pending = self.paper.pending_signals(2000.0)
        self.assertEqual(len(pending), 1)
        self.paper.set_label(pending[0]["id"], 1, 0.0565, 6.0)

        ds = self.paper.signal_dataset()
        self.assertEqual(self.paper.count("signals", "labeled=1"), 1)
        self.assertEqual(len(ds["y"]), 1)
        self.assertEqual(ds["y"][0], 1.0)
        self.assertAlmostEqual(ds["ret"][0], 0.0565)
        self.assertAlmostEqual(ds["hold"][0], 6.0)
        self.assertAlmostEqual(ds["ts"][0], 1000.0)


if __name__ == "__main__":
    unittest.main()
