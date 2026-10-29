"""Tests for the closed-signal-only learning gate."""
import json
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zied_quant as zq


class ClosedSignalLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: zq.CFG[k] for k in ("DB", "DIR")}
        zq.CFG["DIR"] = self.tmp.name
        zq.CFG["DB"] = os.path.join(self.tmp.name, "paper.db")
        self.paper = zq.Paper()

    def tearDown(self):
        self.paper.db.close()
        for k, v in self.old.items():
            zq.CFG[k] = v
        self.tmp.cleanup()

    def add_signal(self, ret, closed=True, valid_features=True):
        feats = json.dumps([0.0] * zq.N_FEATURES) if valid_features else "[]"
        cur = self.paper.db.execute(
            "INSERT INTO signals(ts,inst,price,atr,feats,prob,rank,labeled,label,ret,hold,feature_ts) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (1000.0, f"COIN{self.paper.db.execute('SELECT COUNT(*) FROM signals').fetchone()[0]}-USDT",
             100.0, 0.01, feats, 0.6, 1, 1, int(ret > 0), ret, 1.0, 1000.0))
        sid = cur.lastrowid
        self.paper.db.execute(
            "INSERT INTO tracked_signals(signal_id,inst,opened_ts,entry,sl,tp,atr_pct,prob,rank,feats,"
            "status,closed_ts,exit_price,outcome,ret,hold,close_reason) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, f"COIN{sid}-USDT", 1000.0, 100.0, 99.0, 102.0, 0.01, 0.6, 1,
             feats, "CLOSED" if closed else "OPEN", 1100.0 if closed else None,
             101.0 if ret > 0 else 99.0, "WIN" if ret > 0 else "LOSS",
             ret, 1.0, "TP" if ret > 0 else "SL"))
        self.paper.db.commit()

    def test_gate_requires_500_closed_wins_and_500_closed_losses(self):
        for _ in range(500):
            self.add_signal(0.01)
        for _ in range(499):
            self.add_signal(-0.01)
        status = zq.learning_gate_status(self.paper)
        self.assertEqual(status["wins"], 500)
        self.assertEqual(status["losses"], 499)
        self.assertFalse(status["ready"])
        self.add_signal(-0.01)
        status = zq.learning_gate_status(self.paper)
        self.assertEqual(status["wins"], 500)
        self.assertEqual(status["losses"], 500)
        self.assertTrue(status["ready"])

    def test_training_dataset_excludes_open_signals(self):
        self.add_signal(0.01, closed=True)
        self.add_signal(-0.01, closed=False)
        ds = self.paper.closed_signal_dataset()
        self.assertEqual(len(ds["y"]), 1)
        self.assertEqual(float(ds["y"][0]), 1.0)
        self.assertEqual(float(ds["ret"][0]), 0.01)


if __name__ == "__main__":
    unittest.main()
