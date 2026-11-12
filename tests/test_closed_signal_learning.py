"""Tests for the closed-signal-only learning gate."""
import json
import os
import tempfile
import unittest

import numpy as np

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

    def add_signal(self, ret, closed=True, valid_features=True, unknown_feature=False):
        values = [0.0] * zq.N_FEATURES
        if unknown_feature and valid_features:
            values[0] = None
        feats = json.dumps(values) if valid_features else "[]"
        feature_names = json.dumps(list(zq.FEATURE_COLS), separators=(",", ":")) if valid_features else "[]"
        cur = self.paper.db.execute(
            "INSERT INTO signals(ts,inst,price,atr,feats,prob,rank,labeled,label,ret,hold,feature_ts,"
            "feature_count,feature_names,feature_schema_version) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (1000.0, f"COIN{self.paper.db.execute('SELECT COUNT(*) FROM signals').fetchone()[0]}-USDT",
             100.0, 0.01, feats, 0.6, 1, 1, int(ret > 0), ret, 1.0, 1000.0,
             zq.N_FEATURES if valid_features else 0, feature_names, zq.DATA_VER))
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

    def test_nan_is_retained_as_unknown_and_encoded_separately_from_zero(self):
        values = np.zeros((1, zq.N_FEATURES), dtype=np.float64)
        values[0, 0] = np.nan
        self.assertTrue(zq.feature_coverage_mask(values, 0.80)[0])
        brain = zq.Brain(persist=False)
        brain.mu = np.zeros(zq.N_FEATURES, dtype=np.float64)
        brain.sd = np.ones(zq.N_FEATURES, dtype=np.float64)
        encoded = brain._norm(values)
        self.assertEqual(encoded.shape, (1, zq.N_FEATURES * 2))
        self.assertEqual(encoded[0, 0], 0.0)
        self.assertEqual(encoded[0, zq.N_FEATURES], 1.0)
        self.assertEqual(encoded[0, zq.N_FEATURES + 1], 0.0)

    def test_timeout_exit_is_classified_by_net_return_after_costs(self):
        # A small gross gain below the assumed costs is a net loss.
        ret, outcome = zq.classify_exit_return(100.0, 100.2)
        self.assertLess(ret, 0.0)
        self.assertEqual(outcome, "LOSS")

        # A larger close is a net win after costs.
        ret, outcome = zq.classify_exit_return(100.0, 101.0)
        self.assertGreater(ret, 0.0)
        self.assertEqual(outcome, "WIN")

    def test_gate_uses_total_verified_samples_not_500_per_class(self):
        # The model may start with an imbalanced but non-empty real dataset.
        for _ in range(590):
            self.add_signal(0.01)
        for _ in range(9):
            self.add_signal(-0.01)
        status = zq.learning_gate_status(self.paper)
        self.assertEqual(status["wins"], 590)
        self.assertEqual(status["losses"], 9)
        self.assertFalse(status["ready"])
        self.add_signal(-0.01)
        status = zq.learning_gate_status(self.paper)
        self.assertEqual(status["wins"], 590)
        self.assertEqual(status["losses"], 10)
        self.assertTrue(status["ready"])
        self.assertEqual(status["total"], 600)

    def test_adaptive_memory_learning_starts_with_20_real_closed_outcomes(self):
        for _ in range(10):
            self.add_signal(0.01)
        for _ in range(10):
            self.add_signal(-0.01)
        self.paper.db.execute("UPDATE tracked_signals SET regime='trend'")
        self.paper.db.commit()
        weights, samples, active = self.paper.adaptive_ranking_weights("trend")
        self.assertEqual(samples, 20)
        self.assertTrue(active)
        self.assertEqual(set(weights), {
            "ret1h", "ret4h", "ret1d", "ret15m", "trend",
            "volume_ratio", "adx", "obv", "liquidity",
            "atr_penalty", "overbought_penalty",
        })

    def test_old_feature_schema_is_aligned_by_name_and_new_features_remain_unknown(self):
        old_names = list(zq.FEATURE_COLS[:-3])
        old_values = [float(i + 1) for i in range(len(old_names))]
        row = {
            "feats": json.dumps(old_values),
            "feature_names": json.dumps(old_names),
            "feature_count": len(old_names),
            "ts": 1000.0,
            "inst": "LEGACY-USDT",
            "label": 1,
            "ret": 0.01,
            "hold": 1.0,
        }
        ds = zq._rows_to_ds([row])
        self.assertEqual(ds["X"].shape, (1, zq.N_FEATURES))
        self.assertTrue(np.allclose(ds["X"][0, :-3], old_values))
        self.assertTrue(np.isnan(ds["X"][0, -3:]).all())
        self.assertTrue(zq.feature_coverage_mask(ds["X"], 0.80)[0])

    def test_legacy_feature_vectors_without_name_manifest_are_not_misinterpreted(self):
        row = {
            "feats": json.dumps([0.0] * zq.N_FEATURES),
            "ts": 1000.0,
            "inst": "UNKNOWN-SCHEMA-USDT",
            "label": 1,
            "ret": 0.01,
            "hold": 1.0,
        }
        ds = zq._rows_to_ds([row])
        self.assertEqual(len(ds["y"]), 0)

    def test_training_dataset_keeps_all_outcomes_above_threshold(self):
        # The 500/500 rule is a minimum gate, not a per-class cap.
        self.add_signal(0.01, unknown_feature=True)
        for _ in range(500):
            self.add_signal(0.01)
        for _ in range(503):
            self.add_signal(-0.01)
        status = zq.learning_gate_status(self.paper)
        self.assertTrue(status["ready"])
        ds = self.paper.closed_signal_dataset()
        self.assertEqual(len(ds["y"]), 1004)
        self.assertEqual(int((ds["y"] == 1).sum()), 501)
        self.assertEqual(int((ds["y"] == 0).sum()), 503)
        train = zq.collect_training_data(self.paper)
        self.assertEqual(len(train["y"]), 1004)
        self.assertTrue(np.isnan(train["X"][0, 0]))

    def test_training_dataset_excludes_open_signals(self):
        self.add_signal(0.01, closed=True)
        self.add_signal(-0.01, closed=False)
        ds = self.paper.closed_signal_dataset()
        self.assertEqual(len(ds["y"]), 1)
        self.assertEqual(float(ds["y"][0]), 1.0)
        self.assertEqual(float(ds["ret"][0]), 0.01)


if __name__ == "__main__":
    unittest.main()
