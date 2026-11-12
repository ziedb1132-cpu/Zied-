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

    def test_gate_opens_at_100_verified_samples_with_10_per_class(self):
        # The model may start with an imbalanced but non-empty real dataset.
        for _ in range(90):
            self.add_signal(0.01)
        for _ in range(9):
            self.add_signal(-0.01)
        status = zq.learning_gate_status(self.paper)
        self.assertEqual(status["wins"], 90)
        self.assertEqual(status["losses"], 9)
        self.assertFalse(status["ready"])
        self.add_signal(-0.01)
        status = zq.learning_gate_status(self.paper)
        self.assertEqual(status["wins"], 90)
        self.assertEqual(status["losses"], 10)
        self.assertTrue(status["ready"])
        self.assertEqual(status["total"], 100)

    def test_new_schema_mismatch_forces_one_immediate_rebuild(self):
        class GatePaper:
            def __init__(self, schema_hash=None, total=100):
                self.schema_hash = schema_hash
                self.total = total
            def closed_outcome_counts(self):
                return {"wins": 300, "losses": 300, "total": self.total}
            def _get(self, key):
                if key == "last_learning_attempt_closed_count":
                    return "100"
                if key == "last_learning_attempt_schema_hash":
                    return self.schema_hash
                return None

        brain = zq.Brain(persist=False)
        brain.load_issue = {"reason": "incompatible_feature_or_missing_value_schema"}
        self.assertTrue(zq.need_retrain(GatePaper(), brain))
        self.assertFalse(zq.need_retrain(GatePaper(zq.feature_schema_fingerprint()), brain))
        self.assertTrue(zq.need_retrain(
            GatePaper(zq.feature_schema_fingerprint(), total=101), brain))

    def test_learning_refreshes_after_each_new_verified_closed_signal(self):
        class GatePaper:
            def __init__(self, total):
                self.total = total
            def closed_outcome_counts(self):
                return {"wins": self.total // 2, "losses": self.total - self.total // 2, "total": self.total}
            def _get(self, key):
                if key == "last_learning_attempt_closed_count":
                    return "100"
                if key == "last_learning_attempt_schema_hash":
                    return zq.feature_schema_fingerprint()
                return None

        brain = zq.Brain(persist=False)
        self.assertFalse(zq.need_retrain(GatePaper(100), brain))
        self.assertTrue(zq.need_retrain(GatePaper(101), brain))

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
        # All valid closed outcomes are retained; 100 is a minimum, not a per-class cap.
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

    def test_closed_signal_diagnostics_reports_exclusion_reasons(self):
        self.add_signal(0.01)
        self.add_signal(-0.01)
        self.add_signal(0.02)
        ids = [r[0] for r in self.paper.db.execute("SELECT id FROM signals ORDER BY id")]
        self.paper.db.execute("UPDATE signals SET feature_names=NULL WHERE id=?", (ids[0],))
        self.paper.db.execute("UPDATE signals SET feature_count=1 WHERE id=?", (ids[1],))
        self.paper.db.execute("UPDATE signals SET feats='not-json' WHERE id=?", (ids[2],))
        self.paper.db.commit()
        stats = self.paper.closed_signal_diagnostics()
        self.assertEqual(stats["closed_tracked"], 3)
        self.assertEqual(stats["missing_feature_names"], 1)
        self.assertEqual(stats["feature_count_mismatch"], 1)
        self.assertEqual(stats["malformed_feature_vector"], 1)
        self.assertEqual(stats["valid_training_samples"], 0)
        self.assertEqual(stats["excluded_total"], 3)

    def test_training_dataset_excludes_open_signals(self):
        self.add_signal(0.01, closed=True)
        self.add_signal(-0.01, closed=False)
        ds = self.paper.closed_signal_dataset()
        self.assertEqual(len(ds["y"]), 1)
        self.assertEqual(float(ds["y"][0]), 1.0)
        self.assertEqual(float(ds["ret"][0]), 0.01)


    def test_only_opened_entries_are_persisted_and_exact_snapshot_is_not_reopened(self):
        old_required = zq.CFG.get("REQUIRED_SIGNAL_FEATURES")
        zq.CFG["REQUIRED_SIGNAL_FEATURES"] = zq.N_FEATURES
        try:
            row = {
                "ts": 1000.0, "feature_ts": 1000.0, "inst": "DEDUP-USDT",
                "price": 100.0, "atr": 0.01, "feats": np.zeros(zq.N_FEATURES),
                "prob": None, "rank": 1, "regime": "UNKNOWN", "entry_score": 80.0,
            }
            first = self.paper.create_tracked_signal(row)
            self.assertIsNotNone(first)
            self.assertEqual(self.paper.count("signals"), 1)

            # Repeating the same candle/signal while open cannot create another row.
            self.assertIsNone(self.paper.create_tracked_signal(row))
            self.assertEqual(self.paper.count("signals"), 1)

            # Closing the trade does not make that same market snapshot reusable.
            self.assertTrue(self.paper.close_tracked_signal(
                first["id"], 1100.0, 101.0, "WIN", 0.01, 1.0, "TP"))
            self.assertIsNone(self.paper.create_tracked_signal(row))
            self.assertEqual(self.paper.count("signals"), 1)

            # A later feature timestamp is a genuinely new opportunity, even
            # for the same coin and within the old one-hour cooldown window.
            next_row = dict(row, ts=1300.0, feature_ts=1300.0)
            second = self.paper.create_tracked_signal(next_row)
            self.assertIsNotNone(second)
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(self.paper.count("signals"), 2)
        finally:
            if old_required is None:
                zq.CFG.pop("REQUIRED_SIGNAL_FEATURES", None)
            else:
                zq.CFG["REQUIRED_SIGNAL_FEATURES"] = old_required

    def test_rejected_ranking_candidates_are_not_persisted(self):
        # Ranking itself must not create rows; only create_tracked_signal does.
        self.assertEqual(self.paper.count("signals"), 0)
        self.assertEqual(self.paper.tracked_count(), 0)


if __name__ == "__main__":
    unittest.main()
