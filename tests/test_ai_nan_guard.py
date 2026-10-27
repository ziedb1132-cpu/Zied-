"""Regression tests for fail-closed AI probability validation."""
import json
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zied_quant as zq


def fake_brain(probabilities):
    brain = zq.Brain(persist=False)
    brain.mode = "nn"
    brain.nn = object()
    brain.predict_all = lambda X: {"nn": np.asarray(probabilities, dtype=float)}
    return brain


class AINanGuardTests(unittest.TestCase):
    def test_paper_experience_migration_uses_entry_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(zq.CFG, {"DB": os.path.join(tmp, "paper.db")}):
                paper = zq.Paper()
                feats = json.dumps([0.0] * zq.N_FEATURES)
                paper.db.execute(
                    "INSERT INTO trades(inst,open_ts,feats,close_ts,status) VALUES(?,?,?,?,?)",
                    ("BTC-USDT", 1000.0, feats, 5000.0, "CLOSED"))
                paper.db.execute(
                    "INSERT INTO experience(ts,inst,feats,label,pnl,ret) VALUES(?,?,?,?,?,?)",
                    (5000.0, "BTC-USDT", feats, 1, 2.0, 0.02))
                paper.db.commit()
                paper._migrate()
                row = paper.db.execute("SELECT ts FROM experience").fetchone()
                self.assertEqual(row["ts"], 1000.0)
                paper.db.close()

    def test_get_universe_skips_tickers_with_invalid_prices(self):
        instruments = [
            {"instId": "BTC-USDT", "quoteCcy": "USDT", "baseCcy": "BTC", "state": "live"},
            {"instId": "EMPTY-USDT", "quoteCcy": "USDT", "baseCcy": "EMPTY", "state": "live"},
            {"instId": "BAD-USDT", "quoteCcy": "USDT", "baseCcy": "BAD", "state": "live"},
        ]
        tickers = [
            {"instId": "BTC-USDT", "volCcy24h": "5000000", "last": "65000", "askPx": "65001", "bidPx": "64999"},
            {"instId": "EMPTY-USDT", "volCcy24h": "5000000", "last": "", "askPx": "", "bidPx": ""},
            {"instId": "BAD-USDT", "volCcy24h": "nan", "last": "1", "askPx": "1", "bidPx": "1"},
        ]
        def fake_okx(path, params=None, **kwargs):
            if path.endswith("/public/instruments"):
                return instruments
            if path.endswith("/market/tickers"):
                return tickers
            return []
        with patch.object(zq, "okx_get", side_effect=fake_okx), patch.dict(zq.CFG, {"MIN_VOL_USDT": 300000}):
            result = zq.get_universe()
        self.assertEqual(list(result.index), ["BTC-USDT"])
        self.assertEqual(result.loc["BTC-USDT", "last"], 65000.0)

    def test_get_prices_skips_empty_and_invalid_okx_tickers(self):
        tickers = [
            {"instId": "BTC-USDT", "last": "65000"},
            {"instId": "EMPTY-USDT", "last": ""},
            {"instId": "NONE-USDT", "last": None},
            {"instId": "BAD-USDT", "last": "not-a-price"},
            {"instId": "NAN-USDT", "last": "nan"},
            {"instId": "INF-USDT", "last": "inf"},
            {"instId": "ZERO-USDT", "last": "0"},
        ]
        with patch.object(zq, "okx_get", return_value=tickers):
            self.assertEqual(zq.get_prices(), {"BTC-USDT": 65000.0})

    def test_ai_predict_accepts_finite_probabilities(self):
        brain = fake_brain([0.62, 0.81])
        result = brain.predict(np.zeros((2, zq.N_FEATURES), dtype=float))
        self.assertTrue(np.allclose(result, [0.62, 0.81]))

    def test_ai_predict_rejects_invalid_probabilities(self):
        for bad in ([np.nan], [np.inf], [-0.01], [1.01]):
            with self.subTest(bad=bad):
                brain = fake_brain(bad)
                with self.assertRaises(ValueError):
                    brain.predict(np.zeros((1, zq.N_FEATURES), dtype=float))

    def test_ai_predict_rejects_nonfinite_features(self):
        brain = fake_brain([0.7])
        X = np.zeros((1, zq.N_FEATURES), dtype=float)
        X[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "input contains"):
            brain.predict(X)

    def test_model_save_persists_only_selected_gbm_and_writes_metadata_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(zq.CFG, {
                "DIR": tmp, "MODEL": os.path.join(tmp, "model.pt"),
                "GBM": os.path.join(tmp, "gbm.pkl"),
                "META": os.path.join(tmp, "meta.json"),
            }):
                brain = zq.Brain(persist=False)
                brain.mode = "gbm"
                brain.gbm = {"simple": "picklable test model"}
                brain.nn = None
                brain.mu = np.zeros(zq.N_FEATURES)
                brain.sd = np.ones(zq.N_FEATURES)
                brain.thrs = {"gbm": 0.6}
                brain.auc = 0.61
                brain.info = {"n": 900}
                brain.save(zq.N_FEATURES)
                with open(os.path.join(tmp, "meta.json"), encoding="utf-8") as fh:
                    meta = json.load(fh)
                self.assertEqual(meta["mode"], "gbm")
                self.assertTrue(meta["has_gbm"])
                self.assertFalse(meta["has_nn"])
                self.assertFalse(os.path.exists(os.path.join(tmp, "model.pt")))
                with open(os.path.join(tmp, "gbm.pkl"), "rb") as fh:
                    self.assertEqual(pickle.load(fh), {"simple": "picklable test model"})
                self.assertEqual([n for n in os.listdir(tmp) if n.endswith(".tmp")], [])

    def test_model_save_rejects_invalid_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(zq.CFG, {
                "DIR": tmp, "MODEL": os.path.join(tmp, "model.pt"),
                "GBM": os.path.join(tmp, "gbm.pkl"),
                "META": os.path.join(tmp, "meta.json"),
            }):
                brain = zq.Brain(persist=False)
                brain.mode = "gbm"
                brain.gbm = {"simple": "test"}
                brain.mu = np.zeros(2)
                brain.sd = np.ones(2)
                with self.assertRaisesRegex(RuntimeError, "normalization"):
                    brain.save(zq.N_FEATURES)
                self.assertFalse(os.path.exists(os.path.join(tmp, "meta.json")))

    def test_load_removes_orphan_model_files_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(zq.CFG, {
                "DIR": tmp, "MODEL": os.path.join(tmp, "model.pt"),
                "GBM": os.path.join(tmp, "gbm.pkl"),
                "META": os.path.join(tmp, "meta.json"),
            }):
                with open(os.path.join(tmp, "model.pt"), "wb") as fh:
                    fh.write(b"stale model")
                with open(os.path.join(tmp, "gbm.pkl"), "wb") as fh:
                    fh.write(b"stale gbm")
                brain = zq.Brain(persist=True)
                self.assertFalse(brain.ready)
                self.assertFalse(os.path.exists(os.path.join(tmp, "model.pt")))
                self.assertFalse(os.path.exists(os.path.join(tmp, "gbm.pkl")))
                self.assertFalse(os.path.exists(os.path.join(tmp, "meta.json")))


if __name__ == "__main__":
    unittest.main()
