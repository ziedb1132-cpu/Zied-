"""Regression tests for fail-closed AI probability validation."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zied_quant as zq


def fake_brain(probabilities):
    brain = zq.Brain(persist=False)
    brain.mode = "nn"
    brain.nn = object()  # mark the model as available; inference is stubbed below
    brain.predict_all = lambda X: {"nn": np.asarray(probabilities, dtype=float)}
    return brain


def test_ai_predict_accepts_finite_probabilities():
    brain = fake_brain([0.62, 0.81])
    result = brain.predict(np.zeros((2, zq.N_FEATURES), dtype=float))
    assert np.allclose(result, [0.62, 0.81])


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-0.01], [1.01]])
def test_ai_predict_rejects_invalid_probabilities(bad):
    brain = fake_brain(bad)
    with pytest.raises(ValueError):
        brain.predict(np.zeros((1, zq.N_FEATURES), dtype=float))


def test_ai_predict_rejects_nonfinite_features():
    brain = fake_brain([0.7])
    X = np.zeros((1, zq.N_FEATURES), dtype=float)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="input contains"):
        brain.predict(X)



def test_model_save_persists_only_selected_gbm_and_writes_metadata_last(tmp_path, monkeypatch):
    monkeypatch.setitem(zq.CFG, "DIR", str(tmp_path))
    monkeypatch.setitem(zq.CFG, "MODEL", str(tmp_path / "model.pt"))
    monkeypatch.setitem(zq.CFG, "GBM", str(tmp_path / "gbm.pkl"))
    monkeypatch.setitem(zq.CFG, "META", str(tmp_path / "meta.json"))

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

    import json
    import pickle
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["mode"] == "gbm"
    assert meta["has_gbm"] is True
    assert meta["has_nn"] is False
    assert not (tmp_path / "model.pt").exists()
    with (tmp_path / "gbm.pkl").open("rb") as fh:
        assert pickle.load(fh) == {"simple": "picklable test model"}
    assert not list(tmp_path.glob("*.tmp"))


def test_model_save_rejects_invalid_normalization(tmp_path, monkeypatch):
    monkeypatch.setitem(zq.CFG, "DIR", str(tmp_path))
    monkeypatch.setitem(zq.CFG, "MODEL", str(tmp_path / "model.pt"))
    monkeypatch.setitem(zq.CFG, "GBM", str(tmp_path / "gbm.pkl"))
    monkeypatch.setitem(zq.CFG, "META", str(tmp_path / "meta.json"))
    brain = zq.Brain(persist=False)
    brain.mode = "gbm"
    brain.gbm = {"simple": "test"}
    brain.mu = np.zeros(2)
    brain.sd = np.ones(2)
    with pytest.raises(RuntimeError, match="normalization"):
        brain.save(zq.N_FEATURES)
    assert not (tmp_path / "meta.json").exists()
