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
