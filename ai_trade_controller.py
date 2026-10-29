#!/usr/bin/env python3
"""Use validated winning-pattern memory in live paper-trade ranking and AI scoring.

The 200-win threshold activates pattern discovery only; it never blocks trading.
Pattern memory is an auxiliary signal, not a substitute for model validation or risk gates.
"""
import json
import os

import numpy as np


PATTERN_FILE = "winning_patterns.json"
MAX_FEATURES = 40
MODEL_SCORE_ADJUSTMENT = 0.08


def _load_pattern_memory(zq):
    path = os.path.join(zq.CFG["DIR"], PATTERN_FILE)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if int(data.get("winning_samples", 0)) < 200 or int(data.get("losing_samples", 0)) < 30:
            return []
        if int(data.get("feature_count", 0)) != int(zq.N_FEATURES):
            return []
        known = {name: i for i, name in enumerate(zq.FEATURE_COLS)}
        patterns = []
        for p in data.get("top_recurring_features", [])[:MAX_FEATURES]:
            name = p.get("feature")
            if name not in known:
                continue
            try:
                wm = float(p["winner_median"])
                lm = float(p["loser_median"])
                wq = np.asarray(p["winner_q25_q75"], dtype=float)
                lq = np.asarray(p["loser_q25_q75"], dtype=float)
                effect = float(p.get("robust_effect", 0.0))
            except (TypeError, ValueError, KeyError):
                continue
            if (not np.isfinite([wm, lm, effect]).all() or wq.shape != (2,)
                    or lq.shape != (2,) or not np.isfinite(wq).all()
                    or not np.isfinite(lq).all() or effect <= 0):
                continue
            patterns.append((known[name], wm, lm, wq, lq, min(effect, 5.0)))
        return patterns
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []


def pattern_scores(zq, X):
    """Return a 0..1 relative match score; 0.5 means no directional evidence."""
    arr = np.asarray(X, dtype=np.float64)
    one = arr.ndim == 1
    if one:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != zq.N_FEATURES:
        return np.full(arr.shape[0] if arr.ndim else 1, 0.5)
    patterns = _load_pattern_memory(zq)
    if not patterns or not np.isfinite(arr).all():
        return np.full(len(arr), 0.5)
    evidence, weights = [], []
    for idx, wm, lm, wq, lq, effect in patterns:
        values = arr[:, idx]
        wscale = max(float(wq[1] - wq[0]), abs(wm) * 0.02, 1e-6)
        lscale = max(float(lq[1] - lq[0]), abs(lm) * 0.02, 1e-6)
        dw = np.abs(values - wm) / wscale
        dl = np.abs(values - lm) / lscale
        # Positive evidence when the current value is closer to winner conditions.
        e = 1.0 / (1.0 + np.exp(-np.clip(dl - dw, -6, 6)))
        in_winner_band = (values >= wq[0]) & (values <= wq[1])
        e = np.where(in_winner_band, np.maximum(e, 0.75), e)
        evidence.append(e)
        weights.append(effect)
    matrix = np.vstack(evidence)
    score = np.average(matrix, axis=0, weights=np.asarray(weights, dtype=float))
    return np.clip(score, 0.0, 1.0)


def install(zq):
    """Patch candidate ranking and validated AI inference to consume discovered patterns."""
    if getattr(zq, "_pattern_controller_installed", False):
        return
    original_rank = zq.rank_universe

    def rank_with_patterns(X, vol24):
        base = original_rank(X, vol24)
        try:
            ps = pattern_scores(zq, X.loc[base.index, zq.FEATURE_COLS].to_numpy())
            # A modest ranking influence helps matching setups enter the AI's candidate set.
            adjusted = base.to_numpy(dtype=float) + (ps - 0.5) * 0.40
            return __import__("pandas").Series(adjusted, index=base.index).sort_values(ascending=False)
        except Exception as exc:
            zq.log.warning("pattern ranking overlay unavailable: %s", exc)
            return base

    zq.rank_universe = rank_with_patterns

    original_predict = zq.Brain.predict

    def predict_with_patterns(self, X):
        probabilities = np.asarray(original_predict(self, X), dtype=float)
        try:
            ps = pattern_scores(zq, X)
            # Keep the validated model in control; pattern evidence can only make a small adjustment.
            adjusted = probabilities + (ps - 0.5) * MODEL_SCORE_ADJUSTMENT
            return np.clip(adjusted, 0.0, 1.0)
        except Exception as exc:
            zq.log.warning("pattern prediction overlay unavailable: %s", exc)
            return probabilities

    zq.Brain.predict = predict_with_patterns
    zq._pattern_controller_installed = True
