#!/usr/bin/env python3
"""Guarded diagnostics for the seven Zied Quant intelligence improvements.

This module is deliberately observational: it never changes model weights,
probabilities, entry thresholds, exit policies, or the 500/500 learning gate.
"""
import json
import os
import time

import numpy as np


def _write_json(directory, filename, payload):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp, path)
    return path


def calibration_metrics(y_true, probabilities, bins=10):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0.0, 1.0])
    y, p = y[valid], np.clip(p[valid], 1e-7, 1.0 - 1e-7)
    if len(y) < 30:
        return {"status": "insufficient_gate_samples", "n": int(len(y))}
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows, ece = [], 0.0
    for i in range(len(edges) - 1):
        mask = ((p >= edges[i]) & (p < edges[i + 1])
                if i < len(edges) - 2 else
                (p >= edges[i]) & (p <= edges[i + 1]))
        if not mask.any():
            continue
        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        count = int(mask.sum())
        ece += count / len(y) * abs(confidence - observed)
        rows.append({"bin_low": float(edges[i]), "bin_high": float(edges[i + 1]),
                     "count": count, "mean_probability": confidence,
                     "observed_win_rate": observed})
    return {"status": "ok", "n": int(len(y)),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
            "expected_calibration_error": float(ece), "bins": rows}


def _permutation_importance(zq, brain, fit_snapshot, max_rows=256):
    X = np.asarray(fit_snapshot.get("X_gate"), dtype=np.float64)
    y = np.asarray(fit_snapshot.get("gate_y"), dtype=np.float64)
    if X.ndim != 2 or len(X) != len(y) or X.shape[1] != zq.N_FEATURES or len(y) < 100:
        return {"status": "insufficient_or_missing_gate_features", "feature_count": int(zq.N_FEATURES)}
    # Deterministic, bounded holdout sample keeps the hourly runner predictable.
    if len(y) > max_rows:
        take = np.linspace(0, len(y) - 1, max_rows).astype(int)
        X, y = X[take], y[take]
    base = np.asarray(brain.predict(X), dtype=np.float64)
    base_auc = float(zq.auc_score(y, base))
    rng = np.random.RandomState(1729)
    rows = []
    for j, name in enumerate(zq.FEATURE_COLS):
        permuted = X.copy()
        permuted[:, j] = permuted[rng.permutation(len(permuted)), j]
        p = np.asarray(brain.predict(permuted), dtype=np.float64)
        drop = base_auc - float(zq.auc_score(y, p))
        rows.append({"feature": str(name), "permutation_auc_drop": float(drop)})
    rows.sort(key=lambda item: item["permutation_auc_drop"], reverse=True)
    return {"status": "ok", "method": "permutation_importance_on_untouched_gate",
            "n": int(len(y)), "feature_count": int(X.shape[1]), "baseline_auc": base_auc,
            "top_features": rows[:30],
            "warning": "Associational diagnostic only; not causal proof and not used to alter the model."}


def post_fit_diagnostics(zq, brain, info):
    snapshot = getattr(brain, "last_fit", {}) or {}
    directory = zq.CFG["DIR"]
    calibration = calibration_metrics(snapshot.get("gate_y", []), snapshot.get("gate_p", []))
    calibration.update({"updated_at": time.time(), "candidate_accepted": bool(info.get("accepted"))})
    _write_json(directory, "model_calibration.json", calibration)

    importance = {"status": "skipped_candidate_not_promoted"}
    if info.get("accepted") and brain.ready:
        importance = _permutation_importance(zq, brain, snapshot)
    importance.update({"updated_at": time.time(), "candidate_accepted": bool(info.get("accepted"))})
    _write_json(directory, "feature_importance.json", importance)

    zq.log.info("model calibration diagnostics: %s", {
        k: calibration.get(k) for k in ("status", "n", "brier", "log_loss", "expected_calibration_error")
    })
    zq.log.info("feature importance diagnostics: status=%s features=%d",
                importance.get("status"), len(importance.get("top_features", [])))
    return {"calibration": calibration, "feature_importance": importance}


def feature_drift_report(zq, brain, feature_frame):
    if not brain.ready or brain.mu is None or brain.sd is None or feature_frame is None or feature_frame.empty:
        return {"status": "model_or_features_unavailable"}
    X = np.asarray(feature_frame[zq.FEATURE_COLS], dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != zq.N_FEATURES or not np.isfinite(X).all():
        return {"status": "invalid_feature_matrix"}
    mu = np.asarray(brain.mu, dtype=np.float64)
    sd = np.maximum(np.asarray(brain.sd, dtype=np.float64), 1e-8)
    zmean = (X.mean(axis=0) - mu) / sd
    spread = X.std(axis=0) / sd
    drifted = np.abs(zmean) >= 2.0
    unstable = (spread >= 2.0) | (spread <= 0.5)
    rows = []
    for i, name in enumerate(zq.FEATURE_COLS):
        if drifted[i] or unstable[i]:
            rows.append({"feature": str(name), "mean_shift_sigma": float(zmean[i]),
                         "std_ratio": float(spread[i])})
    rows.sort(key=lambda item: max(abs(item["mean_shift_sigma"]), abs(np.log(max(item["std_ratio"], 1e-8)))),
              reverse=True)
    result = {"status": "ok", "updated_at": time.time(), "n": int(len(X)),
              "feature_count": int(X.shape[1]), "drifted_feature_count": int(drifted.sum()),
              "unstable_scale_feature_count": int(unstable.sum()),
              "drift_fraction": float(drifted.mean()),
              "top_drifted_features": rows[:30],
              "warning": "Monitoring only; drift does not disable trading or change thresholds."}
    _write_json(zq.CFG["DIR"], "feature_drift.json", result)
    return result


def regime_outcome_report(paper):
    rows = paper.db.execute(
        "SELECT regime, COUNT(*) AS n, AVG(ret) AS avg_ret, "
        "SUM(CASE WHEN ret>0 THEN ret ELSE 0 END) AS gp, "
        "SUM(CASE WHEN ret<0 THEN -ret ELSE 0 END) AS gl "
        "FROM tracked_signals WHERE status='CLOSED' AND ret IS NOT NULL "
        "GROUP BY regime ORDER BY n DESC"
    ).fetchall()
    out = []
    for row in rows:
        gp, gl = float(row["gp"] or 0.0), float(row["gl"] or 0.0)
        out.append({"regime": row["regime"], "n": int(row["n"]),
                    "avg_net_return": float(row["avg_ret"] or 0.0),
                    "profit_factor": gp / gl if gl > 0 else None})
    return out


def portfolio_correlation_filter(zq, paper, candidate, threshold=0.85, cache=None):
    """Reject only strongly correlated open-position overlap; unknown data fails open."""
    opened = paper.open_tracked_signals()
    open_coins = sorted({str(row["inst"]) for row in opened if str(row["inst"]) != str(candidate)})
    if not open_coins:
        return {"allowed": True, "max_abs_correlation": 0.0, "compared": 0}
    cache = {} if cache is None else cache
    best, compared = 0.0, 0
    try:
        for coin in [candidate] + open_coins:
            if coin not in cache:
                cache[coin] = zq._hourly_returns(coin)
        a = cache.get(candidate)
        if a is None:
            return {"allowed": True, "max_abs_correlation": None, "compared": 0,
                    "reason": "candidate_returns_unavailable"}
        for coin in open_coins:
            b = cache.get(coin)
            if b is None:
                continue
            joined = __import__("pandas").concat([a, b], axis=1, join="inner").tail(72).dropna()
            if len(joined) < 30:
                continue
            corr = float(joined.corr().iloc[0, 1])
            if np.isfinite(corr):
                best = max(best, abs(corr))
                compared += 1
    except Exception as exc:
        zq.log.warning("portfolio correlation check unavailable for %s: %s", candidate, exc)
        return {"allowed": True, "max_abs_correlation": None, "compared": compared,
                "reason": "correlation_check_error"}
    return {"allowed": best < float(threshold), "max_abs_correlation": best,
            "compared": compared, "threshold": float(threshold)}


def install(zq):
    """Attach read-only drift monitoring to the live scan feature matrix."""
    zq.FEATURE_DRIFT_HOOK = lambda brain, frame: feature_drift_report(zq, brain, frame)
