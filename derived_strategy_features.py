#!/usr/bin/env python3
"""Versioned, causal derived features from independently validated symbolic strategies.

A strategy is a feature candidate, not a trading instruction. Only explicitly
approved records are emitted into the feature matrix. Expressions are restricted
to feature comparisons (no eval, arbitrary code, or future candle references).
"""
import hashlib
import json
import os
import re
import tempfile

SCHEMA_VERSION = 1
FEATURE_PREFIX = "derived_strategy_"
_ALLOWED_OPS = {"gt", "lt"}


def strategy_feature_name(spec):
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return FEATURE_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def validate_spec(spec, known_features):
    if not isinstance(spec, dict) or spec.get("kind") != "symbolic_rule":
        raise ValueError("Unsupported strategy spec")
    conditions = spec.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Strategy must contain conditions")
    for cond in conditions:
        if (not isinstance(cond, dict)
                or cond.get("feature") not in known_features
                or cond.get("op") not in _ALLOWED_OPS
                or not isinstance(cond.get("threshold"), (int, float))):
            raise ValueError("Invalid strategy condition")
    return True


def approve_record(spec, evidence, closed_wins, closed_losses, known_features):
    """Return an approved feature record only after the requested evidence gates.

    evidence must contain independent chronological train/validation/holdout
    results; holdout must be positive and validation must not be used as holdout.
    Global win/loss counts remain the user's 500/500 minimum learning gate.
    """
    validate_spec(spec, known_features)
    if int(closed_wins) < 500 or int(closed_losses) < 500:
        return None
    required = ("train", "validation", "holdout")
    if not all(isinstance(evidence.get(k), dict) for k in required):
        return None
    for key in required:
        row = evidence[key]
        if int(row.get("trades", 0)) < 20 or float(row.get("avg_net") or 0) <= 0:
            return None
    record = {
        "schema": SCHEMA_VERSION,
        "feature_name": strategy_feature_name(spec),
        "spec": spec,
        "status": "approved",
        "approved_at": int(evidence.get("approved_at", 0) or 0),
        "evidence": {k: evidence[k] for k in required},
        "closed_wins_at_approval": int(closed_wins),
        "closed_losses_at_approval": int(closed_losses),
    }
    return record


def feature_values(features, record):
    """Compute a causal 0/1 feature aligned to the supplied feature frame."""
    import numpy as np
    validate_spec(record.get("spec"), set(features.columns))
    mask = np.ones(len(features), dtype=bool)
    for cond in record["spec"]["conditions"]:
        vals = features[cond["feature"]].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        test = vals > float(cond["threshold"]) if cond["op"] == "gt" else vals < float(cond["threshold"])
        mask &= valid & test
    return mask.astype("float32")


def approved_feature_matrix(features, registry):
    import pandas as pd
    records = registry.get("features", []) if isinstance(registry, dict) else []
    cols = {}
    for record in records:
        if record.get("status") != "approved":
            continue
        name = record.get("feature_name") or strategy_feature_name(record.get("spec", {}))
        if name in cols:
            continue
        try:
            cols[name] = feature_values(features, record)
        except (ValueError, KeyError, TypeError):
            continue
    return pd.DataFrame(cols, index=features.index)


def save_registry(path, registry):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".derived_features_", suffix=".tmp",
                                dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(registry, stream, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
