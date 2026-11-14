#!/usr/bin/env python3
"""Cautious bridge from forward-shadow evolutionary evidence to primary AI inference.

Only matured, profitable, regime-and-symbol-approved shadow profiles may softly
calibrate an already-valid primary-model probability. This never creates labels,
changes model weights, changes positions, or bypasses the primary entry gates.
"""
import json
import logging
import os

import numpy as np

log = logging.getLogger("zied.evolution_ai_bridge")

MIN_TRADES = 20
MIN_PROFIT_FACTOR = 1.20
MAX_DRAWDOWN = 0.20
MAX_BLEND_WEIGHT = 0.10
MAX_REPORT_AGE_SECONDS = 72 * 60 * 60


def _load_report(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _matching_shadow_regime(zied_quant, row):
    """Reproduce strategy_meta_selector.market_regime from primary features."""
    try:
        atr = float(row["1H_atr_pct"])
        ema21_dist = float(row["1H_ema_dist21"])
        ema50_dist = float(row["1H_ema_dist50"])
        slope = float(row["1H_ema_slope21"])
        gap = (1.0 + ema50_dist) / (1.0 + ema21_dist) - 1.0
        if np.isfinite([atr, gap, slope]).all():
            if atr >= 0.035:
                return "HIGH_VOL"
            if gap >= 0.008 and slope > 0:
                return "BULL"
            if gap <= -0.008 and slope < 0:
                return "BEAR"
            return "RANGE"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    # Compatibility fallback for callers with a custom regime column/classifier.
    return str(zied_quant.classify_market_regime(row))


def _valid_profile(profile):
    if not isinstance(profile, dict):
        return False
    try:
        trades = int(profile.get("trades") or 0)
        win_rate = float(profile.get("win_rate"))
        avg_net = float(profile.get("avg_net", profile.get("shrunk_avg_net")))
        profit_factor = float(profile.get("profit_factor"))
        drawdown = float(profile.get("max_drawdown"))
    except (TypeError, ValueError):
        return False
    return (
        profile.get("paper_approved") is True
        and trades >= MIN_TRADES
        and np.isfinite([win_rate, avg_net, profit_factor, drawdown]).all()
        and 0.0 <= win_rate <= 1.0
        and avg_net > 0.0
        and profit_factor >= MIN_PROFIT_FACTOR
        and 0.0 <= drawdown <= MAX_DRAWDOWN
    )


def apply_strategy_feedback(zied_quant, frame, predictions, report_path):
    """Blend approved symbol+regime shadow win-rate into primary probabilities.

    The blend is deliberately small and sample-shrunk. Missing/stale/malformed
    evidence leaves primary probabilities unchanged. The function is safe to
    call on every scan and does not mutate the input frame or model.
    """
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("primary probabilities must be finite and aligned with feature rows")
    if not len(values):
        return values

    report = _load_report(report_path)
    try:
        report_age = __import__("time").time() - float(report.get("updated_at"))
    except (TypeError, ValueError):
        return values
    if not np.isfinite(report_age) or report_age < -300 or report_age > MAX_REPORT_AGE_SECONDS:
        return values

    selected = report.get("selected_by_symbol_regime")
    if not isinstance(selected, dict) or not selected:
        return values

    adjusted = values.copy()
    applied = 0
    for offset, (symbol, row) in enumerate(frame.iterrows()):
        try:
            regime = _matching_shadow_regime(zied_quant, row)
            profiles = selected.get(str(symbol) + "|" + regime, [])
            if not isinstance(profiles, list):
                continue
            valid = [item for item in profiles if _valid_profile(item)]
            if not valid:
                continue

            # Combine the approved strategies conservatively, weighting each by
            # its forward sample size. No historical train/validation/holdout
            # score is used here; only independently matured shadow-paper trades.
            weights = []
            rates = []
            for item in valid:
                n = int(item["trades"])
                weight = min(MAX_BLEND_WEIGHT, n / (n + 100.0) * MAX_BLEND_WEIGHT)
                if weight > 0:
                    weights.append(weight)
                    rates.append(float(item["win_rate"]))
            if not weights:
                continue
            total_weight = float(sum(weights))
            shadow_rate = float(np.average(rates, weights=weights))
            blend = min(MAX_BLEND_WEIGHT, total_weight / len(weights))
            adjusted[offset] = (1.0 - blend) * adjusted[offset] + blend * shadow_rate
            applied += 1
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            continue

    if applied:
        log.info(
            "evolution feedback applied to primary AI: rows=%d/%d max_blend=%.3f source=forward_shadow_paper",
            applied, len(values), MAX_BLEND_WEIGHT,
        )
    return np.clip(adjusted, 0.0, 1.0)
