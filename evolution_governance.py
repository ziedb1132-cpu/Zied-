#!/usr/bin/env python3
"""Persistent, research-only governance for autonomous strategy evolution.

Tracks a research Champion/Challenger, adaptive mutation pressure, holdout
robustness, feature diagnostics, opportunity quality and performance drift.
It never edits production model weights, signals, orders, portfolio or risk gates.
"""
import hashlib
import json
import math
import os
import time


def _number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _read(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False,
                  separators=(",", ":"), default=str)
    os.replace(tmp, path)


def _holdout_quality(row):
    validation = row.get("validation") or row.get("tune") or {}
    holdout = row.get("holdout_observation") or row.get("test") or {}
    trades = int(holdout.get("trades") or 0)
    avg = _number(holdout.get("avg_net"))
    pf = _number(holdout.get("profit_factor"), 0.0)
    val_avg = _number(validation.get("avg_net"))
    per_symbol = row.get("per_symbol") or []
    symbol_results = [
        _number((item.get("holdout") or {}).get("avg_net"))
        for item in per_symbol if isinstance(item, dict)
    ]
    symbol_results = [x for x in symbol_results if x is not None]
    positive_symbols = sum(x > 0 for x in symbol_results)
    symbol_ratio = positive_symbols / len(symbol_results) if symbol_results else None
    gap = (val_avg - avg) if val_avg is not None and avg is not None else None
    reasons = []
    if trades < 10:
        reasons.append("holdout_trades_below_10")
    if avg is None or avg <= 0:
        reasons.append("holdout_expectancy_not_positive")
    if pf <= 1.05:
        reasons.append("holdout_profit_factor_not_above_1.05")
    if symbol_ratio is not None and len(symbol_results) >= 3 and symbol_ratio < 0.5:
        reasons.append("cross_symbol_consistency_below_50pct")
    if gap is not None and gap > max(0.01, abs(val_avg or 0.0) * 2.0):
        reasons.append("large_validation_holdout_gap")
    return {
        "passed": not reasons, "reasons": reasons, "trades": trades,
        "avg_net": avg, "profit_factor": pf, "validation_avg_net": val_avg,
        "generalization_gap": gap, "positive_symbols": positive_symbols,
        "tested_symbols": len(symbol_results), "positive_symbol_ratio": symbol_ratio,
    }


def _candidate_from_discovery(discovery):
    candidates = []
    for row in (discovery or {}).get("latest_experiments", []):
        if not isinstance(row, dict) or row.get("status") != "validation_pass":
            continue
        quality = _holdout_quality(row)
        validation = row.get("validation") or {}
        candidates.append({
            "id": row.get("id"), "generation": row.get("generation"),
            "score": _number(row.get("score"), -100.0),
            "spec": row.get("spec"), "genome": row.get("genome"),
            "validation": validation, "holdout": quality,
            "source": "autonomous_feature_discovery",
        })
    candidates.sort(key=lambda x: (
        bool(x["holdout"]["passed"]),
        _number(x["holdout"].get("avg_net"), -999.0),
        x["score"]), reverse=True)
    return candidates[0] if candidates else None


def _candidate_from_strategy(strategy):
    if not isinstance(strategy, dict) or strategy.get("status") != "accepted":
        return None
    champion = strategy.get("champion") or strategy.get("latest_tested") or {}
    test = champion.get("test") or {}
    holdout = {
        "passed": int(test.get("trades") or 0) >= 30
                  and _number(test.get("avg_net"), -1.0) > 0
                  and _number(test.get("profit_factor"), 0.0) > 1.05,
        "reasons": [], "trades": int(test.get("trades") or 0),
        "avg_net": _number(test.get("avg_net")),
        "profit_factor": _number(test.get("profit_factor"), 0.0),
        "validation_avg_net": _number((champion.get("tune") or {}).get("avg_net")),
        "generalization_gap": None, "positive_symbols": None,
        "tested_symbols": len(strategy.get("symbols") or []),
        "positive_symbol_ratio": None,
    }
    if not holdout["passed"]:
        holdout["reasons"] = ["independent_holdout_gate_failed"]
    return {
        "id": "strategy-" + hashlib.sha256(json.dumps(champion.get("spec", {}), sort_keys=True).encode()).hexdigest()[:12],
        "generation": strategy.get("last_run"), "score": _number((champion.get("tune") or {}).get("avg_net"), -100.0),
        "spec": champion.get("spec"), "genome": None,
        "validation": champion.get("tune") or {}, "holdout": holdout,
        "source": "six_strategy_families",
    }


def update_governance(state_dir, discovery=None, strategy=None,
                      feature_quality=None, feature_importance=None,
                      feature_drift=None, regime_report=None, closed_performance=None):
    """Update durable research governance and return a compact dashboard report."""
    path = os.path.join(state_dir, "evolution_governance.json")
    memory = _read(path)
    history = list(memory.get("history") or [])
    previous_champion = memory.get("research_champion")
    candidates = [
        item for item in (
            _candidate_from_discovery(discovery),
            _candidate_from_strategy(strategy),
        ) if item is not None
    ]
    candidates.sort(key=lambda x: (
        bool((x.get("holdout") or {}).get("passed")),
        _number((x.get("holdout") or {}).get("avg_net"), -999.0),
        _number(x.get("score"), -100.0)), reverse=True)
    challenger = candidates[0] if candidates else None
    promoted = False
    decision = "no_qualified_challenger"
    champion = previous_champion

    if challenger:
        if not challenger["holdout"]["passed"]:
            decision = "challenger_rejected_by_independent_holdout"
        elif not champion:
            champion = challenger
            promoted = True
            decision = "initial_research_champion"
        else:
            old_hold = champion.get("holdout") or {}
            new_hold = challenger.get("holdout") or {}
            old_avg = _number(old_hold.get("avg_net"), -999.0)
            new_avg = _number(new_hold.get("avg_net"), -999.0)
            old_pf = _number(old_hold.get("profit_factor"), 0.0)
            new_pf = _number(new_hold.get("profit_factor"), 0.0)
            # A research-only replacement requires a measurable net improvement
            # and no material PF regression. Production remains separately gated.
            if (challenger.get("id") != champion.get("id")
                    and new_avg >= old_avg + 0.0005
                    and new_pf >= max(1.05, old_pf * 0.95)):
                champion = challenger
                promoted = True
                decision = "challenger_promoted_research_only"
            else:
                decision = "champion_retained"

    gen = int((discovery or {}).get("generation") or 0)
    best_score = max(
        [_number(x.get("score"), -100.0)
         for x in ((discovery or {}).get("latest_experiments") or [])
         if isinstance(x, dict)] or [-100.0])
    prior_scores = [ _number(x.get("best_score"), None) for x in history[-3:] ]
    prior_scores = [x for x in prior_scores if x is not None]
    mutation_rate = 0.40
    if len(prior_scores) >= 2 and max(prior_scores) - min(prior_scores) < 1.0:
        mutation_rate = 0.65
        mutation_reason = "plateau_in_last_generations"
    elif prior_scores and best_score > max(prior_scores) + 1.0:
        mutation_rate = 0.28
        mutation_reason = "recent_improvement_preserve_elites"
    else:
        mutation_reason = "balanced_exploration"

    quality = feature_quality if isinstance(feature_quality, dict) else {}
    importance = feature_importance if isinstance(feature_importance, dict) else {}
    drift = feature_drift if isinstance(feature_drift, dict) else {}
    regime = regime_report if isinstance(regime_report, dict) else {}
    closed_perf = closed_performance if isinstance(closed_performance, dict) else {}
    top_features = importance.get("top_features") or quality.get("top_label_associations") or []
    feature_rows = [
        {"feature": x.get("feature"), "importance": _number(
            x.get("permutation_auc_drop", x.get("label_correlation")))}
        for x in top_features[:15] if isinstance(x, dict)
    ]
    drift_fraction = _number(drift.get("drift_fraction"))
    recent_perf = [ _number(x.get("champion_avg_net")) for x in history[-3:] ]
    recent_perf = [x for x in recent_perf if x is not None]
    degradation = bool(
        (drift_fraction is not None and drift_fraction >= 0.20)
        or (len(recent_perf) >= 2 and recent_perf[-1] < recent_perf[0] * 0.70)
    )
    realized_degradation = bool(closed_perf.get("degraded"))
    health = "warning" if (degradation or realized_degradation) else (
        "monitoring" if drift.get("status") not in (None, "ok") else "stable"
    )
    regime_rows = regime.get("selected_by_regime") or regime.get("recommendations") or regime.get("profiles") or []
    opportunities = {
        "validated_candidates": int((discovery or {}).get("validation_passed") or 0),
        "regime_recommendations": (sum(len(v) for v in regime_rows.values() if isinstance(v, list)) if isinstance(regime_rows, dict) else len(regime_rows) if isinstance(regime_rows, list) else 0),
        "selection_rule": "positive net expectancy, PF > 1.05, cross-symbol consistency and independent holdout",
    }

    champion_hold = (champion or {}).get("holdout") or {}
    history.append({
        "generation": gen, "best_score": best_score,
        "champion_avg_net": _number(champion_hold.get("avg_net")),
        "champion_id": (champion or {}).get("id"), "timestamp": time.time(),
    })
    history = history[-100:]
    memory.update({
        "schema": 1, "updated_at": time.time(), "history": history,
        "research_champion": champion, "last_challenger": challenger,
        "mutation_rate": mutation_rate,
    })
    _write(path, memory)
    return {
        "status": "ok", "generation": gen,
        "research_champion": champion, "challenger": challenger,
        "challenger_promoted": promoted, "decision": decision,
        "mutation_rate_next_generation": mutation_rate,
        "mutation_reason": mutation_reason,
        "overfit_audit": (challenger or {}).get("holdout"),
        "feature_evaluation": {
            "status": quality.get("status", "not_available"),
            "feature_count": quality.get("feature_count"),
            "constant_feature_count": quality.get("constant_feature_count"),
            "high_missing_feature_count": quality.get("high_missing_feature_count"),
            "top_features": feature_rows,
        },
        "performance_monitor": {
            "status": health, "drift_fraction": drift_fraction,
            "drifted_feature_count": drift.get("drifted_feature_count"),
            "unstable_scale_feature_count": drift.get("unstable_scale_feature_count"),
            "history_points": len(history), "closed_signal_performance": closed_perf,
            "note": "Monitoring warning only; it does not pause or alter production trading.",
        },
        "opportunity_quality": opportunities,
        "production_promotion": {
            "enabled": False,
            "gate": "500 closed winners + 500 closed losers + independent model holdout",
        },
        "note": "Research Champion/Challenger only. No production weights, live signals, orders, portfolio or risk settings are changed.",
    }
