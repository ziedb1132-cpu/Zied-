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


def performance_risk_summary(returns):
    """Describe realized payoff asymmetry without treating a small sample as proof."""
    values = []
    for value in ([] if returns is None else returns):
        number = _number(value)
        if number is not None:
            values.append(number)
    wins = [x for x in values if x > 0]
    losses = [x for x in values if x < 0]
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    payoff = (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss is not None else None
    breakeven = (abs(avg_loss) / (avg_win + abs(avg_loss))
                 if avg_win is not None and avg_loss is not None else None)
    return {
        "trades": len(values), "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(values) if values else None,
        "avg_net": sum(values) / len(values) if values else None,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff_ratio": payoff, "break_even_win_rate": breakeven,
        "profit_factor": pf,
        "largest_loss": min(losses) if losses else None,
        "status": "ok" if len(values) >= 30 else "insufficient_history",
    }


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


def _validation_quality(metrics):
    """Validation-only eligibility gate; never consult the reusable final holdout."""
    metrics = metrics if isinstance(metrics, dict) else {}
    trades = int(metrics.get("trades") or 0)
    avg = _number(metrics.get("avg_net"))
    pf = _number(metrics.get("profit_factor"), 0.0)
    reasons = []
    if trades < 30:
        reasons.append("validation_trades_below_30")
    if avg is None or avg <= 0:
        reasons.append("validation_expectancy_not_positive")
    if pf <= 1.05:
        reasons.append("validation_profit_factor_not_above_1.05")
    return {"passed": not reasons, "reasons": reasons,
            "trades": trades, "avg_net": avg, "profit_factor": pf}


def _candidate_from_discovery(discovery):
    candidates = []
    for row in (discovery or {}).get("latest_experiments", []):
        if not isinstance(row, dict) or row.get("status") != "validation_pass":
            continue
        validation = row.get("validation") or {}
        validation_gate = _validation_quality(validation)
        if not validation_gate["passed"]:
            continue
        quality = _holdout_quality(row)
        candidates.append({
            "id": row.get("id"), "generation": row.get("generation"),
            "score": _number(row.get("score"), -100.0),
            "spec": row.get("spec"), "genome": row.get("genome"),
            "validation": validation, "validation_gate": validation_gate,
            "holdout": quality, "source": "autonomous_feature_discovery",
        })
    # Select by tune/validation only. The same chronological holdout is
    # reused across generations, so it must never rank candidates or promote
    # a research champion; doing so would turn it into a repeatedly queried
    # validation set and leak information into model selection.
    candidates.sort(key=lambda x: (
        _number((x.get("validation") or {}).get("avg_net"), -999.0),
        _number((x.get("validation") or {}).get("profit_factor"), 0.0),
        x["score"]), reverse=True)
    return candidates[0] if candidates else None


def _candidate_from_strategy(strategy):
    if not isinstance(strategy, dict) or strategy.get("status") != "accepted":
        return None
    champion = strategy.get("champion") or strategy.get("latest_tested") or {}
    validation = champion.get("tune") or champion.get("validation") or {}
    validation_gate = _validation_quality(validation)
    if not validation_gate["passed"]:
        return None
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
        "validation": validation, "validation_gate": validation_gate,
        "holdout": holdout, "source": "six_strategy_families",
    }


def update_governance(state_dir, discovery=None, strategy=None,
                      feature_quality=None, feature_importance=None,
                      feature_drift=None, regime_report=None, closed_performance=None):
    """Update durable research governance and return a compact dashboard report."""
    path = os.path.join(state_dir, "evolution_governance.json")
    memory = _read(path)
    history = list(memory.get("history") or [])
    previous_champion = memory.get("research_champion")
    # Discard legacy validation-only champions; they have not earned forward
    # shadow confirmation under the current governance contract.
    if previous_champion and previous_champion.get("source") != "forward_shadow_paper":
        previous_champion = None
    # A genetic candidate becomes a research Champion only after genuinely
    # forward, matured shadow-paper evidence. Reused historical holdout slices
    # never promote candidates or influence the breeding pool.
    shadow = regime_report if isinstance(regime_report, dict) else {}
    shadow_rows = shadow.get("selected_by_regime") or {}
    candidates = []
    if isinstance(shadow_rows, dict):
        for regime_name, rows in shadow_rows.items():
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                trades = int(row.get("trades") or 0)
                avg = _number(row.get("avg_net"))
                pf = _number(row.get("profit_factor"), 0.0)
                dd = _number(row.get("max_drawdown"), 1.0)
                spec = row.get("spec")
                if (not row.get("strategy_id") or not isinstance(spec, dict) or not spec
                        or trades < 30 or avg is None or avg <= 0
                        or pf < 1.20 or dd is None or dd > 0.20):
                    continue
                candidates.append({
                    "id": row["strategy_id"], "generation": int((discovery or {}).get("generation") or 0),
                    "score": avg, "spec": spec, "genome": None,
                    "regime": str(regime_name),
                    "shadow": {"trades": trades, "avg_net": avg, "profit_factor": pf,
                               "max_drawdown": dd, "win_rate": _number(row.get("win_rate"))},
                    "source": "forward_shadow_paper",
                })
    candidates.sort(key=lambda x: (
        _number((x.get("shadow") or {}).get("avg_net"), -999.0),
        _number((x.get("shadow") or {}).get("profit_factor"), 0.0),
        -_number((x.get("shadow") or {}).get("max_drawdown"), 1.0)), reverse=True)
    challenger = candidates[0] if candidates else None
    promoted = False
    decision = "no_shadow_validated_challenger"
    champion = previous_champion

    if challenger:
        new_shadow = challenger["shadow"]
        old_shadow = (champion or {}).get("shadow") or {}
        if not champion:
            champion = challenger
            promoted = True
            decision = "initial_shadow_validated_research_champion"
        elif challenger.get("id") == champion.get("id"):
            champion = challenger
            decision = "shadow_champion_refreshed"
        elif (_number(new_shadow.get("avg_net"), -999.0)
              >= _number(old_shadow.get("avg_net"), -999.0) + 0.0005
              and _number(new_shadow.get("profit_factor"), 0.0)
              >= max(1.20, _number(old_shadow.get("profit_factor"), 0.0) * 0.95)):
            champion = challenger
            promoted = True
            decision = "shadow_validated_challenger_promoted"
        else:
            decision = "shadow_champion_retained"

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
    recent_perf = closed_perf.get("recent") or {}
    structurally_unprofitable = bool(
        int(recent_perf.get("trades") or 0) >= 30
        and _number(recent_perf.get("avg_net"), 0.0) <= 0.0
        and _number(recent_perf.get("profit_factor"), 0.0) <= 1.05
    )
    health = "warning" if (degradation or realized_degradation or structurally_unprofitable) else (
        "monitoring" if drift.get("status") not in (None, "ok") else "stable"
    )
    regime_rows = regime.get("selected_by_regime") or regime.get("recommendations") or regime.get("profiles") or []
    opportunities = {
        "validated_candidates": int((discovery or {}).get("validation_passed") or 0),
        "regime_recommendations": (sum(len(v) for v in regime_rows.values() if isinstance(v, list)) if isinstance(regime_rows, dict) else len(regime_rows) if isinstance(regime_rows, list) else 0),
        "selection_rule": "positive net expectancy, PF > 1.05, cross-symbol consistency and independent holdout",
    }

    champion_shadow = (champion or {}).get("shadow") or {}
    history.append({
        "generation": gen, "best_score": best_score,
        "champion_avg_net": _number(champion_shadow.get("avg_net")),
        "champion_id": (champion or {}).get("id"), "timestamp": time.time(),
    })
    history = history[-100:]
    memory.update({
        "schema": 1, "updated_at": time.time(), "history": history,
        "research_champion": champion, "last_challenger": challenger,
        "ai_strategy_feedback": {str(x.get("regime")): x for x in candidates},
        "mutation_rate": mutation_rate,
    })
    _write(path, memory)
    return {
        "status": "ok", "generation": gen,
        "research_champion": champion, "challenger": challenger,
        "ai_strategy_feedback": {str(x.get("regime")): x for x in candidates},
        "challenger_promoted": promoted, "decision": decision,
        "mutation_rate_next_generation": mutation_rate,
        "mutation_reason": mutation_reason,
        "overfit_audit": (challenger or {}).get("holdout"),
        "holdout_usage": "reused_historical_holdout_is_diagnostic_only; research_champion_requires_30_matured_forward_shadow_trades_pf_1.20_positive_net_expectancy_and_drawdown_at_most_20pct",
        "feature_evaluation": {
            "status": quality.get("status", "not_available"),
            "feature_count": quality.get("feature_count"),
            "constant_feature_count": quality.get("constant_feature_count"),
            "high_missing_feature_count": quality.get("high_missing_feature_count"),
            "top_features": feature_rows,
        },
        "performance_monitor": {
            "status": health,
            "risk_flag": ("recent_net_edge_not_proven" if structurally_unprofitable
                          else ("performance_degraded" if (degradation or realized_degradation)
                                else "no_current_warning")),
            "drift_fraction": drift_fraction,
            "drifted_feature_count": drift.get("drifted_feature_count"),
            "unstable_scale_feature_count": drift.get("unstable_scale_feature_count"),
            "history_points": len(history), "closed_signal_performance": closed_perf,
            "note": "Monitoring warning only; it does not pause or alter production trading.",
        },
        "opportunity_quality": opportunities,
        "production_promotion": {
            "enabled": False,
            "gate": "independent model holdout and net-performance validation; no fixed win/loss quota",
        },
        "note": "Research Champion/Challenger selection requires matured forward shadow-paper evidence. Reused historical holdout is diagnostic only. AI strategy feedback is exported as research context; production model weights and live signals are not changed.",
    }
