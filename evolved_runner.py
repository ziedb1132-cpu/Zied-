#!/usr/bin/env python3
"""Run Zied Quant with deep learning, guarded evolution and research backtests."""
import json
import os
import sys
import time

import zied_quant
import deep_learning_evolution
import evolution_engine
import strategy_evolver
import candle_news_intelligence
import ai_trade_controller
import exit_evolution
import strategy_discovery_lab
import autonomous_strategy_engine
import strategy_meta_selector
import advanced_intelligence
import evolution_governance

# Install the same guarded intelligence stack used by the scheduled scanner.
deep_learning_evolution.install(zied_quant)
candle_news_intelligence.install(zied_quant)
ai_trade_controller.install(zied_quant)

_original_hist_depth = zied_quant.hist_depth
def _expanded_hist_depth():
    depth = _original_hist_depth()
    depth["1D"] = max(int(depth["1D"]), 320)
    return depth
zied_quant.hist_depth = _expanded_hist_depth

evolution_engine.install(zied_quant)
exit_evolution.install(zied_quant)
advanced_intelligence.install(zied_quant)
zied_quant.PORTFOLIO_CORRELATION_HOOK = (
    lambda paper, inst, cache: advanced_intelligence.portfolio_correlation_filter(
        zied_quant, paper, inst, zied_quant.CFG["MAX_OPEN_CORR"], cache
    )
)

_original_retrain_with_diagnostics = zied_quant.retrain
def _retrain_with_intelligence_diagnostics(brain, paper, universe=None, reason="manual", refresh=False):
    result = _original_retrain_with_diagnostics(brain, paper, universe, reason, refresh)
    try:
        if isinstance(result, dict) and not result.get("skipped"):
            advanced_intelligence.post_fit_diagnostics(zied_quant, brain, result)
    except Exception as exc:
        zied_quant.log.warning("advanced model diagnostics skipped: %s", exc)
    return result
zied_quant.retrain = _retrain_with_intelligence_diagnostics

_original_scan = zied_quant.scan
def _scan_with_strategy_discovery(brain, paper, universe=None, cycle=0, trade=True):
    # Strategy research runs once, from run_comprehensive_backtest after the
    # core walk-forward backtest. Do not repeat it during the scan.
    return _original_scan(brain, paper, universe, cycle, trade)

zied_quant.scan = _scan_with_strategy_discovery


def run_comprehensive_backtest(universe=None, paper=None):
    """Run all isolated research layers after the core purged walk-forward model test."""
    state_dir = zied_quant.CFG["DIR"]
    os.makedirs(state_dir, exist_ok=True)
    report = {
        "schema": 1,
        "created_at": time.time(),
        "mode": "research_only",
        "production_model_changed": False,
        "live_signals_changed": False,
        "portfolio_changed": False,
        "learning_gate": "unchanged: 500 closed winners AND 500 closed losers",
        "layers": {},
    }
    universe = universe if universe is not None else zied_quant.get_universe()
    if universe is None or universe.empty:
        raise RuntimeError("Comprehensive backtest: OKX returned an empty universe")

    # This audit uses recent matured signal snapshots and is independent of the
    # 500-win + 500-loss model activation gate.
    try:
        quality_paper = paper if paper is not None else zied_quant.Paper()
        quality = advanced_intelligence.feature_quality_report_from_db(
            zied_quant, quality_paper)
        advanced_intelligence._write_json(state_dir, "feature_quality.json", quality)
        report["feature_quality_audit"] = quality
    except Exception as exc:
        report["feature_quality_audit"] = {"status": "error", "error": str(exc)}

    # 150 explainable candidates spanning trend, breakout, reversal, momentum,
    # range-breakout and volume-breakout families; chronological validation/holdout.
    try:
        result = strategy_evolver.run_evolution(
            universe, zied_quant.get_candles, state_dir, telegram=None, max_coins=8)
        report["layers"]["six_strategy_families"] = result
    except Exception as exc:
        report["layers"]["six_strategy_families"] = {"status": "error", "error": str(exc)}

    # Run the strategy discovery lab once and keep its output in this canonical report.
    try:
        lab = strategy_discovery_lab.run_lab(zied_quant, paper)
        report["layers"]["strategy_discovery_lab"] = lab
    except Exception as exc:
        report["layers"]["strategy_discovery_lab"] = {"status": "error", "error": str(exc)}

    # Evolve symbolic rules over the causal 546-feature library and preserve its archive.
    try:
        discovery = autonomous_strategy_engine.run_discovery(
            universe, zied_quant.get_candles, state_dir, telegram=None,
            batch_size=10, max_coins=5)
        report["layers"]["autonomous_feature_discovery"] = discovery
    except Exception as exc:
        discovery = None
        report["layers"]["autonomous_feature_discovery"] = {"status": "error", "error": str(exc)}

    # Independent shadow portfolio measures each strategy by BULL/BEAR/RANGE/HIGH_VOL.
    try:
        meta = strategy_meta_selector.run_meta_selector(
            universe, zied_quant.get_candles, state_dir,
            discovery_report=discovery, telegram=None, max_symbols=5)
        report["layers"]["regime_shadow_portfolio"] = meta
    except Exception as exc:
        report["layers"]["regime_shadow_portfolio"] = {"status": "error", "error": str(exc)}

    # Compare current, TP, SL, trailing and time-exit variants on matured closed-signal
    # 5m paths. This collects counterfactual samples; promotion remains separately gated.
    try:
        exit_paper = paper if paper is not None else zied_quant.Paper()
        evaluated = exit_evolution._evaluate(exit_paper, zied_quant)
        report["layers"]["exit_policy_evolution"] = {
            "status": "ok",
            "new_matured_signals_evaluated": int(evaluated),
            "policy_count": len(exit_evolution.POLICIES),
            "policy_names": [p[0] for p in exit_evolution.POLICIES],
            "feedback": exit_evolution._exit_learning_report(exit_paper),
            "promotion_gate": "500 wins + 500 losses plus independent holdout",
            "live_integration": False,
        }
    except Exception as exc:
        report["layers"]["exit_policy_evolution"] = {"status": "error", "error": str(exc)}

    # Persist the research Champion/Challenger decision, adaptive mutation pressure,
    # feature diagnostics, overfit audit and performance-drift monitoring.
    try:
        def _read_optional_report(filename):
            try:
                with open(os.path.join(state_dir, filename), encoding="utf-8") as handle:
                    value = json.load(handle)
                return value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                return {}
        def _closed_signal_performance(paper_obj, window=30):
            if paper_obj is None:
                return {"status": "paper_database_unavailable"}
            try:
                with paper_obj.lock:
                    rows = paper_obj.db.execute(
                        "SELECT ret FROM tracked_signals WHERE status='CLOSED' "
                        "AND ret IS NOT NULL ORDER BY closed_ts DESC LIMIT ?",
                        (window * 2,),
                    ).fetchall()
                values = []
                for row in rows:
                    try:
                        value = float(row["ret"])
                        if __import__("math").isfinite(value):
                            values.append(value)
                    except (TypeError, ValueError, KeyError):
                        continue
                recent, previous = values[:window], values[window:window * 2]
                def _stats(sample):
                    if not sample:
                        return {"trades": 0, "avg_net": None, "profit_factor": None}
                    gains = sum(x for x in sample if x > 0)
                    losses = -sum(x for x in sample if x < 0)
                    return {"trades": len(sample), "avg_net": sum(sample) / len(sample),
                            "profit_factor": gains / losses if losses > 0 else (999.0 if gains > 0 else 0.0)}
                recent_stats, previous_stats = _stats(recent), _stats(previous)
                degraded = False
                if len(recent) >= 20 and len(previous) >= 20:
                    degraded = (
                        (previous_stats["avg_net"] is not None
                         and previous_stats["avg_net"] > 0
                         and recent_stats["avg_net"] < previous_stats["avg_net"] * 0.70)
                        or (previous_stats["profit_factor"] > 1.05
                            and recent_stats["profit_factor"] < previous_stats["profit_factor"] * 0.70)
                    )
                return {"status": "ok" if len(values) >= 20 else "insufficient_history",
                        "recent": recent_stats, "previous": previous_stats,
                        "degraded": bool(degraded), "window_size": window,
                        "note": "Descriptive monitoring only; it never pauses or changes trading."}
            except Exception as exc:
                return {"status": "error", "error": str(exc), "degraded": False}

        governance = evolution_governance.update_governance(
            state_dir,
            discovery=discovery,
            strategy=report["layers"].get("six_strategy_families"),
            feature_quality=report.get("feature_quality_audit"),
            feature_importance=_read_optional_report("feature_importance.json"),
            feature_drift=_read_optional_report("feature_drift.json"),
            regime_report=report["layers"].get("regime_shadow_portfolio"),
            closed_performance=_closed_signal_performance(paper),
        )
        report["layers"]["evolution_governance"] = governance
    except Exception as exc:
        zied_quant.log.exception("evolution governance skipped: %s", exc)
        report["layers"]["evolution_governance"] = {
            "status": "error", "error": str(exc),
            "production_promotion": {"enabled": False,
                "gate": "500 closed winners + 500 closed losers + independent model holdout"},
        }

    # Include the core model walk-forward result produced by zied_quant.py backtest.
    core_path = os.path.join(state_dir, "backtest.json")
    try:
        with open(core_path, encoding="utf-8") as handle:
            report["layers"]["model_walk_forward"] = json.load(handle)
    except (OSError, ValueError):
        report["layers"]["model_walk_forward"] = {"status": "missing_core_backtest"}

    # Snapshot capital/risk assumptions used by the core portfolio replay for auditability.
    keys = ("MAX_OPEN", "POS_FRAC", "RISK_PER_TRADE", "MAX_PORTFOLIO_RISK",
            "BT_COST", "BARRIER_H")
    report["capital_and_risk_assumptions"] = {
        key: zied_quant.CFG.get(key) for key in keys if key in zied_quant.CFG
    }
    report["note"] = (
        "Research feedback only. No backtest result is injected as a training label, "
        "no live policy is changed, and no paper position is modified. AI Learning may "
        "consume only matured real closed-signal outcomes under its existing 500/500 gate."
    )
    output = os.path.join(state_dir, "comprehensive_backtest.json")
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp, output)

    statuses = {name: value.get("status", "completed")
                for name, value in report["layers"].items() if isinstance(value, dict)}
    zied_quant.tg(
        "🧪 Zied Quant — Comprehensive Backtest\n"
        + "\n".join(f"{name}: {status}" for name, status in statuses.items())
        + "\nResearch only — live strategy unchanged."
    )
    zied_quant.log.info("comprehensive backtest saved: %s", output)
    return report


# Run the full research suite immediately after the normal walk-forward test
# in every scheduled analysis cycle. It reuses the cycle's universe and Paper DB.
_original_cycle_backtest = zied_quant.run_cycle_backtest

def _cycle_backtest_with_comprehensive_research(universe, paper=None, folds=5):
    # The workflow can chain many short runs. Heavy model + strategy research is
    # therefore limited to once per rolling 24 hours using the durable Paper DB.
    if paper is not None and not zied_quant.daily_backtest_due(paper):
        zied_quant.log.info("heavy walk-forward/comprehensive research skipped: daily guard is active")
        return {"status": "skipped_daily_guard"}
    core_result = _original_cycle_backtest(universe, paper, folds)
    if core_result is None:
        return None
    if paper is not None:
        zied_quant.mark_daily_backtest_attempt(paper)
    try:
        run_comprehensive_backtest(universe=universe, paper=paper)
    except Exception as exc:
        zied_quant.log.exception("cycle comprehensive backtest failed: %s", exc)
        zied_quant.tg(f"⚠️ تعذر إكمال Backtest الشامل في هذه الدورة: {exc}")
    return core_result

zied_quant.run_cycle_backtest = _cycle_backtest_with_comprehensive_research


if len(sys.argv) > 1 and sys.argv[1] == "backtest":
    # First run the existing purged ML walk-forward backtest unchanged.
    zied_quant.main(["zied_quant.py", *sys.argv[1:]])
    run_comprehensive_backtest()
else:
    zied_quant.main(["zied_quant.py", *sys.argv[1:]])
