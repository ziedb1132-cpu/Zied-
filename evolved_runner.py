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
    result = _original_scan(brain, paper, universe, cycle, trade)
    try:
        uni = universe if universe is not None else zied_quant.get_universe()
        if not uni.empty:
            outcome = strategy_evolver.run_evolution(
                uni, zied_quant.get_candles, zied_quant.CFG["DIR"], zied_quant.tg)
            zied_quant.log.info("strategy evolution: %s", outcome.get("status"))
        lab_outcome = strategy_discovery_lab.run_lab(zied_quant, paper)
        zied_quant.log.info("strategy discovery lab: %s", lab_outcome.get("status"))
        discovery = autonomous_strategy_engine.run_discovery(
            uni, zied_quant.get_candles, zied_quant.CFG["DIR"], zied_quant.tg,
            batch_size=10, max_coins=5)
        meta_report = strategy_meta_selector.run_meta_selector(
            uni, zied_quant.get_candles, zied_quant.CFG["DIR"],
            discovery_report=discovery, telegram=zied_quant.tg, max_symbols=5)
        zied_quant.log.info(
            "strategy meta-selector: cycle=%s shadow_closed=%s approved_regimes=%s",
            meta_report.get("cycle"), meta_report.get("closed_shadow_trades"),
            sum(len(v) for v in meta_report.get("selected_by_regime", {}).values()))
    except Exception as exc:
        zied_quant.log.warning("strategy evolution skipped after error: %s", exc)
    return result
zied_quant.scan = _scan_with_strategy_discovery


def run_comprehensive_backtest():
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
    universe = zied_quant.get_universe()
    if universe is None or universe.empty:
        raise RuntimeError("Comprehensive backtest: OKX returned an empty universe")

    # 150 explainable candidates spanning trend, breakout, reversal, momentum,
    # range-breakout and volume-breakout families; chronological validation/holdout.
    try:
        result = strategy_evolver.run_evolution(
            universe, zied_quant.get_candles, state_dir, telegram=None, max_coins=8)
        report["layers"]["six_strategy_families"] = result
    except Exception as exc:
        report["layers"]["six_strategy_families"] = {"status": "error", "error": str(exc)}

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
        paper = zied_quant.Paper()
        evaluated = exit_evolution._evaluate(paper, zied_quant)
        report["layers"]["exit_policy_evolution"] = {
            "status": "ok",
            "new_matured_signals_evaluated": int(evaluated),
            "policy_count": len(exit_evolution.POLICIES),
            "policy_names": [p[0] for p in exit_evolution.POLICIES],
            "feedback": exit_evolution._exit_learning_report(paper),
            "promotion_gate": "500 wins + 500 losses plus independent holdout",
            "live_integration": False,
        }
        paper.close()
    except Exception as exc:
        report["layers"]["exit_policy_evolution"] = {"status": "error", "error": str(exc)}

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


if len(sys.argv) > 1 and sys.argv[1] == "backtest":
    # First run the existing purged ML walk-forward backtest unchanged.
    zied_quant.main(["zied_quant.py", *sys.argv[1:]])
    run_comprehensive_backtest()
else:
    zied_quant.main(["zied_quant.py", *sys.argv[1:]])
