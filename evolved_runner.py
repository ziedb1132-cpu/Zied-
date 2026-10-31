#!/usr/bin/env python3
"""Run Zied Quant with 300-feature deep learning and guarded model evolution."""
import sys
import zied_quant
import deep_learning_evolution
import evolution_engine
import strategy_evolver
import candle_news_intelligence
import ai_trade_controller
import exit_evolution
import strategy_discovery_lab
import autonomous_strategy_engine
import advanced_intelligence

# Expand the schema/model before installing guarded train/tune/holdout promotion.
deep_learning_evolution.install(zied_quant)
# Candlestick patterns and cached RSS headline sentiment adjust ranking only;
# the 300-feature model schema and portfolio risk controls remain unchanged.
candle_news_intelligence.install(zied_quant)
# Discovered winner/loser patterns influence ranking only after 500 closed wins and 500 closed losses.
# The live model itself trains only on closed tracked-signal outcomes.
ai_trade_controller.install(zied_quant)
# EMA-200 on the daily timeframe needs at least 200 daily candles in bootstrap data.
_original_hist_depth = zied_quant.hist_depth
def _expanded_hist_depth():
    depth = _original_hist_depth()
    depth["1D"] = max(int(depth["1D"]), 320)
    return depth
zied_quant.hist_depth = _expanded_hist_depth

evolution_engine.install(zied_quant)
# Learn TP/SL ATR multipliers from completed counterfactual 1H paths.
# Existing signals are untouched; only newly created paper signals use promoted settings.
exit_evolution.install(zied_quant)

# Seven-part intelligence integration:
# (1) feature importance, (2) existing purged chronological walk-forward,
# (3) existing regime classifier/stats, (4) existing guarded exit evolution,
# (5) probability calibration diagnostics, (6) live feature-drift monitoring,
# (7) correlation-aware paper portfolio selection.
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

# Unified research pipeline: run strategy evolution and discovery lab in every scan cycle.
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
        # Ten new symbolic strategy experiments per cycle. Research-only; no live
        # ranking, AI weights, entry/exit rules, or portfolio settings are changed.
        discovery = autonomous_strategy_engine.run_discovery(
            uni, zied_quant.get_candles, zied_quant.CFG["DIR"], zied_quant.tg,
            batch_size=10, max_coins=5)
        zied_quant.log.info(
            "autonomous strategy discovery: generation=%s generated=%s passed=%s",
            discovery.get("generation"), discovery.get("generated"),
            discovery.get("validation_passed"))
    except Exception as exc:
        zied_quant.log.warning("strategy evolution skipped after error: %s", exc)
    return result

zied_quant.scan = _scan_with_strategy_discovery
zied_quant.main(["zied_quant.py", *sys.argv[1:]])
