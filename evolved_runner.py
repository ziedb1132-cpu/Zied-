#!/usr/bin/env python3
"""Run Zied Quant with 300-feature deep learning and guarded model evolution."""
import sys
import zied_quant
import deep_learning_evolution
import evolution_engine
import strategy_evolver
import candle_news_intelligence

# Expand the schema/model before installing guarded train/tune/holdout promotion.
deep_learning_evolution.install(zied_quant)
# Candlestick patterns and cached RSS headline sentiment adjust ranking only;
# the 300-feature model schema and portfolio risk controls remain unchanged.
candle_news_intelligence.install(zied_quant)
# EMA-200 on the daily timeframe needs at least 200 daily candles in bootstrap data.
_original_hist_depth = zied_quant.hist_depth
def _expanded_hist_depth():
    depth = _original_hist_depth()
    depth["1D"] = max(int(depth["1D"]), 320)
    return depth
zied_quant.hist_depth = _expanded_hist_depth

evolution_engine.install(zied_quant)

# Research-only strategy discovery is throttled to once per 23 hours by persistent state.
_original_scan = zied_quant.scan
def _scan_with_strategy_discovery(brain, paper, universe=None, cycle=0, trade=True):
    result = _original_scan(brain, paper, universe, cycle, trade)
    try:
        uni = universe if universe is not None else zied_quant.get_universe()
        if not uni.empty:
            outcome = strategy_evolver.run_evolution(
                uni, zied_quant.get_candles, zied_quant.CFG["DIR"], zied_quant.tg)
            zied_quant.log.info("strategy evolution: %s", outcome.get("status"))
    except Exception as exc:
        zied_quant.log.warning("strategy evolution skipped after error: %s", exc)
    return result

zied_quant.scan = _scan_with_strategy_discovery
zied_quant.main(["zied_quant.py", *sys.argv[1:]])
