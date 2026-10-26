#!/usr/bin/env python3
"""Run Zied Quant with guarded model evolution and daily strategy discovery."""
import sys
import zied_quant
import evolution_engine
import strategy_evolver

evolution_engine.install(zied_quant)

# Discovery is research-only and throttled to once per 23 hours by persistent state.
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
