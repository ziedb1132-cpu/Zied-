#!/usr/bin/env python3
"""Regime-aware meta-selector with an isolated shadow paper portfolio.

It learns from matured shadow trades only. It never changes live signals,
positions, risk settings, or the production model.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np

from autonomous_strategy_engine import build_features, _signals

FEE_AND_SLIPPAGE = 0.0035
MIN_SHADOW_TRADES = 30
MIN_PROFIT_FACTOR = 1.20
MIN_EXPECTANCY = 0.001
MAX_DRAWDOWN = 0.20
MAX_ACTIVE_STRATEGIES = 3
MAX_SYMBOLS = 5
MAX_CANDIDATES = 30
MIN_SYMBOL_SHADOW_TRADES = 20
SYMBOL_PRIOR_STRENGTH = 40


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    os.replace(tmp, path)


def market_regime(features):
    row = features.iloc[-1]
    atr = float(row["atr_pct"])
    gap = float(row["ema_gap"])
    slope = float(row["ema_slope"])
    if not np.isfinite(atr + gap + slope):
        return "UNKNOWN"
    if atr >= 0.035:
        return "HIGH_VOL"
    if gap >= 0.008 and slope > 0:
        return "BULL"
    if gap <= -0.008 and slope < 0:
        return "BEAR"
    return "RANGE"


def _metrics(trades):
    if not trades:
        return {"trades": 0, "win_rate": None, "avg_net": None,
                "profit_factor": 0.0, "max_drawdown": None}
    returns = np.asarray([float(x["net_return"]) for x in trades], dtype=float)
    gp = float(returns[returns > 0].sum())
    gl = float(-returns[returns < 0].sum())
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    peaks = np.maximum.accumulate(equity)
    dd = float(np.max(1.0 - equity / peaks))
    return {"trades": int(len(returns)),
            "win_rate": float(np.mean(returns > 0)),
            "avg_net": float(np.mean(returns)),
            "profit_factor": gp / gl if gl else (999.0 if gp else 0.0),
            "max_drawdown": dd}


def _symbol_adjusted_expectancy(symbol_avg, regime_avg, samples):
    """Shrink coin-specific expectancy toward its regime prior to limit overfit."""
    n = max(0, int(samples))
    weight = n / (n + SYMBOL_PRIOR_STRENGTH)
    return float(weight * float(symbol_avg) + (1.0 - weight) * float(regime_avg))


def _close_position(position, price, reason, now_bar, history):
    gross = float(price) / float(position["entry_price"]) - 1.0
    net = gross - FEE_AND_SLIPPAGE
    history.append({
        "strategy_id": position["strategy_id"],
        "symbol": position["symbol"],
        "entry_regime": position["entry_regime"],
        "entry_price": position["entry_price"],
        "exit_price": float(price),
        "net_return": float(net),
        "exit_reason": reason,
        "entry_bar": position["entry_bar"],
        "exit_bar": now_bar,
        "closed_at": time.time(),
    })


def run_meta_selector(universe, get_candles, state_dir, discovery_report=None,
                      telegram=None, max_symbols=MAX_SYMBOLS):
    os.makedirs(state_dir, exist_ok=True)
    memory_path = os.path.join(state_dir, "strategy_meta_selector_memory.json")
    report_path = os.path.join(state_dir, "strategy_meta_selector_report.json")
    memory = _load(memory_path) or {
        "schema": 1, "cycle": 0, "positions": {}, "pending_entries": {}, "closed_trades": [],
        "last_seen_bar": {}, "strategy_specs": {}, "last_report": {}
    }
    memory.setdefault("positions", {})
    memory.setdefault("pending_entries", {})
    memory.setdefault("closed_trades", [])
    memory.setdefault("last_seen_bar", {})
    memory.setdefault("strategy_specs", {})
    memory["cycle"] = int(memory.get("cycle", 0)) + 1

    archive = (discovery_report or {}).get("latest_experiments", [])
    champion = (discovery_report or {}).get("champion")

    # Shadow paper is an experiment layer, not a reward reserved for strategies
    # that already passed validation. Monitor fresh rejected candidates too;
    # only the stricter independent shadow metrics below can approve a regime.
    fresh = [x for x in archive
             if x.get("id") and isinstance(x.get("spec"), dict)]
    fresh.sort(key=lambda x: (
        x.get("status") == "validation_pass",
        float(x.get("score", -100.0))
    ), reverse=True)
    candidates = list(fresh)
    if champion and champion.get("id") and isinstance(champion.get("spec"), dict):
        if all(x.get("id") != champion.get("id") for x in candidates):
            candidates.append(champion)

    # Continue monitoring prior experiments without allowing them to crowd
    # out this cycle's fresh candidates.
    for sid, spec in memory["strategy_specs"].items():
        if all(x.get("id") != sid for x in candidates):
            candidates.append({"id": sid, "spec": spec, "status": "paper_candidate"})
    candidates = candidates[:MAX_CANDIDATES]
    for item in candidates:
        if item.get("id") and isinstance(item.get("spec"), dict):
            memory["strategy_specs"][item["id"]] = item["spec"]

    candidate_status_counts = defaultdict(int)
    for item in candidates:
        candidate_status_counts[str(item.get("status", "paper_candidate"))] += 1

    market = {}
    for symbol in list(universe.index[:max_symbols]):
        try:
            df = get_candles(symbol, "1H", 1200)
            if df is None or len(df) < 100:
                continue
            df = df.reset_index(drop=True)
            feat = build_features(df)
            regime = market_regime(feat)
            last = len(df) - 1
            bar = str(df.iloc[last].get("ts", last))
            close = float(df.iloc[last]["c"])
            high = float(df.iloc[last]["h"])
            low = float(df.iloc[last]["l"])
            atr_pct = float(feat.iloc[last]["atr_pct"])
            if not np.isfinite(close + high + low + atr_pct) or close <= 0 or atr_pct <= 0:
                continue
            market[str(symbol)] = {"df": df, "features": feat, "regime": regime,
                                   "bar": bar, "close": close, "high": high,
                                   "low": low, "atr_pct": atr_pct}
        except Exception:
            continue

    # First mature existing shadow positions; each bar is counted once.
    for key, pos in list(memory["positions"].items()):
        item = market.get(pos["symbol"])
        if not item:
            continue
        if item["bar"] == pos.get("last_bar"):
            continue
        pos["last_bar"] = item["bar"]
        pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
        sl_hit = item["low"] <= float(pos["stop_price"])
        tp_hit = item["high"] >= float(pos["target_price"])
        if sl_hit or tp_hit:
            # Conservative same-candle assumption: stop wins if both barriers touch.
            price = float(pos["stop_price"] if sl_hit else pos["target_price"])
            _close_position(pos, price, "SL" if sl_hit else "TP", item["bar"], memory["closed_trades"])
            del memory["positions"][key]
        elif pos["bars_held"] >= int(pos["max_hold"]):
            _close_position(pos, item["close"], "TIME_EXIT", item["bar"], memory["closed_trades"])
            del memory["positions"][key]

    # Fill queued signals at the next observed hourly candle open.
    for key, pending in list(memory["pending_entries"].items()):
        item = market.get(pending["symbol"])
        if not item or item["bar"] == pending.get("signal_bar"):
            continue
        entry = float(item["df"].iloc[-1]["o"])
        if not np.isfinite(entry) or entry <= 0:
            del memory["pending_entries"][key]
            continue
        atr = float(pending["atr_pct"])
        spec = memory["strategy_specs"].get(pending["strategy_id"], {})
        memory["positions"][key] = {
            "strategy_id": pending["strategy_id"], "symbol": pending["symbol"],
            "entry_regime": pending["entry_regime"], "entry_price": entry,
            "stop_price": entry * (1.0 - float(spec.get("sl_atr", 1.5)) * atr),
            "target_price": entry * (1.0 + float(spec.get("tp_atr", 2.0)) * atr),
            "max_hold": int(spec.get("max_hold", 24)), "bars_held": 0,
            "entry_bar": item["bar"], "last_bar": item["bar"], "opened_at": time.time()
        }
        del memory["pending_entries"][key]

    # New signals are queued, never filled on the candle that generated them.
    signal_checks = 0
    signals_detected = 0
    candidate_hits = defaultdict(int)
    for symbol, item in market.items():
        for candidate in candidates:
            sid, spec = candidate.get("id"), candidate.get("spec")
            if not sid or not isinstance(spec, dict):
                continue
            key = sid + "|" + symbol
            if (key in memory["positions"] or key in memory["pending_entries"]
                    or memory["last_seen_bar"].get(key) == item["bar"]):
                continue
            try:
                signal_checks += 1
                hit = bool(_signals(item["features"], spec)[-1])
                if hit:
                    signals_detected += 1
                    candidate_hits[sid] += 1
                    memory["pending_entries"][key] = {
                        "strategy_id": sid, "symbol": symbol, "entry_regime": item["regime"],
                        "signal_bar": item["bar"], "atr_pct": item["atr_pct"]
                    }
                memory["last_seen_bar"][key] = item["bar"]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
    # Learn a separate performance profile for every strategy x entry regime.
    groups = defaultdict(list)
    for trade in memory["closed_trades"]:
        groups[(trade["strategy_id"], trade["entry_regime"])].append(trade)
    profiles = {}
    eligible = defaultdict(list)
    for (sid, regime), trades in groups.items():
        m = _metrics(trades)
        passed = (m["trades"] >= MIN_SHADOW_TRADES
                  and m["avg_net"] is not None and m["avg_net"] > MIN_EXPECTANCY
                  and m["profit_factor"] >= MIN_PROFIT_FACTOR
                  and m["max_drawdown"] <= MAX_DRAWDOWN)
        profiles.setdefault(sid, {})[regime] = {**m, "paper_approved": bool(passed)}
        if passed:
            eligible[regime].append({"strategy_id": sid, "spec": memory["strategy_specs"].get(sid, {}), **m})

    # A selector proposal is a top-three ensemble by matured shadow performance.
    # Coin-specific evidence is shrunk toward its regime-wide prior to limit overfit.
    selected = {}
    for regime, rows in eligible.items():
        rows.sort(key=lambda x: (x["avg_net"], x["profit_factor"], -x["max_drawdown"]), reverse=True)
        selected[regime] = rows[:MAX_ACTIVE_STRATEGIES]

    regime_avg = {}
    for regime in {x["entry_regime"] for x in memory["closed_trades"]}:
        values = [float(x["net_return"]) for x in memory["closed_trades"]
                  if x["entry_regime"] == regime]
        if values:
            regime_avg[regime] = float(np.mean(values))
    symbol_groups = defaultdict(list)
    for trade in memory["closed_trades"]:
        symbol_groups[(trade["strategy_id"], trade["entry_regime"], trade["symbol"])].append(trade)
    symbol_profiles = {}
    selected_by_symbol_regime = defaultdict(list)
    for (sid, regime, symbol), trades in symbol_groups.items():
        m = _metrics(trades)
        prior = regime_avg.get(regime, 0.0)
        shrunk_avg = _symbol_adjusted_expectancy(m["avg_net"] or 0.0, prior, m["trades"])
        regime_approved = any(x["strategy_id"] == sid for x in eligible.get(regime, []))
        passed = (regime_approved and m["trades"] >= MIN_SYMBOL_SHADOW_TRADES
                  and shrunk_avg > MIN_EXPECTANCY
                  and m["profit_factor"] >= MIN_PROFIT_FACTOR
                  and m["max_drawdown"] <= MAX_DRAWDOWN)
        profile = {**m, "shrunk_avg_net": shrunk_avg,
                   "prior_regime_avg_net": prior, "paper_approved": bool(passed)}
        symbol_profiles.setdefault(symbol, {}).setdefault(regime, {})[sid] = profile
        if passed:
            selected_by_symbol_regime[(symbol, regime)].append({"strategy_id": sid, "spec": memory["strategy_specs"].get(sid, {}), **profile})
    selected_by_symbol_regime = {
        f"{symbol}|{regime}": sorted(rows, key=lambda x: (
            x["shrunk_avg_net"], x["profit_factor"], -x["max_drawdown"]), reverse=True
        )[:MAX_ACTIVE_STRATEGIES]
        for (symbol, regime), rows in selected_by_symbol_regime.items()
    }

    memory["closed_trades"] = memory["closed_trades"][-10000:]
    memory["last_report"] = {
        "status": "ok", "cycle": memory["cycle"], "updated_at": time.time(), "market_regimes": {
            s: x["regime"] for s, x in market.items()},
        "candidates_monitored": len(candidates),
        "candidate_status_counts": dict(candidate_status_counts),
        "symbols_with_market_data": len(market),
        "signal_checks": int(signal_checks),
        "signals_detected": int(signals_detected),
        "candidate_signal_hits": dict(candidate_hits),
        "open_shadow_positions": len(memory["positions"]),
        "pending_shadow_entries": len(memory["pending_entries"]),
        "closed_shadow_trades": len(memory["closed_trades"]),
        "profiles": profiles, "selected_by_regime": selected,
        "symbol_profiles": symbol_profiles,
        "selected_by_symbol_regime": selected_by_symbol_regime,
        "approval_rules": {"min_trades": MIN_SHADOW_TRADES,
                           "min_symbol_trades": MIN_SYMBOL_SHADOW_TRADES,
                           "symbol_prior_strength": SYMBOL_PRIOR_STRENGTH,
                           "min_profit_factor": MIN_PROFIT_FACTOR,
                           "min_expectancy": MIN_EXPECTANCY,
                           "max_drawdown": MAX_DRAWDOWN},
        "live_integration": "bounded_primary_ai_probability_overlay_only",
        "ai_feedback_ready": bool(selected_by_symbol_regime),
        "ai_feedback_source": "matured_symbol_and_regime_shadow_trades_only",
        "note": "Approved symbol+regime shadow profiles may softly calibrate primary AI probabilities through the bounded evolution bridge; no model weights, labels, positions, orders, or risk controls are changed."
    }
    _save(memory_path, memory)
    _save(report_path, memory["last_report"])
    if telegram:
        approved = sum(len(v) for v in selected.values())
        telegram("🧠 Strategy Meta-Selector (shadow only)\n"
                 f"Cycle: {memory['cycle']} | Candidates: {len(candidates)}\n"
                 f"Open shadow: {len(memory['positions'])} | Closed: {len(memory['closed_trades'])}\n"
                 f"Regime approvals: {approved} | Primary AI overlay: symbol+regime approved shadow only")
    return memory["last_report"]
