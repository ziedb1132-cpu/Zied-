#!/usr/bin/env python3
"""Research-only strategy generator: mutate rule parameters, then evaluate chronologically.

This module NEVER places orders and does not change the live model's entries. It records
an experimental champion only after an untouched holdout clears conservative net-cost gates.
"""
import itertools
import json
import os
import time
import numpy as np
import pandas as pd

COST = 0.0020  # round-trip fees + slippage reserve; conservative fixed 20 bps
HOLD = 24
MIN_TRADES = 30


def _features(df):
    c = pd.to_numeric(df["c"], errors="coerce")
    h = pd.to_numeric(df["h"], errors="coerce")
    l = pd.to_numeric(df["l"], errors="coerce")
    v = pd.to_numeric(df.get("vol", pd.Series(np.ones(len(df)), index=df.index)), errors="coerce")
    ema_fast = c.ewm(span=12, adjust=False).mean()
    ema_slow = c.ewm(span=36, adjust=False).mean()
    ema_long = c.ewm(span=72, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    prior_high = h.shift(1).rolling(24, min_periods=24).max()
    prior_low = l.shift(1).rolling(24, min_periods=24).min()
    vol_mean = v.shift(1).rolling(24, min_periods=24).mean()
    return c, h, l, v, ema_fast, ema_slow, ema_long, rsi, prior_high, prior_low, vol_mean


def candidate_specs():
    """Small, explainable mutations rather than an unconstrained parameter explosion."""
    specs = []
    for fast, slow, rsi_min in itertools.product((8, 12, 16), (30, 36, 48), (48, 52, 56)):
        if fast < slow:
            specs.append({"kind": "trend_pullback", "fast": fast, "slow": slow, "rsi_min": rsi_min})
    for lookback, vol_mult in itertools.product((12, 24, 36, 48), (1.0, 1.25, 1.5)):
        specs.append({"kind": "breakout", "lookback": lookback, "vol_mult": vol_mult})
    for rsi_low in (20, 25, 30, 35):
        specs.append({"kind": "rsi_reversal", "rsi_low": rsi_low})
    return specs


def signal_for(df, spec):
    c, h, l, v, ef, es, el, rsi, ph, pl, vm = _features(df)
    kind = spec["kind"]
    if kind == "trend_pullback":
        fast = c.ewm(span=spec["fast"], adjust=False).mean()
        slow = c.ewm(span=spec["slow"], adjust=False).mean()
        sig = (fast > slow) & (c > el) & (rsi >= spec["rsi_min"]) & (rsi < 72)
    elif kind == "breakout":
        high = h.shift(1).rolling(spec["lookback"], min_periods=spec["lookback"]).max()
        sig = (c > high) & (v > vm * spec["vol_mult"]) & (c > el)
    else:
        low = spec["rsi_low"]
        sig = (rsi.shift(1) < low) & (rsi >= low) & (c > es)
    return sig.fillna(False).to_numpy(dtype=bool)


def evaluate(df, signals, start, end):
    """Non-overlapping entries, next-bar open entry, fixed 24h exit, costs deducted."""
    n = len(df)
    op = pd.to_numeric(df["o"], errors="coerce").to_numpy(float)
    cl = pd.to_numeric(df["c"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(df["h"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["l"], errors="coerce").to_numpy(float)
    rets, i, last_exit = [], max(0, int(start)), -1
    end = min(int(end), n - HOLD - 1)
    while i < end:
        # Signal at candle i close, enter next candle open, exit by 24 bars.
        if signals[i] and i > last_exit and np.isfinite(op[i+1]) and op[i+1] > 0:
            entry = op[i+1]
            exit_i = i + HOLD
            if exit_i >= n or not np.isfinite(cl[exit_i]):
                break
            # Conservative fixed-horizon mark-to-market, net of costs.
            rets.append(cl[exit_i] / entry - 1.0 - COST)
            last_exit = exit_i
            i = exit_i
        else:
            i += 1
    x = np.asarray(rets, dtype=float)
    wins = x[x > 0].sum()
    losses = -x[x < 0].sum()
    return {"trades": int(len(x)), "avg_net": float(x.mean()) if len(x) else None,
            "win_rate": float((x > 0).mean()) if len(x) else None,
            "profit_factor": float(wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0)}


def evaluate_candidate(frames, spec):
    """Aggregate multiple coins, then use strictly chronological train/tune/holdout windows."""
    totals = {"train": [], "tune": [], "test": []}
    for df in frames:
        if df is None or len(df) < 300:
            continue
        sig = signal_for(df, spec)
        n = len(df)
        # Boundaries have a 24-bar embargo to reduce label overlap across splits.
        a, b = int(n * .60), int(n * .80)
        for name, start, end in (
            ("train", 80, a - HOLD),
            ("tune", a + HOLD, b - HOLD),
            ("test", b + HOLD, n - HOLD - 1),
        ):
            m = evaluate(df, sig, start, end)
            # Keep returns unavailable at this summary layer; aggregate net metrics
            # by collecting the split-level metrics with trade-count weighting.
            if m["trades"] and m["avg_net"] is not None:
                totals[name].append(m)
    out = {}
    for name, chunks in totals.items():
        ntr = sum(x["trades"] for x in chunks)
        if not ntr:
            out[name] = {"trades": 0, "avg_net": None, "win_rate": None, "profit_factor": 0.0}
            continue
        avg = sum(x["avg_net"] * x["trades"] for x in chunks) / ntr
        win = sum(x["win_rate"] * x["trades"] for x in chunks) / ntr
        # Conservative aggregate PF proxy from positive/negative average components.
        gross_win = sum(max(x["avg_net"], 0) * x["trades"] for x in chunks)
        gross_loss = sum(max(-x["avg_net"], 0) * x["trades"] for x in chunks)
        pf = gross_win / gross_loss if gross_loss else (999.0 if gross_win else 0.0)
        out[name] = {"trades": int(ntr), "avg_net": float(avg), "win_rate": float(win), "profit_factor": float(pf)}
    return out


def run_evolution(universe, get_candles, state_dir, telegram=None, max_coins=8):
    """Download 1H candles for liquid symbols and save the best research candidate."""
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "strategy_evolution.json")
    now = time.time()
    try:
        old = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if now - float(old.get("last_run", 0)) < 23 * 3600:
        return {"status": "skipped_recent", "champion": old.get("champion")}
    frames, symbols = [], []
    for inst in list(universe.index[:max_coins]):
        try:
            d = get_candles(inst, "1H", 1200)
            if d is not None and len(d) >= 300:
                frames.append(d.reset_index(drop=True))
                symbols.append(str(inst))
        except Exception:
            continue
    if len(frames) < 3:
        result = {"status": "insufficient_data", "last_run": now, "symbols": symbols,
                  "message": "Need at least 3 symbols with 300+ hourly candles."}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
        return result
    results = []
    for spec in candidate_specs():
        splits = evaluate_candidate(frames, spec)
        results.append({"spec": spec, **splits})
    # Only tuning can rank candidates; holdout is consulted once for the selected candidate.
    eligible = [r for r in results if r["tune"]["trades"] >= MIN_TRADES
                and r["tune"]["avg_net"] is not None and r["tune"]["avg_net"] > 0
                and r["tune"]["profit_factor"] > 1.05]
    eligible.sort(key=lambda r: (r["tune"]["avg_net"], r["tune"]["profit_factor"]), reverse=True)
    champion = eligible[0] if eligible else None
    accepted = bool(champion and champion["test"]["trades"] >= MIN_TRADES
                    and champion["test"]["avg_net"] is not None and champion["test"]["avg_net"] > 0
                    and champion["test"]["profit_factor"] > 1.05)
    previous = old.get("champion")
    if not accepted:
        champion_to_store = previous
    else:
        champion_to_store = {"spec": champion["spec"], "tune": champion["tune"], "test": champion["test"],
                             "accepted_at": now, "symbols": symbols, "cost_round_trip": COST,
                             "max_hold_hours": HOLD}
    result = {"status": "accepted" if accepted else "no_candidate_passed",
              "last_run": now, "symbols": symbols, "candidate_count": len(results),
              "eligible_on_tuning": len(eligible), "latest_tested": champion,
              "champion": champion_to_store,
              "note": "Research-only: does not alter live entries. Holdout gate requires >=30 trades, positive net expectancy after 20 bps costs, PF > 1.05."}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
    if telegram:
        if accepted:
            s = champion["spec"]
            telegram("🧬 استراتيجية تجريبية جديدة اجتازت الاختبار المنفصل\n"
                     f"القاعدة: {s}\n"
                     f"اختبار holdout: {champion['test']['trades']} صفقة | "
                     f"متوسط صافٍ {champion['test']['avg_net']*100:.3f}% | "
                     f"PF {champion['test']['profit_factor']:.2f}\n"
                     "هذه نتيجة بحثية فقط ولا تغيّر إشارات التداول الحية.")
        else:
            telegram(f"🧬 تطور الاستراتيجيات: لم تجتز أي استراتيجية شروط الاختبار المنفصل. "
                     f"تم فحص {len(results)} تركيبة؛ لم يتم استبدال الاستراتيجية السابقة.")
    return result
