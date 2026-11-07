#!/usr/bin/env python3
"""Autonomous symbolic strategy discovery for Zied Quant.

Research-only: generated strategies are evaluated and persisted, but NEVER wired
into live entries or allowed to modify the existing AI model/portfolio controls.
"""
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd

COST = 0.0035
HORIZON = 24
BATCH_SIZE = 10
MIN_TRADES = 20
MAX_ARCHIVE = 2000
LEGACY_FEATURES = (
    "ret1", "ret3", "ret6", "ret12", "rsi", "ema_gap", "ema_slope",
    "atr_pct", "range_pos", "volume_ratio", "body_pct", "upper_wick",
    "lower_wick", "breakout_dist", "close_location",
)
LEGACY_THRESHOLDS = {
    "ret1": (-0.025, -0.01, 0.0, 0.01, 0.025),
    "ret3": (-0.04, -0.02, 0.0, 0.02, 0.04),
    "ret6": (-0.06, -0.03, 0.0, 0.03, 0.06),
    "ret12": (-0.10, -0.05, 0.0, 0.05, 0.10),
    "rsi": (25.0, 35.0, 45.0, 50.0, 55.0, 65.0, 75.0),
    "ema_gap": (-0.03, -0.01, 0.0, 0.01, 0.03),
    "ema_slope": (-0.02, -0.01, 0.0, 0.01, 0.02),
    "atr_pct": (0.005, 0.01, 0.02, 0.04, 0.06),
    "range_pos": (0.2, 0.35, 0.5, 0.65, 0.8),
    "volume_ratio": (0.7, 1.0, 1.25, 1.5, 2.0, 3.0),
    "body_pct": (-0.03, -0.01, 0.0, 0.01, 0.03),
    "upper_wick": (0.1, 0.25, 0.4, 0.6, 0.8),
    "lower_wick": (0.1, 0.25, 0.4, 0.6, 0.8),
    "breakout_dist": (-0.03, -0.01, 0.0, 0.01, 0.03),
    "close_location": (0.2, 0.35, 0.5, 0.65, 0.8),
}

from feature_library import (
    PERIODS as FEATURE_PERIODS,
    ROLL_WINDOWS as FEATURE_ROLL_WINDOWS,
    LIBRARY_VERSION as FEATURE_LIBRARY_VERSION,
    build_feature_library,
    feature_thresholds,
)

# The 546 generated features are selectable by the genetic strategy builder.
# Keep the original 15 names available so archived strategies remain readable.
_FEATURE_FAMILIES = (
    "ret", "rsi", "ema_gap", "ema_slope", "atr_pct", "range_pos",
    "volume_ratio", "body_pct", "upper_wick", "lower_wick",
    "breakout_dist", "close_location", "downside_vol", "upside_vol",
    "efficiency", "volume_trend", "vwap_dist", "drawdown",
)
_CORE_CONCEPTS = LEGACY_FEATURES
_LIBRARY_FEATURES = (
    [f"{family}_{period}" for family in _FEATURE_FAMILIES for period in FEATURE_PERIODS]
    + [f"{name}__{kind}{window}" for name in _CORE_CONCEPTS
       for window in FEATURE_ROLL_WINDOWS for kind in ("z", "rank")]
    + [f"interaction__{left}__{right}" for i, left in enumerate(_CORE_CONCEPTS)
       for right in _CORE_CONCEPTS[i + 1:]]
    + [f"core__{name}" for name in _CORE_CONCEPTS]
)
FEATURES = tuple(LEGACY_FEATURES) + tuple(_LIBRARY_FEATURES)
THRESHOLDS = {
    **LEGACY_THRESHOLDS,
    **feature_thresholds(_LIBRARY_FEATURES),
}



def _build_core_features(df):
    """Causal OHLCV feature set; all rolling references use current/past bars only."""
    o = pd.to_numeric(df["o"], errors="coerce")
    h = pd.to_numeric(df["h"], errors="coerce")
    l = pd.to_numeric(df["l"], errors="coerce")
    c = pd.to_numeric(df["c"], errors="coerce")
    v = pd.to_numeric(df.get("vol", pd.Series(1.0, index=df.index)), errors="coerce")
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-12))
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    prior_high = h.shift(1).rolling(20, min_periods=10).max()
    candle_range = (h-l).replace(0, np.nan)
    return pd.DataFrame({
        "ret1": c.pct_change(1), "ret3": c.pct_change(3),
        "ret6": c.pct_change(6), "ret12": c.pct_change(12),
        "rsi": rsi, "ema_gap": ema20/(ema50+1e-12)-1,
        "ema_slope": ema20.pct_change(5), "atr_pct": atr/(c+1e-12),
        "range_pos": (c-l.rolling(20, min_periods=10).min()) /
                     (h.rolling(20, min_periods=10).max()-l.rolling(20, min_periods=10).min()+1e-12),
        "volume_ratio": v/(v.shift(1).rolling(20, min_periods=10).mean()+1e-12),
        "body_pct": (c-o)/(o+1e-12),
        "upper_wick": (h-pd.concat([o,c],axis=1).max(axis=1))/(candle_range+1e-12),
        "lower_wick": (pd.concat([o,c],axis=1).min(axis=1)-l)/(candle_range+1e-12),
        "breakout_dist": c/(prior_high+1e-12)-1,
        "close_location": (c-l)/(candle_range+1e-12),
    }, index=df.index).replace([np.inf,-np.inf],np.nan)

def build_features(df):
    """Return legacy features plus the complete 546-column causal library."""
    core = _build_core_features(df)
    expanded = build_feature_library(df, core=core)
    combined = pd.concat([core, expanded], axis=1)
    # Validate the actual names, not only the column count. A same-sized but
    # differently named matrix would otherwise be silently converted to NaNs
    # by the model feature-alignment layer.
    if not combined.columns.is_unique:
        raise ValueError("Duplicate feature names in autonomous feature matrix")
    expected = set(_LIBRARY_FEATURES)
    actual = set(expanded.columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if len(_LIBRARY_FEATURES) != 546 or len(expanded.columns) != 546 or missing or unexpected:
        raise RuntimeError(
            "Feature library schema mismatch: "
            f"expected=546 actual={len(expanded.columns)} "
            f"missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    return combined



def _spec_id(spec):
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _new_spec(rng, generation, parent=None, mutation_rate=0.40):
    mutation_rate = min(0.80, max(0.15, float(mutation_rate)))
    count = rng.randint(2, 7)
    conditions = []
    used = set()
    if parent and rng.random() < 0.75:
        parent_spec = parent.get("spec", parent)
        conditions = [dict(x) for x in parent_spec.get("conditions", [])]
        # Mutate, remove, or add conditions; no fixed strategy-complexity ceiling.
        if conditions and rng.random() < mutation_rate * 0.75:
            conditions.pop(rng.randrange(len(conditions)))
        if conditions and rng.random() < mutation_rate:
            item = conditions[rng.randrange(len(conditions))]
            feature = item["feature"]
            item["threshold"] = rng.choice(THRESHOLDS[feature])
            item["op"] = rng.choice(["gt", "lt"])
        count = max(2, count + len(conditions))
    # Repeated features are allowed with different thresholds/operators; this
    # permits arbitrarily deep rule trees without a hard condition-count ceiling.
    while len(conditions) < count:
        feature = rng.choice(FEATURES)
        conditions.append({"feature": feature, "op": rng.choice(["gt", "lt"]),
                           "threshold": rng.choice(THRESHOLDS[feature])})
    rng.shuffle(conditions)
    spec = {
        "version": 1, "kind": "symbolic_rule",
        "conditions": conditions,
        "sl_atr": rng.choice([1.0, 1.25, 1.5, 1.75, 2.0, 2.5]),
        "tp_atr": rng.choice([1.25, 1.5, 2.0, 2.5, 3.0, 4.0]),
        "max_hold": rng.choice([6, 12, 18, 24, 36, 48]),
        "generation": int(generation),
        "parent_id": parent.get("id") if parent else None,
    }
    spec["id"] = _spec_id({k:v for k,v in spec.items() if k not in ("generation","parent_id","id")})
    return spec



def _crossover_specs(rng, parent_a, parent_b, generation, mutation_rate=0.40):
    """Combine two parent genomes and adapt mutation pressure to recent progress."""
    mutation_rate = min(0.80, max(0.15, float(mutation_rate)))
    a = parent_a.get("spec", parent_a)
    b = parent_b.get("spec", parent_b)
    left = [dict(x) for x in a.get("conditions", [])]
    right = [dict(x) for x in b.get("conditions", [])]
    pool = []
    for cond in left + right:
        if cond not in pool and rng.random() < 0.5:
            pool.append(cond)
    if len(pool) < 2:
        pool = (left[:1] + right[:1]) or left or right
    rng.shuffle(pool)
    conditions = pool[:12]
    child = {
        "version": 1, "kind": "symbolic_rule", "conditions": conditions,
        "sl_atr": rng.choice([a.get("sl_atr", 1.5), b.get("sl_atr", 1.5)]),
        "tp_atr": rng.choice([a.get("tp_atr", 2.0), b.get("tp_atr", 2.0)]),
        "max_hold": rng.choice([a.get("max_hold", 24), b.get("max_hold", 24)]),
        "generation": int(generation),
        "parent_id": [parent_a.get("id"), parent_b.get("id")],
        "crossover": True,
    }
    # Mutate input-feature genes and trading-policy genes after crossover.
    if rng.random() < mutation_rate:
        i = rng.randrange(len(child["conditions"]))
        feature = rng.choice(FEATURES)
        child["conditions"][i] = {"feature": feature, "op": rng.choice(["gt", "lt"]),
                                  "threshold": rng.choice(THRESHOLDS[feature])}
    if rng.random() < mutation_rate * 0.65 and len(child["conditions"]) < 12:
        feature = rng.choice(FEATURES)
        child["conditions"].append({"feature": feature, "op": rng.choice(["gt", "lt"]),
                                    "threshold": rng.choice(THRESHOLDS[feature])})
    if rng.random() < mutation_rate * 0.50 and len(child["conditions"]) > 2:
        child["conditions"].pop(rng.randrange(len(child["conditions"])))
    if rng.random() < mutation_rate:
        child["sl_atr"] = rng.choice([1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
    if rng.random() < mutation_rate:
        child["tp_atr"] = rng.choice([1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    if rng.random() < mutation_rate:
        child["max_hold"] = rng.choice([6, 12, 18, 24, 36, 48])
    child["id"] = _spec_id({k: v for k, v in child.items()
                            if k not in ("generation", "parent_id", "id", "crossover")})
    child["input_features"] = sorted({x["feature"] for x in child["conditions"]})
    return child


def _genome_summary(spec):
    conditions = spec.get("conditions", [])
    return {
        "input_feature_count": len({x.get("feature") for x in conditions}),
        "input_features": sorted({x.get("feature") for x in conditions}),
        "condition_count": len(conditions),
        "sl_atr": spec.get("sl_atr"), "tp_atr": spec.get("tp_atr"),
        "max_hold": spec.get("max_hold"),
        "crossover": bool(spec.get("crossover", False)),
        "parent_id": spec.get("parent_id"),
    }

def _signals(features, spec):
    mask = np.ones(len(features), dtype=bool)
    for cond in spec["conditions"]:
        values = features[cond["feature"]].to_numpy(float)
        valid = np.isfinite(values)
        mask &= valid & ((values > cond["threshold"]) if cond["op"] == "gt"
                         else (values < cond["threshold"]))
    return mask


def _evaluate(df, features, spec, start, end):
    op = pd.to_numeric(df["o"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(df["h"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["l"], errors="coerce").to_numpy(float)
    cl = pd.to_numeric(df["c"], errors="coerce").to_numpy(float)
    atr = features["atr_pct"].to_numpy(float)
    sig = _signals(features, spec)
    hold = int(spec["max_hold"])
    end = min(int(end), len(df)-2)
    rets, i, last_exit = [], max(60, int(start)), -1
    while i < end:
        if not sig[i] or i <= last_exit or not np.isfinite(op[i+1]) or op[i+1] <= 0:
            i += 1
            continue
        entry_i = i+1
        entry = op[entry_i]
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue
        sl = entry*(1-float(spec["sl_atr"])*a)
        tp = entry*(1+float(spec["tp_atr"])*a)
        stop_i = min(len(df)-1, entry_i+hold-1)
        exit_price = cl[stop_i]
        exit_i = stop_i
        for j in range(entry_i, stop_i+1):
            # Conservative same-candle handling: if both barriers touched, assume SL.
            if lo[j] <= sl:
                exit_price, exit_i = sl, j
                break
            if hi[j] >= tp:
                exit_price, exit_i = tp, j
                break
        ret = exit_price/entry-1-COST
        if np.isfinite(ret):
            rets.append(float(ret))
        last_exit = exit_i
        i = exit_i+1
    x = np.asarray(rets, dtype=float)
    wins, losses = float(x[x>0].sum()), float(-x[x<0].sum())
    pf = wins/losses if losses > 0 else (999.0 if wins > 0 else 0.0)
    if not len(x):
        return {"trades":0,"avg_net":None,"win_rate":None,"profit_factor":0.0,
                "max_drawdown":None,"sharpe":None}
    equity = np.r_[1.0, np.cumprod(1+x)]
    peaks = np.maximum.accumulate(equity)
    dd = equity/peaks-1
    std = float(x.std(ddof=1)) if len(x)>1 else 0.0
    return {"trades":int(len(x)),"avg_net":float(x.mean()),
            "win_rate":float((x>0).mean()),"profit_factor":float(pf),
            "gross_profit":float(wins),"gross_loss":float(losses),
            "max_drawdown":float(abs(dd.min())),"sharpe":float(x.mean()/std*np.sqrt(len(x))) if std>0 else 0.0}


def _aggregate(rows):
    valid=[r for r in rows if r["trades"]]
    n=sum(r["trades"] for r in valid)
    if not n:
        return {"trades":0,"avg_net":None,"win_rate":None,"profit_factor":0.0,
                "max_drawdown":None,"sharpe":None}
    avg=sum(r["avg_net"]*r["trades"] for r in valid)/n
    win=sum(r["win_rate"]*r["trades"] for r in valid)/n
    gp=sum(float(r.get("gross_profit",0.0)) for r in valid)
    gl=sum(float(r.get("gross_loss",0.0)) for r in valid)
    return {"trades":n,"avg_net":float(avg),"win_rate":float(win),
            "profit_factor":float(gp/gl) if gl>0 else (999.0 if gp>0 else 0.0),
            "gross_profit":float(gp),"gross_loss":float(gl),
            "max_drawdown":float(max((r["max_drawdown"] or 0) for r in valid)),
            "sharpe":float(np.mean([r["sharpe"] or 0 for r in valid]))}


def _score(metrics, complexity, stability):
    if metrics["trades"] < MIN_TRADES or metrics["avg_net"] is None:
        return -100.0, {"insufficient_trades":True}
    avg = metrics["avg_net"]
    pf = min(metrics["profit_factor"], 3.0)
    dd = metrics["max_drawdown"] or 1.0
    sharpe = max(-2.0,min(3.0,metrics["sharpe"] or 0.0))
    # Transparent research score; complexity and cross-symbol instability are penalized.
    raw = (35*max(-1,min(1,avg*100)) + 20*max(0,min(1,(pf-1)/1.5))
           + 15*max(0,min(1,sharpe/2)) + 15*max(0,1-dd/0.30)
           + 15*max(0,1-stability))
    penalty = min(35.0, 1.5*max(0,complexity-2))
    return float(raw-penalty), {"complexity_penalty":float(penalty),
                                "instability_penalty":float(15*min(1,stability))}


def _load(path):
    try:
        with open(path,encoding="utf-8") as f:
            obj=json.load(f)
        if isinstance(obj,dict):
            return obj
    except (OSError,ValueError):
        pass
    return {"schema":1,"generation":0,"experiments":[],"archive":[],"champion":None}


def run_discovery(universe, get_candles, state_dir, telegram=None, batch_size=BATCH_SIZE, max_coins=5):
    """Generate ten fresh symbolic candidates per cycle, evaluate, and persist all research state."""
    os.makedirs(state_dir,exist_ok=True)
    path=os.path.join(state_dir,"autonomous_strategy_memory.json")
    memory=_load(path)
    generation=int(memory.get("generation",0))+1
    governance = _load(os.path.join(state_dir, "evolution_governance.json"))
    mutation_rate = float(governance.get("mutation_rate", memory.get("mutation_rate", 0.40)))
    mutation_rate = min(0.80, max(0.15, mutation_rate))
    rng=random.Random(int(memory.get("seed",1729))+generation*7919)
    frames=[]; symbols=[]
    for inst in list(universe.index[:max_coins]):
        try:
            d=get_candles(inst,"1H",1200)
            if d is not None and len(d)>=400:
                frames.append(d.reset_index(drop=True)); symbols.append(str(inst))
        except Exception:
            continue
    if len(frames)<3:
        result={"status":"insufficient_data","generation":generation,
                "symbols":symbols,"batch_size":int(batch_size),
                "message":"Need at least 3 symbols with 400+ hourly candles."}
        memory.update({"generation":generation,"last_run":time.time(),"last_result":result})
        _atomic_write(path,memory)
        _atomic_write(os.path.join(state_dir,"autonomous_strategy_report.json"),result)
        return result
    features=[build_features(d) for d in frames]
    archive=memory.get("archive",[])
    parents=sorted([x for x in archive if x.get("status") in ("validation_pass","paper_candidate")],
                   key=lambda x:float(x.get("score",-100)),reverse=True)[:20]
    known={x.get("id") for x in archive}
    candidates=[]
    attempts=0
    while len(candidates)<int(batch_size) and attempts<int(batch_size)*40:
        attempts+=1
        if len(parents) >= 2 and rng.random() < 0.60:
            p1, p2 = rng.sample(parents, 2)
            spec = _crossover_specs(rng, p1, p2, generation, mutation_rate)
        else:
            parent = rng.choice(parents) if parents and rng.random() < 0.75 else None
            spec = _new_spec(rng, generation, parent, mutation_rate)
            spec["input_features"] = sorted({x["feature"] for x in spec["conditions"]})
        if spec["id"] not in known and all(x["id"] != spec["id"] for x in candidates):
            candidates.append(spec)
    results=[]
    for spec in candidates:
        train_rows=[]; tune_rows=[]; hold_rows=[]
        per_symbol=[]
        for df,feat in zip(frames,features):
            n=len(df)
            a=_evaluate(df,feat,spec,int(n*.10),int(n*.60))
            b=_evaluate(df,feat,spec,int(n*.62),int(n*.80))
            c=_evaluate(df,feat,spec,int(n*.82),n-50)
            train_rows.append(a); tune_rows.append(b); hold_rows.append(c)
            per_symbol.append({"train":a,"tune":b,"holdout":c})
        train=_aggregate(train_rows); tune=_aggregate(tune_rows)
        # Use cross-symbol dispersion of tuning expectancy as a stability penalty.
        avgs=[x["avg_net"] for x in tune_rows if x["avg_net"] is not None]
        stability=float(np.std(avgs)/(abs(np.mean(avgs))+1e-6)) if len(avgs)>1 else 1.0
        score,penalties=_score(tune,len(spec["conditions"]),stability)
        # Holdout is reported for monitoring, but never used to rank, mutate, or select parents.
        hold=_aggregate(hold_rows)
        status="rejected"
        if (train["trades"]>=MIN_TRADES and tune["trades"]>=MIN_TRADES
                and tune["avg_net"] is not None and tune["avg_net"]>0
                and tune["profit_factor"]>1.05 and score>25):
            status="validation_pass"
        row={"id":spec["id"],"spec":spec,"genome":_genome_summary(spec),"generation":generation,"status":status,
             "score":score,"score_details":penalties,"train":train,"validation":tune,
             "holdout_observation":hold,"per_symbol":per_symbol,
             "symbols":symbols,"cost_round_trip":COST,"created_at":time.time()}
        results.append(row)
    archive.extend(results)
    archive=archive[-MAX_ARCHIVE:]
    # Elitist breeding pool persists across workflow restarts.
    valid=sorted(
        [x for x in archive if x.get("status")=="validation_pass"],
        key=lambda x:float(x.get("score",-100)),reverse=True
    )
    elite_count=max(2, min(20, int(math.ceil(len(valid)*0.10)))) if valid else 0
    population=valid[:elite_count]
    champion=max(valid,key=lambda x:float(x.get("score",-100))) if valid else memory.get("champion")
    result={"status":"ok","generation":generation,"generated":len(candidates),
            "tested":len(results),"validation_passed":sum(x["status"]=="validation_pass" for x in results),
            "archive_size":len(archive),"symbols":symbols,"champion":champion,
            "breeding_pool_size":len(population),"crossover_candidates":sum(bool(x.get("spec",{}).get("crossover")) for x in results),
            "genome_scope":"561 causal input features + thresholds/operators + SL/TP ATR + max-hold genes",
            "feature_count":len(_LIBRARY_FEATURES),"total_feature_columns":len(FEATURES),
            "feature_library_version":FEATURE_LIBRARY_VERSION,
            "latest_experiments":results,
            "note":"Genetic evolution uses mutation + two-parent crossover + elitist breeding over selected 561-feature input genes, thresholds/operators and exit genes. Chronological train/tune; holdout is never used to mutate genomes or choose breeding parents. It is used only for independent research Champion/Challenger confirmation. Production promotion remains behind the 500-win/500-loss gate and independent model holdout."}
    generation_history=list(memory.get("generation_history") or [])
    generation_best=max([float(x.get("score",-100)) for x in results] or [-100.0])
    generation_history.append({"generation":generation,"best_score":generation_best,
                               "validation_passed":int(result["validation_passed"])})
    generation_history=generation_history[-100:]
    result["mutation_rate_used"]=mutation_rate
    result["elite_fraction"]=0.10
    result["elite_count"]=elite_count
    result["generation_history_size"]=len(generation_history)
    memory.update({"schema":3,"generation":generation,"seed":int(memory.get("seed",1729)),
                   "last_run":time.time(),"archive":archive,"population":population,"champion":champion,
                   "mutation_rate":mutation_rate,"generation_history":generation_history,
                   "last_result":{k:v for k,v in result.items() if k!="latest_experiments"}})
    _atomic_write(path,memory)
    # Publish a compact dashboard report; the full experiment archive stays in
    # durable state and is not copied wholesale into dashboard_data.json.
    _atomic_write(os.path.join(state_dir,"autonomous_strategy_report.json"),result)
    if telegram:
        telegram("🧬 Autonomous Strategy Discovery\n"
                 f"Generation: {generation} | New candidates: {len(results)} | "
                 f"Passed validation: {result['validation_passed']}\n"
                 f"Features: {len(_LIBRARY_FEATURES)} | Archive: {len(archive)} | Mode: research-only; no live changes.")
    return result


def _atomic_write(path,obj):
    tmp=path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    os.replace(tmp,path)
