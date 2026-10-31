#!/usr/bin/env python3
"""Autonomous symbolic strategy discovery for Zied Quant.

Research-only: generated strategies are evaluated and persisted, but NEVER wired
into live entries or allowed to modify the existing AI model/portfolio controls.
"""
import hashlib
import json
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
FEATURES = (
    "ret1", "ret3", "ret6", "ret12", "rsi", "ema_gap", "ema_slope",
    "atr_pct", "range_pos", "volume_ratio", "body_pct", "upper_wick",
    "lower_wick", "breakout_dist", "close_location",
)
THRESHOLDS = {
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


def build_features(df):
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


def _spec_id(spec):
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _new_spec(rng, generation, parent=None):
    count = rng.randint(2, 7)
    conditions = []
    used = set()
    if parent and rng.random() < 0.65:
        parent_spec = parent.get("spec", parent)
        conditions = [dict(x) for x in parent_spec.get("conditions", [])]
        # Mutate, remove, or add conditions; no fixed strategy-complexity ceiling.
        if conditions and rng.random() < 0.35:
            conditions.pop(rng.randrange(len(conditions)))
        if conditions and rng.random() < 0.55:
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
    equity = np.cumprod(1+x)
    peaks = np.maximum.accumulate(np.r_[1.0,equity])[1:]
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
        return result
    features=[build_features(d) for d in frames]
    archive=memory.get("archive",[])
    parents=sorted([x for x in archive if x.get("status") in ("validation_pass","paper_candidate")],
                   key=lambda x:float(x.get("score",-100)),reverse=True)[:20]
    known={x.get("id") for x in archive}
    candidates=[]
    attempts=0
    while len(candidates)<int(batch_size) and attempts<int(batch_size)*30:
        attempts+=1
        parent=rng.choice(parents) if parents and rng.random()<0.65 else None
        spec=_new_spec(rng,generation,parent)
        if spec["id"] not in known and all(x["id"]!=spec["id"] for x in candidates):
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
        row={"id":spec["id"],"spec":spec,"generation":generation,"status":status,
             "score":score,"score_details":penalties,"train":train,"validation":tune,
             "holdout_observation":hold,"per_symbol":per_symbol,
             "symbols":symbols,"cost_round_trip":COST,"created_at":time.time()}
        results.append(row)
    archive.extend(results)
    archive=archive[-MAX_ARCHIVE:]
    valid=[x for x in archive if x.get("status")=="validation_pass"]
    champion=max(valid,key=lambda x:float(x.get("score",-100))) if valid else memory.get("champion")
    result={"status":"ok","generation":generation,"generated":len(candidates),
            "tested":len(results),"validation_passed":sum(x["status"]=="validation_pass" for x in results),
            "archive_size":len(archive),"symbols":symbols,"champion":champion,
            "latest_experiments":results,
            "note":"Research-only. Validation uses chronological train/tune blocks; holdout is observational and never used for ranking or breeding. No live strategy/model changes."}
    memory.update({"schema":1,"generation":generation,"seed":int(memory.get("seed",1729)),
                   "last_run":time.time(),"archive":archive,"champion":champion,
                   "last_result":{k:v for k,v in result.items() if k!="latest_experiments"}})
    _atomic_write(path,memory)
    # Publish a compact dashboard report; the full experiment archive stays in
    # durable state and is not copied wholesale into dashboard_data.json.
    _atomic_write(os.path.join(state_dir,"autonomous_strategy_report.json"),result)
    if telegram:
        telegram("🧬 Autonomous Strategy Discovery\n"
                 f"Generation: {generation} | New candidates: {len(results)} | "
                 f"Passed validation: {result['validation_passed']}\n"
                 f"Archive: {len(archive)} | Mode: research-only; no live changes.")
    return result


def _atomic_write(path,obj):
    tmp=path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    os.replace(tmp,path)
