#!/usr/bin/env python3
"""Causal OHLCV feature library for autonomous strategy discovery.

All features use current/past candles only. No future labels or forward shifts.
The generated feature names are deterministic and versioned.
"""
import numpy as np
import pandas as pd

PERIODS = (3, 5, 8, 10, 14, 20, 30, 40, 50, 80, 100, 200)
ROLL_WINDOWS = (5, 10, 20, 40, 80, 120, 200)
LIBRARY_VERSION = 1


def build_feature_library(df, core=None):
    o = pd.to_numeric(df["o"], errors="coerce")
    h = pd.to_numeric(df["h"], errors="coerce")
    l = pd.to_numeric(df["l"], errors="coerce")
    c = pd.to_numeric(df["c"], errors="coerce")
    v = pd.to_numeric(df.get("vol", pd.Series(1.0, index=df.index)), errors="coerce")
    eps = 1e-12
    prev = c.shift(1)
    logc = np.log(c.where(c > 0))
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    candle_range = (h-l).replace(0, np.nan)
    typical = (h+l+c)/3
    raw = {}

    # 18 indicator families evaluated over 12 independent lookback periods.
    for n in PERIODS:
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        rsi = 100 - 100/(1+gain/(loss+eps))
        ema = c.ewm(span=n, adjust=False).mean()
        ema_fast = c.ewm(span=max(2, n//2), adjust=False).mean()
        atr = tr.ewm(alpha=1/n, adjust=False).mean()
        lo, hi = l.rolling(n, min_periods=max(2,n//2)).min(), h.rolling(n, min_periods=max(2,n//2)).max()
        prior_hi = h.shift(1).rolling(n, min_periods=max(2,n//2)).max()
        vol_mean = v.rolling(n, min_periods=max(2,n//2)).mean()
        lr = logc.diff()
        raw[f"ret_{n}"] = logc.diff(n)
        raw[f"rsi_{n}"] = (rsi-50)/50
        raw[f"ema_gap_{n}"] = c/(ema+eps)-1
        raw[f"ema_slope_{n}"] = ema.pct_change(max(2,n//4))
        raw[f"atr_pct_{n}"] = atr/(c+eps)
        raw[f"range_pos_{n}"] = (c-lo)/(hi-lo+eps)
        raw[f"volume_ratio_{n}"] = v/(vol_mean+eps)
        raw[f"body_pct_{n}"] = (c-o)/(o+eps)
        raw[f"upper_wick_{n}"] = (h-pd.concat([o,c],axis=1).max(axis=1))/(candle_range+eps)
        raw[f"lower_wick_{n}"] = (pd.concat([o,c],axis=1).min(axis=1)-l)/(candle_range+eps)
        raw[f"breakout_dist_{n}"] = c/(prior_hi+eps)-1
        raw[f"close_location_{n}"] = (c-l)/(candle_range+eps)
        raw[f"downside_vol_{n}"] = lr.clip(upper=0).rolling(n).std()
        raw[f"upside_vol_{n}"] = lr.clip(lower=0).rolling(n).std()
        raw[f"efficiency_{n}"] = logc.diff(n).abs()/(lr.abs().rolling(n).sum()+eps)
        raw[f"volume_trend_{n}"] = v.rolling(max(2,n//3)).mean()/(vol_mean+eps)
        vwap = (typical*v).rolling(n).sum()/(v.rolling(n).sum()+eps)
        raw[f"vwap_dist_{n}"] = c/(vwap+eps)-1
        raw[f"drawdown_{n}"] = c/(c.rolling(n).max()+eps)-1

    out = pd.DataFrame(raw, index=df.index).replace([np.inf,-np.inf],np.nan)

    # Expand the 15 core concepts with rolling robust z-scores and percentile ranks.
    if core is not None:
        for name in core.columns:
            x = pd.to_numeric(core[name], errors="coerce")
            for w in ROLL_WINDOWS:
                minp = max(3, w//2)
                mean = x.rolling(w, min_periods=minp).mean()
                std = x.rolling(w, min_periods=minp).std().replace(0,np.nan)
                out[f"{name}__z{w}"] = ((x-mean)/(std+eps)).clip(-8,8)
                # Rolling percentile rank is causal; only the current rolling window is used.
                out[f"{name}__rank{w}"] = x.rolling(w, min_periods=minp).rank(pct=True)

        # Pairwise interactions use bounded rolling z-scores to avoid scale dominance.
        names = list(core.columns)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                def zscore(series):
                    wmean = series.rolling(40, min_periods=20).mean()
                    wstd = series.rolling(40, min_periods=20).std()
                    return ((series-wmean)/(wstd+eps)).clip(-3,3)/3
                a, b = zscore(core[left]), zscore(core[right])
                out[f"interaction__{left}__{right}"] = a*b

        for name in names:
            out[f"core__{name}"] = pd.to_numeric(core[name], errors="coerce")

    out = out.replace([np.inf,-np.inf],np.nan)
    out.attrs["feature_library_version"] = LIBRARY_VERSION
    return out


def feature_thresholds(feature_names):
    result = {}
    for name in feature_names:
        if "__rank" in name or name.startswith("range_pos_") or name.startswith("close_location_"):
            result[name] = (0.10,0.20,0.35,0.50,0.65,0.80,0.90)
        elif "__z" in name or name.startswith("interaction__"):
            result[name] = (-0.75,-0.50,-0.25,0.0,0.25,0.50,0.75)
        elif name.startswith("rsi_"):
            result[name] = (-0.60,-0.40,-0.20,0.0,0.20,0.40,0.60)
        elif name.startswith(("volume_ratio_","volume_trend_")):
            result[name] = (0.5,0.75,1.0,1.25,1.5,2.0,3.0)
        elif name.startswith(("atr_pct_","downside_vol_","upside_vol_")):
            result[name] = (0.003,0.006,0.01,0.015,0.02,0.03,0.05)
        elif name.startswith(("ret_","ema_gap_","ema_slope_","body_pct_","breakout_dist_","vwap_dist_","drawdown_")):
            result[name] = (-0.08,-0.04,-0.02,-0.01,0.0,0.01,0.02,0.04,0.08)
        elif name.startswith(("efficiency_","upper_wick_","lower_wick_")):
            result[name] = (0.1,0.2,0.35,0.5,0.65,0.8,0.9)
        else:
            result[name] = (-1.0,-0.5,0.0,0.5,1.0)
    return result


def feature_count():
    # Number of columns is computed on a deterministic synthetic OHLCV frame in tests.
    return len(PERIODS)*18 + len(ROLL_WINDOWS)*2*15 + (15*14)//2 + 15
