#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zied Quant v2
=============
OKX Spot (USDT, no stablecoins)
  -> 150 features = 5 timeframes x 28 technical + 10 market-context features
  -> cross-sectional ranking (best -> worst)
  -> models: 20-layer deep net + gradient boosting + ensemble (best chosen on a time-split validation)
  -> logged signal -> future-candle triple-barrier label for learning
  -> signals are labelled from future candles for self-learning
  -> walk-forward backtest, Telegram alerts + Telegram commands.

SIGNAL-ONLY. No virtual wallet, simulated positions, or real orders. Public OKX endpoints only (no API key).

Commands:
    python zied_quant.py train        # build dataset (cached) + train models
    python zied_quant.py refresh      # re-download history, then train
    python zied_quant.py backtest [folds]   # walk-forward out-of-sample backtest
    python zied_quant.py scan         # one scan, no trading
    python zied_quant.py once         # one full cycle (schedulers / GitHub Actions)
    python zied_quant.py run          # endless loop (VPS / Docker)
    python zied_quant.py label        # label matured logged signals
    python zied_quant.py stats        # paper-trading statistics

Env (optional):
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID   alerts + remote commands
    OKX_BASE                           e.g. https://aws.okx.com if www.okx.com is blocked
"""
import os, sys, time, json, math, pickle, sqlite3, logging, threading, functools, datetime, shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from numpy.lib.stride_tricks import sliding_window_view

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zied")

# =====================================================================
# CONFIG
# =====================================================================
CFG = dict(
    OKX=os.getenv("OKX_BASE", "https://www.okx.com"),
    TFS=["5m", "15m", "1H", "4H", "1D"],
    TF_SEC={"5m": 300, "15m": 900, "1H": 3600, "4H": 14400, "1D": 86400},
    BASE_TF="1H",
    SCAN_DEPTH=260,                    # covers the 200-bar causal feature library plus warm-up on every timeframe
    HIST_DAYS=30,                      # history used for the bootstrap dataset
    TRAIN_TOP=40,                      # coins (by volume) in the bootstrap dataset
    MIN_VOL_USDT=0,                    # scan every live non-stable SPOT-USDT pair; rank liquidity as a feature
    # labels = triple barrier on 1H candles (same SL/TP as paper trading)
    BARRIER_H=24,                      # max holding, in 1H bars
    SL_ATR=1.5, TP_ATR=3.0,
    BT_COST=0.0035,                    # assumed round-trip cost (fees+slippage+spread) for labels/backtest
    # scanning / signals
    TOP_AI=50, LOG_TOP=50,
    SCAN_EVERY_SEC=900, MONITOR_EVERY_SEC=30, RANK_MSG_EVERY=4,
    PROB_MIN=None,                     # None = auto threshold from validation
    EXPLORATION_PROB_MIN=0.65,          # rank-based paper exploration while no validated AI model exists
    # historical backtest constraints only; no live or virtual portfolio
    MAX_OPEN=5, POS_FRAC=0.15, RISK_PER_TRADE=0.005, MAX_PORTFOLIO_RISK=0.02,
    MAX_OPEN_CORR=0.85,                 # reject a new paper entry if abs(correlation) to an open coin is higher
    MAX_SIGNAL_RANK=10,                 # wider runtime pool keeps all existing quality gates
    # self-learning
    RETRAIN_SIGNALS=500,               # bootstrap cache retry interval; separate from closed-memory learning
    DIR="zied_data",
)
os.makedirs(CFG["DIR"], exist_ok=True)
CFG["DB"] = os.path.join(CFG["DIR"], "paper.db")
CFG["MODEL"] = os.path.join(CFG["DIR"], "model.pt")
CFG["GBM"] = os.path.join(CFG["DIR"], "gbm.pkl")
CFG["META"] = os.path.join(CFG["DIR"], "meta.json")
CFG["TRAINSET"] = os.path.join(CFG["DIR"], "train.npz")

TFS, BASE = CFG["TFS"], CFG["BASE_TF"]
EPS = 1e-12
FEATURE_DRIFT_HOOK = None
PRIMARY_PREDICTION_HOOK = None
PORTFOLIO_CORRELATION_HOOK = None
DATA_VER = 3

# =====================================================================
# TELEGRAM
# =====================================================================
def tg(msg):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        log.info("[TG off] %s", msg.replace("\n", " | "))
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": msg[:4000]}, timeout=10)
    except Exception as e:
        log.warning("telegram failed: %s", e)

# =====================================================================
# OKX DATA
# =====================================================================
_thr_lock, _last_call = threading.Lock(), [0.0]

def okx_get(path, params=None, tries=4, gap=0.07):
    """Rate-limited OKX request with HTTP/JSON validation and bounded retry backoff."""
    for i in range(max(1, int(tries))):
        with _thr_lock:
            dt = time.time() - _last_call[0]
            if dt < gap:
                time.sleep(gap - dt)
            _last_call[0] = time.time()
        try:
            response = requests.get(CFG["OKX"] + path, params=params, timeout=(5, 15))
            response.raise_for_status()
            j = response.json()
            if not isinstance(j, dict):
                raise ValueError("OKX response is not a JSON object")
            if str(j.get("code")) == "0" and isinstance(j.get("data"), list):
                return j["data"]
            if str(j.get("code")) == "50011":
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = max(float(retry_after), 1.0) if retry_after else 1.5 * (i + 1)
                except (TypeError, ValueError):
                    delay = 1.5 * (i + 1)
                time.sleep(min(12.0, delay))
                continue
            log.warning("OKX %s -> code=%s msg=%s", path, j.get("code"), j.get("msg"))
            return []
        except (requests.RequestException, ValueError) as exc:
            delay = min(8.0, 0.5 * (2 ** i))
            log.warning("OKX request failed (%s, attempt %d/%d): %s; retry in %.1fs",
                        path, i + 1, max(1, int(tries)), exc, delay)
            if i + 1 < max(1, int(tries)):
                time.sleep(delay)
    return []

STABLE_EXTRA = {"DAI", "EURT", "EURC", "EURI", "AEUR", "XSGD", "GYEN", "BRZ", "USDT"}

def is_stable(base):
    b = base.upper()
    return "USD" in b or b in STABLE_EXTRA          # USDC, FDUSD, TUSD, PYUSD, RLUSD...

def get_universe():
    """All live SPOT *-USDT pairs, no stablecoins, with 24h volume, ask and spread."""
    ins = okx_get("/api/v5/public/instruments", {"instType": "SPOT"})
    ok = {i["instId"] for i in ins
          if i["quoteCcy"] == "USDT" and i["state"] == "live" and not is_stable(i["baseCcy"])}
    rows = []
    for t in okx_get("/api/v5/market/tickers", {"instType": "SPOT"}):
        inst = t.get("instId")
        if inst not in ok:
            continue
        try:
            vol = float(t.get("volCcy24h") or 0)
            last = float(t.get("last") or 0)
            ask = float(t.get("askPx") or last)
            bid = float(t.get("bidPx") or 0)
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (vol, last, ask, bid)):
            continue
        if vol < CFG["MIN_VOL_USDT"] or last <= 0 or ask <= 0 or bid <= 0 or ask < bid:
            continue
        spread = (ask - bid) / last
        if not np.isfinite(spread) or spread < 0 or spread > 0.10:
            continue
        rows.append(dict(inst=inst, vol=vol, last=last, ask=ask, bid=bid, spread=spread))
    if not rows:
        return pd.DataFrame(columns=["vol", "last", "ask", "spread"])
    return pd.DataFrame(rows).set_index("inst").sort_values("vol", ascending=False)

def get_prices():
    """Return only valid positive last prices; OKX may include empty fields."""
    prices = {}
    for t in okx_get("/api/v5/market/tickers", {"instType": "SPOT"}):
        inst = t.get("instId")
        raw = t.get("last")
        if not inst or raw in (None, ""):
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price) or price <= 0:
            continue
        prices[inst] = price
    return prices

def _candle_cache_path(inst, bar):
    # Persistent cache for every symbol/timeframe; Actions restores zied_data between runs.
    safe_inst = str(inst).replace("/", "_").replace("-", "_")
    return os.path.join(CFG["DIR"], "candles", f"{safe_inst}_{bar}.csv")

def _normalize_candles(rows, bar, depth):
    if not rows:
        return None
    rows = [(list(r) + ["1"])[:9] for r in rows]
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "vol", "volccy", "volq", "confirm"])
    df = df[df["confirm"].astype(str) == "1"].copy()
    if df.empty:
        return None
    for k in ("o", "h", "l", "c", "vol"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "o", "h", "l", "c", "vol"])
    df["ts"] = df["ts"].astype("int64")
    numeric = df[["o", "h", "l", "c", "vol"]].to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric).all(axis=1)
    valid &= (df[["o", "h", "l", "c"]].to_numpy(dtype=np.float64) > 0).all(axis=1)
    valid &= (df["vol"].to_numpy(dtype=np.float64) >= 0)
    valid &= (df["h"].to_numpy(dtype=np.float64) >=
              df[["o", "c", "l"]].max(axis=1).to_numpy(dtype=np.float64))
    valid &= (df["l"].to_numpy(dtype=np.float64) <=
              df[["o", "c", "h"]].min(axis=1).to_numpy(dtype=np.float64))
    now_ms = int(time.time() * 1000)
    tf_ms = CFG["TF_SEC"][bar] * 1000
    valid &= (df["ts"].to_numpy(dtype=np.int64) > 0)
    valid &= (df["ts"].to_numpy(dtype=np.int64) <= now_ms + tf_ms)
    df = df.loc[valid].copy()
    if df.empty:
        return None
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["avail"] = pd.to_datetime(df["ts"] + tf_ms, unit="ms")
    return df.iloc[-depth:].reset_index(drop=True)

def get_candles(inst, bar, depth):
    """Read persisted closed candles and incrementally merge recent bars for all timeframes."""
    path = _candle_cache_path(inst, bar)
    cached = None
    try:
        if os.path.isfile(path):
            cached = pd.read_csv(path)
            if not cached.empty:
                cached["ts"] = pd.to_numeric(cached["ts"], errors="coerce")
                cached = cached.dropna(subset=["ts"])
                cached["ts"] = cached["ts"].astype("int64")
                cached = cached.drop_duplicates("ts").sort_values("ts")
                cached["avail"] = pd.to_datetime(cached["ts"] + CFG["TF_SEC"][bar] * 1000, unit="ms")
                cached = cached.iloc[-depth:].reset_index(drop=True)
    except Exception as exc:
        log.warning("candle cache read failed for %s %s: %s", inst, bar, exc)
        cached = None

    now_ms = int(time.time() * 1000)
    tf_ms = CFG["TF_SEC"][bar] * 1000
    # Incremental refresh is safe only when the cache already covers the full requested lookback.
    # Otherwise fetch the complete depth; a short cache plus 3 new candles leaves long-window
    # features (e.g. 120/200-bar ranks and z-scores) permanently NaN.
    cache_ok = cached is not None and len(cached) >= depth
    # If the cache is current, request only a few latest bars instead of the full candle payload.
    if cache_ok:
        age = now_ms - int(cached["ts"].iloc[-1]) - tf_ms
        cache_ok = age <= 3 * tf_ms

    hist = depth > 1400
    api_path = "/api/v5/market/history-candles" if hist else "/api/v5/market/candles"
    lim = 3 if cache_ok else (100 if hist else 300)
    gap = 0.11 if hist else 0.07
    out, after = [], None
    while len(out) < (lim if cache_ok else depth):
        params = {"instId": inst, "bar": bar, "limit": lim}
        if after:
            params["after"] = after
        rows = okx_get(api_path, params, gap=gap)
        if not rows:
            break
        out += rows
        after = rows[-1][0]
        if len(rows) < lim:
            break
        if cache_ok:
            break
    fresh = _normalize_candles(out, bar, depth)
    if cache_ok and fresh is not None:
        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = merged.drop_duplicates("ts", keep="last").sort_values("ts").iloc[-depth:].reset_index(drop=True)
        merged["avail"] = pd.to_datetime(merged["ts"] + tf_ms, unit="ms")
        result = merged
    elif fresh is not None:
        result = fresh
    elif cache_ok:
        result = cached
    else:
        return None

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        result.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    except Exception as exc:
        log.warning("candle cache write failed for %s %s: %s", inst, bar, exc)
    return result.iloc[-depth:].reset_index(drop=True)

def hist_depth():
    d = CFG["HIST_DAYS"]
    return {"5m": d * 288 + 100, "15m": d * 96 + 100, "1H": d * 24 + 400,
            "4H": d * 6 + 300, "1D": d + 120}

def fetch_frames(inst, depth):
    """Fetch frames independently; tolerate one missing non-base timeframe.

    Missing frames remain NaN in the feature matrix. They are never fabricated
    or copied from another timeframe. The 1H base frame remains mandatory.
    """
    fr, missing = {}, []
    for tf in TFS:
        d = get_candles(inst, tf, depth if isinstance(depth, int) else depth[tf])
        if d is None or len(d) < 60:
            missing.append(tf)
            continue
        fr[tf] = d
    if BASE not in fr:
        log.info("data coverage %s: missing required base timeframe %s; skipped", inst, BASE)
        return None
    if len(missing) > 1:
        log.info("data coverage %s: missing=%s; exceeds one-timeframe tolerance; skipped",
                 inst, ",".join(missing))
        return None
    if missing:
        log.info("data coverage %s: missing timeframe=%s; observed features retained",
                 inst, ",".join(missing))
    return fr

def fetch_btc(h1, h4):
    b1, b4 = get_candles("BTC-USDT", "1H", h1), get_candles("BTC-USDT", "4H", h4)
    if b1 is None or b4 is None:
        raise RuntimeError("BTC data unavailable")
    return {"1H": b1, "4H": b4}

def to_sec(idx):
    return np.asarray((pd.DatetimeIndex(idx) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1),
                      dtype=np.float64)

# =====================================================================
# FEATURES : 28 technical x 5 timeframes + 10 market-context = 150
# =====================================================================
NAMES = ["ret1", "ret3", "ret5", "ret10", "ret20", "vol20", "rsi7", "rsi14",
         "macd", "macd_hist", "stoch_k", "bb_pos", "bb_width", "atr_pct", "adx14", "cci20",
         "mfi14", "obv_slope", "vol_ratio", "ema_dist21", "ema_dist50",
         "ema_slope21", "dist_high20", "dist_low20", "range_pct", "body_ratio",
         "upper_wick", "lower_wick", "ret40", "ret60", "downside_vol20", "upside_vol20",
         "efficiency20", "range_pos20", "vol_trend20", "vwap_dist20", "atr_change5", "ema_cross21_50"]
CTX_NAMES = ["beta48", "corr48", "rel_ret24", "btc_ret24", "btc_vol24", "btc_regime",
             "hour_sin", "hour_cos", "dow_sin", "log_dvol24"]
TF_COLS = [f"{tf}_{n}" for tf in TFS for n in NAMES]
CTX_COLS = [f"ctx_{n}" for n in CTX_NAMES]
FEATURE_COLS = TF_COLS + CTX_COLS
N_FEATURES = len(FEATURE_COLS)
assert N_FEATURES == 200


def feature_schema_fingerprint():
    """Stable identity for the exact ordered feature columns used by a model."""
    import hashlib
    payload = "\n".join(str(name) for name in FEATURE_COLS).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + EPS))

def tf_features(df):
    o, h, l, c, v = df["o"], df["h"], df["l"], df["c"], df["vol"]
    lc = np.log(c)
    f = {}
    for n in (1, 3, 5, 10, 20):
        f[f"ret{n}"] = lc.diff(n)
    f["vol20"] = lc.diff().rolling(20).std()
    f["rsi7"], f["rsi14"] = _rsi(c, 7), _rsi(c, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_hist"] = macd / c, (macd - sig) / c
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["stoch_k"] = (c - ll) / (hh - ll + EPS)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - ma20) / (2 * sd20 + EPS)
    f["bb_width"] = 4 * sd20 / (ma20 + EPS)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr_pct"] = atr / c
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / (atr + EPS)
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / (atr + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    f["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    tp = (h + l + c) / 3
    mad = (tp - tp.rolling(20).mean()).abs().rolling(20).mean()
    f["cci20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad + EPS)
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    f["mfi14"] = 100 - 100 / (1 + pos / (neg + EPS))
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope"] = (obv - obv.shift(10)) / (v.rolling(10).sum() + EPS)
    f["vol_ratio"] = v / (v.rolling(20).mean() + EPS)
    e21, e50 = c.ewm(span=21, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    f["ema_dist21"], f["ema_dist50"] = c / e21 - 1, c / e50 - 1
    f["ema_slope21"] = e21.pct_change(5)
    f["dist_high20"] = c / h.rolling(20).max() - 1
    f["dist_low20"] = c / l.rolling(20).min() - 1
    rng = h - l + EPS
    f["range_pct"] = (h - l) / c
    f["body_ratio"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    # Extra multi-horizon, volatility-regime, trend-quality and volume features.
    lr = lc.diff()
    f["ret40"], f["ret60"] = lc.diff(40), lc.diff(60)
    f["downside_vol20"] = lr.clip(upper=0).rolling(20).std()
    f["upside_vol20"] = lr.clip(lower=0).rolling(20).std()
    f["efficiency20"] = lc.diff(20).abs() / (lr.abs().rolling(20).sum() + EPS)
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    f["range_pos20"] = (c - lo20) / (hi20 - lo20 + EPS)
    f["vol_trend20"] = v.rolling(5).mean() / (v.rolling(20).mean() + EPS)
    typical = (h + l + c) / 3
    f["vwap_dist20"] = c / ((typical * v).rolling(20).sum() / (v.rolling(20).sum() + EPS) + EPS) - 1
    f["atr_change5"] = atr / (atr.shift(5) + EPS) - 1
    f["ema_cross21_50"] = e21 / (e50 + EPS) - 1
    out = pd.DataFrame(f)[NAMES].replace([np.inf, -np.inf], np.nan)
    out.index = pd.DatetimeIndex(df["avail"].values)
    return out

def ctx_features(coin, btc):
    """10 market-context features on the coin's 1H index (relation to BTC, regime, time, liquidity)."""
    idx = pd.DatetimeIndex(coin["avail"].values)
    c = pd.Series(coin["c"].values, index=idx)
    v = pd.Series(coin["vol"].values, index=idx)
    b1 = pd.Series(btc["1H"]["c"].values, index=pd.DatetimeIndex(btc["1H"]["avail"].values))
    b = b1[~b1.index.duplicated()].reindex(idx)
    rc, rb = np.log(c).diff(), np.log(b).diff()
    out = pd.DataFrame(index=idx)
    out["beta48"] = rc.rolling(48).cov(rb) / (rb.rolling(48).var() + EPS)
    out["corr48"] = rc.rolling(48).corr(rb)
    out["rel_ret24"] = np.log(c).diff(24) - np.log(b).diff(24)
    out["btc_ret24"] = np.log(b).diff(24)
    out["btc_vol24"] = rb.rolling(24).std()
    b4 = btc["4H"]
    reg = pd.Series((b4["c"] / b4["c"].ewm(span=50, adjust=False).mean() - 1).values,
                    index=pd.DatetimeIndex(b4["avail"].values))
    out["btc_regime"] = reg[~reg.index.duplicated()].reindex(idx, method="ffill")
    out["hour_sin"] = np.sin(2 * np.pi * np.asarray(idx.hour) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * np.asarray(idx.hour) / 24)
    out["dow_sin"] = np.sin(2 * np.pi * np.asarray(idx.dayofweek) / 7)
    out["log_dvol24"] = np.log((c * v).rolling(24).sum() + 1)
    out.columns = CTX_COLS
    return out.replace([np.inf, -np.inf], np.nan)[CTX_COLS]

def all_tf_features(frames):
    """Build observed frames; unavailable timeframes stay explicitly NaN."""
    F = {}
    base_index = pd.DatetimeIndex(frames[BASE]["avail"].values)
    for tf in TFS:
        if frames.get(tf) is None:
            F[tf] = pd.DataFrame(np.nan, index=base_index,
                                 columns=[f"{tf}_{n}" for n in NAMES])
            continue
        x = tf_features(frames[tf])
        x.columns = [f"{tf}_{n}" for n in NAMES]
        F[tf] = x
    return F

def live_vector(frames, btc):
    """Latest closed-candle features of every timeframe + context -> 150 values."""
    F = all_tf_features(frames)
    parts = [F[tf].iloc[-1] for tf in TFS] + [ctx_features(frames[BASE], btc).iloc[-1]]
    # Runtime intelligence features are appended by installed news/event layers after scanning.
    # Reindex instead of strict column selection so their not-yet-enriched values
    # remain explicit NaN and can be filled by rank wrappers without dropping the coin.
    return pd.concat(parts).reindex(FEATURE_COLS)

def aligned_matrix(frames, btc):
    """Historical rows on BASE_TF; other TFs use the last candle already closed (no look-ahead)."""
    F = all_tf_features(frames)
    idx = F[BASE].index
    parts = [F[BASE]] + [F[tf].reindex(idx, method="ffill") for tf in TFS if tf != BASE]
    parts.append(ctx_features(frames[BASE], btc))
    # Dynamic intelligence layers append columns after this module is imported.
    # Reindex keeps the active schema stable and represents unavailable historical
    # features as NaN (unknown), rather than raising KeyError or fabricating values.
    return pd.concat(parts, axis=1).reindex(columns=FEATURE_COLS)

# =====================================================================
# TRIPLE-BARRIER LABELS (same SL/TP logic as paper trading)
# =====================================================================
def sl_atr_multiplier(atr_pct):
    """Widen the stop progressively for more volatile coins; TP is unchanged."""
    a = float(atr_pct)
    if a < 0.012:
        return 1.8
    if a < 0.025:
        return 2.0
    return 2.2


def barrier_batch(c, h, l, atr):
    """Row i enters at c[i]; path = candles i+1 .. i+H. Returns (net_return, hold_bars).
    SL is assumed first when SL and TP hit inside the same candle (conservative)."""
    H = CFG["BARRIER_H"]
    n = len(c)
    ret = np.full(n, np.nan); hold = np.full(n, np.nan)
    if n <= H:
        return ret, hold
    c, h, l, atr = (np.asarray(x, dtype=np.float64) for x in (c, h, l, atr))
    m = n - H
    Hw = sliding_window_view(h[1:], H)
    Lw = sliding_window_view(l[1:], H)
    Cw = sliding_window_view(c[1:], H)
    a = np.clip(atr[:m], 0.004, 0.06)
    entry = c[:m]
    sl_mult = np.where(a < 0.012, 1.8, np.where(a < 0.025, 2.0, 2.2))
    sl = entry * (1 - sl_mult * a); tp = entry * (1 + CFG["TP_ATR"] * a)
    hit_sl, hit_tp = Lw <= sl[:, None], Hw >= tp[:, None]
    i_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), H)
    i_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), H)
    sl_first = (i_sl < H) & (i_sl <= i_tp)
    tp_first = (i_tp < H) & (i_tp < i_sl)
    gross = np.where(sl_first, -sl_mult * a,
                     np.where(tp_first, CFG["TP_ATR"] * a, Cw[:, -1] / entry - 1))
    ret[:m] = gross - CFG["BT_COST"]
    hold[:m] = np.where(sl_first, i_sl + 1, np.where(tp_first, i_tp + 1, H))
    ret[:m] = np.where(np.isnan(atr[:m]), np.nan, ret[:m])
    return ret, hold

def barrier_single(entry, atr, hi, lo, cl):
    H = CFG["BARRIER_H"]
    c = np.r_[entry, cl[:H]]; h = np.r_[entry, hi[:H]]; l = np.r_[entry, lo[:H]]
    r, hd = barrier_batch(c, h, l, np.full(len(c), atr))
    return float(r[0]), int(r[0] > 0), float(hd[0])

# =====================================================================
# DATASETS
# =====================================================================
DS_KEYS = ("X", "y", "ts", "ret", "hold", "coin")

def empty_ds():
    return dict(X=np.zeros((0, N_FEATURES), np.float32), y=np.zeros(0), ts=np.zeros(0),
                ret=np.zeros(0), hold=np.zeros(0), coin=np.array([], dtype="<U16"))

def ds_concat(parts):
    parts = [p for p in parts if p is not None and len(p["y"])]
    if not parts:
        return empty_ds()
    return {k: np.concatenate([p[k] for p in parts]) for k in DS_KEYS}

def build_trainset(universe):
    insts = list(universe.index[:CFG["TRAIN_TOP"]])
    depth = hist_depth()
    log.info("downloading %d days of history for %d coins ...", CFG["HIST_DAYS"], len(insts))
    btc = fetch_btc(depth["1H"], depth["4H"])
    def one(inst):
        try:
            fr = fetch_frames(inst, depth)
            if fr is None:
                return None
            M = aligned_matrix(fr, btc)
            b = fr[BASE]
            ret, hold = barrier_batch(b["c"].values, b["h"].values, b["l"].values, M["1H_atr_pct"].values)
            D = M.assign(_ret=ret, _hold=hold)
            values = D[FEATURE_COLS].to_numpy(dtype=np.float64)
            # Training rows must be fully observed: keep partial timeframe support
            # for live scanning, but never impute or train on fabricated/missing values.
            valid = (np.isfinite(D["_ret"].to_numpy(dtype=np.float64))
                     & np.isfinite(D["_hold"].to_numpy(dtype=np.float64))
                     & feature_coverage_mask(values, 0.80))
            D = D.loc[valid]
            if D.empty:
                return None
            return dict(X=D[FEATURE_COLS].values.astype(np.float32), y=(D["_ret"] > 0).values.astype(np.float32),
                        ts=to_sec(D.index), ret=D["_ret"].values, hold=D["_hold"].values,
                        coin=np.array([inst] * len(D), dtype="<U16"))
        except Exception as e:
            log.warning("history %s failed: %s", inst, e); return None
    with ThreadPoolExecutor(4) as ex:
        ds = ds_concat(list(ex.map(one, insts)))
    if not len(ds["y"]):
        raise RuntimeError("empty training set (OKX unreachable?)")
    np.savez_compressed(CFG["TRAINSET"], ver=DATA_VER, **ds)
    log.info("trainset: %d rows, win rate %.1f%%", len(ds["y"]), 100 * ds["y"].mean())
    return ds

def load_trainset():
    if not os.path.exists(CFG["TRAINSET"]):
        return None
    try:
        z = np.load(CFG["TRAINSET"])
        if int(z["ver"]) != DATA_VER or z["X"].shape[1] != N_FEATURES:
            return None
        return {k: z[k] for k in DS_KEYS}
    except Exception:
        return None

# =====================================================================
# MODELS : 20-layer net + gradient boosting + ensemble
# =====================================================================
if torch is not None:
    class Block(nn.Module):
        def __init__(self, d, p):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p),
                                     nn.Linear(d, d), nn.LayerNorm(d))
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(x + self.net(x))

    class ZiedNet(nn.Module):             # 1 input + 18 hidden (9 residual blocks) + 1 output = 20 Linear layers
        def __init__(self, n_in, d=192, p=0.15):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.LayerNorm(d), nn.GELU())
            self.blocks = nn.Sequential(*[Block(d, p) for _ in range(9)])
            self.out = nn.Linear(d, 1)
        def forward(self, x):
            return self.out(self.blocks(self.inp(x)))

    def n_linear(m):
        return sum(isinstance(x, nn.Linear) for x in m.modules())

    def _train_net(Xf, yf, Xe, ye, epochs):
        torch.manual_seed(0)
        Xt, yt = torch.tensor(Xf), torch.tensor(yf, dtype=torch.float32)
        Xv, yv = torch.tensor(Xe), torch.tensor(ye, dtype=torch.float32)
        net = ZiedNet(Xf.shape[1])
        assert n_linear(net) == 20, n_linear(net)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.BCEWithLogitsLoss()
        best, state, bad = 1e9, None, 0
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                if len(b) < 2:
                    continue
                opt.zero_grad(); loss = lossf(net(Xt[b]).squeeze(1), yt[b]); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            sch.step(); net.eval()
            with torch.no_grad():
                vl = lossf(net(Xv).squeeze(1), yv).item()
            if vl < best - 1e-4:
                best, bad = vl, 0; state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        if state is None:
            raise RuntimeError("net training diverged")
        net.load_state_dict(state); net.eval()
        return net

    def _nn_predict(net, X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()

class GBM:
    """LightGBM if installed, otherwise scikit-learn HistGradientBoosting."""
    def __init__(self):
        self.m = None; self.kind = None

    def fit(self, Xf, yf, Xe, ye):
        try:
            import lightgbm as lgb
            self.m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31,
                                        min_child_samples=50, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)
            self.m.fit(Xf, yf, eval_set=[(Xe, ye)], callbacks=[lgb.early_stopping(50, verbose=False)])
            self.kind = "lightgbm"; return True
        except ImportError:
            pass
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                    min_samples_leaf=50, early_stopping=True,
                                                    validation_fraction=0.15, random_state=0)
            self.m.fit(Xf, yf); self.kind = "sklearn"; return True
        except ImportError:
            return False

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]

def chronological_boundary(ts, target_index):
    """Move a chronological split to the first row of its timestamp group.

    This prevents observations sharing the same market timestamp (often different
    coins from the same candle) from leaking across train/validation boundaries.
    ts must already be sorted in ascending order.
    """
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if not len(ts):
        return 0
    i = min(max(int(target_index), 0), len(ts))
    if i == len(ts):
        return len(ts)
    return int(np.searchsorted(ts, ts[i], side="left"))


def auc_score(y, p):
    """Tie-correct ROC AUC using average ranks (Mann-Whitney definition)."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError(f"AUC length mismatch: {len(y)} labels vs {len(p)} scores")
    if not len(y) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("AUC requires non-empty binary labels and finite scores")
    npos = int((y == 1).sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    sorted_ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        # One-based ranks; equal scores receive their average rank.
        sorted_ranks[i:j] = ((i + 1) + j) / 2.0
        i = j
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = sorted_ranks
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

class Brain:
    def __init__(self, persist=True):
        self.nn = self.gbm = None
        self.mu = self.sd = None
        self.thrs, self.mode, self.auc, self.info = {}, None, None, {}
        self.persist = persist
        self.load_issue = None
        self._recovering_backup = False
        if persist:
            self.load()

    @property
    def ready(self):
        return self.mode is not None and (self.nn is not None or self.gbm is not None)

    @property
    def thr(self):
        return self.thrs.get(self.mode, 0.5)

    def _norm(self, X):
        values = np.asarray(X, dtype=np.float64)
        # Impute only with means learned from the chronological fit block, and
        # append an explicit 0/1 missingness channel for every source feature.
        # Unknown is therefore distinguishable from a genuine numeric zero.
        missing = np.isnan(values)
        values = np.where(missing, np.asarray(self.mu, dtype=np.float64), values)
        scaled = np.clip((values - self.mu) / self.sd, -5, 5).astype(np.float32)
        return np.concatenate((scaled, missing.astype(np.float32)), axis=1)

    def describe(self):
        if not self.ready:
            return "لا يوجد نموذج بعد"
        return (f"النمط: {self.mode} | AUC={self.auc:.3f} | العتبة={self.thr:.3f}\n"
                f"شبكة 20 طبقة: {'نعم' if self.nn is not None else 'لا'} | "
                f"GBM: {'نعم' if self.gbm is not None else 'لا'}\n"
                f"عيّنات التدريب: {self.info.get('n', '?')}")

    # ---------- persistence ----------
    def _restore_previous_model(self):
        """Recover the last committed model after an interrupted multi-file promotion."""
        backup = {key: CFG[key] + ".bak" for key in ("MODEL", "GBM", "META")}
        try:
            with open(backup["META"], encoding="utf-8") as fh:
                meta = json.load(fh)
            if (int(meta.get("n_in", -1)) != 2 * N_FEATURES
                    or int(meta.get("raw_n_in", -1)) != N_FEATURES
                    or int(meta.get("missing_mask_version", 0)) != 1
                    or int(meta.get("feature_schema_version", 0)) != 2
                    or meta.get("feature_schema_hash") != feature_schema_fingerprint()):
                return False
            for key, flag in (("MODEL", "has_nn"), ("GBM", "has_gbm")):
                if meta.get(flag) and not os.path.isfile(backup[key]):
                    return False
            # Restore model components first and metadata last (the commit marker).
            for key, flag in (("MODEL", "has_nn"), ("GBM", "has_gbm")):
                target = CFG[key]
                if meta.get(flag):
                    tmp = target + ".restore.tmp"
                    shutil.copy2(backup[key], tmp)
                    os.replace(tmp, target)
                else:
                    try: os.remove(target)
                    except FileNotFoundError: pass
            tmp = CFG["META"] + ".restore.tmp"
            shutil.copy2(backup["META"], tmp)
            os.replace(tmp, CFG["META"])
            log.warning("recovered last committed AI model from backup: mode=%s auc=%s",
                        meta.get("mode"), meta.get("auc"))
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.warning("previous AI model backup is unavailable or invalid: %s", exc)
            return False
        finally:
            for key in ("MODEL", "GBM", "META"):
                try: os.remove(CFG[key] + ".restore.tmp")
                except FileNotFoundError: pass

    def save(self, n_in):
        """Promote a validated model while retaining a recoverable copy of the previous champion."""
        need_nn = self.mode in ("nn", "ens")
        need_gbm = self.mode in ("gbm", "ens")
        if self.mode not in ("nn", "gbm", "ens"):
            raise RuntimeError(f"cannot save unknown model mode: {self.mode!r}")
        if need_nn and (self.nn is None or torch is None):
            raise RuntimeError("selected AI mode requires a neural network, but it is unavailable")
        if need_gbm and self.gbm is None:
            raise RuntimeError("selected AI mode requires GBM, but it is unavailable")
        raw_n_in = int(n_in)
        model_n_in = raw_n_in * 2
        if self.mu is None or self.sd is None or len(self.mu) != raw_n_in or len(self.sd) != raw_n_in:
            raise RuntimeError("refusing to save model with invalid normalization vectors")

        os.makedirs(CFG["DIR"], exist_ok=True)
        model_tmp, gbm_tmp, meta_tmp = CFG["MODEL"] + ".tmp", CFG["GBM"] + ".tmp", CFG["META"] + ".tmp"
        backup_tmp = {key: CFG[key] + ".bak.tmp" for key in ("MODEL", "GBM", "META")}
        backup = {key: CFG[key] + ".bak" for key in ("MODEL", "GBM", "META")}
        for path in (model_tmp, gbm_tmp, meta_tmp, *backup_tmp.values()):
            try: os.remove(path)
            except FileNotFoundError: pass
        try:
            if need_nn:
                torch.save({"state": self.nn.state_dict(), "n_in": model_n_in}, model_tmp)
            if need_gbm:
                with open(gbm_tmp, "wb") as fh:
                    pickle.dump(self.gbm, fh, protocol=pickle.HIGHEST_PROTOCOL)
            meta = dict(mu=np.asarray(self.mu, dtype=float).tolist(),
                        sd=np.asarray(self.sd, dtype=float).tolist(), thrs=self.thrs, mode=self.mode,
                        auc=self.auc, has_nn=need_nn, has_gbm=need_gbm,
                        n_in=model_n_in, raw_n_in=raw_n_in,
                        missing_mask_version=1,
                        feature_schema_version=2,
                        feature_schema_hash=feature_schema_fingerprint(),
                        info=self.info)
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())

            # Snapshot only a complete current champion; never back up a partial candidate.
            try:
                with open(CFG["META"], encoding="utf-8") as fh:
                    old_meta = json.load(fh)
                old_ok = (int(old_meta.get("n_in", -1)) == 2 * N_FEATURES
                          and int(old_meta.get("raw_n_in", -1)) == N_FEATURES
                          and int(old_meta.get("missing_mask_version", 0)) == 1
                          and old_meta.get("feature_schema_hash") == feature_schema_fingerprint()
                          and (not old_meta.get("has_nn") or os.path.isfile(CFG["MODEL"]))
                          and (not old_meta.get("has_gbm") or os.path.isfile(CFG["GBM"])))
                if old_ok:
                    for key, flag in (("MODEL", "has_nn"), ("GBM", "has_gbm")):
                        if old_meta.get(flag):
                            shutil.copy2(CFG[key], backup_tmp[key])
                    shutil.copy2(CFG["META"], backup_tmp["META"])
                    for key in ("MODEL", "GBM", "META"):
                        if os.path.exists(backup_tmp[key]):
                            os.replace(backup_tmp[key], backup[key])
                        elif key != "META":
                            try: os.remove(backup[key])
                            except FileNotFoundError: pass
            except (OSError, ValueError, TypeError, KeyError):
                log.warning("existing AI champion could not be snapshotted; current files remain untouched until promotion")

            # Candidate components first; metadata is the final commit marker.
            try:
                if need_nn: os.replace(model_tmp, CFG["MODEL"])
                if need_gbm: os.replace(gbm_tmp, CFG["GBM"])
                os.replace(meta_tmp, CFG["META"])
                if not need_nn:
                    try: os.remove(CFG["MODEL"])
                    except FileNotFoundError: pass
                if not need_gbm:
                    try: os.remove(CFG["GBM"])
                    except FileNotFoundError: pass
            except Exception:
                self._restore_previous_model()
                raise
            log.info("validated AI champion saved: mode=%s nn=%s gbm=%s previous_champion_backup=%s",
                     self.mode, need_nn, need_gbm, os.path.exists(backup["META"]))
        finally:
            for path in (model_tmp, gbm_tmp, meta_tmp, *backup_tmp.values()):
                try: os.remove(path)
                except FileNotFoundError: pass

    def load(self):
        if not os.path.exists(CFG["META"]):
            if self._restore_previous_model():
                return self.load()
            # No current or recoverable champion: remove untrusted orphan components.
            for path in (CFG["MODEL"], CFG["GBM"]):
                if os.path.exists(path):
                    log.warning("removing orphan model artifact without meta.json: %s", path)
                    try: os.remove(path)
                    except OSError as exc: log.error("could not remove orphan model artifact %s: %s", path, exc)
            return
        try:
            meta = json.load(open(CFG["META"], encoding="utf-8"))
            expected_hash = feature_schema_fingerprint()
            schema_ok = (
                int(meta.get("n_in", -1)) == 2 * N_FEATURES
                and int(meta.get("raw_n_in", -1)) == N_FEATURES
                and int(meta.get("missing_mask_version", 0)) == 1
                and int(meta.get("feature_schema_version", 0)) == 2
                and meta.get("feature_schema_hash") == expected_hash
            )
            if not schema_ok:
                self.nn = self.gbm = self.mode = None
                self.mu = self.sd = None
                self.load_issue = {
                    "reason": "incompatible_feature_or_missing_value_schema",
                    "saved_raw_features": meta.get("raw_n_in"),
                    "current_raw_features": N_FEATURES,
                    "saved_model_inputs": meta.get("n_in"),
                    "expected_model_inputs": 2 * N_FEATURES,
                    "saved_missing_mask_version": meta.get("missing_mask_version"),
                    "saved_feature_schema_version": meta.get("feature_schema_version"),
                    "feature_order_matches": meta.get("feature_schema_hash") == expected_hash,
                }
                log.warning(
                    "saved AI model rejected safely: raw_features=%s/%s model_inputs=%s/%s "
                    "missing_mask=%s feature_schema_version=%s feature_order_match=%s; "
                    "verified closed-signal retraining will rebuild it when the learning gate is ready",
                    meta.get("raw_n_in"), N_FEATURES, meta.get("n_in"), 2 * N_FEATURES,
                    meta.get("missing_mask_version"), meta.get("feature_schema_version"),
                    meta.get("feature_schema_hash") == expected_hash,
                )
                return
            self.load_issue = None
            if meta.get("has_nn") and not os.path.exists(CFG["MODEL"]):
                raise FileNotFoundError("committed neural model file is missing")
            if meta.get("has_gbm") and not os.path.exists(CFG["GBM"]):
                raise FileNotFoundError("committed GBM file is missing")
            mu = np.asarray(meta["mu"], dtype=np.float64)
            sd = np.asarray(meta["sd"], dtype=np.float64)
            if (mu.shape != (N_FEATURES,) or sd.shape != (N_FEATURES,)
                    or not np.isfinite(mu).all() or not np.isfinite(sd).all()
                    or (sd <= 0).any()):
                raise ValueError("saved model normalization vectors are invalid for the current feature schema")
            self.mu, self.sd = mu, sd
            self.thrs, self.auc, self.info = meta["thrs"], meta["auc"], meta.get("info", {})
            if meta["has_nn"] and torch is not None:
                ck = torch.load(CFG["MODEL"], weights_only=True)
                self.nn = ZiedNet(ck["n_in"]); self.nn.load_state_dict(ck["state"]); self.nn.eval()
            if meta["has_gbm"]:
                with open(CFG["GBM"], "rb") as fh: self.gbm = pickle.load(fh)
            mode = meta["mode"]
            if mode in ("nn", "ens") and self.nn is None:
                mode = "gbm" if self.gbm is not None else None
            if mode == "ens" and self.gbm is None: mode = "nn"
            if mode == "gbm" and self.gbm is None: mode = "nn" if self.nn is not None else None
            self.mode = mode
            if self.ready:
                log.info("model loaded: mode=%s auc=%.3f thr=%.3f", self.mode, self.auc, self.thr)
        except Exception as e:
            log.warning("could not load model: %s", e)
            self.nn = self.gbm = self.mode = None
            self.mu = self.sd = None
            self.load_issue = {"reason": "model_load_error", "error": str(e)}
            if not self._recovering_backup:
                self._recovering_backup = True
                try:
                    if self._restore_previous_model():
                        self.load()
                finally:
                    self._recovering_backup = False

    # ---------- inference ----------
    def predict_all(self, X):
        Xn = self._norm(X); out = {}
        if self.nn is not None:
            out["nn"] = _nn_predict(self.nn, Xn)
        if self.gbm is not None:
            out["gbm"] = self.gbm.predict(Xn)
        if len(out) == 2:
            out["ens"] = (out["nn"] + out["gbm"]) / 2
        return out

    def predict(self, X):
        """Return validated probabilities; invalid model output must never reach trade selection."""
        if not self.ready:
            raise RuntimeError("AI model is not ready")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"AI input shape invalid: {X.shape}; expected (*, {N_FEATURES})")
        if np.isinf(X).any():
            raise ValueError("AI input contains infinite features; only NaN is treated as unknown")
        if not feature_coverage_mask(X, 0.80).all():
            raise ValueError("AI input has more than 20% missing features in at least one row")
        out = np.asarray(self.predict_all(X)[self.mode], dtype=np.float64).reshape(-1)
        if len(out) != len(X):
            raise ValueError(f"AI returned {len(out)} probabilities for {len(X)} rows")
        if not np.isfinite(out).all():
            raise ValueError("AI returned NaN or infinite probabilities")
        if ((out < 0.0) | (out > 1.0)).any():
            raise ValueError("AI probabilities fall outside [0, 1]")
        return out

    # ---------- training ----------
    def fit(self, X, y, ts, ret=None, epochs=80, force=False):
        """Fit on older data, select model/threshold on a recent tune block, and quality-check
        on a later untouched gate block. Walk-forward may force-fit solely for evaluation."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        ts = np.asarray(ts, dtype=np.float64)
        ret = None if ret is None else np.asarray(ret, dtype=np.float64)
        valid = feature_coverage_mask(X, 0.80) & np.isfinite(y) & np.isfinite(ts)
        if ret is not None:
            valid &= np.isfinite(ret)
        X, y, ts = X[valid], y[valid], ts[valid]
        if ret is not None:
            ret = ret[valid]
        o = np.argsort(ts, kind="stable")
        X = X[o]; y = y[o]; ts = ts[o]
        ret = None if ret is None else ret[o]
        n = len(X)
        if n < 600:
            raise ValueError(f"not enough samples ({n})")
        nv_target = max(int(n * 0.2), 200)
        # The recent tail is split chronologically: the first block selects the model
        # and probability threshold; the later block is an untouched quality gate.
        ng_target = max(100, nv_target // 2)
        # Move both boundaries to the first row for that timestamp. A timestamp can
        # contain many coins, so row-count splits alone can put the same candle time
        # in both selection and gate blocks.
        selection_start = chronological_boundary(ts, n - nv_target)
        gate_start = chronological_boundary(ts, n - ng_target)
        ns, ng, nv = gate_start - selection_start, n - gate_start, n - selection_start
        if ns < 50 or ng < 100:
            raise ValueError("selection or untouched gate block too small after timestamp grouping")
        ntr = int(np.searchsorted(ts, ts[selection_start] - CFG["BARRIER_H"] * 3600, side="left"))
        if ntr < 400:
            raise ValueError("training part too small after embargo")
        # Purge the inner fit/early-stop boundary by the full outcome horizon.
        # Keep all rows with a shared timestamp together at this boundary as well.
        nf = chronological_boundary(ts, int(ntr * 0.85))
        inner_train_end = int(np.searchsorted(
            ts, ts[nf] - CFG["BARRIER_H"] * 3600, side="left"))
        if inner_train_end < 400 or ntr - nf < 50:
            raise ValueError("training part too small after inner validation embargo")
        # Fit normalization only on the purged model-fit rows.
        fit_values = X[:inner_train_end]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(fit_values, axis=0)
            sd = np.nanstd(fit_values, axis=0)
        # A feature absent throughout the fit block carries no evidence: neutralize
        # its imputation/scale rather than letting NaN poison every model input.
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        tmp = Brain.__new__(Brain); tmp.mu, tmp.sd = mu, sd
        Xtr = tmp._norm(X[:ntr])
        Xs, Xg = tmp._norm(X[selection_start:gate_start]), tmp._norm(X[gate_start:])
        ys, yg = y[selection_start:gate_start], y[gate_start:]
        Xfit, yfit = Xtr[:inner_train_end], y[:inner_train_end]
        Xearly, yearly = Xtr[nf:ntr], y[nf:ntr]
        log.info("training split: fit=%d embargo=%d early_stop=%d selection=%d untouched_gate=%d",
                 len(Xfit), nf - inner_train_end, len(Xearly), len(Xs), len(Xg))
        models, ps = {}, {}
        g = GBM()
        if g.fit(Xfit, yfit, Xearly, yearly):
            models["gbm"] = g; ps["gbm"] = g.predict(Xs)
        if torch is not None:
            net = _train_net(Xfit, yfit, Xearly, yearly, epochs)
            models["nn"] = net; ps["nn"] = _nn_predict(net, Xs)
        if not ps:
            raise RuntimeError("no model available: install torch and/or lightgbm / scikit-learn")
        if len(ps) == 2:
            ps["ens"] = (ps["nn"] + ps["gbm"]) / 2
        selection_aucs = {k: auc_score(ys, p) for k, p in ps.items()}
        mode = max(selection_aucs, key=selection_aucs.get)
        # Choose the probability threshold on the selection block using net returns,
        # never on the untouched gate. The gate remains the final acceptance test.
        def select_threshold(probabilities):
            probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
            if ret is None:
                return float(np.quantile(probabilities, 0.90))
            selection_ret = ret[selection_start:gate_start]
            best = None
            # Conservative grid: test a few high-confidence tails, with enough samples
            # to avoid choosing a threshold from only a handful of lucky observations.
            for q in (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
                threshold = float(np.quantile(probabilities, q))
                mask = probabilities >= threshold
                count = int(mask.sum())
                if count < 20:
                    continue
                mean_net = float(np.nanmean(selection_ret[mask]))
                if not np.isfinite(mean_net):
                    continue
                candidate = (mean_net, count, threshold)
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[2] > best[2]
                ):
                    best = candidate
            return best[2] if best is not None else float(np.quantile(probabilities, 0.90))

        thrs = {k: select_threshold(p) for k, p in ps.items()}
        log.info("selection thresholds: %s", {
            k: {"threshold": round(v, 4),
                "selected": int(np.sum(ps[k] >= v))}
            for k, v in thrs.items()
        })
        # The later gate evaluates the selected model once; it cannot tune the threshold.
        pg_all = {}
        if "gbm" in models:
            pg_all["gbm"] = models["gbm"].predict(Xg)
        if "nn" in models:
            pg_all["nn"] = _nn_predict(models["nn"], Xg)
        if len(pg_all) == 2:
            pg_all["ens"] = (pg_all["nn"] + pg_all["gbm"]) / 2
        pg = pg_all[mode]
        gate_auc = auc_score(yg, pg)
        gate_mask = pg >= thrs[mode]
        gate_selected = int(gate_mask.sum())
        info = dict(n=int(n), n_val=int(nv), n_selection=int(ns), n_gate=int(ng),
                    aucs=selection_aucs, gate_auc=float(gate_auc), mode=mode,
                    auc=float(gate_auc), thr=thrs[mode], gate_selected=gate_selected,
                    win_rate=float(y.mean()), gbm_kind=g.kind if "gbm" in models else None)
        if ret is not None:
            rg = ret[gate_start:]
            info["val_all_net"] = float(np.nanmean(rg))
            info["val_top_net"] = float(np.nanmean(rg[gate_mask])) if gate_mask.any() else None
        gate_edge = (ret is not None and gate_selected >= 20
                     and info.get("val_top_net") is not None
                     and np.isfinite(info["val_top_net"])
                     and np.isfinite(info["val_all_net"])
                     and info["val_top_net"] > 0.0
                     and info["val_top_net"] > info["val_all_net"])
        # Require a useful untouched net edge and at least 30 selected examples;
        # a tiny lucky subset must not promote a production model.
        gate_edge = bool(gate_edge and gate_selected >= 30
                         and info.get("val_top_net") >= info.get("val_all_net", 0.0) + 0.001)
        quality_ok = (gate_auc >= 0.52
                      and (self.auc is None or gate_auc >= self.auc - 0.03)
                      and gate_edge)
        usable = bool(force or quality_ok)
        info["accepted"] = bool(quality_ok)
        info["forced_for_evaluation"] = bool(force and not quality_ok)
        log.info("model validation: n=%d selection=%d gate=%d selected=%s selection_AUC=%.4f gate_AUC=%.4f threshold=%.4f gate_selected=%d accepted=%s forced=%s previous_gate_auc=%s",
                 n, ns, ng, mode, selection_aucs[mode], gate_auc, thrs[mode],
                 gate_selected, bool(quality_ok), bool(force),
                 "none" if self.auc is None else f"{self.auc:.4f}")
        if not quality_ok:
            log.warning("candidate model failed untouched quality gate; usable only if forced for walk-forward evaluation")
        if usable:
            self.nn, self.gbm = models.get("nn"), models.get("gbm")
            self.mu, self.sd, self.thrs, self.mode, self.auc, self.info = mu, sd, thrs, mode, gate_auc, info
            if self.persist:
                self.save(X.shape[1])
        self.last_fit = dict(info=info, pv=ps, yv=ys, gate_p=pg, gate_y=yg,
                             X_gate=np.asarray(X[gate_start:], dtype=np.float64))
        return info

# =====================================================================
# PAPER TRADING + SIGNAL LOG + SELF-LEARNING MEMORY
# =====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, inst TEXT, price REAL, atr REAL, feats TEXT,
  prob REAL, rank INTEGER, labeled INTEGER DEFAULT 0, label INTEGER, ret REAL, hold REAL,
  feature_ts REAL, feature_count INTEGER, feature_names TEXT, feature_schema_version INTEGER);
CREATE TABLE IF NOT EXISTS tracked_signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id INTEGER UNIQUE, inst TEXT NOT NULL,
  opened_ts REAL NOT NULL, entry REAL NOT NULL, sl REAL NOT NULL, tp REAL NOT NULL,
  atr_pct REAL NOT NULL, prob REAL, rank INTEGER, feats TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'OPEN', closed_ts REAL, exit_price REAL,
  outcome TEXT, ret REAL, hold REAL, close_reason TEXT,
  regime TEXT, entry_score REAL, peak_price REAL, trailing_sl REAL, trail_active INTEGER DEFAULT 0,
  feature_count INTEGER, feature_names TEXT, feature_schema_version INTEGER);
CREATE INDEX IF NOT EXISTS idx_tracked_status ON tracked_signals(status, inst);
CREATE INDEX IF NOT EXISTS idx_signals_label_time ON signals(labeled, feature_ts, ts);
CREATE INDEX IF NOT EXISTS idx_signals_inst_time ON signals(inst, feature_ts, ts);
CREATE INDEX IF NOT EXISTS idx_tracked_closed_time ON tracked_signals(status, closed_ts, signal_id);
CREATE INDEX IF NOT EXISTS idx_tracked_open_inst ON tracked_signals(status, inst);
"""

def locked(f):
    @functools.wraps(f)
    def w(self, *a, **k):
        with self.lock:
            return f(self, *a, **k)
    return w

def impact(usdt, vol24):
    return CFG["IMPACT_K"] * math.sqrt(max(usdt, 0.0) / max(vol24 or 1e6, 1.0))

def feature_coverage_mask(X, min_fraction=0.80):
    """Accept rows with enough real observations; missingness is encoded separately."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros(len(values) if values.ndim else 0, dtype=bool)
    # NaN is an explicit unknown; +/-Inf is invalid data and rejects the row.
    return (~np.isinf(values).any(axis=1)) & (np.isfinite(values).mean(axis=1) >= float(min_fraction))


def _rows_to_ds(rows, label_key="label", ret_key="ret"):
    """Build current-schema rows by feature name, preserving unknowns as NaN.

    Historical snapshots may have a different feature count/order. When their
    stored feature_names manifest is valid, map only exact name matches into the
    active schema. Newly introduced features remain unknown and are handled by
    Brain._norm's missingness channel; removed/renamed features are never guessed.
    Snapshots without a trustworthy name manifest are excluded rather than
    silently interpreting old vector positions as current features.
    """
    X, y, ts, ret, hold, coin = [], [], [], [], [], []
    current_names = list(FEATURE_COLS)
    current_index = {name: i for i, name in enumerate(current_names)}
    for r in rows:
        try:
            values = json.loads(r["feats"])
            raw_names = r["feature_names"] if "feature_names" in r.keys() else None
            names = json.loads(raw_names) if isinstance(raw_names, str) else raw_names
            if (not isinstance(values, list) or not isinstance(names, list)
                    or len(values) != len(names) or not names
                    or not all(isinstance(name, str) and name for name in names)
                    or len(set(names)) != len(names)):
                continue
            feature_count = r["feature_count"] if "feature_count" in r.keys() else None
            if feature_count is not None and int(feature_count) != len(values):
                continue
            vector = np.full(N_FEATURES, np.nan, dtype=np.float64)
            for name, value in zip(names, values):
                target = current_index.get(name)
                if target is None:
                    continue
                vector[target] = np.nan if value is None else float(value)
            if vector.shape != (N_FEATURES,):
                continue
            X.append(vector); y.append(r[label_key]); ts.append(r["ts"])
            ret.append(r[ret_key] if r[ret_key] is not None else np.nan)
            hold.append(r["hold"] if "hold" in r.keys() and r["hold"] is not None else np.nan)
            coin.append(r["inst"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not X:
        return empty_ds()
    return dict(X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.float32), ts=np.array(ts, dtype=np.float64),
                ret=np.array(ret, dtype=np.float64), hold=np.array(hold, dtype=np.float64),
                coin=np.array(coin, dtype="<U16"))

class Paper:
    """Signal history and model metadata only; no wallet, positions, or trade ledger."""
    def __init__(self):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(CFG["DB"], check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        # Keep paper signal tracking durable; remove only obsolete wallet/order ledgers.
        for legacy_table in ("trades", "pending_orders", "experience"):
            self.db.execute(f"DROP TABLE IF EXISTS {legacy_table}")
        signal_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(signals)")}
        if "feature_ts" not in signal_columns:
            self.db.execute("ALTER TABLE signals ADD COLUMN feature_ts REAL")
        if "prob_source" not in signal_columns:
            self.db.execute("ALTER TABLE signals ADD COLUMN prob_source TEXT NOT NULL DEFAULT 'legacy_unknown'")
        for column, declaration in (
            ("feature_count", "INTEGER"), ("feature_names", "TEXT"),
            ("feature_schema_version", "INTEGER")
        ):
            if column not in signal_columns:
                self.db.execute(f"ALTER TABLE signals ADD COLUMN {column} {declaration}")
        tracked_columns = {row["name"] for row in self.db.execute("PRAGMA table_info(tracked_signals)")}
        for column, declaration in (
            ("regime", "TEXT"), ("entry_score", "REAL"), ("peak_price", "REAL"),
            ("trailing_sl", "REAL"), ("trail_active", "INTEGER DEFAULT 0"),
            ("exit_sl_mult", "REAL"), ("exit_tp_mult", "REAL"),
            ("trail_activation_atr", "REAL"), ("trail_distance_atr", "REAL"),
            ("max_hold_bars", "INTEGER"), ("prob_source", "TEXT NOT NULL DEFAULT 'legacy_unknown'"),
            ("feature_count", "INTEGER"), ("feature_names", "TEXT"),
            ("feature_schema_version", "INTEGER")
        ):
            if column not in tracked_columns:
                self.db.execute(f"ALTER TABLE tracked_signals ADD COLUMN {column} {declaration}")
        self.db.execute("UPDATE signals SET feature_ts=ts WHERE feature_ts IS NULL")
        self.db.execute("UPDATE tracked_signals SET peak_price=entry WHERE peak_price IS NULL")
        # Idempotent indexes accelerate pending labels, chronological training,
        # closed-signal joins and per-symbol open-signal checks.
        self.db.executescript("""
            CREATE INDEX IF NOT EXISTS idx_signals_label_time
                ON signals(labeled, feature_ts, ts);
            CREATE INDEX IF NOT EXISTS idx_signals_inst_time
                ON signals(inst, feature_ts, ts);
            CREATE INDEX IF NOT EXISTS idx_tracked_closed_time
                ON tracked_signals(status, closed_ts, signal_id);
            CREATE INDEX IF NOT EXISTS idx_tracked_open_inst
                ON tracked_signals(status, inst);
        """)
        self.db.commit()

    @locked
    def _get(self, k):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return None if r is None else r["v"]

    @locked
    def _set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))
        self.db.commit()

    @locked
    def count(self, table, where="1=1"):
        if table != "signals":
            return 0
        return self.db.execute(f"SELECT COUNT(*) c FROM signals WHERE {where}").fetchone()["c"]

    @locked
    def log_signals(self, rows):
        now = time.time()
        # Ranking snapshots use the active runtime schema (300 in evolved_runner).
        # Unit-level/base imports may still use the core 200-column schema; actual
        # tracked trading signals are strictly guarded at 300 in create_tracked_signal.
        expected_count = int(N_FEATURES)
        feature_names = json.dumps(list(FEATURE_COLS), ensure_ascii=False, separators=(",", ":"))
        schema_version = int(DATA_VER)
        prepared = []
        for r in rows:
            vector = np.asarray(r["feats"], dtype=np.float64).reshape(-1)
            if (vector.size != expected_count or np.isinf(vector).any()
                    or not feature_coverage_mask(vector.reshape(1, -1), 0.80)[0]):
                raise ValueError(f"Invalid feature snapshot for {r.get('inst')}: expected {expected_count} values with >=80% known")
            encoded_vector = json.dumps([float(v) if np.isfinite(v) else None for v in vector],
                                        allow_nan=False, separators=(",", ":"))
            prepared.append((
                float(r.get("ts", now)), r["inst"], float(r["price"]), float(r["atr"]),
                encoded_vector,
                None if r.get("prob") is None or np.isnan(r["prob"]) else float(r["prob"]),
                int(r["rank"]), float(r.get("feature_ts", r.get("ts", now))),
                str(r.get("prob_source") or "ranking_exploration"),
                expected_count, feature_names, schema_version
            ))
        self.db.executemany(
            "INSERT INTO signals(ts,inst,price,atr,feats,prob,rank,feature_ts,prob_source,"
            "feature_count,feature_names,feature_schema_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            prepared)
        self.db.commit()

    @locked
    def pending_signals(self, cutoff_ts):
        return [dict(r) for r in self.db.execute(
            "SELECT id,ts,COALESCE(feature_ts,ts) AS feature_ts,inst,price,atr "
            "FROM signals WHERE labeled=0 AND ts<=? ORDER BY ts", (cutoff_ts,))]

    @locked
    def set_label(self, sid, label, ret, hold):
        self.db.execute("UPDATE signals SET labeled=1,label=?,ret=?,hold=? WHERE id=?",
                        (label, ret, hold, sid))
        self.db.commit()

    @locked
    def signal_dataset(self):
        return _rows_to_ds(self.db.execute(
            "SELECT COALESCE(feature_ts,ts) AS ts,inst,feats,label,ret,hold,feature_names,feature_count "
            "FROM signals WHERE labeled=1 ORDER BY feature_ts,ts").fetchall())

    @locked
    def closed_signal_dataset(self):
        """Training rows from tracked signals only, and only after their closure."""
        return _rows_to_ds(self.db.execute(
            "SELECT COALESCE(s.feature_ts,s.ts) AS ts,s.inst,s.feats,s.feature_names,s.feature_count,"
            "CASE WHEN s.ret>0 THEN 1 ELSE 0 END AS label,s.ret,s.hold "
            "FROM signals s JOIN tracked_signals t ON t.signal_id=s.id "
            "WHERE t.status='CLOSED' AND s.labeled=1 AND s.ret IS NOT NULL "
            "AND s.ret != 0 ORDER BY t.closed_ts,s.id").fetchall())

    @locked
    def closed_outcome_counts(self):
        # Count only outcomes whose feature snapshots are usable by the trainer.
        # A row with too little source data cannot open the supervised model gate.
        ds = self.closed_signal_dataset()
        if not len(ds["y"]):
            return dict(wins=0, losses=0, total=0)
        valid = (feature_coverage_mask(ds["X"], 0.80)
                 & np.isfinite(ds["y"]) & np.isin(ds["y"], [0, 1])
                 & np.isfinite(ds["ts"]) & np.isfinite(ds["ret"]) & (ds["ret"] != 0))
        labels = ds["y"][valid]
        wins = int(np.sum(labels == 1))
        losses = int(np.sum(labels == 0))
        return dict(wins=wins, losses=losses, total=wins + losses)

    @locked
    def adaptive_ranking_weights(self, regime, min_samples=20, lookback=500):
        """Estimate conservative regime-specific factor weights from realized net outcomes."""
        base = {
            "ret1h": 0.20, "ret4h": 0.20, "ret1d": 0.15, "ret15m": 0.10,
            "trend": 0.10, "volume_ratio": 0.08, "adx": 0.05, "obv": 0.05,
            "liquidity": 0.05, "atr_penalty": -0.10, "overbought_penalty": -0.10,
        }
        rows = self.db.execute(
            "SELECT feats,ret FROM tracked_signals WHERE status='CLOSED' AND regime=? "
            "AND ret IS NOT NULL ORDER BY closed_ts DESC LIMIT ?",
            (str(regime), int(lookback))).fetchall()
        if len(rows) < int(min_samples):
            return base, len(rows), False
        idx = {name: i for i, name in enumerate(FEATURE_COLS)}
        factors = {k: [] for k in base}
        returns = []
        for row in rows:
            try:
                v = np.asarray(json.loads(row["feats"]), dtype=np.float64)
                ret = float(row["ret"])
                if v.size != N_FEATURES or not np.isfinite(v).all() or not np.isfinite(ret):
                    continue
                def mean_tf(name):
                    return float(np.mean([v[idx[f"{tf}_{name}"]] for tf in TFS]))
                factors["ret1h"].append(v[idx["1H_ret20"]])
                factors["ret4h"].append(v[idx["4H_ret20"]])
                factors["ret1d"].append(v[idx["1D_ret10"]])
                factors["ret15m"].append(v[idx["15m_ret20"]])
                factors["trend"].append(mean_tf("ema_dist21"))
                factors["volume_ratio"].append(v[idx["1H_vol_ratio"]])
                factors["adx"].append(v[idx["1H_adx14"]])
                factors["obv"].append(v[idx["1H_obv_slope"]])
                factors["liquidity"].append(v[idx["ctx_log_dvol24"]])
                factors["atr_penalty"].append(v[idx["1H_atr_pct"]])
                factors["overbought_penalty"].append(max(v[idx["1H_rsi14"]] - 75.0, 0.0))
                returns.append(ret)
            except (ValueError, TypeError, KeyError, IndexError):
                continue
        n = len(returns)
        if n < int(min_samples):
            return base, n, False
        y = np.asarray(returns, dtype=np.float64)
        learned = {}
        # Correlation is shrunk toward zero and tightly bounded to prevent small
        # samples or outliers from radically changing the established ranking.
        for name, values in factors.items():
            x = np.asarray(values, dtype=np.float64)
            if len(x) != n or np.std(x) < 1e-12 or np.std(y) < 1e-12:
                learned[name] = 1.0
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
            direction = -1.0 if name in ("atr_penalty", "overbought_penalty") else 1.0
            aligned = float(np.clip(corr * direction, -0.5, 0.5))
            shrink = min(1.0, n / 200.0)
            learned[name] = float(np.clip(1.0 + aligned * shrink, 0.75, 1.25))
        adjusted = {k: base[k] * learned[k] for k in base}
        # Preserve the original total positive and negative weight budgets.
        for group, target in ((1, sum(v for v in base.values() if v > 0)),
                              (-1, sum(abs(v) for v in base.values() if v < 0))):
            keys = [k for k, v in adjusted.items() if v * group > 0]
            total = sum(abs(adjusted[k]) for k in keys)
            if total > 0:
                for k in keys:
                    adjusted[k] = group * abs(adjusted[k]) * target / total
        return adjusted, n, True

    @locked
    def regime_stats(self, regime):
        row = self.db.execute(
            "SELECT COUNT(*) AS n, AVG(ret) AS avg_ret, "
            "SUM(CASE WHEN ret>0 THEN ret ELSE 0 END) AS gross_profit, "
            "SUM(CASE WHEN ret<0 THEN -ret ELSE 0 END) AS gross_loss "
            "FROM tracked_signals WHERE status='CLOSED' AND regime=? AND ret IS NOT NULL",
            (str(regime),)).fetchone()
        return dict(n=int(row["n"] or 0), avg_ret=float(row["avg_ret"] or 0.0),
                    pf=(float(row["gross_profit"] or 0.0) / float(row["gross_loss"])
                        if float(row["gross_loss"] or 0.0) > 0 else None))

    @locked
    def update_trailing(self, sid, peak_price, trailing_sl, active):
        self.db.execute(
            "UPDATE tracked_signals SET peak_price=?, trailing_sl=?, trail_active=? "
            "WHERE id=? AND status='OPEN'",
            (float(peak_price), float(trailing_sl), int(bool(active)), int(sid)))
        self.db.commit()

    @locked
    def create_tracked_signal(self, row):
        now = float(row.get("ts", time.time()))
        inst = str(row["inst"])
        if self.db.execute("SELECT 1 FROM tracked_signals WHERE inst=? AND status='OPEN'", (inst,)).fetchone():
            return None
        if self.db.execute("SELECT 1 FROM tracked_signals WHERE inst=? AND opened_ts>=?",
                           (inst, now - CFG["TF_SEC"]["1H"])).fetchone():
            return None
        entry = float(row["price"])
        atr_pct = min(max(float(row["atr"]), 0.004), 0.06)
        sl_mult = sl_atr_multiplier(atr_pct)
        sl = entry * (1.0 - sl_mult * atr_pct)
        tp = entry * (1.0 + CFG["TP_ATR"] * atr_pct)
        regime = str(row.get("regime") or "UNKNOWN")
        entry_score = float(row.get("entry_score", 0.0))
        required_count = int(CFG.get("REQUIRED_SIGNAL_FEATURES", N_FEATURES))
        if int(N_FEATURES) != required_count:
            raise RuntimeError(f"Tracked signals require {required_count} runtime features; got {N_FEATURES}")
        vector = np.asarray(row["feats"], dtype=np.float64).reshape(-1)
        if (vector.size != required_count or np.isinf(vector).any()
                or not feature_coverage_mask(vector.reshape(1, -1), 0.80)[0]):
            raise ValueError(f"Invalid feature snapshot for {inst}: expected {required_count} values with >=80% known")
        feats = json.dumps([float(v) if np.isfinite(v) else None for v in vector],
                           allow_nan=False, separators=(",", ":"))
        feature_names = json.dumps(list(FEATURE_COLS), ensure_ascii=False, separators=(",", ":"))
        schema_version = int(DATA_VER)
        cur = self.db.execute(
            "INSERT INTO signals(ts,inst,price,atr,feats,prob,rank,labeled,feature_ts,prob_source,"
            "feature_count,feature_names,feature_schema_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, inst, entry, atr_pct, feats, None if row.get("prob") is None else float(row["prob"]),
             int(row["rank"]), 0, float(row.get("feature_ts", now)),
             str(row.get("prob_source") or "ranking_exploration"),
             required_count, feature_names, schema_version))
        sid = int(cur.lastrowid)
        self.db.execute(
            "INSERT INTO tracked_signals(signal_id,inst,opened_ts,entry,sl,tp,atr_pct,prob,rank,feats,regime,entry_score,peak_price,trailing_sl,trail_active,prob_source,"
            "feature_count,feature_names,feature_schema_version) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, inst, now, entry, sl, tp, atr_pct,
             None if row.get("prob") is None else float(row["prob"]), int(row["rank"]), feats,
             regime, entry_score, entry, sl, 0,
             str(row.get("prob_source") or "ranking_exploration"),
             required_count, feature_names, schema_version))
        self.db.commit()
        return dict(id=sid, inst=inst, entry=entry, sl=sl, tp=tp, prob=row.get("prob"),
                    rank=int(row["rank"]), regime=regime, entry_score=entry_score,
                    feature_count=required_count, feature_schema_version=schema_version)

    @locked
    def open_tracked_signals(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM tracked_signals WHERE status='OPEN' ORDER BY opened_ts")]

    @locked
    def close_tracked_signal(self, sid, closed_ts, exit_price, outcome, ret, hold, reason):
        row = self.db.execute("SELECT signal_id FROM tracked_signals WHERE id=? AND status='OPEN'", (sid,)).fetchone()
        if not row:
            return False
        self.db.execute(
            "UPDATE tracked_signals SET status='CLOSED',closed_ts=?,exit_price=?,outcome=?,ret=?,hold=?,close_reason=? WHERE id=? AND status='OPEN'",
            (float(closed_ts), float(exit_price), outcome, float(ret), float(hold), reason, sid))
        # Keep the learning label consistent with realized net return after costs.
        net_ret = float(ret)
        label = 1 if net_ret > 0 else 0
        self.db.execute("UPDATE signals SET labeled=1,label=?,ret=?,hold=? WHERE id=?",
                        (label, net_ret, float(hold), row["signal_id"]))
        self.db.commit()
        return True

    @locked
    def tracked_count(self, where="1=1"):
        return int(self.db.execute(f"SELECT COUNT(*) c FROM tracked_signals WHERE {where}").fetchone()["c"])

# =====================================================================
# SIGNAL LABELLING# =====================================================================
# SIGNAL LABELLING (triple barrier on real later prices)
# =====================================================================
def future_candles_for_signal(candles, created_ts, timeframe="1H"):
    """Return only candles whose opening time is not before signal creation."""
    if candles is None or candles.empty:
        return candles
    created_ms = int(float(created_ts) * 1000)
    interval_ms = CFG["TF_SEC"][timeframe] * 1000
    first_future_start = ((created_ms + interval_ms - 1) // interval_ms) * interval_ms
    return candles[candles["ts"] >= first_future_start]


def label_signals(paper):
    H = CFG["BARRIER_H"]
    now = time.time()
    total_unlabelled = paper.count("signals", "labeled=0")
    cutoff = now - (H + 2) * 3600
    rows = paper.pending_signals(cutoff)
    if not rows:
        log.info("signal labelling: total=%d unlabelled=%d mature=0 labelled_now=0 "
                 "(waiting for %dh horizon + safety buffer)",
                 paper.count("signals"), total_unlabelled, H + 2)
        return 0
    by = {}
    for r in rows:
        by.setdefault(r["inst"], []).append(r)
    n = no_candles = insufficient = errors = 0
    for inst, rs in by.items():
        oldest = min(r["ts"] for r in rs)
        depth = min(1400, max(H + 5, int((now - oldest) / 3600) + H + 5))
        try:
            d = get_candles(inst, "1H", depth)
        except Exception as exc:
            errors += len(rs)
            log.warning("signal labelling: candle fetch failed for %s: %s", inst, exc)
            continue
        if d is None or d.empty:
            no_candles += len(rs)
            log.warning("signal labelling: no 1H candles for %s (%d mature signals)", inst, len(rs))
            continue
        for r in rs:
            # Outcomes must begin after the actual signal/entry time, not merely
            # after feature availability: the candle containing a mid-hour signal
            # includes price movement that occurred before the entry.
            path = future_candles_for_signal(d, r["ts"])
            if len(path) < H:
                insufficient += 1
                continue
            try:
                ret, lab, hold = barrier_single(r["price"], r["atr"], path["h"].values,
                                                path["l"].values, path["c"].values)
                if not np.isfinite(ret) or not np.isfinite(hold) or lab not in (0, 1):
                    errors += 1
                    log.warning("signal labelling: invalid barrier outcome signal_id=%s inst=%s", r["id"], inst)
                    continue
                paper.set_label(r["id"], lab, ret, hold)
                n += 1
            except Exception as exc:
                errors += 1
                log.warning("signal labelling: outcome failed signal_id=%s inst=%s: %s", r["id"], inst, exc)
    log.info("signal labelling: total=%d unlabelled=%d mature=%d labelled_now=%d "
             "no_candles=%d insufficient_future_candles=%d errors=%d",
             paper.count("signals"), total_unlabelled, len(rows), n, no_candles, insufficient, errors)
    return n

def classify_exit_return(entry, exit_price):
    """Classify an exit using net return after the configured round-trip costs."""
    entry, exit_price = float(entry), float(exit_price)
    if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0 or exit_price <= 0:
        raise ValueError("entry and exit_price must be finite positive prices")
    ret = exit_price / entry - 1.0 - CFG["BT_COST"]
    if abs(ret) < 1e-12:
        ret = 0.0
    return ret, ("WIN" if ret > 0 else "LOSS" if ret < 0 else "FLAT")


def monitor_tracked_signals(paper):
    """Resolve paper signals from completed 5m candles; stop-loss wins candle ties."""
    opened = paper.open_tracked_signals()
    if not opened:
        return 0
    monitor_tf = "5m"
    max_hold_bars = 24 * 60 // 5  # preserve the existing 24-hour maximum horizon
    now, closed = time.time(), 0
    for s in opened:
        try:
            # 288 completed 5m bars cover 24 hours; keep extra bars for boundary alignment.
            d = get_candles(s["inst"], monitor_tf, max_hold_bars + 5)
            path = future_candles_for_signal(d, s["opened_ts"], monitor_tf)
            if path is None or path.empty:
                continue
            elapsed = 0
            for _, candle in path.iterrows():
                if float(candle["ts"]) / 1000 + CFG["TF_SEC"][monitor_tf] > now:
                    continue
                elapsed += 1
                low, high, close = float(candle["l"]), float(candle["h"]), float(candle["c"])
                reason = outcome = None
                exit_price = close
                # Evaluate the stop already active before this candle; do not use this
                # candle's high to move a stop and then retroactively trigger it on its low.
                active_sl = max(float(s["sl"]), float(s.get("trailing_sl") or s["sl"]))
                if low <= active_sl:
                    reason, outcome, exit_price = "TRAIL" if active_sl > float(s["sl"]) + 1e-12 else "SL", "SL", active_sl
                elif high >= s["tp"]:
                    reason, outcome, exit_price = "TP", "TP", float(s["tp"])
                elif elapsed >= int(s.get("max_hold_bars") or max_hold_bars):
                    reason = "TIME"
                if reason is None:
                    old_peak = float(s.get("peak_price") or s["entry"])
                    peak = max(old_peak, high)
                    entry_price = float(s["entry"])
                    atr_pct = float(s.get("atr_pct") or 0.0)
                    if atr_pct <= 0:
                        atr_pct = max(1e-6, (float(s["tp"]) - entry_price) /
                                      (entry_price * max(float(s.get("exit_tp_mult") or 3.0), 1e-6)))
                    tp_atr = max(1e-6, (float(s["tp"]) - entry_price) / (entry_price * atr_pct))
                    activation_atr = float(s.get("trail_activation_atr") or (0.30 * tp_atr))
                    distance_atr = float(s.get("trail_distance_atr") or 1.8)
                    trail, active = exit_evolution.trailing_stop_update(
                        entry_price, atr_pct, peak,
                        float(s.get("trailing_sl") or s["sl"]),
                        bool(s.get("trail_active")), activation_atr, distance_atr,
                        tp_distance=float(s["tp"]) - entry_price)
                    # Live monitoring and shadow backtests use the same TP-relative
                    # 30% activation / 5% ratchet rule; the stop is monotonic.
                    trail = max(float(s["sl"]), float(s.get("trailing_sl") or s["sl"]), trail)
                    paper.update_trailing(s["id"], peak, trail, active)
                    s["peak_price"], s["trailing_sl"], s["trail_active"] = peak, trail, int(active)
                if reason:
                    ret, net_outcome = classify_exit_return(s["entry"], exit_price)
                    if reason == "TIME":
                        outcome = net_outcome
                    elif reason == "TRAIL":
                        outcome = "LOSS" if ret < 0 else ("FLAT" if ret == 0 else "WIN")
                    elif reason == "TP":
                        outcome = "WIN" if ret > 0 else ("FLAT" if ret == 0 else "LOSS")
                    else:
                        outcome = "LOSS" if ret < 0 else ("FLAT" if ret == 0 else "WIN")
                    if paper.close_tracked_signal(s["id"], now, exit_price, outcome, ret, elapsed, reason):
                        closed += 1
                        emoji = "✅" if reason == "TP" else ("⏳" if reason == "TIME" else ("🔒" if reason == "TRAIL" else "🛑"))
                        tg(f"{emoji} إغلاق إشارة ورقية: {s['inst']}\n"
                           f"السبب: {reason} | الدخول: {s['entry']:.8g} | الخروج: {exit_price:.8g}\n"
                           f"العائد الصافي التقديري بعد التكاليف: {ret*100:+.2f}% | المدة: {elapsed * 5} دقيقة")
                    break
        except Exception as exc:
            log.warning("paper signal monitor failed for %s: %s", s["inst"], exc)
    log.info("paper signal monitor (5m): open=%d closed_now=%d", paper.tracked_count("status='OPEN'"), closed)
    return closed


# =====================================================================
# RANKING (best -> worst)
# =====================================================================
def rank_universe(X, vol24, paper=None, regimes=None, spread24=None):
    """Cross-sectional ranking with conservative regime-specific learned factor weights."""
    z = lambda s: ((s - s.median()) / (s.std() + 1e-9)).clip(-3, 3)
    trend = sum(X[f"{tf}_ema_dist21"] for tf in TFS) / len(TFS)
    overbought = (X["1H_rsi14"] - 75).clip(lower=0)
    factors = {
        "ret1h": z(X["1H_ret20"]), "ret4h": z(X["4H_ret20"]),
        "ret1d": z(X["1D_ret10"]), "ret15m": z(X["15m_ret20"]),
        "trend": z(trend), "volume_ratio": z(X["1H_vol_ratio"]),
        "adx": z(X["1H_adx14"]), "obv": z(X["1H_obv_slope"]),
        "liquidity": z(np.log(vol24.reindex(X.index))),
        "atr_penalty": z(X["1H_atr_pct"]), "overbought_penalty": z(overbought),
        "spread_penalty": z((spread24.reindex(X.index) if spread24 is not None else pd.Series(0.0, index=X.index))),
    }
    base = {"ret1h":.20,"ret4h":.20,"ret1d":.15,"ret15m":.10,"trend":.10,
            "volume_ratio":.08,"adx":.05,"obv":.05,"liquidity":.05,
            "atr_penalty":-.10,"overbought_penalty":-.10,"spread_penalty":-.08}
    score = pd.Series(0.0, index=X.index)
    diagnostics = {}
    regime_series = regimes.reindex(X.index) if regimes is not None else pd.Series("UNKNOWN", index=X.index)
    for regime in regime_series.dropna().unique():
        coins = regime_series.index[regime_series == regime]
        weights, samples, adaptive = (paper.adaptive_ranking_weights(regime)
                                      if paper is not None else (base, 0, False))
        # Sum observed evidence only; missing factors are not treated as zero.
        local_weights = {k: float(weights.get(k, base[k])) for k in base}
        weighted = pd.DataFrame({k: factors[k].loc[coins] * local_weights[k] for k in base}, index=coins)
        observed_weight = pd.DataFrame({
            k: factors[k].loc[coins].notna().astype(float) * abs(local_weights[k])
            for k in base
        }, index=coins).sum(axis=1)
        part = weighted.sum(axis=1, min_count=1)
        part = part * sum(abs(v) for v in local_weights.values()) / observed_weight.replace(0.0, np.nan)
        score.loc[coins] = part
        diagnostics[str(regime)] = {"samples": int(samples), "adaptive": bool(adaptive),
                                    "weights": weights}
    score.attrs["adaptive_diagnostics"] = diagnostics
    return score.sort_values(ascending=False)


def classify_market_regime(features):
    """Classify the broad market using only information available at signal time."""
    btc = float(features.get("ctx_btc_regime", np.nan))
    adx = float(features.get("1H_adx14", np.nan))
    atr = float(features.get("1H_atr_pct", np.nan))
    if not np.isfinite([btc, adx, atr]).all():
        return "UNKNOWN"
    if atr >= 0.035:
        return "HIGH_VOL"
    if btc >= 0.015 and adx >= 18:
        return "BULL"
    if btc <= -0.015 and adx >= 18:
        return "BEAR"
    if adx < 18:
        return "RANGE"
    return "TREND"


def evaluate_entry_quality(features):
    """Prefer pullbacks/rebounds and reject late, overextended entries; score 0..100."""
    f = features
    rsi = float(f["1H_rsi14"])
    atr = max(float(f["1H_atr_pct"]), 1e-6)
    dist = float(f["1H_ema_dist21"])
    ret15 = float(f["15m_ret20"])
    wick = float(f["1H_lower_wick"])
    body = float(f["1H_body_ratio"])
    adx = float(f["1H_adx14"])
    score = 50.0
    score += 15 if 42 <= rsi <= 62 else (7 if 35 <= rsi <= 70 else -18)
    score += 14 if -0.8 * atr <= dist <= 0.8 * atr else (4 if dist <= 1.4 * atr else -18)
    score += 10 if ret15 > 0 else (-8 if ret15 < -0.01 else 0)
    score += 8 if wick >= 0.25 else 0
    score += 5 if body > 0 else 0
    score += 8 if 18 <= adx <= 40 else (2 if adx > 40 else -5)
    stretched = rsi >= 75 and dist > 1.5 * atr
    return float(np.clip(score, 0.0, 100.0)), bool(stretched)


def regime_entry_threshold(paper, regime):
    """Adapt entry-quality strictness to realized net performance in this market regime."""
    stats = paper.regime_stats(regime)
    if stats["n"] < 30:
        return 55.0, stats
    pf = stats.get("pf")
    avg = float(stats.get("avg_ret") or 0.0)
    if avg <= 0.0 or (pf is not None and pf < 0.90):
        return 68.0, stats
    if avg < 0.001 or (pf is not None and pf < 1.05):
        return 62.0, stats
    if stats["n"] >= 60 and pf is not None and pf >= 1.20:
        return 52.0, stats
    return 57.0, stats


def effective_probability_threshold(brain):
    """Use the model's validated, net-return-selected threshold unless explicitly overridden."""
    configured = CFG.get("PROB_MIN")
    if configured is not None:
        value = float(configured)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("CFG.PROB_MIN must be finite and within [0, 1]")
        return value
    if brain is not None and brain.ready:
        value = float(brain.thr)
        if np.isfinite(value) and 0.0 <= value <= 1.0:
            return value
        raise ValueError("validated model threshold is invalid")
    return 0.55

# =====================================================================
# WALK-FORWARD BACKTEST
# =====================================================================
def simulate(idx, ds):
    """Replay time-ordered signals with max positions, one per coin and aggregate stop-risk budget."""
    ts, ret, hold, coin = ds["ts"][idx], ds["ret"][idx], ds["hold"][idx], ds["coin"][idx]
    # Some small unit fixtures contain only outcomes; use a neutral ATR in that case.
    atr = (ds["X"][idx, FEATURE_COLS.index("1H_atr_pct")] if "X" in ds
           else np.full(len(idx), 0.01, dtype=np.float64))
    open_, taken, events = [], [], []
    for i in np.argsort(ts, kind="stable"):
        t = ts[i]
        if not np.isfinite([ret[i], hold[i], atr[i]]).all():
            continue
        open_ = [p for p in open_ if p[0] > t]
        if len(open_) >= CFG["MAX_OPEN"] or any(p[1] == coin[i] for p in open_):
            continue
        a = float(np.clip(atr[i], 0.004, 0.06))
        sl_mult = sl_atr_multiplier(a)
        size_frac = min(CFG["POS_FRAC"], CFG["RISK_PER_TRADE"] / (sl_mult * a))
        risk_frac = size_frac * sl_mult * a
        if sum(p[2] for p in open_) + risk_frac > CFG["MAX_PORTFOLIO_RISK"] + 1e-12:
            continue
        end = t + (hold[i] if not np.isnan(hold[i]) else CFG["BARRIER_H"]) * 3600
        open_.append((end, coin[i], risk_frac)); taken.append(ret[i]); events.append((end, ret[i], size_frac))
    eq = peak = 1.0; mdd = 0.0
    for _, r, size_frac in sorted(events):
        eq *= 1 + size_frac * r; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    r = np.array(taken)
    if not len(r):
        return dict(n=0, win=0.0, avg=0.0, pf=0.0, total=0.0, mdd=0.0)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=int(len(r)), win=float((r > 0).mean()), avg=float(r.mean()),
                pf=float(gp / gl) if gl > 0 else float("inf"), total=float(eq - 1), mdd=float(mdd))

def walk_forward(ds, folds=5, epochs=30, min_train_frac=0.4):
    o = np.argsort(ds["ts"], kind="stable")
    ds = {k: v[o] for k, v in ds.items()}
    ts = ds["ts"]
    edges = np.linspace(ts.min() + (ts.max() - ts.min()) * min_train_frac, ts.max() + 1, folds + 1)
    emb = CFG["BARRIER_H"] * 3600
    oos = {}                                   # mode -> {"idx": [], "sig": [], "p": []}
    for k in range(folds):
        te = (ts >= edges[k]) & (ts < edges[k + 1]); tr = ts < edges[k] - emb
        if te.sum() < 50 or tr.sum() < 800:
            continue
        b = Brain(persist=False)
        try:
            b.fit(ds["X"][tr], ds["y"][tr], ds["ts"][tr], ds["ret"][tr], epochs=epochs, force=True)
        except Exception as e:
            log.warning("fold %d skipped: %s", k, e); continue
        for mode, p in b.predict_all(ds["X"][te]).items():
            d = oos.setdefault(mode, {"idx": [], "sig": [], "p": []})
            d["idx"].append(np.where(te)[0]); d["sig"].append(p >= b.thrs[mode]); d["p"].append(p)
        log.info("fold %d/%d done (train=%d test=%d)", k + 1, folds, tr.sum(), te.sum())
    if not oos:
        raise RuntimeError("backtest produced no out-of-sample fold (not enough data)")
    rep = {}
    for mode, d in oos.items():
        idx, sig, p = np.concatenate(d["idx"]), np.concatenate(d["sig"]), np.concatenate(d["p"])
        rep[mode] = dict(auc=auc_score(ds["y"][idx], p), n_oos=int(len(idx)),
                         base_win=float(ds["y"][idx].mean()), base_avg=float(np.nanmean(ds["ret"][idx])),
                         **simulate(idx[sig], ds))
    return rep

def backtest_report(rep):
    lines = ["📉 Backtest walk-forward (خارج العينة، بعد الرسوم)"]
    for mode, r in sorted(rep.items(), key=lambda kv: -kv[1]["avg"]):
        name = {"nn": "شبكة 20 طبقة", "gbm": "GBM", "ens": "دمج"}.get(mode, mode)
        lines.append(f"\n{name}: AUC {r['auc']:.3f} | صفقات {r['n']} | نجاح {r['win'] * 100:.0f}% "
                     f"(الأساس {r['base_win'] * 100:.0f}%)\n"
                     f"متوسط/صفقة {r['avg'] * 100:+.2f}% (الأساس {r['base_avg'] * 100:+.2f}%) | PF {r['pf']:.2f} | "
                     f"عائد {r['total'] * 100:+.1f}% | أقصى تراجع {r['mdd'] * 100:.1f}%")
    best = max(rep.values(), key=lambda r: r["avg"])
    ok = best["n"] >= 30 and best["avg"] > 0 and best["pf"] > 1.1 and best["avg"] > best["base_avg"]
    lines.append("\n" + ("✅ توجد أفضلية أولية، تحتاج تأكيداً بـ paper trading." if ok
                         else "⚠️ لا توجد أفضلية موثوقة بعد: لا تعتمد على الإشارات."))
    return "\n".join(lines)

# =====================================================================
# TELEGRAM COMMANDS
# =====================================================================
def stats_msg(paper, prices=None):
    total = paper.count("signals")
    labelled = paper.count("signals", "labeled=1")
    pending = paper.count("signals", "labeled=0")
    ds = paper.signal_dataset()
    n = len(ds["y"])
    wins = int(np.sum(ds["y"] > 0.5)) if n else 0
    avg = float(np.mean(ds["ret"]) * 100) if n else 0.0
    opened = paper.tracked_count("status='OPEN'")
    tracked = paper.tracked_count("status='CLOSED'")
    tracked_wins = paper.tracked_count("status='CLOSED' AND ret>0")
    closed_counts = paper.closed_outcome_counts()
    closed_rows = [dict(r) for r in paper.db.execute(
        "SELECT inst,ret,close_reason,regime,rank,entry_score,atr_pct,hold "
        "FROM tracked_signals WHERE status='CLOSED' AND ret IS NOT NULL ORDER BY closed_ts,id")]
    realized = np.asarray([float(x["ret"]) for x in closed_rows], dtype=float)
    gross_profit = float(realized[realized > 0].sum()) if len(realized) else 0.0
    gross_loss = float(-realized[realized < 0].sum()) if len(realized) else 0.0
    pf_realized = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    # Signals may overlap and have no common position size in the core ledger.
    # Use equal-notional additive returns for diagnostics; do not pretend every
    # signal compounds the full account as if trades were strictly sequential.
    equity_path = 1.0 + np.concatenate(([0.0], np.cumsum(realized))) if len(realized) else np.array([1.0])
    peaks = np.maximum.accumulate(equity_path)
    drawdowns = np.divide(peaks - equity_path, peaks,
                          out=np.zeros_like(peaks), where=peaks > 1e-12)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
    cumulative_signal_return = float(np.sum(realized)) if len(realized) else 0.0
    realized_avg = float(realized.mean() * 100) if len(realized) else 0.0
    regime_lines = []
    for regime in ("BULL", "BEAR", "RANGE", "TREND", "HIGH_VOL", "UNKNOWN"):
        group = [float(x["ret"]) for x in closed_rows if x["regime"] == regime]
        if group:
            regime_lines.append(f"{regime}: {len(group)} | WR {sum(v>0 for v in group)/len(group)*100:.1f}% | Avg {np.mean(group)*100:+.2f}%")
    # Compact loss diagnostics by exit reason, rank band and entry quality.
    reason_lines = []
    for reason in ("SL", "TRAIL", "TP", "TIME"):
        vals = [float(x["ret"]) for x in closed_rows if x["close_reason"] == reason]
        if vals:
            reason_lines.append(f"{reason}: {len(vals)} | خسائر {sum(v < 0 for v in vals)} | متوسط {np.mean(vals)*100:+.2f}%")
    segment_lines = []
    segments = (
        ("رتبة 1-3", lambda x: x["rank"] is not None and 1 <= int(x["rank"]) <= 3),
        ("رتبة 4-7", lambda x: x["rank"] is not None and 4 <= int(x["rank"]) <= 7),
        ("رتبة 8-10", lambda x: x["rank"] is not None and 8 <= int(x["rank"]) <= 10),
        ("جودة دخول <60", lambda x: x["entry_score"] is not None and float(x["entry_score"]) < 60),
        ("جودة دخول 60-74", lambda x: x["entry_score"] is not None and 60 <= float(x["entry_score"]) < 75),
        ("جودة دخول 75+", lambda x: x["entry_score"] is not None and float(x["entry_score"]) >= 75),
    )
    for label, predicate in segments:
        vals = [float(x["ret"]) for x in closed_rows if predicate(x)]
        if vals:
            segment_lines.append(f"{label}: n={len(vals)}, خسائر={sum(v < 0 for v in vals)}, متوسط={np.mean(vals)*100:+.2f}%")
    losing_coins = {}
    for row in closed_rows:
        if float(row["ret"]) < 0:
            losing_coins.setdefault(row["inst"], []).append(float(row["ret"]))
    worst_coins = " | ".join(
        f"{coin} ({len(vals)} خسائر، متوسط {np.mean(vals)*100:+.2f}%)"
        for coin, vals in sorted(losing_coins.items(), key=lambda kv: (-len(kv[1]), np.mean(kv[1])))[:5])
    gate_text = (f"بيانات النموذج كافية ({closed_counts['total']}/{MIN_CLOSED_MODEL_SAMPLES})"
                 if closed_counts["total"] >= MIN_CLOSED_MODEL_SAMPLES
                 else f"يتعلم من الذاكرة المتاحة؛ النموذج الإشرافي ينتظر {MIN_CLOSED_MODEL_SAMPLES} عينة موثقة")
    return (f"📊 تقييم سجل الإشارات الورقية\n"
            f"سجل التعلم: {total} | مكتملة التقييم: {labelled} | تنتظر: {pending}\n"
            f"مفتوحة: {opened} | مغلقة: {tracked} | رابحة: {tracked_wins}\n"
            f"حالة التعلم: {gate_text} | عينات مغلقة: {closed_counts['total']} (رابحة {closed_counts['wins']} / خاسرة {closed_counts['losses']})\n"
            f"رابحة حسب بيانات التدريب: {wins} | خاسرة: {n-wins} | نسبة النجاح: {(wins/n*100 if n else 0):.1f}%\n"
            f"متوسط العائد النظري لسجل التعلم: {avg:+.2f}%\n"
            f"\n📈 الأداء المحقق للإشارات المغلقة (صافي تقديري بعد التكلفة):\n"
            f"عدد الصفقات: {len(realized)} | نسبة النجاح: {(np.mean(realized>0)*100 if len(realized) else 0):.1f}%\n"
            f"متوسط الصفقة: {realized_avg:+.2f}% | Profit Factor: {pf_realized:.2f}\n"
            f"تراكم عوائد الإشارات غير الموزونة: {cumulative_signal_return*100:+.2f}% | أقصى تراجع نسبي: {max_dd*100:.2f}%\n"
            f"الأداء حسب حالة السوق: " + (" | ".join(regime_lines) if regime_lines else "لا توجد نتائج كافية بعد") +
            f"\n\n🔎 تشخيص الخسائر حسب سبب الإغلاق: " + (" | ".join(reason_lines) if reason_lines else "لا توجد بيانات") +
            f"\nحسب ترتيب العملة وجودة الدخول: " + (" | ".join(segment_lines) if segment_lines else "لا توجد بيانات") +
            f"\nالعملات الأكثر تكراراً في الخسائر: " + (worst_coins or "لا توجد خسائر مسجلة"))

def top_msg():
    p = os.path.join(CFG["DIR"], "ranking.csv")
    if not os.path.exists(p):
        return "لا يوجد تصنيف بعد."
    R = pd.read_csv(p, index_col=0).head(10)
    score_col = "ranking_score" if "ranking_score" in R.columns else "score"
    return "📊 أفضل 10 عملات — ترتيب القوة النسبية\n" + "\n".join(
        f"{int(r['rank'])}. {i.replace('-USDT', '')} | Ranking Score {float(r[score_col]):+.2f}"
        + (f" | AI Probability {r['ai_probability'] * 100:.1f}%"
           if "ai_probability" in R.columns and not pd.isna(r["ai_probability"])
           else " | AI: غير متاح") for i, r in R.iterrows())

def handle_command(text, paper, brain):
    cmd = text.split()[0].split("@")[0].lower()
    if cmd in ("/stats", "/status", "/signals", "/portfolio", "/محفظتي"):
        return stats_msg(paper)
    if cmd == "/top":
        return top_msg()
    if cmd == "/pause":
        paper._set("paused", "1")
        return "⏸ تم إيقاف تسجيل الإشارات الجديدة؛ سيستمر تقييم الإشارات المسجلة."
    if cmd == "/resume":
        paper._set("paused", "0")
        return "▶️ تم استئناف تسجيل الإشارات."
    if cmd == "/model":
        return "🧠 " + brain.describe()
    return "الأوامر: /stats /signals /top /model /pause /resume"

def poll_commands(paper, brain, timeout=0):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return 0
    params = {"timeout": timeout}
    off = paper._get("tg_offset")
    if off:
        params["offset"] = int(off) + 1
    try:
        j = requests.get(f"https://api.telegram.org/bot{tok}/getUpdates", params=params, timeout=timeout + 10).json()
    except Exception:
        return 0
    n = 0
    for u in j.get("result", []):
        paper._set("tg_offset", u["update_id"])
        m = u.get("message") or {}
        if str(m.get("chat", {}).get("id")) != str(chat):         # only the owner chat is obeyed
            continue
        text = (m.get("text") or "").strip()
        if text.startswith("/"):
            tg(handle_command(text, paper, brain)); n += 1
    return n

# =====================================================================
# PIPELINE
# =====================================================================
MIN_CLOSED_MODEL_SAMPLES = 600
MIN_CLOSED_CLASS_SAMPLES = 10
LEARNING_REFRESH_SAMPLES = 1


def learning_gate_status(paper):
    """Use all verified closed outcomes; no arbitrary 500-win/500-loss quota."""
    counts = paper.closed_outcome_counts()
    ready = (counts["total"] >= MIN_CLOSED_MODEL_SAMPLES
             and counts["wins"] >= MIN_CLOSED_CLASS_SAMPLES
             and counts["losses"] >= MIN_CLOSED_CLASS_SAMPLES)
    return dict(**counts, ready=ready,
                min_samples=MIN_CLOSED_MODEL_SAMPLES,
                min_per_class=MIN_CLOSED_CLASS_SAMPLES)


def enforce_closed_outcome_gate(brain, paper):
    """Keep validated closed-signal models; never erase memory for a sample-count shortfall."""
    status = learning_gate_status(paper)
    legacy_model = brain.ready and brain.info.get("learning_source") != "closed_tracked_signals_v1"
    if legacy_model:
        brain.nn = brain.gbm = None
        brain.mu = brain.sd = None
        brain.thrs, brain.mode, brain.auc, brain.info = {}, None, None, {}
        # Remove old model artifacts so the dashboard and next runner cannot mistake them for an active model.
        for model_path in (CFG["MODEL"], CFG["GBM"], CFG["META"]):
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
            except OSError as exc:
                log.warning("could not remove non-compliant model artifact %s: %s", model_path, exc)
    # Pattern/strategy memory is retained and may inform conservative ranking before
    # the supervised model has enough balanced, verified outcomes.
    return status


def collect_training_data(paper, universe=None, refresh=False, with_trades=False):
    # Learn only from observed, completed tracked signals; never invent missing outcomes.
    status = learning_gate_status(paper)
    ds = paper.closed_signal_dataset()
    # Do not manufacture missing feature values for closed-signal learning.
    # Keep NaN as an explicit unknown: Brain._norm encodes both the normalized
    # value and a separate missingness indicator. Reject only infinities, rows
    # below the established 80% observed-feature threshold, or invalid labels.
    valid = (feature_coverage_mask(ds["X"], 0.80)
             & np.isfinite(ds["y"])
             & np.isin(ds["y"], [0, 1])
             & np.isfinite(ds["ts"])
             & np.isfinite(ds["ret"]))
    excluded_incomplete = int(len(valid) - np.sum(valid))
    ds = {k: v[valid] for k, v in ds.items()}
    if excluded_incomplete:
        log.info("closed-signal memory: excluded %d rows below 80%% feature coverage or with invalid labels/timestamps/returns (NaN retained as unknown)",
                 excluded_incomplete)
    wins = int(np.sum(ds["y"] == 1))
    losses = int(np.sum(ds["y"] == 0))
    log.info("verified closed-signal memory: samples=%d wins=%d losses=%d (minimum total=%d, per class=%d)",
             len(ds["y"]), wins, losses, MIN_CLOSED_MODEL_SAMPLES, MIN_CLOSED_CLASS_SAMPLES)
    if len(ds["y"]) < MIN_CLOSED_MODEL_SAMPLES:
        raise RuntimeError(f"waiting for {MIN_CLOSED_MODEL_SAMPLES} verified closed outcomes; have {len(ds['y'])}")
    if min(wins, losses) < MIN_CLOSED_CLASS_SAMPLES:
        raise RuntimeError(f"insufficient real examples in one outcome class; wins={wins}, losses={losses}, minimum={MIN_CLOSED_CLASS_SAMPLES}")
    return ds

def retrain(brain, paper, universe=None, reason="manual", refresh=False):
    status = learning_gate_status(paper)
    if not status["ready"]:
        log.info("AI supervised model waiting for verified data: samples=%d/%d wins=%d losses=%d (memory learning remains active)",
                 status["total"], MIN_CLOSED_MODEL_SAMPLES, status["wins"], status["losses"])
        return {"accepted": False, "skipped": True, "reason": "insufficient_verified_closed_data",
                "wins": status["wins"], "losses": status["losses"], "n": status["total"]}
    ds = collect_training_data(paper, universe, refresh)
    info = brain.fit(ds["X"], ds["y"], ds["ts"], ds["ret"])
    if info.get("accepted") and brain.ready:
        # Stamp and persist provenance so future runs never reactivate a legacy model.
        info["learning_source"] = "closed_tracked_signals_v1"
        brain.info["learning_source"] = "closed_tracked_signals_v1"
        brain.save(N_FEATURES)
    paper._set("last_train_sig", paper.count("signals", "labeled=1"))
    aucs = " | ".join(f"{k}:{v:.3f}" for k, v in info["aucs"].items())
    top = info.get("val_top_net")
    baseline = info.get("baseline_test_avg_net", info.get("val_all_net"))
    test_n, test_pf = info.get("test_n"), info.get("test_pf")
    rates = [info.get(k) for k in ("train_positive_rate", "tune_positive_rate", "test_positive_rate")]
    diag = ""
    if baseline is not None and np.isfinite(baseline):
        diag += f"\nمتوسط كل إشارات الاختبار: {baseline * 100:+.2f}%"
    if test_n is not None and test_pf is not None:
        diag += f" | صفقات النموذج: {int(test_n)} | PF: {float(test_pf):.2f}"
    if all(v is not None and np.isfinite(v) for v in rates):
        diag += f"\nنسبة العينات الرابحة train/tune/test: {rates[0] * 100:.1f}%/{rates[1] * 100:.1f}%/{rates[2] * 100:.1f}%"
    tg(f"🧠 تدريب النماذج [{reason}]\nعيّنات: {info['n']} | AUC اختيار ({aucs})"
       f" | Gate AUC: {info.get('gate_auc', info.get('auc', 0.0)):.3f}\n"
       f"النمط المختار: {info['mode']}"
       + (f" | متوسط صفقات الإشارة في التحقق: {top * 100:+.2f}%" if top is not None and np.isfinite(top) else "")
       + diag
       + f"\n{'تم اعتماد النموذج ✅' if info['accepted'] else 'لم يُعتمد (أسوأ من السابق أو ضعيف) ⚠️'}")
    return info

def need_retrain(paper, brain=None):
    """Refresh supervised learning after each newly verified closed outcome.

    The 600-sample gate is required before the first fit. After that, every
    additional verified close triggers a refresh attempt. A candidate still
    must pass the untouched net-performance gate before it can become active.
    """
    status = learning_gate_status(paper)
    if not status["ready"]:
        return False
    previous = int(paper._get("last_learning_attempt_closed_count") or 0)
    if previous <= 0:
        return True
    if (brain is not None and not brain.ready
            and isinstance(brain.load_issue, dict)
            and brain.load_issue.get("reason") == "incompatible_feature_or_missing_value_schema"
            and paper._get("last_learning_attempt_schema_hash") != feature_schema_fingerprint()):
        # A newly observed schema mismatch must trigger one rebuild immediately,
        # even when the previous code already recorded an attempt at this sample count.
        return True
    retry_samples = LEARNING_REFRESH_SAMPLES
    return status["total"] - previous >= retry_samples

def _hourly_returns(inst):
    d = get_candles(inst, "1H", 100)
    if d is None:
        return None
    return pd.Series(np.log(d["c"]).diff().values, index=d["ts"].values).dropna()

def max_corr_with_open(inst, open_insts, cache):
    if not open_insts:
        return 0.0
    for k in [inst] + list(open_insts):
        if k not in cache:
            cache[k] = _hourly_returns(k)
    a, best = cache[inst], 0.0
    if a is None:
        return 0.0
    for o in open_insts:
        b = cache[o]
        if b is None:
            continue
        x = pd.concat([a, b], axis=1, join="inner").tail(72)
        if len(x) >= 30:
            best = max(best, float(x.corr().iloc[0, 1]))
    return best

def scan(brain, paper, universe=None, cycle=0, trade=True):
    uni = universe if universe is not None else get_universe()
    log.info("universe: %d coins", len(uni))
    if uni.empty:
        return None
    btc = fetch_btc(CFG["SCAN_DEPTH"], CFG["SCAN_DEPTH"])
    def one(inst):
        try:
            fr = fetch_frames(inst, CFG["SCAN_DEPTH"])
            if fr is None:
                return inst, None
            # Historical rows use the 1H candle availability/close timestamp.
            # Store the same timestamp for live features instead of scan wall-clock time.
            feature_ts = float(pd.Timestamp(fr[BASE]["avail"].iloc[-1]).timestamp())
            return inst, (live_vector(fr, btc), feature_ts)
        except Exception as e:
            log.warning("%s: %s", inst, e); return inst, None
    with ThreadPoolExecutor(4) as ex:
        res = dict(ex.map(one, uni.index))
    valid = {k: v for k, v in res.items() if v is not None}
    X_raw = pd.DataFrame({k: v[0] for k, v in valid.items()}).T
    feature_ts = {k: v[1] for k, v in valid.items()}
    coverage = feature_coverage_mask(X_raw.to_numpy(dtype=np.float64), 0.80) if not X_raw.empty else np.zeros(0, dtype=bool)
    X = X_raw.loc[coverage].copy()
    failed_frames = len(uni) - len(valid)
    rows_before_dropna = len(X_raw)
    rows_after_dropna = len(X)
    log.info("scan data diagnostics: universe=%d frame_failures=%d frame_success=%d feature_rows=%d "
             "usable_feature_rows=%d dropped_low_coverage=%d feature_columns=%d",
             len(uni), failed_frames, len(valid), rows_before_dropna, rows_after_dropna,
             rows_before_dropna - rows_after_dropna, X_raw.shape[1] if not X_raw.empty else 0)
    if X.empty:
        if not X_raw.empty:
            nan_counts = X_raw.isna().sum().sort_values(ascending=False)
            log.warning("no data: no feature rows reached 80%% coverage; top_unknown_features=%s",
                        rows_before_dropna,
                        json.dumps({str(k): int(v) for k, v in nan_counts.head(12).items()}, ensure_ascii=False))
        else:
            log.warning("no data: all %d/%d coins failed candle/frame retrieval or feature generation",
                        failed_frames, len(uni))
        return None
    if callable(FEATURE_DRIFT_HOOK):
        try:
            drift = FEATURE_DRIFT_HOOK(brain, X)
            if drift.get("status") == "ok":
                log.info("feature drift: %.1f%% of features shifted >=2 training std; %d/%d unstable scales",
                         drift["drift_fraction"] * 100, drift["unstable_scale_feature_count"],
                         drift["feature_count"])
        except Exception as exc:
            log.warning("feature drift monitoring skipped: %s", exc)
    current_regimes = pd.Series({i: classify_market_regime(X.loc[i]) for i in X.index})
    score = rank_universe(X, uni["vol"], paper=paper, regimes=current_regimes,
                           spread24=uni["spread"] if "spread" in uni.columns else None)
    for regime, info in score.attrs.get("adaptive_diagnostics", {}).items():
        log.info("adaptive ranking regime=%s samples=%d mode=%s weights=%s", regime,
                 info["samples"], "LEARNED" if info["adaptive"] else "BASELINE",
                 json.dumps(info["weights"], sort_keys=True))
    R = pd.DataFrame({"ranking_score": score, "score": score})
    R.attrs["adaptive_diagnostics"] = score.attrs.get("adaptive_diagnostics", {})
    R["rank"] = np.arange(1, len(R) + 1)
    R["ai_probability"] = np.nan
    R["prob"] = np.nan
    R["prediction_source"] = "RANKING_EXPLORATION"
    entry_quality = {i: evaluate_entry_quality(X.loc[i]) for i in R.index}
    R["entry_score"] = [entry_quality[i][0] for i in R.index]
    R["stretched"] = [entry_quality[i][1] for i in R.index]
    R["regime"] = [current_regimes.loc[i] for i in R.index]
    ai_prediction_error = None
    learning_mode = not brain.ready
    if learning_mode:
        log.warning("AI model unavailable: ranking-only exploration; AI probability remains NULL")
    top = list(R.index[:CFG["TOP_AI"]])
    if brain.ready:
        try:
            predictions = np.asarray(brain.predict(X.loc[top, FEATURE_COLS].values), dtype=float)
            if len(predictions) != len(top) or not np.isfinite(predictions).all():
                raise ValueError("model returned invalid probability vector")
            if callable(PRIMARY_PREDICTION_HOOK):
                predictions = np.asarray(
                    PRIMARY_PREDICTION_HOOK(brain, X.loc[top], predictions), dtype=float)
                if len(predictions) != len(top) or not np.isfinite(predictions).all():
                    raise ValueError("primary news-fusion hook returned invalid probabilities")
            R.loc[top, "ai_probability"] = predictions
            R.loc[top, "prob"] = predictions
            R.loc[top, "prediction_source"] = "VALIDATED_AI"
        except Exception as e:
            ai_prediction_error = str(e)
            log.exception("AI prediction invalid; all new entries will be blocked")
    # Ranking-only exploration intentionally leaves AI probability empty.
    R.to_csv(os.path.join(CFG["DIR"], "ranking.csv"))
    log.info("top 5:\n%s", R.head(5))
    if not trade:
        return R
    monitor_tracked_signals(paper)
    if paper._get("paused") == "1":
        return R
    paper.log_signals([dict(inst=i, price=float(uni.loc[i, "last"]), atr=float(X.loc[i, "1H_atr_pct"]),
                            feats=X.loc[i, FEATURE_COLS].values, prob=R.loc[i, "prob"], rank=int(R.loc[i, "rank"]),
                            prob_source=str(R.loc[i, "prediction_source"]), feature_ts=feature_ts[i])
                       for i in R.index[:CFG["LOG_TOP"]]])
    if ai_prediction_error is not None:
        log.error("AI prediction failed (%s); entries blocked for this cycle", ai_prediction_error)
        tg("⚠️ تعذر حساب احتمال النموذج؛ تم منع الدخول في هذه الدورة.")
        return R
    if brain.ready and not np.isfinite(R["ai_probability"].dropna().to_numpy(dtype=float)).all():
        log.error("validated AI returned invalid probabilities; entries blocked")
        tg("⚠️ احتمالات النموذج غير صالحة؛ تم منع الدخول في هذه الدورة.")
        return R
    prob_min = effective_probability_threshold(brain)
    # Widen the ranked candidate pool without relaxing any entry-quality,
    # validated-AI, correlation, duplicate or portfolio-risk checks.
    max_signal_rank = max(10, int(CFG.get("MAX_SIGNAL_RANK", 10)))
    eligible_rows = []
    correlation_cache = {}
    # Explicit rejection counters make a no-entry scan diagnosable from Actions logs.
    reject = {"stretched": 0, "entry_quality": 0, "ai_probability": 0, "correlation": 0}
    candidate_pool = R[R["rank"] <= max_signal_rank]
    log.info("entry diagnostics: ranked=%d candidate_pool=%d model_ready=%s paused=%s prob_min=%.3f",
             len(R), len(candidate_pool), bool(brain.ready), paper._get("paused") == "1", float(prob_min))
    for inst, rr in R.iterrows():
        if int(rr["rank"]) > max_signal_rank:
            continue
        threshold, regime_stats = regime_entry_threshold(paper, rr["regime"])
        # Unknown optional features can be learned, but execution still requires
        # every field used by entry-quality and ATR risk calculations.
        required_entry_features = (
            "1H_rsi14", "1H_atr_pct", "1H_ema_dist21", "15m_ret20",
            "1H_lower_wick", "1H_body_ratio", "1H_adx14",
        )
        if not np.isfinite(X.loc[inst, list(required_entry_features)].to_numpy(dtype=float)).all():
            reject["entry_quality"] += 1
            continue
        if bool(rr["stretched"]):
            reject["stretched"] += 1
            continue
        if float(rr["entry_score"]) < threshold:
            reject["entry_quality"] += 1
            continue
        if brain.ready and (pd.isna(rr["ai_probability"]) or float(rr["ai_probability"]) < float(prob_min)):
            reject["ai_probability"] += 1
            continue
        if callable(PORTFOLIO_CORRELATION_HOOK):
            try:
                correlation = PORTFOLIO_CORRELATION_HOOK(paper, inst, correlation_cache)
                if not correlation.get("allowed", True):
                    reject["correlation"] += 1
                    log.info("portfolio diversification: skipped %s; max_abs_corr=%.3f threshold=%.3f",
                             inst, float(correlation.get("max_abs_correlation") or 0.0),
                             float(correlation.get("threshold", CFG["MAX_OPEN_CORR"])))
                    continue
            except Exception as exc:
                # Missing market data must not silently change signal eligibility.
                log.warning("portfolio correlation check failed open for %s: %s", inst, exc)
        eligible_rows.append(inst)
    log.info("entry rejection summary: candidate_pool=%d eligible=%d rejected_stretched=%d "
             "rejected_entry_quality=%d rejected_ai_probability=%d rejected_correlation=%d",
             len(candidate_pool), len(eligible_rows), reject["stretched"], reject["entry_quality"],
             reject["ai_probability"], reject["correlation"])
    sort_columns = ["ai_probability", "ranking_score"] if brain.ready else ["ranking_score"]
    eligible = R.loc[eligible_rows].sort_values(sort_columns, ascending=False, na_position="last")
    created = []
    for inst, rr in eligible.iterrows():
        # A paper long is priced at the executable ask, not the ticker's last trade.
        # BT_COST remains the conservative all-in round-trip fee/slippage assumption.
        entry_quote = float(uni.loc[inst, "ask"])
        if not np.isfinite(entry_quote) or entry_quote <= 0:
            log.warning("paper entry skipped for %s: invalid ask quote", inst)
            continue
        sig = paper.create_tracked_signal(dict(
            inst=inst, price=entry_quote, atr=float(X.loc[inst, "1H_atr_pct"]),
            feats=X.loc[inst, FEATURE_COLS].values,
            prob=(float(rr["ai_probability"]) if brain.ready else None),
            prob_source=("validated_ai" if brain.ready else "ranking_exploration"), rank=int(rr["rank"]),
            feature_ts=feature_ts[inst], regime=str(rr["regime"]),
            entry_score=float(rr["entry_score"])))
        if sig:
            created.append(sig)
            probability_line = (f"احتمال AI المعتمد: {float(sig['prob'])*100:.1f}%"
                                if brain.ready and sig.get("prob") is not None
                                else "المصدر: ترتيب استكشافي فقط — لا يوجد احتمال AI")
            tg(f"🟢 إشارة تداول ورقي #{sig['id']}\n"
               f"العملة: {inst} | الترتيب: {sig['rank']}/50 | نقاط الترتيب: {float(rr['ranking_score']):+.2f}\n"
               f"{probability_line}\n"
               f"الدخول الافتراضي: {sig['entry']:.8g}\n"
               f"وقف الخسارة: {sig['sl']:.8g} | الهدف: {sig['tp']:.8g}\n"
               f"جودة الدخول: {sig['entry_score']:.0f}/100 | حالة السوق: {sig['regime']}\n"
               f"وقف متحرك يتفعّل بعد +1.5 ATR؛ متابعة حتى TP/SL/Trailing أو انتهاء {CFG['BARRIER_H']} ساعة. لا توجد أوامر حقيقية.")
    log.info("paper signal creation: candidate_pool=%d eligible=%d created=%d "
             "already-open coins skipped; rank_limit=%d; threshold=%.3f",
             len(candidate_pool), len(eligible), len(created), max_signal_rank, float(prob_min))
    return R

def bootstrap_training_due(paper, now=None):
    """Retry model bootstrap when new outcomes arrive, or periodically if durable history is trainable."""
    now = time.time() if now is None else float(now)
    last = float(paper._get("bootstrap_train_attempt_ts") or 0)
    if last <= 0:
        return True
    new_labels = paper.count("signals", "labeled=1") - int(paper._get("bootstrap_attempt_labels") or 0)

    # The GitHub Actions runner is ephemeral: train.npz is the durable historical
    # cache restored from state-data. If it already contains enough usable rows,
    # don't leave a missing model idle for a full day while waiting for new labels.
    # GitHub Actions runs hourly. When the model is missing, retry from the
    # durable cache after one workflow interval instead of waiting six hours.
    # This only schedules another validation attempt; it never bypasses promotion gates.
    cached_rows = 0
    try:
        cached = load_trainset()
        if cached is not None:
            X = np.asarray(cached["X"], dtype=np.float64)
            y = np.asarray(cached["y"], dtype=np.float64)
            ts = np.asarray(cached["ts"], dtype=np.float64)
            valid = (feature_coverage_mask(X, 0.80) & np.isfinite(y)
                     & np.isfinite(ts) & np.isin(y, [0, 1]))
            cached_rows = int(valid.sum())
    except Exception as exc:
        log.warning("bootstrap cache check failed: %s", exc)

    cache_retry_due = cached_rows >= 1200 and now - last >= 3600
    if cached_rows:
        log.info("bootstrap memory cache: usable_rows=%d; accelerated_retry=%s",
                 cached_rows, cache_retry_due)
    return (now - last >= 86400
            or new_labels >= int(CFG["RETRAIN_SIGNALS"])
            or cache_retry_due)


def mark_bootstrap_training_attempt(paper, now=None):
    """Persist the attempt and data counters even if validation rejects the candidate."""
    now = time.time() if now is None else float(now)
    paper._set("bootstrap_train_attempt_ts", now)
    paper._set("bootstrap_attempt_labels", paper.count("signals", "labeled=1"))


def run_cycle_backtest(universe, paper=None, folds=5):
    """Run the walk-forward research backtest as part of each analysis cycle."""
    try:
        ds = load_trainset()
        if ds is None:
            ds = build_trainset(universe)
        rep = walk_forward(ds, folds=folds)
        txt = backtest_report(rep)
        path = os.path.join(CFG["DIR"], "backtest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        log.info("cycle backtest completed: %s", path)
        tg("🔄 Backtest داخل دورة التحليل\n" + txt)
        return rep
    except Exception as e:
        log.exception("cycle backtest failed: %s", e)
        tg(f"⚠️ تعذر إكمال Backtest في هذه الدورة: {e}")
        return None

def daily_training_due(paper, now=None):
    """Allow at most one AI training attempt per rolling 24 hours across workflow runs."""
    now = time.time() if now is None else float(now)
    last = float(paper._get("daily_ai_training_attempt_ts") or 0)
    return last <= 0 or now - last >= 86400


def mark_daily_training_attempt(paper, now=None):
    """Persist the training-attempt time before training, including rejected/failed attempts."""
    now = time.time() if now is None else float(now)
    paper._set("daily_ai_training_attempt_ts", now)


def daily_backtest_due(paper, now=None):
    """Persist a rolling 24-hour backtest guard so reruns cannot repeat the heavy job."""
    now = time.time() if now is None else float(now)
    last = float(paper._get("daily_backtest_attempt_ts") or 0)
    return last <= 0 or now - last >= 86400


def mark_daily_backtest_attempt(paper, now=None):
    now = time.time() if now is None else float(now)
    paper._set("daily_backtest_attempt_ts", now)


def once():
    """One full cycle for schedulers (GitHub Actions)."""
    brain, paper = Brain(), Paper()
    uni = get_universe()
    if uni.empty:
        raise RuntimeError("OKX returned no tradable pairs (blocked? try OKX_BASE=https://aws.okx.com)")
    poll_commands(paper, brain)
    monitor_tracked_signals(paper)
    label_signals(paper)
    status = enforce_closed_outcome_gate(brain, paper)
    if status["ready"] and need_retrain(paper, brain):
        try:
            retrain(brain, paper, uni, "تعلّم من الإشارات المغلقة")
        except Exception as e:
            log.exception("closed-signal retraining failed: %s", e)
            tg(f"⚠️ فشل تحديث التعلم من الإشارات المغلقة: {e}")
        finally:
            paper._set("last_learning_attempt_closed_count", status["total"])
            paper._set("last_learning_attempt_schema_hash", feature_schema_fingerprint())
    elif not status["ready"]:
        log.info("AI memory learning active; supervised model waiting for verified samples=%d/%d (wins=%d losses=%d)",
                 status["total"], MIN_CLOSED_MODEL_SAMPLES, status["wins"], status["losses"])
    cycle = int(paper._get("cycle") or 0)
    scan(brain, paper, uni, cycle)
    # Keep the full analysis pipeline in this same cycle: scan and walk-forward backtest.
    run_cycle_backtest(uni, paper)
    paper._set("cycle", cycle + 1)
    status = learning_gate_status(paper)
    if status["ready"] and need_retrain(paper, brain):
        try:
            retrain(brain, paper, uni, "تحديث بعد إشارة مغلقة")
        except Exception as e:
            log.exception("post-closure learning failed: %s", e)
        finally:
            paper._set("last_learning_attempt_closed_count", status["total"])
            paper._set("last_learning_attempt_schema_hash", feature_schema_fingerprint())
    # Statistics are available on demand via /stats; no wallet or exchange orders.

def telegram_loop(paper, brain, stop):
    while not stop.is_set():
        try:
            poll_commands(paper, brain, timeout=25)
        except Exception as e:
            log.warning("telegram loop: %s", e)
        stop.wait(2)

def run():
    """Endless loop (VPS / Docker)."""
    brain, paper = Brain(), Paper()
    stop = threading.Event()
    uni = get_universe()
    if uni.empty:
        raise RuntimeError("OKX returned no tradable pairs (OKX unreachable?)")
    threading.Thread(target=telegram_loop, args=(paper, brain, stop), daemon=True).start()
    tg("🚀 Zied Quant بدأ العمل: يستفيد من الذاكرة والنتائج المغلقة المتاحة؛ لا تُختلق بيانات ناقصة، ويُعتمد النموذج بعد تحقق عينة كافية.")
    cycle = 0
    try:
        while True:
            t0 = time.time()
            try:
                monitor_tracked_signals(paper)
                label_signals(paper)
                status = enforce_closed_outcome_gate(brain, paper)
                if status["ready"] and need_retrain(paper, brain):
                    try:
                        retrain(brain, paper, None, "تعلّم من الإشارات المغلقة")
                    except Exception as e:
                        log.exception("closed-signal retraining failed: %s", e)
                        tg(f"⚠️ فشل تحديث التعلم: {e}")
                    finally:
                        paper._set("last_learning_attempt_closed_count", status["total"])
                    paper._set("last_learning_attempt_schema_hash", feature_schema_fingerprint())
                elif not status["ready"]:
                    log.info("AI memory learning active; supervised model waiting: samples=%d/%d wins=%d losses=%d",
                             status["total"], MIN_CLOSED_MODEL_SAMPLES, status["wins"], status["losses"])
                scan(brain, paper, None, cycle)
                run_cycle_backtest(uni, paper)
                cycle += 1
            except Exception as e:
                log.exception("scan failed: %s", e)
                tg(f"⚠️ خطأ في المسح: {e}")
            status = learning_gate_status(paper)
            if status["ready"] and need_retrain(paper, brain):
                try:
                    retrain(brain, paper, None, "تحديث بعد إشارة مغلقة")
                except Exception as e:
                    log.exception("post-closure learning failed: %s", e)
                finally:
                    paper._set("last_learning_attempt_closed_count", status["total"])
                paper._set("last_learning_attempt_schema_hash", feature_schema_fingerprint())
            time.sleep(max(5, CFG["SCAN_EVERY_SEC"] - (time.time() - t0)))
    except KeyboardInterrupt:
        stop.set()
        tg("⏹ Zied Quant توقف")

def main(argv):
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd in ("train", "refresh", "backtest", "scan", "once", "run") and torch is None:
        try:
            import sklearn  # noqa: F401
            log.warning("torch missing: the 20-layer net is disabled, using gradient boosting only")
        except ImportError:
            sys.exit("pip install torch scikit-learn lightgbm  (at least one model library is required)")
    if cmd in ("train", "refresh"):
        b, p = Brain(), Paper()
        enforce_closed_outcome_gate(b, p)
        print(retrain(b, p, get_universe(), "تدريب يدوي", refresh=(cmd == "refresh")))
    elif cmd == "backtest":
        # Research backtest remains available, but its historical candles are never used to train/promote the live model.
        p = Paper(); folds = int(argv[2]) if len(argv) > 2 else 5
        ds = build_trainset(get_universe())
        rep = walk_forward(ds, folds=folds)
        txt = backtest_report(rep); print(txt)
        json.dump(rep, open(os.path.join(CFG["DIR"], "backtest.json"), "w"), indent=2)
        tg(txt)
    elif cmd == "scan":
        b, p = Brain(), Paper()
        enforce_closed_outcome_gate(b, p)
        print(scan(b, p, trade=False).head(30))
    elif cmd == "once":
        once()
    elif cmd == "label":
        print(label_signals(Paper()))
    elif cmd == "stats":
        print(stats_msg(Paper()))
    else:
        run()

# Install the adaptive paper-exit selector after all core classes/functions exist.
import exit_evolution
exit_evolution.install(sys.modules[__name__])

if __name__ == "__main__":
    main(sys.argv)
