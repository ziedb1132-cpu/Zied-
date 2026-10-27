#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zied Quant v2
=============
OKX Spot (USDT, no stablecoins)
  -> 150 features = 5 timeframes x 28 technical + 10 market-context features
  -> cross-sectional ranking (best -> worst)
  -> models: 20-layer deep net + gradient boosting + ensemble (best chosen on a time-split validation)
  -> BUY signal -> paper trading with risk management and realistic costs
  -> every signal is logged and labelled later with triple-barrier outcomes (self-learning)
  -> walk-forward backtest, Telegram alerts + Telegram commands.

PAPER ONLY. No real order is ever sent. Public OKX endpoints only (no API key).

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
import os, sys, time, json, math, pickle, sqlite3, logging, threading, functools
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
    SCAN_DEPTH=300,                    # candles per TF for the live scan
    HIST_DAYS=30,                      # history used for the bootstrap dataset
    TRAIN_TOP=40,                      # coins (by volume) in the bootstrap dataset
    MIN_VOL_USDT=300_000,              # min 24h quote volume to be in the universe
    # labels = triple barrier on 1H candles (same SL/TP as paper trading)
    BARRIER_H=24,                      # max holding, in 1H bars
    SL_ATR=1.5, TP_ATR=3.0,
    BT_COST=0.0035,                    # assumed round-trip cost (fees+slippage+spread) for labels/backtest
    # scanning / signals
    TOP_AI=30, LOG_TOP=20,
    SCAN_EVERY_SEC=900, MONITOR_EVERY_SEC=30, RANK_MSG_EVERY=4,
    BTC_FILTER=-0.04,                  # no new entries if BTC 4H dist to EMA50 < this
    PROB_MIN=None,                     # None = auto threshold from validation
    # paper trading + risk
    START_BALANCE=1000.0,
    POS_FRAC=0.15,                     # max fraction of equity per trade
    RISK_PER_TRADE=0.005,              # equity risked per trade -> smaller size when volatility is high
    MAX_OPEN=5,
    MAX_PORTFOLIO_RISK=0.02,           # aggregate stop-loss risk across open + pending positions
    MAX_PORTFOLIO_EXPOSURE=0.75,       # maximum portfolio notional exposure / equity
    LIMIT_TTL_H=6,                     # cancel unfilled paper LIMIT orders after six hours
    CORR_MAX=0.85,                     # skip a coin too correlated with an open position
    DAILY_LOSS=0.03,                   # stop new entries after -3% realised today (UTC)
    LOSS_STREAK=4, COOLDOWN_H=6,       # kill-switch: N losses in a row -> pause
    MAX_HOLD_H=24,
    FEE=0.001, SLIP=0.0005, IMPACT_K=0.1,
    # self-learning
    RETRAIN_EVERY=20,                  # closed paper trades
    RETRAIN_SIGNALS=300,               # newly labelled logged signals
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
DATA_VER = 2

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
    for i in range(tries):
        with _thr_lock:                               # global throttle
            dt = time.time() - _last_call[0]
            if dt < gap:
                time.sleep(gap - dt)
            _last_call[0] = time.time()
        try:
            j = requests.get(CFG["OKX"] + path, params=params, timeout=15).json()
            if j.get("code") == "0":
                return j["data"]
            if j.get("code") == "50011":              # rate limit
                time.sleep(1.5 * (i + 1)); continue
            log.warning("OKX %s -> %s %s", path, j.get("code"), j.get("msg"))
            return []
        except Exception:
            time.sleep(1 + i)
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
        if vol < CFG["MIN_VOL_USDT"] or last <= 0 or ask <= 0 or bid < 0:
            continue
        spread = (ask - bid) / last if bid > 0 else 0.002
        if not np.isfinite(spread) or spread < 0:
            continue
        rows.append(dict(inst=inst, vol=vol, last=last, ask=ask, spread=spread))
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

def get_candles(inst, bar, depth):
    """Closed candles, oldest first. depth > 1400 switches to the history endpoint."""
    hist = depth > 1400
    path = "/api/v5/market/history-candles" if hist else "/api/v5/market/candles"
    lim, gap = (100, 0.11) if hist else (300, 0.07)
    out, after = [], None
    while len(out) < depth:
        p = {"instId": inst, "bar": bar, "limit": lim}
        if after:
            p["after"] = after
        r = okx_get(path, p, gap=gap)
        if not r:
            break
        out += r
        after = r[-1][0]
        if len(r) < lim:
            break
    if not out:
        return None
    out = [(list(r) + ["1"])[:9] for r in out]
    df = pd.DataFrame(out, columns=["ts", "o", "h", "l", "c", "vol", "volccy", "volq", "confirm"])
    df = df[df["confirm"] == "1"].copy()                  # closed candles only
    for k in ("o", "h", "l", "c", "vol"):
        df[k] = df[k].astype(float)
    df["ts"] = df["ts"].astype("int64")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["avail"] = pd.to_datetime(df["ts"] + CFG["TF_SEC"][bar] * 1000, unit="ms")
    return df.iloc[-depth:].reset_index(drop=True)

def hist_depth():
    d = CFG["HIST_DAYS"]
    return {"5m": d * 288 + 100, "15m": d * 96 + 100, "1H": d * 24 + 400,
            "4H": d * 6 + 300, "1D": d + 120}

def fetch_frames(inst, depth):
    fr = {}
    for tf in TFS:
        d = get_candles(inst, tf, depth if isinstance(depth, int) else depth[tf])
        if d is None or len(d) < 60:
            return None
        fr[tf] = d
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
         "upper_wick", "lower_wick"]
CTX_NAMES = ["beta48", "corr48", "rel_ret24", "btc_ret24", "btc_vol24", "btc_regime",
             "hour_sin", "hour_cos", "dow_sin", "log_dvol24"]
TF_COLS = [f"{tf}_{n}" for tf in TFS for n in NAMES]
CTX_COLS = [f"ctx_{n}" for n in CTX_NAMES]
FEATURE_COLS = TF_COLS + CTX_COLS
N_FEATURES = len(FEATURE_COLS)
assert N_FEATURES == 150

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
    F = {}
    for tf in TFS:
        x = tf_features(frames[tf]); x.columns = [f"{tf}_{n}" for n in NAMES]; F[tf] = x
    return F

def live_vector(frames, btc):
    """Latest closed-candle features of every timeframe + context -> 150 values."""
    F = all_tf_features(frames)
    parts = [F[tf].iloc[-1] for tf in TFS] + [ctx_features(frames[BASE], btc).iloc[-1]]
    return pd.concat(parts)[FEATURE_COLS]

def aligned_matrix(frames, btc):
    """Historical rows on BASE_TF; other TFs use the last candle already closed (no look-ahead)."""
    F = all_tf_features(frames)
    idx = F[BASE].index
    parts = [F[BASE]] + [F[tf].reindex(idx, method="ffill") for tf in TFS if tf != BASE]
    parts.append(ctx_features(frames[BASE], btc))
    return pd.concat(parts, axis=1)[FEATURE_COLS]

# =====================================================================
# TRIPLE-BARRIER LABELS (same SL/TP logic as paper trading)
# =====================================================================
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
    sl = entry * (1 - CFG["SL_ATR"] * a); tp = entry * (1 + CFG["TP_ATR"] * a)
    hit_sl, hit_tp = Lw <= sl[:, None], Hw >= tp[:, None]
    i_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), H)
    i_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), H)
    sl_first = (i_sl < H) & (i_sl <= i_tp)
    tp_first = (i_tp < H) & (i_tp < i_sl)
    gross = np.where(sl_first, -CFG["SL_ATR"] * a,
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
            D = M.assign(_ret=ret, _hold=hold).dropna()
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
        if persist:
            self.load()

    @property
    def ready(self):
        return self.mode is not None and (self.nn is not None or self.gbm is not None)

    @property
    def thr(self):
        return self.thrs.get(self.mode, 0.5)

    def _norm(self, X):
        return np.clip((np.asarray(X, dtype=np.float64) - self.mu) / self.sd, -5, 5).astype(np.float32)

    def describe(self):
        if not self.ready:
            return "لا يوجد نموذج بعد"
        return (f"النمط: {self.mode} | AUC={self.auc:.3f} | العتبة={self.thr:.3f}\n"
                f"شبكة 20 طبقة: {'نعم' if self.nn is not None else 'لا'} | "
                f"GBM: {'نعم' if self.gbm is not None else 'لا'}\n"
                f"عيّنات التدريب: {self.info.get('n', '?')}")

    # ---------- persistence ----------
    def save(self, n_in):
        """Persist only the model selected for inference; write metadata last as commit marker."""
        need_nn = self.mode in ("nn", "ens")
        need_gbm = self.mode in ("gbm", "ens")
        if self.mode not in ("nn", "gbm", "ens"):
            raise RuntimeError(f"cannot save unknown model mode: {self.mode!r}")
        if need_nn and (self.nn is None or torch is None):
            raise RuntimeError("selected AI mode requires a neural network, but it is unavailable")
        if need_gbm and self.gbm is None:
            raise RuntimeError("selected AI mode requires GBM, but it is unavailable")
        if self.mu is None or self.sd is None or len(self.mu) != n_in or len(self.sd) != n_in:
            raise RuntimeError("refusing to save model with invalid normalization vectors")

        os.makedirs(CFG["DIR"], exist_ok=True)
        model_tmp, gbm_tmp, meta_tmp = CFG["MODEL"] + ".tmp", CFG["GBM"] + ".tmp", CFG["META"] + ".tmp"
        for path in (model_tmp, gbm_tmp, meta_tmp):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        try:
            if need_nn:
                torch.save({"state": self.nn.state_dict(), "n_in": n_in}, model_tmp)
            if need_gbm:
                with open(gbm_tmp, "wb") as fh:
                    pickle.dump(self.gbm, fh, protocol=pickle.HIGHEST_PROTOCOL)
            meta = dict(mu=np.asarray(self.mu, dtype=float).tolist(),
                        sd=np.asarray(self.sd, dtype=float).tolist(), thrs=self.thrs, mode=self.mode,
                        auc=self.auc, has_nn=need_nn, has_gbm=need_gbm,
                        n_in=n_in, info=self.info)
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            # Metadata is replaced last, so incomplete temporary files are never treated as a model.
            if need_nn:
                os.replace(model_tmp, CFG["MODEL"])
            if need_gbm:
                os.replace(gbm_tmp, CFG["GBM"])
            os.replace(meta_tmp, CFG["META"])
            # Remove obsolete components only after the new metadata is safely committed.
            if not need_nn:
                try: os.remove(CFG["MODEL"])
                except FileNotFoundError: pass
            if not need_gbm:
                try: os.remove(CFG["GBM"])
                except FileNotFoundError: pass
            log.info("model artifacts saved: mode=%s nn=%s gbm=%s", self.mode, need_nn, need_gbm)
        finally:
            for path in (model_tmp, gbm_tmp, meta_tmp):
                try: os.remove(path)
                except FileNotFoundError: pass

    def load(self):
        if not os.path.exists(CFG["META"]):
            # Without metadata there is no trustworthy normalization, selected mode, or validation score.
            # These model files are stale/orphaned and must not survive another state save.
            for path in (CFG["MODEL"], CFG["GBM"]):
                if os.path.exists(path):
                    log.warning("removing orphan model artifact without meta.json: %s", path)
                    try:
                        os.remove(path)
                    except OSError as exc:
                        log.error("could not remove orphan model artifact %s: %s", path, exc)
            return
        try:
            meta = json.load(open(CFG["META"]))
            if meta["n_in"] != N_FEATURES:
                log.info("saved model has %s features, expected %s -> ignored", meta["n_in"], N_FEATURES)
                return
            self.mu, self.sd = np.array(meta["mu"]), np.array(meta["sd"])
            self.thrs, self.auc, self.info = meta["thrs"], meta["auc"], meta.get("info", {})
            if meta["has_nn"] and torch is not None and os.path.exists(CFG["MODEL"]):
                ck = torch.load(CFG["MODEL"], weights_only=True)
                self.nn = ZiedNet(ck["n_in"]); self.nn.load_state_dict(ck["state"]); self.nn.eval()
            if meta["has_gbm"] and os.path.exists(CFG["GBM"]):
                with open(CFG["GBM"], "rb") as fh:
                    self.gbm = pickle.load(fh)
            mode = meta["mode"]
            if mode in ("nn", "ens") and self.nn is None:        # torch missing here -> degrade gracefully
                mode = "gbm" if self.gbm is not None else None
            if mode == "ens" and self.gbm is None:
                mode = "nn"
            if mode == "gbm" and self.gbm is None:
                mode = "nn" if self.nn is not None else None
            self.mode = mode
            if self.ready:
                log.info("model loaded: mode=%s auc=%.3f thr=%.3f", self.mode, self.auc, self.thr)
        except Exception as e:
            log.warning("could not load model: %s", e)
            self.nn = self.gbm = self.mode = None

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
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"AI input shape invalid: {X.shape}; expected (*, {N_FEATURES})")
        if not np.isfinite(X).all():
            raise ValueError("AI input contains NaN or infinite values")
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
        """Time-ordered split with an embargo; early stopping uses the tail of the training part only,
        so the validation block (last 20%) stays untouched for model choice and the threshold."""
        o = np.argsort(ts, kind="stable")
        X = np.asarray(X, dtype=np.float64)[o]; y = np.asarray(y, dtype=np.float32)[o]
        ts = np.asarray(ts, dtype=np.float64)[o]
        ret = None if ret is None else np.asarray(ret, dtype=np.float64)[o]
        n = len(X)
        if n < 600:
            raise ValueError(f"not enough samples ({n})")
        nv = max(int(n * 0.2), 200)
        ntr = int(np.searchsorted(ts, ts[n - nv] - CFG["BARRIER_H"] * 3600))
        if ntr < 400:
            raise ValueError("training part too small after embargo")
        # Fit preprocessing only on the actual model-fit segment; do not use early-stop
        # validation rows to estimate feature means/standard deviations.
        nf = int(ntr * 0.85)
        mu = X[:nf].mean(0); sd = X[:nf].std(0) + 1e-8
        tmp = Brain.__new__(Brain); tmp.mu, tmp.sd = mu, sd
        Xtr, Xv = tmp._norm(X[:ntr]), tmp._norm(X[n - nv:])
        ytr, yv = y[:ntr], y[n - nv:]
        models, pv = {}, {}
        g = GBM()
        if g.fit(Xtr[:nf], ytr[:nf], Xtr[nf:], ytr[nf:]):
            models["gbm"] = g; pv["gbm"] = g.predict(Xv)
        if torch is not None:
            net = _train_net(Xtr[:nf], ytr[:nf], Xtr[nf:], ytr[nf:], epochs)
            models["nn"] = net; pv["nn"] = _nn_predict(net, Xv)
        if not pv:
            raise RuntimeError("no model available: install torch and/or lightgbm / scikit-learn")
        if len(pv) == 2:
            pv["ens"] = (pv["nn"] + pv["gbm"]) / 2
        aucs = {k: auc_score(yv, p) for k, p in pv.items()}
        mode = max(aucs, key=aucs.get)
        thrs = {k: float(np.quantile(p, 0.90)) for k, p in pv.items()}
        info = dict(n=int(n), n_val=int(nv), aucs=aucs, mode=mode, auc=aucs[mode], thr=thrs[mode],
                    win_rate=float(y.mean()), gbm_kind=g.kind if "gbm" in models else None)
        if ret is not None:
            rv = ret[n - nv:]; m = pv[mode] >= thrs[mode]
            info["val_all_net"] = float(np.nanmean(rv))
            info["val_top_net"] = float(np.nanmean(rv[m])) if m.any() else None
        accepted = force or (aucs[mode] >= 0.52 and (self.auc is None or aucs[mode] >= self.auc - 0.03))
        info["accepted"] = bool(accepted)
        log.info("model validation: n=%d val=%d selected=%s AUC=%.4f threshold=%.4f accepted=%s previous_auc=%s",
                 n, nv, mode, aucs[mode], thrs[mode], bool(accepted),
                 "none" if self.auc is None else f"{self.auc:.4f}")
        if not accepted:
            log.warning("candidate model rejected by quality gate; no model artifacts will be written")
        if accepted:
            self.nn, self.gbm = models.get("nn"), models.get("gbm")
            self.mu, self.sd, self.thrs, self.mode, self.auc, self.info = mu, sd, thrs, mode, aucs[mode], info
            if self.persist:
                self.save(X.shape[1])
        self.last_fit = dict(info=info, pv=pv, yv=yv)
        return info

# =====================================================================
# PAPER TRADING + SIGNAL LOG + SELF-LEARNING MEMORY
# =====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, inst TEXT, open_ts REAL, feature_ts REAL, entry REAL, qty REAL,
  usdt_in REAL, sl REAL, tp REAL, prob REAL, rank INTEGER, feats TEXT,
  status TEXT DEFAULT 'OPEN', close_ts REAL, exit REAL, pnl REAL, pnl_pct REAL, reason TEXT,
  spread REAL, vol24 REAL);
CREATE TABLE IF NOT EXISTS experience(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, inst TEXT, feats TEXT, label INTEGER, pnl REAL, ret REAL);
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, inst TEXT, price REAL, atr REAL, feats TEXT,
  prob REAL, rank INTEGER, labeled INTEGER DEFAULT 0, label INTEGER, ret REAL, hold REAL);
CREATE TABLE IF NOT EXISTS pending_orders(
  id INTEGER PRIMARY KEY AUTOINCREMENT, inst TEXT, created_ts REAL, expires_ts REAL,
  limit_px REAL, qty REAL, reserved_usdt REAL, sl REAL, tp REAL, prob REAL, rank INTEGER,
  feats TEXT, spread REAL, vol24 REAL, status TEXT DEFAULT 'PENDING');
"""

def locked(f):
    @functools.wraps(f)
    def w(self, *a, **k):
        with self.lock:
            return f(self, *a, **k)
    return w

def impact(usdt, vol24):
    return CFG["IMPACT_K"] * math.sqrt(max(usdt, 0.0) / max(vol24 or 1e6, 1.0))

def _rows_to_ds(rows, label_key="label", ret_key="ret"):
    X, y, ts, ret, hold, coin = [], [], [], [], [], []
    for r in rows:
        f = json.loads(r["feats"])
        if len(f) != N_FEATURES:
            continue
        X.append(f); y.append(r[label_key]); ts.append(r["ts"])
        ret.append(r[ret_key] if r[ret_key] is not None else np.nan)
        hold.append(r["hold"] if "hold" in r.keys() and r["hold"] is not None else np.nan)
        coin.append(r["inst"])
    if not X:
        return empty_ds()
    return dict(X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.float32), ts=np.array(ts, dtype=np.float64),
                ret=np.array(ret, dtype=np.float64), hold=np.array(hold, dtype=np.float64),
                coin=np.array(coin, dtype="<U16"))

class Paper:
    def __init__(self):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(CFG["DB"], check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()
        if self._get("cash") is None:
            self._set("cash", CFG["START_BALANCE"])

    def _migrate(self):
        for table, cols in {"trades": {"spread": "REAL", "vol24": "REAL", "feature_ts": "REAL"}, "experience": {"ret": "REAL"}}.items():
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for c, t in cols.items():
                if c not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {c} {t}")
        # A LIMIT order's features are observed when the order is created, which can
        # precede its fill by up to LIMIT_TTL_H. Keep the sample timestamp at feature
        # observation time, not fill/close time, to preserve chronological evaluation.
        self.db.execute(
            "UPDATE trades SET feature_ts=COALESCE(("
            "SELECT p.created_ts FROM pending_orders p WHERE p.inst=trades.inst "
            "AND p.feats=trades.feats AND p.status='FILLED' AND p.created_ts<=trades.open_ts "
            "ORDER BY ABS(p.created_ts-trades.open_ts) ASC LIMIT 1), open_ts) "
            "WHERE feature_ts IS NULL"
        )
        self.db.execute(
            "UPDATE experience SET ts=(SELECT COALESCE(t.feature_ts,t.open_ts) FROM trades t "
            "WHERE t.inst=experience.inst AND t.feats=experience.feats "
            "AND t.close_ts=experience.ts ORDER BY t.id DESC LIMIT 1) "
            "WHERE EXISTS (SELECT 1 FROM trades t WHERE t.inst=experience.inst "
            "AND t.feats=experience.feats AND t.close_ts=experience.ts)"
        )
        self.db.commit()

    @locked
    def _get(self, k):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return None if r is None else r["v"]

    @locked
    def _set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v))); self.db.commit()

    @locked
    def count(self, table, where="1=1"):
        return self.db.execute(f"SELECT COUNT(*) c FROM {table} WHERE {where}").fetchone()["c"]

    @locked
    def cash(self):
        return float(self._get("cash"))

    @locked
    def open_trades(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM trades WHERE status='OPEN'")]

    @locked
    def pending_orders(self):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM pending_orders WHERE status='PENDING' ORDER BY created_ts")]

    @locked
    def equity(self, prices=None):
        # Reserved cash remains an asset: pending LIMIT orders must not make equity disappear.
        eq = self.cash()
        eq += sum(float(o["reserved_usdt"]) for o in self.pending_orders())
        for t in self.open_trades():
            px = (prices or {}).get(t["inst"], t["entry"]); eq += t["qty"] * px
        return eq

    @locked
    def portfolio_risk_used(self):
        """Estimated USDT at risk to stop-loss across open positions and pending LIMIT orders."""
        risk = 0.0
        for t in self.open_trades():
            if t["entry"] > 0:
                risk += float(t["usdt_in"]) * max(0.0, (t["entry"] - t["sl"]) / t["entry"])
        for o in self.pending_orders():
            if o["limit_px"] > 0:
                risk += float(o["reserved_usdt"]) * max(0.0, (o["limit_px"] - o["sl"]) / o["limit_px"])
        return risk

    # ---------- risk management ----------
    @locked
    def risk_gate(self, equity):
        """None if new entries are allowed, else the reason (string)."""
        if self._get("paused") == "1":
            return "paused"
        if time.time() < float(self._get("pause_until") or 0):
            return "cooldown"
        day0 = time.time() // 86400 * 86400
        pnl = self.db.execute("SELECT COALESCE(SUM(pnl),0) s FROM trades WHERE status='CLOSED' AND close_ts>=?",
                              (day0,)).fetchone()["s"]
        if pnl <= -CFG["DAILY_LOSS"] * equity:
            return "daily_loss"
        return None

    # ---------- trading ----------
    @locked
    def place_limit(self, inst, limit_px, atr_pct, prob, rank, feats, vol24, spread):
        """Reserve paper cash for a buy LIMIT order; create a trade only after a later candle fills it."""
        open_trades = self.open_trades()
        pending = self.pending_orders()
        if any(t["inst"] == inst for t in open_trades) or any(o["inst"] == inst for o in pending):
            return None
        if len(open_trades) + len(pending) >= CFG["MAX_OPEN"]:
            return None
        if not np.isfinite(limit_px) or limit_px <= 0:
            return None
        eq = self.equity()
        a = float(np.clip(atr_pct, 0.004, 0.06))
        usdt = min(self.cash(), CFG["POS_FRAC"] * eq,
                   CFG["RISK_PER_TRADE"] * eq / (CFG["SL_ATR"] * a))
        if usdt < 10:
            return None
        sl = float(limit_px) * (1 - CFG["SL_ATR"] * a)
        tp = float(limit_px) * (1 + CFG["TP_ATR"] * a)
        used_risk = self.portfolio_risk_used()
        new_risk = usdt * (limit_px - sl) / limit_px
        if used_risk + new_risk > CFG["MAX_PORTFOLIO_RISK"] * eq + 1e-9:
            log.info("%s skipped: portfolio stop-risk budget %.2f + %.2f > %.2f USDT",
                     inst, used_risk, new_risk, CFG["MAX_PORTFOLIO_RISK"] * eq)
            return None
        exposure = sum(float(t["usdt_in"]) for t in open_trades) + sum(float(o["reserved_usdt"]) for o in pending)
        if exposure + usdt > CFG["MAX_PORTFOLIO_EXPOSURE"] * eq + 1e-9:
            log.info("%s skipped: portfolio exposure limit", inst)
            return None
        now = time.time()
        feats_json = json.dumps([float(x) for x in feats])
        qty = usdt * (1 - CFG["FEE"]) / float(limit_px)
        cur = self.db.execute(
            "INSERT INTO pending_orders(inst,created_ts,expires_ts,limit_px,qty,reserved_usdt,sl,tp,prob,rank,feats,spread,vol24,status) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'PENDING')",
            (inst, now, now + CFG["LIMIT_TTL_H"] * 3600, float(limit_px), qty, usdt, sl, tp,
             prob, rank, feats_json, spread, vol24))
        self._set("cash", self.cash() - usdt)
        self.db.commit()
        return dict(id=cur.lastrowid, inst=inst, limit_px=float(limit_px), sl=sl, tp=tp,
                    usdt=usdt, qty=qty, expires_ts=now + CFG["LIMIT_TTL_H"] * 3600)

    @locked
    def check_limit_orders(self):
        """Fill only from a closed 5m candle that starts after order creation; otherwise expire/refund."""
        now = time.time()
        filled = expired = 0
        for o in self.pending_orders():
            d = get_candles(o["inst"], "5m", 300)
            hit = None
            if d is not None and len(d):
                # Strict start-time filter avoids filling from a candle that began before the order existed.
                path = d[(d["ts"] >= int(o["created_ts"] * 1000)) &
                         (d["ts"] < int(o["expires_ts"] * 1000))]
                for _, candle in path.iterrows():
                    if float(candle["l"]) <= float(o["limit_px"]):
                        hit = candle
                        break
            if hit is not None:
                fill_ts = float(hit["ts"]) / 1000.0 + CFG["TF_SEC"]["5m"]
                self.db.execute(
                    "INSERT INTO trades(inst,open_ts,feature_ts,entry,qty,usdt_in,sl,tp,prob,rank,feats,spread,vol24) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (o["inst"], fill_ts, o["created_ts"], o["limit_px"], o["qty"], o["reserved_usdt"],
                     o["sl"], o["tp"], o["prob"], o["rank"], o["feats"], o["spread"], o["vol24"]))
                self.db.execute("UPDATE pending_orders SET status='FILLED' WHERE id=?", (o["id"],))
                self.db.commit()
                tg(f"✅ تنفيذ LIMIT افتراضي {o['inst']}\nالسعر: {o['limit_px']:.8g}\n"
                   f"SL: {o['sl']:.8g} | TP: {o['tp']:.8g}\nالمبلغ المحجوز: {o['reserved_usdt']:.2f} USDT")
                filled += 1
            elif now >= float(o["expires_ts"]):
                self._set("cash", self.cash() + float(o["reserved_usdt"]))
                self.db.execute("UPDATE pending_orders SET status='EXPIRED' WHERE id=?", (o["id"],))
                self.db.commit()
                tg(f"⌛ انتهت صلاحية أمر LIMIT الافتراضي {o['inst']} دون تنفيذ. "
                   f"أُعيد {o['reserved_usdt']:.2f} USDT إلى النقد المتاح.")
                expired += 1
        return dict(filled=filled, expired=expired)

    @locked
    def open(self, inst, ask, atr_pct, prob, rank, feats, vol24, spread):
        if any(t["inst"] == inst for t in self.open_trades()):
            return None
        if len(self.open_trades()) >= CFG["MAX_OPEN"]:
            return None
        eq = self.equity()
        a = float(np.clip(atr_pct, 0.004, 0.06))
        usdt = min(self.cash(), CFG["POS_FRAC"] * eq, CFG["RISK_PER_TRADE"] * eq / (CFG["SL_ATR"] * a))
        if usdt < 10:
            return None
        entry = ask * (1 + CFG["SLIP"] + impact(usdt, vol24))
        qty = usdt * (1 - CFG["FEE"]) / entry
        sl, tp = entry * (1 - CFG["SL_ATR"] * a), entry * (1 + CFG["TP_ATR"] * a)
        now = time.time()
        self.db.execute(
            "INSERT INTO trades(inst,open_ts,feature_ts,entry,qty,usdt_in,sl,tp,prob,rank,feats,spread,vol24) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (inst, now, now, entry, qty, usdt, sl, tp, prob, rank,
             json.dumps([float(x) for x in feats]), spread, vol24))
        self._set("cash", self.cash() - usdt)
        return dict(inst=inst, entry=entry, sl=sl, tp=tp, usdt=usdt)

    @locked
    def _close(self, t, price, reason):
        exit_px = price * (1 - CFG["SLIP"] - (t.get("spread") or 0) / 2 - impact(t["usdt_in"], t.get("vol24")))
        proceeds = t["qty"] * exit_px * (1 - CFG["FEE"])
        pnl = proceeds - t["usdt_in"]; pct = pnl / t["usdt_in"]
        now = time.time()
        self.db.execute("UPDATE trades SET status='CLOSED',close_ts=?,exit=?,pnl=?,pnl_pct=?,reason=? WHERE id=?",
                        (now, exit_px, pnl, pct, reason, t["id"]))
        # Timestamp the sample when its features were observed, not when it filled
        # or closed. This matters for LIMIT orders whose feature vector can be hours old.
        feature_ts = t.get("feature_ts") or t["open_ts"]
        self.db.execute("INSERT INTO experience(ts,inst,feats,label,pnl,ret) VALUES(?,?,?,?,?,?)",
                        (float(feature_ts), t["inst"], t["feats"], int(pnl > 0), pnl, pct))
        self._set("cash", self.cash() + proceeds); self.db.commit()
        tg(f"{'✅' if pnl > 0 else '🔴'} إغلاق صفقة ورقية {t['inst']}\nالسبب: {reason}\n"
           f"دخول {t['entry']:.6g} ← خروج {exit_px:.6g}\nالربح: {pnl:+.2f} USDT ({pct * 100:+.2f}%)")
        self._streak_check()

    @locked
    def _streak_check(self):
        n = CFG["LOSS_STREAK"]; since = float(self._get("streak_from") or 0)
        rows = self.db.execute("SELECT pnl FROM trades WHERE status='CLOSED' AND close_ts>? "
                               "ORDER BY close_ts DESC LIMIT ?", (since, n)).fetchall()
        if len(rows) == n and all(r["pnl"] <= 0 for r in rows):
            self._set("pause_until", time.time() + CFG["COOLDOWN_H"] * 3600)
            self._set("streak_from", time.time())
            tg(f"⛔ {n} خسائر متتالية: إيقاف مؤقت للصفقات الجديدة {CFG['COOLDOWN_H']} ساعات")

    @locked
    def check(self, prices):
        """Fast mode (long-running loop): compare the latest price with SL/TP."""
        closed = 0
        for t in self.open_trades():
            px = prices.get(t["inst"])
            if px is None:
                continue
            if px <= t["sl"]:
                self._close(t, px, "وقف الخسارة"); closed += 1
            elif px >= t["tp"]:
                self._close(t, px, "جني الأرباح"); closed += 1
            elif time.time() - t["open_ts"] > CFG["MAX_HOLD_H"] * 3600:
                self._close(t, px, "انتهاء المدة"); closed += 1
        return closed

    @locked
    def check_candles(self):
        """Scheduler mode (hourly): scan 5m candles since entry for SL/TP (SL first if both)."""
        closed = 0
        trades = self.open_trades()
        prices = get_prices() if trades else {}
        for t in trades:
            px = reason = None
            d = get_candles(t["inst"], "5m", 300)
            if d is not None:
                for _, k in d[d["ts"] >= t["open_ts"] * 1000].iterrows():
                    if k["l"] <= t["sl"]:
                        px, reason = t["sl"], "وقف الخسارة"; break
                    if k["h"] >= t["tp"]:
                        px, reason = t["tp"], "جني الأرباح"; break
            if px is None and time.time() - t["open_ts"] > CFG["MAX_HOLD_H"] * 3600 and t["inst"] in prices:
                px, reason = prices[t["inst"]], "انتهاء المدة"
            if px is not None:
                self._close(t, px, reason); closed += 1
        return closed

    # ---------- signal log (self-learning) ----------
    @locked
    def log_signals(self, rows):
        now = time.time()
        self.db.executemany("INSERT INTO signals(ts,inst,price,atr,feats,prob,rank) VALUES(?,?,?,?,?,?,?)",
                            [(now, r["inst"], r["price"], r["atr"], json.dumps([float(x) for x in r["feats"]]),
                              None if r["prob"] is None or np.isnan(r["prob"]) else float(r["prob"]), r["rank"])
                             for r in rows])
        self.db.commit()

    @locked
    def pending_signals(self, cutoff_ts):
        return [dict(r) for r in self.db.execute(
            "SELECT id,ts,inst,price,atr FROM signals WHERE labeled=0 AND ts<=? ORDER BY ts", (cutoff_ts,))]

    @locked
    def set_label(self, sid, label, ret, hold):
        self.db.execute("UPDATE signals SET labeled=1,label=?,ret=?,hold=? WHERE id=?", (label, ret, hold, sid))
        self.db.commit()

    @locked
    def signal_dataset(self):
        return _rows_to_ds(self.db.execute(
            "SELECT ts,inst,feats,label,ret,hold FROM signals WHERE labeled=1 ORDER BY ts").fetchall())

    @locked
    def experience_dataset(self):
        return _rows_to_ds(self.db.execute(
            "SELECT ts,inst,feats,label,ret FROM experience ORDER BY ts").fetchall())

    @locked
    def stats(self, prices=None):
        cl = [dict(r) for r in self.db.execute("SELECT * FROM trades WHERE status='CLOSED'")]
        n = len(cl); wins = [t for t in cl if t["pnl"] > 0]; loss = [t for t in cl if t["pnl"] <= 0]
        gp, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in loss)
        eq = self.equity(prices)
        return dict(trades=n, open=len(self.open_trades()), winrate=(len(wins) / n if n else 0),
                    pnl=sum(t["pnl"] for t in cl),
                    profit_factor=(gp / gl if gl > 0 else (float("inf") if gp else 0)),
                    equity=eq, ret_pct=100 * (eq / CFG["START_BALANCE"] - 1))

# =====================================================================
# SIGNAL LABELLING (triple barrier on real later prices)
# =====================================================================
def label_signals(paper):
    H = CFG["BARRIER_H"]
    rows = paper.pending_signals(time.time() - (H + 2) * 3600)
    if not rows:
        return 0
    by = {}
    for r in rows:
        by.setdefault(r["inst"], []).append(r)
    n = 0
    for inst, rs in by.items():
        oldest = min(r["ts"] for r in rs)
        d = get_candles(inst, "1H", min(1400, int((time.time() - oldest) / 3600) + H + 5))
        if d is None:
            continue
        for r in rs:
            path = d[d["ts"] >= r["ts"] * 1000]
            if len(path) < H:
                continue
            ret, lab, hold = barrier_single(r["price"], r["atr"], path["h"].values, path["l"].values, path["c"].values)
            paper.set_label(r["id"], lab, ret, hold); n += 1
    log.info("labelled %d logged signals", n)
    return n

# =====================================================================
# RANKING (best -> worst)
# =====================================================================
def rank_universe(X, vol24):
    z = lambda s: ((s - s.median()) / (s.std() + 1e-9)).clip(-3, 3)
    trend = sum(X[f"{tf}_ema_dist21"] for tf in TFS) / len(TFS)
    overbought = (X["1H_rsi14"] - 75).clip(lower=0)
    score = (0.20 * z(X["1H_ret20"]) + 0.20 * z(X["4H_ret20"]) + 0.15 * z(X["1D_ret10"])
             + 0.10 * z(X["15m_ret20"]) + 0.10 * z(trend) + 0.08 * z(X["1H_vol_ratio"])
             + 0.05 * z(X["1H_adx14"]) + 0.05 * z(X["1H_obv_slope"])
             + 0.05 * z(np.log(vol24.reindex(X.index)))
             - 0.10 * z(X["1H_atr_pct"]) - 0.10 * z(overbought))
    return score.sort_values(ascending=False)

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
        open_ = [p for p in open_ if p[0] > t]
        if len(open_) >= CFG["MAX_OPEN"] or any(p[1] == coin[i] for p in open_):
            continue
        a = float(np.clip(atr[i], 0.004, 0.06))
        size_frac = min(CFG["POS_FRAC"], CFG["RISK_PER_TRADE"] / (CFG["SL_ATR"] * a))
        risk_frac = size_frac * CFG["SL_ATR"] * a
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
    s = paper.stats(prices if prices is not None else get_prices())
    gate = paper.risk_gate(s["equity"])
    return (f"📈 ملخص Paper Trading\nالرصيد الكلي: {s['equity']:.2f} USDT ({s['ret_pct']:+.2f}%)\n"
            f"صفقات مغلقة: {s['trades']} | مفتوحة: {s['open']}\n"
            f"نسبة النجاح: {s['winrate'] * 100:.1f}% | Profit factor: {s['profit_factor']:.2f}\n"
            f"الحالة: {'متوقف (' + gate + ')' if gate else 'نشط'}")

def positions_msg(paper):
    ts = paper.open_trades()
    if not ts:
        return "لا توجد صفقات مفتوحة."
    px = get_prices(); lines = []
    for t in ts:
        p = px.get(t["inst"], t["entry"])
        lines.append(f"{t['inst']}: {(p / t['entry'] - 1) * 100:+.2f}% | SL {t['sl']:.6g} | TP {t['tp']:.6g}")
    return "📌 الصفقات المفتوحة\n" + "\n".join(lines)

def portfolio_msg(paper):
    """Full virtual-portfolio snapshot for Telegram; paper trading only."""
    prices = get_prices()
    s = paper.stats(prices)
    cash = paper.cash()
    open_trades = paper.open_trades()
    unrealized = 0.0
    lines = [
        "💼 المحفظة الافتراضية — Zied Quant",
        "⚠️ تداول ورقي فقط؛ لا توجد أوامر حقيقية.",
        f"💵 النقد المتاح: {cash:.2f} USDT",
        f"⏳ أوامر LIMIT المعلّقة: {len(paper.pending_orders())} | المحجوز: {sum(float(o['reserved_usdt']) for o in paper.pending_orders()):.2f} USDT",
        f"🛡️ المخاطر المقدّرة عند SL: {paper.portfolio_risk_used():.2f} USDT / {CFG['MAX_PORTFOLIO_RISK'] * 100:.1f}% كحد أقصى من حقوق المحفظة",
        f"📊 قيمة المحفظة: {s['equity']:.2f} USDT ({s['ret_pct']:+.2f}%)",
        f"📈 الربح/الخسارة الكلي مقابل رأس المال الأولي: {s['equity'] - CFG['START_BALANCE']:+.2f} USDT",
        f"✅ الربح/الخسارة المحقق للصفقات المغلقة: {s['pnl']:+.2f} USDT",
        f"🔓 صفقات مفتوحة: {len(open_trades)} | مغلقة: {s['trades']}",
        f"🎯 نسبة النجاح: {s['winrate'] * 100:.1f}% | Profit Factor: {s['profit_factor']:.2f}",
        "",
        "📌 تفاصيل الصفقات المفتوحة:"
    ]
    if not open_trades:
        lines.append("لا توجد صفقات مفتوحة حالياً.")
    else:
        for t in open_trades:
            px = prices.get(t["inst"], t["entry"])
            upnl = t["qty"] * px - t["usdt_in"]
            unrealized += upnl
            pct = (px / t["entry"] - 1) * 100
            lines.append(
                f"• {t['inst']} | دخول {t['entry']:.6g} | الآن {px:.6g}\n"
                f"  عائم {upnl:+.2f} USDT ({pct:+.2f}%) | SL {t['sl']:.6g} | TP {t['tp']:.6g}"
            )
    lines.insert(7, f"🌊 الربح/الخسارة غير المحقق للصفقات المفتوحة: {unrealized:+.2f} USDT")
    # Telegram limits messages to 4096 characters; current open-position cap is five.
    return "\n".join(lines)[:3900]

def top_msg():
    p = os.path.join(CFG["DIR"], "ranking.csv")
    if not os.path.exists(p):
        return "لا يوجد تصنيف بعد."
    R = pd.read_csv(p, index_col=0).head(10)
    return "📊 أفضل 10 عملات\n" + "\n".join(
        f"{int(r['rank'])}. {i.replace('-USDT', '')} نقاط {r['score']:+.2f}"
        + (f" AI {r['prob'] * 100:.0f}%" if not pd.isna(r["prob"]) else "") for i, r in R.iterrows())

def handle_command(text, paper, brain):
    cmd = text.split()[0].split("@")[0].lower()
    if cmd in ("/stats", "/status"):
        return stats_msg(paper)
    if cmd == "/positions":
        return positions_msg(paper)
    if cmd in ("/portfolio", "/محفظتي"):
        return portfolio_msg(paper)
    if cmd == "/top":
        return top_msg()
    if cmd == "/pause":
        paper._set("paused", "1"); return "⏸ تم إيقاف الصفقات الجديدة (المتابعة مستمرة)."
    if cmd == "/resume":
        paper._set("paused", "0"); paper._set("pause_until", "0"); return "▶️ تم استئناف الصفقات الجديدة."
    if cmd == "/model":
        return "🧠 " + brain.describe()
    if cmd == "/signals":
        return (f"سجل الإشارات: {paper.count('signals')} | موسومة: {paper.count('signals', 'labeled=1')} | "
                f"صفقات مغلقة: {paper.count('experience')}")
    return ("الأوامر: /stats /positions /portfolio /top /model /signals /pause /resume")

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
def collect_training_data(paper, universe=None, refresh=False, with_trades=True):
    base = None if refresh else load_trainset()
    if base is None:
        universe = universe if universe is not None else get_universe()
        base = build_trainset(universe)
    signals = paper.signal_dataset()
    exp = paper.experience_dataset() if with_trades else empty_ds()

    def report_source(name, ds):
        n = len(ds["y"])
        if not n:
            log.info("training data source=%s n=0", name)
            return
        y = np.asarray(ds["y"], dtype=np.float64)
        ret = np.asarray(ds["ret"], dtype=np.float64)
        ts = np.asarray(ds["ts"], dtype=np.float64)
        finite_ret = np.isfinite(ret)
        mismatch = (np.mean((y[finite_ret] > 0.5) != (ret[finite_ret] > 0))
                    if finite_ret.any() else float("nan"))
        coin = np.asarray(ds["coin"]).astype(str)
        finite_ts = np.isfinite(ts)
        keys = np.char.add(np.char.add(coin[finite_ts], "|"),
                           ts[finite_ts].astype(np.int64).astype(str))
        duplicate_rows = int(len(keys) - len(np.unique(keys)))
        ts_range = (f"{int(ts[finite_ts].min())}..{int(ts[finite_ts].max())}"
                    if finite_ts.any() else "invalid")
        ret_mean = float(np.mean(ret[finite_ret])) if finite_ret.any() else float("nan")
        ret_win = float(np.mean(ret[finite_ret] > 0)) if finite_ret.any() else float("nan")
        log.info(
            "training data source=%s n=%d label_win_rate=%.3f return_win_rate=%.3f "
            "mean_net_return=%.5f ret_finite=%d/%d ts_finite=%d/%d "
            "label_return_mismatch=%.4f duplicate_coin_timestamp=%d ts_range=%s",
            name, n, float(np.mean(y > 0.5)), ret_win, ret_mean,
            int(finite_ret.sum()), n, int(finite_ts.sum()), n,
            float(mismatch), duplicate_rows, ts_range)

    report_source("historical", base)
    report_source("labelled_signals", signals)
    if with_trades:
        report_source("paper_experience", exp)
    parts = [base, signals]
    if with_trades:
        parts += [exp] * 3                                # real paper-trade outcomes weigh x3
    merged = ds_concat(parts)
    valid = (np.isfinite(merged["X"]).all(axis=1)
             & np.isfinite(merged["y"])
             & np.isin(merged["y"], [0, 1])
             & np.isfinite(merged["ts"])
             & np.isfinite(merged["ret"]))
    dropped = int(len(valid) - valid.sum())
    if dropped:
        log.warning("training data integrity: dropping %d/%d rows with invalid features/labels/timestamps/returns",
                    dropped, len(valid))
        merged = {k: v[valid] for k, v in merged.items()}
    if len(merged["y"]):
        log.info("training data merged: n=%d win_rate=%.3f", len(merged["y"]),
                 float(np.mean(merged["y"] > 0.5)))
    return merged

def retrain(brain, paper, universe=None, reason="manual", refresh=False):
    ds = collect_training_data(paper, universe, refresh)
    info = brain.fit(ds["X"], ds["y"], ds["ts"], ds["ret"])
    paper._set("last_train_n", paper.count("experience"))
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
    tg(f"🧠 تدريب النماذج [{reason}]\nعيّنات: {info['n']} | AUC ({aucs})\n"
       f"النمط المختار: {info['mode']}"
       + (f" | متوسط صفقات الإشارة في التحقق: {top * 100:+.2f}%" if top is not None and np.isfinite(top) else "")
       + diag
       + f"\n{'تم اعتماد النموذج ✅' if info['accepted'] else 'لم يُعتمد (أسوأ من السابق أو ضعيف) ⚠️'}")
    return info

def need_retrain(paper):
    new_trades = paper.count("experience") - int(paper._get("last_train_n") or 0)
    new_sigs = paper.count("signals", "labeled=1") - int(paper._get("last_train_sig") or 0)
    return new_trades >= CFG["RETRAIN_EVERY"] or new_sigs >= CFG["RETRAIN_SIGNALS"]

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
    if trade:
        try:
            paper.check_limit_orders()
        except Exception as e:
            log.warning("LIMIT order check failed: %s", e)
    uni = universe if universe is not None else get_universe()
    log.info("universe: %d coins", len(uni))
    if uni.empty:
        return None
    btc = fetch_btc(CFG["SCAN_DEPTH"], CFG["SCAN_DEPTH"])
    def one(inst):
        try:
            fr = fetch_frames(inst, CFG["SCAN_DEPTH"])
            return inst, (None if fr is None else live_vector(fr, btc))
        except Exception as e:
            log.warning("%s: %s", inst, e); return inst, None
    with ThreadPoolExecutor(4) as ex:
        res = dict(ex.map(one, uni.index))
    X = pd.DataFrame({k: v for k, v in res.items() if v is not None}).T.dropna()
    if X.empty:
        log.warning("no data"); return None
    score = rank_universe(X, uni["vol"])
    R = pd.DataFrame({"score": score}); R["rank"] = np.arange(1, len(R) + 1); R["prob"] = np.nan
    ai_prediction_error = None
    if brain.ready:
        top = list(R.index[:CFG["TOP_AI"]])
        try:
            R.loc[top, "prob"] = brain.predict(X.loc[top, FEATURE_COLS].values)
        except Exception as e:
            ai_prediction_error = str(e)
            log.exception("AI prediction invalid; all new entries will be blocked")
    R.to_csv(os.path.join(CFG["DIR"], "ranking.csv"))
    log.info("top 5:\n%s", R.head(5))
    if cycle % CFG["RANK_MSG_EVERY"] == 0:
        tg(top_msg())
    if not trade:
        return R
    # log the top coins of every scan: they become labelled training examples later
    paper.log_signals([dict(inst=i, price=float(uni.loc[i, "last"]), atr=float(X.loc[i, "1H_atr_pct"]),
                            feats=X.loc[i, FEATURE_COLS].values, prob=R.loc[i, "prob"], rank=int(R.loc[i, "rank"]))
                       for i in R.index[:CFG["LOG_TOP"]]])
    if not brain.ready:
        log.error("AI model unavailable; scan is ranking-only and no entries are allowed")
        tg("⚠️ Zied Quant: نموذج AI غير جاهز أو لم يجتز شروط الاعتماد. تم إيقاف الدخول في صفقات جديدة؛ الترتيب الظاهر ليس توصية شراء. راجع سجل التدريب.")
        return R
    if ai_prediction_error is not None or not np.isfinite(R["prob"].dropna().to_numpy(dtype=float)).all():
        log.error("AI probabilities invalid (%s); scan is ranking-only and no entries are allowed",
                  ai_prediction_error or "non-finite values")
        tg("⚠️ Zied Quant: خرجت احتمالات AI بقيم غير صالحة (NaN/Inf). تم منع جميع أوامر الدخول في هذه الدورة.")
        return R
    gate = paper.risk_gate(paper.equity(uni["last"].to_dict()))
    if gate:
        log.info("entries blocked: %s", gate); return R
    if float(X["ctx_btc_regime"].iloc[0]) < CFG["BTC_FILTER"]:
        log.info("BTC bearish -> no new entries"); return R
    thr = CFG["PROB_MIN"] if CFG["PROB_MIN"] is not None else brain.thr
    sig = R[R["prob"] >= thr].sort_values("prob", ascending=False)
    cache = {}
    for inst, r in sig.iterrows():
        open_inst = [t["inst"] for t in paper.open_trades()]
        pending_inst = [o["inst"] for o in paper.pending_orders()]
        active = open_inst + pending_inst
        if inst in active or len(active) >= CFG["MAX_OPEN"]:
            continue
        corr = max_corr_with_open(inst, active, cache)
        if corr > CFG["CORR_MAX"]:
            log.info("%s skipped: corr %.2f with an open/pending position", inst, corr); continue
        last = float(uni.loc[inst, "last"])
        ask = float(uni.loc[inst, "ask"])
        spread = float(uni.loc[inst, "spread"])
        atr_pct = float(X.loc[inst, "1H_atr_pct"])
        # Place a modestly discounted limit: no more than 0.30% below last, but account for spread.
        discount = max(spread / 2.0, min(0.25 * float(np.clip(atr_pct, 0.004, 0.06)), 0.003))
        limit_px = min(ask, last * (1.0 - discount))
        tg(f"🟡 إشارة شراء LIMIT {inst}\nاحتمال AI: {r['prob'] * 100:.1f}% (العتبة {thr * 100:.1f}%)\n"
           f"الترتيب: #{int(r['rank'])} | السعر الحالي: {last:.8g} | سعر LIMIT: {limit_px:.8g}")
        t = paper.place_limit(inst, limit_px, atr_pct, float(r["prob"]), int(r["rank"]),
                              X.loc[inst, FEATURE_COLS].values, float(uni.loc[inst, "vol"]), spread)
        if t:
            tg(f"📋 أمر LIMIT افتراضي معلّق {inst}\nالسعر المحدد: {t['limit_px']:.8g}\n"
               f"SL: {t['sl']:.8g} | TP: {t['tp']:.8g}\n"
               f"المحجوز: {t['usdt']:.2f} USDT | الصلاحية: {CFG['LIMIT_TTL_H']} ساعات")
        else:
            log.info("%s LIMIT skipped by cash, slot, exposure or aggregate-risk controls", inst)
    return R

def once():
    """One full cycle for schedulers (GitHub Actions)."""
    brain, paper = Brain(), Paper()
    uni = get_universe()
    if uni.empty:
        raise RuntimeError("OKX returned no tradable pairs (blocked? try OKX_BASE=https://aws.okx.com)")
    if not brain.ready:
        try:
            retrain(brain, paper, uni, "أول تدريب")
        except Exception as e:
            log.exception("first training failed: %s", e); tg(f"⚠️ فشل أول تدريب: {e}")
    poll_commands(paper, brain)
    paper.check_candles()
    label_signals(paper)
    cycle = int(paper._get("cycle") or 0)
    scan(brain, paper, uni, cycle)
    paper._set("cycle", cycle + 1)
    # Send a complete portfolio snapshot to Telegram after every hourly GitHub Actions cycle.
    tg(portfolio_msg(paper))
    if need_retrain(paper):
        try:
            retrain(brain, paper, uni, "تعلّم ذاتي")
        except Exception as e:
            log.warning("retrain: %s", e)
    if time.time() - float(paper._get("last_daily") or 0) > 86400:
        tg(stats_msg(paper)); paper._set("last_daily", time.time())

def monitor_loop(paper, stop):
    while not stop.is_set():
        try:
            if paper.open_trades():
                paper.check(get_prices())
        except Exception as e:
            log.warning("monitor: %s", e)
        stop.wait(CFG["MONITOR_EVERY_SEC"])

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
    if not brain.ready:
        retrain(brain, paper, uni, "أول تدريب")
    threading.Thread(target=monitor_loop, args=(paper, stop), daemon=True).start()
    threading.Thread(target=telegram_loop, args=(paper, brain, stop), daemon=True).start()
    tg("🚀 Zied Quant v2 بدأ العمل (Paper Trading فقط). أرسل /help للأوامر.")
    cycle, last_daily = 0, time.time()
    try:
        while True:
            t0 = time.time()
            try:
                label_signals(paper)
                scan(brain, paper, None, cycle)
            except Exception as e:
                log.exception("scan failed: %s", e); tg(f"⚠️ خطأ في المسح: {e}")
            cycle += 1
            if need_retrain(paper):
                try:
                    retrain(brain, paper, None, "تعلّم ذاتي")
                except Exception as e:
                    log.warning("retrain: %s", e)
            if time.time() - last_daily > 86400:
                tg(stats_msg(paper)); last_daily = time.time()
            time.sleep(max(5, CFG["SCAN_EVERY_SEC"] - (time.time() - t0)))
    except KeyboardInterrupt:
        stop.set(); tg("⏹ Zied Quant توقف")

def main(argv):
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd in ("train", "refresh", "backtest", "scan", "once", "run") and torch is None:
        try:
            import sklearn  # noqa: F401
            log.warning("torch missing: the 20-layer net is disabled, using gradient boosting only")
        except ImportError:
            sys.exit("pip install torch scikit-learn lightgbm  (at least one model library is required)")
    if cmd in ("train", "refresh"):
        b, p = Brain(), Paper(); print(retrain(b, p, get_universe(), "يدوي", refresh=(cmd == "refresh")))
    elif cmd == "backtest":
        p = Paper(); folds = int(argv[2]) if len(argv) > 2 else 5
        ds = collect_training_data(p, with_trades=False)
        rep = walk_forward(ds, folds=folds)
        txt = backtest_report(rep); print(txt)
        json.dump(rep, open(os.path.join(CFG["DIR"], "backtest.json"), "w"), indent=2)
        tg(txt)
    elif cmd == "scan":
        b, p = Brain(), Paper(); print(scan(b, p, trade=False).head(30))
    elif cmd == "once":
        once()
    elif cmd == "label":
        print(label_signals(Paper()))
    elif cmd == "stats":
        print(stats_msg(Paper()))
    else:
        run()

if __name__ == "__main__":
    main(sys.argv)
