#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zied Quant
==========
OKX Spot (USDT, no stablecoins) -> 150 features (30 x 5 timeframes)
-> cross-sectional ranking -> 20-layer deep net -> BUY signal
-> paper trading (SQLite) + self-learning + Telegram alerts.

PAPER ONLY. No real order is ever sent. Public OKX endpoints only (no API key).

Usage:
    python zied_quant.py train     # download history + train the 20-layer net
    python zied_quant.py scan      # one scan (ranking + signals), no loop
    python zied_quant.py run       # full loop: scan + paper trades + monitor + learn
    python zied_quant.py once      # one full cycle (for schedulers / GitHub Actions)
    python zied_quant.py stats     # paper-trading statistics

Env (optional, for Telegram):
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os, sys, time, json, sqlite3, logging, threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

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
    BASE_TF="1H",                      # timeframe used to build training rows
    SCAN_DEPTH=300,                    # candles per TF for live scan
    TRAIN_DEPTH={"5m": 1400, "15m": 1400, "1H": 1400, "4H": 900, "1D": 400},
    TRAIN_TOP=40,                      # coins (by volume) used for bootstrap training
    MIN_VOL_USDT=300_000,              # min 24h quote volume to be in the universe
    LABEL_H=4,                         # forward horizon in BASE_TF bars
    LABEL_THR=0.010,                   # +1.0% forward return = positive label
    TOP_AI=30,                         # coins analysed by the deep net each scan
    SCAN_EVERY_SEC=900,                # 15 min
    MONITOR_EVERY_SEC=30,
    RANK_MSG_EVERY=4,                  # send ranking to Telegram every N scans
    # paper trading
    START_BALANCE=1000.0,
    POS_FRAC=0.15,                     # fraction of equity per trade
    MAX_OPEN=5,
    FEE=0.001,                         # OKX spot taker ~0.1% per side
    SLIP=0.0005,
    SL_ATR=1.5, TP_ATR=3.0,
    MAX_HOLD_H=24,
    BTC_FILTER=-0.04,                  # no new entries if BTC 4H dist to EMA50 < this
    RETRAIN_EVERY=20,                  # retrain after N newly closed trades
    PROB_MIN=None,                     # None = auto threshold from validation
    DIR="zied_data",
)
os.makedirs(CFG["DIR"], exist_ok=True)
CFG["DB"] = os.path.join(CFG["DIR"], "paper.db")
CFG["MODEL"] = os.path.join(CFG["DIR"], "model.pt")
CFG["TRAINSET"] = os.path.join(CFG["DIR"], "train.npz")

TFS, BASE = CFG["TFS"], CFG["BASE_TF"]
EPS = 1e-12

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
                      json={"chat_id": chat, "text": msg}, timeout=10)
    except Exception as e:
        log.warning("telegram failed: %s", e)

# =====================================================================
# OKX DATA
# =====================================================================
_thr_lock, _last_call = threading.Lock(), [0.0]

def okx_get(path, params=None, tries=4):
    for i in range(tries):
        with _thr_lock:                               # global throttle ~14 req/s
            dt = time.time() - _last_call[0]
            if dt < 0.07:
                time.sleep(0.07 - dt)
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
    """All live SPOT *-USDT pairs, no stablecoins, with 24h volume + last/ask."""
    ins = okx_get("/api/v5/public/instruments", {"instType": "SPOT"})
    ok = {i["instId"] for i in ins
          if i["quoteCcy"] == "USDT" and i["state"] == "live" and not is_stable(i["baseCcy"])}
    rows = []
    for t in okx_get("/api/v5/market/tickers", {"instType": "SPOT"}):
        if t["instId"] in ok:
            vol = float(t.get("volCcy24h") or 0)
            if vol >= CFG["MIN_VOL_USDT"]:
                rows.append(dict(inst=t["instId"], vol=vol, last=float(t["last"]),
                                 ask=float(t.get("askPx") or t["last"])))
    return pd.DataFrame(rows).set_index("inst").sort_values("vol", ascending=False)

def get_prices():
    return {t["instId"]: float(t["last"]) for t in
            okx_get("/api/v5/market/tickers", {"instType": "SPOT"})}

def get_candles(inst, bar, depth):
    out, after = [], None
    while len(out) < depth:
        p = {"instId": inst, "bar": bar, "limit": 300}
        if after:
            p["after"] = after
        r = okx_get("/api/v5/market/candles", p)
        if not r:
            break
        out += r
        after = r[-1][0]
        if len(r) < 300:
            break
    if not out:
        return None
    df = pd.DataFrame(out, columns=["ts", "o", "h", "l", "c", "vol", "volccy", "volq", "confirm"])
    df = df[df["confirm"] == "1"].copy()                  # closed candles only
    for k in ("o", "h", "l", "c", "vol"):
        df[k] = df[k].astype(float)
    df["ts"] = df["ts"].astype("int64")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["avail"] = pd.to_datetime(df["ts"] + CFG["TF_SEC"][bar] * 1000, unit="ms")
    return df.iloc[-depth:].reset_index(drop=True)

# =====================================================================
# FEATURES : 30 per timeframe x 5 timeframes = 150
# =====================================================================
NAMES = ["ret1", "ret3", "ret5", "ret10", "ret20", "vol10", "vol20", "rsi7", "rsi14",
         "macd", "macd_hist", "stoch_k", "bb_pos", "bb_width", "atr_pct", "adx14", "cci20",
         "mfi14", "obv_slope", "vol_ratio", "ema_dist9", "ema_dist21", "ema_dist50",
         "ema_slope21", "dist_high20", "dist_low20", "range_pct", "body_ratio",
         "upper_wick", "lower_wick"]
FEATURE_COLS = [f"{tf}_{n}" for tf in TFS for n in NAMES]
assert len(FEATURE_COLS) == 150

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
    r = lc.diff()
    f["vol10"], f["vol20"] = r.rolling(10).std(), r.rolling(20).std()
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
    e9, e21, e50 = (c.ewm(span=s, adjust=False).mean() for s in (9, 21, 50))
    f["ema_dist9"], f["ema_dist21"], f["ema_dist50"] = c / e9 - 1, c / e21 - 1, c / e50 - 1
    f["ema_slope21"] = e21.pct_change(5)
    f["dist_high20"] = c / h.rolling(20).max() - 1
    f["dist_low20"] = c / l.rolling(20).min() - 1
    rng = h - l + EPS
    f["range_pct"] = (h - l) / c
    f["body_ratio"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    out = pd.DataFrame(f)[NAMES].replace([np.inf, -np.inf], np.nan)
    out.index = df["avail"].values
    return out

def all_tf_features(frames):
    F = {}
    for tf in TFS:
        x = tf_features(frames[tf]); x.columns = [f"{tf}_{n}" for n in NAMES]; F[tf] = x
    return F

def live_vector(frames):
    """Latest closed-candle features of every timeframe -> 150 values."""
    F = all_tf_features(frames)
    return pd.concat([F[tf].iloc[-1] for tf in TFS])[FEATURE_COLS]

def aligned_matrix(frames):
    """Historical rows on BASE_TF; other TFs use the last candle already closed (no look-ahead)."""
    F = all_tf_features(frames)
    idx = F[BASE].index
    parts = [F[BASE]] + [F[tf].reindex(idx, method="ffill") for tf in TFS if tf != BASE]
    return pd.concat(parts, axis=1)[FEATURE_COLS]

def fetch_frames(inst, depth):
    fr = {}
    for tf in TFS:
        d = get_candles(inst, tf, depth if isinstance(depth, int) else depth[tf])
        if d is None or len(d) < 60:
            return None
        fr[tf] = d
    return fr

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
# DEEP NET : 20 Linear layers (1 input + 18 hidden in 9 residual blocks + 1 output)
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

    class ZiedNet(nn.Module):
        def __init__(self, n_in, d=192, p=0.15):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.LayerNorm(d), nn.GELU())
            self.blocks = nn.Sequential(*[Block(d, p) for _ in range(9)])
            self.out = nn.Linear(d, 1)
        def forward(self, x):
            return self.out(self.blocks(self.inp(x)))

def n_linear(m):
    return sum(isinstance(x, nn.Linear) for x in m.modules())

def auc_score(y, p):
    y = np.asarray(y); o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

class Brain:
    def __init__(self):
        self.net = None; self.mu = self.sd = None; self.thr = 0.5; self.auc = None
        self.load()

    @property
    def ready(self):
        return self.net is not None

    def _norm(self, X):
        return np.clip((np.asarray(X, dtype=np.float64) - self.mu) / self.sd, -5, 5).astype(np.float32)

    def load(self):
        if torch is None or not os.path.exists(CFG["MODEL"]):
            return
        ck = torch.load(CFG["MODEL"], weights_only=False)
        self.net = ZiedNet(ck["n_in"]); self.net.load_state_dict(ck["state"]); self.net.eval()
        self.mu, self.sd, self.thr, self.auc = ck["mu"], ck["sd"], ck["thr"], ck["auc"]
        log.info("model loaded (auc=%.3f thr=%.3f)", self.auc, self.thr)

    def predict(self, X):
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.tensor(self._norm(X))).squeeze(1)).numpy()

    def fit(self, X, y, ts, epochs=80):
        if torch is None:
            raise RuntimeError("pip install torch")
        o = np.argsort(ts); X, y = np.asarray(X)[o], np.asarray(y)[o].astype(np.float32)
        n = len(X); nv = max(int(n * 0.2), 100)
        if n < 600:
            raise ValueError(f"not enough samples ({n})")
        mu = X[:n - nv].mean(0); sd = X[:n - nv].std(0) + 1e-8
        tmp = Brain.__new__(Brain); tmp.mu, tmp.sd = mu, sd
        Xt, Xv = torch.tensor(tmp._norm(X[:n - nv])), torch.tensor(tmp._norm(X[n - nv:]))
        yt, yv = torch.tensor(y[:n - nv]), torch.tensor(y[n - nv:])
        net = ZiedNet(X.shape[1])
        assert n_linear(net) == 20, n_linear(net)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.BCEWithLogitsLoss()
        best, state, bad = 1e9, None, 0
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
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
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv).squeeze(1)).numpy()
        auc = auc_score(yv.numpy(), pv)
        thr = float(np.quantile(pv, 0.90))             # top 10% of validation predictions
        info = dict(auc=auc, thr=thr, val_loss=best, n=n, pos_rate=float(y.mean()))
        if auc < 0.52 or (self.auc is not None and auc < self.auc - 0.03):
            info["accepted"] = False
            return info
        self.net, self.mu, self.sd, self.thr, self.auc = net, mu, sd, thr, auc
        torch.save(dict(state=net.state_dict(), n_in=X.shape[1], mu=mu, sd=sd, thr=thr, auc=auc),
                   CFG["MODEL"])
        info["accepted"] = True
        return info

# =====================================================================
# TRAINING DATA (bootstrap from history)
# =====================================================================
def build_trainset(universe):
    insts = list(universe.index[:CFG["TRAIN_TOP"]])
    log.info("downloading history for %d coins ...", len(insts))
    def one(inst):
        fr = fetch_frames(inst, CFG["TRAIN_DEPTH"])
        if fr is None:
            return None
        M = aligned_matrix(fr)
        close = fr[BASE].set_index("avail")["c"]
        fwd = close.shift(-CFG["LABEL_H"]) / close - 1
        M = M.assign(_y=(fwd > CFG["LABEL_THR"]).astype(float), _f=fwd).dropna()
        M["_ts"] = M.index.astype("int64") // 10**9
        return M
    with ThreadPoolExecutor(4) as ex:
        parts = [p for p in ex.map(one, insts) if p is not None and len(p)]
    D = pd.concat(parts)
    np.savez_compressed(CFG["TRAINSET"], X=D[FEATURE_COLS].values, y=D["_y"].values, ts=D["_ts"].values)
    log.info("trainset: %d rows, positive rate %.1f%%", len(D), 100 * D["_y"].mean())
    return D[FEATURE_COLS].values, D["_y"].values, D["_ts"].values

def load_trainset():
    if not os.path.exists(CFG["TRAINSET"]):
        return None
    z = np.load(CFG["TRAINSET"]); return z["X"], z["y"], z["ts"]

# =====================================================================
# PAPER TRADING + SELF-LEARNING MEMORY
# =====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, inst TEXT, open_ts REAL, entry REAL, qty REAL,
  usdt_in REAL, sl REAL, tp REAL, prob REAL, rank INTEGER, feats TEXT,
  status TEXT DEFAULT 'OPEN', close_ts REAL, exit REAL, pnl REAL, pnl_pct REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS experience(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, inst TEXT, feats TEXT, label INTEGER, pnl REAL);
"""

class Paper:
    def __init__(self):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(CFG["DB"], check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if self._get("cash") is None:
            self._set("cash", CFG["START_BALANCE"])

    def _get(self, k):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return None if r is None else r["v"]

    def _set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v))); self.db.commit()

    def cash(self):
        return float(self._get("cash"))

    def open_trades(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM trades WHERE status='OPEN'")]

    def equity(self, prices=None):
        eq = self.cash()
        for t in self.open_trades():
            px = (prices or {}).get(t["inst"], t["entry"]); eq += t["qty"] * px
        return eq

    def open(self, inst, price, atr_pct, prob, rank, feats):
        with self.lock:
            if any(t["inst"] == inst for t in self.open_trades()):
                return None
            if len(self.open_trades()) >= CFG["MAX_OPEN"]:
                return None
            usdt = min(self.cash(), CFG["POS_FRAC"] * self.equity())
            if usdt < 10:
                return None
            entry = price * (1 + CFG["SLIP"])
            qty = usdt * (1 - CFG["FEE"]) / entry
            a = float(np.clip(atr_pct, 0.004, 0.06))
            sl, tp = entry * (1 - CFG["SL_ATR"] * a), entry * (1 + CFG["TP_ATR"] * a)
            self.db.execute(
                "INSERT INTO trades(inst,open_ts,entry,qty,usdt_in,sl,tp,prob,rank,feats) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (inst, time.time(), entry, qty, usdt, sl, tp, prob, rank, json.dumps([float(x) for x in feats])))
            self._set("cash", self.cash() - usdt)
            return dict(inst=inst, entry=entry, sl=sl, tp=tp, usdt=usdt)

    def _close(self, t, price, reason):
        proceeds = t["qty"] * price * (1 - CFG["FEE"])
        pnl = proceeds - t["usdt_in"]; pct = pnl / t["usdt_in"]
        self.db.execute("UPDATE trades SET status='CLOSED',close_ts=?,exit=?,pnl=?,pnl_pct=?,reason=? WHERE id=?",
                        (time.time(), price, pnl, pct, reason, t["id"]))
        self.db.execute("INSERT INTO experience(ts,inst,feats,label,pnl) VALUES(?,?,?,?,?)",
                        (time.time(), t["inst"], t["feats"], int(pnl > 0), pnl))
        self._set("cash", self.cash() + proceeds); self.db.commit()
        icon = "✅" if pnl > 0 else "🔴"
        tg(f"{icon} إغلاق صفقة ورقية {t['inst']}\nالسبب: {reason}\n"
           f"دخول {t['entry']:.6g} ← خروج {price:.6g}\nالربح: {pnl:+.2f} USDT ({pct * 100:+.2f}%)")

    def check(self, prices):
        closed = 0
        with self.lock:
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

    def check_candles(self):
        """Hourly mode: scan 5m candles since entry for SL/TP hits (SL first if both in one candle)."""
        closed = 0
        with self.lock:
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

    def experience(self):
        rows = self.db.execute("SELECT ts,feats,label FROM experience ORDER BY ts").fetchall()
        if not rows:
            return None
        return (np.array([json.loads(r["feats"]) for r in rows]),
                np.array([r["label"] for r in rows], dtype=float),
                np.array([r["ts"] for r in rows]))

    def stats(self, prices=None):
        cl = [dict(r) for r in self.db.execute("SELECT * FROM trades WHERE status='CLOSED'")]
        n = len(cl); wins = [t for t in cl if t["pnl"] > 0]; loss = [t for t in cl if t["pnl"] <= 0]
        gp, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in loss)
        eq = self.equity(prices)
        return dict(trades=n, open=len(self.open_trades()), winrate=(len(wins) / n if n else 0),
                    pnl=sum(t["pnl"] for t in cl), profit_factor=(gp / gl if gl > 0 else float("inf") if gp else 0),
                    equity=eq, ret_pct=100 * (eq / CFG["START_BALANCE"] - 1))

# =====================================================================
# PIPELINE
# =====================================================================
def retrain(brain, paper, universe=None, reason="manual"):
    parts = []
    base = load_trainset()
    if base is None:
        universe = universe if universe is not None else get_universe()
        base = build_trainset(universe)
    parts.append(base)
    exp = paper.experience()
    if exp is not None:                               # real paper-trade outcomes, weight x3
        for _ in range(3):
            parts.append(exp)
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    ts = np.concatenate([p[2] for p in parts])
    info = brain.fit(X, y, ts)
    paper._set("last_train_n", paper.db.execute("SELECT COUNT(*) c FROM experience").fetchone()["c"])
    tg(f"🧠 تدريب الشبكة (20 طبقة) [{reason}]\nعيّنات: {info['n']} | AUC: {info['auc']:.3f}\n"
       f"{'تم اعتماد النموذج ✅' if info['accepted'] else 'لم يُعتمد (أسوأ من السابق) ⚠️'}")
    return info

def scan(brain, paper, universe=None, cycle=0, trade=True):
    uni = universe if universe is not None else get_universe()
    log.info("universe: %d coins", len(uni))
    def one(inst):
        fr = fetch_frames(inst, CFG["SCAN_DEPTH"])
        return inst, (None if fr is None else live_vector(fr))
    with ThreadPoolExecutor(4) as ex:
        res = dict(ex.map(one, uni.index))
    X = pd.DataFrame({k: v for k, v in res.items() if v is not None}).T
    X = X.dropna()
    if X.empty:
        log.warning("no data"); return None
    score = rank_universe(X, uni["vol"])
    R = pd.DataFrame({"score": score}); R["rank"] = np.arange(1, len(R) + 1); R["prob"] = np.nan
    if brain.ready:
        top = list(R.index[:CFG["TOP_AI"]])
        R.loc[top, "prob"] = brain.predict(X.loc[top, FEATURE_COLS].values)
    R.to_csv(os.path.join(CFG["DIR"], "ranking.csv"))
    log.info("top 5:\n%s", R.head(5))
    if cycle % CFG["RANK_MSG_EVERY"] == 0:
        lines = [f"{int(r['rank'])}. {i.replace('-USDT', '')}  نقاط {r['score']:+.2f}"
                 + (f"  AI {r['prob'] * 100:.0f}%" if not np.isnan(r["prob"]) else "")
                 for i, r in R.head(10).iterrows()]
        tg("📊 أفضل 10 عملات الآن (OKX Spot)\n" + "\n".join(lines))
    if not (brain.ready and trade):
        return R
    # BTC regime filter
    btc = X.loc["BTC-USDT", "4H_ema_dist50"] if "BTC-USDT" in X.index else 0
    if btc < CFG["BTC_FILTER"]:
        log.info("BTC bearish (%.3f) -> no new entries", btc); return R
    thr = CFG["PROB_MIN"] or brain.thr
    sig = R[(R["prob"] >= thr)].sort_values("prob", ascending=False)
    for inst, r in sig.iterrows():
        px = float(uni.loc[inst, "ask"])
        atr = float(X.loc[inst, "1H_atr_pct"])
        tg(f"🟢 إشارة شراء {inst}\nاحتمال AI: {r['prob'] * 100:.1f}% (العتبة {thr * 100:.1f}%)\n"
           f"الترتيب: #{int(r['rank'])} | السعر: {px:.6g}")
        t = paper.open(inst, px, atr, float(r["prob"]), int(r["rank"]), X.loc[inst, FEATURE_COLS].values)
        if t:
            tg(f"📥 فتح صفقة ورقية {inst}\nدخول {t['entry']:.6g} | SL {t['sl']:.6g} | TP {t['tp']:.6g}\n"
               f"الحجم: {t['usdt']:.2f} USDT")
    return R

def monitor_loop(paper, stop, retrain_flag):
    while not stop.is_set():
        try:
            if paper.open_trades():
                if paper.check(get_prices()):
                    retrain_flag.set()
        except Exception as e:
            log.warning("monitor: %s", e)
        stop.wait(CFG["MONITOR_EVERY_SEC"])

def stats_msg(paper):
    s = paper.stats(get_prices())
    return (f"📈 ملخص Paper Trading\nالرصيد الكلي: {s['equity']:.2f} USDT ({s['ret_pct']:+.2f}%)\n"
            f"صفقات مغلقة: {s['trades']} | مفتوحة: {s['open']}\n"
            f"نسبة النجاح: {s['winrate'] * 100:.1f}% | Profit factor: {s['profit_factor']:.2f}")

def once():
    """One full cycle, for schedulers (GitHub Actions): check trades, scan, learn, report."""
    brain, paper = Brain(), Paper()
    uni = get_universe()
    if not brain.ready:
        retrain(brain, paper, uni, "أول تدريب")
    paper.check_candles()
    cycle = int(paper._get("cycle") or 0)
    scan(brain, paper, uni, cycle)
    paper._set("cycle", cycle + 1)
    done = paper.db.execute("SELECT COUNT(*) c FROM experience").fetchone()["c"]
    if done - int(paper._get("last_train_n") or 0) >= CFG["RETRAIN_EVERY"]:
        retrain(brain, paper, uni, f"تعلّم ذاتي بعد {done} صفقة")
    if time.time() - float(paper._get("last_daily") or 0) > 86400:
        tg(stats_msg(paper)); paper._set("last_daily", time.time())

def run():
    brain, paper = Brain(), Paper()
    stop, flag = threading.Event(), threading.Event()
    uni = get_universe()
    if not brain.ready:
        retrain(brain, paper, uni, "أول تدريب")
        if not brain.ready:
            log.error("model not accepted; running ranking only")
    threading.Thread(target=monitor_loop, args=(paper, stop, flag), daemon=True).start()
    tg("🚀 Zied Quant بدأ العمل (Paper Trading فقط)")
    cycle, last_daily = 0, time.time()
    try:
        while True:
            t0 = time.time()
            try:
                scan(brain, paper, None, cycle)
            except Exception as e:
                log.exception("scan failed: %s", e); tg(f"⚠️ خطأ في المسح: {e}")
            cycle += 1
            done = paper.db.execute("SELECT COUNT(*) c FROM experience").fetchone()["c"]
            last = int(paper._get("last_train_n") or 0)
            if done - last >= CFG["RETRAIN_EVERY"]:
                try: retrain(brain, paper, None, f"تعلّم ذاتي بعد {done} صفقة")
                except Exception as e: log.warning("retrain: %s", e)
            flag.clear()
            if time.time() - last_daily > 86400:
                tg(stats_msg(paper)); last_daily = time.time()
            time.sleep(max(5, CFG["SCAN_EVERY_SEC"] - (time.time() - t0)))
    except KeyboardInterrupt:
        stop.set(); tg("⏹ Zied Quant توقف")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd in ("train", "scan", "run", "once") and torch is None:
        sys.exit("pip install torch  (required)")
    if cmd == "train":
        b, p = Brain(), Paper(); print(retrain(b, p, get_universe(), "يدوي"))
    elif cmd == "scan":
        b, p = Brain(), Paper(); print(scan(b, p, trade=False).head(30))
    elif cmd == "once":
        once()
    elif cmd == "stats":
        print(stats_msg(Paper()))
    else:
        run()
