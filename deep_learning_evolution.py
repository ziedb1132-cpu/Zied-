#!/usr/bin/env python3
"""300-feature, 30-linear-layer deep-learning extension and winning-pattern discovery.

Paper trading only. Keeps existing risk gates and temporal validation intact.
"""
import json
import os
import sqlite3
import time
import types

import numpy as np
import pandas as pd


EXTRA_NAMES = [
    "ret2", "ret8", "ret15", "ret30", "ret90",
    "rsi21", "rsi28",
    "ema_dist12", "ema_dist100", "ema_dist200",
    "ema_slope50", "ema_slope100",
    "vol_ratio5", "vol_ratio50",
    "atr_pct7", "atr_pct28",
    "range_pos50", "breakout_dist50", "drawdown20", "gap_pct",
]
WIN_THRESHOLD = 500
LOSS_THRESHOLD = 500


def _extra_features(df):
    o = pd.to_numeric(df["o"], errors="coerce")
    h = pd.to_numeric(df["h"], errors="coerce")
    l = pd.to_numeric(df["l"], errors="coerce")
    c = pd.to_numeric(df["c"], errors="coerce")
    v = pd.to_numeric(df["vol"], errors="coerce")
    lc = np.log(c.where(c > 0))
    f = {}
    for n in (2, 8, 15, 30, 90):
        f[f"ret{n}"] = lc.diff(n)
    def rsi(n):
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + up / (dn + 1e-12))
    f["rsi21"], f["rsi28"] = rsi(21), rsi(28)
    for n in (12, 100, 200):
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"ema_dist{n}"] = c / (ema + 1e-12) - 1
    e50 = c.ewm(span=50, adjust=False).mean()
    e100 = c.ewm(span=100, adjust=False).mean()
    f["ema_slope50"], f["ema_slope100"] = e50.pct_change(5), e100.pct_change(10)
    f["vol_ratio5"] = v / (v.rolling(5).mean() + 1e-12)
    f["vol_ratio50"] = v / (v.rolling(50).mean() + 1e-12)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct7"] = tr.ewm(alpha=1/7, adjust=False).mean() / (c + 1e-12)
    f["atr_pct28"] = tr.ewm(alpha=1/28, adjust=False).mean() / (c + 1e-12)
    lo50, hi50 = l.rolling(50).min(), h.rolling(50).max()
    f["range_pos50"] = (c-lo50) / (hi50-lo50+1e-12)
    prior_hi = h.shift(1).rolling(50).max()
    f["breakout_dist50"] = c / (prior_hi + 1e-12) - 1
    hi20 = c.rolling(20).max()
    f["drawdown20"] = c / (hi20 + 1e-12) - 1
    f["gap_pct"] = o / (c.shift(1) + 1e-12) - 1
    return pd.DataFrame(f, index=df.index)[EXTRA_NAMES].replace([np.inf, -np.inf], np.nan)


def _install_features(zq):
    old_names = list(zq.NAMES)
    old_tf = zq.tf_features
    # Give the original function its own globals so expanded NAMES is thread-safe.
    old_globals = dict(old_tf.__globals__)
    old_globals["NAMES"] = old_names
    old_tf_isolated = types.FunctionType(
        old_tf.__code__, old_globals, old_tf.__name__, old_tf.__defaults__, old_tf.__closure__
    )
    new_names = old_names + EXTRA_NAMES
    def expanded_tf_features(df):
        base = old_tf_isolated(df)
        extra = _extra_features(df)
        extra.index = base.index
        return pd.concat([base, extra], axis=1)[new_names]
    zq.NAMES = new_names
    zq.TF_COLS = [f"{tf}_{n}" for tf in zq.TFS for n in new_names]
    zq.CTX_COLS = [f"ctx_{n}" for n in zq.CTX_NAMES]
    zq.FEATURE_COLS = zq.TF_COLS + zq.CTX_COLS
    zq.N_FEATURES = len(zq.FEATURE_COLS)
    if zq.N_FEATURES != 300:
        raise RuntimeError(f"Expected 300 features, got {zq.N_FEATURES}")
    zq.tf_features = expanded_tf_features
    # Old cached arrays/models use 200 columns and must be rebuilt, not padded.
    zq.DATA_VER = 4
    zq.CFG["TOP_AI"] = 50
    zq.CFG["LOG_TOP"] = 50
    zq.CFG["MAX_OPEN"] = 10
    # Smaller notional per position lets up to 10 slots fit within the existing 75% exposure cap.
    zq.CFG["POS_FRAC"] = 0.075


def _install_30_layer_net(zq):
    if zq.torch is None:
        zq.log.warning("PyTorch unavailable: 30-layer deep network disabled; only GBM can run")
        return
    torch, nn = zq.torch, zq.nn
    class Block(nn.Module):
        def __init__(self, d, p):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p),
                nn.Linear(d, d), nn.LayerNorm(d)
            )
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(x + self.net(x))
    class ZiedDeepNet(nn.Module):
        # 1 input linear + 14 residual blocks x 2 linear + 1 output linear = 30.
        def __init__(self, n_in, d=192, p=0.15):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, d), nn.LayerNorm(d), nn.GELU())
            self.blocks = nn.Sequential(*[Block(d, p) for _ in range(14)])
            self.out = nn.Linear(d, 1)
        def forward(self, x):
            return self.out(self.blocks(self.inp(x)))
    def n_linear(model):
        return sum(isinstance(layer, nn.Linear) for layer in model.modules())
    def train_net(Xf, yf, Xe, ye, epochs):
        torch.manual_seed(0)
        Xt, yt = torch.tensor(np.asarray(Xf, dtype=np.float32)), torch.tensor(np.asarray(yf, dtype=np.float32))
        Xv, yv = torch.tensor(np.asarray(Xe, dtype=np.float32)), torch.tensor(np.asarray(ye, dtype=np.float32))
        net = ZiedDeepNet(Xf.shape[1])
        assert n_linear(net) == 30, n_linear(net)
        opt = torch.optim.AdamW(net.parameters(), lr=7e-4, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
        lossf = nn.BCEWithLogitsLoss()
        best, state, bad = float("inf"), None, 0
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 256):
                b = perm[i:i+256]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = lossf(net(Xt[b]).squeeze(1), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():
                val_loss = lossf(net(Xv).squeeze(1), yv).item()
            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        if state is None:
            raise RuntimeError("30-layer deep network training did not produce a valid checkpoint")
        net.load_state_dict(state)
        net.eval()
        return net
    zq.ZiedNet = ZiedDeepNet
    zq._train_net = train_net
    old_describe = zq.Brain.describe
    def describe(self):
        result = old_describe(self)
        return result.replace("شبكة 20 طبقة", "شبكة Deep Learning من 30 طبقة")
    zq.Brain.describe = describe


def discover_winning_patterns(zq, paper):
    """Discover winner-associated features from the canonical, labelled signals table.

    The old implementation queried an 'experience' table, but Paper.__init__
    deliberately drops that legacy table. The canonical source is signals:
    each row is labelled from later candles and contains the original 300-feature
    vector plus net return. Do not count an outcome as a win unless both its
    label and net return agree.
    """
    db_path = zq.CFG["DB"]
    if not os.path.exists(db_path):
        return {"status": "no_database", "winning_samples": 0, "losing_samples": 0}

    try:
        with sqlite3.connect(db_path, timeout=10) as db:
            rows = db.execute(
                "SELECT COALESCE(s.feature_ts, s.ts) AS feature_ts, s.feats, "
                "CASE WHEN s.ret>0 THEN 1 ELSE 0 END AS label, s.ret "
                "FROM signals s JOIN tracked_signals t ON t.signal_id=s.id "
                "WHERE t.status='CLOSED' AND s.labeled=1 AND s.ret IS NOT NULL "
                "ORDER BY t.closed_ts, s.id"
            ).fetchall()
    except sqlite3.Error as exc:
        zq.log.warning("pattern discovery could not read labelled signals: %s", exc)
        return {"status": "database_error", "error": str(exc)}

    parsed = []
    for feature_ts, feats_json, label, ret in rows:
        try:
            vector = np.asarray(json.loads(feats_json), dtype=np.float64)
            label, ret = int(label), float(ret)
            if (vector.shape == (zq.N_FEATURES,)
                    and np.isfinite(vector).all()
                    and label in (0, 1)
                    and np.isfinite(ret)):
                parsed.append((float(feature_ts or 0.0), vector, label, ret))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    winners = [r for r in parsed if r[2] == 1 and r[3] > 0.0]
    losers = [r for r in parsed if r[2] == 0 and r[3] < 0.0]
    path = os.path.join(zq.CFG["DIR"], "winning_patterns.json")
    if len(winners) < WIN_THRESHOLD or len(losers) < LOSS_THRESHOLD:
        # Old patterns must not remain active before the stricter evidence gate.
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                zq.log.warning("could not remove stale winning-pattern memory: %s", exc)
        return {"status": "waiting_for_closed_outcomes",
                "winning_samples": len(winners), "losing_samples": len(losers),
                "required_winners": WIN_THRESHOLD, "required_losers": LOSS_THRESHOLD}

    old_count = int(paper._get("last_pattern_closed_count") or 0)
    closed_count = len(winners) + len(losers)
    if closed_count <= old_count and os.path.exists(path):
        return {"status": "no_new_closed_signal",
                "winning_samples": len(winners), "losing_samples": len(losers)}

    W = np.vstack([r[1] for r in winners])
    L = np.vstack([r[1] for r in losers])
    wm, lm = np.median(W, axis=0), np.median(L, axis=0)
    pooled = np.vstack([W, L])
    pooled_median = np.median(pooled, axis=0)
    scale = np.median(np.abs(pooled - pooled_median), axis=0) * 1.4826
    effect = np.abs(wm - lm) / np.maximum(scale, 1e-8)
    effect = np.nan_to_num(effect, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(effect)[::-1]
    patterns = []
    for j in order[:40]:
        patterns.append({
            "feature": zq.FEATURE_COLS[int(j)],
            "winner_median": float(wm[j]),
            "loser_median": float(lm[j]),
            "robust_effect": float(effect[j]),
            "winner_q25_q75": [float(x) for x in np.quantile(W[:, j], [0.25, 0.75])],
            "loser_q25_q75": [float(x) for x in np.quantile(L[:, j], [0.25, 0.75])],
        })

    result = {
        "version": 2, "updated_at": time.time(), "feature_count": int(zq.N_FEATURES),
        "winning_samples": len(winners), "losing_samples": len(losers),
        "total_usable_samples": len(parsed),
        "source": "closed tracked signals only; outcomes and net returns from later-candle evaluation",
        "method": "robust median difference scaled by pooled MAD; discovery only, not proof of predictive edge",
        "top_recurring_features": patterns,
    }
    os.makedirs(zq.CFG["DIR"], exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp, path)

    if closed_count > old_count:
        zq.tg("🧬 اكتشاف أنماط الصفقات الرابحة\\n"
              f"العينات الرابحة: {len(winners)} | الخاسرة للمقارنة: {len(losers)}\\n"
              "تم حفظ أهم 40 ميزة من سجل الإشارات المكتملة في winning_patterns.json. "
              "الأنماط مساعدة فقط ولا تُعد دليلًا على ربح مستقبلي قبل اختبار زمني مستقل.")
        paper._set("last_pattern_closed_count", closed_count)
    return {"status": "patterns_updated", "winning_samples": len(winners),
            "losing_samples": len(losers), "features_reported": len(patterns), "path": path}


def install(zq):
    _install_features(zq)
    _install_30_layer_net(zq)
    old_report = zq.backtest_report
    def backtest_report_30_layers(report):
        return old_report(report).replace("شبكة 20 طبقة", "شبكة Deep Learning من 30 طبقة")
    zq.backtest_report = backtest_report_30_layers
    old_scan = zq.scan
    def scan_with_pattern_discovery(brain, paper, universe=None, cycle=0, trade=True):
        result = old_scan(brain, paper, universe, cycle, trade)
        if trade:
            try:
                outcome = discover_winning_patterns(zq, paper)
                zq.log.info("winning-pattern discovery: %s", outcome)
            except Exception as exc:
                zq.log.warning("winning-pattern discovery skipped: %s", exc)
        return result
    zq.scan = scan_with_pattern_discovery
