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
MIN_PATTERN_TOTAL = 100
MIN_PATTERN_PER_CLASS = 1
WIN_THRESHOLD = MIN_PATTERN_PER_CLASS
LOSS_THRESHOLD = MIN_PATTERN_PER_CLASS


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
    # Load only independently approved symbolic features. Their names become part
    # of the model schema, so the cache/model version changes when the registry does.
    registry_path = os.path.join(zq.CFG["DIR"], "derived_strategy_features.json")
    try:
        with open(registry_path, "r", encoding="utf-8") as fh:
            registry = json.load(fh)
    except (OSError, ValueError):
        registry = {"features": []}
    approved = [r for r in registry.get("features", [])
                if isinstance(r, dict) and r.get("status") == "approved"]
    derived_names = sorted({
        r.get("feature_name") for r in approved
        if isinstance(r.get("feature_name"), str) and r.get("feature_name")
    })
    approved = [r for r in registry.get("features", [])
                if isinstance(r, dict) and r.get("status") == "approved"]
    # Bring the full causal feature library used by autonomous strategy research
    # into the model's input matrix, not only strategies that passed the approval gate.
    from autonomous_strategy_engine import _LIBRARY_FEATURES, build_features as build_discovery_features
    research_names = list(_LIBRARY_FEATURES)
    # Candlestick patterns are now explicit supervised inputs, not dashboard-only
    # diagnostics or a ranking adjustment. The same causal detector is used for
    # historical training rows and live inference rows.
    import candle_news_intelligence as candle_intel
    # Load newly discovered pattern names before freezing this process's model schema.
    # Names detected later in this run are persisted and become columns next startup.
    candle_intel.configure_pattern_registry(
        os.path.join(zq.CFG["DIR"], "candle_pattern_registry.json")
    )
    PATTERN_FEATURE_NAMES = candle_intel.PATTERN_FEATURE_NAMES
    pattern_feature_matrix = candle_intel.pattern_feature_matrix

    # Deduplicate the expanded schema while preserving stable feature order.
    new_names = list(dict.fromkeys(
        old_names + EXTRA_NAMES + research_names + derived_names + list(PATTERN_FEATURE_NAMES)
    ))

    def expanded_tf_features(df):
        base = old_tf_isolated(df)
        extra = _extra_features(df)
        extra.index = base.index
        try:
            candle_patterns = pattern_feature_matrix(df)
            if len(candle_patterns) != len(base):
                raise ValueError(
                    f"Candlestick pattern row mismatch: {len(candle_patterns)} != {len(base)}"
                )
            candle_patterns.index = base.index
        except Exception as exc:
            zq.log.warning("candlestick model features unavailable: %s", exc)
            candle_patterns = pd.DataFrame(
                np.nan, index=base.index, columns=PATTERN_FEATURE_NAMES
            )
        combined = pd.concat([base, extra, candle_patterns], axis=1)
        try:
            discovery = build_discovery_features(df)
            # The legacy feature function indexes rows by candle timestamps,
            # while the research builder preserves df's original RangeIndex.
            # Align the research matrix to the exact same candle timestamps
            # before reindexing columns; otherwise every research value becomes
            # NaN even though the features were computed successfully.
            if len(discovery) != len(base):
                raise ValueError(
                    f"Research feature row mismatch: {len(discovery)} != {len(base)}"
                )
            discovery.index = base.index
            # Research features are causal OHLCV transformations. Align to the same
            # closed-candle index; absent columns remain NaN and are removed by
            # the existing dataset's dropna, never filled with future data.
            # Assemble research and derived columns separately, then concatenate
            # once. Repeated DataFrame column insertion fragments memory and slows
            # feature construction without changing the feature values or schema.
            research_columns = {
                name: discovery[name].reindex(combined.index) if name in discovery else np.nan
                for name in research_names
            }
            if derived_names:
                from derived_strategy_features import approved_feature_matrix
                derived = approved_feature_matrix(discovery, registry)
                derived_columns = {
                    name: derived[name].reindex(combined.index) if name in derived else np.nan
                    for name in derived_names
                }
                research_columns.update(derived_columns)
            if research_columns:
                research_frame = pd.DataFrame(research_columns, index=combined.index)
                combined = pd.concat([combined, research_frame], axis=1)
        except Exception as exc:
            zq.log.warning("research/approved feature computation skipped: %s", exc)
            for name in research_names + derived_names:
                combined[name] = np.nan
        return combined.reindex(columns=new_names)
    zq.NAMES = new_names
    zq.TF_COLS = [f"{tf}_{n}" for tf in zq.TFS for n in new_names]
    zq.CTX_COLS = [f"ctx_{n}" for n in zq.CTX_NAMES]
    zq.FEATURE_COLS = zq.TF_COLS + zq.CTX_COLS
    zq.N_FEATURES = len(zq.FEATURE_COLS)
    if zq.N_FEATURES < 300:
        raise RuntimeError(f"Expected at least 300 features, got {zq.N_FEATURES}")
    zq.tf_features = expanded_tf_features
    # Schema hash forces cached matrices and models to rebuild when approved features change.
    import hashlib
    schema = "|".join(zq.FEATURE_COLS).encode("utf-8")
    zq.DATA_VER = 4 + int(hashlib.sha256(schema).hexdigest()[:8], 16)
    zq.CFG["REQUIRED_SIGNAL_FEATURES"] = zq.N_FEATURES
    zq.CFG["TOP_AI"] = 100
    zq.CFG["LOG_TOP"] = 100
    zq.CFG["MAX_SIGNAL_RANK"] = 30
    zq.CFG["MAX_OPEN"] = 10
    # Smaller notional per position lets up to 10 slots fit within the existing 75% exposure cap.
    zq.CFG["POS_FRAC"] = 0.075
    zq.log.info(
        "primary AI feature schema installed: total=%d timeframes=%d genetic_library=%d_per_timeframe "
        "candlestick_patterns=%d_per_timeframe approved_derived=%d_per_timeframe",
        zq.N_FEATURES, len(zq.TFS), len(research_names), len(PATTERN_FEATURE_NAMES), len(derived_names),
    )


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
            # Scale each residual branch to keep a 30-linear-layer network stable
            # on wide, sparse feature vectors.
            return self.act(x + 0.1 * self.net(x))
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
        # Reject malformed rows before they can poison the optimizer or validation loss.
        Xf = np.asarray(Xf, dtype=np.float32)
        yf = np.asarray(yf, dtype=np.float32).reshape(-1)
        Xe = np.asarray(Xe, dtype=np.float32)
        ye = np.asarray(ye, dtype=np.float32).reshape(-1)
        if Xf.ndim != 2 or Xe.ndim != 2 or Xf.shape[1] != Xe.shape[1]:
            raise ValueError("30-layer network received incompatible feature matrices")
        fit_ok = np.isfinite(Xf).all(axis=1) & np.isfinite(yf) & np.isin(yf, (0.0, 1.0))
        val_ok = np.isfinite(Xe).all(axis=1) & np.isfinite(ye) & np.isin(ye, (0.0, 1.0))
        Xf, yf, Xe, ye = Xf[fit_ok], yf[fit_ok], Xe[val_ok], ye[val_ok]
        if len(Xf) < 20 or len(Xe) < 8:
            raise ValueError(
                f"30-layer network needs >=20 fit and >=8 validation rows; got {len(Xf)}/{len(Xe)}"
            )
        torch.manual_seed(0)
        Xt, yt = torch.from_numpy(Xf.copy()), torch.from_numpy(yf.copy())
        Xv, yv = torch.from_numpy(Xe.copy()), torch.from_numpy(ye.copy())
        if not (torch.isfinite(Xt).all() and torch.isfinite(yt).all()
                and torch.isfinite(Xv).all() and torch.isfinite(yv).all()):
            raise ValueError("30-layer network inputs remain non-finite after filtering")
        net = ZiedDeepNet(Xf.shape[1])
        assert n_linear(net) == 30, n_linear(net)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
        lossf = nn.BCEWithLogitsLoss()
        best, state, bad, valid_epochs, updates = float("inf"), None, 0, 0, 0
        for _ in range(max(1, int(epochs))):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 256):
                b = perm[i:i+256]
                if len(b) < 2:
                    continue
                opt.zero_grad(set_to_none=True)
                logits = net(Xt[b]).squeeze(1)
                loss = lossf(logits, yt[b])
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                if not torch.isfinite(grad_norm):
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                updates += 1
            sch.step()
            net.eval()
            with torch.no_grad():
                val_logits = net(Xv).squeeze(1)
                val_loss_tensor = lossf(val_logits, yv)
                val_loss = float(val_loss_tensor.item())
            if not np.isfinite(val_loss):
                # Do not checkpoint a numerically invalid network. GBM remains eligible.
                bad += 1
                if bad >= 2:
                    break
                continue
            valid_epochs += 1
            if val_loss < best - 1e-6:
                best, bad = val_loss, 0
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        if state is None:
            raise RuntimeError(
                "30-layer network produced no finite validation checkpoint "
                f"(valid_epochs={valid_epochs}, optimizer_updates={updates})"
            )
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
                "CASE WHEN s.ret>0 THEN 1 ELSE 0 END AS label, s.ret, "
                "s.feature_names, s.feature_count "
                "FROM signals s JOIN tracked_signals t ON t.signal_id=s.id "
                "WHERE t.status='CLOSED' AND s.labeled=1 AND s.ret IS NOT NULL "
                "ORDER BY t.closed_ts, s.id"
            ).fetchall()
    except sqlite3.Error as exc:
        zq.log.warning("pattern discovery could not read labelled signals: %s", exc)
        return {"status": "database_error", "error": str(exc)}

    parsed = []
    current_names = list(zq.FEATURE_COLS)
    current_index = {name: i for i, name in enumerate(current_names)}
    for feature_ts, feats_json, label, ret, names_json, feature_count in rows:
        try:
            raw_values = json.loads(feats_json)
            names = json.loads(names_json) if isinstance(names_json, str) else names_json
            label, ret = int(label), float(ret)
            if (not isinstance(raw_values, list) or not isinstance(names, list)
                    or len(raw_values) != len(names) or not names
                    or len(set(names)) != len(names)
                    or not all(isinstance(name, str) and name for name in names)
                    or (feature_count is not None and int(feature_count) != len(raw_values))):
                continue
            vector = np.full(zq.N_FEATURES, np.nan, dtype=np.float64)
            for name, value in zip(names, raw_values):
                idx = current_index.get(name)
                if idx is not None:
                    vector[idx] = np.nan if value is None else float(value)
            coverage = float(np.isfinite(vector).mean())
            stamp = float(feature_ts or 0.0)
            if (not np.isinf(vector).any() and coverage >= 0.80
                    and label in (0, 1) and np.isfinite(ret) and np.isfinite(stamp)):
                parsed.append((stamp, vector, label, ret))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue

    winners = [r for r in parsed if r[2] == 1 and r[3] > 0.0]
    losers = [r for r in parsed if r[2] == 0 and r[3] < 0.0]
    path = os.path.join(zq.CFG["DIR"], "winning_patterns.json")
    min_total = int(getattr(zq, "MIN_CLOSED_MODEL_SAMPLES", MIN_PATTERN_TOTAL))
    min_class = int(getattr(zq, "MIN_CLOSED_CLASS_SAMPLES", MIN_PATTERN_PER_CLASS))
    if len(winners) + len(losers) < min_total or len(winners) < min_class or len(losers) < min_class:
        # Old patterns must not remain active before the stricter evidence gate.
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                zq.log.warning("could not remove stale winning-pattern memory: %s", exc)
        return {"status": "waiting_for_closed_outcomes",
                "winning_samples": len(winners), "losing_samples": len(losers),
                "required_total": min_total, "required_per_class": min_class}

    old_count = int(paper._get("last_pattern_closed_count") or 0)
    closed_count = len(winners) + len(losers)
    if closed_count <= old_count and os.path.exists(path):
        return {"status": "no_new_closed_signal",
                "winning_samples": len(winners), "losing_samples": len(losers)}

    W = np.vstack([r[1] for r in winners])
    L = np.vstack([r[1] for r in losers])
    patterns = []
    for j, name in enumerate(zq.FEATURE_COLS):
        wv, lv = W[:, j][np.isfinite(W[:, j])], L[:, j][np.isfinite(L[:, j])]
        if not len(wv) or not len(lv):
            continue
        wm, lm = float(np.median(wv)), float(np.median(lv))
        pooled = np.concatenate([wv, lv])
        pooled_median = float(np.median(pooled))
        scale = float(np.median(np.abs(pooled - pooled_median)) * 1.4826)
        effect = abs(wm - lm) / max(scale, 1e-8)
        if not np.isfinite(effect):
            continue
        patterns.append({
            "feature": name, "winner_median": wm, "loser_median": lm,
            "robust_effect": float(effect),
            "winner_q25_q75": [float(x) for x in np.quantile(wv, [0.25, 0.75])],
            "loser_q25_q75": [float(x) for x in np.quantile(lv, [0.25, 0.75])],
        })
    patterns.sort(key=lambda item: item["robust_effect"], reverse=True)
    patterns = patterns[:40]

    result = {
        "version": 2, "updated_at": time.time(), "feature_count": int(zq.N_FEATURES),
        "winning_samples": len(winners), "losing_samples": len(losers),
        "total_usable_samples": len(parsed),
        "source": "closed tracked signals only; outcomes and net returns from later-candle evaluation",
        "method": "per-feature finite-value robust median difference scaled by pooled MAD; unknown values are retained as missing and excluded only from that feature statistic; discovery only, not proof of predictive edge",
        "minimum_total_samples": min_total, "minimum_per_class": min_class,
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
