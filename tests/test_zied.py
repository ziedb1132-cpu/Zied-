"""Offline tests: no network, no torch required (the net tests are skipped without torch).
Run:  python -m pytest -q      (or: python tests/test_zied.py)
"""
import os, sys, json, tempfile, time, types
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zied_quant as zq

# ---------------------------------------------------------------- helpers
END = int(time.time() // 86400) * 86400 * 1000            # candle grid anchor (ms)

def fake_candles(inst, bar, depth):
    step = zq.CFG["TF_SEC"][bar] * 1000
    n = int(depth)
    seed = (abs(hash(inst)) % 10_000) + 1
    rng = np.random.default_rng(seed + zq.CFG["TF_SEC"][bar])
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004 * (zq.CFG["TF_SEC"][bar] / 3600) ** 0.5, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    ts = END - (n - np.arange(n)) * step
    d = pd.DataFrame(dict(ts=ts.astype("int64"), o=o, h=h, l=l, c=c, vol=rng.uniform(1, 10, n) * 1000))
    d["avail"] = pd.to_datetime(d["ts"] + step, unit="ms")
    return d

def fake_frames(inst, depth):
    return {tf: fake_candles(inst, tf, depth if isinstance(depth, int) else depth[tf]) for tf in zq.TFS}

def tmp_env():
    d = tempfile.mkdtemp()
    for k, f in (("DB", "paper.db"), ("MODEL", "model.pt"), ("GBM", "gbm.pkl"),
                 ("META", "meta.json"), ("TRAINSET", "train.npz")):
        zq.CFG[k] = os.path.join(d, f)
    zq.CFG["DIR"] = d
    return d

def mute_telegram():
    sent = []
    zq.tg = lambda m: sent.append(m)
    return sent

# ---------------------------------------------------------------- features
def test_feature_count_and_names():
    assert zq.N_FEATURES == 200
    assert len(zq.NAMES) == 38 and len(zq.CTX_NAMES) == 10
    assert len(set(zq.FEATURE_COLS)) == 200

def test_live_vector_complete():
    fr = fake_frames("AAA-USDT", 300); btc = {"1H": fake_candles("BTC-USDT", "1H", 300),
                                              "4H": fake_candles("BTC-USDT", "4H", 300)}
    v = zq.live_vector(fr, btc)
    assert len(v) == 200 and int(v.isna().sum()) == 0

def test_no_lookahead():
    btc = {"1H": fake_candles("BTC-USDT", "1H", 800), "4H": fake_candles("BTC-USDT", "4H", 400)}
    fr = fake_frames("AAA-USDT", {"5m": 4000, "15m": 1500, "1H": 800, "4H": 400, "1D": 200})
    full = zq.aligned_matrix(fr, btc)
    t = full.dropna().index[len(full.dropna()) // 2]
    cut = {tf: d[d["avail"] <= t].reset_index(drop=True) for tf, d in fr.items()}
    cb = {k: d[d["avail"] <= t].reset_index(drop=True) for k, d in btc.items()}
    part = zq.aligned_matrix(cut, cb).loc[t]
    assert np.allclose(full.loc[t].values.astype(float), part.values.astype(float), equal_nan=True)

# ---------------------------------------------------------------- labels
def _brute(c, h, l, atr, i, H):
    a = min(max(atr, 0.004), 0.06); e = c[i]
    sl, tp = e * (1 - zq.CFG["SL_ATR"] * a), e * (1 + zq.CFG["TP_ATR"] * a)
    for k in range(1, H + 1):
        if l[i + k] <= sl:
            return -zq.CFG["SL_ATR"] * a - zq.CFG["BT_COST"], k
        if h[i + k] >= tp:
            return zq.CFG["TP_ATR"] * a - zq.CFG["BT_COST"], k
    return c[i + H] / e - 1 - zq.CFG["BT_COST"], H

def test_barrier_matches_bruteforce():
    rng = np.random.default_rng(1); n = 200; H = zq.CFG["BARRIER_H"]
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h, l = c * 1.004, c * 0.996; atr = np.full(n, 0.01)
    r, hd = zq.barrier_batch(c, h, l, atr)
    for i in range(0, n - H, 7):
        er, eh = _brute(c, h, l, 0.01, i, H)
        assert abs(r[i] - er) < 1e-9 and hd[i] == eh
    assert np.isnan(r[n - H:]).all()

def test_barrier_sl_first_on_tie_and_single():
    H = zq.CFG["BARRIER_H"]
    hi = np.full(H, 101.0); lo = np.full(H, 99.0); cl = np.full(H, 100.0)
    hi[2] = 110.0; lo[2] = 90.0                                  # both barriers in one candle
    r, lab, hold = zq.barrier_single(100.0, 0.01, hi, lo, cl)
    assert lab == 0 and r < 0 and hold == 3
    hi2 = np.full(H, 101.0); lo2 = np.full(H, 99.9); hi2[4] = 105.0
    r, lab, hold = zq.barrier_single(100.0, 0.01, hi2, lo2, cl)
    assert lab == 1 and hold == 5

# ---------------------------------------------------------------- universe
def test_stable_filter():
    for s in ("USDC", "DAI", "FDUSD", "TUSD", "PYUSD", "RLUSD", "USDG", "EURT"):
        assert zq.is_stable(s), s
    for s in ("BTC", "ETH", "SOL", "XRP"):
        assert not zq.is_stable(s), s

# ---------------------------------------------------------------- signal-only persistence
def test_signal_only_store_and_commands():
    tmp_env(); mute_telegram()
    p = zq.Paper(); b = zq.Brain(persist=False)
    feats = np.zeros(zq.N_FEATURES)
    p.log_signals([dict(inst="AAA-USDT", price=100.0, atr=0.01, feats=feats, prob=0.6, rank=1)])
    assert p.count("signals") == 1
    assert p.count("signals", "labeled=0") == 1
    # No virtual portfolio, trade, or closed-trade learning tables are retained.
    tables = {r["name"] for r in p.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "signals" in tables and "meta" in tables
    assert not ({"trades", "pending_orders", "experience", "tracked_signals"} & tables)
    zq.handle_command("/pause", p, b); assert p._get("paused") == "1"
    zq.handle_command("/resume@mybot", p, b); assert p._get("paused") == "0"
    assert "سجل الإشارات" in zq.handle_command("/signals", p, b)
    assert "/positions" not in zq.handle_command("/help", p, b)

# ---------------------------------------------------------------- data + models
def _patch_network(monkey=True):
    zq.get_candles = fake_candles
    zq.CFG["HIST_DAYS"] = 14
    zq.CFG["TRAIN_TOP"] = 8

def _universe():
    names = [f"C{i}-USDT" for i in range(8)]
    return pd.DataFrame(dict(vol=np.linspace(9e6, 1e6, 8), last=100.0, ask=100.1, spread=0.001), index=names)

def test_trainset_and_fit_and_backtest():
    tmp_env(); mute_telegram(); _patch_network()
    zq.fetch_btc = lambda a, b: {"1H": fake_candles("BTC-USDT", "1H", a), "4H": fake_candles("BTC-USDT", "4H", b)}
    ds = zq.build_trainset(_universe())
    assert ds["X"].shape[1] == 200 and len(ds["y"]) > 1500 and not np.isnan(ds["X"]).any()
    assert set(np.unique(ds["y"])) <= {0.0, 1.0}
    assert zq.load_trainset() is not None
    b = zq.Brain(persist=True)
    info = b.fit(ds["X"], ds["y"], ds["ts"], ds["ret"], epochs=3, force=True)
    assert b.ready and info["mode"] in ("nn", "gbm", "ens")
    p = b.predict(ds["X"][:5]); assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()
    b2 = zq.Brain(persist=True); assert b2.ready and b2.mode == b.mode          # save/load round trip
    rep = zq.walk_forward(ds, folds=3, epochs=2)
    assert rep and all(0 <= r["auc"] <= 1 for r in rep.values())
    assert "Backtest" in zq.backtest_report(rep)

def test_simulate_respects_limits():
    n = 40
    ds = dict(ts=np.arange(n) * 60.0, ret=np.full(n, 0.01), hold=np.full(n, 24.0),
              coin=np.array([f"C{i % 3}" for i in range(n)]))
    r = zq.simulate(np.arange(n), ds)
    assert r["n"] <= zq.CFG["MAX_OPEN"] and r["n"] == 3                # only 3 distinct coins can be open

def test_label_signals_end_to_end():
    tmp_env(); mute_telegram(); _patch_network()
    p = zq.Paper(); H = zq.CFG["BARRIER_H"]
    old = time.time() - (H + 6) * 3600
    p.log_signals([dict(inst="AAA-USDT", price=100.0, atr=0.01, feats=np.zeros(zq.N_FEATURES), prob=0.6, rank=1)])
    p.db.execute("UPDATE signals SET ts=?", (old,)); p.db.commit()
    zq.get_candles = lambda inst, bar, depth: (lambda d: d.assign(ts=d["ts"] - 0))(fake_candles(inst, bar, depth))
    # make candles start after the signal time
    def shifted(inst, bar, depth):
        d = fake_candles(inst, bar, depth); d["ts"] = (old * 1000 + (np.arange(len(d)) + 1) * 3_600_000).astype("int64"); return d
    zq.get_candles = shifted
    assert zq.label_signals(p) == 1
    ds = p.signal_dataset(); assert len(ds["y"]) == 1 and ds["hold"][0] >= 1


def test_scan_and_once_pipeline():
    tmp_env(); sent = mute_telegram(); _patch_network()
    uni = _universe()
    zq.get_universe = lambda: uni
    zq.fetch_frames = lambda inst, depth: fake_frames(inst, depth)
    zq.fetch_btc = lambda a, b: {"1H": fake_candles("BTC-USDT", "1H", a), "4H": fake_candles("BTC-USDT", "4H", b)}
    zq.get_prices = lambda: {i: 100.0 for i in uni.index}
    zq.poll_commands = lambda *a, **k: 0
    zq.CFG["SCAN_DEPTH"] = 300
    zq.CFG["BTC_FILTER"] = -1.0  # synthetic test candles should not trip the live BTC gate
    b = zq.Brain(persist=False); p = zq.Paper()
    R = zq.scan(b, p, uni, 0, trade=True)                       # no model yet: ranking + signal logging only
    assert len(R) > 0 and R["rank"].is_monotonic_increasing and p.count("signals") == min(zq.CFG["LOG_TOP"], len(R))
    ds = zq.build_trainset(uni)
    b.fit(ds["X"], ds["y"], ds["ts"], ds["ret"], epochs=2, force=True)
    zq.CFG["PROB_MIN"] = 0.0                                   # force signals to exercise the entry path
    R = zq.scan(b, p, uni, 1, trade=True)
    assert R["prob"].notna().sum() > 0 and p.count("signals") > 0
    assert p.count("signals") > 0  # scan records signals only; it must not create virtual positions
    zq.CFG["PROB_MIN"] = None
    zq.once()                                                  # full scheduler cycle must not raise
    assert any("إشارة شراء" in m for m in sent)

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:                                   # noqa
                import traceback; traceback.print_exc(); print("FAIL", name, e); fails += 1
    sys.exit(1 if fails else 0)

# ---------------------------------------------------------------- bootstrap training retry policy
class _BootstrapPaper:
    def __init__(self):
        self.meta = {}
        self.experience = 0
        self.labelled = 0

    def _get(self, key):
        return self.meta.get(key)

    def _set(self, key, value):
        self.meta[key] = str(value)

    def count(self, table, where="1=1"):
        if table == "experience":
            return self.experience
        if table == "signals" and where == "labeled=1":
            return self.labelled
        raise AssertionError((table, where))



def test_chronological_boundary_keeps_equal_timestamps_together():
    ts = np.array([1, 2, 3, 3, 3, 4, 5, 5, 6], dtype=float)
    # A nominal row split in the middle of timestamp 3 must move to its first row.
    boundary = zq.chronological_boundary(ts, 4)
    assert boundary == 2
    assert ts[boundary - 1] < ts[boundary]
    # A boundary at the start of a timestamp group remains unchanged.
    assert zq.chronological_boundary(ts, 5) == 5
    # The end boundary is valid and does not index beyond the array.
    assert zq.chronological_boundary(ts, len(ts)) == len(ts)


def test_bootstrap_training_retry_is_persistently_throttled(monkeypatch):
    # With no durable cache, retain the normal 24-hour cooldown.
    monkeypatch.setattr(zq, "load_trainset", lambda: None)
    paper = _BootstrapPaper()
    assert zq.bootstrap_training_due(paper, now=1000)
    zq.mark_bootstrap_training_attempt(paper, now=1000)
    assert not zq.bootstrap_training_due(paper, now=1001)
    assert not zq.bootstrap_training_due(paper, now=1000 + 86399)
    assert zq.bootstrap_training_due(paper, now=1000 + 86400)


def test_bootstrap_training_retries_after_enough_new_signal_labels():
    paper = _BootstrapPaper()
    zq.mark_bootstrap_training_attempt(paper, now=1000)
    paper.labelled = int(zq.CFG["RETRAIN_SIGNALS"])
    assert zq.bootstrap_training_due(paper, now=1001)

def test_bootstrap_training_retries_hourly_with_sufficient_durable_cache(monkeypatch):
    rows = 1200
    cache = {
        "X": np.zeros((rows, zq.N_FEATURES), dtype=np.float32),
        "y": np.array([0, 1] * (rows // 2), dtype=np.float32),
        "ts": np.arange(rows, dtype=np.float64),
    }
    monkeypatch.setattr(zq, "load_trainset", lambda: cache)
    paper = _BootstrapPaper()
    zq.mark_bootstrap_training_attempt(paper, now=1000)

    assert not zq.bootstrap_training_due(paper, now=1000 + 3599)
    assert zq.bootstrap_training_due(paper, now=1000 + 3600)

