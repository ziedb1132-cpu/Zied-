#!/usr/bin/env python3
"""Research-only discovery lab for missed signals, entry timing and historical analogues.

This module is deliberately isolated from live ranking, model weights and paper entries.
It writes only its own research tables and JSON report.
"""
import json
import os
import time
import sqlite3
import numpy as np

COST = 0.0035
HORIZON = 24
MIN_ANALOGUES = 8


def _ensure(paper):
    with paper.lock:
        paper.db.executescript("""
        CREATE TABLE IF NOT EXISTS discovery_missed(
          signal_id INTEGER PRIMARY KEY, inst TEXT NOT NULL, feature_ts REAL NOT NULL,
          rank INTEGER, prob REAL, label INTEGER, net_ret REAL, processed_ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS discovery_timing(
          signal_id INTEGER NOT NULL, delay_bars INTEGER NOT NULL,
          net_ret REAL NOT NULL, hold_bars INTEGER NOT NULL,
          PRIMARY KEY(signal_id,delay_bars));
        CREATE TABLE IF NOT EXISTS discovery_similarity(
          signal_id INTEGER PRIMARY KEY, nearest_count INTEGER NOT NULL,
          win_rate REAL NOT NULL, avg_net REAL NOT NULL, created_ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS discovery_experiments(
          experiment_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
          train_json TEXT, tune_json TEXT, test_json TEXT,
          status TEXT NOT NULL, created_ts REAL NOT NULL);
        """)
        paper.db.commit()


def _timing_replay(candles, signal_ts, entry_price, delays, cost):
    """Replay delayed entries only on candles strictly after the original signal timestamp."""
    if candles is None or candles.empty:
        return []
    d = candles.copy()
    d["ts"] = d["ts"].astype(float) / 1000.0
    d = d[d["ts"] >= float(signal_ts)].sort_values("ts").reset_index(drop=True)
    if len(d) < HORIZON + max(delays) + 1:
        return []
    out = []
    for delay in delays:
        # The signal was known at its feature close. Entry is the open of the
        # first candle after the requested delay; no same-candle look-ahead.
        i = int(delay)
        if i >= len(d):
            continue
        entry = float(d.iloc[i]["o"])
        if not np.isfinite(entry) or entry <= 0:
            continue
        path = d.iloc[i:min(i + HORIZON, len(d))]
        if len(path) < HORIZON:
            continue
        # Same conservative barriers as the primary strategy, evaluated at
        # the original signal ATR and existing configured multipliers.
        # ATR is supplied separately by caller in the signal row.
        out.append((delay, entry, path))
    return out


def _evaluate_timing(zq, paper, row):
    try:
        candles = zq.get_candles(row["inst"], "1H", min(120, HORIZON + 12))
        if candles is None or candles.empty:
            return 0
        atr = float(np.clip(float(row["atr"]), 0.004, 0.06))
        results = []
        for delay, entry, path in _timing_replay(candles, row["feature_ts"], row["price"], (0, 1, 2, 3), COST):
            sl = entry * (1 - zq.CFG["SL_ATR"] * atr)
            tp = entry * (1 + zq.CFG["TP_ATR"] * atr)
            exit_price, hold = float(path.iloc[-1]["c"]), len(path)
            for j, candle in enumerate(path.itertuples(index=False), 1):
                lo, hi = float(candle.l), float(candle.h)
                if lo <= sl:
                    exit_price, hold = sl, j
                    break
                if hi >= tp:
                    exit_price, hold = tp, j
                    break
            net = exit_price / entry - 1.0 - COST
            if np.isfinite(net):
                results.append((int(row["id"]), int(delay), float(net), int(hold)))
        if results:
            with paper.lock:
                paper.db.executemany(
                    "INSERT OR IGNORE INTO discovery_timing(signal_id,delay_bars,net_ret,hold_bars) VALUES(?,?,?,?)",
                    results)
                paper.db.commit()
            return 1
    except Exception as exc:
        zq.log.debug("discovery timing replay skipped signal %s: %s", row.get("id"), exc)
    return 0


def _missed_and_timing(zq, paper, limit=120):
    """Use matured logged candidates that were not opened as paper positions."""
    with paper.lock:
        rows = [dict(r) for r in paper.db.execute(
            "SELECT s.id,s.inst,COALESCE(s.feature_ts,s.ts) feature_ts,s.ts,s.price,s.atr,s.rank,s.prob,s.label,s.ret "
            "FROM signals s WHERE s.labeled=1 AND s.ret IS NOT NULL "
            "AND NOT EXISTS(SELECT 1 FROM tracked_signals t WHERE t.signal_id=s.id) "
            "AND NOT EXISTS(SELECT 1 FROM discovery_timing d WHERE d.signal_id=s.id) "
            "ORDER BY s.ts DESC LIMIT ?", (int(limit),)).fetchall()]
    done = 0
    for row in rows:
        try:
            with paper.lock:
                paper.db.execute(
                    "INSERT OR IGNORE INTO discovery_missed VALUES(?,?,?,?,?,?,?,?)",
                    (row["id"], row["inst"], row["feature_ts"], row["rank"], row["prob"],
                     int(row["label"]), float(row["ret"]), time.time()))
                paper.db.commit()
            _evaluate_timing(zq, paper, row)
            done += 1
        except Exception as exc:
            zq.log.warning("missed-signal lab row %s skipped: %s", row["id"], exc)
    return done


def _similarity(zq, paper, limit=80):
    """Nearest historical labelled feature vectors; exclude same signal and same timestamp."""
    with paper.lock:
        latest = [dict(r) for r in paper.db.execute(
            "SELECT id,inst,COALESCE(feature_ts,ts) feature_ts,feats FROM signals "
            "WHERE labeled=0 ORDER BY ts DESC LIMIT ?", (int(limit),)).fetchall()]
        history = paper.db.execute(
            "SELECT id,COALESCE(feature_ts,ts) feature_ts,feats,ret FROM signals "
            "WHERE labeled=1 AND ret IS NOT NULL AND ret != 0 ORDER BY ts DESC LIMIT 5000").fetchall()
    if len(history) < MIN_ANALOGUES:
        return 0
    parsed = []
    for h in history:
        try:
            x = np.asarray(json.loads(h["feats"]), dtype=float)
            if x.shape == (zq.N_FEATURES,) and np.isfinite(x).all():
                parsed.append((int(h["id"]), float(h["feature_ts"]), x, float(h["ret"])))
        except Exception:
            continue
    if len(parsed) < MIN_ANALOGUES:
        return 0
    H = np.vstack([x[2] for x in parsed])
    med = np.nanmedian(H, axis=0)
    scale = np.nanmedian(np.abs(H - med), axis=0) * 1.4826 + 1e-6
    Hn = np.clip((H - med) / scale, -8, 8)
    saved = 0
    for row in latest:
        try:
            q = np.asarray(json.loads(row["feats"]), dtype=float)
            if q.shape != (zq.N_FEATURES,) or not np.isfinite(q).all():
                continue
            qn = np.clip((q - med) / scale, -8, 8)
            dist = np.sqrt(np.mean((Hn - qn) ** 2, axis=1))
            order = np.argsort(dist)[:min(25, len(dist))]
            # Similarity threshold is relative to this query's nearest-neighbour distances.
            cutoff = max(float(np.quantile(dist[order], .75)), 0.05)
            chosen = [parsed[i] for i in order if dist[i] <= cutoff]
            if len(chosen) < MIN_ANALOGUES:
                chosen = [parsed[i] for i in order[:MIN_ANALOGUES]]
            rets = np.asarray([x[3] for x in chosen], dtype=float)
            with paper.lock:
                paper.db.execute(
                    "INSERT OR REPLACE INTO discovery_similarity VALUES(?,?,?,?,?)",
                    (int(row["id"]), len(rets), float((rets > 0).mean()),
                     float(rets.mean()), time.time()))
                paper.db.commit()
            saved += 1
        except Exception as exc:
            zq.log.debug("historical analogue skipped: %s", exc)
    return saved


def _report(paper, state_dir):
    with paper.lock:
        missed = paper.db.execute(
            "SELECT COUNT(*) n, SUM(label=1) wins, SUM(label=0) losses, AVG(net_ret) avg "
            "FROM discovery_missed").fetchone()
        timing = paper.db.execute(
            "SELECT delay_bars,COUNT(*) n,AVG(net_ret) avg,SUM(CASE WHEN net_ret>0 THEN 1 ELSE 0 END)*1.0/COUNT(*) win "
            "FROM discovery_timing GROUP BY delay_bars ORDER BY delay_bars").fetchall()
        sims = paper.db.execute(
            "SELECT COUNT(*) n,AVG(win_rate) win,AVG(avg_net) avg FROM discovery_similarity").fetchone()
    report = {
        "updated_at": time.time(),
        "mode": "research_only",
        "missed_signals": {k: (int(missed[k] or 0) if k in ("n","wins","losses") else
                               float(missed[k]) if missed[k] is not None else None)
                           for k in ("n","wins","losses","avg")},
        "entry_timing": [{"delay_bars": int(r["delay_bars"]), "samples": int(r["n"]),
                          "avg_net": float(r["avg"]), "win_rate": float(r["win"])} for r in timing],
        "historical_similarity": {"signals": int(sims["n"] or 0),
                                  "analogue_win_rate": float(sims["win"] or 0),
                                  "analogue_avg_net": float(sims["avg"] or 0)},
        "note": "Research only. No live signal, model, entry, exit, risk, or portfolio setting is modified."
    }
    os.makedirs(state_dir, exist_ok=True)
    tmp = os.path.join(state_dir, "strategy_discovery_lab.json.tmp")
    dst = os.path.join(state_dir, "strategy_discovery_lab.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp, dst)
    return report


def run_lab(zq, paper):
    """Run bounded research work; safe to call after each normal scan."""
    _ensure(paper)
    now = time.time()
    last = float(paper._get("discovery_lab_last_run") or 0)
    if now - last < 6 * 3600:
        return {"status": "skipped_recent"}
    missed_n = _missed_and_timing(zq, paper)
    similar_n = _similarity(zq, paper)
    report = _report(paper, zq.CFG["DIR"])
    paper._set("discovery_lab_last_run", now)
    zq.log.info("Zied Strategy Discovery Lab: missed=%d timing_evaluated=%d analogues=%d",
                missed_n, missed_n, similar_n)
    return {"status": "ok", "missed_processed": missed_n,
            "similarity_processed": similar_n, "report": report}
