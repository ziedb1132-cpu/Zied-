#!/usr/bin/env python3
"""Persistent, conservative counterfactual learning for paper-signal exits.

Learns SL/TP ATR multipliers from completed 1H paths. Existing open signals are
never changed; promoted settings apply only to newly-created paper signals.
"""
import math
import time
import numpy as np

# Include the current fixed settings as the baseline and a bounded, explainable grid.
CANDIDATES = (
    (1.00, 1.50), (1.00, 2.00),
    (1.25, 2.00), (1.25, 2.50),
    (1.50, 2.50), (1.50, 3.00), (1.50, 4.00),
    (2.00, 3.00), (2.00, 4.00), (2.00, 5.00),
    (2.50, 4.00), (2.50, 5.00),
)
BASELINE = (1.50, 3.00)
MIN_TRAIN = 40
MIN_TEST = 15
COST = None  # use the same round-trip cost as the main program


def volatility_regime(atr_pct):
    a = float(atr_pct)
    return "LOW" if a < 0.012 else ("MID" if a < 0.025 else "HIGH")


def simulate_exit(entry, atr_pct, highs, lows, closes, sl_mult, tp_mult, cost):
    """Replay long-only barriers; conservative SL-first handling for same-bar ties."""
    entry, atr_pct = float(entry), float(atr_pct)
    sl = entry * (1.0 - sl_mult * atr_pct)
    tp = entry * (1.0 + tp_mult * atr_pct)
    for i, (high, low) in enumerate(zip(highs, lows)):
        if float(low) <= sl:
            exit_price, reason = sl, "SL"
            hold = i + 1
            break
        if float(high) >= tp:
            exit_price, reason = tp, "TP"
            hold = i + 1
            break
    else:
        if len(closes) == 0:
            return None
        exit_price, reason, hold = float(closes[-1]), "TIME", len(closes)
    net = exit_price / entry - 1.0 - float(cost)
    if not np.isfinite(net):
        return None
    return {"net_ret": float(net), "hold": int(hold), "reason": reason}


def _profit_factor(values):
    x = np.asarray(values, dtype=float)
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    return gains / losses if losses > 0 else (999.0 if gains > 0 else 0.0)


def _ensure_tables(paper):
    with paper.lock:
        paper.db.executescript("""
        CREATE TABLE IF NOT EXISTS exit_candidate_results(
          tracked_id INTEGER NOT NULL, inst TEXT NOT NULL, opened_ts REAL NOT NULL,
          regime TEXT NOT NULL, sl_mult REAL NOT NULL, tp_mult REAL NOT NULL,
          net_ret REAL NOT NULL, hold REAL NOT NULL, reason TEXT NOT NULL,
          PRIMARY KEY(tracked_id, sl_mult, tp_mult));
        CREATE INDEX IF NOT EXISTS idx_exit_results_regime_time
          ON exit_candidate_results(regime, opened_ts);
        CREATE TABLE IF NOT EXISTS exit_champions(
          regime TEXT PRIMARY KEY, sl_mult REAL NOT NULL, tp_mult REAL NOT NULL,
          train_n INTEGER NOT NULL, test_n INTEGER NOT NULL,
          test_avg REAL NOT NULL, test_pf REAL NOT NULL, updated_ts REAL NOT NULL);
        """)
        paper.db.commit()


def _promote(paper, zq):
    """Tune on the older 80%; promote only when the newest 20% beats baseline."""
    promoted = []
    with paper.lock:
        regimes = [r[0] for r in paper.db.execute(
            "SELECT DISTINCT regime FROM exit_candidate_results").fetchall()]
        for regime in regimes:
            rows = paper.db.execute(
                "SELECT tracked_id,opened_ts,sl_mult,tp_mult,net_ret FROM exit_candidate_results "
                "WHERE regime=? ORDER BY opened_ts,tracked_id", (regime,)).fetchall()
            ids = sorted({int(r["tracked_id"]) for r in rows})
            if len(ids) < MIN_TRAIN + MIN_TEST:
                continue
            split = int(len(ids) * 0.80)
            train_ids, test_ids = set(ids[:split]), set(ids[split:])
            if len(train_ids) < MIN_TRAIN or len(test_ids) < MIN_TEST:
                continue
            by_pair = {}
            for r in rows:
                pair = (float(r["sl_mult"]), float(r["tp_mult"]))
                by_pair.setdefault(pair, {})[int(r["tracked_id"])] = float(r["net_ret"])
            base = by_pair.get(BASELINE, {})
            baseline_test = [base[i] for i in test_ids if i in base]
            if len(baseline_test) < MIN_TEST:
                continue
            trials = []
            for pair, values in by_pair.items():
                tr = [values[i] for i in train_ids if i in values]
                te = [values[i] for i in test_ids if i in values]
                if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
                    continue
                trials.append((float(np.mean(tr)), pair, te))
            if not trials:
                continue
            _, pair, test_values = max(trials, key=lambda v: v[0])
            test_avg = float(np.mean(test_values))
            test_pf = _profit_factor(test_values)
            baseline_avg = float(np.mean(baseline_test))
            # Require positive net results, PF > 1.05, and a meaningful holdout
            # improvement over the unchanged 1.5/3.0 ATR baseline.
            if (pair == BASELINE or test_avg <= 0 or test_pf <= 1.05
                    or test_avg <= baseline_avg + 0.0005):
                continue
            paper.db.execute(
                "INSERT OR REPLACE INTO exit_champions "
                "(regime,sl_mult,tp_mult,train_n,test_n,test_avg,test_pf,updated_ts) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (regime, pair[0], pair[1], len(train_ids), len(test_values),
                 test_avg, test_pf, time.time()))
            promoted.append((regime, pair[0], pair[1], len(test_values), test_avg, test_pf))
        paper.db.commit()
    for item in promoted:
        zq.log.info("AI EXIT PROMOTED regime=%s SL=%.2f ATR TP=%.2f ATR holdout_n=%d avg_net=%.4f PF=%.3f",
                    *item)
        zq.tg("🧠 AI طوّر إعدادات المخارج\n"
              f"حالة التقلب: {item[0]}\nSL: {item[1]:.2f} ATR | TP: {item[2]:.2f} ATR\n"
              f"اختبار مستقل: {item[3]} إشارة | متوسط صافي: {item[4]*100:+.2f}% | PF: {item[5]:.2f}")


def _evaluate_mature_signals(paper, zq):
    h = int(zq.CFG["BARRIER_H"])
    now = time.time()
    _ensure_tables(paper)
    with paper.lock:
        pending = [dict(r) for r in paper.db.execute(
            "SELECT t.id,t.inst,t.opened_ts,t.entry,t.atr_pct FROM tracked_signals t "
            "WHERE t.opened_ts<=? AND NOT EXISTS "
            "(SELECT 1 FROM exit_candidate_results e WHERE e.tracked_id=t.id) "
            "ORDER BY t.opened_ts LIMIT 500", (now - (h + 1) * 3600,)).fetchall()]
    if not pending:
        return 0
    grouped = {}
    for row in pending:
        grouped.setdefault(row["inst"], []).append(row)
    evaluated = 0
    for inst, signals in grouped.items():
        try:
            candles = zq.get_candles(inst, "1H", min(100, h + 10))
            if candles is None or candles.empty:
                continue
        except Exception as exc:
            zq.log.warning("exit learner candle fetch failed %s: %s", inst, exc)
            continue
        for signal in signals:
            try:
                path = zq.future_candles_for_signal(candles, signal["opened_ts"])
                # Exclude the currently-forming candle. Evaluate only after the full
                # configured horizon is available, avoiding partial future paths.
                path = path[path["ts"].astype(float) + 3600000 <= now * 1000]
                if len(path) < h:
                    continue
                path = path.iloc[:h]
                atr = float(np.clip(signal["atr_pct"], 0.004, 0.06))
                regime = volatility_regime(atr)
                results = []
                for sl_mult, tp_mult in CANDIDATES:
                    # Keep maximum initial stop distance bounded at 8%.
                    if sl_mult * atr > 0.08:
                        continue
                    outcome = simulate_exit(
                        signal["entry"], atr, path["h"].to_numpy(),
                        path["l"].to_numpy(), path["c"].to_numpy(),
                        sl_mult, tp_mult, zq.CFG["BT_COST"])
                    if outcome is not None:
                        results.append((signal["id"], inst, signal["opened_ts"], regime,
                                        sl_mult, tp_mult, outcome["net_ret"],
                                        outcome["hold"], outcome["reason"]))
                if results:
                    with paper.lock:
                        paper.db.executemany(
                            "INSERT OR IGNORE INTO exit_candidate_results "
                            "(tracked_id,inst,opened_ts,regime,sl_mult,tp_mult,net_ret,hold,reason) "
                            "VALUES(?,?,?,?,?,?,?,?,?)", results)
                        paper.db.commit()
                    evaluated += 1
            except Exception as exc:
                zq.log.warning("exit learner evaluation failed id=%s %s: %s",
                               signal["id"], inst, exc)
    if evaluated:
        zq.log.info("AI exit learner: evaluated=%d matured_signals=%d candidate_pairs=%d",
                    evaluated, len(pending), len(CANDIDATES))
        _promote(paper, zq)
    return evaluated


def _select(paper, atr_pct):
    regime = volatility_regime(atr_pct)
    with paper.lock:
        row = paper.db.execute(
            "SELECT sl_mult,tp_mult FROM exit_champions WHERE regime=?", (regime,)).fetchone()
    if row is None:
        # Match the main scanner's volatility-adaptive stop; keep the target unchanged.
        a = float(atr_pct)
        sl_mult = 1.8 if a < 0.012 else (2.0 if a < 0.025 else 2.2)
        return sl_mult, BASELINE[1]
    return float(row["sl_mult"]), float(row["tp_mult"])


def install(zq):
    if getattr(zq, "_exit_evolution_installed", False):
        return
    original_create = zq.Paper.create_tracked_signal

    def create_with_learned_exits(self, row):
        _ensure_tables(self)
        signal = original_create(self, row)
        if signal is None:
            return None
        atr = float(np.clip(float(row["atr"]), 0.004, 0.06))
        sl_mult, tp_mult = _select(self, atr)
        entry = float(signal["entry"])
        sl = entry * (1.0 - sl_mult * atr)
        tp = entry * (1.0 + tp_mult * atr)
        with self.lock:
            self.db.execute("UPDATE tracked_signals SET sl=?,tp=? WHERE id=? AND status='OPEN'",
                            (sl, tp, signal["id"]))
            self.db.commit()
        signal.update(sl=sl, tp=tp, sl_mult=sl_mult, tp_mult=tp_mult)
        zq.log.info("signal exits %s: regime=%s SL=%.2f ATR TP=%.2f ATR",
                    signal["inst"], volatility_regime(atr), sl_mult, tp_mult)
        return signal

    original_monitor = zq.monitor_tracked_signals

    def monitor_with_exit_learning(paper):
        try:
            _evaluate_mature_signals(paper, zq)
        except Exception as exc:
            zq.log.warning("AI exit learning skipped safely: %s", exc)
        return original_monitor(paper)

    zq.Paper.create_tracked_signal = create_with_learned_exits
    zq.monitor_tracked_signals = monitor_with_exit_learning
    zq._exit_evolution_installed = True
