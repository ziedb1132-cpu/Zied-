#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative walk-forward shadow learner for the complete paper exit policy.

It compares SL, TP, trailing activation/distance and time-exit horizons on the
same matured 5m candle paths. Promotion is blocked until the main 500-win /
500-loss closed-outcome gate is satisfied. Existing open signals are untouched.
"""
import time
import numpy as np

MIN_TRAIN = 50
MIN_TEST = 20
BASELINE_NAME = "current"
# (name, SL multiplier adjustment, TP ATR, trailing activation ATR,
#  trailing distance ATR, maximum holding in 5m bars)
POLICIES = (
    ("current",       0.00, 3.0, 1.5, 1.8, 288),
    ("tight_target",  0.00, 2.5, 1.2, 1.6, 144),
    ("wide_target",   0.00, 4.0, 2.0, 2.0, 288),
    ("early_trail",   0.00, 3.5, 1.2, 1.5, 288),
    ("wide_trail",    0.00, 4.0, 1.8, 2.2, 288),
    ("faster_time",   0.00, 3.0, 1.5, 1.8, 144),
    ("sl_tighter",   -0.20, 3.0, 1.5, 1.8, 288),
    ("sl_wider",      0.20, 3.0, 1.5, 1.8, 288),
)


def _base_sl(atr):
    return 1.8 if atr < 0.012 else (2.0 if atr < 0.025 else 2.2)


def _regime(atr):
    return "LOW" if atr < 0.012 else ("MID" if atr < 0.025 else "HIGH")


def _ensure(paper):
    with paper.lock:
        paper.db.executescript("""
        CREATE TABLE IF NOT EXISTS exit_policy_shadow_v2(
          tracked_id INTEGER NOT NULL, regime TEXT NOT NULL, policy TEXT NOT NULL,
          net_ret REAL NOT NULL, reason TEXT NOT NULL, hold_bars INTEGER NOT NULL,
          opened_ts REAL NOT NULL,
          PRIMARY KEY(tracked_id,policy));
        CREATE INDEX IF NOT EXISTS idx_exit_policy_v2
          ON exit_policy_shadow_v2(regime,opened_ts);
        CREATE TABLE IF NOT EXISTS exit_policy_champion_v2(
          regime TEXT PRIMARY KEY, policy TEXT NOT NULL, train_n INTEGER NOT NULL,
          test_n INTEGER NOT NULL, test_avg REAL NOT NULL, test_pf REAL NOT NULL,
          baseline_avg REAL NOT NULL, updated_ts REAL NOT NULL);
        """)
        paper.db.commit()


def _simulate(entry, atr, highs, lows, closes, cost, sl_mult, tp_mult,
              activation_atr, distance_atr, max_bars):
    sl = entry * (1.0 - sl_mult * atr)
    tp = entry * (1.0 + tp_mult * atr)
    peak = entry
    trail = sl
    active = False
    last = min(len(closes), int(max_bars))
    for i in range(last):
        hi, lo = float(highs[i]), float(lows[i])
        # Stops active before this candle are checked first; same-candle
        # ambiguity is resolved conservatively in favour of the stop.
        active_sl = max(sl, trail)
        if lo <= active_sl:
            px, reason = active_sl, ("TRAIL" if active_sl > sl + 1e-12 else "SL")
            hold = i + 1
            break
        if hi >= tp:
            px, reason, hold = tp, "TP", i + 1
            break
        peak = max(peak, hi)
        if not active and peak >= entry * (1.0 + max(activation_atr * atr, 0.01)):
            active = True
        if active:
            trail = max(trail, entry * 1.01, peak * (1.0 - distance_atr * atr))
    else:
        if last <= 0:
            return None
        px, reason, hold = float(closes[last - 1]), "TIME", last
    net = px / entry - 1.0 - float(cost)
    if not np.isfinite(net):
        return None
    return float(net), reason, int(hold)


def _pf(values):
    a = np.asarray(values, dtype=float)
    gains = float(a[a > 0].sum())
    losses = float(-a[a < 0].sum())
    return gains / losses if losses > 0 else (999.0 if gains > 0 else 0.0)


def _promote(paper, zq):
    gate = zq.learning_gate_status(paper)
    if not gate["ready"]:
        zq.log.info("AI exit-policy gate closed: wins=%d/500 losses=%d/500",
                    gate["wins"], gate["losses"])
        return []
    promoted = []
    with paper.lock:
        regimes = [r[0] for r in paper.db.execute(
            "SELECT DISTINCT regime FROM exit_policy_shadow_v2").fetchall()]
        for regime in regimes:
            rows = paper.db.execute(
                "SELECT tracked_id,opened_ts,policy,net_ret FROM exit_policy_shadow_v2 "
                "WHERE regime=? ORDER BY opened_ts,tracked_id", (regime,)).fetchall()
            ids = sorted({int(r["tracked_id"]) for r in rows},
                         key=lambda sid: min(float(x["opened_ts"]) for x in rows
                                             if int(x["tracked_id"]) == sid))
            if len(ids) < MIN_TRAIN + MIN_TEST:
                continue
            split = int(len(ids) * 0.80)
            train_ids, test_ids = set(ids[:split]), set(ids[split:])
            if len(train_ids) < MIN_TRAIN or len(test_ids) < MIN_TEST:
                continue
            by_policy = {}
            for row in rows:
                by_policy.setdefault(row["policy"], {})[int(row["tracked_id"])] = float(row["net_ret"])
            baseline = by_policy.get(BASELINE_NAME, {})
            base_test = [baseline[i] for i in test_ids if i in baseline]
            if len(base_test) < MIN_TEST:
                continue
            trials = []
            for name, values in by_policy.items():
                train = [values[i] for i in train_ids if i in values]
                test = [values[i] for i in test_ids if i in values]
                if len(train) >= MIN_TRAIN and len(test) >= MIN_TEST:
                    trials.append((float(np.mean(train)), name, test))
            if not trials:
                continue
            _, winner, test = max(trials, key=lambda x: x[0])
            test_avg, test_pf = float(np.mean(test)), _pf(test)
            baseline_avg = float(np.mean(base_test))
            # Do not promote merely for beating the baseline by noise.
            if (winner == BASELINE_NAME or test_avg <= 0 or test_pf <= 1.05
                    or test_avg <= baseline_avg + 0.001):
                continue
            policy = next(p for p in POLICIES if p[0] == winner)
            paper.db.execute(
                "INSERT OR REPLACE INTO exit_policy_champion_v2 "
                "(regime,policy,train_n,test_n,test_avg,test_pf,baseline_avg,updated_ts) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (regime, winner, len(train_ids), len(test), test_avg, test_pf,
                 baseline_avg, time.time()))
            promoted.append((regime, winner, len(test), test_avg, test_pf, baseline_avg))
        paper.db.commit()
    for regime, name, n, avg, pf, base in promoted:
        zq.log.info("AI EXIT POLICY promoted regime=%s policy=%s holdout=%d avg=%.4f PF=%.2f baseline=%.4f",
                    regime, name, n, avg, pf, base)
        zq.tg("🧠 اعتماد سياسة مخارج جديدة بعد اختبار مستقل\n"
              f"تقلب السوق: {regime} | السياسة: {name}\n"
              f"اختبار: {n} إشارة | صافي متوسط: {avg*100:+.2f}% | PF: {pf:.2f}\n"
              f"الأساس الحالي: {base*100:+.2f}%")
    return promoted


def _evaluate(paper, zq):
    _ensure(paper)
    gate = zq.learning_gate_status(paper)
    if not gate["ready"]:
        return 0
    horizon = max(p[5] for p in POLICIES)
    # Keep a 25-hour grace window so hourly / two-hour Actions schedules can
    # still replay the complete 24-hour path from the retained 5m candles.
    replay_depth = horizon + 300
    now = time.time()
    with paper.lock:
        pending = [dict(r) for r in paper.db.execute(
            "SELECT t.id,t.inst,t.opened_ts,t.entry,t.atr_pct FROM tracked_signals t "
            "WHERE t.status='CLOSED' AND t.opened_ts>=? AND t.opened_ts<=? AND NOT EXISTS "
            "(SELECT 1 FROM exit_policy_shadow_v2 e WHERE e.tracked_id=t.id) "
            "ORDER BY t.opened_ts LIMIT 300",
            (now - replay_depth * 300, now - (horizon + 3) * 300)).fetchall()]
    if not pending:
        return 0
    grouped = {}
    for row in pending:
        grouped.setdefault(row["inst"], []).append(row)
    evaluated = 0
    for inst, signals in grouped.items():
        try:
            candles = zq.get_candles(inst, "5m", replay_depth + 20)
            if candles is None or candles.empty:
                continue
        except Exception as exc:
            zq.log.warning("exit-policy candle fetch failed %s: %s", inst, exc)
            continue
        for signal in signals:
            try:
                path = zq.future_candles_for_signal(candles, signal["opened_ts"], "5m")
                path = path[path["ts"].astype(float) + 300000 <= now * 1000]
                if len(path) < horizon:
                    continue
                path = path.iloc[:horizon]
                atr = float(np.clip(signal["atr_pct"], 0.004, 0.06))
                base_sl = _base_sl(atr)
                regime = _regime(atr)
                records = []
                for name, sl_delta, tp, activation, distance, max_bars in POLICIES:
                    sl_mult = min(2.4, max(1.4, base_sl + sl_delta))
                    result = _simulate(float(signal["entry"]), atr,
                        path["h"].to_numpy(), path["l"].to_numpy(),
                        path["c"].to_numpy(), zq.CFG["BT_COST"],
                        sl_mult, tp, activation, distance, max_bars)
                    if result is not None:
                        net, reason, hold = result
                        records.append((int(signal["id"]), regime, name, net,
                                        reason, hold, float(signal["opened_ts"])))
                if records:
                    with paper.lock:
                        paper.db.executemany(
                            "INSERT OR IGNORE INTO exit_policy_shadow_v2 "
                            "(tracked_id,regime,policy,net_ret,reason,hold_bars,opened_ts) "
                            "VALUES(?,?,?,?,?,?,?)", records)
                        paper.db.commit()
                    evaluated += 1
            except Exception as exc:
                zq.log.warning("exit-policy evaluation failed id=%s: %s", signal["id"], exc)
    if evaluated:
        zq.log.info("AI exit-policy shadow evaluation: signals=%d policies=%d",
                    evaluated, len(POLICIES))
        _promote(paper, zq)
    return evaluated


def _selected(paper, atr):
    regime = _regime(atr)
    with paper.lock:
        row = paper.db.execute(
            "SELECT policy FROM exit_policy_champion_v2 WHERE regime=?", (regime,)).fetchone()
    name = row["policy"] if row else BASELINE_NAME
    return next((p for p in POLICIES if p[0] == name), POLICIES[0])


def install(zq):
    if getattr(zq, "_exit_policy_v2_installed", False):
        return
    original_create = zq.Paper.create_tracked_signal

    def create_with_policy(self, row):
        _ensure(self)
        signal = original_create(self, row)
        if signal is None:
            return None
        atr = float(np.clip(float(row["atr"]), 0.004, 0.06))
        name, sl_delta, tp, activation, distance, max_bars = _selected(self, atr)
        sl_mult = min(2.4, max(1.4, _base_sl(atr) + sl_delta))
        entry = float(signal["entry"])
        sl = entry * (1.0 - sl_mult * atr)
        target = entry * (1.0 + tp * atr)
        with self.lock:
            self.db.execute(
                "UPDATE tracked_signals SET sl=?,tp=?,exit_sl_mult=?,exit_tp_mult=?,"
                "trail_activation_atr=?,trail_distance_atr=?,max_hold_bars=? "
                "WHERE id=? AND status='OPEN'",
                (sl, target, sl_mult, tp, activation, distance, max_bars, signal["id"]))
            self.db.commit()
        signal.update(sl=sl, tp=target, exit_policy=name, sl_mult=sl_mult,
                      tp_mult=tp, trail_activation_atr=activation,
                      trail_distance_atr=distance, max_hold_bars=max_bars)
        zq.log.info("paper exit policy=%s regime=%s SL=%.2f ATR TP=%.2f ATR trail=%.2f/%.2f ATR time=%dh",
                    name, _regime(atr), sl_mult, tp, activation, distance,
                    max_bars * 5 // 60)
        return signal

    original_monitor = zq.monitor_tracked_signals

    def monitor_and_learn(paper):
        try:
            _evaluate(paper, zq)
        except Exception as exc:
            zq.log.warning("AI exit-policy learning skipped safely: %s", exc)
        return original_monitor(paper)

    zq.Paper.create_tracked_signal = create_with_policy
    zq.monitor_tracked_signals = monitor_and_learn
    zq._exit_policy_v2_installed = True
