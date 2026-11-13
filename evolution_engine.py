#!/usr/bin/env python3
"""Conservative train/tune/holdout promotion policy for Zied Quant."""
import numpy as np

def trade_metrics(returns, selected):
    r = np.asarray(returns, dtype=np.float64)
    m = np.asarray(selected, dtype=bool)
    x = r[m & np.isfinite(r)]
    if not len(x):
        return {"n": 0, "avg": float("nan"), "pf": 0.0, "win": 0.0}
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    pf = gains / losses if losses > 0 else (999.0 if gains > 0 else 0.0)
    return {"n": int(len(x)), "avg": float(x.mean()), "pf": float(pf),
            "win": float((x > 0).mean())}

def holdout_edge_passes(selected, baseline, min_trades=8, min_pf=1.05, min_edge=0.001):
    """Require a sufficiently sampled, positive net edge over the unfiltered baseline."""
    if not isinstance(selected, dict) or not isinstance(baseline, dict):
        return False
    try:
        n = int(selected.get("n") or 0)
        avg = float(selected.get("avg"))
        pf = float(selected.get("pf"))
        base_avg = float(baseline.get("avg"))
    except (TypeError, ValueError):
        return False
    return bool(n >= int(min_trades) and np.isfinite([avg, pf, base_avg]).all()
                and avg > 0.0 and pf > float(min_pf)
                and avg >= base_avg + float(min_edge))


def chronological_train_tune_holdout_masks(ts, block, embargo):
    """Build chronological masks without splitting rows sharing one market timestamp."""
    zq = __import__("zied_quant")
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if not len(ts) or not np.isfinite(ts).all() or np.any(np.diff(ts) < 0):
        raise ValueError("timestamps must be finite and sorted")
    # Use timestamp-group boundaries: different coins from the same candle must
    # stay in the same partition, otherwise the holdout is not independent.
    test_start_i = zq.chronological_boundary(ts, len(ts) - block)
    tune_start_i = zq.chronological_boundary(ts, test_start_i - block)
    tune_start, test_start = ts[tune_start_i], ts[test_start_i]
    train_mask = ts < tune_start - embargo
    tune_mask = (ts >= tune_start) & (ts < test_start - embargo)
    test_mask = ts >= test_start
    return train_mask, tune_mask, test_mask


def fit_partition_requirements(n):
    """Minimum evidence and starting chronological block size for retraining."""
    n = int(n)
    if n < 100:
        raise ValueError(f"not enough verified closed-signal samples ({n}; need 100)")
    # Start with a 15% later block, then expand it dynamically if the 24h
    # outcome-horizon embargo leaves too few tuning rows. Never relax the embargo.
    block = max(int(n * 0.15), 15)
    return block, 40, 15, 15


def adaptive_chronological_partition(ts, initial_block, embargo,
                                      min_train=40, min_tune=15, min_test=15):
    """Expand later blocks until all post-embargo partitions are usable."""
    n = len(ts)
    best = None
    max_block = max(int(initial_block), int(n * 0.40))
    for block in range(int(initial_block), max_block + 1):
        masks = chronological_train_tune_holdout_masks(ts, block, embargo)
        actual = tuple(int(mask.sum()) for mask in masks)
        best = (block, masks, actual)
        # An 80/20 train/early-stop split must retain at least 30/10 rows.
        fit_rows = int(actual[0] * 0.80)
        early_rows = actual[0] - fit_rows
        if (actual[0] >= min_train and actual[1] >= min_tune
                and actual[2] >= min_test and fit_rows >= 30 and early_rows >= 10):
            return block, masks, actual
    actual = best[2] if best else (0, 0, 0)
    raise ValueError(
        f"train/tune/holdout block too small after 24h embargo: "
        f"best={actual}, need at least ({min_train}, {min_tune}, {min_test}) "
        f"with train fit/early-stop >=30/10; verified samples={n}"
    )


def evolved_fit(self, X, y, ts, ret=None, epochs=80, force=False):
    """Train on past data; tune on a later block; promote only after a holdout passes."""
    zq = __import__("zied_quant")
    np = zq.np
    order = np.argsort(ts, kind="stable")
    X = np.asarray(X, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float32)[order]
    ts = np.asarray(ts, dtype=np.float64)[order]
    ret = None if ret is None else np.asarray(ret, dtype=np.float64)[order]
    n = len(X)
    block, min_train, min_tune, min_test = fit_partition_requirements(n)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
        raise ValueError("closed-signal training requires real winning and losing examples")
    embargo = zq.CFG["BARRIER_H"] * 3600
    block, (train_mask, tune_mask, test_mask), actual = adaptive_chronological_partition(
        ts, block, embargo, min_train, min_tune, min_test
    )
    Xraw, ytr = X[train_mask], y[train_mask]
    ytune, ytest = y[tune_mask], y[test_mask]
    if any(len(np.unique(part)) < 2 for part in (ytr, ytune, ytest)):
        raise ValueError("chronological train/tune/holdout must each contain real wins and losses; collect more closed signals")
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(Xraw, axis=0), np.nanstd(Xraw, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    norm = zq.Brain.__new__(zq.Brain)
    norm.mu, norm.sd = mu, sd
    Xtr, Xtune, Xtest = norm._norm(Xraw), norm._norm(X[tune_mask]), norm._norm(X[test_mask])
    rtune = np.full(int(tune_mask.sum()), np.nan) if ret is None else ret[tune_mask]
    rtest = np.full(int(test_mask.sum()), np.nan) if ret is None else ret[test_mask]
    split = int(len(Xtr) * 0.80)
    if split < 30 or len(Xtr) - split < 10:
        raise ValueError(
            f"training split too small for early stopping: fit={split}, "
            f"early_stop={len(Xtr) - split}; need at least 30/10"
        )
    models, tune_p, test_p = {}, {}, {}
    gbm = zq.GBM()
    if gbm.fit(Xtr[:split], ytr[:split], Xtr[split:], ytr[split:]):
        models["gbm"] = gbm
        tune_p["gbm"], test_p["gbm"] = gbm.predict(Xtune), gbm.predict(Xtest)
    if zq.torch is not None:
        net = zq._train_net(Xtr[:split], ytr[:split], Xtr[split:], ytr[split:], epochs)
        models["nn"] = net
        tune_p["nn"], test_p["nn"] = zq._nn_predict(net, Xtune), zq._nn_predict(net, Xtest)
    if not tune_p:
        raise RuntimeError("no model available: install torch and/or lightgbm / scikit-learn")
    if len(tune_p) == 2:
        tune_p["ens"] = (tune_p["nn"] + tune_p["gbm"]) / 2
        test_p["ens"] = (test_p["nn"] + test_p["gbm"]) / 2
    tune_auc = {k: zq.auc_score(ytune, p) for k, p in tune_p.items()}
    # Evolve the confidence threshold using only the tuning block. The final
    # holdout is never used to select a threshold or model.
    thresholds, tune_stats = {}, {}
    min_tune_trades = max(5, int(np.ceil(len(rtune) * 0.20)))
    for name, probs in tune_p.items():
        trials = []
        for q in (0.70, 0.75, 0.80, 0.85, 0.90, 0.925):
            threshold = float(np.quantile(probs, q))
            metrics = trade_metrics(rtune, probs >= threshold)
            if metrics["n"] >= min_tune_trades and metrics["avg"] > 0 and metrics["pf"] > 1.05:
                trials.append((metrics["avg"], threshold, metrics))
        if trials:
            _, thresholds[name], tune_stats[name] = max(trials, key=lambda item: item[0])
        else:
            thresholds[name] = float(np.quantile(probs, 0.85))
            tune_stats[name] = trade_metrics(rtune, probs >= thresholds[name])
    eligible = [k for k in tune_p if tune_auc[k] >= 0.52 and tune_stats[k]["n"] >= min_tune_trades
                and tune_stats[k]["avg"] > 0 and tune_stats[k]["pf"] > 1.05]
    mode = max(eligible, key=lambda k: tune_stats[k]["avg"]) if eligible else max(tune_auc, key=tune_auc.get)
    hold_auc = {k: zq.auc_score(ytest, p) for k, p in test_p.items()}
    final = trade_metrics(rtest, test_p[mode] >= thresholds[mode])
    base_test = trade_metrics(rtest, np.ones(len(rtest), dtype=bool))
    info = {"n": int(n), "n_train": int(train_mask.sum()), "n_tune": int(tune_mask.sum()),
            "n_test": int(test_mask.sum()), "tune_aucs": tune_auc, "aucs": hold_auc,
            "mode": mode, "auc": hold_auc[mode], "thr": thresholds[mode],
            "min_tune_trades": int(min_tune_trades), "min_holdout_trades": 8,
            "tune_n": int(tune_stats[mode]["n"]), "tune_avg_net": float(tune_stats[mode]["avg"]),
            "tune_pf": float(tune_stats[mode]["pf"]), "test_n": int(final["n"]),
            "test_avg_net": float(final["avg"]), "test_pf": float(final["pf"]),
            "test_win_rate": float(final["win"]), "win_rate": float(y.mean()),
            "train_positive_rate": float(ytr.mean()), "tune_positive_rate": float(ytune.mean()),
            "test_positive_rate": float(ytest.mean()),
            "baseline_test_avg_net": float(base_test["avg"]) if np.isfinite(base_test["avg"]) else None,
            "gbm_kind": gbm.kind if "gbm" in models else None}
    info["val_all_net"] = float(np.nanmean(rtest)) if np.isfinite(rtest).any() else None
    info["val_top_net"] = float(final["avg"]) if np.isfinite(final["avg"]) else None
    zq.log.info(
        "training diagnostics: split(train/tune/holdout)=%d/%d/%d positive_rate=%.3f/%.3f/%.3f "
        "AUC(tune)=%s AUC(holdout)=%s baseline_holdout_net=%s selected_holdout_net=%s "
        "selected_trades=%d PF=%.3f threshold=%.5f",
        int(train_mask.sum()), int(tune_mask.sum()), int(test_mask.sum()),
        float(ytr.mean()), float(ytune.mean()), float(ytest.mean()),
        {k: round(float(v), 4) for k, v in tune_auc.items()},
        {k: round(float(v), 4) for k, v in hold_auc.items()},
        None if not np.isfinite(base_test["avg"]) else round(float(base_test["avg"]), 6),
        None if not np.isfinite(final["avg"]) else round(float(final["avg"]), 6),
        int(final["n"]), float(final["pf"]), float(thresholds[mode]))
    holdout_edge_ok = holdout_edge_passes(final, base_test)
    info["holdout_edge_ok"] = bool(holdout_edge_ok)
    info["holdout_edge_required"] = 0.001
    accepted = force or (hold_auc[mode] >= 0.52 and holdout_edge_ok
        and (self.auc is None or hold_auc[mode] >= self.auc - 0.03))
    info["accepted"] = bool(accepted)
    if accepted:
        self.nn, self.gbm = models.get("nn"), models.get("gbm")
        self.mu, self.sd, self.thrs, self.mode = mu, sd, thresholds, mode
        self.auc, self.info = hold_auc[mode], info
        if self.persist:
            self.save(X.shape[1])
    self.last_fit = {"info": info, "tune_p": tune_p, "test_p": test_p, "ytune": ytune, "ytest": ytest}
    return info

def install(module):
    module.Brain.fit = evolved_fit
