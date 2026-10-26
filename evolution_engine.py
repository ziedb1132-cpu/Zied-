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
    pf = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
    return {"n": int(len(x)), "avg": float(x.mean()), "pf": float(pf),
            "win": float((x > 0).mean())}

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
    if n < 1200:
        raise ValueError(f"not enough samples for train/tune/holdout ({n}; need 1200)")
    block = max(int(n * 0.15), 200)
    tune_start, test_start = ts[n - 2 * block], ts[n - block]
    embargo = zq.CFG["BARRIER_H"] * 3600
    train_mask = ts < tune_start - embargo
    tune_mask = (ts >= tune_start) & (ts < test_start - embargo)
    test_mask = ts >= test_start
    if train_mask.sum() < 400 or tune_mask.sum() < 100 or test_mask.sum() < 100:
        raise ValueError("train/tune/holdout block too small after time embargo")
    Xraw, ytr = X[train_mask], y[train_mask]
    mu, sd = Xraw.mean(0), Xraw.std(0) + 1e-8
    norm = zq.Brain.__new__(zq.Brain)
    norm.mu, norm.sd = mu, sd
    Xtr, Xtune, Xtest = norm._norm(Xraw), norm._norm(X[tune_mask]), norm._norm(X[test_mask])
    ytune, ytest = y[tune_mask], y[test_mask]
    rtune = np.full(int(tune_mask.sum()), np.nan) if ret is None else ret[tune_mask]
    rtest = np.full(int(test_mask.sum()), np.nan) if ret is None else ret[test_mask]
    split = int(len(Xtr) * 0.85)
    if split < 300 or len(Xtr) - split < 50:
        raise ValueError("training split too small for early stopping")
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
    for name, probs in tune_p.items():
        trials = []
        for q in (0.70, 0.75, 0.80, 0.85, 0.90, 0.925):
            threshold = float(np.quantile(probs, q))
            metrics = trade_metrics(rtune, probs >= threshold)
            if metrics["n"] >= 30 and metrics["avg"] > 0 and metrics["pf"] > 1.05:
                trials.append((metrics["avg"], threshold, metrics))
        if trials:
            _, thresholds[name], tune_stats[name] = max(trials, key=lambda item: item[0])
        else:
            thresholds[name] = float(np.quantile(probs, 0.85))
            tune_stats[name] = trade_metrics(rtune, probs >= thresholds[name])
    eligible = [k for k in tune_p if tune_auc[k] >= 0.52 and tune_stats[k]["n"] >= 30
                and tune_stats[k]["avg"] > 0 and tune_stats[k]["pf"] > 1.05]
    mode = max(eligible, key=lambda k: tune_stats[k]["avg"]) if eligible else max(tune_auc, key=tune_auc.get)
    hold_auc = {k: zq.auc_score(ytest, p) for k, p in test_p.items()}
    final = trade_metrics(rtest, test_p[mode] >= thresholds[mode])
    info = {"n": int(n), "n_train": int(train_mask.sum()), "n_tune": int(tune_mask.sum()),
            "n_test": int(test_mask.sum()), "tune_aucs": tune_auc, "aucs": hold_auc,
            "mode": mode, "auc": hold_auc[mode], "thr": thresholds[mode],
            "tune_n": int(tune_stats[mode]["n"]), "tune_avg_net": float(tune_stats[mode]["avg"]),
            "tune_pf": float(tune_stats[mode]["pf"]), "test_n": int(final["n"]),
            "test_avg_net": float(final["avg"]), "test_pf": float(final["pf"]),
            "test_win_rate": float(final["win"]), "win_rate": float(y.mean()),
            "gbm_kind": gbm.kind if "gbm" in models else None}
    info["val_all_net"] = float(np.nanmean(rtest)) if np.isfinite(rtest).any() else None
    info["val_top_net"] = float(final["avg"]) if np.isfinite(final["avg"]) else None
    accepted = force or (hold_auc[mode] >= 0.52 and final["n"] >= 30
        and np.isfinite(final["avg"]) and final["avg"] > 0 and final["pf"] > 1.05
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
