"""Fit and evaluate the do-no-harm detector on recorded real updates.

Training traces and evaluation traces are disjoint: the weights are fitted on a
fixed set of seeds of the plain sign-flip attack (plus honest-only runs), then
every detector is scored on held-out seeds and on attack families that were never
used for fitting.

    python -m experiments.fit_dnh TRACE_DIR [--out results/dnh_fit.json]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trust.dnh_detector import DoNoHarmDetector, CalibratedDetector, loo_features
from trust.model_trust import ModelTrustEvaluator
from trust import alt_detectors

FIT_SEEDS = (1000, 1001)           # plain sign-flip, ratios 0.0 / 0.24 / 0.40
MALICIOUS = ("poisoning", "free_rider")


def load(path):
    z = np.load(path)
    return z["round"], z["cid"], z["role"], z["delta"].astype(np.float64)


def rounds_of(path):
    rd, cid, role, delta = load(path)
    for r in sorted(set(rd.tolist())):
        m = np.where(rd == r)[0]
        yield r, [int(c) for c in cid[m]], [str(x) for x in role[m]], delta[m]


def fit_logistic(X, y, w_pos=1.0, w_neg=1.0, l2=1e-4):
    sw = np.where(y == 1, w_pos, w_neg)

    def f(b):
        z = X @ b[1:] + b[0]
        ll = sw * (np.logaddexp(0, z) - y * z)
        return ll.sum() / sw.sum() + l2 * (b[1:] ** 2).sum()

    res = minimize(f, np.zeros(X.shape[1] + 1), method="L-BFGS-B")
    return res.x


def training_table(files):
    X, y = [], []
    for f in files:
        for r, cids, roles, D in rounds_of(f):
            ratio, cos, norms, live = loo_features(D)
            for i, ro in enumerate(roles):
                if ro not in ("honest", "poisoning") or not live[i]:
                    continue
                X.append([np.log2(max(ratio[i], 1e-6)), -cos[i]])
                y.append(1 if ro == "poisoning" else 0)
    return np.array(X), np.array(y)


def calibrate_threshold(weights, files, quantile=0.995, ewma=0.5):
    """Threshold = a high quantile of the evidence honest clients reach (false-positive budget)."""
    ev_honest = []
    for f in files:
        det = DoNoHarmDetector(weights=weights, ewma=ewma, threshold=float("inf"))
        for r, cids, roles, D in rounds_of(f):
            _, ev = det.decide({c: d for c, d in zip(cids, D)})
            ev_honest += [ev[c] for c, ro in zip(cids, roles) if ro == "honest" and np.isfinite(ev.get(c, np.inf))]
    return float(np.quantile(ev_honest, quantile))


def honest_calibration(files, q=0.995):
    """Quantiles of the honest statistics on ATTACKER-FREE traces (no attack label is used)."""
    R, C = [], []
    for f in files:
        for r, cids, roles, D in rounds_of(f):
            ratio, cos, norms, live = loo_features(D)
            for i, ro in enumerate(roles):
                if ro == "honest" and live[i]:
                    R.append(ratio[i]); C.append(cos[i])
    return float(np.quantile(R, q)), float(np.quantile(C, 1.0 - q))


def run_calibrated(path, ratio_hi, cos_lo, **kw):
    det = CalibratedDetector(ratio_hi, cos_lo, **kw)
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for r, cids, roles, D in rounds_of(path):
        excl, _ = det.decide({c: d for c, d in zip(cids, D)})
        for k, v in confusion(excl, cids, roles).items():
            tot[k] += v
    return tot


def confusion(excl, cids, roles):
    c = dict(tp=0, fp=0, fn=0, tn=0)
    for cid, ro in zip(cids, roles):
        mal, ex = ro in MALICIOUS, cid in excl
        c["tp" if (mal and ex) else "fp" if ex else "fn" if mal else "tn"] += 1
    return c


def run_deployed(path):
    """Model-trust part of the deployed rule (joint trust < 0.4 at alpha = 0.5, behavioural prior 0.7)."""
    ev = ModelTrustEvaluator()
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for r, cids, roles, D in rounds_of(path):
        sc = ev.evaluate_model_updates({c: d for c, d in zip(cids, D)})
        # joint trust < 0.4; a model score <= 0.2 also drives behavioural trust down (see TrustEngine)
        excl = {c for c, s in sc.items() if s <= 0.2}
        for k, v in confusion(excl, cids, roles).items():
            tot[k] += v
    return tot


def run_dnh(path, **kw):
    det = DoNoHarmDetector(**kw)
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for r, cids, roles, D in rounds_of(path):
        excl, _ = det.decide({c: d for c, d in zip(cids, D)})
        for k, v in confusion(excl, cids, roles).items():
            tot[k] += v
    return tot


def run_shadow(path, fn):
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for r, cids, roles, D in rounds_of(path):
        upd = {c: d for c, d in zip(cids, D)}
        excl = fn(upd)
        for k, v in confusion(excl, cids, roles).items():
            tot[k] += v
    return tot


def rates(c):
    nm, nh = c["tp"] + c["fn"], c["fp"] + c["tn"]
    return (c["tp"] / nm if nm else None), (c["fp"] / nh if nh else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--out", default="results/dnh_fit.json")
    ap.add_argument("--w_neg", type=float, default=1.0)
    ap.add_argument("--quantile", type=float, default=0.995)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.trace_dir, "trace_*.npz")))
    files_sf = [f for f in files if os.path.basename(f).startswith("trace_sign_flip_")]
    fit_files = [f for f in files_sf if any("_s%d" % s in f for s in FIT_SEEDS)]
    X, y = training_table(fit_files)
    b = fit_logistic(X, y, w_neg=a.w_neg)
    weights = {"b0": float(b[0]), "b_ratio": float(b[1]), "b_cos": float(b[2])}
    weights["threshold"] = calibrate_threshold({k: weights[k] for k in ("b0", "b_ratio", "b_cos")}, fit_files, a.quantile)
    print("fit on %d rows (%d poisoned): %s" % (len(y), y.sum(), weights))
    held = [f for f in sorted(glob.glob(os.path.join(a.trace_dir, "trace_*.npz"))) if f not in fit_files]
    out = {"weights": weights, "fit_files": [os.path.basename(f) for f in fit_files], "eval": {}}
    honest_files = [f for f in files if "_r0.00_" in f and any("_s%d" % sd in f for sd in FIT_SEEDS)]
    R_hi, C_lo = honest_calibration(honest_files)
    out["calibration"] = {"ratio_hi": R_hi, "cos_lo": C_lo, "files": [os.path.basename(f) for f in honest_files]}
    print("label-free calibration on attacker-free runs: ratio_hi %.3f  cos_lo %.3f" % (R_hi, C_lo))
    detectors = {
        "deployed": run_deployed,
        "dnh_v2": lambda p: run_calibrated(p, R_hi, C_lo),
        "distance_mad": lambda p: run_shadow(p, alt_detectors.distance_mad_exclusion),
        "dnh": lambda p: run_dnh(p, weights={k: weights[k] for k in ("b0", "b_ratio", "b_cos")}, threshold=weights["threshold"]),
    }
    for f in held:
        name = os.path.basename(f)[6:-4]
        out["eval"][name] = {}
        line = []
        for dn, fn in detectors.items():
            c = fn(f)
            tpr, fpr = rates(c)
            out["eval"][name][dn] = dict(c, tpr=tpr, fpr=fpr)
            line.append("%s TPR %s FPR %s" % (dn, "N/A" if tpr is None else "%.2f" % tpr, "%.3f" % fpr))
        print("%-34s %s" % (name, " | ".join(line)), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
