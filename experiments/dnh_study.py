"""Closed-loop downstream study of the do-no-harm exclusion rule.

Pre-registered success criteria (fixed before running):
  (S1) at 0 % attackers: completion within 2 pp of FedAvg (paired mean; the
       deployed trust-aware rule lost ~10 pp in the active benchmark);
  (S2) at 24 % and 40 % attackers: completion not below the trust-aware rule and
       above undefended FedAvg (paired 95 % CI excluding zero at 40 %).
Cells use the benchmark's seeds, the same protocol and attacks for every method.

    python -m experiments.dnh_study [--seeds 12] [--ratios 0 0.24 0.4] [--poison TYPE --strength S] [--out DIR]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import stats

from configs.config_loader import load_config
from experiments import experiment_runner as er
from experiments.learner_study import paired

RESULTS = os.path.join(er.PROJECT_ROOT, "results")
METHODS = ("fedavg", "trust_aware", "do_no_harm")


def run(n_seeds, ratios, out_dir, poison=None, strength=None, extra=()):
    meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    seeds = list(meta["seeds"])[:n_seeds]
    config = load_config()
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(6, os.cpu_count() or 2))))
    fl = config["federated_learning"]
    eval_config = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                                      seed_offset=fl.get("eval_seed_offset", 500_000))
    os.makedirs(out_dir, exist_ok=True)
    tag = poison or "sign_flip"
    path = os.path.join(out_dir, "dnh_study_%s.json" % tag)
    results = json.load(open(path)) if os.path.exists(path) else {}
    for ratio in ratios:
        g = results.setdefault(str(ratio), {})
        for m in METHODS + tuple(extra):
            if m in g and len(g[m].get("per_seed_results", [])) == len(seeds):
                print("  r=%s %s (cached)" % (ratio, m), flush=True)
                continue
            t0 = time.time()
            kw = dict(attack_ratio=ratio, mobility_mode=config["mobility"].get("scenario", "highway"))
            if poison:
                kw["poison_type"] = poison
            if strength is not None:
                kw["poison_strength"] = strength
            g[m] = er.run_multiseed_experiment(seeds, method=m, config=config, eval_config=eval_config, **kw)
            print("  r=%s %-12s %.1f%%  (%.0f s)" % (ratio, m, g[m]["completion_rate"]["mean"], time.time() - t0), flush=True)
            json.dump(results, open(path, "w"))
    summary = {}
    for ratio, g in results.items():
        for m in g:
            if m == "fedavg":
                continue
            if "fedavg" in g:
                summary["r=%s %s - fedavg" % (ratio, m)] = paired(g[m], g["fedavg"])
        if "do_no_harm" in g and "trust_aware" in g:
            summary["r=%s do_no_harm - trust_aware" % ratio] = paired(g["do_no_harm"], g["trust_aware"])
        for m in g:   # detection quality
            c = g[m].get("pooled_confusion_counts")
            if c and (c["tp"] + c["fn"] + c["fp"] + c["tn"]):
                summary["r=%s %s detection" % (ratio, m)] = {
                    "tpr": c["tp"] / max(1, c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None,
                    "fpr": c["fp"] / max(1, c["fp"] + c["tn"]), **c}
    json.dump({"seeds": seeds, "poison": tag, "summary": summary}, open(os.path.join(out_dir, "dnh_summary_%s.json" % tag), "w"), indent=1)
    for k, v in summary.items():
        if "mean_pp" in v:
            print("%-40s %+6.1f pp  CI [%+.1f, %+.1f]  n=%d" % (k, v["mean_pp"], v["ci_low"], v["ci_high"], v["n"]))
        else:
            print("%-40s TPR %s FPR %.3f" % (k, v["tpr"], v["fpr"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.24, 0.4])
    ap.add_argument("--poison", default=None)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(RESULTS, "dnh_study"))
    a = ap.parse_args()
    run(a.seeds, a.ratios, a.out, a.poison, a.strength, tuple(a.extra))
