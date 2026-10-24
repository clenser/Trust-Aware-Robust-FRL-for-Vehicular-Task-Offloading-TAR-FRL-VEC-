"""Confirmation of the exploratory 40 % norm-matched result on fresh seeds (pre-specified).

docs/DNH_DETECTOR.md section 9 reports, on 12 seeds and after the 24 % result was known, that
against norm-matched (stealth) sign-flip attackers at 40 % the label-free rule (``do_no_harm_v2``)
beats Multi-Krum and that Multi-Krum is worse than no defence.  This study re-tests exactly that
on 12 NEW seeds (the benchmark seeds + 1000; none overlaps section 9) with the tests fixed here,
before running:

  C1  completion, do_no_harm_v2 - Multi-Krum  > 0
  C2  completion, do_no_harm_v2 - FedAvg      > 0
  C3  completion, Multi-Krum   - FedAvg       < 0   (Multi-Krum is harmful)
Holm-corrected across the three.  A pooled 24-seed estimate (section 9 seeds + these) is reported
as secondary and is not part of the confirmation.

    python -m experiments.stealth_confirmation
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from experiments import experiment_runner as er
from experiments.hypothesis_study import holm, paired_test

RESULTS = os.path.join(er.PROJECT_ROOT, "results")
METHODS = ("fedavg", "do_no_harm_v2", "multi_krum")


def run(out_dir, n_seeds=12):
    meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    seeds = [s + 1000 for s in meta["seeds"]][:n_seeds]
    config = load_config()
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(7, os.cpu_count() or 2))))
    fl = config["federated_learning"]
    eval_config = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                                      seed_offset=fl.get("eval_seed_offset", 500_000))
    mob = config["mobility"].get("scenario", "highway")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stealth_confirmation.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for m in METHODS:
        if m in results and len(results[m].get("per_seed_results", [])) == len(seeds):
            print("  %s (cached)" % m, flush=True)
            continue
        t0 = time.time()
        results[m] = er.run_multiseed_experiment(seeds, config=config, eval_config=eval_config, attack_ratio=0.4,
                                                 mobility_mode=mob, method=m, poison_type="stealth_sign_flip")
        c = results[m].get("pooled_confusion_counts") or {}
        tpr = c["tp"] / max(1, c["tp"] + c["fn"]) if c else float("nan")
        fpr = c["fp"] / max(1, c["fp"] + c["tn"]) if c else float("nan")
        print("  %-14s %.1f%%  TPR %.2f FPR %.3f (%.0f s)" % (m, results[m]["completion_rate"]["mean"], tpr, fpr, time.time() - t0), flush=True)
        json.dump(results, open(path, "w"))
    R = results
    tests = [("C1", "v2 - Multi-Krum", R["do_no_harm_v2"], R["multi_krum"], +1),
             ("C2", "v2 - FedAvg", R["do_no_harm_v2"], R["fedavg"], +1),
             ("C3", "Multi-Krum - FedAvg", R["multi_krum"], R["fedavg"], -1)]
    rows = []
    for pid, desc, a, b, d in tests:
        t = paired_test(a, b)
        t.update(id=pid, description=desc, direction=d)
        rows.append(t)
    adj = holm([r["p"] for r in rows])
    for r, ap in zip(rows, adj):
        r["p_holm"] = float(ap)
        excl = (r["ci_low"] > 0) if r["direction"] > 0 else (r["ci_high"] < 0)
        r["verdict"] = "CONFIRMED" if (excl and ap < 0.05) else ("raw CI only" if excl else "not confirmed")
    json.dump({"seeds": seeds, "tests": rows}, open(os.path.join(out_dir, "stealth_confirmation_verdicts.json"), "w"), indent=1)
    for r in rows:
        print("%s %-22s %+6.2f pp [%+.2f, %+.2f] Holm p=%.3f %s" % (
            r["id"], r["description"], r["mean"], r["ci_low"], r["ci_high"], r["p_holm"], r["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "stealth_confirmation"))
    ap.add_argument("--seeds", type=int, default=12)
    a = ap.parse_args()
    run(a.out, a.seeds)
