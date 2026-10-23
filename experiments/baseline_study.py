"""H2 against the standard robust aggregators (pre-specified).

Roadmap H2: trust-aware aggregation gives higher task reliability and model
robustness than conventional aggregation under increasing malicious
participation.  The hypothesis study compared the do-no-harm rule with FedAvg;
this study adds the robust aggregators a reviewer will ask for.

Comparisons (do_no_harm minus baseline, completion, paired by seed), Holm-corrected
across all nine:
  B1-B3  vs Multi-Krum, coordinate median, trimmed mean at 40 % attackers (24 seeds)
  B4-B6  same at 24 % attackers (24 seeds)
  B7-B9  same with no attackers (12 seeds; the cost of each defence)
Reference cells for do_no_harm and FedAvg come from the hypothesis study (24 and 40 %)
and the do-no-harm study (0 %), which use the identical seeds, protocol and attack.
FLTrust is not included: it needs a trusted root dataset that the federation here
does not have (see docs/BASELINES.md).

    python -m experiments.baseline_study
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from configs.config_loader import load_config
from experiments import experiment_runner as er
from experiments.hypothesis_study import holm, paired_test

RESULTS = os.path.join(er.PROJECT_ROOT, "results")
BASELINES = ("multi_krum", "coordinate_median", "trimmed_mean")


def run(out_dir, baselines=BASELINES, tag="baseline"):
    hyp = json.load(open(os.path.join(RESULTS, "hypothesis_study", "hypothesis_study.json")))
    dnh0 = json.load(open(os.path.join(RESULTS, "dnh_study", "dnh_study_sign_flip.json")))["0.0"]
    meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    base = list(meta["seeds"])
    seeds24 = (base + [s + 1000 for s in base])[:24]
    seeds12 = seeds24[:12]
    config = load_config()
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(7, os.cpu_count() or 2))))
    fl = config["federated_learning"]
    eval_config = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                                      seed_offset=fl.get("eval_seed_offset", 500_000))
    mob = config["mobility"].get("scenario", "highway")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s_study.json" % tag)
    results = json.load(open(path)) if os.path.exists(path) else {}

    plan = [(0.4, seeds24), (0.24, seeds24), (0.0, seeds12)]
    for ratio, sd in plan:
        for m in baselines:
            name = "r%.2f_%s" % (ratio, m)
            if name in results and len(results[name].get("per_seed_results", [])) == len(sd):
                print("  %s (cached)" % name, flush=True)
                continue
            t0 = time.time()
            results[name] = er.run_multiseed_experiment(sd, config=config, eval_config=eval_config,
                                                        attack_ratio=ratio, mobility_mode=mob, method=m)
            c = results[name]
            pc = c.get("pooled_confusion_counts") or {}
            print("  %-26s %.1f%%  (%.0f s)" % (name, c["completion_rate"]["mean"], time.time() - t0), flush=True)
            json.dump(results, open(path, "w"))

    ref = {
        (0.4, "do_no_harm"): hyp["r0.40_do_no_harm"], (0.4, "fedavg"): hyp["r0.40_fedavg"],
        (0.24, "do_no_harm"): hyp["r0.24_do_no_harm"], (0.24, "fedavg"): hyp["r0.24_fedavg"],
        (0.0, "do_no_harm"): dnh0["do_no_harm"], (0.0, "fedavg"): dnh0["fedavg"],
    }
    rows = []
    for k, (ratio, _) in enumerate(plan):
        for m in baselines:
            t = paired_test(ref[(ratio, "do_no_harm")], results["r%.2f_%s" % (ratio, m)])
            t.update(id=("F%d" if tag == "fltrust" else "B%d") % (len(rows) + 1), baseline=m, ratio=ratio)
            rows.append(t)
    adj = holm([r["p"] for r in rows])
    for r, ap in zip(rows, adj):
        r["p_holm"] = float(ap)
        if r["ci_low"] > 0 and ap < 0.05:
            r["verdict"] = "do_no_harm BETTER"
        elif r["ci_high"] < 0 and ap < 0.05:
            r["verdict"] = "do_no_harm WORSE"
        elif r["ci_low"] > 0:
            r["verdict"] = "better (raw CI only)"
        elif r["ci_high"] < 0:
            r["verdict"] = "worse (raw CI only)"
        else:
            r["verdict"] = "no difference resolved"
    table = {}
    for ratio, _ in plan:
        row = {"fedavg": ref[(ratio, "fedavg")]["completion_rate"]["mean"],
               "do_no_harm": ref[(ratio, "do_no_harm")]["completion_rate"]["mean"]}
        for m in baselines:
            row[m] = results["r%.2f_%s" % (ratio, m)]["completion_rate"]["mean"]
        table[str(ratio)] = row
    json.dump({"tests": rows, "means": table}, open(os.path.join(out_dir, "%s_verdicts.json" % tag), "w"), indent=1)
    print("\nMean completion %:")
    for ratio, row in table.items():
        print("  %5s  %s" % (ratio, "  ".join("%s %.1f" % (k, v) for k, v in row.items())))
    print("\n%-4s %-18s %-6s %9s  %-18s %7s  %s" % ("id", "baseline", "ratio", "mean", "95% CI", "p_holm", "verdict"))
    for r in rows:
        print("%-4s %-18s %-6s %+9.2f  [%+.2f, %+.2f] %7.3f  %s" % (
            r["id"], r["baseline"], r["ratio"], r["mean"], r["ci_low"], r["ci_high"], r["p_holm"], r["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "baseline_study"))
    ap.add_argument("--family", choices=["robust", "fltrust"], default="robust")
    a = ap.parse_args()
    if a.family == "fltrust":
        # Separate pre-specified family (three tests, own Holm correction).
        run(a.out, ("fltrust",), "fltrust")
    else:
        run(a.out)
