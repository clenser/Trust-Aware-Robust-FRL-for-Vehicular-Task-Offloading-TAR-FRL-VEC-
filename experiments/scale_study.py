"""Does the advantage of the do-no-harm rule hold at larger fleets?

Pre-specified (set before running), 24 % attackers, sign-flip, 8 paired seeds, fleets of 50 and
100 vehicles (the benchmark uses 25; RSU capacity is unchanged, so larger fleets also load the
edge more):

  S1  completion, do_no_harm - FedAvg,    50 vehicles
  S2  completion, do_no_harm - Multi-Krum, 50 vehicles
  S3  completion, do_no_harm - FedAvg,    100 vehicles
  S4  completion, do_no_harm - Multi-Krum, 100 vehicles
Holm-corrected across the four.  The aggregation time per round is also reported.

    python -m experiments.scale_study
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
METHODS = ("fedavg", "do_no_harm", "multi_krum")


def run(out_dir):
    meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    seeds = list(meta["seeds"])[:8]
    config = load_config()
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(7, os.cpu_count() or 2))))
    fl = config["federated_learning"]
    eval_config = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                                      seed_offset=fl.get("eval_seed_offset", 500_000))
    mob = config["mobility"].get("scenario", "highway")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scale_study.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for n in (50, 100):
        for m in METHODS:
            name = "n%d_%s" % (n, m)
            if name in results and len(results[name].get("per_seed_results", [])) == len(seeds):
                print("  %s (cached)" % name, flush=True)
                continue
            t0 = time.time()
            results[name] = er.run_multiseed_experiment(seeds, config=config, eval_config=eval_config, attack_ratio=0.24,
                                                        mobility_mode=mob, method=m, num_vehicles=n)
            print("  %-20s %.1f%% (%.0f s)" % (name, results[name]["completion_rate"]["mean"], time.time() - t0), flush=True)
            json.dump(results, open(path, "w"))
    rows = []
    for n in (50, 100):
        for base in ("fedavg", "multi_krum"):
            t = paired_test(results["n%d_do_no_harm" % n], results["n%d_%s" % (n, base)])
            t.update(id="S%d" % (len(rows) + 1), vehicles=n, baseline=base)
            rows.append(t)
    adj = holm([r["p"] for r in rows])
    for r, ap in zip(rows, adj):
        r["p_holm"] = float(ap)
        r["verdict"] = ("DNH BETTER" if (r["ci_low"] > 0 and ap < 0.05) else
                        "DNH WORSE" if (r["ci_high"] < 0 and ap < 0.05) else "no difference resolved")
    agg = {}
    for name, c in results.items():
        t = [x["aggregation_time_ms"] for s in c["per_seed_results"] for x in s["training_rounds"] if x.get("aggregation_time_ms")]
        agg[name] = float(np.median(t)) if t else None
    json.dump({"tests": rows, "median_aggregation_ms_loaded_machine": agg}, open(os.path.join(out_dir, "scale_verdicts.json"), "w"), indent=1)
    for r in rows:
        print("%s %3d vehicles DNH - %-10s %+6.2f pp [%+.2f, %+.2f] Holm p=%.3f %s" % (
            r["id"], r["vehicles"], r["baseline"], r["mean"], r["ci_low"], r["ci_high"], r["p_holm"], r["verdict"]))
    print("median aggregation ms:", {k: (round(v) if v else None) for k, v in agg.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "scale_study"))
    run(ap.parse_args().out)
