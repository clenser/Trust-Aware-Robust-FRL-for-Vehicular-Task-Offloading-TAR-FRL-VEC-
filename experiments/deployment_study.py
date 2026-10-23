"""H2 / H4 under trace-calibrated edge deployments (realistic measured-utilisation load).

Pre-specified question: does the advantage of the do-no-harm rule over FedAvg (24 % attackers)
survive when the edge side is calibrated to the Alibaba trace?  Every seed draws a different
deployment (RSU capacity spread and co-tenant load factor are sampled from the measured
distributions, seeded by the environment seed), and the study runs the trough (05:00) and the
evening (20:00) hour with the measured-utilisation load model (``cpu_trace``).

  D1  completion, do_no_harm - FedAvg, 24 % attackers, 05:00, cpu_trace   (12 seeds)
  D2  same at 20:00
Holm-corrected across the two.

    python -m experiments.deployment_study
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from experiments import experiment_runner as er
from experiments.hypothesis_study import holm, paired_test

RESULTS = os.path.join(er.PROJECT_ROOT, "results")


def run(out_dir):
    meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    seeds = list(meta["seeds"])[:12]
    config = load_config()
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(7, os.cpu_count() or 2))))
    fl = config["federated_learning"]
    eval_config = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                                      seed_offset=fl.get("eval_seed_offset", 500_000))
    mob = config["mobility"].get("scenario", "highway")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "deployment_study.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for hour in (5.0, 20.0):
        cfg = copy.deepcopy(config)
        cfg["edge_calibration"].update({"enabled": True, "load_model": "cpu_trace", "start_hour": hour})
        for m in ("fedavg", "do_no_harm"):
            name = "h%02d_%s" % (hour, m)
            if name in results and len(results[name].get("per_seed_results", [])) == len(seeds):
                print("  %s (cached)" % name, flush=True)
                continue
            t0 = time.time()
            results[name] = er.run_multiseed_experiment(seeds, config=cfg, eval_config=eval_config, attack_ratio=0.24,
                                                        mobility_mode=mob, method=m)
            print("  %-18s %.1f%% (%.0f s)" % (name, results[name]["completion_rate"]["mean"], time.time() - t0), flush=True)
            json.dump(results, open(path, "w"))
    rows = []
    for pid, hour in (("D1", 5), ("D2", 20)):
        t = paired_test(results["h%02d_do_no_harm" % hour], results["h%02d_fedavg" % hour])
        t.update(id=pid, hour=hour)
        rows.append(t)
    adj = holm([r["p"] for r in rows])
    for r, ap in zip(rows, adj):
        r["p_holm"] = float(ap)
        r["verdict"] = ("SUPPORTED" if (r["ci_low"] > 0 and ap < 0.05) else
                        "raw CI only" if r["ci_low"] > 0 else "not established")
    json.dump({"tests": rows}, open(os.path.join(out_dir, "deployment_verdicts.json"), "w"), indent=1)
    for r in rows:
        print("%s %02d:00  DNH - FedAvg %+.2f pp [%+.2f, %+.2f] Holm p=%.3f  %s" % (
            r["id"], r["hour"], r["mean"], r["ci_low"], r["ci_high"], r["p_holm"], r["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "deployment_study"))
    run(ap.parse_args().out)
