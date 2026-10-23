"""Edge-load study: does the co-tenant load model change the baseline picture?

Pre-registered question (set before running): the original edge calibration
scales the bandwidth diurnal profile to the VM *core allocation* ratio (an upper
bound on contention).  VM_CPU.csv shows tenants actually use ~8 % of their
cores.  Hypothesis EL1: the completion-rate gap between Resource-Aware and
Nearest-RSU is larger under the allocation-view load than under the
measured-utilisation (cpu_trace) load, because heavy contention is what makes
load-blind routing costly.  EL2: under cpu_trace the loads are so small that
the completion rate equals the no-calibration run within a paired CI.

Heuristics only (no training): cheap, paired seeds, same workloads.

    python -m experiments.edge_load_study [--seeds N] [--out DIR]
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from configs.config_loader import load_config
from experiments import experiment_runner as er
from experiments.learner_study import paired

RESULTS = os.path.join(er.PROJECT_ROOT, "results")
BASELINES = ("Local_Only", "Nearest_RSU", "Resource_Aware")
SETTINGS = {
    "none": {},
    "alloc_h05": {"enabled": True, "start_hour": 5.0, "load_model": "bandwidth_profile"},
    "alloc_h20": {"enabled": True, "start_hour": 20.0, "load_model": "bandwidth_profile"},
    "cpu_h05": {"enabled": True, "start_hour": 5.0, "load_model": "cpu_trace"},
    "cpu_h20": {"enabled": True, "start_hour": 20.0, "load_model": "cpu_trace"},
}


# Ablation (pre-registered after the main study): which calibration component costs Nearest-RSU?
OFF = {"capacity_heterogeneity": False, "backhaul_rtt": False}
ABLATION = {
    "none": {},
    "cap_only": {"enabled": True, "start_hour": 20.0, "peak_background_share": 0.0, "capacity_heterogeneity": True, "backhaul_rtt": False},
    "rtt_only": {"enabled": True, "start_hour": 20.0, "peak_background_share": 0.0, "capacity_heterogeneity": False, "backhaul_rtt": True},
    "cpu_load_only": {"enabled": True, "start_hour": 20.0, "load_model": "cpu_trace", **OFF},
    "alloc_load_only": {"enabled": True, "start_hour": 20.0, "load_model": "bandwidth_profile", **OFF},
}


def run(n_seeds, out_dir, ablation=False):
    SETTINGS_ = ABLATION if ablation else SETTINGS
    meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    seeds = list(meta["sweep_seeds"])
    seeds = (seeds + [s + 1000 for s in seeds] + [s + 2000 for s in seeds] + [s + 3000 for s in seeds])[:n_seeds]
    config = load_config()
    fl = config["federated_learning"]
    ev = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                             seed_offset=fl.get("eval_seed_offset", 500_000))
    common = dict(attack_ratio=config["adversarial"].get("attack_ratio", 0.24),
                  mobility_mode=config["mobility"].get("scenario", "highway"))
    res = {}
    for name, ec in SETTINGS_.items():
        cfg = copy.deepcopy(config)
        if ec:
            cfg.setdefault("edge_calibration", {}).update(ec)
        res[name] = {}
        for b in BASELINES:
            res[name][b] = er.run_multiseed_experiment(seeds, config=cfg, eval_config=ev, baseline=b, **common)
            print("%-10s %-15s %5.1f%%" % (name, b, res[name][b]["completion_rate"]["mean"]), flush=True)
    summary = {}
    for name in SETTINGS_:
        summary[name + ": Resource_Aware - Nearest_RSU"] = paired(res[name]["Resource_Aware"], res[name]["Nearest_RSU"])
        summary[name + ": Resource_Aware - Local_Only"] = paired(res[name]["Resource_Aware"], res[name]["Local_Only"])
    for b in BASELINES:
        for name in SETTINGS_:
            if name != "none":
                summary["%s: %s vs none" % (name, b)] = paired(res[name][b], res["none"][b])
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"seeds": seeds, "summary": summary, "results": res}, open(os.path.join(out_dir, "edge_load_ablation.json" if ablation else "edge_load_study.json"), "w"))
    for k, v in summary.items():
        print("%-48s %+6.2f pp  CI [%+.2f, %+.2f] n=%d" % (k, v["mean_pp"], v["ci_low"], v["ci_high"], v["n"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(RESULTS, "edge_load_study"))
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    run(a.seeds, a.out, a.ablation)
