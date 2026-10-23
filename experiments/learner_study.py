"""Learner study: can the offloading policy be made to beat the heuristics?

Pre-registered question (set before running): *with the heuristics' own delay
estimates as inputs (state v2.2) and bandit-style targets (gamma = 0), does the
federated DQN beat the best simple heuristic under hidden-state conditions
(faulty RSU, co-tenant load, V2V sabotage), with a paired 95 % CI that excludes
zero, without losing on the default environment?*

Variants (all use the same federated protocol, seeds and attacks):
  base        state v2.1, gamma 0.95   (the active benchmark's learner)
  est         state v2.2, gamma 0.95
  est_bandit  state v2.2, gamma 0.0
  est_bandit_masked  est_bandit with the trust features hidden
Settings:
  default     the benchmark environment
  rotating    a faulty RSU (60 % drop) that moves every 40 steps
  peak        edge calibration at the 20:00 co-tenant peak
  dataplane   malicious V2V helpers, 30 % compute-rich vehicles, trust-aware peers

    python -m experiments.learner_study [--seeds N] [--settings a b] [--variants x y] [--out DIR]
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from configs.config_loader import load_config
from experiments import experiment_runner as er

RESULTS = os.path.join(er.PROJECT_ROOT, "results")

VARIANTS = {
    "base": {"environment_model": {"state_features": "v2.1"}, "federated_learning": {"gamma_discount": 0.95}},
    "est": {"environment_model": {"state_features": "v2.2"}, "federated_learning": {"gamma_discount": 0.95}},
    "est_bandit": {"environment_model": {"state_features": "v2.2"}, "federated_learning": {"gamma_discount": 0.0}},
    # Same learner with every trust feature hidden: isolates what behavioural trust adds.
    # est_bandit plus each RSU's advertised co-tenant load (state v2.3).
    "est_bandit_load": {"environment_model": {"state_features": "v2.3"}, "federated_learning": {"gamma_discount": 0.0}},
    "est_bandit_masked": {"environment_model": {"state_features": "v2.2", "mask_trust_features": True},
                          "federated_learning": {"gamma_discount": 0.0}},
}
RICH = {"vehicle_compute": {"rich_fraction": 0.3, "rich_cpu_ghz": [4.0, 8.0]}}
SETTINGS = {
    "default": {},
    "rotating": {"adversarial": {"faulty_rsus": [{"index": "random", "drop_prob": 0.6, "delay_factor": 1.0,
                                                    "rotate_every_steps": 40}]}},
    "peak": {"edge_calibration": {"enabled": True, "start_hour": 20.0}},
    # Realistic load: measured-utilisation co-tenant model (docs/EDGE_TRACE_USE.md).
    "cpu_peak": {"edge_calibration": {"enabled": True, "load_model": "cpu_trace", "start_hour": 20.0}},
    "cpu_trough": {"edge_calibration": {"enabled": True, "load_model": "cpu_trace", "start_hour": 5.0}},
    "dataplane": {"mobility": RICH, "adversarial": {"data_plane": {"enabled": True}},
                  "network": {"v2v_peer_selection": "trust_aware"}},
}
BASELINES = {"default": ("Resource_Aware", "Nearest_RSU"), "rotating": ("Resource_Aware", "Trust_Greedy"),
             "peak": ("Resource_Aware", "Nearest_RSU"), "dataplane": ("Resource_Aware", "Trust_V2V"),
             "cpu_peak": ("Resource_Aware", "Nearest_RSU"), "cpu_trough": ("Resource_Aware", "Nearest_RSU")}


def _merge(cfg, mods):
    for sec, kv in mods.items():
        cfg.setdefault(sec, {})
        for k, v in kv.items():
            if isinstance(v, dict) and isinstance(cfg[sec].get(k), dict):
                cfg[sec][k].update(v)
            else:
                cfg[sec][k] = v
    return cfg


def paired(a, b):
    A = {r["seed"]: r["completion_rate"] for r in a["per_seed_results"]}
    B = {r["seed"]: r["completion_rate"] for r in b["per_seed_results"]}
    x = np.array([A[s] - B[s] for s in A if s in B])
    n = len(x)
    from scipy import stats
    m = float(x.mean())
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean_pp": m, "ci_low": m - h, "ci_high": m + h, "n": n}


def run(n_seeds, settings, variants, out_dir, quick=False):
    parent_meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    seeds = list(parent_meta["sweep_seeds"])
    seeds = (seeds + [s + 1000 for s in seeds] + [s + 2000 for s in seeds])[:n_seeds]
    config = load_config()
    if quick:
        seeds = seeds[:2]
        config["federated_learning"].update({"num_rounds": 2, "steps_per_round": 4, "local_epochs": 2,
                                             "eval_episodes": 1, "eval_steps": 4})
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(6, os.cpu_count() or 2))))
    fl = config["federated_learning"]
    eval_config = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                                      seed_offset=fl.get("eval_seed_offset", 500_000))
    common = dict(attack_ratio=config["adversarial"].get("attack_ratio", 0.24),
                  mobility_mode=config["mobility"].get("scenario", "highway"))
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, "learner_study.json")
    results = json.load(open(partial)) if os.path.exists(partial) else {}

    def cell(setting, name, cfg, **kw):
        g = results.setdefault(setting, {})
        if name in g and len(g[name].get("per_seed_results", [])) == len(seeds):
            print("    %s / %s (cached)" % (setting, name), flush=True)
            return
        t0 = time.time()
        g[name] = er.run_multiseed_experiment(seeds, config=cfg, eval_config=eval_config, **kw, **common)
        print("    %s / %s: %.1f%%  (%.0f s)" % (setting, name, g[name]["completion_rate"]["mean"], time.time() - t0), flush=True)
        json.dump(results, open(partial, "w"))

    for st in settings:
        base_cfg = _merge(copy.deepcopy(config), SETTINGS[st])
        for b in BASELINES[st]:
            cell(st, b, base_cfg, baseline=b)
        for v in variants:
            cell(st, "learned_" + v, _merge(copy.deepcopy(base_cfg), VARIANTS[v]), method="trust_aware")
    # summary
    summary = {}
    for st, g in results.items():
        ref = g.get("Resource_Aware")
        for name, c in g.items():
            if name.startswith("learned_") and ref:
                summary["%s/%s vs Resource_Aware" % (st, name)] = paired(c, ref)
    json.dump({"seeds": seeds, "summary": summary}, open(os.path.join(out_dir, "learner_summary.json"), "w"), indent=1)
    for k, v in summary.items():
        print("%-45s %+6.1f pp  CI [%+.1f, %+.1f]  n=%d" % (k, v["mean_pp"], v["ci_low"], v["ci_high"], v["n"]))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default=os.path.join(RESULTS, "learner_study"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(a.seeds, a.settings, a.variants, a.out, a.quick)
