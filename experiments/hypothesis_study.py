"""Pre-specified confirmation study of the roadmap hypotheses H1-H5.

The comparisons below, and the Holm correction across them, are fixed before any
run.  "Trust-aware" here is the do-no-harm rule (strategy ``do_no_harm``); the
older joint-trust rule appears in H3 because that hypothesis is about the tiers.
Each hypothesis is stated with the attack-ratio scope under which it is tested;
the default 24 % ratio and the high-contamination 40 % ratio are both reported.

  P1  H2  completion, do_no_harm - FedAvg, 40 % attackers          (24 seeds)
  P2  H2  completion, do_no_harm - FedAvg, 24 % attackers          (24 seeds)
  P3  H1  completed-task latency (ms), do_no_harm - FedAvg, 40 %   (24 seeds; < 0 is better)
  P4  H1  completed-task latency, do_no_harm - FedAvg, 24 %        (24 seeds)
  P5  H3  completion, joint trust (alpha 0.5) - behavioural only (alpha 1), 40 %
  P6  H3  same at 24 %
  P7  H3  completion, do_no_harm - behavioural only, 40 %
  P8  H4  do_no_harm - FedAvg, 40 %, 40 % of clients offline per round (missed broadcasts)
  P9  H4  do_no_harm - FedAvg, 40 %, synthetic NGSIM-style trace mobility
  P10 H4  do_no_harm - FedAvg, 40 %, authentic NGSIM US-101 replay
  (H5 overhead is measured separately: docs/DNH_DETECTOR.md.)

    python -m experiments.hypothesis_study [--seeds 24] [--out DIR]
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import stats

from configs.config_loader import load_config
from experiments import experiment_runner as er

RESULTS = os.path.join(er.PROJECT_ROOT, "results")


def _vals(cell, key):
    return {r["seed"]: r[key] for r in cell["per_seed_results"]}


def paired_test(a, b, key="completion_rate"):
    A, B = _vals(a, key), _vals(b, key)
    x = np.array([A[s] - B[s] for s in A if s in B], dtype=float)
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(0.975, n - 1) * se)
    p = float(stats.ttest_1samp(x, 0.0).pvalue) if se > 0 else 1.0
    return {"mean": m, "ci_low": m - h, "ci_high": m + h, "p": p, "n": n}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def run(n_seeds, out_dir):
    meta = json.load(open(os.path.join(RESULTS, "run_20261023_022608_env_v21_coverage_features", "metadata.json")))
    base = list(meta["seeds"])
    seeds = (base + [s + 1000 for s in base])[:n_seeds]
    seeds12 = seeds[:12]
    config = load_config()
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(6, os.cpu_count() or 2))))
    fl = config["federated_learning"]
    eval_config = er.EvaluationConfig(episodes=fl.get("eval_episodes", 5), steps_per_episode=fl.get("eval_steps", 30),
                                      seed_offset=fl.get("eval_seed_offset", 500_000))
    mob = config["mobility"].get("scenario", "highway")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hypothesis_study.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    def cell(name, sd, cfg=None, **kw):
        if name in results and len(results[name].get("per_seed_results", [])) == len(sd):
            print("  %s (cached)" % name, flush=True)
            return
        t0 = time.time()
        kw.setdefault("mobility_mode", mob)
        results[name] = er.run_multiseed_experiment(sd, config=cfg or config, eval_config=eval_config, **kw)
        c = results[name]
        print("  %-34s %.1f%%  %.0f ms  (%.0f s)" % (name, c["completion_rate"]["mean"],
              c["avg_completed_latency_ms"]["mean"], time.time() - t0), flush=True)
        json.dump(results, open(path, "w"))

    for r in (0.24, 0.4):
        cell("r%.2f_fedavg" % r, seeds, attack_ratio=r, method="fedavg")
        cell("r%.2f_do_no_harm" % r, seeds, attack_ratio=r, method="do_no_harm")
        cell("r%.2f_joint" % r, seeds, attack_ratio=r, method="trust_aware", alpha=0.5)
        cell("r%.2f_behavioural_only" % r, seeds, attack_ratio=r, method="trust_aware", alpha=1.0)
    cfg_c = copy.deepcopy(config)
    cfg_c["federated_learning"].update({"connectivity_dropout": 0.4, "missed_broadcast": True})
    cell("conn_fedavg", seeds12, cfg_c, attack_ratio=0.4, method="fedavg")
    cell("conn_do_no_harm", seeds12, cfg_c, attack_ratio=0.4, method="do_no_harm")
    for tag, mode in (("trace", "trace"), ("ngsim", "authentic_trace")):
        cell("%s_fedavg" % tag, seeds12, attack_ratio=0.4, method="fedavg", mobility_mode=mode)
        cell("%s_do_no_harm" % tag, seeds12, attack_ratio=0.4, method="do_no_harm", mobility_mode=mode)

    R = results
    tests = [
        ("P1", "H2", "completion, do_no_harm - FedAvg, 40 %", R["r0.40_do_no_harm"], R["r0.40_fedavg"], "completion_rate", +1),
        ("P2", "H2", "completion, do_no_harm - FedAvg, 24 %", R["r0.24_do_no_harm"], R["r0.24_fedavg"], "completion_rate", +1),
        ("P3", "H1", "completed latency (ms), do_no_harm - FedAvg, 40 %", R["r0.40_do_no_harm"], R["r0.40_fedavg"], "avg_completed_latency_ms", -1),
        ("P4", "H1", "completed latency (ms), do_no_harm - FedAvg, 24 %", R["r0.24_do_no_harm"], R["r0.24_fedavg"], "avg_completed_latency_ms", -1),
        ("P5", "H3", "completion, joint - behavioural only, 40 %", R["r0.40_joint"], R["r0.40_behavioural_only"], "completion_rate", +1),
        ("P6", "H3", "completion, joint - behavioural only, 24 %", R["r0.24_joint"], R["r0.24_behavioural_only"], "completion_rate", +1),
        ("P7", "H3", "completion, do_no_harm - behavioural only, 40 %", R["r0.40_do_no_harm"], R["r0.40_behavioural_only"], "completion_rate", +1),
        ("P8", "H4", "completion, do_no_harm - FedAvg, 40 % offline per round", R["conn_do_no_harm"], R["conn_fedavg"], "completion_rate", +1),
        ("P9", "H4", "completion, do_no_harm - FedAvg, synthetic-trace mobility", R["trace_do_no_harm"], R["trace_fedavg"], "completion_rate", +1),
        ("P10", "H4", "completion, do_no_harm - FedAvg, authentic NGSIM replay", R["ngsim_do_no_harm"], R["ngsim_fedavg"], "completion_rate", +1),
    ]
    rows = []
    for pid, hyp, desc, a, b, key, direction in tests:
        t = paired_test(a, b, key)
        t.update(id=pid, hypothesis=hyp, description=desc, direction=direction)
        rows.append(t)
    adj = holm([r["p"] for r in rows])
    for r, ap in zip(rows, adj):
        r["p_holm"] = float(ap)
        in_dir = (r["mean"] * r["direction"]) > 0
        excl = (r["ci_low"] > 0) if r["direction"] > 0 else (r["ci_high"] < 0)
        if in_dir and excl and ap < 0.05:
            r["verdict"] = "SUPPORTED"
        elif in_dir and excl:
            r["verdict"] = "SUPPORTED (raw CI only; not after Holm)"
        elif (not in_dir) and ((r["ci_high"] < 0) if r["direction"] > 0 else (r["ci_low"] > 0)):
            r["verdict"] = "CONTRADICTED"
        else:
            r["verdict"] = "not established"
    json.dump({"seeds": seeds, "tests": rows}, open(os.path.join(out_dir, "hypothesis_verdicts.json"), "w"), indent=1)
    print("\n%-4s %-3s %-58s %9s  %-18s %7s  %s" % ("id", "H", "comparison", "mean", "95% CI", "p_holm", "verdict"))
    for r in rows:
        print("%-4s %-3s %-58s %+9.2f  [%+.2f, %+.2f] %7.3f  %s" % (
            r["id"], r["hypothesis"], r["description"], r["mean"], r["ci_low"], r["ci_high"], r["p_holm"], r["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(RESULTS, "hypothesis_study"))
    a = ap.parse_args()
    run(a.seeds, a.out)
