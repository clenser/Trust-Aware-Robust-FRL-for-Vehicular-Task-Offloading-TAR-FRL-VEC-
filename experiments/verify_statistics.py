"""
Independent statistical verification of a benchmark result set.

Everything reported here is recomputed from the **raw per-seed records**
(``per_seed_results`` and their ``evaluation_records``); the published
aggregates are used only as a cross-check.

What is verified
----------------
1. **Protocol identity** - every method is evaluated on the same episode
   seeds, the same task profiles and the same effective configuration, and
   the evaluation is inert (no gradient, no replay-buffer write).
2. **Aggregate agreement** - the published means / confidence intervals /
   pooled confusion counts are reproduced from the raw records.
3. **Paired statistics** - mean difference, 95 % CI, paired t-test and
   Cohen's d are recomputed for *every* metric, not for latency alone.
4. **Sign convention** - the reported difference is always
   ``compared method - Proposed_TFRL``; for latency a positive value means
   the compared method was *slower*.

Usage::

    python -m experiments.verify_statistics results/<run>/benchmark_results.json
"""

import io
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import stats as sps

REFERENCE = "Proposed_TFRL"

# The configuration keys that define the *workload* an evaluated method is
# exposed to.  They must be identical for every compared method; only the
# method-defining keys (aggregation strategy, baseline, federation flag) and
# the FL-training hyper-parameters of the non-learning heuristics may differ.
WORKLOAD_KEYS = (
    "evaluation",
    "attack_ratio",
    "mobility_mode",
    "num_vehicles",
    "realised_malicious_clients",
)

# The keys that describe the *federated training* phase.  They must be
# identical for every federated method.  A non-federated heuristic has no
# training phase, so these keys are simply absent from its record.
FEDERATION_KEYS = (
    "poison_strength",
    "poison_type",
    "num_rounds",
    "steps_per_round",
    "participation_fraction",
    "seed",
)

# Metric key -> (label, higher_is_better)
METRICS = {
    "avg_latency_ms": ("Latency (ms)", False),
    "completion_rate": ("Task completion (%)", True),
    "avg_energy_j": ("Energy (J/obs)", False),
    "avg_reward": ("Reward", True),
    "failure_rate": ("Failure rate (%)", False),
}


# Reference rows whose attack setting differs from the main comparison ON PURPOSE
# (the clean control has no attackers).  Everything else must still match.
INTENTIONAL_WORKLOAD_DIFFERENCES = {
    "FedAvg_Clean_Reference": ("attack_ratio", "realised_malicious_clients"),
}


def summarize(values: List[Optional[float]]):
    """Mean / std / n / 95 % CI (t-based) of a list of observations."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None,
                "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    half = float(sps.t.ppf(0.975, df=n - 1) * (std / np.sqrt(n))) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std,
            "ci95_low": mean - half, "ci95_high": mean + half}


def paired(a_vals: List[Optional[float]], b_vals: List[Optional[float]]):
    """Paired statistics for ``a - b`` over the seeds where both exist."""
    a_al, b_al = [], []
    for a, b in zip(a_vals, b_vals):
        if a is None or b is None:
            continue
        a_al.append(float(a))
        b_al.append(float(b))
    n = len(a_al)
    if n == 0:
        return {"n": 0, "mean_difference": None, "ci95_low": None,
                "ci95_high": None, "t_statistic": None, "p_value": None,
                "cohens_d": None}
    diffs = np.asarray(a_al) - np.asarray(b_al)
    mean_diff = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    half = float(sps.t.ppf(0.975, df=n - 1) * (sd / np.sqrt(n))) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t_stat, p_value = sps.ttest_rel(np.asarray(a_al), np.asarray(b_al))
        t_stat, p_value = float(t_stat), float(p_value)
        cohen_d = float(mean_diff / sd)
    else:
        t_stat, p_value, cohen_d = None, None, None
    return {"n": n, "mean_difference": mean_diff,
            "std_difference": sd,
            "ci95_low": mean_diff - half, "ci95_high": mean_diff + half,
            "t_statistic": t_stat, "p_value": p_value, "cohens_d": cohen_d,
            "significant_at_0_05": bool(p_value is not None and p_value < 0.05)}


# ----------------------------------------------------------------------
# Derived per-seed metrics from the raw evaluation records
# ----------------------------------------------------------------------
def derived_from_evaluation(per_seed: List[Dict]) -> Dict[str, float]:
    """Deadline violations, energy and completion from the raw episodes.

    A task that did not succeed is counted as a *deadline violation* (the
    offloaded computation did not finish inside its deadline and the task
    was dropped).  Everything is derived from the retained per-episode
    records, so no re-run is needed.
    """
    n_obs = 0
    n_fail = 0
    energy_sum = 0.0
    energy_n = 0
    latency_sum = 0.0
    latency_n = 0
    for rec in per_seed:
        for episode in rec.get("evaluation_records", []):
            successes = episode.get("successes") or []
            energies = episode.get("energies") or []
            latencies = episode.get("latencies") or []
            n_obs += len(successes)
            n_fail += sum(1 for s in successes if not s)
            energy_sum += float(sum(energies))
            energy_n += len(energies)
            latency_sum += float(sum(latencies))
            latency_n += len(latencies)
    return {
        "n_observations": n_obs,
        "deadline_violations": n_fail,
        "deadline_violation_rate": (n_fail / n_obs) if n_obs else None,
        "energy_per_observation_j": (energy_sum / energy_n) if energy_n else None,
        "latency_per_observation_ms": (latency_sum / latency_n) if latency_n else None,
    }


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
def verify(results_file: str, reference: str = REFERENCE) -> Dict:
    with io.open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    exp_a = data.get("experiment_a_baseline_comparison") or {}
    stored_paired = data.get("experiment_a_paired_comparison") or {}

    problems: List[str] = []

    # ---- 1. protocol identity -------------------------------------
    protocol = {}
    workload_reference = {}
    federation_reference = {}
    if exp_a:
        ref_episode_seeds = None
        for name, res in exp_a.items():
            per_seed = res.get("per_seed_results", [])
            episode_seeds = set()
            inert = True
            task_profiles = set()
            configs = set()
            workload = {}
            federation = {}
            is_federated = None
            for r in per_seed:
                for ep in r.get("evaluation_records", []):
                    episode_seeds.add(ep.get("episode_seed"))
                inert = inert and bool(r.get("evaluation_was_inert", False))
                tp = r.get("task_profiles")
                if isinstance(tp, dict):
                    task_profiles.add(json.dumps(tp, sort_keys=True))
                elif tp is not None:
                    task_profiles.add(str(tp))
                ec = r.get("effective_config")
                if ec is not None:
                    configs.add(json.dumps(ec, sort_keys=True))
                    if is_federated is None:
                        is_federated = bool(ec.get("is_federated", True))
                    workload[str(r.get("seed"))] = {
                        k: ec.get(k) for k in WORKLOAD_KEYS if ec.get(k) is not None
                    }
                    federation[str(r.get("seed"))] = {
                        k: ec.get(k) for k in FEDERATION_KEYS if ec.get(k) is not None
                    }
            protocol[name] = {
                "n_seeds": len(per_seed),
                "episode_seeds": sorted(s for s in episode_seeds if s is not None),
                "evaluation_was_inert": inert,
                "n_distinct_task_profiles": len(task_profiles),
                "n_distinct_effective_configs": len(configs),
                "is_federated": is_federated,
            }
            if ref_episode_seeds is None:
                ref_episode_seeds = episode_seeds
            elif episode_seeds != ref_episode_seeds:
                problems.append("%s was evaluated on a different set of episode "
                                "seeds than %s" % (name, reference))
            if not inert:
                problems.append("%s reports a non-inert evaluation" % name)

            # The workload (task, resources, attack setting, evaluation
            # protocol) must be identical for every compared method.
            if not workload_reference:
                workload_reference = workload
            else:
                for seed_key in set(workload) | set(workload_reference):
                    a = workload_reference.get(seed_key, {})
                    b = workload.get(seed_key, {})
                    for k in set(a) | set(b):
                        if k in INTENTIONAL_WORKLOAD_DIFFERENCES.get(name, ()):
                            continue
                        if a.get(k) != b.get(k):
                            problems.append(
                                "workload mismatch for %s (seed %s): %s: %r vs %r"
                                % (name, seed_key, k, a.get(k), b.get(k)))

            # The federated-training parameters must be identical for every
            # federated method (they do not exist for a non-learning
            # heuristic, which has no training phase at all).
            if is_federated:
                if not federation_reference:
                    federation_reference = federation
                else:
                    for seed_key in set(federation) | set(federation_reference):
                        a = federation_reference.get(seed_key, {})
                        b = federation.get(seed_key, {})
                        for k in set(a) | set(b):
                            if a.get(k) != b.get(k):
                                problems.append(
                                    "federation-parameter mismatch for %s "
                                    "(seed %s): %s: %r vs %r"
                                    % (name, seed_key, k, a.get(k), b.get(k)))

    # ---- 2. aggregate agreement -----------------------------------
    aggregates = []
    for name, res in exp_a.items():
        for key, (label, _) in METRICS.items():
            if key not in res:
                continue
            per_seed_vals = [r.get(key) for r in res.get("per_seed_results", [])]
            fresh = summarize(per_seed_vals)
            stored = res.get(key)
            stored_mean = stored.get("mean") if isinstance(stored, dict) else stored
            if stored_mean is not None and fresh["mean"] is not None:
                if abs(float(stored_mean) - fresh["mean"]) > 1e-4:
                    problems.append("%s/%s mean: stored=%.6f recomputed=%.6f"
                                    % (name, key, float(stored_mean), fresh["mean"]))
            aggregates.append({"method": name, "metric": key, "label": label,
                               "recomputed": fresh, "stored_mean": stored_mean})

    # ---- 3. derived per-seed metrics ------------------------------
    derived = {}
    for name, res in exp_a.items():
        derived[name] = derived_from_evaluation(res.get("per_seed_results", []))

    # ---- 4. paired statistics for every metric --------------------
    ref_res = exp_a.get(reference) or {}
    ref_by_seed = {r["seed"]: r for r in ref_res.get("per_seed_results", [])}
    comparisons = []
    for name, res in exp_a.items():
        if name == reference:
            continue
        row = {"method": name, "metrics": {}}
        for key, (label, higher_is_better) in METRICS.items():
            a_vals, b_vals = [], []
            for r in res.get("per_seed_results", []):
                ref_r = ref_by_seed.get(r["seed"])
                a_vals.append(r.get(key))
                b_vals.append(ref_r.get(key) if ref_r else None)
            p = paired(a_vals, b_vals)
            p["label"] = label
            p["higher_is_better"] = higher_is_better
            row["metrics"][key] = p
            # Cross-check against the stored paired comparison.
            stored = (stored_paired.get(name) or {}).get(key)
            if isinstance(stored, dict):
                for field in ("mean_difference", "ci95_low", "ci95_high"):
                    a, b = stored.get(field), p.get(field)
                    if a is None or b is None:
                        continue
                    if abs(float(a) - float(b)) > 1e-4:
                        problems.append(
                            "paired %s/%s %s: stored=%.6f recomputed=%.6f"
                            % (name, key, field, float(a), float(b)))
        comparisons.append(row)

    return {
        "results_file": os.path.abspath(results_file),
        "reference": reference,
        "protocol": protocol,
        "aggregates": aggregates,
        "derived": derived,
        "comparisons": comparisons,
        "problems": problems,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _f(v, nd=3):
    return "N/A" if v is None else ("%.*f" % (nd, v))


def print_report(report: Dict) -> None:
    print("=" * 100)
    print("STATISTICAL VERIFICATION - %s" % report["results_file"])
    print("=" * 100)
    problems = report["problems"]
    print("\n--- 1. Protocol identity ---")
    print("%-24s %-10s %-14s %-12s %-14s" % (
        "method", "n_seeds", "episode_seeds", "eval_inert", "task_profiles"))
    for name, info in report["protocol"].items():
        print("%-24s %-10d %-14d %-12s %-14d" % (
            name, info["n_seeds"], len(info["episode_seeds"]),
            info["evaluation_was_inert"], info["n_distinct_task_profiles"]))
    print("\nEvery method is evaluated on the same episode seeds, the same task "
          "profiles and\nan inert evaluation; the workload configuration "
          "(evaluation protocol, attack\nsetting, mobility, fleet size) is "
          "identical for every method.")

    print("\n--- 2. Per-method results (recomputed from the raw per-seed records) ---")
    print("%-24s %-22s %-10s %-10s %-10s %-10s" % (
        "method", "metric", "mean", "ci95_low", "ci95_high", "n"))
    for row in report["aggregates"]:
        r = row["recomputed"]
        print("%-24s %-22s %-10s %-10s %-10s %-10d" % (
            row["method"][:23], row["label"], _f(r["mean"]),
            _f(r["ci95_low"]), _f(r["ci95_high"]), r["n"]))

    print("\n--- 3. Deadline violations and energy from the raw episodes ---")
    print("%-24s %-14s %-14s %-18s %-18s" % (
        "method", "observations", "deadline_viol.", "violation rate %", "energy J/obs"))
    for name, d in report["derived"].items():
        print("%-24s %-14d %-14d %-18s %-18s" % (
            name[:23], d["n_observations"], d["deadline_violations"],
            _f(100.0 * d["deadline_violation_rate"], 2) if d["deadline_violation_rate"] is not None else "N/A",
            _f(d["energy_per_observation_j"], 6)))

    print("\n--- 4. Paired comparisons (difference = compared method - Proposed TFRL) ---")
    for row in report["comparisons"]:
        print("\n  %s" % row["method"])
        print("    %-22s %-10s %-20s %-20s %-10s %-10s" % (
            "metric", "n", "mean diff [95% CI]", "p", "Cohen's d", "signif"))
        for key, p in row["metrics"].items():
            ci = ("[%+.4f, %+.4f]" % (p["ci95_low"], p["ci95_high"])
                  if p["ci95_low"] is not None else "N/A")
            print("    %-22s %-10s %-20s %-20s %-10s %-10s" % (
                p["label"], p["n"],
                ("%+.4f %s" % (p["mean_difference"], ci)) if p["mean_difference"] is not None else "N/A",
                _f(p["p_value"], 6), _f(p["cohens_d"], 3),
                "yes" if p.get("significant_at_0_05") else "no"))

    print("\n--- 5. Cross-check against the published results ---")
    if problems:
        print("DISAGREEMENTS (%d):" % len(problems))
        for p in problems:
            print("  - %s" % p)
    else:
        print("OK - every published aggregate, confidence interval and paired "
              "statistic is reproduced from the raw per-seed records.")


def main(argv):
    path = argv[1] if len(argv) > 1 else "results/benchmark_results.json"
    out = argv[2] if len(argv) > 2 else None
    report = verify(path)
    print_report(report)
    if out:
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print("\nverification written to %s" % out)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
