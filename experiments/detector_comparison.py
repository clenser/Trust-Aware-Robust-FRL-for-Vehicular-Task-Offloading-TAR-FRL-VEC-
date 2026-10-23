"""
Detector comparison on the held-out validation scenarios.

Every detector is applied to the SAME clients, data partitions, seeds and
attacks.  Only held-out seed bases are used (never the ones the exclusion
threshold was selected on), so the comparison is not tuned on its own test
data.  The deployed detector is compared with the shadow rules of
``trust.alt_detectors``.

For each held-out scenario and detector the pooled confusion counts over
``N_SEEDS`` independent seed bases are reported with Wilson 95 % intervals.
Where a scenario has no malicious (or no honest) client the corresponding
rate is N/A, never 0.

    python -m experiments.detector_comparison
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.trust_validation import (HELDOUT_SCENARIOS, ScenarioSpec, build_scenario,
                                          evaluate_scenario)
from trust import alt_detectors

HELDOUT_SEED_BASES = list(range(9001, 9021))      # 20 independent draws
N_CLIENTS = 20
DIM = 64
DETECTORS = ("deployed_trust", "krum", "cosine_median", "distance_mad", "no_detection")


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def run():
    specs = [ScenarioSpec(**{**s.__dict__, "n_clients": N_CLIENTS, "dim": DIM})
             for s in HELDOUT_SCENARIOS]
    out = {}
    for spec in specs:
        counts = {d: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for d in DETECTORS}
        for base in HELDOUT_SEED_BASES:
            seed = base * 100 + spec.seed_index
            updates, malicious, idle = build_scenario(spec, seed)
            # The deployed detector applies the participation rule itself.
            res = evaluate_scenario(updates, malicious, seed=seed, idle=idle)
            for k, v in res["confusion"].items():
                counts["deployed_trust"][k] += v
            # The alternatives see the same detection population.
            population = {c: u for c, u in updates.items() if c not in set(idle)}
            mal = set(malicious)
            for name, fn in alt_detectors.ALTERNATIVES.items():
                s = alt_detectors.score(fn(population), population.keys(), mal)
                for k, v in s.items():
                    counts[name][k] += v
        out[spec.name] = {}
        for d in DETECTORS:
            c = counts[d]
            nm, nh = c["tp"] + c["fn"], c["fp"] + c["tn"]
            out[spec.name][d] = {
                **c,
                "tpr": (c["tp"] / nm) if nm else None,
                "tpr_ci95": wilson(c["tp"], nm),
                "fpr": (c["fp"] / nh) if nh else None,
                "fpr_ci95": wilson(c["fp"], nh),
            }
    return out


def to_markdown(results):
    lines = ["# Detector comparison on held-out scenarios", "",
             "Identical clients, partitions, seeds and attacks for every detector; "
             "%d independent held-out seed bases (%d-%d), %d clients per scenario. "
             "Pooled over seeds; Wilson 95 %% intervals; N/A = undefined denominator." %
             (len(HELDOUT_SEED_BASES), HELDOUT_SEED_BASES[0], HELDOUT_SEED_BASES[-1], N_CLIENTS), "",
             "Krum assumes a fixed Byzantine fraction of 0.25 (a design constant, not the "
             "realised attacker count).  `no_detection` is the reference row.", ""]
    head = "| Scenario | " + " | ".join("%s TPR / FPR" % d for d in DETECTORS[:4]) + " |"
    lines += [head, "|" + "---|" * 5]

    def cell(r):
        t = "N/A" if r["tpr"] is None else "%.2f" % r["tpr"]
        f = "N/A" if r["fpr"] is None else "%.3f" % r["fpr"]
        return "%s / %s" % (t, f)

    for scen, row in results.items():
        lines.append("| %s | %s |" % (scen, " | ".join(cell(row[d]) for d in DETECTORS[:4])))
    # Aggregate over adversarial and honest scenarios.
    lines += ["", "## Pooled over all adversarial scenarios (TPR) and all honest scenarios (FPR)", "",
              "| Detector | TP | FN | TPR [95 % CI] | FP | TN | FPR [95 % CI] |", "|---|---|---|---|---|---|---|"]
    for d in DETECTORS:
        tp = sum(r[d]["tp"] for r in results.values())
        fn = sum(r[d]["fn"] for r in results.values())
        fp = sum(r[d]["fp"] for r in results.values())
        tn = sum(r[d]["tn"] for r in results.values())
        tci, fci = wilson(tp, tp + fn), wilson(fp, fp + tn)
        lines.append("| %s | %d | %d | %.3f [%s, %s] | %d | %d | %.4f [%s, %s] |" % (
            d, tp, fn, tp / max(1, tp + fn), tci[0], tci[1], fp, tn, fp / max(1, fp + tn), fci[0], fci[1]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    res = run()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(out_dir, "detector_comparison.json"), "w") as f:
        json.dump({"seed_bases": HELDOUT_SEED_BASES, "n_clients": N_CLIENTS, "dim": DIM,
                   "scenarios": res}, f, indent=2)
    md = to_markdown(res)
    with open(os.path.join(out_dir, "detector_comparison.md"), "w") as f:
        f.write(md)
    print(md)
