"""Calibration of the trust detector.

Protocol
--------
1. The exclusion threshold ``tau`` is selected on the **validation** suite
   only (seed base 1).  The selection rule is pre-registered:

       a. Let ``P`` be the set of candidate thresholds whose mean validation
          detection rate is maximal (the *detection plateau*).
       b. Let ``F`` be the subset of ``P`` whose number of excluded honest
          clients on the honest validation scenarios is minimal (the
          *false-positive plateau*).
       c. ``tau*`` is the **median** of ``F`` - the most central, least
          extreme operating point of the plateau.

   The median (rather than an endpoint) is used deliberately: the lowest
   threshold of the plateau is the most aggressive exclusion rule and the
   highest is the most fragile one.

2. The selected threshold is then evaluated **once** on the **held-out**
   suite (independent seeds *and* additional scenario shapes that were never
   used for tuning, including unseen attack strengths).

3. The held-out results are reported but are **never** fed back into the
   selection.  :func:`select_exclusion_threshold` accepts only validation
   results, so it is structurally impossible for held-out data to influence
   the threshold.

Run as a script to print the full calibration report.
"""

import io
import json
import os
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.trust_validation import (
    run_validation_suite, run_heldout_suite, summarize,
)

VALIDATION_SEED = 1
HELDOUT_SEED = 9001

N_CLIENTS = 20
DIM = 64

CANDIDATE_TAUS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

HONEST_SCENARIOS = (
    "honest_iid",
    "honest_non_iid",
    "honest_benign_outlier",
    "honest_varying_participation",
    "honest_fully_disjoint",
    "honest_small_legitimate_update",
    "honest_idle_clients",
    "honest_idle_majority",
)


def honest_false_positives(results) -> int:
    """Total number of honest clients excluded across the honest scenarios."""
    summary = summarize(results)
    return sum(summary[name]["excluded_honest"]
               for name in HONEST_SCENARIOS if name in summary)


def detection_rate(results) -> Optional[float]:
    """Mean detection rate over the adversarial validation scenarios."""
    summary = summarize(results)
    tprs = [row["tpr"] for name, row in summary.items()
            if name.startswith("adversarial_") and row["tpr"] is not None]
    if not tprs:
        return None
    return float(sum(tprs) / len(tprs))


def select_exclusion_threshold(candidates=CANDIDATE_TAUS,
                               validation_results_by_tau=None,
                               n_clients: int = N_CLIENTS,
                               dim: int = DIM,
                               seed_base: int = VALIDATION_SEED) -> Tuple[float, list]:
    """Select the exclusion threshold on the VALIDATION suite only.

    ``validation_results_by_tau`` may be supplied to avoid recomputation; it
    must be a mapping ``{tau: validation_results}``.  There is deliberately
    no parameter through which held-out results could be supplied.

    The rule is the pre-registered plateau rule described in the module
    docstring: maximal validation detection rate, then minimal number of
    excluded honest clients, then the median of the surviving interval.
    """
    if validation_results_by_tau is None:
        validation_results_by_tau = {
            tau: run_validation_suite(n_clients, dim, exclusion_threshold=tau,
                                       seed_base=seed_base)
            for tau in candidates
        }

    table = []
    for tau in candidates:
        results = validation_results_by_tau[tau]
        fps = honest_false_positives(results)
        tpr = detection_rate(results)
        table.append({
            "tau": tau,
            "honest_false_positives": fps,
            "mean_validation_tpr": tpr,
            "admissible": fps == 0,
        })

    # (a) detection plateau
    tprs = [r["mean_validation_tpr"] for r in table
            if r["mean_validation_tpr"] is not None]
    if tprs:
        best_tpr = max(tprs)
        plateau = [r for r in table if r["mean_validation_tpr"] == best_tpr]
    else:
        plateau = list(table)

    # (b) false-positive plateau inside the detection plateau
    min_fp = min(r["honest_false_positives"] for r in plateau)
    survivors = sorted(r["tau"] for r in plateau
                       if r["honest_false_positives"] == min_fp)

    # (c) the median of the surviving interval
    n = len(survivors)
    selected = float(survivors[n // 2]) if n % 2 else \
        float((survivors[n // 2 - 1] + survivors[n // 2]) / 2.0)

    for row in table:
        row["in_detection_plateau"] = row in plateau
        row["in_selected_interval"] = row["tau"] in survivors

    return selected, table


def heldout_report(tau: float, heldout_results=None,
                   n_clients: int = N_CLIENTS, dim: int = DIM,
                   seed_base: int = HELDOUT_SEED) -> dict:
    """Evaluate an already-selected threshold on the held-out suite."""
    if heldout_results is None:
        heldout_results = run_heldout_suite(n_clients, dim, exclusion_threshold=tau,
                                            seed_base=seed_base)
    summary = summarize(heldout_results)
    rows = []
    for name, row in summary.items():
        rows.append({
            "scenario": name,
            "description": row["description"],
            "tp": row["detected_malicious"],
            "fp": row["excluded_honest"],
            "fn": row["missed_malicious"],
            "tn": row["honest_retained"],
            "n_honest": row["n_honest"],
            "n_malicious": row["n_malicious"],
            "tpr": row["tpr"],
            "fpr": row["fpr"],
            "fnr": row["fnr"],
            "precision": row["precision"],
        })
    return {"tau": tau, "seed_base": seed_base, "scenarios": rows}


def _fmt(v, nd=4):
    return "N/A" if v is None else ("%.*f" % (nd, v))


def print_report(selected: float, table: list, heldout: dict) -> None:
    print("=" * 96)
    print("TRUST DETECTOR CALIBRATION")
    print("=" * 96)
    print("Threshold selection uses the VALIDATION suite only (seed base %d)."
          % VALIDATION_SEED)
    print("Held-out suite (seed base %d) is evaluated once with the selected"
          % HELDOUT_SEED)
    print("threshold and is never fed back into the selection.")
    print()
    print("  %-8s %-22s %-22s %-10s %-10s" % (
        "tau", "honest FP (validation)", "mean TPR (validation)",
        "det. plateau", "selected"))
    for row in table:
        print("  %-8.2f %-22d %-22s %-10s %-10s" % (
            row["tau"], row["honest_false_positives"],
            _fmt(row["mean_validation_tpr"], 3),
            "yes" if row.get("in_detection_plateau") else "",
            "yes" if row.get("in_selected_interval") else ""))
    print()
    print("SELECTED tau = %.2f" % selected)
    print()
    print("=" * 96)
    print("HELD-OUT RESULTS (independent seeds + unseen scenario shapes)")
    print("=" * 96)
    print("  %-42s %-22s %-8s %-8s %-8s %-8s" % (
        "scenario", "tp,fp,fn,tn", "TPR", "FPR", "FNR", "prec."))
    for row in heldout["scenarios"]:
        print("  %-42s %-22s %-8s %-8s %-8s %-8s" % (
            row["scenario"][:41],
            "%d,%d,%d,%d" % (row["tp"], row["fp"], row["fn"], row["tn"]),
            _fmt(row["tpr"], 3), _fmt(row["fpr"], 3),
            _fmt(row["fnr"], 3), _fmt(row["precision"], 3)))


def run_calibration(output_json: str = None) -> dict:
    """Full calibration: select on validation, evaluate on held-out."""
    selected, table = select_exclusion_threshold()
    heldout = heldout_report(selected)
    if output_json:
        with io.open(output_json, "w", encoding="utf-8") as f:
            json.dump({"selected_tau": selected, "validation_table": table,
                       "heldout": heldout}, f, indent=2)
    return {"selected_tau": selected, "validation_table": table, "heldout": heldout}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    res = run_calibration(out)
    print_report(res["selected_tau"], res["validation_table"], res["heldout"])
    if out:
        print("\ncalibration written to %s" % out)
