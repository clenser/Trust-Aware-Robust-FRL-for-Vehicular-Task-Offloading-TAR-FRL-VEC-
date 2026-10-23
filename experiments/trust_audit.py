"""
Trust-detector audit.

Recomputes **every** reported detection metric directly from the underlying
TP / FP / FN / TN counts, states the denominator explicitly, and separates
*undefined* metrics (no denominator) from *zero-valued* ones (a defined rate
that happens to be zero).

It also documents the relationship between the attack ratio and the
false-positive rate, which is a property of the population-relative scoring
rule rather than an artefact of the reporting.
"""

import io
import json
import os
from typing import Dict, Optional

# Metric denominators, stated explicitly.
DENOMINATORS = {
    "tpr": "TP + FN  (malicious clients: the positive class)",
    "fpr": "FP + TN  (honest clients: the negative class)",
    "fnr": "TP + FN  (malicious clients: the positive class)",
    "precision": "TP + FP  (clients the detector flagged)",
    "recall": "TP + FN  (malicious clients: the positive class)",
    "accuracy": "TP + FP + FN + TN  (every decision)",
}

FIELD_FOR = {
    "attack_detection_rate": "tpr",
    "false_positive_rate": "fpr",
    "false_negative_rate": "fnr",
    "detection_precision": "precision",
}


def rates_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Optional[float]]:
    """Recompute every detection rate from the raw confusion counts.

    A rate whose denominator is zero is ``None`` (undefined / N-A).  A rate
    whose denominator is non-zero but whose numerator is zero is ``0.0`` - a
    real, measured zero, not a missing value.
    """
    def r(num, den):
        return (num / den) if den > 0 else None

    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_decisions": tp + fp + fn + tn,
        "n_malicious": tp + fn,
        "n_honest": fp + tn,
        "tpr": r(tp, tp + fn),
        "fpr": r(fp, fp + tn),
        "fnr": r(fn, tp + fn),
        "precision": r(tp, tp + fp),
        "recall": r(tp, tp + fn),
        "accuracy": r(tp + tn, tp + fp + fn + tn),
    }


def check_against_record(record: Dict) -> Dict[str, object]:
    """Verify that a stored result record agrees with its raw counts."""
    c = record.get("confusion_counts") or {}
    recomputed = rates_from_counts(c.get("tp", 0), c.get("fp", 0),
                                  c.get("fn", 0), c.get("tn", 0))
    problems = []
    for key, field in FIELD_FOR.items():
        stored = record.get(key)
        fresh = recomputed.get(field)
        if stored is None and fresh is None:
            continue
        if stored is None or fresh is None:
            problems.append("%s: stored=%r recomputed=%r" % (key, stored, fresh))
            continue
        if abs(float(stored) - float(fresh)) > 1e-9:
            problems.append("%s: stored=%.10f recomputed=%.10f" % (key, stored, fresh))
    return {"recomputed": recomputed, "problems": problems,
            "denominators": DENOMINATORS}


def _collect(data: Dict):
    """Yield (section, scenario, seed, record) for every detection result."""
    for section, value in data.items():
        if not isinstance(value, dict):
            continue
        if "pooled_confusion_counts" in value:
            rec = dict(value)
            rec["confusion_counts"] = value["pooled_confusion_counts"]
            yield section, "-", "pooled", rec
            for r in value.get("per_seed_results", []):
                yield section, "-", r.get("seed"), r
            continue
        for name, res in value.items():
            if not isinstance(res, dict) or "pooled_confusion_counts" not in res:
                continue
            rec = dict(res)
            rec["confusion_counts"] = res["pooled_confusion_counts"]
            yield section, name, "pooled", rec
            for r in res.get("per_seed_results", []):
                yield section, name, r.get("seed"), r


def audit_run(results_file: str) -> Dict[str, object]:
    """Audit every stored detection metric in a results file."""
    with io.open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    total_problems = 0
    for section, scenario, seed, record in _collect(data):
        chk = check_against_record(record)
        total_problems += len(chk["problems"])
        rows.append({
            "section": section,
            "scenario": scenario,
            "seed": seed,
            "counts": chk["recomputed"],
            "problems": chk["problems"],
        })

    return {
        "results_file": results_file,
        "denominators": DENOMINATORS,
        "rows": rows,
        "total_problems": total_problems,
    }


def _f(v):
    return "N/A" if v is None else "%.4f" % v


def print_audit(report: Dict[str, object], title: str = "TRUST-METRIC AUDIT",
                only_with_decisions: bool = True) -> None:
    print("=" * 104)
    print(title)
    print("=" * 104)
    print("Denominators:")
    for k, v in DENOMINATORS.items():
        print("  %-10s %s" % (k, v))
    print()
    print("%-38s %-20s %-6s %-24s %-8s %-8s %-8s %-8s" % (
        "section", "scenario", "seed", "tp,fp,fn,tn", "TPR", "FPR", "FNR", "prec."))
    current = None
    for row in report["rows"]:
        c = row["counts"]
        if only_with_decisions and c["n_decisions"] == 0:
            continue
        if row["section"] != current:
            current = row["section"]
            print("  [%s]" % current)
        print("%-38s %-20s %-6s %-24s %-8s %-8s %-8s %-8s" % (
            "", row["scenario"][:19], str(row["seed"])[:5],
            "%d,%d,%d,%d" % (c["tp"], c["fp"], c["fn"], c["tn"]),
            _f(c["tpr"]), _f(c["fpr"]), _f(c["fnr"]), _f(c["precision"])))
    print()
    print("TOTAL DISAGREEMENTS WITH THE RAW COUNTS: %d" % report["total_problems"])


def write_audit(report: Dict[str, object], path: str) -> None:
    """Write the audit as a machine-readable artefact."""
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    rep = audit_run(path)
    print_audit(rep)
    if out:
        write_audit(rep, out)
        print("\naudit written to %s" % out)
