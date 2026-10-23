"""
Detector study: protocol comparison, threshold sensitivity and uncertainty.

This module produces the scientific evidence for the trust detector that the
benchmark alone cannot provide:

Part A - **Participation-rule comparison.**  The deployed protocol and the
    candidate protocol are compared on *identical* scenarios and seeds.  The
    decision is taken on the VALIDATION suite; the held-out suite is
    assessed afterwards and never informs the decision.

Part B - **Threshold sensitivity.**  The exclusion threshold is swept over a
    grid on the validation suite only.  The selected threshold is then
    assessed once on the untouched held-out suite.

Part C - **Uncertainty.**  Every reported rate carries a Wilson score
    interval, so a rate of 0.0000 over 950 decisions is distinguishable from
    a rate of 0.0000 over 15 decisions.

Part D - **Heterogeneity and participation sweeps.**  How the false-positive
    rate and the detection rate move as the honest population becomes more
    heterogeneous and as more clients are non-participants.

Part E - **Realistic (simulated-protocol) scenarios.**  The synthetic
    harness cannot model the interaction between the population statistics
    and a partially idle population, so the same comparison is repeated on
    the real federated protocol (procedural highway, synthetic trace and the
    authentic NGSIM trace) with identical seeds.

Usage::

    python -m experiments.detector_study --out results/detector_study.json
"""

import io
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from experiments.trust_validation import (
    BASE_VALIDATION_SCENARIOS, HELDOUT_SCENARIOS, ScenarioSpec,
    run_suite, summarize,
)

TAU_GRID = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
VALIDATION_SEED_BASE = 1
HELDOUT_SEED_BASE = 9001


# ----------------------------------------------------------------------
# Uncertainty
# ----------------------------------------------------------------------
def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> Tuple:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because several reported rates are
    0 or 1, where the normal approximation is not valid.
    """
    if total <= 0:
        return (None, None)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_with_ci(value: Optional[float], successes: int, total: int) -> Dict:
    lo, hi = wilson_interval(successes, total)
    return {"value": value, "n": total, "ci95_low": lo, "ci95_high": hi}


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------
def pool_counts(results: Dict[str, Dict], scenarios: Optional[List[str]] = None
                ) -> Dict[str, int]:
    """Pool the raw confusion counts of the selected scenarios."""
    tp = fp = fn = tn = 0
    for name, res in results.items():
        if scenarios is not None and name not in scenarios:
            continue
        c = res["confusion"]
        tp += c["tp"]; fp += c["fp"]; fn += c["fn"]; tn += c["tn"]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def rates_from_counts(c: Dict[str, int]) -> Dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    honest = fp + tn
    malicious = tp + fn
    return {
        "confusion": c,
        "tpr": (tp / malicious) if malicious else None,
        "fpr": (fp / honest) if honest else None,
        "fnr": (fn / malicious) if malicious else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "tpr_ci": rate_with_ci((tp / malicious) if malicious else None, tp, malicious),
        "fpr_ci": rate_with_ci((fp / honest) if honest else None, fp, honest),
        "honest_denominator": honest,
        "malicious_denominator": malicious,
    }


# ----------------------------------------------------------------------
# Part A / B: synthetic protocol comparison and threshold sensitivity
# ----------------------------------------------------------------------
def _honest_scenarios(specs) -> List[str]:
    return [s.name for s in specs if not s.name.startswith("adversarial_")]


def protocol_comparison(tau: float = 0.40) -> Dict:
    """Compare the participation rule on validation and held-out suites.

    The rule is a *protocol* decision, so the selection criterion is
    pre-registered:

      * the honest false-positive count over all honest scenarios of the
        VALIDATION suite must be minimal, and
      * the free-riding attacker must still be detected (TPR = 1).

    The held-out suite is then assessed with the selected rule.
    """
    out = {}
    for label, specs, seed_base in (
            ("validation", BASE_VALIDATION_SCENARIOS, VALIDATION_SEED_BASE),
            ("heldout", HELDOUT_SCENARIOS, HELDOUT_SEED_BASE)):
        per_rule = {}
        for rule in (False, True):
            results = run_suite(list(specs), seed_base=seed_base,
                                exclusion_threshold=tau,
                                exclude_non_participants=rule)
            s = summarize(results)
            honest_fp = sum(s[n]["excluded_honest"] for n in _honest_scenarios(specs)
                            if n in s)
            honest_den = sum(s[n]["n_honest"] for n in _honest_scenarios(specs)
                             if n in s)
            fr = s.get("adversarial_free_rider_with_idle")
            fr_tpr = fr["tpr"] if fr else None
            pooled = rates_from_counts(pool_counts(results))
            per_rule[str(rule)] = {
                "honest_false_positives": honest_fp,
                "honest_denominator": honest_den,
                "honest_fpr": (honest_fp / honest_den) if honest_den else None,
                "free_rider_tpr": fr_tpr,
                "pooled": pooled,
                "per_scenario": {
                    name: {
                        "confusion": res["confusion"],
                        "rates": res["rates"],
                        "non_participants": res.get("non_participants", []),
                    } for name, res in results.items()},
            }
        out[label] = per_rule

    # --- the decision, on validation only ----------------------------
    val = out["validation"]
    def score(rule):
        r = val[str(rule)]
        fr_ok = 1.0 if (r["free_rider_tpr"] is None or r["free_rider_tpr"] >= 1.0) else 0.0
        return (r["honest_false_positives"], -fr_ok)

    selected_rule = min((False, True), key=score)
    out["selected_rule"] = selected_rule
    out["selection_basis"] = "validation suite only (seed base %d)" % VALIDATION_SEED_BASE
    return out


def threshold_sensitivity() -> Dict:
    """Sweep the exclusion threshold on validation, assess on held-out."""
    val = {}
    for tau in TAU_GRID:
        results = run_suite(list(BASE_VALIDATION_SCENARIOS),
                            seed_base=VALIDATION_SEED_BASE,
                            exclusion_threshold=tau)
        s = summarize(results)
        honest_fp = sum(s[n]["excluded_honest"] for n in _honest_scenarios(BASE_VALIDATION_SCENARIOS)
                        if n in s)
        honest_den = sum(s[n]["n_honest"] for n in _honest_scenarios(BASE_VALIDATION_SCENARIOS)
                         if n in s)
        attack = [s[n]["tpr"] for n in s if n.startswith("adversarial_")
                  and s[n]["tpr"] is not None]
        pooled = rates_from_counts(pool_counts(results))
        val[str(tau)] = {
            "honest_false_positives": honest_fp,
            "honest_denominator": honest_den,
            "honest_fpr": (honest_fp / honest_den) if honest_den else None,
            "mean_attack_tpr": float(np.mean(attack)) if attack else None,
            "pooled": pooled,
        }

    # Pre-registered selection: maximal detection, then minimal honest false
    # positives, then the median of the surviving interval (the most central
    # non-degenerate operating point).
    tprs = [r["mean_attack_tpr"] for r in val.values() if r["mean_attack_tpr"] is not None]
    best_tpr = max(tprs) if tprs else None
    plateau = [k for k, r in val.items()
               if best_tpr is not None and r["mean_attack_tpr"] == best_tpr]
    min_fp = min(val[k]["honest_false_positives"] for k in plateau)
    survivors = sorted(float(k) for k in plateau
                       if val[k]["honest_false_positives"] == min_fp)
    n = len(survivors)
    selected = survivors[n // 2] if n % 2 else (survivors[n // 2 - 1] + survivors[n // 2]) / 2.0

    held = {}
    for tau in TAU_GRID:
        results = run_suite(list(HELDOUT_SCENARIOS), seed_base=HELDOUT_SEED_BASE,
                            exclusion_threshold=tau)
        s = summarize(results)
        honest_fp = sum(s[n]["excluded_honest"] for n in _honest_scenarios(HELDOUT_SCENARIOS)
                        if n in s)
        honest_den = sum(s[n]["n_honest"] for n in _honest_scenarios(HELDOUT_SCENARIOS)
                         if n in s)
        attack = [s[n]["tpr"] for n in s if n.startswith("adversarial_")
                  and s[n]["tpr"] is not None]
        pooled = rates_from_counts(pool_counts(results))
        held[str(tau)] = {
            "honest_false_positives": honest_fp,
            "honest_denominator": honest_den,
            "honest_fpr": (honest_fp / honest_den) if honest_den else None,
            "mean_attack_tpr": float(np.mean(attack)) if attack else None,
            "pooled": pooled,
        }

    return {"grid": list(TAU_GRID), "validation": val, "heldout": held,
            "selected_tau": selected,
            "selected_interval": survivors,
            "selection_basis": "validation suite only (seed base %d)" % VALIDATION_SEED_BASE}


# ----------------------------------------------------------------------
# Part D: heterogeneity and participation sweeps
# ----------------------------------------------------------------------
def heterogeneity_sweep(tau: float = 0.40) -> Dict:
    """How does the honest false-positive rate move with heterogeneity?

    ``shared_component`` controls how much of the honest population's update
    direction is common: 1.0 is IID, 0.0 is fully disjoint.
    """
    out = {}
    for sc in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        spec = ScenarioSpec("honest_non_iid_sweep", heterogeneity="non_iid",
                            shared_component=sc, seed_index=2)
        res = run_suite([spec], seed_base=VALIDATION_SEED_BASE,
                        exclusion_threshold=tau)
        s = summarize(res)
        row = s["honest_non_iid_sweep"]
        out[str(sc)] = {
            "confusion": res["honest_non_iid_sweep"]["confusion"],
            "fpr": row["fpr"],
            "fpr_ci": rate_with_ci(row["fpr"], row["excluded_honest"], row["n_honest"]),
        }
    return out


def participation_sweep(tau: float = 0.40) -> Dict:
    """How do the rates move as more clients are non-participants?

    A free rider (experience withheld) must stay detectable while honest
    non-participants must not be counted as false positives.
    """
    out = {}
    for n_idle in (0, 2, 4, 6, 8, 10, 12, 14):
        spec = ScenarioSpec("participation_sweep", heterogeneity="non_iid",
                            shared_component=0.6, idle_clients=n_idle,
                            free_rider_clients=2, seed_index=8)
        row = {}
        for rule in (False, True):
            res = run_suite([spec], seed_base=VALIDATION_SEED_BASE,
                            exclusion_threshold=tau,
                            exclude_non_participants=rule)
            c = res["participation_sweep"]["confusion"]
            r = rates_from_counts(c)
            row["with_rule" if rule else "without_rule"] = {
                "confusion": c,
                "tpr": r["tpr"], "fpr": r["fpr"],
                "honest_denominator": r["honest_denominator"],
                "malicious_denominator": r["malicious_denominator"],
            }
        out[str(n_idle)] = row
    return out


# ----------------------------------------------------------------------
# Part E: realistic protocol
# ----------------------------------------------------------------------
def realistic_protocol(seeds, modes=("highway", "trace", "authentic_trace"),
                       tau: float = 0.40) -> Dict:
    """Repeat the comparison on the real federated protocol.

    The participation rule is a configuration flag, so both protocols are
    executed on exactly the same seeds and scenarios.
    """
    from experiments.detector_diagnostics import DetectorProbe, aggregate
    from experiments.experiment_runner import run_simulation_trial, EvaluationConfig
    from configs.config_loader import load_config

    config = load_config()
    eval_config = EvaluationConfig()
    out = {}
    for mode in modes:
        for rule in (False, True):
            cfg = json.loads(json.dumps(config))
            cfg.setdefault("trust_management", {})["exclude_non_participants"] = rule
            per_seed = {}
            for seed in seeds:
                probe = DetectorProbe()
                probe.install()
                try:
                    run_simulation_trial(seed=seed, config=cfg,
                                         eval_config=eval_config,
                                         mobility_mode=mode)
                finally:
                    probe.uninstall()
                per_seed[str(seed)] = aggregate(probe.rounds)
            tp = fp = fn = tn = 0
            mech = {}
            for info in per_seed.values():
                c = info["confusion"]
                tp += c["tp"]; fp += c["fp"]; fn += c["fn"]; tn += c["tn"]
                for k, v in info["fp_by_mechanism"].items():
                    mech[k] = mech.get(k, 0) + v
            pooled = rates_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
            pooled["fp_by_mechanism"] = mech
            out.setdefault(mode, {})["with_rule" if rule else "without_rule"] = {
                "pooled": pooled,
                "per_seed": {k: v["confusion"] for k, v in per_seed.items()},
            }
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _f(v, nd=4):
    return "N/A" if v is None else ("%.*f" % (nd, v))


def render(report: Dict) -> str:
    L = []
    a = L.append

    a("### A. Participation-rule comparison (synthetic harness)\n")
    a("Identical scenarios and seeds; the rule is selected on the validation suite.\n")
    a("| Suite | Rule | Honest FP | Honest denom. | Honest FPR | Free-rider TPR | Pooled TPR | Pooled FPR |")
    a("|---|---|---|---|---|---|---|---|")
    for suite in ("validation", "heldout"):
        for rule in ("False", "True"):
            r = report["protocol_comparison"][suite][rule]
            p = r["pooled"]
            a("| %s | %s | %d | %d | %s | %s | %s | %s |" % (
                suite, "excluded" if rule == "True" else "all judged",
                r["honest_false_positives"], r["honest_denominator"],
                _f(r["honest_fpr"]), _f(r["free_rider_tpr"]),
                _f(p["tpr"]), _f(p["fpr"])))
    a("")
    a("Selected rule (validation only): **%s**\n" %
      ("exclude non-participants" if report["protocol_comparison"]["selected_rule"]
       else "judge every participant"))

    a("### B. Threshold sensitivity (validation selection, held-out assessment)\n")
    ts = report["threshold_sensitivity"]
    a("| tau | validation honest FPR | validation mean attack TPR | held-out honest FPR | held-out mean attack TPR |")
    a("|---|---|---|---|---|")
    for tau in ts["grid"]:
        v = ts["validation"][str(tau)]
        h = ts["heldout"][str(tau)]
        a("| %.2f | %s | %s | %s | %s |" % (
            tau, _f(v["honest_fpr"]), _f(v["mean_attack_tpr"], 3),
            _f(h["honest_fpr"]), _f(h["mean_attack_tpr"], 3)))
    a("")
    a("Selected tau on validation: **%.2f** (interval %s); held-out at that tau: "
      "honest FPR %s, mean attack TPR %s\n" % (
          ts["selected_tau"], ts["selected_interval"],
          _f(ts["heldout"][str(ts["selected_tau"])]["honest_fpr"]),
          _f(ts["heldout"][str(ts["selected_tau"])]["mean_attack_tpr"], 3)))

    a("### C. Honest false-positive rate vs. honest heterogeneity\n")
    a("| Shared component | tp,fp,fn,tn | Honest FPR [95 % Wilson CI] |")
    a("|---|---|---|")
    for sc, row in sorted(report["heterogeneity_sweep"].items(), key=lambda kv: -float(kv[0])):
        c = row["confusion"]
        ci = row["fpr_ci"]
        a("| %s | %d,%d,%d,%d | %s [%s, %s] |" % (
            sc, c["tp"], c["fp"], c["fn"], c["tn"], _f(row["fpr"]),
            _f(ci["ci95_low"]), _f(ci["ci95_high"])))
    a("")

    a("### D. Participation sweep\n")
    a("| Idle honest clients | Rule | Confusion (tp,fp,fn,tn) | TPR | FPR |")
    a("|---|---|---|---|---|")
    for n_idle, row in sorted(report["participation_sweep"].items(), key=lambda kv: int(kv[0])):
        for key, label in (("without_rule", "all judged"),
                           ("with_rule", "exclude non-participants")):
            r = row[key]
            c = r["confusion"]
            a("| %s | %s | %d,%d,%d,%d | %s | %s |" % (
                n_idle, label, c["tp"], c["fp"], c["fn"], c["tn"],
                _f(r["tpr"], 3), _f(r["fpr"], 3)))
    a("")

    if report.get("realistic_protocol"):
        a("### E. Realistic protocol (real federated rounds, identical seeds)\n")
        a("| Mobility mode | Rule | Confusion (tp,fp,fn,tn) | Honest denom. | TPR [95 % CI] | FPR [95 % CI] | FP by mechanism |")
        a("|---|---|---|---|---|---|---|")
        for mode, rules in report["realistic_protocol"].items():
            for key, label in (("without_rule", "all judged"),
                               ("with_rule", "exclude non-participants")):
                p = rules[key]["pooled"]
                c = p["confusion"]
                mech = ", ".join("%s=%d" % (k, v)
                                 for k, v in sorted(p.get("fp_by_mechanism", {}).items()))
                a("| %s | %s | %d,%d,%d,%d | %d | %s [%s, %s] | %s [%s, %s] | %s |" % (
                    mode, label, c["tp"], c["fp"], c["fn"], c["tn"],
                    p["honest_denominator"],
                    _f(p["tpr"]), _f(p["tpr_ci"]["ci95_low"]), _f(p["tpr_ci"]["ci95_high"]),
                    _f(p["fpr"]), _f(p["fpr_ci"]["ci95_low"]), _f(p["fpr_ci"]["ci95_high"]),
                    mech or "-"))
        a("")
    return "\n".join(L)


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/detector_study.json")
    p.add_argument("--md", default="results/detector_study.md")
    p.add_argument("--realistic-seeds", type=int, nargs="*", default=[42, 101, 202])
    p.add_argument("--skip-realistic", action="store_true")
    args = p.parse_args(argv[1:])

    report = {
        "protocol_comparison": protocol_comparison(),
        "threshold_sensitivity": threshold_sensitivity(),
        "heterogeneity_sweep": heterogeneity_sweep(),
        "participation_sweep": participation_sweep(),
    }
    if not args.skip_realistic:
        report["realistic_protocol"] = realistic_protocol(args.realistic_seeds)

    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    md = render(report)
    with io.open(args.md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print("\nwritten to %s and %s" % (args.out, args.md))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
