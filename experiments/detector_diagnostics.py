"""
Detector diagnostics: why does the trust evaluator exclude a client?

This module runs the *deployed* detector unchanged and instruments it, so
that every exclusion can be attributed to a mechanism instead of being
reported as an aggregate rate.

It answers three questions:

1. Which statistical signal caused each exclusion?
2. How do the population statistics (the robust scale and its fallback
   chain) behave on the different mobility sources?
3. How does the false-positive rate move as honest heterogeneity and
   participation patterns change?

Nothing here changes the detector.  The instrumentation is applied by
wrapping the two public entry points at run time, so the production code
path is untouched and the deployed configuration is exactly the one that
produced the active benchmark.

Usage::

    python -m experiments.detector_diagnostics --seeds 42 101 202 \
        --modes highway authentic_trace --out results/detector_diagnostics.json
"""

import io
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from trust.model_trust import ModelTrustEvaluator
from trust.trust_engine import TrustEngine
from trust.trust_model import _canonical_id


# ----------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------
class DetectorProbe:
    """Captures the per-client internals of the trust decision.

    The probe wraps (and does not modify) ``evaluate_model_updates`` and
    ``filter_trusted_participants``.  The decision itself is still taken by
    the deployed code; the probe only records what the deployed code saw.
    """

    def __init__(self):
        self.rounds: List[Dict] = []
        self._pending: Optional[Dict] = None
        self._orig_eval = ModelTrustEvaluator.evaluate_model_updates
        self._orig_filter = TrustEngine.filter_trusted_participants
        self._installed = False

    # -- installation -------------------------------------------------
    def install(self):
        if self._installed:
            return
        probe = self

        def eval_wrapper(self, client_updates_dict):
            scores = probe._orig_eval(self, client_updates_dict)
            probe._pending = {
                "details": {cid: dict(d) for cid, d in self.last_details.items()},
                "scores": dict(scores),
                "evaluator": self,
            }
            return scores

        def filter_wrapper(self, client_updates_dict, ground_truth_roles=None):
            trusted, excluded, metrics = probe._orig_filter(
                self, client_updates_dict, ground_truth_roles=ground_truth_roles)
            probe._record(self, client_updates_dict, trusted, excluded, metrics,
                          ground_truth_roles)
            return trusted, excluded, metrics

        ModelTrustEvaluator.evaluate_model_updates = eval_wrapper
        TrustEngine.filter_trusted_participants = filter_wrapper
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        ModelTrustEvaluator.evaluate_model_updates = self._orig_eval
        TrustEngine.filter_trusted_participants = self._orig_filter
        self._installed = False

    # -- recording ----------------------------------------------------
    def _record(self, engine, updates, trusted, excluded, metrics,
                ground_truth_roles):
        pending = self._pending or {"details": {}, "scores": {}}
        details = pending["details"]
        scores = pending["scores"]
        excluded_set = set(excluded)
        roles = dict(ground_truth_roles or {})

        clients = []
        for cid in updates:
            canon = _canonical_id(cid)
            role = roles.get(cid, roles.get(canon, "honest"))
            is_malicious = role is not None and role != "honest"
            d = details.get(cid, {})
            model_trust = float(scores.get(cid, 0.5))
            behavioural = float(engine.behavior_tracker.get_trust(canon))
            joint = float(engine.final_trust_scores.get(canon, 1.0))
            clients.append({
                "client_id": int(cid) if isinstance(cid, (int, np.integer)) else str(cid),
                "role": role,
                "is_malicious": bool(is_malicious),
                "excluded": cid in excluded_set,
                "model_trust": round(model_trust, 6),
                "behavioural_trust": round(behavioural, 6),
                "joint_trust": round(joint, 6),
                "reason": d.get("reason", "unknown"),
                "agreement": round(float(d.get("agreement", float("nan"))), 6)
                if "agreement" in d else None,
                "agreement_z": round(float(d.get("agreement_z", float("nan"))), 6)
                if "agreement_z" in d else None,
                "norm_ratio": round(float(d.get("norm_ratio", float("nan"))), 6)
                if "norm_ratio" in d else None,
                "dist_z": round(float(d.get("dist_z", float("nan"))), 6)
                if "dist_z" in d else None,
                "ratio_z": round(float(d.get("ratio_z", float("nan"))), 6)
                if "ratio_z" in d else None,
            })

        self.rounds.append({
            "eligible_clients": metrics.get("total_clients"),
            "excluded_count": metrics.get("excluded_count"),
            "true_positives": metrics.get("true_positives"),
            "false_positives": metrics.get("false_positives"),
            "true_negatives": metrics.get("true_negatives"),
            "false_negatives": metrics.get("false_negatives"),
            "clients": clients,
        })
        self._pending = None


# ----------------------------------------------------------------------
# Mechanism attribution
# ----------------------------------------------------------------------
def classify_mechanism(client: Dict, z_threshold: float = 3.0,
                       norm_low_ratio: float = 0.15,
                       norm_high_ratio: float = 3.0) -> str:
    """Attribute an exclusion to the signal that caused it.

    The order mirrors the deployed scoring code exactly.
    """
    if not client["excluded"]:
        return "not_excluded"
    reason = client.get("reason")
    if reason == "free_riding":
        return "free_riding_by_norm_ratio"
    a_z = client.get("agreement_z")
    d_z = client.get("dist_z")
    ratio = client.get("norm_ratio")
    extreme_agreement = (a_z is not None and a_z < -z_threshold)
    extreme_distance = (d_z is not None and d_z > z_threshold)
    scaled_up = (ratio is not None and ratio >= norm_high_ratio)
    if extreme_agreement and (scaled_up or extreme_distance):
        return "agreement_and_axis_outlier"
    if scaled_up:
        return "scale_outlier"
    if extreme_agreement:
        return "agreement_outlier"
    if client["model_trust"] <= 0.20:
        return "joint_trust_capped_by_model_trust"
    return "joint_trust_below_threshold"


def aggregate(rounds: List[Dict], z_threshold: float = 3.0) -> Dict:
    """Pool the per-client records into confusion counts and mechanisms."""
    tp = fp = fn = tn = 0
    by_mech: Dict[str, int] = {}
    fp_stats = {"agreement_z": [], "dist_z": [], "norm_ratio": []}
    tn_stats = {"agreement_z": [], "dist_z": [], "norm_ratio": []}
    fp_clients: List[Dict] = []
    for rnd in rounds:
        for c in rnd["clients"]:
            mech = classify_mechanism(c, z_threshold=z_threshold)
            if c["excluded"] and c["is_malicious"]:
                tp += 1
            elif c["excluded"] and not c["is_malicious"]:
                fp += 1
                by_mech[mech] = by_mech.get(mech, 0) + 1
                fp_clients.append(dict(c, mechanism=mech))
                for k in fp_stats:
                    if c.get(k) is not None:
                        fp_stats[k].append(float(c[k]))
            elif c["is_malicious"]:
                fn += 1
            else:
                tn += 1
                for k in tn_stats:
                    if c.get(k) is not None:
                        tn_stats[k].append(float(c[k]))

    def summary(vals):
        if not vals:
            return {"n": 0}
        a = np.asarray(vals, dtype=float)
        return {"n": int(a.size), "mean": float(a.mean()),
                "median": float(np.median(a)),
                "p05": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)),
                "min": float(a.min()), "max": float(a.max())}

    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "tpr": (tp / (tp + fn)) if (tp + fn) else None,
        "fpr": (fp / (fp + tn)) if (fp + tn) else None,
        "fnr": (fn / (tp + fn)) if (tp + fn) else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "honest_denominator": fp + tn,
        "malicious_denominator": tp + fn,
        "fp_by_mechanism": by_mech,
        "false_positive_statistics": {k: summary(v) for k, v in fp_stats.items()},
        "true_negative_statistics": {k: summary(v) for k, v in tn_stats.items()},
        "false_positive_clients": fp_clients,
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run_diagnostics(seeds, mobility_modes, attack_ratio: float = 0.24,
                    label: str = "diagnostic") -> Dict:
    """Run the deployed detector on the given mobility modes and seeds."""
    from experiments.experiment_runner import run_simulation_trial, EvaluationConfig
    from configs.config_loader import load_config

    config = load_config()
    eval_config = EvaluationConfig()
    out = {"seeds": list(seeds), "mobility_modes": list(mobility_modes),
           "attack_ratio": attack_ratio, "runs": {}}

    for mode in mobility_modes:
        per_seed = {}
        for seed in seeds:
            probe = DetectorProbe()
            probe.install()
            try:
                rec = run_simulation_trial(seed=seed, config=config,
                                           eval_config=eval_config,
                                           mobility_mode=mode,
                                           attack_ratio=attack_ratio)
            finally:
                probe.uninstall()
            agg = aggregate(probe.rounds)
            per_seed[str(seed)] = {
                "per_round": [{
                    "eligible_clients": r["eligible_clients"],
                    "excluded_count": r["excluded_count"],
                    "true_positives": r["true_positives"],
                    "false_positives": r["false_positives"],
                    "true_negatives": r["true_negatives"],
                    "false_negatives": r["false_negatives"],
                } for r in probe.rounds],
                "summary": agg,
                "avg_latency_ms": rec.get("avg_latency_ms"),
                "completion_rate": rec.get("completion_rate"),
                "avg_energy_j": rec.get("avg_energy_j"),
            }
            print("  %-16s seed %-5s  tp=%d fp=%d fn=%d tn=%d  FPR=%s  mechanisms=%s" % (
                mode, seed, agg["confusion"]["tp"], agg["confusion"]["fp"],
                agg["confusion"]["fn"], agg["confusion"]["tn"],
                "n/a" if agg["fpr"] is None else "%.4f" % agg["fpr"],
                agg["fp_by_mechanism"]))
        out["runs"][mode] = per_seed
    return out


def pooled_for_mode(per_seed: Dict) -> Dict:
    """Pool the per-seed confusion counts of one mobility mode."""
    tp = fp = fn = tn = 0
    mech: Dict[str, int] = {}
    for seed_info in per_seed.values():
        c = seed_info["summary"]["confusion"]
        tp += c["tp"]; fp += c["fp"]; fn += c["fn"]; tn += c["tn"]
        for k, v in seed_info["summary"]["fp_by_mechanism"].items():
            mech[k] = mech.get(k, 0) + v
    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "tpr": (tp / (tp + fn)) if (tp + fn) else None,
        "fpr": (fp / (fp + tn)) if (fp + tn) else None,
        "fnr": (fn / (tp + fn)) if (tp + fn) else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "honest_denominator": fp + tn,
        "malicious_denominator": tp + fn,
        "fp_by_mechanism": mech,
    }


def render(report: Dict) -> str:
    lines = []
    lines.append("### Detector diagnostics (deployed detector, instrumented)\n")
    lines.append("Seeds: %s | attack ratio: %s\n" % (report["seeds"], report["attack_ratio"]))
    lines.append("| Mobility mode | tp | fp | fn | tn | Honest denom. | TPR | FPR | FP by mechanism |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for mode, per_seed in report["runs"].items():
        p = pooled_for_mode(per_seed)
        c = p["confusion"]
        mech = ", ".join("%s=%d" % (k, v) for k, v in sorted(p["fp_by_mechanism"].items()))
        lines.append("| %s | %d | %d | %d | %d | %d | %s | %s | %s |" % (
            mode, c["tp"], c["fp"], c["fn"], c["tn"], p["honest_denominator"],
            "N/A" if p["tpr"] is None else "%.4f" % p["tpr"],
            "N/A" if p["fpr"] is None else "%.4f" % p["fpr"],
            mech or "-"))
    lines.append("")
    return "\n".join(lines)


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 101, 202])
    p.add_argument("--modes", nargs="*", default=["highway", "authentic_trace"])
    p.add_argument("--attack-ratio", type=float, default=0.24)
    p.add_argument("--out", default="results/detector_diagnostics.json")
    args = p.parse_args(argv[1:])

    report = run_diagnostics(args.seeds, args.modes, args.attack_ratio)
    out = args.out
    if out:
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=float)
        print("\nwritten to %s" % out)
    print()
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
