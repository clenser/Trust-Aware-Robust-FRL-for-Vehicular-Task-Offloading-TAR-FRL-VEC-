"""
Generate the research package artefacts from the active run.

* ``results/claims_report.md``   - claim / evidence / limitation / artefact.
* ``results/detector_evaluation.md`` - per-scenario detection table with
  denominators and Wilson confidence intervals.

Everything is read from the retained artefacts of the validated active run
and recomputed with the verification tools, so the published tables cannot
drift away from the underlying records.

Usage::

    python -m experiments.claims_report            # write both artefacts
"""

import io
import json
import math
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.verify_statistics import verify, REFERENCE
from experiments.trust_audit import audit_run
from experiments.detector_study import wilson_interval, rate_with_ci

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")


def _active_run(results_dir=RESULTS):
    from web.app import _active_run_entry
    entry = _active_run_entry(results_dir)
    if entry is None:
        raise SystemExit("no active, validated run in results/manifest.json")
    return entry


def _fmt(v, nd=3):
    return "N/A" if v is None else ("%.*f" % (nd, v))


def _ci(lo, hi, nd=4):
    if lo is None or hi is None:
        return "N/A"
    return "[%.*f, %.*f]" % (nd, lo, nd, hi)


# ----------------------------------------------------------------------
# Detector evaluation table
# ----------------------------------------------------------------------
def detector_evaluation(data: Dict, audit: Dict) -> str:
    """Per-scenario detection table with denominators and uncertainty."""
    rows: List[Dict] = []
    for r in audit["rows"]:
        c = r["counts"]
        rows.append({
            "experiment": r["section"],
            "scenario": r["scenario"],
            "seed": r["seed"],
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
            "honest_denominator": c["n_honest"],
            "malicious_denominator": c["n_malicious"],
            "tpr": c["tpr"], "fpr": c["fpr"], "fnr": c["fnr"],
            "precision": c["precision"],
        })

    L = []
    a = L.append
    a("### Detector evaluation table (active run)\n")
    a("Every rate is recomputed from the raw TP/FP/FN/TN counts by "
      "`experiments/trust_audit.py`; the 95 % intervals are Wilson score "
      "intervals for a binomial proportion.  `N/A` marks a metric whose "
      "denominator is zero (mathematically undefined), never a measured "
      "zero.\n")

    a("#### Pooled per experiment and scenario\n")
    a("| Experiment | Scenario | tp | fp | fn | tn | Honest denom. | Malicious denom. | TPR [95 % CI] | FPR [95 % CI] | FNR | Precision |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in rows:
        if r["seed"] != "pooled":
            continue
        key = (r["experiment"], r["scenario"])
        if key in seen:
            continue
        seen.add(key)
        tpr_ci = rate_with_ci(r["tpr"], r["tp"], r["malicious_denominator"])
        fpr_ci = rate_with_ci(r["fpr"], r["fp"], r["honest_denominator"])
        a("| %s | %s | %d | %d | %d | %d | %d | %d | %s %s | %s %s | %s | %s |" % (
            r["experiment"], r["scenario"], r["tp"], r["fp"], r["fn"], r["tn"],
            r["honest_denominator"], r["malicious_denominator"],
            _fmt(r["tpr"]), _ci(tpr_ci["ci95_low"], tpr_ci["ci95_high"]),
            _fmt(r["fpr"]), _ci(fpr_ci["ci95_low"], fpr_ci["ci95_high"]),
            _fmt(r["fnr"]), _fmt(r["precision"])))
    a("")

    a("#### Per-seed counts of the main comparison\n")
    a("| Seed | tp | fp | fn | tn | Honest denom. | TPR | FPR [95 % CI] |")
    a("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["experiment"] != "experiment_a_baseline_comparison" or r["scenario"] != REFERENCE:
            continue
        if r["seed"] == "pooled":
            continue
        fpr_ci = rate_with_ci(r["fpr"], r["fp"], r["honest_denominator"])
        a("| %s | %d | %d | %d | %d | %d | %s | %s %s |" % (
            r["seed"], r["tp"], r["fp"], r["fn"], r["tn"], r["honest_denominator"],
            _fmt(r["tpr"]), _fmt(r["fpr"]),
            _ci(fpr_ci["ci95_low"], fpr_ci["ci95_high"])))
    a("")

    # ---- the held-out synthetic scenarios --------------------------
    cal_path = os.path.join(RESULTS, "trust_calibration.json")
    if os.path.isfile(cal_path):
        with io.open(cal_path, encoding="utf-8") as f:
            cal = json.load(f)
        a("#### Held-out validation scenarios (synthetic harness, tau = %.2f)\n"
          % cal.get("heldout", {}).get("tau", 0.40))
        a("Seed base %s - never used for threshold selection.\n"
          % cal.get("heldout", {}).get("seed_base", 9001))
        a("| Scenario | tp | fp | fn | tn | Honest denom. | TPR [95 % CI] | FPR [95 % CI] | FNR | Precision |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for row in cal.get("heldout", {}).get("scenarios", []):
            tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
            honest = row.get("n_honest", fp + tn)
            malicious = row.get("n_malicious", tp + fn)
            tpr_ci = rate_with_ci(row.get("tpr"), tp, malicious)
            fpr_ci = rate_with_ci(row.get("fpr"), fp, honest)
            a("| %s | %d | %d | %d | %d | %d | %s %s | %s %s | %s | %s |" % (
                row["scenario"], tp, fp, fn, tn, honest,
                _fmt(row.get("tpr"), 4),
                _ci(tpr_ci["ci95_low"], tpr_ci["ci95_high"]),
                _fmt(row.get("fpr"), 4),
                _ci(fpr_ci["ci95_low"], fpr_ci["ci95_high"]),
                _fmt(row.get("fnr"), 4), _fmt(row.get("precision"), 4)))
        a("")
    return "\n".join(L)


# ----------------------------------------------------------------------
# Statistics helpers for the claim verdicts
# ----------------------------------------------------------------------
from scipy import stats as _st                                   # noqa: E402
from experiments.experiment_runner import paired_comparison      # noqa: E402

#: Rule that turns a paired confidence interval into a verdict.  Written down
#: once, applied mechanically to every claim below.
VERDICT_RULE = ("SUPPORTED: the 95 % CI of the paired difference excludes 0 in the claimed "
                "direction.  CONTRADICTED: it excludes 0 in the opposite direction.  "
                "NOT ESTABLISHED: it contains 0 (the data neither show nor rule out the effect).")


def _seed_map(cell, key="completion_rate"):
    return {r["seed"]: r.get(key) for r in cell.get("per_seed_results", [])}


def paired_cells(a, b, key="completion_rate"):
    """Paired a - b over the seeds both cells have, with the minimum detectable effect."""
    A, B = _seed_map(a, key), _seed_map(b, key)
    seeds = [s for s in A if s in B and A[s] is not None and B[s] is not None]
    res = paired_comparison([A[s] for s in seeds], [B[s] for s in seeds])
    n = res["n"]
    sd = res.get("std_difference")
    if n > 1 and sd:
        # Two-sided alpha 0.05, power 0.80, paired t design.
        mde = (_st.t.ppf(0.975, n - 1) + _st.t.ppf(0.80, n - 1)) * sd / math.sqrt(n)
    else:
        mde = None
    res["mde_80"] = mde
    return res


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone), order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: (pvals[i] is None, pvals[i]))
    m = len([p for p in pvals if p is not None])
    adj = [None] * len(pvals)
    running = 0.0
    for rank, i in enumerate(idx):
        if pvals[i] is None:
            continue
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def verdict(res, direction):
    """direction = +1 (claimed a > b) or -1 (claimed a < b)."""
    if res["n"] < 2 or res.get("ci95_low") is None:
        return "NOT TESTABLE"
    lo, hi = res["ci95_low"], res["ci95_high"]
    if direction > 0:
        return "SUPPORTED" if lo > 0 else ("CONTRADICTED" if hi < 0 else "NOT ESTABLISHED")
    return "SUPPORTED" if hi < 0 else ("CONTRADICTED" if lo > 0 else "NOT ESTABLISHED")


def _cell(res, nd=2):
    if res["n"] == 0 or res.get("mean_difference") is None:
        return "N/A"
    return "%+.*f [%+.*f, %+.*f]" % (nd, res["mean_difference"], nd, res["ci95_low"], nd, res["ci95_high"])


def _rate(k, n):
    if not n:
        return "N/A"
    lo, hi = wilson_interval(k, n)
    return "%.3f [%.3f, %.3f]" % (k / n, lo, hi)


def _stat(s, nd=2):
    if not s or s.get("mean") is None:
        return "N/A"
    return "%.*f ± %.*f" % (nd, s["mean"], nd, s["ci95"] or 0.0)


def _method_table(group, title_col="Method / scenario"):
    rows = [(k, v) for k, v in group.items() if isinstance(v, dict) and "per_seed_results" in v]
    if not rows:
        return ["_No records._", ""]
    out = ["| %s | Seeds | Completion, %% | Latency (completed), ms | Reward | Energy, J/task | FL traffic, MiB |" % title_col,
           "|---|---|---|---|---|---|---|"]
    for name, v in rows:
        comm = v.get("comm_overhead_kb")
        comm = comm["mean"] if isinstance(comm, dict) else comm
        out.append("| %s | %d | %s | %s | %s | %s | %s |" % (
            name, v["num_seeds"], _stat(v["completion_rate"]), _stat(v.get("avg_completed_latency_ms")),
            _stat(v["avg_reward"]), _stat(v["avg_energy_j"], 4),
            "%.1f" % (comm / 1024.0) if comm else "0"))
    out.append("")
    return out


def _detection_row(name, c):
    return "| %s | %d | %d | %d | %d | %s | %s |" % (
        name, c["tp"], c["fn"], c["fp"], c["tn"],
        _rate(c["tp"], c["tp"] + c["fn"]), _rate(c["fp"], c["fp"] + c["tn"]))


# ----------------------------------------------------------------------
# Claims report
# ----------------------------------------------------------------------
def build(results_dir=RESULTS):
    entry = _active_run(results_dir)
    run_dir = entry.get("directory") or os.path.join(results_dir, entry["run_id"])
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT, run_dir)
    results_file = os.path.join(run_dir, "benchmark_results.json")
    meta_file = os.path.join(run_dir, "metadata.json")

    with io.open(results_file, encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    if os.path.isfile(meta_file):
        with io.open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

    stats = verify(results_file)
    audit = audit_run(results_file)
    return {"entry": entry, "run_dir": run_dir, "results_file": results_file,
            "metadata": meta, "stats": stats, "audit": audit, "data": data}


def _headline_claims(A):
    """The pre-specified downstream claims, each tested once on paired seeds."""
    P, F = A["Proposed_TFRL"], A["Standard_FedAvg_RL"]
    C, O = A["FedAvg_Clean_Reference"], A["Oracle_Exclusion_Upper_Bound"]
    N, R = A["Nearest_RSU"], A["Resource_Aware"]
    K, ND = A["Robust_FL_MultiKrum"], A["Non_Federated_D3QN"]
    spec = [
        ("H1", "The poisoning attack degrades the undefended (FedAvg) policy", F, C, -1,
         "FedAvg under attack minus FedAvg without attackers"),
        ("H2", "Trust-aware aggregation recovers task completion relative to FedAvg under attack", P, F, +1,
         "Proposed_TFRL minus FedAvg, both under attack"),
        ("H3", "Removing the true attackers (oracle) recovers completion relative to FedAvg", O, F, +1,
         "Oracle exclusion minus FedAvg; bounds what ANY detector can gain"),
        ("H4", "A residual completion loss relative to the clean reference remains under the trust-aware defence", P, C, -1,
         "Proposed_TFRL minus clean FedAvg (negative = still degraded after the defence)"),
        ("H5", "Trust-aware aggregation beats Multi-Krum", P, K, +1, "Proposed_TFRL minus Multi-Krum"),
        ("H6", "Federated learning beats non-federated learning (no attackers)", C, ND, +1,
         "Clean FedAvg minus independent per-vehicle D3QN (reference rows differ in attack ratio only for FedAvg)"),
        ("H7", "The learned trust-aware policy beats Nearest-RSU", P, N, +1, "Proposed_TFRL minus Nearest_RSU"),
        ("H8", "The learned trust-aware policy beats the greedy resource-aware heuristic", P, R, +1,
         "Proposed_TFRL minus Resource_Aware"),
    ]
    rows = []
    for hid, text, a, b, direction, note in spec:
        res = paired_cells(a, b)
        rows.append({"id": hid, "text": text, "res": res, "direction": direction, "note": note})
    adj = holm([r["res"]["p_value"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
        r["verdict"] = verdict(r["res"], r["direction"])
        # A Holm-adjusted p above 0.05 downgrades SUPPORTED to NOT ESTABLISHED.
        if r["verdict"] == "SUPPORTED" and p is not None and p >= 0.05:
            r["verdict"] = "NOT ESTABLISHED (fails Holm correction)"
    return rows


def render(info: Dict) -> str:
    data = info["data"]
    stats = info["stats"]
    audit = info["audit"]
    meta = info["metadata"]
    entry = info["entry"]
    run_id = entry["run_id"]
    A = data["experiment_a_baseline_comparison"]
    P = A["Proposed_TFRL"]

    L = []
    a = L.append

    a("# Claims, evidence, limitations and artefacts\n")
    a("Generated by `python -m experiments.claims_report` from the artefacts of the validated "
      "active run `%s`.  Every number is recomputed from the retained per-seed records; every verdict is "
      "produced by the rule below, not written by hand.\n" % run_id)

    a("## Provenance of this report\n")
    a("| Item | Value |")
    a("|---|---|")
    a("| Active run | `%s` |" % run_id)
    a("| Validation status | %s |" % meta.get("validation_status"))
    a("| Seeds (main comparison) | %s |" % (meta.get("seeds"),))
    a("| Sweep seeds | %s |" % (meta.get("sweep_seeds"),))
    a("| Evaluation protocol | %s |" % (meta.get("evaluation_config"),))
    a("| Training protocol | %s |" % ({k: meta.get("training_protocol", {}).get(k) for k in
                                      ("num_rounds", "steps_per_round", "local_epochs_gradient_steps_per_round")},))
    a("| Environment model | %s |" % (meta.get("environment_model"),))
    a("| Attack settings | %s |" % (meta.get("attack_settings"),))
    a("| Code fingerprint | %s |" % meta.get("code_revision", {}).get("content_sha256", "n/a"))
    a("| Resumed after interruption | %s |" % meta.get("resumed_from_partial_results"))
    a("| Aggregate cross-check | %s |" % ("OK - no disagreement" if not stats["problems"]
                                         else "%d disagreements" % len(stats["problems"])))
    a("| Detection-metric audit | %d disagreements over %d records |" % (audit["total_problems"], len(audit["rows"])))
    a("")

    # ---------------------------------------------------------------- headline
    a("## 1. Downstream claims (task completion, paired by seed)\n")
    a("Verdict rule: %s  The eight claims were specified before the confidence intervals were computed; "
      "the p-values are Holm-adjusted across the family of eight.\n" % VERDICT_RULE)
    a("| # | Claim | Paired difference, pp [95 % CI] | n | raw p | Holm p | Min. detectable (80 % power), pp | Verdict |")
    a("|---|---|---|---|---|---|---|---|")
    claims = _headline_claims(A)
    for c in claims:
        r = c["res"]
        a("| %s | %s | %s | %d | %s | %s | %s | **%s** |" % (
            c["id"], c["text"], _cell(r), r["n"], _fmt(r["p_value"], 4), _fmt(c["p_holm"], 4),
            _fmt(r["mde_80"], 1), c["verdict"]))
    a("")
    for c in claims:
        a("* **%s** - %s." % (c["id"], c["note"]))
    a("")
    sd_note = [c["res"]["mde_80"] for c in claims if c["res"]["mde_80"]]
    if sd_note:
        a("**Statistical power.**  With %d paired seeds the experiment can resolve completion differences of "
          "roughly %.1f-%.1f percentage points at 80 %% power.  A difference smaller than that cannot be shown "
          "or excluded by this design; a `NOT ESTABLISHED` verdict is therefore *absence of evidence*, not "
          "evidence of no effect.\n" % (claims[0]["res"]["n"], min(sd_note), max(sd_note)))

    # ---------------------------------------------------------------- per-method
    a("## 2. Per-method results\n")
    a("Latency is over completed tasks only (a failed task has no end-to-end latency; failures are measured by the "
      "completion rate).  FL traffic is the measured fp32 payload of all uploads and broadcasts.\n")
    L.extend(_method_table(A))

    # ---------------------------------------------------------------- detection
    a("## 3. Detection on real federated updates\n")
    a("Ground truth scores the detector only; it is never an input.  Counts are pooled over seeds and rounds "
      "(%d seeds); the 95 %% intervals are Wilson intervals and treat decisions as independent, which they are "
      "not (the same client is judged every round), so they are **optimistic**.  All detectors saw exactly the same "
      "updates (shadow evaluation).\n" % P["num_seeds"])
    a("| Detector | TP | FN | FP | TN | TPR [95 % CI] | FPR [95 % CI] |")
    a("|---|---|---|---|---|---|---|")
    a(_detection_row("deployed (joint trust, tau = 0.40)", P["pooled_confusion_counts"]))
    for name, c in (P.get("pooled_shadow_confusion") or {}).items():
        if name == "no_detection":
            continue
        a(_detection_row("shadow: %s" % name, c))
    a(_detection_row("Multi-Krum as the aggregator", A["Robust_FL_MultiKrum"]["pooled_confusion_counts"]))
    a("")
    per_seed_fp = [(r["seed"], r["confusion_counts"]["fp"], r["confusion_counts"]["fp"] + r["confusion_counts"]["tn"])
                   for r in P["per_seed_results"]]
    worst = sorted(per_seed_fp, key=lambda x: -x[1])[:3]
    a("False exclusions are concentrated: the three seeds with the most false positives account for %d of %d "
      "(seed, FP, honest decisions): %s.\n" % (sum(w[1] for w in worst), sum(x[1] for x in per_seed_fp), worst))

    # ---------------------------------------------------------------- attack families
    fam = data.get("experiment_f_attack_families")
    if fam:
        a("## 4. Attack families\n")
        a("Detection is the deployed detector's pooled TPR / FPR; the last column is the paired completion "
          "difference Proposed_TFRL minus FedAvg under the same attack (n = sweep seeds; exploratory, not "
          "multiplicity-corrected).\n")
        a("| Attack | TPR [95 % CI] | FPR [95 % CI] | distance_mad TPR | krum TPR | Proposed - FedAvg, pp [95 % CI] | n |")
        a("|---|---|---|---|---|---|---|")
        attacks = sorted({k.split("/")[0] for k in fam})
        for atk in attacks:
            pa, fa = fam.get(atk + "/trust_aware"), fam.get(atk + "/fedavg")
            if not pa or not fa or "per_seed_results" not in pa:
                continue
            c = pa["pooled_confusion_counts"]
            sh = pa.get("pooled_shadow_confusion") or {}
            dm, kr = sh.get("distance_mad"), sh.get("krum")
            res = paired_cells(pa, fa)
            a("| %s | %s | %s | %s | %s | %s | %d |" % (
                atk, _rate(c["tp"], c["tp"] + c["fn"]), _rate(c["fp"], c["fp"] + c["tn"]),
                _rate(dm["tp"], dm["tp"] + dm["fn"]) if dm else "N/A",
                _rate(kr["tp"], kr["tp"] + kr["fn"]) if kr else "N/A", _cell(res), res["n"]))
        a("")

    # ---------------------------------------------------------------- ratio trade-off
    tr, fr_ = data.get("experiment_c_attack_ratio_sweep"), data.get("experiment_c2_attack_ratio_fedavg")
    if tr and fr_:
        a("## 4b. Benefit and cost of the defence by attack ratio\n")
        a("Paired difference Proposed_TFRL minus FedAvg at the same attack ratio and seeds (exploratory; the sweep "
          "uses the sweep seeds, the 0.24 row uses the seeds it shares with the sweep cells).  A negative value at "
          "ratio 0 is the **cost of false exclusions**; a positive value at a high ratio is the benefit.\n")
        a("| Attack ratio | Proposed - FedAvg, pp [95 % CI] | n | Deployed FPR [95 % CI] | Deployed TPR [95 % CI] | Verdict |")
        a("|---|---|---|---|---|---|")
        for ratio in sorted(tr, key=float):
            if ratio not in fr_ or "per_seed_results" not in tr[ratio]:
                continue
            res = paired_cells(tr[ratio], fr_[ratio])
            c = tr[ratio]["pooled_confusion_counts"]
            a("| %s | %s | %d | %s | %s | %s |" % (
                ratio, _cell(res), res["n"], _rate(c["fp"], c["fp"] + c["tn"]),
                _rate(c["tp"], c["tp"] + c["fn"]), verdict(res, +1 if float(ratio) > 0 else -1)))
        a("")
        a("_Verdict column: for ratio 0 the tested claim is 'the defence costs completion' (direction -1); for "
          "positive ratios it is 'the defence helps' (direction +1)._\n")

    # ---------------------------------------------------------------- other experiments
    titles = [
        ("experiment_b_mobility_sweep", "5a. Mobility source (trust-aware)"),
        ("experiment_b2_mobility_fedavg", "5b. Mobility source (FedAvg)"),
        ("experiment_c_attack_ratio_sweep", "6a. Attack ratio (trust-aware)"),
        ("experiment_c2_attack_ratio_fedavg", "6b. Attack ratio (FedAvg)"),
        ("experiment_d_ablation", "7. Ablation of the trust components"),
        ("experiment_e_alpha_sensitivity", "8. Sensitivity to the behavioural weight alpha"),
        ("experiment_g_participation", "9. Partial participation and connectivity gating"),
        ("experiment_h_heterogeneity", "10. Task-distribution heterogeneity"),
        ("experiment_i_reward_trust_bonus", "11. Reward with the legacy trust bonus (lambda_trust = 2.5)"),
        ("experiment_j_environment_v1", "12. Environment version 1 (silent RSU re-association, uncontended uplink, trust bonus)"),
        ("experiment_k_mixed_threat", "13. Mixed threat: poisoning clients plus task-dropping helpers"),
    ]
    for key, heading in titles:
        grp = data.get(key)
        if not grp:
            continue
        a("## %s\n" % heading)
        L.extend(_method_table(grp))
        confs = {n: v["pooled_confusion_counts"] for n, v in grp.items()
                 if isinstance(v, dict) and "pooled_confusion_counts" in v
                 and sum(v["pooled_confusion_counts"].values()) > 0}
        if confs:
            a("| Scenario | TP | FN | FP | TN | TPR [95 % CI] | FPR [95 % CI] |")
            a("|---|---|---|---|---|---|---|")
            for n, c in confs.items():
                a(_detection_row(n, c))
            a("")
        if key == "experiment_j_environment_v1" and "Nearest_RSU" in grp:
            a("| Learned policy minus Nearest_RSU (same environment) | Paired difference, pp [95 % CI] | n |")
            a("|---|---|---|")
            for mname in ("trust_aware", "fedavg"):
                if mname in grp and "per_seed_results" in grp[mname]:
                    r_ = paired_cells(grp[mname], grp["Nearest_RSU"])
                    a("| v1 environment: %s | %s | %d |" % (mname, _cell(r_), r_["n"]))
            for mname, ref in (("trust_aware", "Proposed_TFRL"), ("fedavg", "Standard_FedAvg_RL")):
                r_ = paired_cells(A[ref], A["Nearest_RSU"])
                a("| v2.1 environment (main comparison): %s | %s | %d |" % (mname, _cell(r_), r_["n"]))
            a("")
        if key == "experiment_k_mixed_threat":
            a("_In this experiment the ground-truth label counts a task-dropping helper as malicious although its "
              "model update is honest, so a model-update detector cannot reach TPR 1.0 by construction (about 40 % "
              "of the positives are such helpers)._\n")

    # ---------------------------------------------------------------- state-feature change
    prev = os.path.join(ROOT, "results", "run_20261022_215233_env_v2_protocol", "benchmark_results.json")
    if os.path.isfile(prev) and run_id != "run_20261022_215233_env_v2_protocol":
        with io.open(prev, encoding="utf-8") as f:
            old = json.load(f)["experiment_a_baseline_comparison"]
        a("## 14. Effect of the state-representation change (post-hoc; both runs retained)\n")
        a("The first full run (`run_20261022_215233_env_v2_protocol`) used road-length-normalised RSU distances "
          "(17 features).  Its failure log showed that a large share of the learned policy's failures were "
          "'out of coverage', i.e. the policy had not learned RSU coverage.  The state was then changed to "
          "coverage-relative distances plus in-coverage flags (20 features).  **This change was made after "
          "seeing the first run's results**, so it is a design iteration, not a pre-registered one.  Both runs "
          "use the same seeds, environment model and training budget.\n")
        a("| Method | v2.0 features: completion, % | v2.1 features: completion, % | Paired change, pp [95 % CI] |")
        a("|---|---|---|---|")
        for m in ("Proposed_TFRL", "Standard_FedAvg_RL", "FedAvg_Clean_Reference", "Non_Federated_D3QN",
                  "Robust_FL_MultiKrum", "Nearest_RSU", "Resource_Aware", "Local_Only"):
            if m in A and m in old:
                a("| %s | %s | %s | %s |" % (m, _stat(old[m]["completion_rate"]), _stat(A[m]["completion_rate"]),
                                            _cell(paired_cells(A[m], old[m]))))
        a("")

    # ---------------------------------------------------------------- synthetic harness
    cmp_path = os.path.join(RESULTS, "detector_comparison.json")
    if os.path.isfile(cmp_path):
        with io.open(cmp_path, encoding="utf-8") as f:
            cmp_ = json.load(f)
        a("## 15. Detector comparison on the held-out synthetic scenarios\n")
        a("Identical clients, partitions, seeds and attacks for every detector (%d held-out seed bases); source "
          "`results/detector_comparison.json`.\n" % len(cmp_["seed_bases"]))
        a("| Detector | TP | FN | FP | TN | TPR [95 % CI] | FPR [95 % CI] |")
        a("|---|---|---|---|---|---|---|")
        for d in ("deployed_trust", "krum", "cosine_median", "distance_mad"):
            tot = {k: sum(sc[d][k] for sc in cmp_["scenarios"].values()) for k in ("tp", "fn", "fp", "tn")}
            a(_detection_row(d, tot))
        a("")

    # ---------------------------------------------------------------- not supported
    a("## 16. Claims that are NOT supported by this run\n")
    a("| Claim | Status | Why |")
    a("|---|---|---|")
    for c in claims:
        if c["verdict"] != "SUPPORTED":
            a("| %s | **%s** | %s: %s |" % (c["text"], c["verdict"], c["note"], _cell(c["res"])))
    en_p, en_l = P["avg_energy_j"]["mean"], A["Local_Only"]["avg_energy_j"]["mean"]
    if en_l and en_p > en_l:
        a("| Trust-aware offloading is more energy-efficient than local execution | **CONTRADICTED** | "
          "%.1fx more energy per task than local execution (%.4f vs %.4f J) |" % (en_p / en_l, en_p, en_l))
    a("| The detector has a near-zero false-positive rate in general | **CONTRADICTED** | pooled FPR on real "
      "federated updates is %s; it differs from the synthetic harness |" % _rate(
          P["pooled_confusion_counts"]["fp"],
          P["pooled_confusion_counts"]["fp"] + P["pooled_confusion_counts"]["tn"]))
    a("| The results describe a measured real-world workload | **NOT SUPPORTED** | Mobility may be replayed from an "
      "authentic trace; tasks, channel, queues, rewards and attacks are simulated |")
    a("| Behavioural trust improves detection | see section 7/8/13 | at the default weights it cannot exclude a "
      "client with an honest update (`tests/test_trust_geometry.py`) |")
    a("")

    # ---------------------------------------------------------------- figures
    a("## 17. Figures\n")
    a("Every figure is regenerated from the raw per-seed records and recorded in `figures_manifest.json` with its "
      "source file, SHA-256 and record paths.\n")
    a("| Figure | Title |")
    a("|---|---|")
    manifest_path = os.path.join(info["run_dir"], "plots", "figures_manifest.json")
    if os.path.isfile(manifest_path):
        with io.open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        for fig in man.get("figures", []):
            a("| `%s` | %s |" % (fig.get("filename"), fig.get("title")))
    a("")
    return "\n".join(L)


def main(argv):
    info = build()
    claims = render(info)
    detector = detector_evaluation(info["data"], info["audit"])
    with io.open(os.path.join(RESULTS, "claims_report.md"), "w", encoding="utf-8") as f:
        f.write(claims + "\n" + detector)
    print("written to results/claims_report.md (claims + detector evaluation table)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
