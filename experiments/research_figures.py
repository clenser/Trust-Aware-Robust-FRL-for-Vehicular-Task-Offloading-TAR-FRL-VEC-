"""
Research figures that carry the downstream findings (figures 7-10).

Every figure is drawn from the raw per-seed records of a results file, and is
appended to the run's ``figures_manifest.json`` with its source file, SHA-256 and
record paths, in the same schema as ``experiments/plotter.py``.

    python -m experiments.research_figures                      # active run
    python -m experiments.research_figures RESULTS PLOTS_DIR META
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.claims_report import _headline_claims, paired_cells      # noqa: E402
from experiments.detector_study import wilson_interval                    # noqa: E402
from experiments.plotter import sha256_file                               # noqa: E402

#: Filenames of the figures this module produces (appended to the plotter's six).
RESEARCH_FIGURES = ["fig7_downstream_claims.png", "fig8_defence_by_attack_ratio.png",
                    "fig9_detector_comparison.png", "fig10_attack_families.png"]

BLUE, GREEN, RED, GREY, AMBER = "#1f5fbf", "#17803d", "#c0392b", "#6b7380", "#b26a00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})


def _mean_ci(cell, key="completion_rate"):
    s = cell[key]
    return s["mean"], (s["ci95"] or 0.0)


SHORT = {
    "H1": "H1  attack impact: FedAvg - clean FedAvg",
    "H2": "H2  defence recovery: trust-aware - FedAvg",
    "H3": "H3  oracle headroom: oracle - FedAvg",
    "H4": "H4  residual loss: trust-aware - clean FedAvg",
    "H5": "H5  trust-aware - Multi-Krum",
    "H6": "H6  federation gain: clean FedAvg - non-federated",
    "H7": "H7  trust-aware - Nearest-RSU",
    "H8": "H8  trust-aware - greedy resource-aware",
}


def figure_downstream(A, out):
    rows = _headline_claims(A)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xmax = max(r["res"]["ci95_high"] for r in rows)
    for i, r in enumerate(rows):
        res = r["res"]
        m, lo, hi = res["mean_difference"], res["ci95_low"], res["ci95_high"]
        col = GREEN if r["verdict"] == "SUPPORTED" else (RED if r["verdict"] == "CONTRADICTED" else GREY)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=col, ecolor=col, capsize=4)
        ax.text(xmax + 1.5, i, "%s  (p=%.3f, Holm %.3f)" % (r["verdict"].split(" (")[0].lower(), res["p_value"], r["p_holm"] or 1.0),
                va="center", fontsize=8, color=col)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(min(r["res"]["ci95_low"] for r in rows) - 1.5, xmax + 14)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT.get(r["id"], r["id"]) for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference in task completion, percentage points (95%% CI, n = %d seeds)"
                  % rows[0]["res"]["n"])
    ax.set_title("Pre-specified downstream claims (paired by seed; green = supported after Holm)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, "fig7_downstream_claims.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, [r["id"] for r in rows]


def figure_ratio(data, out):
    tr, fa = data["experiment_c_attack_ratio_sweep"], data["experiment_c2_attack_ratio_fedavg"]
    A = data["experiment_a_baseline_comparison"]
    ratios = sorted(tr, key=float)
    xs = [float(r) for r in ratios]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for cells, label, col in ((fa, "FedAvg (undefended)", RED), (tr, "Trust-aware", BLUE)):
        ms = [_mean_ci(cells[r]) for r in ratios]
        ax.errorbar(xs, [m for m, _ in ms], yerr=[c for _, c in ms], marker="o", label=label, color=col, capsize=3)
    nr = A["Nearest_RSU"]["completion_rate"]["mean"]
    ax.axhline(nr, color=GREY, ls="--", label="Nearest-RSU heuristic")
    ox = float(next(iter(sorted(tr, key=float, reverse=False))))
    ax.set_xlabel("Fraction of malicious clients")
    ax.set_ylabel("Task completion rate, % (mean, 95 % CI over seeds)")
    ax.set_title("Benefit and cost of the defence by attack ratio\n(0.24 uses 16 seeds, the other ratios 6)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = os.path.join(out, "fig8_defence_by_attack_ratio.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, ratios


def figure_detectors(A, out):
    P = A["Proposed_TFRL"]
    shadow = dict(P["pooled_shadow_confusion"])
    rows = [("deployed", P["pooled_confusion_counts"])] + [(k, v) for k, v in shadow.items() if k != "no_detection"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    w = 0.38
    xs = np.arange(len(rows))
    for j, (what, col) in enumerate((("TPR", GREEN), ("FPR", RED))):
        vals, errs = [], []
        for _, c in rows:
            k, n = (c["tp"], c["tp"] + c["fn"]) if what == "TPR" else (c["fp"], c["fp"] + c["tn"])
            lo, hi = wilson_interval(k, n)
            vals.append(k / n)
            errs.append([k / n - lo, hi - k / n])
        ax1.bar(xs + (j - 0.5) * w, vals, w, yerr=np.array(errs).T, color=col, label=what, capsize=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([r[0].replace("_", "\n") for r in rows])
    ax1.set_ylim(0, 1.08)
    ax1.legend()
    ax1.set_title("Detectors on the same real federated updates\n(pooled, Wilson 95 % CI, optimistic)")
    nrounds = len(P["per_seed_results"][0]["training_rounds"])
    fp = np.zeros(nrounds)
    hon = np.zeros(nrounds)
    for r in P["per_seed_results"]:
        for i, t in enumerate(r["training_rounds"]):
            fp[i] += t["false_positives"]
            hon[i] += t["false_positives"] + t["true_negatives"]
    ax2.plot(range(1, nrounds + 1), 100 * fp / np.maximum(hon, 1), color=RED, marker="o", ms=3)
    ax2.set_xlabel("Federated round")
    ax2.set_ylabel("Honest clients excluded, %")
    ax2.set_title("False exclusions grow as training converges\n(deployed detector, %d seeds)" % len(P["per_seed_results"]))
    fig.tight_layout()
    path = os.path.join(out, "fig9_detector_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, [r[0] for r in rows]


def figure_attacks(data, out):
    fam = data["experiment_f_attack_families"]
    attacks = sorted({k.split("/")[0] for k in fam})
    names = ("deployed", "distance_mad", "krum")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.26
    xs = np.arange(len(attacks))
    for j, name in enumerate(names):
        vals = []
        for atk in attacks:
            cell = fam[atk + "/trust_aware"]
            c = cell["pooled_confusion_counts"] if name == "deployed" else cell["pooled_shadow_confusion"][name]
            vals.append(c["tp"] / max(1, c["tp"] + c["fn"]))
        ax.bar(xs + (j - 1) * w, vals, w, label=name, color=(BLUE, GREEN, AMBER)[j])
    ax.set_xticks(xs)
    ax.set_xticklabels([a.replace("_", "\n") for a in attacks], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate (TPR)")
    ax.set_title("Detection by attack family (same updates for every detector)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out, "fig10_attack_families.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, attacks


def generate(results_file, output_dir, metadata_file=None):
    with open(results_file, "r") as f:
        data = json.load(f)
    meta = {}
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            meta = json.load(f)
    run_id = meta.get("run_id") or os.path.basename(os.path.dirname(os.path.abspath(results_file)))
    A = data["experiment_a_baseline_comparison"]
    specs = [
        (figure_downstream, (A,), "experiment_a_baseline_comparison",
         "experiment_a_baseline_comparison.<method>.per_seed_results[].completion_rate (paired by seed)",
         "Fig 7: Downstream claims - paired completion differences with Holm-adjusted verdicts",
         "Forest plot of the eight pre-specified paired comparisons of task completion."),
        (figure_ratio, (data,), "experiment_c_attack_ratio_sweep + experiment_c2_attack_ratio_fedavg",
         "experiment_c*.<ratio>.per_seed_results[].completion_rate",
         "Fig 8: Defence benefit and cost by attack ratio",
         "Completion of FedAvg and trust-aware aggregation as the malicious fraction grows."),
        (figure_detectors, (A,), "experiment_a_baseline_comparison.Proposed_TFRL",
         "Proposed_TFRL.per_seed_results[].confusion_counts / training_rounds[] / pooled_shadow_confusion",
         "Fig 9: Detector comparison on real federated updates and false exclusions over rounds",
         "Deployed detector vs shadow rules on identical updates, and the false-exclusion rate per round."),
        (figure_attacks, (data,), "experiment_f_attack_families",
         "experiment_f_attack_families.<attack>/trust_aware.pooled_(shadow_)confusion",
         "Fig 10: Detection by attack family",
         "TPR of the deployed detector and two shadow rules per attack family."),
    ]
    manifest_path = os.path.join(output_dir, "figures_manifest.json")
    manifest = {"figures": []}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    new_names = set()
    made = []
    for fn, args, src, rec, title, desc in specs:
        path, series = fn(*args, output_dir)
        name = os.path.basename(path)
        new_names.add(name)
        made.append(name)
        manifest["figures"] = [e for e in manifest["figures"] if e.get("filename") != name]
        manifest["figures"].append({
            "filename": name, "title": title, "sha256": sha256_file(path), "experiment_id": run_id,
            "source_experiment": src, "source_records": rec,
            "source_results_file": os.path.abspath(results_file),
            "source_results_sha256": sha256_file(results_file),
            "source_metadata_file": os.path.abspath(metadata_file) if metadata_file else None,
            "source_metadata_sha256": sha256_file(metadata_file) if metadata_file else None,
            "description": desc, "series": series})
        print("Generated %s" % title)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return made


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        generate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        generate(os.path.join(root, "results", "benchmark_results.json"), os.path.join(root, "plots"),
                 os.path.join(root, "results", "benchmark_results_metadata.json"))
