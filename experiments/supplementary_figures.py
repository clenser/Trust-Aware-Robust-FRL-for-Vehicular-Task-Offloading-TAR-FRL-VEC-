"""Figures for the supplementary and extended experiments (figures 11-13).

Each figure is drawn from the per-cell completion summaries of a results file
and is appended to ``plots/figures_manifest.json`` with its source file and
SHA-256, in the schema used by ``experiments/plotter.py``.

    python -m experiments.supplementary_figures                    # latest runs
    python -m experiments.supplementary_figures SUPP.json [EXT.json]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.plotter import sha256_file  # noqa: E402

#: Filenames of the figures this module produces.
SUPPLEMENTARY_FIGURES = ["fig11_faulty_edge_node.png", "fig12_intermittent_connectivity.png",
                         "fig13_data_plane_and_edge_calibration.png",
                         "fig14_do_no_harm_closed_loop.png", "fig15_learner_study.png",
                         "fig16_hypothesis_tests.png", "fig17_baselines.png"]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BLUE, GREEN, RED, GREY, AMBER = "#1f5fbf", "#17803d", "#c0392b", "#6b7380", "#b26a00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})


def _bars(ax, cells, title, colors=None):
    names = list(cells)
    means = [cells[n]["completion_rate"]["mean"] for n in names]
    cis = [cells[n]["completion_rate"].get("ci95") or 0.0 for n in names]
    y = np.arange(len(names))
    cols = colors or [BLUE if "learned" in n or "trust_aware" in n else GREY for n in names]
    ax.barh(y, means, xerr=cis, color=cols, capsize=3)
    for i, (m, c) in enumerate(zip(means, cis)):
        ax.text(m + c + 0.8, i, "%.1f" % m, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Task completion, % (mean, 95% CI over seeds)")
    ax.set_title(title, fontsize=10)


def fig_faulty(data, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    _bars(a1, data["experiment_l_faulty_edge_node"], "RSU_2 drops 60% of tasks (fixed fault)")
    _bars(a2, data["experiment_n_rotating_faulty_edge_node"], "Faulty RSU rotates every 40 steps")
    fig.suptitle("Learned policy vs heuristics under an unreliable edge node", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, "fig11_faulty_edge_node.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, ["experiment_l_faulty_edge_node", "experiment_n_rotating_faulty_edge_node"]


def fig_connectivity(data, out):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _bars(ax, data["experiment_m_intermittent_connectivity"],
          "40% of clients offline per round; offline clients miss the broadcast")
    fig.tight_layout()
    path = os.path.join(out, "fig12_intermittent_connectivity.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, ["experiment_m_intermittent_connectivity"]


def fig_extended(data, out):
    groups = [g for g in ("experiment_p_data_plane_v2v", "experiment_q_learned_data_plane",
                          "experiment_r_edge_calibration") if g in data]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.6 * len(groups), 4.8))
    axes = np.atleast_1d(axes)
    titles = {"experiment_p_data_plane_v2v": "P: malicious / selfish V2V helpers",
              "experiment_q_learned_data_plane": "Q: learned policy, data-plane attack",
              "experiment_r_edge_calibration": "R: Alibaba edge-trace calibration"}
    for ax, g in zip(axes, groups):
        cells = data[g]
        cols = [(RED if "attacked" in n else (GREEN if "Trust" in n else GREY)) for n in cells] \
            if g.endswith("v2v") else None
        _bars(ax, cells, titles[g], cols)
    fig.tight_layout()
    path = os.path.join(out, "fig13_data_plane_and_edge_calibration.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, groups


def fig_dnh(path_dir, out):
    ratios = ["0.0", "0.24", "0.4"]
    data = json.load(open(os.path.join(path_dir, "dnh_study_sign_flip.json")))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    w = 0.26
    for j, (m, col, lab) in enumerate((("fedavg", GREY, "FedAvg (no defence)"), ("trust_aware", AMBER, "deployed trust-aware"),
                                       ("do_no_harm", GREEN, "do-no-harm"))):
        vals = [data[r][m]["completion_rate"]["mean"] for r in ratios]
        errs = [data[r][m]["completion_rate"].get("ci95") or 0 for r in ratios]
        ax.bar(np.arange(3) + (j - 1) * w, vals, w, yerr=errs, color=col, capsize=3, label=lab)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["%d %% attackers" % round(float(r) * 100) for r in ratios])
    ax.set_ylim(40, 90)
    ax.set_ylabel("Task completion, % (mean, 95% CI, 12 seeds)")
    ax.set_title("Downstream completion, sign-flip attack")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    for j, (m, col) in enumerate((("trust_aware", AMBER), ("do_no_harm", GREEN))):
        fpr = []
        for r in ratios:
            c = data[r][m]["pooled_confusion_counts"]
            fpr.append(100.0 * c["fp"] / max(1, c["fp"] + c["tn"]))
        ax.bar(np.arange(3) + (j - 0.5) * 0.36, fpr, 0.36, color=col)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["0 %", "24 %", "40 %"])
    ax.set_ylabel("Honest clients excluded, %")
    ax.set_title("False exclusions in the closed loop")
    fig.tight_layout()
    path = os.path.join(out, "fig14_do_no_harm_closed_loop.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, ["dnh_study_sign_flip.json"]


def fig_learner(path_dir, out):
    data = json.load(open(os.path.join(path_dir, "learner_study.json")))
    settings = [s for s in ("default", "rotating", "peak", "dataplane") if s in data]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    series = (("Resource_Aware", GREY), ("learned_est_bandit", BLUE), ("learned_est_bandit_masked", "#6f8fbf"))
    w = 0.26
    for j, (name, col) in enumerate(series):
        vals, errs, xs = [], [], []
        for i, st in enumerate(settings):
            c = data[st].get(name)
            if c:
                vals.append(c["completion_rate"]["mean"])
                errs.append(c["completion_rate"].get("ci95") or 0)
                xs.append(i + (j - 1) * w)
        ax.bar(xs, vals, w, yerr=errs, color=col, capsize=2, label=name.replace("_", " "))
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(["benchmark env", "rotating faulty RSU", "edge-trace peak hour", "data-plane attack + V2V"][:len(settings)])
    ax.set_ylabel("Task completion, % (mean, 95% CI)")
    ax.set_ylim(40, 90)
    ax.set_title("Learned policy (delay-estimate inputs, bandit targets) vs Resource-Aware, 12 seeds", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = os.path.join(out, "fig15_learner_study.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, ["learner_study.json"]


def fig_hypotheses(path_dir, out):
    rows = json.load(open(os.path.join(path_dir, "hypothesis_verdicts.json")))["tests"]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    xmax = max(r["ci_high"] for r in rows)
    for i, r in enumerate(rows):
        ok = r["verdict"] == "SUPPORTED"
        raw = r["verdict"].startswith("SUPPORTED (")
        col = GREEN if ok else (AMBER if raw else GREY)
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]], fmt="o", color=col,
                    ecolor=col, capsize=4)
        ax.text(xmax + 2, i, "%s (Holm p=%.3f)" % (r["verdict"].split(" (")[0].lower() if not raw else "raw CI only", r["p_holm"]),
                va="center", fontsize=8, color=col)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["%s  %s  %s" % (r["id"], r["hypothesis"], r["description"]) for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(min(r["ci_low"] for r in rows) - 3, xmax + 22)
    ax.set_xlabel("Paired difference (pp of completion; ms for latency, where negative is better), 95% CI")
    ax.set_title("Pre-specified tests of the roadmap hypotheses (green = supported after Holm)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, "fig16_hypothesis_tests.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, ["hypothesis_verdicts.json"]


def fig_baselines(path_dir, out):
    a = json.load(open(os.path.join(path_dir, "baseline_verdicts.json")))["means"]
    f = json.load(open(os.path.join(path_dir, "fltrust_verdicts.json")))["means"]
    ratios = ["0.0", "0.24", "0.4"]
    series = (("fedavg", "FedAvg", GREY), ("do_no_harm", "do-no-harm", GREEN), ("multi_krum", "Multi-Krum", BLUE),
              ("coordinate_median", "coord. median", "#8aa4c8"), ("trimmed_mean", "trimmed mean", AMBER),
              ("fltrust", "FLTrust-style (adapted)", RED))
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    w = 0.13
    for j, (key, lab, col) in enumerate(series):
        vals = [(f if key == "fltrust" else a)[r][key] for r in ratios]
        ax.bar(np.arange(3) + (j - 2.5) * w, vals, w, color=col, label=lab)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["0 % attackers (12 seeds)", "24 % attackers (24 seeds)", "40 % attackers (24 seeds)"])
    ax.set_ylabel("Task completion, % (mean)")
    ax.set_ylim(20, 90)
    ax.set_title("do-no-harm vs the standard robust aggregators (sign-flip attack, paired seeds)", fontsize=11)
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    path = os.path.join(out, "fig17_baselines.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, ["baseline_verdicts.json", "fltrust_verdicts.json"]


def _latest(kind):
    manifest = json.load(open(os.path.join(ROOT, "results", "manifest.json")))
    runs = [e for e in manifest if e.get("kind") == kind]
    if not runs:
        return None
    d = os.path.join(ROOT, runs[-1]["directory"])
    return os.path.join(d, "benchmark_results.json"), os.path.join(d, "metadata.json")


def generate(supp=None, ext=None, out=os.path.join(ROOT, "plots")):
    manifest_path = os.path.join(out, "figures_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"figures": []}
    jobs = []
    if supp:
        jobs += [(fig_faulty, supp, "Fig 11: Learned policy vs heuristics under an unreliable edge node",
                  "Fixed and rotating faulty RSU (supplementary experiments L and N)."),
                 (fig_connectivity, supp, "Fig 12: Intermittent connectivity and stale updates",
                  "FedAvg vs trust-aware aggregation with offline clients (supplementary experiment M).")]
    if ext:
        jobs += [(fig_extended, ext, "Fig 13: Data-plane attack, V2V trust and edge-trace calibration",
                  "Extended experiments P, Q and R.")]
    study_jobs = []
    dnh_dir = os.path.join(ROOT, "results", "dnh_study")
    lrn_dir = os.path.join(ROOT, "results", "learner_study")
    if os.path.exists(os.path.join(dnh_dir, "dnh_study_sign_flip.json")):
        study_jobs.append((fig_dnh, dnh_dir, os.path.join(dnh_dir, "dnh_study_sign_flip.json"),
                           "Fig 14: Do-no-harm exclusion vs the deployed rule, closed loop",
                           "Completion and false exclusions of FedAvg, the deployed trust-aware rule and the do-no-harm rule."))
    if os.path.exists(os.path.join(lrn_dir, "learner_study.json")):
        study_jobs.append((fig_learner, lrn_dir, os.path.join(lrn_dir, "learner_study.json"),
                           "Fig 15: Learner study - delay-estimate inputs and bandit targets",
                           "Learned policy variants vs Resource-Aware across four environments."))
    hyp_dir = os.path.join(ROOT, "results", "hypothesis_study")
    if os.path.exists(os.path.join(hyp_dir, "hypothesis_verdicts.json")):
        study_jobs.append((fig_hypotheses, hyp_dir, os.path.join(hyp_dir, "hypothesis_verdicts.json"),
                           "Fig 16: Pre-specified hypothesis tests",
                           "Ten paired comparisons for H1-H4 with Holm-adjusted verdicts."))
    bs_dir = os.path.join(ROOT, "results", "baseline_study")
    if os.path.exists(os.path.join(bs_dir, "fltrust_verdicts.json")):
        study_jobs.append((fig_baselines, bs_dir, os.path.join(bs_dir, "baseline_verdicts.json"),
                           "Fig 17: do-no-harm vs Multi-Krum, median, trimmed mean and an FLTrust-style rule",
                           "Mean completion at 0, 24 and 40 % attackers."))
    made = []
    for fn, d, res, title, desc in study_jobs:
        path, series = fn(d, out)
        name = os.path.basename(path)
        manifest["figures"] = [e for e in manifest["figures"] if e.get("filename") != name]
        manifest["figures"].append({
            "filename": name, "title": title, "sha256": sha256_file(path), "experiment_id": "dnh_and_learner_studies",
            "source_experiment": ", ".join(series), "source_records": "<ratio|setting>.<cell>.completion_rate / pooled_confusion_counts",
            "source_results_file": os.path.abspath(res), "source_results_sha256": sha256_file(res),
            "source_metadata_file": None, "source_metadata_sha256": None,
            "description": desc, "series": series})
        made.append(name)
        print("Generated", title)
    for fn, (res, meta), title, desc in jobs:
        data = json.load(open(res))
        path, series = fn(data, out)
        name = os.path.basename(path)
        md = json.load(open(meta)) if meta and os.path.exists(meta) else {}
        manifest["figures"] = [e for e in manifest["figures"] if e.get("filename") != name]
        manifest["figures"].append({
            "filename": name, "title": title, "sha256": sha256_file(path),
            "experiment_id": md.get("run_id"), "source_experiment": ", ".join(series),
            "source_records": "<group>.<cell>.completion_rate (mean, ci95)",
            "source_results_file": os.path.abspath(res), "source_results_sha256": sha256_file(res),
            "source_metadata_file": os.path.abspath(meta) if meta else None,
            "source_metadata_sha256": sha256_file(meta) if meta and os.path.exists(meta) else None,
            "description": desc, "series": series})
        made.append(name)
        print("Generated", title)
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return made


if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = sys.argv[1]
        s = (res, os.path.join(os.path.dirname(res), "metadata.json"))
        e = None
        if len(sys.argv) > 2:
            e = (sys.argv[2], os.path.join(os.path.dirname(sys.argv[2]), "metadata.json"))
        generate(s, e)
    else:
        generate(_latest("supplementary"), _latest("extended"))
