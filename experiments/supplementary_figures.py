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
                         "fig13_data_plane_and_edge_calibration.png"]

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
    made = []
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
