"""
Publication figure plotter.

Reads the benchmark artefacts and produces the publication figures.

Design rules
------------
* **Every statistic is recomputed from the raw per-seed records**
  (``per_seed_results``), not copied from the aggregated summary.  The
  aggregated summary is used only as a *cross-check*; a disagreement is
  reported, never silently absorbed.
* **Every figure is recorded in a manifest** (``figures_manifest.json``)
  together with the experiment identifier, the source results file, its
  SHA-256, and the JSON path of the raw records the figure was derived
  from.  A figure without a manifest entry cannot be published.
* **Stale figures are removed** from the output directory, but only from
  directories that are *not* protected as historical.  Historical run
  directories are never touched.

Figures
-------
1. Baseline comparison: latency and completion rate with 95 % CI error bars.
2. Paired difference against the reference method (the decisive evidence).
3. Malicious-client ratio sweep.
4. Ablation study.
5. Alpha sensitivity.
6. Mobility-mode comparison.
"""

import os
import json
import hashlib
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as _sps

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 15,
    'lines.linewidth': 2.0,
    'figure.autolayout': True,
})

MANIFEST_NAME = "figures_manifest.json"

# The figures that belong to the current release.  Anything else in an
# output directory is stale and must be removed (or the run refused).
CURRENT_FIGURES = (
    "fig1_baseline_comparison.png",
    "fig2_paired_differences.png",
    "fig3_attack_ratio_sweep.png",
    "fig4_ablation_study.png",
    "fig5_sensitivity_alpha.png",
    "fig6_mobility_sweep.png",
)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def sha256_file(path: str) -> str:
    """SHA-256 of a file, or ``None`` when it does not exist."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats_from_values(values):
    """Mean / std / n / 95 % CI, computed exactly like the runner does."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None, "n": 0,
                "ci95": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(_sps.t.ppf(0.975, df=n - 1) * (std / np.sqrt(n)))
    else:
        half = 0.0
    return {"mean": mean, "std": std, "n": n,
            "ci95": half, "ci95_low": mean - half, "ci95_high": mean + half}


def per_seed_values(res, key):
    """The per-seed observations of a metric, taken from the raw records."""
    if not isinstance(res, dict):
        return []
    out = []
    for r in res.get("per_seed_results", []):
        out.append(r.get(key))
    return out


def _mean(res, key):
    """Mean of a metric from the *recomputed* per-seed statistics."""
    s = per_seed_stats(res, key)
    return s["mean"]


def per_seed_stats(res, key):
    """Statistics recomputed from the raw per-seed records."""
    return stats_from_values(per_seed_values(res, key))


def _ci(res, key):
    """(low, high) of the 95 % CI of the recomputed statistics."""
    s = per_seed_stats(res, key)
    if s["ci95_low"] is None:
        return None
    return (s["ci95_low"], s["ci95_high"])


def _n(res):
    s = per_seed_stats(res, "avg_latency_ms")
    return s["n"]


def _errbars(ax, values, cis, labels):
    """Draw error bars from the CI bounds when they are available."""
    for i, (v, ci) in enumerate(zip(values, cis)):
        if v is None or ci is None:
            continue
        lo = max(0.0, v - ci[0])
        hi = max(0.0, ci[1] - v)
        ax.errorbar(i, v, yerr=[[lo], [hi]], fmt='none', ecolor='#333333',
                    capsize=4, elinewidth=1.2)


# ----------------------------------------------------------------------
# Independent recomputation of the paired comparison
# ----------------------------------------------------------------------
def recompute_paired(method_res, reference_res, key="avg_latency_ms"):
    """Paired difference ``method - reference`` recomputed from raw records.

    The sign convention is explicit: the reported difference is always
    ``method - reference`` (i.e. *reference* = Proposed_TFRL).  A positive
    latency difference therefore means the compared method was *slower*.
    """
    ref_by_seed = {r["seed"]: r for r in reference_res.get("per_seed_results", [])}
    a_vals, b_vals = [], []
    for r in method_res.get("per_seed_results", []):
        ref_r = ref_by_seed.get(r["seed"])
        if ref_r is None:
            continue
        a, b = r.get(key), ref_r.get(key)
        if a is None or b is None:
            continue
        a_vals.append(a)
        b_vals.append(b)
    n = len(a_vals)
    if n == 0:
        return {"n": 0, "mean_difference": None, "std_difference": None,
                "ci95": None, "ci95_low": None, "ci95_high": None,
                "t_statistic": None, "p_value": None, "cohens_d": None}
    diffs = np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)
    mean_diff = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    half = float(_sps.t.ppf(0.975, df=n - 1) * (sd / np.sqrt(n))) if n > 1 else 0.0
    if n > 1 and sd > 0:
        t_stat, p_value = _sps.ttest_rel(np.asarray(a_vals, dtype=float),
                                         np.asarray(b_vals, dtype=float))
        t_stat, p_value = float(t_stat), float(p_value)
        cohen_d = float(mean_diff / sd)
    else:
        t_stat, p_value, cohen_d = None, None, None
    return {"n": n, "mean_difference": mean_diff, "std_difference": sd,
            "ci95": half, "ci95_low": mean_diff - half, "ci95_high": mean_diff + half,
            "t_statistic": t_stat, "p_value": p_value, "cohens_d": cohen_d,
            "significant_at_0_05": (p_value is not None and p_value < 0.05)}


# ----------------------------------------------------------------------
# Cross-check against the stored aggregates
# ----------------------------------------------------------------------
def cross_check_aggregates(data, tolerance=1e-4):
    """Compare the recomputed statistics with the stored aggregates.

    Returns a list of human-readable disagreements.  An empty list means
    the published summary tables agree with the raw per-seed records.
    """
    problems = []

    def compare(section, name, key, stored, recomputed):
        if stored is None and recomputed is None:
            return
        if stored is None or recomputed is None:
            problems.append("%s/%s/%s: stored=%r recomputed=%r"
                            % (section, name, key, stored, recomputed))
            return
        s = stored.get("mean") if isinstance(stored, dict) else stored
        if s is None or recomputed["mean"] is None:
            return
        if abs(float(s) - float(recomputed["mean"])) > tolerance:
            problems.append("%s/%s/%s: stored=%.8f recomputed=%.8f"
                            % (section, name, key, float(s), recomputed["mean"]))

    for section, value in (data or {}).items():
        if not isinstance(value, dict):
            continue
        if "per_seed_results" in value:
            for key in ("avg_latency_ms", "completion_rate", "avg_energy_j", "avg_reward"):
                compare(section, "-", key, value.get(key), per_seed_stats(value, key))
            continue
        for name, res in value.items():
            if not isinstance(res, dict) or "per_seed_results" not in res:
                continue
            for key in ("avg_latency_ms", "completion_rate", "avg_energy_j", "avg_reward"):
                compare(section, name, key, res.get(key), per_seed_stats(res, key))

    # The pooled confusion counts must equal the sum of the per-seed counts.
    for section, value in (data or {}).items():
        if not isinstance(value, dict):
            continue
        groups = [value] if "pooled_confusion_counts" in value else [
            res for res in value.values()
            if isinstance(res, dict) and "pooled_confusion_counts" in res]
        for res in groups:
            pooled = res.get("pooled_confusion_counts") or {}
            agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for r in res.get("per_seed_results", []):
                c = r.get("confusion_counts") or {}
                for k in agg:
                    agg[k] += int(c.get(k, 0) or 0)
            for k in agg:
                if int(pooled.get(k, 0) or 0) != agg[k]:
                    problems.append("%s/%s pooled %s: stored=%d recomputed=%d"
                                    % (section, res.get("method", "-"), k,
                                       int(pooled.get(k, 0) or 0), agg[k]))
    return problems


# ----------------------------------------------------------------------
# Stale-figure cleaning
# ----------------------------------------------------------------------
def protected_plots_dirs(root="results", manifest_name="manifest.json"):
    """Directories whose figures must never be removed.

    A run directory is protected when its manifest entry marks it as
    historical (a ``note``) or superseded (``superseded: true``).
    """
    protected = set()
    manifest_path = os.path.join(root, manifest_name)
    if not os.path.isfile(manifest_path):
        return protected
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return protected
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        is_historical = bool(entry.get("note")) or bool(entry.get("superseded"))
        if not is_historical:
            continue
        directory = entry.get("directory") or ""
        base = os.path.dirname(directory) if os.path.basename(directory) == "plots" \
            else directory
        candidate = os.path.join(base, "plots")
        if os.path.isdir(candidate):
            protected.add(os.path.abspath(candidate))
    return protected


def clean_output_dir(output_dir, keep_filenames, root="results"):
    """Remove stale figures from ``output_dir``.

    Only files matching ``fig*.png`` are considered, and the directory is
    refused if it is a protected historical directory.
    """
    abs_dir = os.path.abspath(output_dir)
    if abs_dir in protected_plots_dirs(root):
        raise RuntimeError("refusing to clean protected historical directory %s"
                           % output_dir)
    removed = []
    if not os.path.isdir(abs_dir):
        return removed
    for name in sorted(os.listdir(abs_dir)):
        if not (name.startswith("fig") and name.endswith(".png")):
            continue
        if name in keep_filenames:
            continue
        os.remove(os.path.join(abs_dir, name))
        removed.append(name)
    return removed


# ----------------------------------------------------------------------
# Figure generation
# ----------------------------------------------------------------------
def generate_all_publication_plots(results_file="results/benchmark_results.json",
                                   output_dir="plots", metadata_file=None,
                                   clean=False, write_manifest=True,
                                   strict=True):
    """Generate every publication figure from a results file.

    Returns the list of generated figure filenames.  When ``clean`` is
    true, stale figures are removed from ``output_dir`` afterwards.
    ``strict`` makes an aggregate cross-check failure an error.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(results_file):
        print("Results file %s not found." % results_file)
        return []

    with open(results_file, "r") as f:
        data = json.load(f)

    meta = {}
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            meta = json.load(f)

    run_id = meta.get("run_id") or os.path.basename(
        os.path.dirname(os.path.abspath(results_file)))
    results_sha = sha256_file(results_file)
    metadata_sha = sha256_file(metadata_file) if metadata_file else None

    generated = []
    figures = []
    n_seeds = meta.get("num_seeds", "?")

    # Latency basis.  Runs from environment model 2 report the latency of
    # COMPLETED tasks; older runs only have the mean that also contains the
    # penalty value booked for failed tasks.  Use the former when it exists and
    # say which one is plotted.
    _first = next((v for v in (data.get("experiment_a_baseline_comparison") or {}).values()
                   if isinstance(v, dict) and "per_seed_results" in v), None)
    LAT_KEY = ("avg_completed_latency_ms"
               if _first and "avg_completed_latency_ms" in (_first.get("per_seed_results") or [{}])[0]
               else "avg_latency_ms")
    LAT_LABEL = ("Latency of completed tasks (ms)" if LAT_KEY == "avg_completed_latency_ms"
                 else "Average task latency incl. failure penalty (ms)")
    LAT_STORED = "completed_latency_ms" if LAT_KEY == "avg_completed_latency_ms" else "latency_ms"

    def record(filename, source_experiment, source_records, description,
               series=None, title=None):
        generated.append(filename)
        figures.append({
            "filename": filename,
            "title": title or filename,
            "sha256": sha256_file(os.path.join(output_dir, filename)),
            "experiment_id": run_id,
            "source_experiment": source_experiment,
            "source_records": source_records,
            "source_results_file": os.path.abspath(results_file),
            "source_results_sha256": results_sha,
            "source_metadata_file": os.path.abspath(metadata_file) if metadata_file else None,
            "source_metadata_sha256": metadata_sha,
            "description": description,
            "series": series or [],
        })

    # ---------------- Figure 1: baseline comparison ----------------
    exp_a = data.get("experiment_a_baseline_comparison") or data.get("experiment_a") or {}
    if exp_a:
        methods = list(exp_a.keys())
        labels = [m.replace("_", " ") for m in methods]
        latencies = [_mean(exp_a[m], LAT_KEY) for m in methods]
        lat_cis = [_ci(exp_a[m], LAT_KEY) for m in methods]
        completion = [_mean(exp_a[m], "completion_rate") for m in methods]
        comp_cis = [_ci(exp_a[m], "completion_rate") for m in methods]
        colors = ['#2ca02c' if 'Proposed' in m else '#1f77b4' for m in methods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        xs = np.arange(len(methods))
        b1 = ax1.bar(xs, [v or 0 for v in latencies], color=colors,
                      edgecolor='black', alpha=0.85)
        _errbars(ax1, latencies, lat_cis, labels)
        ax1.set_ylabel(LAT_LABEL)
        ax1.set_title("(a) Offloading Latency Across Baselines\n(n=%s seeds, 95%% CI)" % n_seeds)
        ax1.set_xticks(xs)
        ax1.set_xticklabels(labels, rotation=25, ha='right')
        for b, v in zip(b1, latencies):
            if v is not None:
                ax1.text(b.get_x() + b.get_width() / 2., b.get_height(),
                         "%.1f" % v, ha='center', va='bottom', fontsize=9)

        b2 = ax2.bar(xs, [v or 0 for v in completion], color=colors,
                      edgecolor='black', alpha=0.85)
        _errbars(ax2, completion, comp_cis, labels)
        ax2.set_ylabel("Task Completion Rate (%)")
        ax2.set_title("(b) Task Completion Rate Across Baselines\n(n=%s seeds, 95%% CI)" % n_seeds)
        ax2.set_ylim(0, 108)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(labels, rotation=25, ha='right')
        for b, v in zip(b2, completion):
            if v is not None:
                ax2.text(b.get_x() + b.get_width() / 2., b.get_height(),
                         "%.1f%%" % v, ha='center', va='bottom', fontsize=9)

        fig.savefig(os.path.join(output_dir, "fig1_baseline_comparison.png"), dpi=200)
        plt.close(fig)
        record("fig1_baseline_comparison.png",
               "experiment_a_baseline_comparison",
               "experiment_a_baseline_comparison.<method>.per_seed_results[]",
               "Latency and task-completion rate per method, recomputed from "
               "the per-seed records of the main comparison.",
               series=methods,
               title="Fig 1: Baseline latency and task-completion rate")
        print("Generated Fig 1: baseline comparison (n=%s seeds)" % n_seeds)

    # ---------------- Figure 2: paired difference ----------------
    reference = "Proposed_TFRL"
    exp_a2 = data.get("experiment_a_baseline_comparison") or {}
    stored_paired = data.get("experiment_a_paired_comparison") or {}
    if exp_a2 and reference in exp_a2:
        names = [m for m in exp_a2.keys() if m != reference]
        recomputed = {}
        for name in names:
            recomputed[name] = recompute_paired(exp_a2[name], exp_a2[reference],
                                                 LAT_KEY)
        means = [recomputed[n]["mean_difference"] for n in names]
        los = [recomputed[n]["ci95_low"] for n in names]
        his = [recomputed[n]["ci95_high"] for n in names]
        pvals = [recomputed[n]["p_value"] for n in names]
        ns = [recomputed[n]["n"] for n in names]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ys = np.arange(len(names))
        for i, (m, lo, hi) in enumerate(zip(means, los, his)):
            if m is None:
                continue
            ax.errorbar(m, i, xerr=[[max(0, m - lo)], [max(0, hi - m)]],
                        fmt='o', color='#1f77b4', capsize=4, ecolor='#333333')
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2)
        ax.set_yticks(ys)
        ax.set_yticklabels([n.replace("_", " ") for n in names])
        ax.set_xlabel(("Paired latency difference (%s): compared method - Proposed TFRL (ms)\n"
                       % LAT_LABEL.split(" (")[0])
                      + "positive = compared method slower; CI crossing 0 = difference not established")
        ax.set_title("Paired Method Differences (95%% CI, n=%s seeds)\n"
                     "recomputed from the per-seed records" % n_seeds)
        for i, (m, p, n) in enumerate(zip(means, pvals, ns)):
            if m is None:
                continue
            sig = "" if p is None else ("  p=%.3g%s" % (p, " *" if p < 0.05 else ""))
            ax.text(m, i + 0.22, "%.2f ms (n=%d)%s" % (m, n or 0, sig),
                    fontsize=8, ha="center")
        fig.savefig(os.path.join(output_dir, "fig2_paired_differences.png"), dpi=200)
        plt.close(fig)
        record("fig2_paired_differences.png",
               "experiment_a_baseline_comparison + experiment_a_paired_comparison",
               "experiment_a_baseline_comparison.<method>.per_seed_results[] "
               "(paired against Proposed_TFRL by seed)",
               "Paired latency difference of every method against Proposed TFRL; "
               "sign convention: difference = compared method - Proposed TFRL.",
               series=names,
               title="Fig 2: Paired latency differences against Proposed TFRL")
        print("Generated Fig 2: paired differences")

        # Cross-check the recomputed pairing against the stored one.
        for name in names:
            stored = (stored_paired.get(name) or {}).get(LAT_STORED) or {}
            fresh = recomputed[name]
            for key in ("mean_difference", "ci95_low", "ci95_high"):
                a, b = stored.get(key), fresh.get(key)
                if a is None or b is None:
                    continue
                if abs(float(a) - float(b)) > 1e-4:
                    msg = ("paired %s %s: stored=%.6f recomputed=%.6f"
                           % (name, key, float(a), float(b)))
                    if strict:
                        raise AssertionError(msg)
                    print("WARNING: " + msg)

    # ---------------- Figure 3: attack ratio sweep ----------------
    exp_c = data.get("experiment_c_attack_ratio_sweep") or {}
    if exp_c:
        ratios = sorted(float(k) for k in exp_c.keys())
        series = {}
        for name, res in exp_c.items():
            if not isinstance(res, dict) or LAT_KEY not in res:
                continue
            series[name] = [_mean(res, LAT_KEY) for r in ratios]
        if series:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            for name, values in series.items():
                ax1.plot([r * 100 for r in ratios], values, marker='o',
                         label=name.replace("_", " "))
            ax1.set_xlabel("Malicious client ratio (%)")
            ax1.set_ylabel(LAT_LABEL)
            ax1.set_title("(a) Latency vs. malicious-client ratio")
            ax1.legend(fontsize=8)
            for name, res in exp_c.items():
                if not isinstance(res, dict) or LAT_KEY not in res:
                    continue
                vals = [_mean(res, LAT_KEY) for r in ratios]
                ax2.plot([r * 100 for r in ratios], vals, marker='o',
                         label=name.replace("_", " "))
            ax2.set_xlabel("Malicious client ratio (%)")
            ax2.set_ylabel(LAT_LABEL)
            ax2.set_title("(b) Detection performance is reported separately")
            fig.savefig(os.path.join(output_dir, "fig3_attack_ratio_sweep.png"), dpi=200)
            plt.close(fig)
            record("fig3_attack_ratio_sweep.png",
                   "experiment_c_attack_ratio_sweep",
                   "experiment_c_attack_ratio_sweep.<ratio>.per_seed_results[]",
                   "Latency as a function of the malicious-client ratio.",
                   series=sorted(exp_c.keys()),
                   title="Fig 3: Latency vs. malicious-client ratio")
            print("Generated Fig 3: attack ratio sweep")

    # ---------------- Figure 4: ablation ----------------
    exp_d = data.get("experiment_d_ablation") or {}
    if exp_d:
        names = list(exp_d.keys())
        lat = [_mean(exp_d[n], LAT_KEY) for n in names]
        comp = [_mean(exp_d[n], "completion_rate") for n in names]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        xs = np.arange(len(names))
        colors = ['#2ca02c' if n == "Full_Proposed" else '#1f77b4' for n in names]
        ax1.bar(xs, [v or 0 for v in lat], color=colors, edgecolor='black', alpha=0.85)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([n.replace("_", " ") for n in names], rotation=25, ha='right')
        ax1.set_ylabel(LAT_LABEL)
        ax1.set_title("(a) Ablation: latency (n=%s seeds)" % n_seeds)
        ax2.bar(xs, [v or 0 for v in comp], color=colors, edgecolor='black', alpha=0.85)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([n.replace("_", " ") for n in names], rotation=25, ha='right')
        ax2.set_ylabel("Completion rate (%)")
        ax2.set_ylim(0, 108)
        ax2.set_title("(b) Ablation: completion rate (n=%s seeds)" % n_seeds)
        fig.savefig(os.path.join(output_dir, "fig4_ablation_study.png"), dpi=200)
        plt.close(fig)
        record("fig4_ablation_study.png",
               "experiment_d_ablation",
               "experiment_d_ablation.<variant>.per_seed_results[]",
               "Ablation of the two trust components.",
               series=names,
               title="Fig 4: Ablation of the trust components")
        print("Generated Fig 4: ablation study")

    # ---------------- Figure 5: alpha sensitivity ----------------
    exp_e = data.get("experiment_e_alpha_sensitivity") or {}
    if exp_e:
        alphas = sorted(float(k) for k in exp_e.keys())
        lat = [_mean(exp_e[str(a)] if str(a) in exp_e else exp_e[a], LAT_KEY)
               for a in alphas]
        comp = [_mean(exp_e[str(a)] if str(a) in exp_e else exp_e[a], "completion_rate")
                for a in alphas]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        ax1.plot(alphas, lat, marker='o', color='#1f77b4')
        ax1.set_xlabel("Joint trust weight alpha")
        ax1.set_ylabel(LAT_LABEL)
        ax1.set_title("(a) Latency vs. alpha")
        ax2.plot(alphas, comp, marker='s', color='#2ca02c')
        ax2.set_xlabel("Joint trust weight alpha")
        ax2.set_ylabel("Completion rate (%)")
        ax2.set_title("(b) Completion vs. alpha")
        fig.savefig(os.path.join(output_dir, "fig5_sensitivity_alpha.png"), dpi=200)
        plt.close(fig)
        record("fig5_sensitivity_alpha.png",
               "experiment_e_alpha_sensitivity",
               "experiment_e_alpha_sensitivity.<alpha>.per_seed_results[]",
               "Sensitivity of latency and completion to the trust-mixing weight.",
               series=[str(a) for a in alphas],
               title="Fig 5: Sensitivity to the joint-trust weight alpha")
        print("Generated Fig 5: alpha sensitivity")

    # ---------------- Figure 6: mobility comparison ----------------
    exp_b = data.get("experiment_b_mobility_sweep") or {}
    usable = {k: v for k, v in exp_b.items()
              if isinstance(v, dict) and LAT_KEY in v}
    if usable:
        names = list(usable.keys())
        lat = [_mean(usable[n], LAT_KEY) for n in names]
        comp = [_mean(usable[n], "completion_rate") for n in names]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        xs = np.arange(len(names))
        ax1.bar(xs, [v or 0 for v in lat], color='#1f77b4', edgecolor='black', alpha=0.85)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, ha='right')
        ax1.set_ylabel(LAT_LABEL)
        ax1.set_title("(a) Latency by mobility mode")
        ax2.bar(xs, [v or 0 for v in comp], color='#2ca02c', edgecolor='black', alpha=0.85)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, ha='right')
        ax2.set_ylabel("Task completion rate (%)")
        ax2.set_ylim(0, 108)
        ax2.set_title("(b) Completion by mobility mode")
        fig.savefig(os.path.join(output_dir, "fig6_mobility_sweep.png"), dpi=200)
        plt.close(fig)
        record("fig6_mobility_sweep.png",
               "experiment_b_mobility_sweep",
               "experiment_b_mobility_sweep.<mode>.per_seed_results[]",
               "Latency and completion by mobility mode.",
               series=names,
               title="Fig 6: Latency and completion by mobility mode")
        print("Generated Fig 6: mobility sweep")

    # ---------------- Aggregate cross-check ----------------
    problems = cross_check_aggregates(data)
    if problems:
        msg = "aggregate cross-check failed:\n  " + "\n  ".join(problems)
        if strict:
            raise AssertionError(msg)
        print("WARNING: " + msg)
    else:
        print("Aggregate cross-check: recomputed statistics agree with the "
              "published summary for every method.")

    # ---------------- Stale-figure cleaning ----------------
    if clean:
        removed = clean_output_dir(output_dir, set(generated))
        if removed:
            print("Removed stale figures: %s" % ", ".join(removed))

    # ---------------- Manifest ----------------
    if write_manifest:
        manifest = {
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_id": run_id,
            "results_file": os.path.abspath(results_file),
            "results_file_sha256": results_sha,
            "metadata_file": os.path.abspath(metadata_file) if metadata_file else None,
            "metadata_file_sha256": metadata_sha,
            "output_dir": os.path.abspath(output_dir),
            "num_seeds": meta.get("num_seeds"),
            "seeds": meta.get("seeds"),
            "cross_check_problems": problems,
            "figures": figures,
        }
        with open(os.path.join(output_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Figure manifest written to %s" % os.path.join(output_dir, MANIFEST_NAME))

    return generated


if __name__ == "__main__":
    import sys
    results = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.json"
    meta = "results/benchmark_results_metadata.json"
    out = "plots"
    clean = False
    args = sys.argv[1:]
    if len(args) > 1:
        out = args[1]
    if "--clean" in args:
        clean = True
    if "--meta" in args:
        meta = args[args.index("--meta") + 1]
    generate_all_publication_plots(results, out, meta, clean=clean)
