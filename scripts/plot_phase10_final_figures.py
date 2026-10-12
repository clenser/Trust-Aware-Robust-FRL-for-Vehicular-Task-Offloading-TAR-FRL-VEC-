"""Phase 10 Final v4: Publication Figure and LaTeX Table Generator.

Reads v4 CSVs and writes to reports/figures/phase10_final_v4/ and
reports/tables/phase10_final_v4/. Every figure/table is generated from the
given results dir only; error bars are Student-t 95% confidence intervals
(computed here, not hard-coded); legends derive from the data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

REPO_ROOT = Path(__file__).resolve().parent.parent


def ci95_half(vals) -> float:
    """Student-t 95% CI half-width for a cell; 0.0 when undefined (n<2 or zero variance)."""
    a = np.asarray(list(vals), dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 2:
        return 0.0
    s = float(a.std(ddof=1))
    if s <= 0:
        return 0.0
    return float(sstats.t.ppf(0.975, df=n - 1) * s / np.sqrt(n))


def group_mean_ci(sub: pd.DataFrame, group_keys: list[str], metric: str) -> pd.DataFrame:
    """Per-cell mean and 95% CI half-width over seeds/runs."""
    grp = sub.groupby(group_keys)[metric]
    return pd.DataFrame({"mean": grp.mean(), "ci": grp.apply(ci95_half)}).reset_index()


def generate_plots_and_tables(
    results_dir: Path = Path("results/phase10_final_v4"),
    out_figs: Path = Path("reports/figures/phase10_final_v4"),
    out_tables: Path = Path("reports/tables/phase10_final_v4"),
) -> None:
    out_figs.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    df_core = pd.read_csv(results_dir / "phase10_experiment_runs.csv")
    df_abl = pd.read_csv(results_dir / "phase10_ablation_runs.csv")
    df_sens = pd.read_csv(results_dir / "phase10_sensitivity_runs.csv")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # ==============================================================
    # Figure 1: Attack Detection F1 vs Fraction (10 Seeds)
    # ==============================================================
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for sg, label, color, marker in [
        ("freeriding_zero_update", "Zero-Update Free-Riding", "#16a34a", "o"),
        ("freeriding_copy_update", "Copy-Update Free-Riding", "#2563eb", "s"),
        ("poisoning_sign_flip", "Sign-Flip Poisoning", "#ea580c", "^"),
        ("poisoning_perturbation", "Gaussian Perturbation", "#dc2626", "v"),
    ]:
        sub = df_core[(df_core["scenario_group"] == sg) & (df_core["strategy"] == "trust_threshold")]
        if sub.empty:
            continue
        cell = group_mean_ci(sub, ["attack_fraction"], "f1").sort_values("attack_fraction")
        ax.errorbar(
            cell["attack_fraction"], cell["mean"], yerr=cell["ci"],
            label=label, color=color, marker=marker, capsize=4, linewidth=1.8
        )

    ax.set_xlabel("Malicious / Free-Riding Fraction ($f$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Detection F1 Score", fontsize=11, fontweight="bold")
    ax.set_title("Detection F1 Score Across Attack Fractions (10 Seeds, 8 Rounds; bars: 95% CI)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="center right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_figs / "detection_f1_vs_fraction.png")
    plt.close(fig)

    # ==============================================================
    # Figure 2: Task Completion Under Zero-Update Free-Riding
    # ==============================================================
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    for strat, label, color, marker in [
        ("trust_threshold", "Trust-Threshold (Proposed)", "#16a34a", "o"),
        ("trust_weighted", "Trust-Weighted", "#2563eb", "s"),
        ("fedavg", "Standard FedAvg", "#dc2626", "x"),
    ]:
        sub = df_core[(df_core["scenario_group"] == "freeriding_zero_update") & (df_core["strategy"] == strat)]
        cell = group_mean_ci(sub, ["attack_fraction"], "completion_rate").sort_values("attack_fraction")
        ax.errorbar(
            cell["attack_fraction"], cell["mean"], yerr=cell["ci"],
            label=label, color=color, marker=marker, capsize=4, linewidth=1.8
        )

    ax.set_xlabel("Zero-Update Free-Rider Fraction ($f$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Held-Out Task Completion Rate", fontsize=11, fontweight="bold")
    ax.set_title("Task Completion Under Zero-Update Free-Riding (10 Seeds; bars: 95% CI)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.55, 1.00)
    ax.legend(frameon=True, loc="lower left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_figs / "completion_vs_fraction_zeroupdate.png")
    plt.close(fig)

    # ==============================================================
    # Figure 3: Honest Non-IID Anomaly FPR Distribution
    # ==============================================================
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    h_sub = df_core[(df_core["scenario_group"] == "honest_non_iid") & (df_core["strategy"] == "trust_threshold")]
    fprs = h_sub["fpr"].dropna()
    ax.hist(fprs, bins=6, color="#0284c7", edgecolor="black", alpha=0.8)
    ax.axvline(fprs.mean(), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean Anomaly FPR: {fprs.mean():.1%}")
    ax.axvline(0.05, color="#16a34a", linestyle=":", linewidth=2, label="5% reference level (descriptive only; no bounded-FPR guarantee)")
    ax.set_xlabel("Honest Non-IID Anomaly Detection Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Seed Count (out of 10)", fontsize=11, fontweight="bold")
    ax.set_title("Honest Non-IID Anomaly Rate Across 10 Seeds (8 Rounds)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_figs / "honest_non_iid_fpr_distribution.png")
    plt.close(fig)

    # ==============================================================
    # Figure 4: Unreliable Execution by Severity
    # ==============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    unrel_rows = df_core[df_core["scenario_group"].str.startswith("unreliable_")]
    if not unrel_rows.empty:
        # Panel A: Faults Injected
        sev_order = ["low", "medium", "high"]
        grp_faults = unrel_rows.groupby("unreliable_severity")[["injected_drops", "injected_fails", "injected_delays"]].mean().reindex(sev_order)
        grp_faults.plot(kind="bar", stacked=True, ax=ax1, color=["#ef4444", "#f97316", "#eab308"], edgecolor="black")
        ax1.set_title("Injected Execution Faults by Severity", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Severity Profile", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Mean Injected Count per Trial", fontsize=10, fontweight="bold")
        ax1.legend(["Drops", "Failures", "Delays"], fontsize=9)

        # Panel B: Held-Out Completion Comparison
        grp_comp = unrel_rows.groupby(["unreliable_severity", "strategy"])["completion_rate"].mean().unstack().reindex(sev_order)
        grp_comp.plot(kind="bar", ax=ax2, colormap="viridis", edgecolor="black")
        ax2.set_title("Clean Held-Out Completion Under Unreliable Participants", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Severity Profile", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Held-Out Completion Rate", fontsize=10, fontweight="bold")
        ax2.set_ylim(0.50, 1.00)
        _handles2, _ = ax2.get_legend_handles_labels()
        ax2.legend(_handles2, list(grp_comp.columns), fontsize=9)

    plt.tight_layout()
    fig.savefig(out_figs / "unreliable_execution_severity.png")
    plt.close(fig)

    # ==============================================================
    # Figure 5: Ablation Study Comparison (7 Configurations)
    # ==============================================================
    # Figure 5: Ablation Study Comparison (7 Configurations).
    # Legend entries derive from the data columns present in the CSV.
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    abl_grp = df_abl.groupby(["ablation_name", "scenario_group"])["completion_rate"].mean().unstack()
    _scen_order = list(abl_grp.columns)
    _scen_labels = {"poisoning_sign_flip": "Sign-Flip Poisoning",
                    "freeriding_zero_update": "Zero-Update Free-Riding",
                    "freeriding_stale_replay": "Stale-Replay Free-Riding"}
    abl_grp.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="black")
    ax.set_title("Mandatory Ablation Study: Held-Out Completion Rate (5 Seeds)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Architectural Configuration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Held-Out Task Completion", fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0.50, 1.00)
    _handles5, _ = ax.get_legend_handles_labels()
    ax.legend(_handles5, [_scen_labels.get(s, s) for s in _scen_order], fontsize=9)
    plt.tight_layout()
    fig.savefig(out_figs / "ablation_comparison.png")
    plt.close(fig)

    # ==============================================================
    # Figure 6: Sensitivity Analysis Subplots (4 Parameters)
    # ==============================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)
    params = [
        ("alpha", "Trust Balance (alpha)", axes[0, 0]),
        ("tau_exc", "Exclusion Threshold (tau_exc)", axes[0, 1]),
        ("lambda_smooth", "Temporal Smoothing (lambda)", axes[1, 0]),
        ("m2_dir_threshold", "Directional Agreement Threshold (m2)", axes[1, 1]),
    ]
    for param_key, param_title, p_ax in params:
        sub = df_sens[df_sens["sweep_param"].isin([param_key, "tau_anomaly" if param_key == "m2_dir_threshold" else param_key])]
        if not sub.empty:
            piv = sub.groupby(["param_value", "scenario_group"])["completion_rate"].mean().unstack()
            present_scens = list(piv.columns)
            piv.plot(ax=p_ax, marker="o", linewidth=2)
            p_ax.set_title(param_title, fontsize=11, fontweight="bold")
            p_ax.set_xlabel("Parameter Value", fontsize=10)
            p_ax.set_ylabel("Completion Rate", fontsize=10)
            p_ax.set_ylim(0.60, 1.00)
            handles, _ = p_ax.get_legend_handles_labels()
            p_ax.legend(handles, present_scens, fontsize=8, title="Scenario (per-panel data)")

    plt.tight_layout()
    fig.savefig(out_figs / "sensitivity_sweeps.png")
    plt.close(fig)

    # ==============================================================
    # LaTeX Tables Generation
    # ==============================================================
    # Table 1: Core Performance Matrix preserving attack fraction
    core_summary = df_core.groupby(["scenario_group", "attack_fraction", "strategy"])[
        ["completion_rate", "mean_latency_s", "f1", "precision", "recall", "fpr", "exclusion_rate"]
    ].agg(["mean", "std"]).round(3)
    core_summary.to_latex(out_tables / "core_performance_matrix.tex")

    # Table 2: Ablation Study
    abl_summary = df_abl.groupby(["ablation_name", "scenario_group"])[
        ["completion_rate", "mean_reward", "f1", "fpr", "exclusion_rate"]
    ].agg(["mean", "std"]).round(3)
    abl_summary.to_latex(out_tables / "ablation_results.tex")

    # Table 3: Sensitivity Sweeps
    sens_summary = df_sens.groupby(["sweep_param", "param_value", "scenario_group"])[
        ["completion_rate", "f1", "fpr", "exclusion_rate"]
    ].agg(["mean", "std"]).round(3)
    sens_summary.to_latex(out_tables / "sensitivity_all.tex")

    # Table 4: Unreliable Execution
    if not unrel_rows.empty:
        unrel_cols = [c for c in ["tasks_exposed", "tasks_with_any_fault", "injected_drops", "injected_fails", "injected_delays", "completion_rate", "mean_latency_s"] if c in unrel_rows.columns]
        if not unrel_cols:
            unrel_cols = ["injected_tasks_exposed", "injected_drops", "injected_fails", "injected_delays", "completion_rate", "mean_latency_s"]
        unrel_summary = unrel_rows.groupby(["unreliable_severity", "strategy"])[unrel_cols].agg(["mean", "std"]).round(3)
        unrel_summary.to_latex(out_tables / "unreliable_execution.tex")

    print(f"Generated Phase 10 figures to {out_figs}")
    print(f"Generated Phase 10 LaTeX tables to {out_tables}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/phase10_final_v4")
    parser.add_argument("--out-figs", default="reports/figures/phase10_final_v4")
    parser.add_argument("--out-tables", default="reports/tables/phase10_final_v4")
    args = parser.parse_args()

    generate_plots_and_tables(Path(args.results_dir), Path(args.out_figs), Path(args.out_tables))
