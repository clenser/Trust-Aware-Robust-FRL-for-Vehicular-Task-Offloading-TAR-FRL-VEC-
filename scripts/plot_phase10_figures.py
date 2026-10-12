"""
Generate publication-quality figures and summary LaTeX tables for Phase 10.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FIGS = REPO_ROOT / "reports" / "figures" / "phase10"
OUT_TABLES = REPO_ROOT / "reports" / "tables" / "phase10"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(REPO_ROOT / "results" / "phase10" / "phase10_experiment_runs.csv")
abl = pd.read_csv(REPO_ROOT / "results" / "phase10" / "phase10_ablation_runs.csv")
sens = pd.read_csv(REPO_ROOT / "results" / "phase10" / "phase10_sensitivity_runs.csv")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Figure 1: Attack Detection F1 vs Fraction (10 Seeds)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
for sg, label, color, marker in [
    ("freeriding_zero_update", "Zero-Update Free-Riding", "#16a34a", "o"),
    ("freeriding_copy_update", "Copy-Update Free-Riding", "#2563eb", "s"),
    ("poisoning_sign_flip", "Sign-Flip Poisoning", "#ea580c", "^"),
    ("poisoning_perturbation", "Gaussian Perturbation", "#dc2626", "v"),
]:
    sub = runs[(runs["scenario_group"] == sg) & (runs["strategy"] == "trust_threshold")]
    grp = sub.groupby("attack_fraction")["f1"].agg(["mean", "std"])
    ax.errorbar(
        grp.index, grp["mean"], yerr=grp["std"],
        label=label, color=color, marker=marker, capsize=4, linewidth=1.8
    )

ax.set_xlabel("Malicious / Free-Riding Fraction ($f$)", fontsize=11, fontweight="bold")
ax.set_ylabel("Detection F1 Score", fontsize=11, fontweight="bold")
ax.set_title("Phase 10: Attack Detection Across Attack Fractions (10 Seeds, 8 Rounds)", fontsize=12, fontweight="bold")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=True, loc="center right", fontsize=9)
plt.tight_layout()
fig.savefig(OUT_FIGS / "detection_f1_vs_fraction.png")
plt.close(fig)

# Figure 2: Completion Rate vs Fraction for Zero-Update (Mitigation at f=0.40)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
for strat, label, color, marker in [
    ("trust_threshold", "Trust-Threshold (Proposed)", "#16a34a", "o"),
    ("trust_weighted", "Trust-Weighted", "#2563eb", "s"),
    ("fedavg", "Standard FedAvg", "#dc2626", "x"),
]:
    sub = runs[(runs["scenario_group"] == "freeriding_zero_update") & (runs["strategy"] == strat)]
    grp = sub.groupby("attack_fraction")["completion_rate"].agg(["mean", "std"])
    ax.errorbar(
        grp.index, grp["mean"], yerr=grp["std"],
        label=label, color=color, marker=marker, capsize=4, linewidth=1.8
    )

ax.set_xlabel("Zero-Update Free-Rider Fraction ($f$)", fontsize=11, fontweight="bold")
ax.set_ylabel("Held-Out Task Completion Rate", fontsize=11, fontweight="bold")
ax.set_title("Phase 10: Task Completion Under Zero-Update Free-Riding (10 Seeds)", fontsize=12, fontweight="bold")
ax.set_ylim(0.60, 0.90)
ax.legend(frameon=True, loc="lower left", fontsize=9)
plt.tight_layout()
fig.savefig(OUT_FIGS / "completion_vs_fraction_zeroupdate.png")
plt.close(fig)

# Figure 3: Honest Non-IID FPR Distribution (10 Seeds)
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
h_sub = runs[(runs["scenario_group"] == "honest_non_iid") & (runs["strategy"] == "trust_threshold")]
ax.hist(h_sub["fpr"], bins=6, color="#0284c7", edgecolor="black", alpha=0.8)
ax.axvline(h_sub["fpr"].mean(), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean FPR: {h_sub['fpr'].mean():.1%}")
ax.axvline(0.05, color="#16a34a", linestyle=":", linewidth=2, label="5% Target Threshold")
ax.set_xlabel("Honest Non-IID False Positive Rate (FPR)", fontsize=11, fontweight="bold")
ax.set_ylabel("Seed Count (out of 10)", fontsize=11, fontweight="bold")
ax.set_title("Phase 10: Honest Non-IID FPR Across 10 Seeds (8 Rounds)", fontsize=12, fontweight="bold")
ax.legend(frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig(OUT_FIGS / "honest_non_iid_fpr_distribution.png")
plt.close(fig)

# LaTeX Table 1: Core Performance Matrix
table1 = runs.groupby(["scenario_group", "strategy"])[["completion_rate", "f1", "precision", "recall", "fpr"]].mean().round(3)
table1.to_latex(OUT_TABLES / "core_performance_matrix.tex")

# LaTeX Table 2: Ablation Study
table2 = abl.groupby("ablation_name")[["completion_rate", "f1", "fpr", "exclusion_rate"]].mean().round(3)
table2.to_latex(OUT_TABLES / "ablation_results.tex")

# LaTeX Table 3: Sensitivity Analysis
table3_alpha = sens[sens["sweep_param"] == "alpha"].groupby("param_value")[["completion_rate", "f1", "exclusion_rate"]].mean().round(3)
table3_alpha.to_latex(OUT_TABLES / "sensitivity_alpha.tex")

print("Generated Phase 10 figures and LaTeX tables successfully!")
