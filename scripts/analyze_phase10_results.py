"""Phase 10 Final Remediation: Reproducible Statistical Analysis Module.

Performs:
1. Confirmatory matched paired t-tests (scipy.stats.ttest_rel), exact p-values,
   Cohen's d_z effect sizes, and Holm-Bonferroni corrections.
2. Exact Student-t confidence intervals across all cells.
3. Summary metrics JSON generation populating overall, by_fraction, ablations,
   and sensitivity without mixing disparate attack fractions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_student_t_ci_95(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Compute exact two-tailed Student-t 95% confidence interval."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": None, "stdev": None, "ci95_half": None, "ci95_low": None, "ci95_high": None, "n": 0}
    if n == 1:
        return {"mean": float(arr[0]), "stdev": 0.0, "ci95_half": 0.0, "ci95_low": float(arr[0]), "ci95_high": float(arr[0]), "n": 1}

    mean = float(np.mean(arr))
    stdev = float(np.std(arr, ddof=1))
    se = stdev / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_half = t_crit * se

    return {
        "mean": mean,
        "stdev": stdev,
        "se": se,
        "t_crit": t_crit,
        "ci95_half": ci_half,
        "ci95_low": mean - ci_half,
        "ci95_high": mean + ci_half,
        "n": n,
        "method": "student_t",
        "confidence_level": 0.95,
    }


def analyze_phase10_results(
    results_dir: Path = Path("results/phase10_final_v3"),
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Perform full statistical analysis and return structured results dictionary."""
    core_csv = results_dir / "phase10_experiment_runs.csv"
    abl_csv = results_dir / "phase10_ablation_runs.csv"
    sens_csv = results_dir / "phase10_sensitivity_runs.csv"

    df_core = pd.read_csv(core_csv)
    df_abl = pd.read_csv(abl_csv)
    df_sens = pd.read_csv(sens_csv)

    analysis_report: dict[str, Any] = {
        "metadata": {
            "results_dir": str(results_dir),
            "core_rows": len(df_core),
            "ablation_rows": len(df_abl),
            "sensitivity_rows": len(df_sens),
            "total_rows": len(df_core) + len(df_abl) + len(df_sens),
        },
        "primary_hypotheses": {},
        "secondary_hypotheses": {},
        "overall": {},
        "by_fraction": {},
        "unreliable_execution": {},
        "ablations": {},
        "sensitivity": {},
    }

    # ==============================================================
    # 1. Primary Pre-Registered Hypothesis Testing (Matched Pairs)
    # ==============================================================
    # H1: Zero-Update Free-Riding (f=0.40): FedAvg vs Trust-Threshold
    sub_zero_40 = df_core[
        (df_core["scenario_group"] == "freeriding_zero_update")
        & (df_core["attack_fraction"] == 0.40)
    ]
    fed_z40 = sub_zero_40[sub_zero_40["strategy"] == "fedavg"].sort_values("seed")
    th_z40 = sub_zero_40[sub_zero_40["strategy"] == "trust_threshold"].sort_values("seed")

    fed_vals = fed_z40["completion_rate"].to_numpy()
    th_vals = th_z40["completion_rate"].to_numpy()
    paired_diff = th_vals - fed_vals
    n_pairs = len(paired_diff)

    mean_diff = float(np.mean(paired_diff))
    std_diff = float(np.std(paired_diff, ddof=1)) if n_pairs > 1 else 0.0
    se_diff = std_diff / np.sqrt(n_pairs) if n_pairs > 0 and std_diff > 0 else 0.0
    t_crit_9 = float(stats.t.ppf(0.975, df=n_pairs - 1)) if n_pairs > 1 else 0.0
    ci_diff_half = t_crit_9 * se_diff
    cohen_dz = mean_diff / std_diff if std_diff > 0 else 0.0

    if std_diff > 1e-12:
        t_res = stats.ttest_rel(th_vals, fed_vals)
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
    else:
        t_stat = float("nan")
        p_val = float("nan")

    try:
        if np.any(paired_diff != 0):
            w_res = stats.wilcoxon(th_vals, fed_vals)
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        else:
            w_stat, w_pval = None, None
    except Exception:
        w_stat, w_pval = None, None

    analysis_report["primary_hypotheses"]["H1_zero_update_f40"] = {
        "description": "Zero-Update Free-Riding f=0.40: Trust-Threshold vs FedAvg held-out completion",
        "sample_size": n_pairs,
        "seeds": fed_z40["seed"].tolist(),
        "fedavg_completions": fed_vals.tolist(),
        "trust_threshold_completions": th_vals.tolist(),
        "paired_differences": paired_diff.tolist(),
        "fedavg_mean": float(np.mean(fed_vals)),
        "fedavg_std": float(np.std(fed_vals, ddof=1)),
        "trust_threshold_mean": float(np.mean(th_vals)),
        "trust_threshold_std": float(np.std(th_vals, ddof=1)),
        "mean_paired_difference": mean_diff,
        "std_paired_difference": std_diff,
        "se_paired_difference": se_diff,
        "ci95_difference_low": mean_diff - ci_diff_half,
        "ci95_difference_high": mean_diff + ci_diff_half,
        "t_statistic": t_stat,
        "p_value_twotailed": p_val,
        "cohen_dz": cohen_dz,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval,
        "statistically_significant_alpha_005": bool(p_val < 0.05) if not np.isnan(p_val) else False,
    }

    # Secondary confirmatory tests across primary comparisons with Holm-Bonferroni
    confirmatory_tests = [
        ("freeriding_zero_update", 0.20, "H_zero_f20"),
        ("poisoning_sign_flip", 0.20, "H_signflip_f20"),
        ("poisoning_sign_flip", 0.40, "H_signflip_f40"),
    ]
    p_values_for_correction = []
    test_keys = []

    for scen, frac, t_key in confirmatory_tests:
        sub = df_core[(df_core["scenario_group"] == scen) & (df_core["attack_fraction"] == frac)]
        sub_fed = sub[sub["strategy"] == "fedavg"].sort_values("seed")
        sub_th = sub[sub["strategy"] == "trust_threshold"].sort_values("seed")
        v_fed = sub_fed["completion_rate"].to_numpy()
        v_th = sub_th["completion_rate"].to_numpy()
        diff = v_th - v_fed
        s_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        m_diff = float(np.mean(diff))
        if s_diff > 1e-12:
            res_t = stats.ttest_rel(v_th, v_fed)
            p_v = float(res_t.pvalue)
            t_s = float(res_t.statistic)
        else:
            p_v = float("nan")
            t_s = float("nan")

        analysis_report["secondary_hypotheses"][t_key] = {
            "scenario": scen,
            "attack_fraction": frac,
            "mean_difference": m_diff,
            "std_difference": s_diff,
            "t_statistic": t_s,
            "p_value_raw": p_v,
        }
        if not np.isnan(p_v):
            p_values_for_correction.append(p_v)
            test_keys.append(t_key)

    # Apply Holm-Bonferroni step-down correction if any valid p-values exist
    if p_values_for_correction:
        sorted_pairs = sorted(zip(p_values_for_correction, test_keys))
        m_tests = len(sorted_pairs)
        for rank, (p_raw, k_test) in enumerate(sorted_pairs):
            alpha_adj = 0.05 / (m_tests - rank)
            analysis_report["secondary_hypotheses"][k_test]["holm_adjusted_alpha"] = alpha_adj
            analysis_report["secondary_hypotheses"][k_test]["holm_significant"] = bool(p_raw < alpha_adj)

    # ==============================================================
    # 2. Overall Scenario Group Summaries
    # ==============================================================
    metrics_to_summarize = [
        "completion_rate", "failure_rate", "miss_rate",
        "mean_latency_s", "mean_energy_j", "mean_reward",
        "exclusion_rate", "f1", "precision", "recall", "fpr"
    ]

    for sg in df_core["scenario_group"].unique():
        analysis_report["overall"][sg] = {}
        for strat in df_core["strategy"].unique():
            sub = df_core[(df_core["scenario_group"] == sg) & (df_core["strategy"] == strat)]
            if sub.empty:
                continue
            analysis_report["overall"][sg][strat] = {}
            for m in metrics_to_summarize:
                analysis_report["overall"][sg][strat][m] = compute_student_t_ci_95(sub[m].dropna().tolist())

    # ==============================================================
    # 3. By-Fraction Attack Summaries (No fraction pooling!)
    # ==============================================================
    attack_groups = ["poisoning_sign_flip", "poisoning_perturbation", "freeriding_zero_update", "freeriding_copy_update"]
    for ag in attack_groups:
        analysis_report["by_fraction"][ag] = {}
        sub_ag = df_core[df_core["scenario_group"] == ag]
        for frac in sorted(sub_ag["attack_fraction"].unique()):
            frac_key = f"f_{int(frac*100)}"
            analysis_report["by_fraction"][ag][frac_key] = {}
            for strat in df_core["strategy"].unique():
                sub = sub_ag[(sub_ag["attack_fraction"] == frac) & (sub_ag["strategy"] == strat)]
                if sub.empty:
                    continue
                analysis_report["by_fraction"][ag][frac_key][strat] = {}
                for m in metrics_to_summarize:
                    analysis_report["by_fraction"][ag][frac_key][strat][m] = compute_student_t_ci_95(sub[m].dropna().tolist())

    # ==============================================================
    # 4. Unreliable Execution Summary by Severity
    # ==============================================================
    unrel_groups = [sg for sg in df_core["scenario_group"].unique() if sg.startswith("unreliable_")]
    for ug in unrel_groups:
        analysis_report["unreliable_execution"][ug] = {}
        sub_ug = df_core[df_core["scenario_group"] == ug]
        for strat in df_core["strategy"].unique():
            sub = sub_ug[sub_ug["strategy"] == strat]
            if sub.empty:
                continue
            analysis_report["unreliable_execution"][ug][strat] = {
                "tasks_exposed": compute_student_t_ci_95(sub["tasks_exposed"].tolist()) if "tasks_exposed" in sub.columns else compute_student_t_ci_95(sub["injected_tasks_exposed"].tolist()),
                "tasks_with_any_fault": compute_student_t_ci_95(sub["tasks_with_any_fault"].tolist()) if "tasks_with_any_fault" in sub.columns else None,
                "injected_drops": compute_student_t_ci_95(sub["injected_drops"].tolist()),
                "injected_fails": compute_student_t_ci_95(sub["injected_fails"].tolist()),
                "injected_delays": compute_student_t_ci_95(sub["injected_delays"].tolist()),
                "completion_rate": compute_student_t_ci_95(sub["completion_rate"].tolist()),
                "mean_latency_s": compute_student_t_ci_95(sub["mean_latency_s"].tolist()),
                "mean_reward": compute_student_t_ci_95(sub["mean_reward"].tolist()),
                "exclusion_rate": compute_student_t_ci_95(sub["exclusion_rate"].tolist()),
                "fpr": compute_student_t_ci_95(sub["fpr"].dropna().tolist()),
            }

    # ==============================================================
    # 5. Ablation Summary
    # ==============================================================
    for abl in df_abl["ablation_name"].unique():
        analysis_report["ablations"][abl] = {}
        sub_abl = df_abl[df_abl["ablation_name"] == abl]
        for scen in sub_abl["scenario_group"].unique():
            sub = sub_abl[sub_abl["scenario_group"] == scen]
            analysis_report["ablations"][abl][scen] = {
                "completion_rate": compute_student_t_ci_95(sub["completion_rate"].tolist()),
                "mean_reward": compute_student_t_ci_95(sub["mean_reward"].tolist()),
                "exclusion_rate": compute_student_t_ci_95(sub["exclusion_rate"].tolist()),
                "f1": compute_student_t_ci_95(sub["f1"].dropna().tolist()),
                "precision": compute_student_t_ci_95(sub["precision"].dropna().tolist()),
                "recall": compute_student_t_ci_95(sub["recall"].dropna().tolist()),
                "fpr": compute_student_t_ci_95(sub["fpr"].dropna().tolist()),
            }

    # ==============================================================
    # 6. Sensitivity Summary
    # ==============================================================
    for param in df_sens["sweep_param"].unique():
        analysis_report["sensitivity"][param] = {}
        sub_param = df_sens[df_sens["sweep_param"] == param]
        for val in sorted(sub_param["param_value"].unique()):
            val_key = f"val_{val}"
            analysis_report["sensitivity"][param][val_key] = {}
            for scen in sub_param["scenario_group"].unique():
                sub = sub_param[(sub_param["param_value"] == val) & (sub_param["scenario_group"] == scen)]
                analysis_report["sensitivity"][param][val_key][scen] = {
                    "completion_rate": compute_student_t_ci_95(sub["completion_rate"].tolist()),
                    "mean_reward": compute_student_t_ci_95(sub["mean_reward"].tolist()),
                    "exclusion_rate": compute_student_t_ci_95(sub["exclusion_rate"].tolist()),
                    "f1": compute_student_t_ci_95(sub["f1"].dropna().tolist()),
                    "fpr": compute_student_t_ci_95(sub["fpr"].dropna().tolist()),
                }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(analysis_report, f, indent=2)
        print(f"Saved complete Phase 10 statistical analysis to {out_json}")

    return analysis_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/phase10_final_v3")
    parser.add_argument("--out-json", default="results/phase10_final_v3/phase10_summary_metrics.json")
    args = parser.parse_args()

    rep = analyze_phase10_results(Path(args.results_dir), Path(args.out_json))
    h1 = rep["primary_hypotheses"]["H1_zero_update_f40"]
    print("\n=======================================================")
    print("Primary Hypothesis H1 (Zero-Update f=0.40):")
    print(f"  FedAvg completion:          {h1['fedavg_mean']:.3f} +/- {h1['fedavg_std']:.3f}")
    print(f"  Trust-Threshold completion: {h1['trust_threshold_mean']:.3f} +/- {h1['trust_threshold_std']:.3f}")
    print(f"  Mean paired diff:           {h1['mean_paired_difference']:+.4f} (95% CI: [{h1['ci95_difference_low']:+.4f}, {h1['ci95_difference_high']:+.4f}])")
    print(f"  Paired t-test:              t({h1['sample_size']-1}) = {h1['t_statistic']}, p = {h1['p_value_twotailed']}")
    print(f"  Statistically significant:  {h1['statistically_significant_alpha_005']}")
    print("=======================================================\n")
