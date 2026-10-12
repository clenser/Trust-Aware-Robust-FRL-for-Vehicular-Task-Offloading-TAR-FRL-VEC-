"""
Streamlit Research Dashboard for Trust-Aware Federated RL in VANETs.
Provides an interactive, publication-grade research interface:
- High-level KPI cards and overview
- Scenario and attack vulnerability matrix
- Federated learning convergence & ablation
- Trust estimation & anomaly detection diagnostics
- Interactive vehicular environment simulator
- Research reproducibility, manifest viewer, and data exports
"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt

from dashboard.styles import get_custom_css
from dashboard.data_loader import (
    get_available_datasets,
    load_experiment_data,
    load_summary_metrics,
    load_manifest,
    verify_dataset_integrity,
    load_config_yaml,
    get_available_checkpoints,
    get_available_figures,
    REPO_ROOT
)
from dashboard.components import (
    render_kpi_header,
    render_scenario_overview_table,
    render_detection_metrics_chart,
    render_performance_comparison_chart,
    render_interactive_simulator
)

# Page configuration
st.set_page_config(
    page_title="VANET Trust-Aware FRL Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.title("VANET FRL Research")
    st.sidebar.caption("Trust-Aware Federated Reinforcement Learning in Vehicular Edge Networks")

    # Dataset selection
    datasets = get_available_datasets()
    if not datasets:
        st.error("No experimental results found in `results/`. Please verify project structure.")
        return

    selected_dataset_label = st.sidebar.selectbox(
        "Experiment Horizon Dataset",
        options=list(datasets.keys()),
        index=0,
        help="Select a results suite. Phase 10 Final suites use an 8-round horizon; older Phase 8 suites use 4-round and 2-round horizons."
    )
    dataset_info = datasets[selected_dataset_label]
    df = load_experiment_data(dataset_info["csv_path"])
    summary_metrics = load_summary_metrics(dataset_info["json_path"])
    manifest = load_manifest(dataset_info["manifest_path"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset:** `{dataset_info['key']}`")
    st.sidebar.markdown(f"**Horizon:** {dataset_info['rounds']} FL Rounds")
    st.sidebar.markdown(f"**Core Runs (this table):** {len(df)}")
    if "ablation_csv_path" in dataset_info:
        abl_df = load_experiment_data(Path(dataset_info["ablation_csv_path"]))
        sens_df = load_experiment_data(Path(dataset_info["sensitivity_csv_path"]))
        st.sidebar.markdown(f"**Ablation Runs:** {len(abl_df)}")
        st.sidebar.markdown(f"**Sensitivity Runs:** {len(sens_df)}")
        st.sidebar.markdown(f"**Suite Total:** {len(df) + len(abl_df) + len(sens_df)}")
        st.sidebar.caption(
            "The main table shows the core matrix only. "
            "Ablation and sensitivity results are in the Federated Convergence tab."
        )
    # Dynamic integrity check
    integrity_info = verify_dataset_integrity(Path(dataset_info.get("manifest_path", "")))
    st.sidebar.markdown(f"**Integrity:** {integrity_info['label']}")
    if integrity_info["status"] != "verified":
        st.sidebar.caption(integrity_info["message"])

    # Filter controls in sidebar
    st.sidebar.markdown("### Experiment Filters")
    available_scenarios = ["All"] + sorted(df["scenario_label"].unique().tolist()) if not df.empty else ["All"]
    selected_scenario = st.sidebar.selectbox("Filter by Scenario", available_scenarios, index=0)

    available_strategies = ["All"] + sorted(df["strategy_label"].unique().tolist()) if not df.empty else ["All"]
    selected_strategy = st.sidebar.selectbox("Filter by Strategy", available_strategies, index=0)

    available_attack_types = ["All"] + sorted(df["attack_type"].unique().tolist()) if not df.empty and "attack_type" in df.columns else ["All"]
    selected_attack_type = st.sidebar.selectbox("Filter by Attack Type", available_attack_types, index=0)

    fractions_list = ["All"] + [f"{f:.1f}" for f in sorted(df["attack_fraction"].unique())] if not df.empty and "attack_fraction" in df.columns else ["All"]
    selected_fraction = st.sidebar.selectbox("Filter by Attack Fraction", fractions_list, index=0)

    seeds_list = ["All"] + [str(s) for s in sorted(df["seed"].unique())] if not df.empty and "seed" in df.columns else ["All"]
    selected_seed = st.sidebar.selectbox("Filter by Seed", seeds_list, index=0)

    # Apply filters
    filtered_df = df.copy()
    if selected_scenario != "All":
        filtered_df = filtered_df[filtered_df["scenario_label"] == selected_scenario]
    if selected_strategy != "All":
        filtered_df = filtered_df[filtered_df["strategy_label"] == selected_strategy]
    if selected_attack_type != "All":
        filtered_df = filtered_df[filtered_df["attack_type"] == selected_attack_type]
    if selected_fraction != "All":
        filtered_df = filtered_df[filtered_df["attack_fraction"] == float(selected_fraction)]
    if selected_seed != "All":
        filtered_df = filtered_df[filtered_df["seed"] == int(selected_seed)]

    # Title Banner
    st.title("Trust-Aware Federated RL Research Dashboard")
    st.markdown(
        "Secure and efficient task offloading in Vehicular Ad-hoc Networks (VANETs) under dynamic edge "
        "and adversarial conditions. Evaluates **Behavioural Trust**, **Model-Update Trust**, and "
        "**Robust Federated Aggregation**."
    )

    # Top KPI Metrics Header
    if not filtered_df.empty:
        render_kpi_header(filtered_df, dataset_info["rounds"])

    st.markdown("---")

    # Main Tabs
    tab_overview, tab_scenarios, tab_fl, tab_trust, tab_simulator, tab_repro = st.tabs([
        "📊 Benchmark Overview",
        "🎯 Scenarios & Attacks",
        "🔄 Federated Convergence",
        "🛡️ Trust & Anomaly Detection",
        "🚗 Vehicular Simulation",
        "📦 Reproducibility & Export"
    ])

    # TAB 1: OVERVIEW
    with tab_overview:
        st.subheader("Executive Research Summary")
        st.markdown(
            """
            This research evaluates a **multi-dimensional trust architecture** to protect Federated Reinforcement Learning (FRL)
            against Byzantine poisoning, free-riding, and unreliable nodes in vehicular edge networks:
            - **Behavioural Trust ($T_{\\text{beh}}$):** Evaluates runtime task execution timeliness, completion, and communication reliability.
            - **Model-Update Trust ($T_{\\text{mod}}$):** Evaluates directional alignment ($m_2$), staleness ($m_3$), originality ($m_4$), and learning progress.
            - **Trust-Aware Aggregation:** Dynamically down-weights low-trust updates and excludes anomalous updates below the configured exclusion threshold (v4 core $\\tau_{\\text{exc}} = 0.35$; see dataset manifest for exact values).
            """
        )

        render_scenario_overview_table(filtered_df)

        st.markdown("---")
        render_performance_comparison_chart(filtered_df, rounds=dataset_info["rounds"])

    # TAB 2: SCENARIOS & ATTACKS
    with tab_scenarios:
        st.subheader("Scenario Matrix & Attack Vulnerability Breakdown")
        n_groups = int(df["scenario_group"].nunique()) if (not df.empty and "scenario_group" in df.columns) else 0
        st.markdown(
            f"Empirical results across {n_groups} scenario groups spanning clean control, model poisoning, "
            "free-riding variants, unreliable execution, and honest non-IID heterogeneity."
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Scenario Breakdown Table")
            if not filtered_df.empty:
                scen_agg = filtered_df.groupby("scenario_label")[
                    ["completion_rate", "mean_latency_s", "f1", "precision", "recall", "fpr"]
                ].mean().reset_index()
                st.dataframe(scen_agg.style.format({
                    "completion_rate": "{:.1%}",
                    "mean_latency_s": "{:.2f} s",
                    "f1": "{:.3f}",
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "fpr": "{:.3f}"
                }), use_container_width=True)

        with col_right:
            st.markdown("#### Exclusion Rates by Strategy")
            if not filtered_df.empty:
                excl_agg = filtered_df.groupby(["strategy_label", "scenario_label"])["exclusion_rate"].mean().reset_index()
                excl_chart = alt.Chart(excl_agg).mark_bar().encode(
                    x=alt.X("exclusion_rate:Q", title="Mean Exclusion Rate", axis=alt.Axis(format=".0%")),
                    y=alt.Y("scenario_label:N", title="Scenario"),
                    color=alt.Color("strategy_label:N", title="Strategy"),
                    tooltip=["scenario_label", "strategy_label", alt.Tooltip("exclusion_rate:Q", format=".1%")]
                ).properties(height=340)
                st.altair_chart(excl_chart, use_container_width=True)

        st.markdown("---")
        render_detection_metrics_chart(filtered_df)

        # Unreliable Execution Fault Accounting Subsection
        unrel_df = pd.DataFrame()
        if not filtered_df.empty and "scenario_group" in filtered_df.columns:
            unrel_mask = filtered_df["scenario_group"].astype(str).str.startswith("unreliable")
            unrel_df = filtered_df[unrel_mask]

        if not unrel_df.empty:
            st.markdown("---")
            st.markdown("#### Unreliable Execution Fault Accounting Breakdown")
            st.caption(
                "Aggregated injected fault counts across all unreliable wrappers (drop > fail > delay precedence). "
                "Unreliable nodes are tracked with `is_malicious = False`."
            )
            fault_cols = ["unreliable_severity", "strategy_label", "tasks_exposed", "tasks_with_any_fault", "injected_drops", "injected_fails", "injected_delays"]
            avail_fault_cols = [c for c in fault_cols if c in unrel_df.columns]
            if len(avail_fault_cols) > 2:
                unrel_agg = unrel_df.groupby(["unreliable_severity", "strategy_label"])[avail_fault_cols[2:]].mean().reset_index()
                st.dataframe(unrel_agg.style.format({
                    "tasks_exposed": "{:.1f}",
                    "tasks_with_any_fault": "{:.1f}",
                    "injected_drops": "{:.1f}",
                    "injected_fails": "{:.1f}",
                    "injected_delays": "{:.1f}"
                }), use_container_width=True)

    # TAB 3: FEDERATED CONVERGENCE
    with tab_fl:
        st.subheader("Federated Learning Convergence & Reward Evolution")
        st.markdown(
            "Evaluation of global DQN convergence across communication rounds under clean and adversarial conditions."
        )

        # Statistical Hypothesis Testing Findings
        if summary_metrics and "primary_hypotheses" in summary_metrics:
            st.markdown("#### Confirmatory Statistical Testing Findings")
            h1 = summary_metrics["primary_hypotheses"].get("H1_zero_update_f40", {})
            if h1:
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.markdown("**Primary Hypothesis H1 (Zero-Update Free-Riding $f=0.40$)**")
                    diff_m = h1.get("mean_paired_difference", 0.0)
                    ci_l = h1.get("ci95_difference_low", 0.0)
                    ci_h = h1.get("ci95_difference_high", 0.0)
                    t_val = h1.get("t_statistic")
                    p_val = h1.get("p_value_twotailed")
                    t_str = f"{t_val:.3f}" if t_val is not None else "N/A"
                    p_str = f"{p_val:.3f}" if p_val is not None else "N/A"
                    st.info(
                        f"- **Paired Diff (Trust - FedAvg):** {diff_m:+.4f} (95% CI: [{ci_l:+.4f}, {ci_h:+.4f}])\n"
                        f"- **Paired Student's t-test:** t(9) = {t_str}, p = {p_str}\n"
                        f"- **Statistical Significance:** {'Yes (p < 0.05)' if h1.get('statistically_significant_alpha_005') else 'No (Fail to reject null)'}"
                    )
                with col_h2:
                    st.markdown("**Secondary Hypotheses (Holm-Bonferroni Corrected)**")
                    sec = summary_metrics.get("secondary_hypotheses", {})
                    sec_rows = []
                    for k, v in sec.items():
                        diff = v.get("mean_difference", 0.0)
                        pval = v.get("p_value_raw")
                        holm_a = v.get("holm_adjusted_alpha")
                        af_val = v.get("attack_fraction")
                        af_str = f"{af_val:.1f}" if af_val is not None else "N/A"
                        sec_rows.append({
                            "Test": k,
                            "Scenario": v.get("scenario", "N/A"),
                            "Fraction": af_str,
                            "Paired Diff": f"{diff:+.4f}",
                            "Raw p": f"{pval:.3f}" if pval is not None else "N/A",
                            "Holm Alpha": f"{holm_a:.3f}" if holm_a is not None else "N/A",
                            "Significant": v.get("holm_significant", False)
                        })
                    if sec_rows:
                        st.dataframe(pd.DataFrame(sec_rows), use_container_width=True)

        figs = get_available_figures(dataset_info["figures_dir"])
        if figs:
            st.markdown("#### Publication Experiment Figures")
            # Render in a clean 2-column grid
            for row_idx in range(0, len(figs), 2):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(str(figs[row_idx]), caption=figs[row_idx].name, use_container_width=True)
                if row_idx + 1 < len(figs):
                    with col_b:
                        st.image(str(figs[row_idx + 1]), caption=figs[row_idx + 1].name, use_container_width=True)
        else:
            st.info("No pre-rendered figures found for this dataset.")

        st.markdown("#### Ablation Results (per-configuration, per-scenario means)")
        if "ablation_csv_path" in dataset_info and Path(dataset_info["ablation_csv_path"]).exists():
            abl_view = load_experiment_data(Path(dataset_info["ablation_csv_path"]))
            if not abl_view.empty:
                abl_agg = abl_view.groupby(["ablation_name", "scenario_group"])[
                    ["completion_rate", "mean_reward", "f1", "fpr", "exclusion_rate"]
                ].mean().reset_index()
                st.dataframe(abl_agg.style.format({
                    "completion_rate": "{:.1%}",
                    "mean_reward": "{:.3f}",
                    "f1": "{:.3f}",
                    "fpr": "{:.3f}",
                    "exclusion_rate": "{:.1%}"
                }), use_container_width=True)
            else:
                st.info("Ablation CSV present but empty.")
        else:
            st.info("No ablation dataset registered for this selection.")

        st.markdown("#### Sensitivity Results (per-parameter, per-value, per-scenario means)")
        if "sensitivity_csv_path" in dataset_info and Path(dataset_info["sensitivity_csv_path"]).exists():
            sens_view = load_experiment_data(Path(dataset_info["sensitivity_csv_path"]))
            if not sens_view.empty:
                sens_agg = sens_view.groupby(["sweep_param", "param_value", "scenario_group"])[
                    ["completion_rate", "mean_reward", "f1", "fpr", "exclusion_rate"]
                ].mean().reset_index()
                st.dataframe(sens_agg.style.format({
                    "param_value": "{:.2f}",
                    "completion_rate": "{:.1%}",
                    "mean_reward": "{:.3f}",
                    "f1": "{:.3f}",
                    "fpr": "{:.3f}",
                    "exclusion_rate": "{:.1%}"
                }), use_container_width=True)
            else:
                st.info("Sensitivity CSV present but empty.")
        else:
            st.info("No sensitivity dataset registered for this selection.")

        st.markdown("#### Strategy Comparison on Degradation")
        if not filtered_df.empty and "degradation_completion" in filtered_df.columns:
            deg_df = filtered_df.groupby(["strategy_label", "scenario_label"])[["degradation_completion", "degradation_reward"]].mean().reset_index()
            st.dataframe(deg_df.style.format({
                "degradation_completion": "{:.2%}",
                "degradation_reward": "{:.3f}"
            }), use_container_width=True)

    # TAB 4: TRUST & ANOMALY DETECTION
    with tab_trust:
        st.subheader("Trust Dynamics & Anomaly Detection Analysis")
        st.markdown(
            "Inspect anomaly detection diagnostics, false positive rate on honest non-IID clients, "
            "and detection recall across attack vectors."
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("#### Honest Non-IID False Positive Rate (FPR)")
            honest_runs = df[df["scenario_group"] == "honest_non_iid"]
            if not honest_runs.empty:
                trust_honest = honest_runs[honest_runs["strategy"] != "fedavg"]
                tot_fp = trust_honest["fp"].sum()
                tot_eval = (trust_honest["fp"] + trust_honest["tn"]).sum()
                micro_fpr = (tot_fp / tot_eval) if tot_eval > 0 else 0.0
                macro_fpr = honest_runs["fpr"].mean()

                st.info(
                    f"**Micro-Aggregated FPR (Trust Strategies):** **{micro_fpr:.2%}** ({int(tot_fp)} FP / {int(tot_eval)} update decisions across {len(trust_honest)} runs)\n\n"
                    f"**Macro Mean Per-Run FPR (All Strategies):** **{macro_fpr:.2%}**"
                )
                st.markdown(
                    "Observed anomaly false positive rate for honest non-IID clients across tested conditions, "
                    "measuring whether heterogeneous local updates avoid being misclassified as adversarial. "
                    "Rates are descriptive for the tested seeds and sample sizes; no bounded-FPR guarantee is claimed."
                )
                
                # Show seed by seed honest runs
                st.dataframe(honest_runs[["seed", "strategy", "fp", "tn", "fpr", "precision", "recall"]], use_container_width=True)

        with col_t2:
            st.markdown("#### Detection Recall Breakdown by Attack Family")
            attack_df = filtered_df[~filtered_df["scenario_group"].isin(["clean_control", "honest_non_iid"])] if not filtered_df.empty else pd.DataFrame()
            if not attack_df.empty:
                recall_by_attack = attack_df.groupby("scenario_label")["recall"].mean().reset_index()
                st.dataframe(recall_by_attack.style.format({"recall": "{:.1%}"}), use_container_width=True)
            else:
                st.info("No adversarial scenario data in current filter.")

    # TAB 5: INTERACTIVE SIMULATOR
    with tab_simulator:
        render_interactive_simulator()

    # TAB 6: REPRODUCIBILITY & EXPORT
    with tab_repro:
        st.subheader("Research Reproducibility, Manifest & Data Export")
        st.markdown(
            "Review verified experiment manifests, configuration checksums, model checkpoints, "
            "and download full tabular data."
        )

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown("#### Run Manifest & Configuration")
            if manifest:
                st.json(manifest)
            else:
                st.info("Manifest file not found.")

        with col_exp2:
            st.markdown("#### Model Checkpoints (Fingerprints)")
            ckpts = get_available_checkpoints()
            if ckpts:
                st.dataframe(pd.DataFrame(ckpts), use_container_width=True)
            else:
                st.info("No checkpoint records found.")

            st.markdown("#### Export Tabular Experiment Data")
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered CSV Data",
                data=csv_data,
                file_name=f"vanet_frl_{dataset_info['key']}_filtered.csv",
                mime="text/csv",
                type="primary"
            )

            if summary_metrics:
                json_data = json.dumps(summary_metrics, indent=2).encode("utf-8")
                st.download_button(
                    label="📥 Download Summary JSON",
                    data=json_data,
                    file_name=f"vanet_frl_{dataset_info['key']}_summary.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
