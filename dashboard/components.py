"""
Interactive UI visual components for the VANET FRL Research Dashboard.
Provides modular visual renderers:
- Metric KPI cards & summaries
- Scenario comparisons & confidence intervals
- Anomaly detection matrices & trade-off curves
- Network topology and task offloading interactive simulator
- Reproducibility verification and manifest inspect
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import baselines
from environment.vanet_env import VehicularEdgeEnv, EnvConfig


def render_kpi_header(df: pd.DataFrame, rounds: int) -> None:
    """Render top-level executive KPI cards."""
    total_runs = len(df)
    unique_scenarios = df["scenario_group"].nunique() if "scenario_group" in df.columns else 0
    unique_strategies = df["strategy"].nunique() if "strategy" in df.columns else 0
    unique_seeds = df["seed"].nunique() if "seed" in df.columns else 0

    mean_comp = df["completion_rate"].mean() * 100 if "completion_rate" in df.columns else 0.0
    mean_lat = df["mean_latency_s"].mean() if "mean_latency_s" in df.columns else 0.0
    mean_f1 = df["f1"].mean() if "f1" in df.columns else 0.0
    
    # Honest Non-IID FPR
    honest_df = df[df["scenario_group"] == "honest_non_iid"] if "scenario_group" in df.columns else pd.DataFrame()
    honest_fpr = honest_df["fpr"].mean() * 100 if not honest_df.empty else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Total Experiment Runs",
            value=f"{total_runs}",
            delta=f"{unique_scenarios} scenarios × {unique_strategies} strats"
        )
    with col2:
        st.metric(
            label="FL Rounds Horizon",
            value=f"{rounds} Rounds",
            delta=f"{unique_seeds} Random Seeds"
        )
    with col3:
        st.metric(
            label="Mean Task Completion",
            value=f"{mean_comp:.1f}%",
            delta=f"Lat: {mean_lat:.2f}s"
        )
    with col4:
        st.metric(
            label="Overall Anomaly F1",
            value=f"{mean_f1:.3f}",
            delta="Robust Aggregation"
        )
    with col5:
        st.metric(
            label="Honest Non-IID FPR",
            value=f"{honest_fpr:.1f}%",
            delta="Target < 5.0%",
            delta_color="inverse" if honest_fpr > 5.0 else "normal"
        )


def render_scenario_overview_table(df: pd.DataFrame) -> None:
    """Render summary table grouped by scenario and strategy."""
    st.subheader("Aggregated Performance by Scenario and Aggregation Strategy")
    st.markdown(
        "Empirical benchmarks across baseline unweighted `FedAvg`, `Trust-Weighted`, "
        "and `Trust-Threshold Robust` aggregation."
    )

    if df.empty:
        st.warning("No experiment records available.")
        return

    # Pivot or group
    group_cols = ["scenario_label", "strategy_label"]
    metric_cols = ["completion_rate", "mean_latency_s", "f1", "precision", "recall", "fpr", "exclusion_rate"]
    
    available_metrics = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(group_cols)[available_metrics].agg(["mean", "std"]).reset_index()

    # Flatten column names for clean display
    flat_cols = []
    for c in summary.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    summary.columns = flat_cols

    # Formatting
    st.dataframe(
        summary.style.format(
            formatter={
                col: "{:.3f}" for col in summary.select_dtypes(include=[float]).columns
            }
        ),
        use_container_width=True,
        height=380
    )


def render_detection_metrics_chart(df: pd.DataFrame) -> None:
    """Render interactive Altair charts for attack detection metrics."""
    st.subheader("Attack Detection Performance across Fractions ($f \\in [0.0, 0.4]$)")
    st.markdown(
        "Evaluation of model-update trust anomaly detection flags: cosine angle deviation ($m_2$), "
        "temporal staleness ($m_3$), originality flag ($m_4$), and combined anomaly score."
    )

    # Filter out clean and non-iid for attack detection curve
    attack_df = df[~df["scenario_group"].isin(["clean_control", "honest_non_iid"])].copy()
    if attack_df.empty:
        st.info("No adversarial scenario data available.")
        return

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**F1 Score vs Malicious Client Ratio**")
        f1_chart_data = attack_df.groupby(["scenario_label", "attack_fraction"])["f1"].mean().reset_index()
        f1_chart = alt.Chart(f1_chart_data).mark_line(point=True).encode(
            x=alt.X("attack_fraction:Q", title="Attack Fraction (f)", axis=alt.Axis(format=".0%")),
            y=alt.Y("f1:Q", title="Mean Detection F1 Score", scale=alt.Scale(domain=[0.0, 1.05])),
            color=alt.Color("scenario_label:N", title="Attack Scenario"),
            tooltip=["scenario_label", alt.Tooltip("attack_fraction:Q", format=".0%"), alt.Tooltip("f1:Q", format=".3f")]
        ).properties(height=320)
        st.altair_chart(f1_chart, use_container_width=True)

    with col_chart2:
        st.markdown("**Precision vs Recall Trade-off by Attack Type**")
        pr_data = attack_df.groupby(["scenario_label"])[["precision", "recall"]].mean().reset_index()
        pr_chart = alt.Chart(pr_data).mark_circle(size=120).encode(
            x=alt.X("recall:Q", title="Detection Recall", scale=alt.Scale(domain=[0.0, 1.05])),
            y=alt.Y("precision:Q", title="Detection Precision", scale=alt.Scale(domain=[0.0, 1.05])),
            color=alt.Color("scenario_label:N", title="Attack Scenario"),
            tooltip=["scenario_label", alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")]
        ).properties(height=320)
        st.altair_chart(pr_chart, use_container_width=True)


def render_performance_comparison_chart(df: pd.DataFrame, rounds: int | None = None) -> None:
    """Render completion rate and latency degradation comparisons."""
    st.subheader("Held-Out Offloading Policy Performance Under Aggregation")
    horizon_note = f"after {rounds} federated rounds" if rounds else "after the dataset's federated horizon"
    st.caption(
        f"Note: These metrics evaluate the global model produced {horizon_note} on a standardized, clean held-out test environment. "
        "Training-time execution faults (such as Scenario C unreliable execution) alter local client training updates "
        "and behavioural trust, while held-out rollout tests model generalization and robustness."
    )

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**Held-Out Completion Rate vs Attack Fraction by Strategy**")
        comp_data = df.groupby(["strategy_label", "attack_fraction"])["completion_rate"].mean().reset_index()
        comp_chart = alt.Chart(comp_data).mark_line(point=True).encode(
            x=alt.X("attack_fraction:Q", title="Attack Fraction (f)", axis=alt.Axis(format=".0%")),
            y=alt.Y("completion_rate:Q", title="Mean Completion Rate", scale=alt.Scale(domain=[0.5, 1.0])),
            color=alt.Color("strategy_label:N", title="Aggregation Strategy"),
            tooltip=["strategy_label", alt.Tooltip("attack_fraction:Q", format=".0%"), alt.Tooltip("completion_rate:Q", format=".1%")]
        ).properties(height=320)
        st.altair_chart(comp_chart, use_container_width=True)

    with col_chart2:
        st.markdown("**Mean Latency (s) vs Attack Fraction**")
        lat_data = df.groupby(["strategy_label", "attack_fraction"])["mean_latency_s"].mean().reset_index()
        lat_chart = alt.Chart(lat_data).mark_line(point=True).encode(
            x=alt.X("attack_fraction:Q", title="Attack Fraction (f)", axis=alt.Axis(format=".0%")),
            y=alt.Y("mean_latency_s:Q", title="Mean Latency (seconds)"),
            color=alt.Color("strategy_label:N", title="Aggregation Strategy"),
            tooltip=["strategy_label", alt.Tooltip("attack_fraction:Q", format=".0%"), alt.Tooltip("mean_latency_s:Q", format=".2f")]
        ).properties(height=320)
        st.altair_chart(lat_chart, use_container_width=True)


def render_interactive_simulator() -> None:
    """Render interactive single-episode simulation runner for visual exploration."""
    st.subheader("Interactive VANET Environment & Policy Simulation")
    st.markdown(
        "Execute a live single-episode simulation on the verified `VehicularEdgeEnv` "
        "to inspect spatial topology, candidate discovery, and task offloading decisions in real-time."
    )

    with st.expander("Simulation Parameters & Controls", expanded=True):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            policy_name = st.selectbox(
                "Offloading Policy",
                options=list(baselines.POLICIES),
                format_func=lambda x: {
                    "local_only": "Local Only (Vehicle CPU)",
                    "random_valid": "Random Valid Node",
                    "nearest_edge": "Nearest Edge Node (RSU)",
                    "resource_aware": "Resource-Aware Heuristic"
                }.get(x, x),
                index=3
            )
            sim_seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
        with sim_col2:
            num_v = st.slider("Number of Vehicles", min_value=2, max_value=12, value=6, step=1)
            num_e = st.slider("Number of RSU Edge Nodes", min_value=1, max_value=4, value=2, step=1)
        with sim_col3:
            mobility = st.selectbox("Mobility Setting", options=["low", "medium", "high"], index=0)
            task_load = st.selectbox("Task Arrival Load", options=["low", "medium", "high"], index=0)

    if st.button("Run Simulation Episode", type="primary"):
        with st.spinner("Executing simulation episode..."):
            cfg = EnvConfig(
                seed=int(sim_seed),
                num_vehicles=int(num_v),
                num_edge_nodes=int(num_e),
                mobility=mobility,
                task_load=task_load,
                num_tasks=40
            )
            env = VehicularEdgeEnv(cfg)
            env.reset()
            policy = baselines.make_policy(policy_name, seed=int(sim_seed))
            step_results, ep_summary = env.run_episode(policy)

            # Metrics
            st.success(f"Episode completed: {ep_summary.num_tasks} tasks evaluated.")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Success Rate", f"{ep_summary.successes}/{ep_summary.num_tasks} ({ep_summary.successes/ep_summary.num_tasks:.1%})")
            with m_col2:
                st.metric("Deadline Misses", f"{ep_summary.deadline_misses}")
            with m_col3:
                st.metric("Mean Latency", f"{ep_summary.mean_latency_s:.2f} s")
            with m_col4:
                st.metric("Offload Fraction", f"{ep_summary.offload_fraction:.1%}")

            # Spatial topology visualization
            st.markdown("#### Vehicular & Edge Node Spatial Coordinates (1D Highway)")
            veh_data = []
            for v in env.vehicles:
                veh_data.append({
                    "Node": f"Vehicle {v.vid}",
                    "Position (m)": v.x_m,
                    "Speed (m/s)": v.speed_mps,
                    "CPU Freq (GHz)": v.cpu_freq_hz / 1e9,
                    "Type": "Vehicle"
                })
            for e in env.edges:
                veh_data.append({
                    "Node": f"Edge {e.eid}",
                    "Position (m)": e.x_m,
                    "Speed (m/s)": 0.0,
                    "CPU Freq (GHz)": e.cpu_freq_hz / 1e9,
                    "Type": "Edge (RSU)"
                })
            
            top_df = pd.DataFrame(veh_data)
            topo_chart = alt.Chart(top_df).mark_circle(size=160).encode(
                x=alt.X("Position (m):Q", title="Road Position (meters)", scale=alt.Scale(domain=[0, cfg.road_length_m])),
                y=alt.Y("Type:N", title="Node Type"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Vehicle", "Edge (RSU)"], range=["#2563eb", "#dc2626"])),
                tooltip=["Node", "Position (m)", "Speed (m/s)", "CPU Freq (GHz)"]
            ).properties(height=200)
            st.altair_chart(topo_chart, use_container_width=True)

            # Node Table
            st.markdown("#### Node Configuration")
            st.dataframe(top_df, use_container_width=True)
