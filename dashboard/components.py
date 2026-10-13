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
            delta="Observed in tested seeds",
            delta_color="off"
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


try:
    from dashboard.simulator_engine import (
        run_traced_episode,
        generate_road_scene_svg,
        get_available_policy_names,
        get_policy_label,
        StepSnapshot,
    )
    from dashboard.three_scene import build_3d_simulation_html
except ModuleNotFoundError:
    from simulator_engine import (
        run_traced_episode,
        generate_road_scene_svg,
        get_available_policy_names,
        get_policy_label,
        StepSnapshot,
    )
    from three_scene import build_3d_simulation_html


def render_interactive_simulator() -> None:
    """Render interactive 2D multi-lane visual road simulation, step scrubber, and telemetry inspector."""
    st.subheader("Interactive 2D Vehicular Environment & Policy Simulation")
    st.markdown(
        "Execute a live simulation on the verified `VehicularEdgeEnv` to inspect 2D road kinematics, "
        "RSU coverage zones ($R_{\\text{comm}} = 300\\text{m}$), dynamic offloading vectors, and trust evolution in real time."
    )

    with st.expander("Simulation Parameters & Configuration", expanded=True):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        available_policies = get_available_policy_names()
        with sim_col1:
            policy_name = st.selectbox(
                "Offloading Policy",
                options=available_policies,
                format_func=get_policy_label,
                index=0,
                help="Select between baseline heuristics or trained DQN / FedDQN checkpoint models."
            )
            sim_seed = st.number_input("Random PRNG Seed", min_value=0, max_value=99999, value=42, step=1)
        with sim_col2:
            num_v = st.slider("Number of Vehicles", min_value=2, max_value=12, value=6, step=1)
            num_e = st.slider("Number of Roadside Edge Units (RSUs)", min_value=1, max_value=4, value=2, step=1)
        with sim_col3:
            mobility = st.selectbox("Mobility Dynamics", options=["low", "medium", "high"], index=0)
            task_load = st.selectbox("Task Arrival Load", options=["low", "medium", "high"], index=0)
            num_tasks = st.slider("Tasks in Episode", min_value=10, max_value=50, value=30, step=5)

    # Simulation execution button or state check
    sim_key = f"sim_cache_{policy_name}_{sim_seed}_{num_v}_{num_e}_{mobility}_{task_load}_{num_tasks}"
    re_run = st.button("▶️ Run Simulation Episode", type="primary")

    if re_run or "sim_snapshots" not in st.session_state or st.session_state.get("sim_key") != sim_key:
        with st.spinner("Executing simulation dynamics and recording step traces..."):
            cfg = EnvConfig(
                seed=int(sim_seed),
                num_vehicles=int(num_v),
                num_edge_nodes=int(num_e),
                mobility=mobility,
                task_load=task_load,
                num_tasks=int(num_tasks)
            )
            snapshots, summary = run_traced_episode(cfg, policy_name=policy_name, max_tasks=int(num_tasks))
            st.session_state["sim_snapshots"] = snapshots
            st.session_state["sim_summary"] = summary
            st.session_state["sim_key"] = sim_key
            st.session_state["sim_cfg"] = cfg
            st.session_state["sim_step_idx"] = 0

    snapshots: list[StepSnapshot] = st.session_state.get("sim_snapshots", [])
    summary: dict = st.session_state.get("sim_summary", {})
    cfg: EnvConfig = st.session_state.get("sim_cfg", None)

    if not snapshots:
        st.warning("No simulation snapshots available. Click 'Run Simulation Episode' to execute.")
        return

    # Top KPI Metrics row for the episode
    st.markdown("#### Episode Performance Summary")
    ep_col1, ep_col2, ep_col3, ep_col4, ep_col5 = st.columns(5)
    with ep_col1:
        st.metric(
            "Tasks Evaluated",
            f"{summary.get('total_tasks', 0)}",
            delta=summary.get('policy_name', '')
        )
    with ep_col2:
        succ_rate = summary.get('success_rate', 0.0) * 100
        st.metric(
            "Success Rate",
            f"{succ_rate:.1f}%",
            delta=f"{summary.get('successes', 0)} completed"
        )
    with ep_col3:
        st.metric(
            "Deadline Misses",
            f"{summary.get('deadline_misses', 0)}",
            delta="Timeout violations",
            delta_color="inverse"
        )
    with ep_col4:
        st.metric(
            "Mean Latency",
            f"{summary.get('mean_latency_s', 0.0):.2f} s",
            delta=f"Energy: {summary.get('mean_energy_j', 0.0):.1f} J"
        )
    with ep_col5:
        st.metric(
            "Offload Fraction",
            f"{summary.get('offload_fraction', 0.0) * 100:.1f}%",
            delta="V2I / V2V ratio"
        )

    st.markdown("---")

    # Interactive 3D WebGL Vehicular Simulation
    st.markdown("#### Realistic Daylight 3D Vehicular Highway Simulation")
    st.caption(
        "Real-time Three.js WebGL 3D simulation featuring realistic daylight lighting, PBR materials, multi-lane highway "
        "(3 lanes at 3.6m width, steel guardrails, roadside vegetation), detailed vehicle models (Sedans, SUVs, Delivery Vans), "
        "and authentic roadside communication units (RSUs). "
        "**Camera Controls:** Left-click drag to rotate/orbit, scroll wheel to zoom, right-click drag to pan. "
        "Use camera view buttons (**Highway 3/4**, **Chase Cam**, **Roadside**, **Top-Down**) and toggles for coverage and network overlay. "
        "Click any 3D vehicle or RSU to inspect live telemetry."
    )

    # Render standalone 3D WebGL scene
    html_3d = build_3d_simulation_html(snapshots, summary, cfg)
    st.components.v1.html(html_3d, height=680)

    # Step Selector for Granular Telemetry Inspector
    st.markdown("---")
    st.markdown("#### Granular Step Telemetry & Discovered Candidate Inspector")
    col_sel, col_info = st.columns([2, 4])
    with col_sel:
        active_step = st.selectbox(
            "Inspect Step Details",
            options=list(range(len(snapshots))),
            format_func=lambda i: f"Step {i+1}: Task #{snapshots[i].task_id} (V{snapshots[i].requester_vid} ➔ {snapshots[i].dest_name}) [{'SUCCESS' if snapshots[i].success else 'DEADLINE MISS'}]",
            index=min(st.session_state.get("sim_step_idx", 0), len(snapshots) - 1)
        )
        st.session_state["sim_step_idx"] = active_step
    with col_info:
        snap_cur = snapshots[active_step]
        outcome_label = "Completed within deadline" if snap_cur.success else ("Deadline exceeded" if snap_cur.deadline_missed else "Execution failed")
        st.caption(
            f"Displaying verified kinematics, channel rates, and edge computing metrics for **Task #{snap_cur.task_id}** "
            f"requested by **Vehicle {snap_cur.requester_vid}** (Status: *{outcome_label}*)."
        )

    snapshot = snapshots[active_step]
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    with t_col1:
        st.markdown(
            f"""
            <div class="telemetry-card">
              <div class="telemetry-title">Task Specifications</div>
              <div class="telemetry-row"><span>Task ID:</span><span class="telemetry-val">#{snapshot.task_id}</span></div>
              <div class="telemetry-row"><span>Requester:</span><span class="telemetry-val">Vehicle {snapshot.requester_vid}</span></div>
              <div class="telemetry-row"><span>Data Size:</span><span class="telemetry-val">{snapshot.data_bits / 8e3:.1f} KB</span></div>
              <div class="telemetry-row"><span>Compute Cycles:</span><span class="telemetry-val">{snapshot.cycles / 1e6:.1f} Mcyc</span></div>
              <div class="telemetry-row"><span>Deadline:</span><span class="telemetry-val">{snapshot.deadline_s:.2f} s</span></div>
              <div class="telemetry-row"><span>Priority:</span><span class="telemetry-val">{snapshot.priority:.0f}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with t_col2:
        st.markdown(
            f"""
            <div class="telemetry-card">
              <div class="telemetry-title">Decision & Target Node</div>
              <div class="telemetry-row"><span>Action:</span><span class="telemetry-val">#{snapshot.action}</span></div>
              <div class="telemetry-row"><span>Destination:</span><span class="telemetry-val">{snapshot.dest_name}</span></div>
              <div class="telemetry-row"><span>Target Kind:</span><span class="telemetry-val">{snapshot.dest_kind.upper()}</span></div>
              <div class="telemetry-row"><span>Dest Trust:</span><span class="telemetry-val">{snapshot.dest_trust:.3f}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with t_col3:
        st.markdown(
            f"""
            <div class="telemetry-card">
              <div class="telemetry-title">Latency & Energy</div>
              <div class="telemetry-row"><span>Transmission:</span><span class="telemetry-val">{snapshot.tx_latency_s:.3f} s</span></div>
              <div class="telemetry-row"><span>Execution:</span><span class="telemetry-val">{snapshot.exe_latency_s:.3f} s</span></div>
              <div class="telemetry-row"><span>Total Latency:</span><span class="telemetry-val">{snapshot.total_latency_s:.3f} s</span></div>
              <div class="telemetry-row"><span>Energy Cost:</span><span class="telemetry-val">{snapshot.energy_j:.2f} J</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with t_col4:
        outcome_badge = (
            '<span class="badge-success">SUCCESS</span>' if snapshot.success
            else ('<span class="badge-danger">DEADLINE MISS</span>' if snapshot.deadline_missed
            else '<span class="badge-warning">FAILURE</span>')
        )
        trust_delta = snapshot.updated_trust - snapshot.dest_trust
        st.markdown(
            f"""
            <div class="telemetry-card">
              <div class="telemetry-title">Outcome & Trust Update</div>
              <div class="telemetry-row"><span>Execution:</span>{outcome_badge}</div>
              <div class="telemetry-row"><span>Pre-Trust:</span><span class="telemetry-val">{snapshot.dest_trust:.3f}</span></div>
              <div class="telemetry-row"><span>Post-Trust:</span><span class="telemetry-val">{snapshot.updated_trust:.3f}</span></div>
              <div class="telemetry-row"><span>Trust Delta:</span><span class="telemetry-val">{trust_delta:+.3f}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Step Candidates Table
    if snapshot.candidates:
        st.markdown("##### Candidate Nodes in Radio Range ($R_{\\text{comm}} = 300\\text{m}$)")
        cand_rows = []
        for c in snapshot.candidates:
            is_chosen = (snapshot.dest_kind == c["kind"] and snapshot.dest_id == c["node_id"])
            cand_rows.append({
                "Selected": "⭐ SELECTED" if is_chosen else "Eligible",
                "Node": c["name"],
                "Distance (m)": f"{c['dist_m']:.1f}",
                "Link Rate (Mbps)": f"{c['link_rate_mbps']:.2f}",
                "CPU (GHz)": f"{c['freq_ghz']:.2f}",
                "Queue (Mcyc)": f"{c['queue_mcyc']:.1f}",
                "Trust Score": f"{c['trust']:.3f}",
            })
        st.dataframe(pd.DataFrame(cand_rows), use_container_width=True)
    else:
        st.info("No candidates within communication range for this task. Local execution was mandatory.")

    # Dynamic Evolution Charts Across Episode
    st.markdown("---")
    st.markdown("#### Episode Trajectory Analytics")
    c_tab1, c_tab2, c_tab3 = st.columns(3)

    # 1. Latency vs Deadline Chart
    with c_tab1:
        st.markdown("**Task Latency vs. Deadline (s)**")
        lat_records = []
        for i, s in enumerate(snapshots):
            lat_records.append({
                "Task": i + 1,
                "Latency": s.total_latency_s,
                "Deadline": s.deadline_s,
                "Outcome": "Success" if s.success else ("Deadline Miss" if s.deadline_missed else "Failed")
            })
        lat_df = pd.DataFrame(lat_records)
        base_line = alt.Chart(lat_df).mark_line(color="#94a3b8", strokeDash=[4, 4]).encode(
            x=alt.X("Task:Q", title="Task Index"),
            y=alt.Y("Deadline:Q", title="Seconds"),
            tooltip=["Task", "Deadline"]
        )
        pts = alt.Chart(lat_df).mark_circle(size=70).encode(
            x="Task:Q",
            y="Latency:Q",
            color=alt.Color("Outcome:N", scale=alt.Scale(
                domain=["Success", "Deadline Miss", "Failed"],
                range=["#10b981", "#ef4444", "#f59e0b"]
            )),
            tooltip=["Task", "Latency", "Deadline", "Outcome"]
        )
        st.altair_chart((base_line + pts).properties(height=240), use_container_width=True)

    # 2. Cumulative Success Rate
    with c_tab2:
        st.markdown("**Cumulative Success Rate (%)**")
        cum_records = []
        n_succ = 0
        for i, s in enumerate(snapshots):
            if s.success:
                n_succ += 1
            cum_records.append({
                "Task": i + 1,
                "Success Rate": (n_succ / (i + 1)) * 100
            })
        cum_df = pd.DataFrame(cum_records)
        cum_chart = alt.Chart(cum_df).mark_line(color="#2563eb", strokeWidth=2).encode(
            x=alt.X("Task:Q", title="Task Index"),
            y=alt.Y("Success Rate:Q", title="Success Rate (%)", scale=alt.Scale(domain=[0, 105])),
            tooltip=["Task", alt.Tooltip("Success Rate:Q", format=".1f")]
        ).properties(height=240)
        st.altair_chart(cum_chart, use_container_width=True)

    # 3. Queue Dynamics
    with c_tab3:
        st.markdown("**Node Queue Backlog (Mcycles)**")
        q_records = []
        for i, s in enumerate(snapshots):
            for e in s.edges:
                q_records.append({
                    "Task": i + 1,
                    "Node": f"RSU {e['eid']}",
                    "Queue Mcyc": e["queue_mcyc"]
                })
        if q_records:
            q_df = pd.DataFrame(q_records)
            q_chart = alt.Chart(q_df).mark_line().encode(
                x=alt.X("Task:Q", title="Task Index"),
                y=alt.Y("Queue Mcyc:Q", title="Queue Backlog (Mcyc)"),
                color=alt.Color("Node:N", title="Edge Node"),
                tooltip=["Task", "Node", "Queue Mcyc"]
            ).properties(height=240)
            st.altair_chart(q_chart, use_container_width=True)

    # Secondary Chart Row: Destination Breakdown & Trust Trajectories
    c2_tab1, c2_tab2 = st.columns(2)
    with c2_tab1:
        st.markdown("**Offloading Destination Distribution**")
        dest_counts = pd.Series([s.dest_kind.upper() for s in snapshots]).value_counts().reset_index()
        dest_counts.columns = ["Destination", "Count"]
        pie = alt.Chart(dest_counts).mark_bar().encode(
            x=alt.X("Count:Q", title="Number of Tasks"),
            y=alt.Y("Destination:N", title="Execution Location", sort="-x"),
            color=alt.Color("Destination:N", scale=alt.Scale(
                domain=["LOCAL", "EDGE", "VEHICLE"],
                range=["#8b5cf6", "#0284c7", "#10b981"]
            )),
            tooltip=["Destination", "Count"]
        ).properties(height=200)
        st.altair_chart(pie, use_container_width=True)

    with c2_tab2:
        st.markdown("**Destination Trust Score Trajectory**")
        trust_records = []
        for i, s in enumerate(snapshots):
            trust_records.append({
                "Task": i + 1,
                "Node": s.dest_name,
                "Trust": s.updated_trust
            })
        trust_df = pd.DataFrame(trust_records)
        trust_chart = alt.Chart(trust_df).mark_line(point=True).encode(
            x=alt.X("Task:Q", title="Task Index"),
            y=alt.Y("Trust:Q", title="Trust Score", scale=alt.Scale(domain=[0.0, 1.05])),
            color=alt.Color("Node:N", title="Observed Node"),
            tooltip=["Task", "Node", alt.Tooltip("Trust:Q", format=".3f")]
        ).properties(height=200)
        st.altair_chart(trust_chart, use_container_width=True)
