"""
Unit and integration tests for the 2D visual simulation engine.
Tests:
- Policy resolution (heuristics + trained DQN / FedDQN checkpoints)
- Step-by-step trace generation
- Telemetry calculations (latency decomposition, candidate tracking)
- SVG markup generation with RSUs, coverage zones, vehicles, and offload vectors
"""

import pytest
from environment.vanet_env import EnvConfig
from dashboard.simulator_engine import (
    get_available_policy_names,
    get_policy_label,
    load_simulation_policy,
    run_traced_episode,
    generate_road_scene_svg,
    StepSnapshot,
)


def test_available_policies():
    """Verify that heuristic policies and existing checkpoints are discovered."""
    policies = get_available_policy_names()
    assert "resource_aware" in policies
    assert "nearest_edge" in policies
    assert "random_valid" in policies
    assert "local_only" in policies
    # Check that labels are human-readable
    for pol in policies:
        label = get_policy_label(pol)
        assert len(label) > 0
        assert isinstance(label, str)


def test_load_simulation_policy():
    """Verify policy loader returns callable and descriptive string."""
    for pol_name in ["resource_aware", "local_only", "nearest_edge"]:
        pol, desc = load_simulation_policy(pol_name, seed=42)
        assert callable(pol)
        assert pol_name in desc


def test_run_traced_episode_structure():
    """Verify run_traced_episode produces accurate snapshots and episode summary."""
    cfg = EnvConfig(
        seed=101,
        num_vehicles=5,
        num_edge_nodes=2,
        mobility="low",
        task_load="low",
        num_tasks=10,
    )
    snapshots, summary = run_traced_episode(cfg, policy_name="resource_aware", max_tasks=10)
    
    assert len(snapshots) == 10
    assert summary["total_tasks"] == 10
    assert summary["successes"] + summary["deadline_misses"] + summary["failures"] == 10
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert summary["mean_latency_s"] >= 0.0

    # Inspect first snapshot
    s0 = snapshots[0]
    assert isinstance(s0, StepSnapshot)
    assert s0.step_idx == 0
    assert s0.task_id == 0
    assert len(s0.vehicles) == 5
    assert len(s0.edges) == 2
    assert s0.dest_kind in ("local", "edge", "vehicle")
    assert s0.total_latency_s >= 0.0
    assert s0.total_latency_s == pytest.approx(s0.tx_latency_s + s0.exe_latency_s, rel=1e-5)
    assert s0.energy_j >= 0.0
    assert 0.0 <= s0.updated_trust <= 1.0


def test_generate_road_scene_svg():
    """Verify SVG markup contains all required visual layers and valid XML structure."""
    cfg = EnvConfig(
        seed=42,
        num_vehicles=6,
        num_edge_nodes=2,
        mobility="medium",
        task_load="medium",
        num_tasks=5,
    )
    snapshots, _ = run_traced_episode(cfg, policy_name="resource_aware", max_tasks=5)
    
    for snap in snapshots:
        svg = generate_road_scene_svg(snap, road_length_m=cfg.road_length_m)
        assert svg.strip().startswith("<svg")
        assert svg.strip().endswith("</svg>")
        # Highway lanes & yellow shoulders
        assert "stroke=\"#eab308\"" in svg
        # RSU towers
        assert "RSU 0" in svg
        # Vehicles
        assert "V0" in svg
        assert f"V{snap.requester_vid}" in svg
        # Coverage gradient / discs
        assert "rsu-cov-grad" in svg
        # Offload indicator
        if snap.dest_kind == "edge":
            assert "animateMotion" in svg
        elif snap.dest_kind == "local":
            assert "LOCAL" in svg or "stroke=\"#8b5cf6\"" in svg


def test_build_3d_simulation_html():
    """Verify Three.js 3D simulation HTML generator embeds WebGL scene and trace data."""
    from dashboard.three_scene import build_3d_simulation_html

    cfg = EnvConfig(
        seed=123,
        num_vehicles=5,
        num_edge_nodes=2,
        mobility="low",
        task_load="low",
        num_tasks=8,
    )
    snapshots, summary = run_traced_episode(cfg, policy_name="resource_aware", max_tasks=8)
    html_doc = build_3d_simulation_html(snapshots, summary, cfg)

    assert "<!DOCTYPE html>" in html_doc
    assert "new THREE.Scene()" in html_doc
    assert "THREE.OrbitControls" in html_doc
    assert "sRGBEncoding" in html_doc
    assert "ACESFilmicToneMapping" in html_doc
    assert "buildRealistic3DVehicle" in html_doc
    assert "buildRealistic3DRsu" in html_doc
    assert "timeline-slider" in html_doc
    assert "inspector-panel" in html_doc
    assert "simTrace" in html_doc
    assert len(html_doc) > 50000  # Confirms asset bundling

