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


def test_strict_policy_rejects_invalid_action(monkeypatch):
    """Verify that strict_policy=True raises ValueError on out-of-bounds or masked action."""
    cfg = EnvConfig(seed=42, num_tasks=5)
    
    # Faulty policy returning invalid action 99
    monkeypatch.setattr(
        "dashboard.simulator_engine.load_simulation_policy",
        lambda *args, **kwargs: (lambda obs, mask, ctx=None: 99, "Faulty Policy")
    )
    
    with pytest.raises(ValueError) as excinfo:
        run_traced_episode(cfg, policy_name="faulty", max_tasks=5, strict_policy=True)
    assert "returned invalid action 99" in str(excinfo.value)


def test_strict_policy_rejects_unexpected_exception(monkeypatch):
    """Verify that strict_policy=True raises RuntimeError on policy crash."""
    cfg = EnvConfig(seed=42, num_tasks=5)

    def crashing_policy(obs, mask, ctx=None):
        raise ZeroDivisionError("Simulated neural crash")

    monkeypatch.setattr(
        "dashboard.simulator_engine.load_simulation_policy",
        lambda *args, **kwargs: (crashing_policy, "Crashing Policy")
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_traced_episode(cfg, policy_name="crashing", max_tasks=5, strict_policy=True)
    assert "Simulated neural crash" in str(excinfo.value)


def test_policy_context_and_fallback_diagnostics():
    """Verify that graceful dashboard mode logs policy errors and invalid action telemetry."""
    cfg = EnvConfig(seed=42, num_tasks=5)

    # 1. Normal run should have 0 exceptions and 0 invalid actions
    _, summary = run_traced_episode(cfg, policy_name="resource_aware", max_tasks=5, strict_policy=False)
    assert summary["policy_exceptions"] == 0
    assert summary["invalid_actions"] == 0

    # 2. Verify local_only strictly has 0 offload fraction
    _, sum_loc = run_traced_episode(cfg, policy_name="local_only", max_tasks=5, strict_policy=True)
    assert sum_loc["offload_fraction"] == 0.0
    assert sum_loc["invalid_actions"] == 0


def test_step_snapshot_handovers_and_timing_serialization():
    """Verify that StepSnapshot correctly captures handovers and serializes into HTML."""
    from dashboard.three_scene import build_3d_simulation_html
    import json

    cfg = EnvConfig(seed=42, num_tasks=5, num_vehicles=4, num_edge_nodes=2)
    snapshots, summary = run_traced_episode(cfg, policy_name="resource_aware", max_tasks=5)

    s0 = snapshots[0]
    assert hasattr(s0, "handovers")
    assert isinstance(s0.handovers, list)
    assert s0.tx_latency_s >= 0.0
    assert s0.exe_latency_s >= 0.0

    html = build_3d_simulation_html(snapshots, summary, cfg)
    assert "const simTrace = " in html
    # Extract simTrace JSON
    prefix = "const simTrace = "
    start_pos = html.find(prefix) + len(prefix)
    end_pos = html.find(";\n    const totalSteps", start_pos)
    trace_json_str = html[start_pos:end_pos]
    data = json.loads(trace_json_str)

    assert data["road_length_m"] == cfg.road_length_m
    assert data["comm_range_m"] == cfg.comm_range_m
    assert len(data["steps"]) == 5
    first_step = data["steps"][0]
    assert "handovers" in first_step
    assert "tx_latency_s" in first_step
    assert "exe_latency_s" in first_step
    assert "total_latency_s" in first_step


def test_3d_scene_controls_and_camera_presets():
    """Verify that all camera presets, controls, HUD badges, and offline assets are present."""
    from dashboard.three_scene import build_3d_simulation_html

    cfg = EnvConfig(seed=101, num_tasks=3)
    snapshots, summary = run_traced_episode(cfg, policy_name="nearest_edge", max_tasks=3)
    html = build_3d_simulation_html(snapshots, summary, cfg)

    # Camera presets
    assert "btn-cam-iso" in html
    assert "btn-cam-chase" in html
    assert "btn-cam-side" in html
    assert "btn-cam-top" in html
    assert "resetCamera()" in html

    # Toggles
    assert "btn-cov" in html
    assert "btn-net" in html
    assert "btn-follow" in html
    assert "toggleCoverage()" in html
    assert "toggleNetworkOverlay()" in html
    assert "toggleCameraFollow()" in html

    # Timeline scrubber
    assert "timeline-slider" in html
    assert "speed-select" in html
    assert "smoothTweenTo" in html

    # Staged HUD
    assert "hud-stage" in html
    assert "WIRELESS TX" in html
    assert "NODE EXECUTION" in html
    assert "LOCAL CPU COMPUTE" in html


def test_deterministic_wheel_rotation_and_vehicle_dynamics():
    """Verify that physical wheel rotation formula and vehicle dynamics are present in JS."""
    from dashboard.three_scene import build_3d_simulation_html

    cfg = EnvConfig(seed=2024, num_tasks=3)
    snapshots, summary = run_traced_episode(cfg, policy_name="resource_aware", max_tasks=3)
    html = build_3d_simulation_html(snapshots, summary, cfg)

    # Deterministic wheel rotation
    assert "curX / wheelRad" in html
    assert "rotation.z = -wheelAngle" in html

    # Acceleration pitch
    assert "v0.accel_mps2" in html
    assert "rotation.z = -pitch" in html

    # Hermite S-curve lane changing
    assert "laneT = alpha * alpha * (3.0 - 2.0 * alpha)" in html
    assert "steerYaw = laneDelta * Math.sin(Math.PI * alpha)" in html
    assert "roll = laneDelta * Math.sin(Math.PI * alpha)" in html



