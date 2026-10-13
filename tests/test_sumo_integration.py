"""Unit and integration tests for Eclipse SUMO microscopic traffic simulation and TraCI adapter."""

import os
from unittest.mock import patch
import pytest

from environment.config import EnvConfig
from environment.sumo_adapter import (
    SumoScenarioConfig,
    SumoTrafficSimulator,
    SumoVehicleState,
    find_sumo_binary,
    get_sumo_home,
    get_sumo_status,
    is_sumo_available,
    SumoNotInstalledError,
)
from environment.sumo_env import SumoVehicularEdgeEnv, HandoverEvent
from dashboard.simulator_engine import run_traced_episode
import baselines


def test_sumo_detection_and_status():
    """Verify SUMO availability detection returns a complete status dictionary."""
    status = get_sumo_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "sumo_home" in status
    assert "sumo_binary" in status
    assert "traci_installed" in status
    assert "instructions" in status

    # On this verified host, SUMO and TraCI should be detected
    if is_sumo_available():
        assert status["available"] is True
        assert status["sumo_binary"] is not None
        assert status["traci_installed"] is True


def test_graceful_fallback_when_sumo_unavailable():
    """Verify that when SUMO is absent, is_sumo_available returns False and errors are informative."""
    with patch("environment.sumo_adapter.find_sumo_binary", return_value=None):
        assert is_sumo_available() is False
        sim = SumoTrafficSimulator()
        with pytest.raises(SumoNotInstalledError) as exc_info:
            sim.start()
        assert "Cannot start SUMO simulation" in str(exc_info.value)


def test_scenario_config_generation(tmp_path):
    """Verify scenario files (.net.xml, .rou.xml, .sumocfg) are correctly verified and generated."""
    scen_cfg = SumoScenarioConfig(
        scenario_dir=tmp_path,
        road_length_m=1200.0,
        density="medium",
        seed=101,
    )
    sim = SumoTrafficSimulator(scen_cfg)
    cfg_file = sim._ensure_scenario_files()
    assert cfg_file.is_file()
    assert (tmp_path / "highway_medium.rou.xml").is_file()
    assert "highway.net.xml" in cfg_file.read_text(encoding="utf-8")


@pytest.mark.skipif(not is_sumo_available(), reason="Requires local Eclipse SUMO installation")
def test_real_sumo_execution_and_telemetry():
    """Execute real SUMO process via TraCI and verify physical vehicle states."""
    cfg = SumoScenarioConfig(seed=42, density="medium", step_length_s=0.1)
    with SumoTrafficSimulator(cfg) as sim:
        assert sim.is_running is True
        time_elapsed = sim.step(50)  # 5.0 simulation seconds
        assert time_elapsed >= 4.9

        vehs = sim.get_active_vehicles()
        assert len(vehs) > 0, "SUMO highway should have inserted vehicles after 5 seconds"
        for v in vehs:
            assert isinstance(v, SumoVehicleState)
            assert v.x_m >= 0.0
            assert v.lane_idx in (0, 1, 2)
            assert v.speed_mps >= 0.0
            assert v.veh_type in ("passenger_sedan", "passenger_suv", "commercial_van")


@pytest.mark.skipif(not is_sumo_available(), reason="Requires local Eclipse SUMO installation")
def test_sumo_env_lifecycle_and_candidates():
    """Verify SumoVehicularEdgeEnv reset, candidate discovery, and observation vector dimensions."""
    env_cfg = EnvConfig(seed=42, num_vehicles=6, num_edge_nodes=2, road_length_m=1200.0)
    env = SumoVehicularEdgeEnv(env_cfg, use_sumo=True)

    obs, mask = env.reset()
    assert len(obs) == env_cfg.obs_dim
    assert len(mask) == env_cfg.action_dim
    assert mask[0] is True  # Action 0 (local) must always be valid

    # Check candidates structure
    cands = env.candidates()
    assert isinstance(cands, list)
    for c in cands:
        assert c.kind in ("edge", "vehicle")
        assert c.dist_m >= 0.0

    env.close()


@pytest.mark.skipif(not is_sumo_available(), reason="Requires local Eclipse SUMO installation")
def test_rsu_coverage_and_handover_tracking():
    """Verify RSU association and handover events under real microscopic vehicle mobility."""
    env_cfg = EnvConfig(seed=42, num_vehicles=6, num_edge_nodes=2, road_length_m=1200.0)
    env = SumoVehicularEdgeEnv(env_cfg, use_sumo=True)
    obs, mask = env.reset()

    # Step through 15 tasks to allow vehicles to traverse coverage zones
    policy = baselines.make_policy("resource_aware", seed=42)
    for _ in range(15):
        cands = env.candidates()
        action = policy(obs, mask, {"candidates": cands, "env": env})
        res = env.step(action)
        obs, mask = res.obs, res.mask

    # Handovers list should have recorded association/handover events
    assert len(env.handovers) > 0
    for h in env.handovers:
        assert isinstance(h, HandoverEvent)
        assert h.event_type in ("associate", "handover", "disconnect")

    env.close()


@pytest.mark.skipif(not is_sumo_available(), reason="Requires local Eclipse SUMO installation")
def test_run_traced_episode_with_sumo():
    """Verify dashboard engine runs full traced episode with SUMO and produces valid summary metrics."""
    env_cfg = EnvConfig(seed=42, num_vehicles=6, num_edge_nodes=2, num_tasks=20, road_length_m=1200.0)
    snapshots, summary = run_traced_episode(env_cfg, policy_name="resource_aware", max_tasks=10, mobility_mode="sumo")

    assert len(snapshots) == 10
    assert summary["total_tasks"] == 10
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert summary["mobility_engine"] == "SUMO Microscopic Traffic (TraCI)"
    assert "total_handovers" in summary
    assert "total_associations" in summary

    # Verify per-step snapshot has rich SUMO fields
    s0 = snapshots[0]
    assert len(s0.vehicles) > 0
    v0 = s0.vehicles[0]
    assert "sumo_vid" in v0
    assert "lane" in v0
    assert "speed_mps" in v0
    assert "connected_rsu" in v0


@pytest.mark.skipif(not is_sumo_available(), reason="Requires local Eclipse SUMO installation")
def test_policy_differentiation_in_simulator_engine():
    """Verify policies make distinct offloading choices in simulator_engine under SUMO traffic."""
    env_cfg = EnvConfig(seed=42, num_vehicles=6, num_edge_nodes=2, num_tasks=20, road_length_m=1200.0)
    
    # 1. Local only must have offload_fraction == 0.0
    _, summary_local = run_traced_episode(env_cfg, policy_name="local_only", max_tasks=15, mobility_mode="sumo")
    assert summary_local["offload_fraction"] == 0.0

    # 2. Nearest edge should offload when RSUs are reachable
    _, summary_nearest = run_traced_episode(env_cfg, policy_name="nearest_edge", max_tasks=15, mobility_mode="sumo")
    assert summary_nearest["offload_fraction"] > 0.0
    assert summary_nearest["offload_fraction"] != summary_local["offload_fraction"]

