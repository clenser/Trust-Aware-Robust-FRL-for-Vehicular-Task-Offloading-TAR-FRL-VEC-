"""Phase 3 tests: baseline behaviour, mask compliance, metrics, saved outputs.

No learned agents, federation, trust, or attacks (later phases).
"""

import csv
import json

import pytest

from baselines import (
    aggregate,
    candidate_score,
    ensure_valid,
    episode_metrics,
    local_only,
    make_policy,
    make_random_valid,
    nearest_edge,
    resource_aware,
)
from environment import EnvConfig, VehicularEdgeEnv
from scripts.run_baselines import main as run_baselines_main


def make_cfg(**over) -> EnvConfig:
    params = dict(seed=0, num_vehicles=4, num_edge_nodes=2, num_tasks=20, link_reliability=1.0)
    params.update(over)
    cfg = EnvConfig(**params)
    cfg.validate()
    return cfg


def controlled_env() -> VehicularEdgeEnv:
    """Reset env then park nodes at known positions (requester is vehicle 0)."""
    env = VehicularEdgeEnv(make_cfg())
    env.reset(0)
    env.vehicles[0].x_m = 0.0
    env._cand_cache = None
    return env


def test_local_only_always_zero():
    assert local_only([0.1] * 22, [True] + [False] * 5, {}) == 0
    assert local_only([0.1] * 22, [True] * 6, {}) == 0


def test_random_valid_mask_compliance_and_reproducibility():
    env = VehicularEdgeEnv(make_cfg())
    env.reset(3)
    p1, p2 = make_random_valid(11), make_random_valid(11)
    seq1, seq2 = [], []
    for _ in range(20):
        obs, mask = env.current_observation()
        a1, a2 = p1(obs, mask, {}), p2(obs, mask, {})
        assert mask[a1] and mask[a2]
        seq1.append(a1)
        seq2.append(a2)
        env.step(a1)
    assert seq1 == seq2  # same seed -> identical choices


def test_nearest_edge_picks_nearest_edge_not_vehicle():
    env = controlled_env()
    for v in env.vehicles[1:]:
        v.x_m = 10.0  # vehicles nearer than any edge, but must be ignored
    env.edges[0].x_m = 100.0
    env.edges[1].x_m = 50.0
    env._cand_cache = None
    obs, mask = env.current_observation()
    cands = env.candidates()
    expected = next(i for i, c in enumerate(cands) if c.kind == "edge" and c.node_id == 1) + 1
    assert nearest_edge(obs, mask, {"candidates": cands, "env": env}) == expected


def test_nearest_edge_tie_break_lower_id():
    env = controlled_env()
    for v in env.vehicles[1:]:
        v.available = False
    env.edges[0].x_m = 100.0
    env.edges[1].x_m = 1900.0  # wrapped distance also 100 on a 2000 m road
    env._cand_cache = None
    obs, mask = env.current_observation()
    cands = env.candidates()
    assert len([c for c in cands if c.kind == "edge"]) == 2
    expected = next(i for i, c in enumerate(cands) if c.kind == "edge" and c.node_id == 0) + 1
    assert nearest_edge(obs, mask, {"candidates": cands, "env": env}) == expected


def test_resource_aware_avoids_congested_node():
    env = controlled_env()
    for v in env.vehicles[1:]:
        v.available = False
    env.edges[0].x_m = 60.0
    env.edges[1].x_m = 60.0
    env.edges[0].queue_cycles = env.cfg.queue_cycles_max  # fully congested
    env.edges[1].queue_cycles = 0.0
    env._cand_cache = None
    obs, mask = env.current_observation()
    cands = env.candidates()
    expected = next(i for i, c in enumerate(cands) if c.kind == "edge" and c.node_id == 1) + 1
    got = resource_aware(obs, mask, {"candidates": cands, "env": env})
    assert got == expected
    # scoring rule itself: free+fast+near beats congested
    assert candidate_score(0.0, 50e9, 8e9, 10.0, 150.0) > candidate_score(50e9, 8e9, 8e9, 10.0, 150.0)


def test_all_policies_fallback_to_local_when_nothing_eligible():
    env = VehicularEdgeEnv(make_cfg())
    env.reset(5)
    for node in env.vehicles[1:] + env.edges:
        node.available = False
    env._cand_cache = None
    obs, mask = env.current_observation()
    assert mask == [True] + [False] * env.cfg.top_k
    ctx = {"candidates": [], "env": env}
    assert local_only(obs, mask, ctx) == 0
    assert make_random_valid(0)(obs, mask, ctx) == 0
    assert nearest_edge(obs, mask, ctx) == 0
    assert resource_aware(obs, mask, ctx) == 0


def test_ensure_valid_falls_back():
    assert ensure_valid(3, [True, False, False, False]) == 0
    assert ensure_valid(1, [True, True, False, False]) == 1
    assert ensure_valid(99, [True]) == 0


def test_make_policy_unknown_raises():
    with pytest.raises(ValueError):
        make_policy("dqn", 0)


def test_episode_metrics_counts_and_rates():
    m = episode_metrics(10, 6, 3, 1, [1.0] * 10, [2.0] * 10, 4, [0.5] * 10)
    assert m["completion_rate"] == pytest.approx(0.6)
    assert m["miss_rate"] == pytest.approx(0.3)
    assert m["failure_rate"] == pytest.approx(0.1)
    assert m["mean_latency_s"] == pytest.approx(1.0)
    assert m["offload_fraction"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        episode_metrics(10, 6, 3, 2, [1.0] * 10, [2.0] * 10, 4, [0.5] * 10)


def test_aggregate_reports_variability():
    rows = [{"completion_rate": 0.5}, {"completion_rate": 0.7}]
    agg = aggregate(rows, ["completion_rate"])
    assert agg["completion_rate"]["mean"] == pytest.approx(0.6)
    assert agg["completion_rate"]["stdev"] > 0
    assert agg["completion_rate"]["n"] == 2


def test_run_baselines_saves_reproducible_outputs(tmp_path):
    out = str(tmp_path / "bl")
    rc = run_baselines_main(["--config", "configs/smoke.yaml", "--seeds", "0,1",
                             "--policies", "local_only,nearest_edge", "--tasks", "10", "--out", out])
    assert rc == 0
    with open(out + ".csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4  # 2 policies x 2 seeds
    assert all(set(("config", "policy", "seed", "completion_rate")) <= set(r) for r in rows)
    with open(out + ".json", encoding="utf-8") as fh:
        data = json.load(fh)
    assert set(data["aggregate"]) == {"local_only", "nearest_edge"}
    assert data["seeds"] == [0, 1]
    # reproducibility: rerun -> identical CSV bytes
    out2 = str(tmp_path / "bl2")
    run_baselines_main(["--config", "configs/smoke.yaml", "--seeds", "0,1",
                        "--policies", "local_only,nearest_edge", "--tasks", "10", "--out", out2])
    assert (tmp_path / "bl.csv").read_bytes() == (tmp_path / "bl2.csv").read_bytes()
