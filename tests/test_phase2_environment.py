"""Phase 2 tests: environment calculations, validity, reproducibility, edge cases.

No RL / federated / trust / attack logic is tested here (later phases).
"""

import math

import numpy as np
import pytest

from environment import EnvConfig, InvalidActionError, VehicularEdgeEnv, from_yaml
from environment.network import link_rate_hz, snr_linear
from specs import formulas as F


def make_cfg(**over) -> EnvConfig:
    params = dict(seed=0, num_vehicles=5, num_edge_nodes=2, num_tasks=20)
    params.update(over)
    cfg = EnvConfig(**params)
    cfg.validate()
    return cfg


def test_same_seed_reproducible():
    a = VehicularEdgeEnv(make_cfg()); a.reset(7)
    b = VehicularEdgeEnv(make_cfg()); b.reset(7)
    seq_a = [(a.step(0).latency_s, a.step(0).energy_j) for _ in range(10)]
    b.reset(7)
    seq_b = [(b.step(0).latency_s, b.step(0).energy_j) for _ in range(10)]
    assert seq_a == seq_b


def test_different_seeds_differ():
    a = VehicularEdgeEnv(make_cfg()); a.reset(1)
    b = VehicularEdgeEnv(make_cfg()); b.reset(2)
    lat_a = [a.step(0).latency_s for _ in range(10)]
    lat_b = [b.step(0).latency_s for _ in range(10)]
    assert lat_a != lat_b


def test_obs_shape_bounds_and_mask():
    env = VehicularEdgeEnv(make_cfg())
    obs, mask = env.reset(3)
    assert len(obs) == env.cfg.obs_dim == 7 + 3 * env.cfg.top_k
    assert len(mask) == env.cfg.action_dim == 1 + env.cfg.top_k
    assert all(0.0 <= x <= 1.0 for x in obs)
    assert all(math.isfinite(x) for x in obs)
    assert mask[0] is True  # local always available
    cands = env.candidates()
    assert mask[1:] == [i < len(cands) for i in range(env.cfg.top_k)]


@pytest.mark.parametrize("top_k", [1, 2, 5, 8])
def test_obs_dim_multiple_k_values(top_k):
    """Declared metadata and actual vector agree for every supported K."""
    env = VehicularEdgeEnv(make_cfg(top_k=top_k))
    assert env.cfg.obs_dim == 7 + 3 * top_k
    assert env.cfg.action_dim == 1 + top_k
    for seed in (0, 1):
        obs, mask = env.reset(seed)
        assert len(obs) == 7 + 3 * top_k
        assert len(mask) == 1 + top_k
        assert all(0.0 <= x <= 1.0 and math.isfinite(x) for x in obs)
        # stepping keeps working with the wider/narrower action space
        valid = [a for a, ok in enumerate(mask) if ok]
        env.step(valid[0])


def test_obs_component_order():
    """Each slot equals the documented normalised quantity, in SPEC §4 order."""
    env = VehicularEdgeEnv(make_cfg(link_reliability=1.0))
    env.reset(21)
    t = env.pending
    req = env.vehicles[t.requester_id]
    cands = env.candidates()
    obs, _ = env.current_observation()
    c = env.cfg
    assert obs[0] == pytest.approx(F.minmax_norm(t.data_bits, 0.0, c.data_bits_max))
    assert obs[1] == pytest.approx(F.minmax_norm(t.cycles, 0.0, c.cycles_max))
    assert obs[2] == pytest.approx(F.minmax_norm(t.deadline_s, 0.0, c.deadline_max_s))
    assert obs[3] == pytest.approx(F.minmax_norm(t.priority, 1.0, 3.0))
    assert obs[4] == pytest.approx(F.minmax_norm(req.utilisation, 0.0, 1.0))
    assert obs[5] == pytest.approx(F.minmax_norm(req.queue_cycles, 0.0, c.queue_cycles_max))
    for i, cand in enumerate(cands):
        node = env.vehicles[cand.node_id] if cand.kind == "vehicle" else env.edges[cand.node_id]
        base = 6 + 3 * i
        assert obs[base] == pytest.approx(F.minmax_norm(cand.dist_m, 0.0, c.comm_range_m))
        assert obs[base + 1] == pytest.approx(c.neutral_trust)
        assert obs[base + 2] == pytest.approx(F.minmax_norm(node.queue_cycles, 0.0, c.queue_cycles_max))
    for i in range(len(cands), c.top_k):
        assert obs[6 + 3 * i: 9 + 3 * i] == [0.0, 0.0, 0.0]  # zero padding
    assert obs[-1] == pytest.approx(len(cands) / c.top_k)  # eligible count last


def test_invalid_actions_raise():
    env = VehicularEdgeEnv(make_cfg())
    env.reset(4)
    n = len(env.candidates())
    with pytest.raises(InvalidActionError):
        env.step(env.cfg.top_k + 1)  # out of range
    with pytest.raises(InvalidActionError):
        env.step(n + 1)  # points at padded/ineligible slot (or out of range when n == top_k)


def test_env_matches_spec_formulas():
    env = VehicularEdgeEnv(make_cfg(link_reliability=1.0))
    env.reset(5)
    req = env.vehicles[env.pending.requester_id]
    cyc = env.pending.cycles
    res = env.step(0)
    # Post-step identity: exec = c/f + q_drained/f = q_post/f_post (drift applied pre-execution).
    assert res.dest_kind == "local" and res.latency_s == pytest.approx(
        req.queue_cycles / req.effective_freq_hz
    )
    assert res.energy_j == pytest.approx(F.task_energy(0.0, 0.0, env.cfg.kappa, cyc, req.cpu_freq_hz))
    assert isinstance(res.success, bool)


def test_outcome_flags_consistent():
    env = VehicularEdgeEnv(make_cfg())
    env.reset(6)
    for _ in range(15):
        obs, mask = env.current_observation()
        res = env.step(0)
        assert res.success == (not res.failed and not res.deadline_missed)
        assert not (res.failed and res.deadline_missed)


def test_impossible_deadline_misses():
    env = VehicularEdgeEnv(make_cfg())
    env.reset(8)
    env.pending.deadline_s = 1e-9  # shorter than any execution
    res = env.step(0)
    assert res.deadline_missed and not res.success


def test_queue_congestion_increases_latency():
    env = VehicularEdgeEnv(make_cfg(link_reliability=1.0))
    env.reset(9)
    req = env.vehicles[env.pending.requester_id]
    req.queue_cycles = 0.0
    fresh_lat = env.step(0).latency_s
    env.reset(9)
    req2 = env.vehicles[env.pending.requester_id]
    req2.queue_cycles = 20e9  # heavy backlog on the same first task stream
    cong_lat = env.step(0).latency_s
    assert cong_lat > fresh_lat


def test_out_of_range_nodes_ineligible():
    env = VehicularEdgeEnv(make_cfg(comm_range_m=1.0))
    env.reset(10)
    # With 1 m range, candidates (if any) must all be within range.
    for c in env.candidates():
        assert c.dist_m <= 1.0
    # Tiny range over 2000 m road usually leaves no candidates at all.
    obs, mask = env.current_observation()
    assert mask[0] is True and sum(mask) >= 1


def test_link_rate_decreases_with_distance():
    cfg = make_cfg()
    near = link_rate_hz(10.0, cfg.bandwidth_hz, cfg.tx_power_w, cfg.pl_k0, cfg.pl_gamma, cfg.noise_w)
    far = link_rate_hz(140.0, cfg.bandwidth_hz, cfg.tx_power_w, cfg.pl_k0, cfg.pl_gamma, cfg.noise_w)
    assert near > far > 0
    assert snr_linear(0.0, cfg.tx_power_w, cfg.pl_k0, cfg.pl_gamma, cfg.noise_w) > 0


def test_mobility_moves_vehicles_in_range():
    env = VehicularEdgeEnv(make_cfg())
    env.reset(11)
    x0 = [v.x_m for v in env.vehicles]
    for _ in range(10):
        env.step(0)
    x1 = [v.x_m for v in env.vehicles]
    assert x0 != x1
    assert all(0.0 <= x < env.cfg.road_length_m for x in x1)


def test_yaml_config_loads():
    cfg = from_yaml("configs/smoke.yaml")
    assert cfg.num_vehicles == 5 and cfg.num_edge_nodes == 2 and cfg.comm_range_m == 150.0
    env = VehicularEdgeEnv(cfg)
    obs, mask = env.reset()
    assert len(obs) == cfg.obs_dim == 7 + 3 * cfg.top_k


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        EnvConfig(num_vehicles=0).validate()
    with pytest.raises(ValueError):
        EnvConfig(mobility="warp").validate()
    with pytest.raises(ValueError):
        EnvConfig(link_reliability=0.0).validate()


def test_episode_summary_counts():
    env = VehicularEdgeEnv(make_cfg())
    env.reset(12)
    _, summary = env.run_episode(lambda o, m: 0, max_tasks=10)
    assert summary.num_tasks == 10
    assert summary.successes + summary.deadline_misses + summary.failures == 10
    assert 0.0 <= summary.offload_fraction <= 1.0
