"""Phase 5 tests: behavioural trust dynamics, disconnect handling, integration.

No federation, model-update trust, or attacks (later phases).
"""

from pathlib import Path

import pytest

from baselines import make_policy, trust_aware
from environment import EnvConfig, VehicularEdgeEnv
from trust import BehaviouralTrustTable, Evidence


def make_table() -> BehaviouralTrustTable:
    return BehaviouralTrustTable(2, 2)


def good_ev(kind="edge", nid=0) -> Evidence:
    return Evidence(kind, nid, True, True, 1.0, 5.0, 0.8, 0.8)


def test_init_and_reset():
    t = make_table()
    assert all(v == 0.5 for v in t.snapshot().values())
    t.observe(0, good_ev())
    t.reset()
    assert all(v == 0.5 for v in t.snapshot().values()) and t.history == []


def test_bounds_under_extremes():
    t = make_table()
    for i in range(50):
        t.observe(i, good_ev())
        assert 0.0 <= t.get("edge", 0) <= 1.0
    for i in range(50, 100):
        t.observe(i, Evidence("edge", 0, False, True, 99.0, 1.0, 0.1, 50.0))
        assert 0.0 <= t.get("edge", 0) <= 1.0


def test_positive_raises_repeated_failures_lower_and_recovery():
    t = make_table()
    t.observe(0, good_ev())
    assert t.get("edge", 0) > 0.5
    lo = t.get("edge", 0)
    for i in range(1, 6):
        t.observe(i, Evidence("edge", 0, False, True, 9.0, 5.0, 0.5, 4.0))
    assert t.get("edge", 0) < lo
    for i in range(6, 16):
        t.observe(i, good_ev())
    assert t.get("edge", 0) > 0.5  # recovery, no permanent malice label


def test_disconnect_is_not_malice():
    t = make_table()
    t.observe(0, good_ev())
    high = t.get("edge", 0)
    for i in range(1, 11):  # node unseen: only neutral ageing, no failure records
        t.age_all(i, except_key=("edge", 1))
    v = t.get("edge", 0)
    assert 0.5 < v < high  # drifted toward prior, never punished toward 0
    assert all(r.event == "idle_age" or (r.kind, r.node_id) != ("edge", 0) or r.task_id == 0
               for r in t.history if (r.kind, r.node_id) == ("edge", 0))


def test_path_failure_vs_deadline_miss_attribution():
    t = make_table()
    r_path = t.observe(0, Evidence("edge", 0, False, False, 2.0, 5.0, 0.5, 0.5))
    assert r_path.event == "path_failure" and r_path.s4 == 0.0
    assert r_path.s2 == 0.5  # missing timeliness carried as prior, not zero
    r_miss = t.observe(1, Evidence("edge", 1, False, True, 9.0, 5.0, 0.5, 4.0))
    assert r_miss.event == "deadline_miss" and r_miss.s4 == 1.0
    assert r_miss.s2 == pytest.approx(max(0.0, 1.0 - 9.0 / 5.0))


def test_nodes_keep_separate_states():
    t = make_table()
    for i in range(5):
        t.observe(i, good_ev("edge", 0))
        t.observe(i, Evidence("edge", 1, False, True, 9.0, 5.0, 0.5, 4.0))
    assert t.get("edge", 0) > 0.6 > t.get("edge", 1)


def make_env(**over) -> VehicularEdgeEnv:
    params = dict(seed=0, num_vehicles=4, num_edge_nodes=2, num_tasks=30, link_reliability=1.0)
    params.update(over)
    cfg = EnvConfig(**params)
    cfg.validate()
    return VehicularEdgeEnv(cfg)


def test_env_obs_dim_order_mask_and_slot_identity():
    env = make_env()
    obs, mask = env.reset(3)
    assert len(obs) == env.cfg.obs_dim == 7 + 3 * env.cfg.top_k
    assert len(mask) == env.cfg.action_dim
    cands = env.candidates()
    for i, cand in enumerate(cands):
        assert obs[6 + 3 * i + 1] == pytest.approx(env.trust.get(cand.kind, cand.node_id))


def test_env_trust_evolves_and_resets():
    env = make_env()
    env.reset(4)
    for _ in range(20):
        obs, mask = env.current_observation()
        cands = env.candidates()
        a = 1 if len(cands) else 0
        env.step(a)
    vals = list(env.trust.snapshot().values())
    assert all(0.0 <= v <= 1.0 for v in vals) and len(set(round(v, 6) for v in vals)) > 1
    assert any(r.event in ("success", "deadline_miss", "path_failure") for r in env.trust.history)
    env.reset(4)
    assert all(v == 0.5 for v in env.trust.snapshot().values()) and env.trust.history == []


def test_env_trust_deterministic():
    def run():
        env = make_env()
        env.reset(9)
        pol = make_policy("resource_aware", 9)
        for _ in range(15):
            obs, mask = env.current_observation()
            env.step(int(pol(obs, mask, {"candidates": env.candidates(), "env": env})))
        return env.trust.snapshot()
    assert run() == run()


def test_trust_aware_policy_and_low_trust_fallback():
    env = make_env()
    env.reset(6)
    obs, mask = env.current_observation()
    ctx = {"candidates": env.candidates(), "env": env}
    a = trust_aware(obs, mask, ctx)
    assert mask[a]  # valid (possibly local fallback at prior 0.5 >= 0.3)
    # all-zero trust forces local fallback
    for key in env.trust.snapshot():
        env.trust._t[key] = 0.0
    obs2, mask2 = env.current_observation()
    assert trust_aware(obs2, mask2, {"candidates": env.candidates(), "env": env}) == 0
    assert trust_aware(obs2, mask2, {"candidates": [], "env": env}) == 0


def test_trust_logging_and_plot(tmp_path):
    from scripts.log_trust import main as log_main
    out = str(tmp_path / "trustdemo")
    rc = log_main(["--config", "configs/smoke.yaml", "--seeds", "0,1",
                   "--policy", "resource_aware", "--tasks", "10", "--out", out])
    assert rc == 0
    assert (tmp_path / "trustdemo_snapshots.csv").is_file()
    assert (tmp_path / "trustdemo_events.csv").is_file()
    assert (tmp_path / "trustdemo.json").is_file()
    import glob
    fig = glob.glob("reports/figures/trust_trustdemo.png")
    assert fig
    for f in fig:  # keep the working tree clean: figure was only a generation check
        Path(f).unlink()


def test_phase3_and_dqn_compatible_with_trust_env():
    env = make_env()
    env.reset(7)
    for name in ("local_only", "random_valid", "nearest_edge", "resource_aware", "trust_aware"):
        pol = make_policy(name, 7)
        for _ in range(5):
            obs, mask = env.current_observation()
            a = int(pol(obs, mask, {"candidates": env.candidates(), "env": env}))
            assert mask[a]
            env.step(a)
    ckpt = Path("checkpoints/dqn_default/checkpoint_final.pt")
    if not ckpt.is_file():
        pytest.skip("Phase 4 checkpoint not present")
    from rl import DQNAgent
    agent = DQNAgent.load(ckpt, eps_start=0.0, eps_end=0.0, seed=0)
    agent.epsilon = 0.0
    env.reset(8)
    for _ in range(10):
        obs, mask = env.current_observation()
        a = agent.act(obs, mask)
        assert mask[a]
        env.step(a)
