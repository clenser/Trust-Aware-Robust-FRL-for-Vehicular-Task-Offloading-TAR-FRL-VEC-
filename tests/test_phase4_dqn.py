"""Phase 4 tests: network dims, masks, replay, targets, checkpoints, separation.

No federation, trust, or attacks (later phases).
"""

import numpy as np
import pytest
import torch

from environment import EnvConfig, VehicularEdgeEnv
from rl import DQNAgent, ReplayBuffer, Transition, seed_all, task_reward
from rl.dqn import build_q_network, masked_greedy, masked_max_next_q


def tiny_agent(**over) -> DQNAgent:
    params = dict(obs_dim=22, action_dim=6, hidden=[16, 16], batch_size=4, capacity=50)
    params.update(over)
    return DQNAgent(**params)


def test_network_input_output_dims():
    net = build_q_network(22, 6, [32, 32])
    out = net(torch.zeros(5, 22))
    assert out.shape == (5, 6)
    with pytest.raises(ValueError):
        build_q_network(0, 6, [8])


def test_masked_greedy_never_selects_invalid():
    q = torch.tensor([[1.0, 99.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True, True]])
    assert int(masked_greedy(q, mask).item()) == 3
    with pytest.raises(ValueError):
        masked_greedy(q, torch.tensor([[False] * 4]))


def test_epsilon_greedy_behavior():
    agent = tiny_agent(seed=0)
    obs, mask = [0.0] * 22, [True, True, False, False, False, False]
    agent.epsilon = 0.0
    greedy = [agent.act(obs, mask) for _ in range(5)]
    assert all(a in (0, 1) for a in greedy) and len(set(greedy)) == 1
    agent.epsilon = 1.0
    picks = {agent.act(obs, mask) for _ in range(50)}
    assert picks <= {0, 1} and picks == {0, 1}  # explores both valid actions
    with pytest.raises(ValueError):
        tiny_agent().act([0.0] * 5, mask)


def test_replay_insert_sample_shapes_and_overflow():
    buf = ReplayBuffer(10)
    for i in range(15):
        buf.push(Transition([0.0] * 22, [True] * 6, i % 6, 1.0, [0.0] * 22, [True] * 6, False))
    assert len(buf) == 10  # FIFO cap
    batch = buf.sample(4, np.random.default_rng(0))
    assert batch["obs"].shape == (4, 22) and batch["action"].shape == (4,)
    assert batch["mask"].dtype == torch.bool
    with pytest.raises(ValueError):
        buf.sample(11, np.random.default_rng(0))


def test_bellman_target_terminal_vs_nonterminal_and_next_mask():
    agent = tiny_agent(seed=1, gamma=0.9)
    with torch.no_grad():  # force known target outputs: action 2 highest everywhere
        for p in agent.target_net.parameters():
            p.zero_()
        agent.target_net[-1].bias.copy_(torch.tensor([-1.0, 0.0, 5.0, -2.0, -3.0, -4.0]))
    mk = lambda done, valid: {
        "reward": torch.tensor([1.0, 1.0]),
        "done": torch.tensor([done, done]),
        "next_obs": torch.zeros(2, 22),
        "next_mask": torch.tensor([[True] * 6, valid]),
    }
    both = agent.bellman_target(mk(False, [True] * 6))
    assert both[0].item() == pytest.approx(1.0 + 0.9 * 5.0)  # bootstrap full max
    assert both[1].item() == pytest.approx(1.0 + 0.9 * 5.0)
    term = agent.bellman_target(mk(True, [True] * 6))
    assert term.tolist() == pytest.approx([1.0, 1.0])  # terminal: no bootstrap
    masked = agent.bellman_target(mk(False, [True, True, False, False, False, False]))
    assert masked[1].item() == pytest.approx(1.0 + 0.9 * 0.0)  # max over valid only


def test_train_step_finite_loss_and_sync():
    agent = tiny_agent(seed=2)
    for i in range(8):
        agent.buffer.push(Transition([0.01 * i] * 22, [True] * 6, i % 6, float(i),
                                     [0.02 * i] * 22, [True] * 6, False))
    loss = agent.train_step()
    assert np.isfinite(loss) and agent.train_steps == 1
    agent.sync_target()
    for p, q in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(p, q)


def test_checkpoint_save_load_consistency(tmp_path):
    agent = tiny_agent(seed=3)
    for i in range(8):
        agent.buffer.push(Transition([0.01 * i] * 22, [True] * 6, i % 6, float(i),
                                     [0.02 * i] * 22, [True] * 6, False))
    agent.train_step()
    path = agent.save(tmp_path / "ckpt.pt")
    # Loaded without hidden override restores saved [16, 16] automatically
    loaded_auto = DQNAgent.load(path, batch_size=4, capacity=50)
    assert loaded_auto.hidden == [16, 16]
    assert loaded_auto.q_net[0].out_features == 16
    assert loaded_auto.epsilon == agent.epsilon and loaded_auto.train_steps == agent.train_steps
    # Loaded with explicit override works as well
    loaded = DQNAgent.load(path, hidden=[16, 16], batch_size=4, capacity=50)
    assert loaded.epsilon == agent.epsilon and loaded.train_steps == agent.train_steps
    obs = [0.05] * 22
    agent.epsilon = loaded.epsilon = loaded_auto.epsilon = 0.0
    assert loaded.act(obs, [True] * 6) == agent.act(obs, [True] * 6) == loaded_auto.act(obs, [True] * 6)


def test_reproducibility_fixed_seed():
    a1, a2 = tiny_agent(seed=7), tiny_agent(seed=7)
    for p, q in zip(a1.q_net.parameters(), a2.q_net.parameters()):
        assert torch.equal(p, q)
    assert a1.act([0.1] * 22, [True] * 6) == a2.act([0.1] * 22, [True] * 6)


def test_train_eval_separation_and_env_dims(tmp_path):
    cfg = EnvConfig(seed=0, num_vehicles=3, num_edge_nodes=1, num_tasks=6)
    env = VehicularEdgeEnv(cfg)
    agent = DQNAgent(obs_dim=env.cfg.obs_dim, action_dim=env.cfg.action_dim,
                     hidden=[8, 8], batch_size=2, capacity=20, seed=4)
    assert agent.obs_dim == len(env.reset(0)[0])
    env.reset(0)
    for i in range(6):  # tiny training rollout with exploration
        obs, mask = env.current_observation()
        res = env.step(agent.act(obs, mask))
        agent.buffer.push(Transition(obs, mask, res.action,
                                     task_reward(res.latency_s, res.energy_j, res.success,
                                                 res.failed, res.deadline_missed, res.dest_trust),
                                     res.obs, res.mask, i == 5))
    agent.train_step()
    path = agent.save(tmp_path / "c.pt")
    ev = DQNAgent.load(path, hidden=[8, 8], batch_size=2, capacity=20,
                       eps_start=0.0, eps_end=0.0, seed=99)
    ev.epsilon = 0.0
    env.reset(1)  # unseen seed
    for _ in range(6):  # greedy eval never explores and stays valid
        obs, mask = env.current_observation()
        a = ev.act(obs, mask)
        assert mask[a]
        env.step(a)


def test_reward_provisional_and_finite_guards():
    r = task_reward(1.0, 1.0, True, False, False, 0.5)
    assert np.isfinite(r) and r > task_reward(9.0, 9.0, False, False, True, 0.5)
    r_high_trust = task_reward(1.0, 1.0, True, False, False, 0.9)
    r_low_trust = task_reward(1.0, 1.0, True, False, False, 0.1)
    assert r_high_trust > r_low_trust
    with pytest.raises(ValueError):
        task_reward(float("nan"), 1.0, True, False, False)
