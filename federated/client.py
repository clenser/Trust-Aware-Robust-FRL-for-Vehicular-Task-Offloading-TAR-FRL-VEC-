"""Simulated federated client: private env + private replay buffer + local DQN.

The client NEVER exposes raw transitions. Its round update to the server is:
{"client_id", "round_trained", "num_samples", "staleness", "state"} —
parameters plus scalar metadata only (SPEC §8 binding rule).
"""

from __future__ import annotations

import copy

import torch

from baselines import episode_metrics
from environment import EnvConfig, VehicularEdgeEnv
from rl import DQNAgent, Transition, task_reward

ALLOWED_UPDATE_KEYS = {"client_id", "round_trained", "num_samples", "staleness", "state"}


class FedClient:
    """One vehicle-associated learner with strictly local experience."""

    def __init__(self, client_id: int, env_cfg: EnvConfig, agent_kwargs: dict, seed: int) -> None:
        env_cfg.validate()
        self.client_id = client_id
        self.env_cfg = copy.deepcopy(env_cfg)
        self.seed = seed
        self.env = VehicularEdgeEnv(self.env_cfg)
        self.agent = DQNAgent(obs_dim=self.env_cfg.obs_dim, action_dim=self.env_cfg.action_dim,
                              seed=seed, **agent_kwargs)
        self.local_episodes_done = 0
        self.last_mean_reward = float("nan")
        self.last_round_samples = 0

    def install_global(self, state: dict[str, torch.Tensor]) -> None:
        """Load the distributed global model into BOTH online and target nets.

        Design (documented): each round restarts from global consensus; the
        local target-sync schedule then continues within the round. Tested.
        """
        self.agent.q_net.load_state_dict(state)
        self.agent.sync_target()

    def local_train(self, num_episodes: int, base_seed: int, num_tasks: int,
                    warmup_steps: int, sync_every: int) -> dict:
        """Train locally for `num_episodes`; return the server-bound update dict."""
        rewards_all: list[float] = []
        succ = miss = fail = off = 0
        lat: list[float] = []
        ene: list[float] = []
        collected = 0
        for e in range(num_episodes):
            self.env.reset(base_seed + self.local_episodes_done)
            self.agent.update_epsilon(self.local_episodes_done)
            self.local_episodes_done += 1
            for step_i in range(num_tasks):
                obs, mask = self.env.current_observation()
                action = self.agent.act(obs, mask)
                res = self.env.step(action)
                reward = task_reward(res.latency_s, res.energy_j, res.success,
                                     res.failed, res.deadline_missed)
                done = step_i == num_tasks - 1
                self.agent.buffer.push(Transition(obs, mask, action, reward,
                                                  res.obs, res.mask, done))
                collected += 1
                rewards_all.append(reward)
                succ += res.success
                miss += res.deadline_missed
                fail += res.failed
                off += res.action > 0
                lat.append(res.latency_s)
                ene.append(res.energy_j)
                if collected > warmup_steps and len(self.agent.buffer) >= self.agent.batch_size:
                    self.agent.train_step()
                    if self.agent.train_steps % sync_every == 0:
                        self.agent.sync_target()
        self.last_mean_reward = sum(rewards_all) / len(rewards_all)
        self.last_round_samples = collected
        metrics = episode_metrics(collected, succ, miss, fail, lat, ene, off, rewards_all)
        return {"metrics": metrics, "transitions": collected}

    def make_update(self, round_trained: int) -> dict:
        """Package parameters + metadata. Contains no observations, actions,
        rewards, or replay data — enforced by test."""
        return {
            "client_id": self.client_id,
            "round_trained": round_trained,
            # n_i: valid local training transitions collected this round (EQ-21 weight).
            "num_samples": self.last_round_samples,
            "staleness": 0,  # filled by the server relative to the current round
            "state": {k: v.detach().cpu().clone() for k, v in self.agent.q_net.state_dict().items()},
        }

    def trust_summary(self) -> dict:
        """Behavioural trust for analysis only (Phase 6 never weights by it)."""
        snap = self.env.trust.snapshot()
        vals = list(snap.values())
        return {"mean_trust": sum(vals) / len(vals), "min_trust": min(vals),
                "max_trust": max(vals), "nodes": len(vals)}
