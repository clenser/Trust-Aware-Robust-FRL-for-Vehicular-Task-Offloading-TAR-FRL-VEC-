"""DQN agent: epsilon-greedy masked action selection, TD learning, checkpoints."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dqn import build_q_network, masked_greedy, masked_max_next_q
from rl.replay_buffer import ReplayBuffer


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, PyTorch; single thread for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class DQNAgent:
    """Conventional DQN with target network and masked discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int] | None = None,
                 lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 32,
                 capacity: int = 10000, eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay_episodes: int = 20, max_grad_norm: float = 10.0,
                 seed: int = 0) -> None:
        if not 0.0 <= eps_end <= eps_start <= 1.0:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        seed_all(seed)
        self.seed = seed
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, max(1, eps_decay_episodes)
        self.epsilon = eps_start
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.hidden = list(hidden or [64, 64])
        self.q_net = build_q_network(obs_dim, action_dim, self.hidden)
        self.target_net = build_q_network(obs_dim, action_dim, self.hidden)
        self.sync_target()
        self.optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity)
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0

    def update_epsilon(self, episode: int) -> float:
        """Linear decay over episodes; returns the new epsilon."""
        frac = min(1.0, episode / self.eps_decay)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)
        return self.epsilon

    def act(self, obs: list[float], mask: list[bool]) -> int:
        """Epsilon-greedy over mask-valid actions only (exploration is uniform-valid)."""
        if len(obs) != self.obs_dim or len(mask) != self.action_dim:
            raise ValueError("obs/mask dimension mismatch with agent")
        valid = [a for a, ok in enumerate(mask) if ok]
        if not valid:
            raise ValueError("no valid action available")
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            q = self.q_net(torch.tensor([obs], dtype=torch.float32))
            return int(masked_greedy(q, torch.tensor([mask], dtype=torch.bool)).item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def bellman_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """y = r + gamma * (1 - done) * max_valid Q_target(s', .). No bootstrap on done."""
        next_q = masked_max_next_q(self.target_net, batch["next_obs"], batch["next_mask"])
        not_done = (~batch["done"]).float()
        return batch["reward"] + self.gamma * not_done * next_q

    def train_step(self) -> float:
        """One gradient step on a random minibatch; returns finite loss or raises."""
        batch = self.buffer.sample(self.batch_size, self.rng)
        q_pred = self.q_net(batch["obs"]).gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = self.bellman_target(batch)
        if not torch.isfinite(target).all():
            raise ValueError("non-finite Bellman target")
        loss = nn.functional.mse_loss(q_pred, target)
        if not math.isfinite(loss.item()):
            raise ValueError(f"non-finite loss: {loss.item()}")
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optim.step()
        self.train_steps += 1
        return float(loss.item())

    # ------------------------------------------------------------ checkpoints
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_state": self.q_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden": self.hidden,
            "seed": self.seed,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "DQNAgent":
        """Restore weights/optimizer/epsilon into a fresh agent (same hyperparams unless overridden)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        hidden = overrides.pop("hidden", ckpt.get("hidden", [64, 64]))
        agent = cls(obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"], hidden=hidden, **overrides)
        agent.q_net.load_state_dict(ckpt["q_state"])
        agent.target_net.load_state_dict(ckpt["target_state"])
        agent.optim.load_state_dict(ckpt["optim_state"])
        agent.epsilon = ckpt["epsilon"]
        agent.train_steps = ckpt["train_steps"]
        return agent
