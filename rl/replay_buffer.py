"""Experience replay buffer for the Phase 4 DQN agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    obs: list[float]
    mask: list[bool]
    action: int
    reward: float
    next_obs: list[float]
    next_mask: list[bool]
    done: bool


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform random minibatch sampling."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, t: Transition) -> None:
        if t.action < 0:
            raise ValueError("action must be >= 0")
        self._buf.append(t)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        """Return a batch of tensors; raises if undersized."""
        if len(self._buf) < batch_size:
            raise ValueError(f"buffer has {len(self._buf)} < {batch_size} transitions")
        idx = rng.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[i] for i in idx]
        return {
            "obs": torch.tensor([t.obs for t in batch], dtype=torch.float32),
            "mask": torch.tensor([t.mask for t in batch], dtype=torch.bool),
            "action": torch.tensor([t.action for t in batch], dtype=torch.int64),
            "reward": torch.tensor([t.reward for t in batch], dtype=torch.float32),
            "next_obs": torch.tensor([t.next_obs for t in batch], dtype=torch.float32),
            "next_mask": torch.tensor([t.next_mask for t in batch], dtype=torch.bool),
            "done": torch.tensor([t.done for t in batch], dtype=torch.bool),
        }
