"""
Experience Replay Buffer for Q-learning.
Stores transitions: (state, action, reward, next_state, done).

Thread-safe: the web console appends transitions from the simulation task
while a federated training worker samples batches concurrently.
"""

import random
import threading
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self._lock = threading.RLock()

    def push(self, state, action, reward, next_state, done):
        with self._lock:
            self.buffer.append((
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done)
            ))

    def sample(self, batch_size):
        with self._lock:
            if len(self.buffer) == 0:
                raise ValueError("Cannot sample from an empty replay buffer")
            effective = min(batch_size, len(self.buffer))
            batch = random.sample(list(self.buffer), effective)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        with self._lock:
            return len(self.buffer)
