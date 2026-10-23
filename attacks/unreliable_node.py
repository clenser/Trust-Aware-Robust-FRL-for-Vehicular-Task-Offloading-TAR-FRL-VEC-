"""
Unreliable Node Attack & Defective Behavior.
Simulates edge computing nodes or vehicle peers that:
- Drop tasks maliciously
- Fake high availability
- Inject artificial processing latency
"""

import numpy as np

class UnreliableNodeBehavior:
    def __init__(self, drop_probability=0.75, latency_multiplier=4.0):
        self.drop_probability = drop_probability
        self.latency_multiplier = latency_multiplier

    def should_drop(self):
        return np.random.rand() < self.drop_probability

    def corrupt_latency(self, normal_latency):
        return normal_latency * np.random.uniform(2.0, self.latency_multiplier)
