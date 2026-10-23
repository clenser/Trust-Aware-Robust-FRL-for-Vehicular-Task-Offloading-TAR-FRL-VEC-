"""
Free-Riding Attack Implementations for Federated RL.
Simulates clients that wish to benefit from the global model without contributing:
1. Zero-Update Free Rider: Submits Delta theta = 0 (exact global weights).
2. Stale-Weights Free Rider: Submits weights from old iterations without training.
3. Random-Noise Free Rider: Submits tiny random updates to bypass naive zero checks.
"""

import copy
import torch

class FreeRider:
    @staticmethod
    def zero_update_attack(global_weights):
        """Returns exact global weights (zero local update)."""
        return copy.deepcopy(global_weights)

    @staticmethod
    def stale_update_attack(historical_weights):
        """Returns older round weights without local gradient steps."""
        if historical_weights is not None:
            return copy.deepcopy(historical_weights)
        return None

    @staticmethod
    def dummy_noise_attack(global_weights, epsilon=1e-5):
        """Returns global weights plus tiny undetectable jitter to fake training."""
        fake_weights = copy.deepcopy(global_weights)
        for k in fake_weights.keys():
            fake_weights[k] = (fake_weights[k].float() + torch.randn_like(fake_weights[k].float()) * epsilon).to(fake_weights[k].dtype)
        return fake_weights
