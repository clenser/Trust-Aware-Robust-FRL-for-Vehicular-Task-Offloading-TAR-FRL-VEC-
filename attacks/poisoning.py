"""
Model Poisoning Attack Implementations for Federated RL.
Supports:
1. Sign-Flipping: Reverses gradient direction to degrade global convergence.
2. Gaussian Noise Attack: Adds high-variance random perturbations.
3. Scaling Attack: Inflates update norm to dominate federated aggregation.
"""

import copy
import torch
import numpy as np

class ModelPoisoner:
    @staticmethod
    def sign_flip_attack(local_weights, global_weights, scale=2.0):
        """
        Delta' = - scale * (theta_local - theta_global)
        theta_poisoned = theta_global + Delta'
        """
        poisoned = copy.deepcopy(local_weights)
        for key in global_weights.keys():
            delta = local_weights[key].float() - global_weights[key].float()
            poisoned[key] = (global_weights[key].float() - scale * delta).to(local_weights[key].dtype)
        return poisoned

    @staticmethod
    def gaussian_noise_attack(local_weights, noise_std=0.5):
        """Adds zero-mean Gaussian perturbation with standard deviation noise_std."""
        poisoned = copy.deepcopy(local_weights)
        for key in poisoned.keys():
            noise = torch.randn_like(poisoned[key].float()) * noise_std
            poisoned[key] = (poisoned[key].float() + noise).to(poisoned[key].dtype)
        return poisoned

    @staticmethod
    def scaling_attack(local_weights, global_weights, scale_factor=5.0):
        """Artificially scales gradient delta by a large factor."""
        poisoned = copy.deepcopy(local_weights)
        for key in global_weights.keys():
            delta = local_weights[key].float() - global_weights[key].float()
            poisoned[key] = (global_weights[key].float() + scale_factor * delta).to(local_weights[key].dtype)
        return poisoned

    @staticmethod
    def partial_sign_flip_attack(local_weights, global_weights, fraction=0.5, scale=1.0, seed=0):
        """Flip the sign of a fixed random ``fraction`` of the update coordinates.

        The remaining coordinates keep their honest value, so the poisoned
        update stays positively correlated with the honest majority.  This is
        the weak attack the detector is documented to miss at small fractions.
        """
        gen = torch.Generator().manual_seed(int(seed))
        poisoned = copy.deepcopy(local_weights)
        for key in global_weights.keys():
            delta = local_weights[key].float() - global_weights[key].float()
            mask = (torch.rand(delta.shape, generator=gen) < fraction).float()
            signed = delta * (1.0 - mask) - scale * delta * mask
            poisoned[key] = (global_weights[key].float() + signed).to(local_weights[key].dtype)
        return poisoned

    @staticmethod
    def apply_attack(local_weights, global_weights, attack_type="sign_flip", strength=1.0):
        if attack_type == "sign_flip":
            return ModelPoisoner.sign_flip_attack(local_weights, global_weights, scale=1.5 * strength)
        elif attack_type == "gaussian_noise":
            return ModelPoisoner.gaussian_noise_attack(local_weights, noise_std=0.4 * strength)
        elif attack_type == "scaling":
            return ModelPoisoner.scaling_attack(local_weights, global_weights, scale_factor=3.0 * strength)
        elif attack_type == "partial_sign_flip":
            # ``strength`` is the flipped fraction of coordinates (0-1).
            return ModelPoisoner.partial_sign_flip_attack(
                local_weights, global_weights, fraction=min(1.0, max(0.0, strength)))
        elif attack_type == "stealth_sign_flip":
            # Norm-matched reversal: -1 x the honest update, so the norm equals
            # the client's own honest update and no norm test can see it.
            return ModelPoisoner.sign_flip_attack(local_weights, global_weights, scale=1.0)
        return local_weights
