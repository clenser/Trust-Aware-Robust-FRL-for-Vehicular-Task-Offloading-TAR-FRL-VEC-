"""
Trust-Aware Multi-Objective Reward Function for Task Offloading.

Formulation:
    R = - alpha_lat * (Delay / Deadline)
        - beta_energy * (Energy / E_norm)
        - gamma_fail * Failure_indicator
        + delta_success * Success_indicator
        + lambda_trust * Target_Node_Trust

Energy normalisation
-------------------
``e_norm`` is the *reference energy* of a representative offloading
interaction, not an arbitrary constant.  With the corrected computation-energy
model (``E = kappa * f^2 * N`` with ``kappa = 1e-30`` J*s^2/cycle, i.e.
1 pJ per cycle at 1 GHz) a 0.22 Gcycle cooperative-perception task executed
on a 4 GHz edge core consumes about 3.5 mJ, and a 0.40 Gcycle
video-analytics task about 6.4 mJ.  ``e_norm = 0.01 J`` (10 mJ) therefore
places a typical offload at a normalized energy of roughly 0.3-0.7, i.e. the
energy term is commensurate with the (bounded) delay term instead of being
numerically irrelevant.

The value is a *modelling normalisation*: it does not assert a physical
budget, and it is documented here so that the reward is interpretable.
"""

import numpy as np

class RewardCalculator:
    def __init__(self, alpha_lat=2.0, beta_energy=1.0, gamma_fail=5.0, delta_success=3.0,
                 lambda_trust=2.5, e_norm=0.01):
        self.alpha_lat = alpha_lat
        self.beta_energy = beta_energy
        self.gamma_fail = gamma_fail
        self.delta_success = delta_success
        self.lambda_trust = lambda_trust
        # Reference energy of a representative offloading interaction (J).
        self.e_norm = float(e_norm)

    def compute_reward(self, success, delay_sec, deadline_sec, energy_j, target_trust_score=0.8):
        """
        Calculates scalar reinforcement learning reward.
        """
        if not success:
            # Harsh penalty for task dropping or deadline violation
            r = -self.gamma_fail - self.alpha_lat * 2.0
            # If target node was untrusted, add slight penalty
            if target_trust_score < 0.4:
                r -= 1.0
            return float(r)

        # Successful completion
        norm_delay = min(delay_sec / max(1e-4, deadline_sec), 2.0)
        # The energy penalty is NOT capped: very different energy outcomes
        # must remain distinguishable to the learner.
        norm_energy = float(energy_j) / self.e_norm

        r = (self.delta_success
             - self.alpha_lat * norm_delay
             - self.beta_energy * norm_energy
             + self.lambda_trust * target_trust_score)

        return float(r)
