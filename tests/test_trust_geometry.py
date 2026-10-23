"""
The reachable exclusion region of the joint trust rule.

T_joint = alpha * T_beh + (1 - alpha) * T_model, excluded when T_joint < tau,
plus an override in ``TrustEngine.compute_joint_trust``: a model trust <= 0.20
caps T_joint at 0.30 (so it is always excluded at tau = 0.40) for 0 < alpha < 1.

These tests document - they do not endorse - what that combination implies, so
that nobody claims more for the "joint" score than the arithmetic supports:

* a client whose behavioural trust is at its neutral prior (0.70, which is what
  every vehicle holds until it is used as a V2V helper) is excluded only
  through the model-trust override or when the model trust is below 0.10;
* behavioural trust alone can exclude a client only when alpha is large
  enough that alpha * T_beh_min + (1 - alpha) * T_model_max < tau.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trust.trust_engine import TrustEngine


def excluded(alpha, t_beh, t_model, tau=0.40):
    eng = TrustEngine(alpha=alpha, exclusion_threshold=tau)
    eng.behavior_tracker.get_record("V1")["trust_score"] = t_beh
    return eng.compute_joint_trust("V1", t_model) < tau


class TestJointTrustGeometry(unittest.TestCase):
    def test_neutral_behaviour_excludes_only_hostile_model_updates(self):
        # alpha = 0.5, T_beh = 0.70 (the prior): the formula alone needs T_model < 0.10;
        # the override extends it to T_model <= 0.20.
        self.assertTrue(excluded(0.5, 0.70, 0.05))
        self.assertTrue(excluded(0.5, 0.70, 0.20))
        self.assertFalse(excluded(0.5, 0.70, 0.25))
        self.assertFalse(excluded(0.5, 0.70, 0.30))   # the 'partial' tier is kept

    def test_behaviour_alone_cannot_exclude_a_model_honest_client_at_default_alpha(self):
        # An unreliable helper (T_beh at its floor 0.05) with an honest update (T_model = 1.0):
        # 0.5 * 0.05 + 0.5 * 1.0 = 0.525 >= tau, so it stays in the aggregation.
        self.assertFalse(excluded(0.5, 0.05, 1.0))

    def test_large_alpha_lets_behaviour_exclude(self):
        self.assertTrue(excluded(0.9, 0.05, 1.0))      # 0.045 + 0.1 = 0.145 < 0.40

    def test_alpha_one_ignores_the_model_update(self):
        self.assertFalse(excluded(1.0, 0.70, 0.05))

    def test_alpha_zero_uses_model_trust_directly(self):
        self.assertTrue(excluded(0.0, 0.70, 0.35))
        self.assertFalse(excluded(0.0, 0.70, 0.45))

    def test_threshold_is_inert_between_the_override_and_the_formula(self):
        # Moving tau between 0.35 and 0.45 changes nothing for a neutral-behaviour client
        # with T_model <= 0.20 (override) or >= 0.25 (formula gives >= 0.475).
        for tau in (0.35, 0.40, 0.45):
            self.assertTrue(excluded(0.5, 0.70, 0.10, tau))
            self.assertFalse(excluded(0.5, 0.70, 0.30, tau))


if __name__ == "__main__":
    unittest.main()
