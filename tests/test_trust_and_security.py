"""
Unit tests for Behavioral Trust, Model-Update Trust, Non-IID safety, and Adversarial Attack Detection.
"""

import unittest
import numpy as np
from trust.trust_model import BehavioralTrustTracker, _canonical_id
from trust.model_trust import ModelTrustEvaluator
from trust.trust_engine import TrustEngine

class TestTrustAndSecurity(unittest.TestCase):
    def test_canonical_id_normalization(self):
        self.assertEqual(_canonical_id(0), "V0")
        self.assertEqual(_canonical_id("0"), "V0")
        self.assertEqual(_canonical_id("V0"), "V0")
        self.assertEqual(_canonical_id(15), "V15")
        self.assertEqual(_canonical_id("V15"), "V15")
        self.assertEqual(_canonical_id("RSU_1"), "RSU_1")

    def test_behavioral_trust_observable_evidence(self):
        tracker = BehavioralTrustTracker(eta=0.8)
        node = 1
        
        # Reliable node completing tasks on time
        for _ in range(5):
            t = tracker.record_interaction(node_id=node, success=True, delay_sec=0.04, deadline_sec=0.10, channel_sinr=18.0, utilization=0.2)
        self.assertGreaterEqual(t, 0.75, f"Expected high trust for reliable node, got {t}")

        # Unreliable node dropping tasks
        bad_node = 2
        for _ in range(10):
            t_bad = tracker.record_interaction(node_id=bad_node, success=False, delay_sec=0.15, deadline_sec=0.10, channel_sinr=5.0, utilization=0.9)
        self.assertLess(t_bad, 0.40, f"Expected low trust for unreliable node, got {t_bad}")

    def test_model_trust_non_iid_honesty_safety(self):
        """
        Validates that honest clients with natural Non-IID gradient variation (Dirichlet alpha=0.4)
        are NOT falsely penalized or excluded.
        """
        evaluator = ModelTrustEvaluator()
        np.random.seed(42)
        
        # 5 honest clients with natural directional divergence due to Non-IID data
        base_direction = np.random.randn(100)
        base_direction /= np.linalg.norm(base_direction)
        
        client_updates = {}
        for i in range(5):
            noise = np.random.randn(100) * 0.35
            v = base_direction + noise
            client_updates[f"V{i}"] = v
            
        scores = evaluator.evaluate_model_updates(client_updates)
        for cid, score in scores.items():
            self.assertGreaterEqual(score, 0.55, f"Honest Non-IID client {cid} falsely scored too low: {score}")

    def test_model_trust_adversarial_detection(self):
        """
        Validates that:
        1. Sign-flip attacks are detected and scored <= 0.15.
        2. Free-riders (zero update) are detected and scored <= 0.10.
        3. Gradient scaling attacks are penalized.
        """
        evaluator = ModelTrustEvaluator()
        np.random.seed(99)
        
        honest_grad = np.random.randn(100)
        client_updates = {
            "V0": honest_grad + np.random.randn(100) * 0.1,
            "V1": honest_grad + np.random.randn(100) * 0.1,
            "V2": honest_grad + np.random.randn(100) * 0.1,
            "V_signflip": -2.0 * honest_grad,               # Adversarial sign-flip
            "V_freerider": np.zeros(100),                   # Zero update free rider
            "V_scaling": 8.0 * honest_grad                  # Gradient explosion
        }
        
        scores = evaluator.evaluate_model_updates(client_updates)
        self.assertGreaterEqual(scores["V0"], 0.70, f"Honest client V0 should have high score: {scores['V0']}")
        self.assertLessEqual(scores["V_signflip"], 0.15, f"Sign-flip attacker must be penalized: {scores['V_signflip']}")
        self.assertLessEqual(scores["V_freerider"], 0.10, f"Free-rider must be penalized: {scores['V_freerider']}")
        self.assertLessEqual(scores["V_scaling"], 0.20, f"Scaling attack must be penalized: {scores['V_scaling']}")

    def test_trust_engine_filtering_and_metrics(self):
        engine = TrustEngine(alpha=0.5, exclusion_threshold=0.40)
        np.random.seed(42)
        
        honest_grad = np.random.randn(100)
        client_updates = {
            0: honest_grad + np.random.randn(100) * 0.1,
            1: honest_grad + np.random.randn(100) * 0.1,
            2: -1.5 * honest_grad,  # Poisoning
            3: np.zeros(100)        # Free-rider
        }
        ground_truth_roles = {
            0: "honest",
            1: "honest",
            2: "poisoning",
            3: "free_rider"
        }
        
        trusted_weights, excluded, metrics = engine.filter_trusted_participants(
            client_updates, ground_truth_roles=ground_truth_roles
        )
        
        self.assertIn(2, excluded, "Poisoning client 2 must be excluded")
        self.assertIn(3, excluded, "Free rider client 3 must be excluded")
        self.assertIn(0, trusted_weights, "Honest client 0 must be accepted")
        self.assertIn(1, trusted_weights, "Honest client 1 must be accepted")
        self.assertEqual(metrics["attack_detection_rate"], 1.0, f"Expected 100% detection rate, got {metrics['attack_detection_rate']}")
        self.assertEqual(metrics["false_positive_rate"], 0.0, f"Expected 0% FPR, got {metrics['false_positive_rate']}")

if __name__ == "__main__":
    unittest.main()
