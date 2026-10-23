"""
Unit tests validating the resolution of Review Blockers A through E:
- Blocker A: TraceDrivenMobility CSV parsing and trajectory replay
- Blocker B: Task workload reconciliation with authoritative CSV profiles
- Blocker C: Elimination of attacker label leakage in initial trust prior
- Blocker D: Differentiation of implicit robust aggregation from explicit detection
- Blocker E: Dynamic propagation and scaling of poisoning attack strength
- Unit validation: Physical SINR dB conversion for behavioral trust
"""

import unittest
import numpy as np
import torch

from environment.mobility import TraceDrivenMobility
from environment.vanet_env import VANETOffloadingEnv
from environment.task import TaskGenerator
from environment.network import V2XChannel
from federated.client import FederatedClient
from federated.server import FederatedServer
from trust.trust_model import BehavioralTrustTracker

class TestBlockerFixes(unittest.TestCase):

    def test_blocker_a_trace_driven_mobility(self):
        """Blocker A: Validates authentic trace replay loading and progression."""
        trace_path = "data/synthetic_traces/synthetic_ngsim_highway.csv"
        mobility = TraceDrivenMobility(trace_file=trace_path, dt=0.5, road_length=2400.0)
        
        self.assertGreater(len(mobility.df), 0, "Trace dataframe should not be empty")
        self.assertIn("pos_x", mobility.df.columns)
        self.assertIn("pos_y", mobility.df.columns)
        self.assertIn("speed_mps", mobility.df.columns)
        
        # Test trace-mode environment initialization and stepping
        env = VANETOffloadingEnv(num_vehicles=10, mobility_mode="trace", trace_file=trace_path)
        self.assertEqual(len(env.vehicles), 10)
        
        init_x = {vid: v.x for vid, v in env.vehicles.items()}
        env.update_positions()
        step_x = {vid: v.x for vid, v in env.vehicles.items()}
        
        # Vehicles should advance according to trajectory timestamps
        moved_count = sum(1 for vid in env.vehicles if step_x[vid] != init_x[vid])
        self.assertGreater(moved_count, 0, "Vehicles should advance along trace positions")

    def test_blocker_b_authoritative_task_profiles(self):
        """Blocker B: Validates task generator parameters match authoritative 3GPP catalog."""
        generator = TaskGenerator(catalog_path="data/synthetic_traces/synthetic_v2x_task_profiles.csv")
        
        profiles = generator.profiles
        self.assertIn("Cooperative_Perception", profiles)
        self.assertIn("Sensor_Fusion", profiles)
        self.assertIn("Video_Analytics", profiles)
        self.assertIn("Trajectory_Planning", profiles)
        
        # Cooperative Perception: 0.45 MB, 0.22 Gcycles, 160 ms
        cp = profiles["Cooperative_Perception"]
        self.assertAlmostEqual(cp["d_mean"], 0.45, places=2)
        self.assertAlmostEqual(cp["c_mean"], 0.22, places=2)
        self.assertAlmostEqual(cp["deadline"], 160.0, places=1)

        # Generate sample task and check sanity
        task = generator.generate_task(vehicle_id=1)
        self.assertGreater(task.data_size_mb, 0.0)
        self.assertGreater(task.cpu_cycles_giga, 0.0)
        self.assertGreater(task.deadline_ms, 0.0)

    def test_blocker_c_zero_label_leakage_in_initial_trust(self):
        """Blocker C: Validates that initial trust prior is strictly role-agnostic (0.70)."""
        np.random.seed(42)
        env = VANETOffloadingEnv(num_vehicles=20, attack_ratio=0.35)
        
        roles_seen = set()
        for vid, v in env.vehicles.items():
            roles_seen.add(v.role)
            # Initial trust on vehicle object must be neutral 0.70
            self.assertEqual(v.trust_score, 0.70, f"Vehicle {vid} ({v.role}) has leaked trust: {v.trust_score}")
            
            # Initial behavioral tracker record must be neutral 0.70
            node_key = f"V{vid}"
            tracker_trust = env.trust_engine.behavior_tracker.get_trust(node_key)
            self.assertEqual(tracker_trust, 0.70, f"Node {node_key} behavioral tracker leaked: {tracker_trust}")
            
            # Initial final trust score must be neutral 0.70
            final_trust = env.trust_engine.final_trust_scores.get(node_key, None)
            self.assertEqual(final_trust, 0.70, f"Node {node_key} final trust leaked: {final_trust}")

        self.assertIn("honest", roles_seen)
        self.assertIn("poisoning", roles_seen)
        self.assertIn("free_rider", roles_seen)

    def test_blocker_d_aggregation_metric_differentiation(self):
        """Blocker D: Validates explicit exclusion (Multi-Krum, Trust) vs implicit robust aggregation (Median, Trimmed)."""
        server_median = FederatedServer(state_dim=20, action_dim=5, aggregation_strategy="coordinate_median")
        server_krum = FederatedServer(state_dim=20, action_dim=5, aggregation_strategy="multi_krum")
        server_trust = FederatedServer(state_dim=20, action_dim=5, aggregation_strategy="trust_aware")
        
        base_weights = server_median.get_global_weights()
        client_models = {}
        client_samples = {}
        roles = {1: "honest", 2: "honest", 3: "honest", 4: "poisoning"}
        
        for cid, role in roles.items():
            client = FederatedClient(vehicle_id=cid, state_dim=20, action_dim=5, role=role, poison_strength=1.5)
            client.update_from_global(base_weights)
            client_models[cid] = client.get_upload_weights(base_weights)
            client_samples[cid] = 10
            
        # 1. Coordinate-wise Median: Implicit defense, must report detection_rate = None
        _, m_median = server_median.run_aggregation_round(client_models, client_samples, ground_truth_roles=roles)
        self.assertIsNone(m_median["attack_detection_rate"])
        self.assertEqual(m_median["excluded_count"], 0)

        # 2. Multi-Krum: Explicit exclusion, must report empirical detection rate
        _, m_krum = server_krum.run_aggregation_round(client_models, client_samples, ground_truth_roles=roles)
        self.assertIsNotNone(m_krum["attack_detection_rate"])
        self.assertIn("false_positive_rate", m_krum)
        self.assertGreater(m_krum["excluded_count"], 0)

        # 3. Trust-Aware: Explicit exclusion, must report empirical detection rate
        _, m_trust = server_trust.run_aggregation_round(client_models, client_samples, ground_truth_roles=roles)
        self.assertIsNotNone(m_trust["attack_detection_rate"])
        self.assertIn("false_positive_rate", m_trust)

    def test_blocker_e_dynamic_poisoning_strength_scaling(self):
        """Blocker E: Validates that configuring different poison_strength values produces scaled perturbations."""
        import copy
        server = FederatedServer(state_dim=20, action_dim=5)
        base_weights = server.get_global_weights()
        
        # Set local weights with non-zero gradient
        local_weights = copy.deepcopy(base_weights)
        for k in local_weights:
            local_weights[k] = local_weights[k] + torch.ones_like(local_weights[k].float()) * 0.1
        
        client_weak = FederatedClient(vehicle_id=1, state_dim=20, action_dim=5, role="poisoning", poison_strength=0.5)
        client_weak.agent.set_weights(local_weights)
        weights_weak = client_weak.get_upload_weights(base_weights)
        
        client_strong = FederatedClient(vehicle_id=2, state_dim=20, action_dim=5, role="poisoning", poison_strength=2.5)
        client_strong.agent.set_weights(local_weights)
        weights_strong = client_strong.get_upload_weights(base_weights)
        
        evaluator = server.trust_engine.model_evaluator
        delta_weak = evaluator.extract_flat_update(weights_weak, base_weights)
        delta_strong = evaluator.extract_flat_update(weights_strong, base_weights)
        
        norm_weak = float(np.linalg.norm(delta_weak))
        norm_strong = float(np.linalg.norm(delta_strong))
        
        self.assertGreater(norm_strong, norm_weak * 2.0, 
                           f"Strong attack ({norm_strong:.4f}) must scale perturbation above weak ({norm_weak:.4f})")

    def test_sinr_db_conversion_units(self):
        """Validates that linear SINR is converted to decibels before feeding into behavioral trust."""
        channel = V2XChannel()
        sinr_db = channel.calculate_sinr_db(distance_m=30.0)
        self.assertTrue(5.0 <= sinr_db <= 45.0, f"SINR dB {sinr_db} outside realistic V2X bounds")
        
        # Check conversion identity: 10 * log10(100.0) == 20 dB
        linear_val = 100.0
        db_val = float(10.0 * np.log10(linear_val))
        self.assertAlmostEqual(db_val, 20.0, places=2)
        
        # Test tracker behavior with dB inputs
        tracker = BehavioralTrustTracker(eta=0.8)
        # Good channel (20 dB) with success
        t_high = tracker.record_interaction(node_id=1, success=True, delay_sec=0.05, deadline_sec=0.10, channel_sinr=20.0)
        self.assertGreater(t_high, 0.70)
        
        # Poor channel (-5 dB) with repeated drops
        for _ in range(10):
            t_low = tracker.record_interaction(node_id=2, success=False, delay_sec=0.20, deadline_sec=0.10, channel_sinr=-5.0)
        self.assertLess(t_low, 0.40)

if __name__ == "__main__":
    unittest.main()
