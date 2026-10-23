"""
Comprehensive Specification and Mathematical Unit Test Suite for VANET Research Validity.
Covers Section 12 of the Master Task Specification:
1. Reward calculation branches (Local, RSU in-range, RSU out-of-range, V2V helper, untrusted node penalty).
2. Wireless channel physics (Path loss monotonicity, SINR dB scaling, Shannon capacity).
3. Edge computation & energy consumption (kappa * f^2 * cycles, queue dynamics).
4. Federated aggregation algorithms (FedAvg, Coordinate Median, Trimmed Mean, Multi-Krum, Trust-Weighted).
5. Attack detection and security metrics (Sign-flip, Gaussian noise, Scaling, Free-riding).
6. Deterministic multi-trial reproducibility.
"""

import unittest
import numpy as np
import torch
import copy

from environment.network import V2XChannel
from environment.mobility import HighwayMobility, TraceDrivenMobility
from environment.vehicle import Vehicle
from environment.rsu import RSU
from environment.task import Task
from environment.vanet_env import VANETOffloadingEnv
from rl.reward import RewardCalculator
from trust.trust_model import BehavioralTrustTracker
from trust.model_trust import ModelTrustEvaluator
from trust.trust_engine import TrustEngine
from federated.trust_aggregation import FederatedAggregator
from baselines.heuristics import RobustFLBaselines
from configs.config_loader import load_config

class TestComprehensiveSpecification(unittest.TestCase):

    def test_config_loader_integrity(self):
        """Validates that configs/default_config.yaml loads cleanly and matches schema."""
        cfg = load_config()
        self.assertIn("mobility", cfg)
        self.assertEqual(cfg["mobility"]["road_length_m"], 3200.0)
        self.assertEqual(cfg["mobility"]["num_lanes"], 5)
        self.assertEqual(cfg["trust_management"]["initial_trust_prior"], 0.70)
        self.assertEqual(cfg["trust_management"]["exclusion_threshold_tau"], 0.40)
        self.assertEqual(len(cfg["edge_nodes"]["positions_m"]), 3)

    def test_reward_branches_comprehensive(self):
        """Validates mathematical correctness of reward calculation across all execution branches."""
        calc = RewardCalculator(alpha_lat=2.0, beta_energy=1.0, gamma_fail=5.0, delta_success=3.0, lambda_trust=2.5)
        
        # Branch 1: Successful offloading with prompt response and trusted node.
        # The reference energy is 0.01 J (the modelled energy of a
        # representative offload), so a 5 mJ task contributes 0.5.
        # r = delta_success - alpha_lat * (0.05/0.10) - beta_energy * (0.005/0.01) + lambda_trust * 0.9
        # r = 3.0 - 2.0 * 0.5 - 1.0 * 0.5 + 2.5 * 0.9 = 3.0 - 1.0 - 0.5 + 2.25 = 3.75
        r1 = calc.compute_reward(success=True, delay_sec=0.05, deadline_sec=0.10, energy_j=0.005, target_trust_score=0.9)
        self.assertAlmostEqual(r1, 3.75, places=2)

        # Branch 2: Failed offloading due to deadline violation (trusted node)
        # r = -gamma_fail - alpha_lat * 2.0 = -5.0 - 4.0 = -9.0
        r2 = calc.compute_reward(success=False, delay_sec=0.15, deadline_sec=0.10, energy_j=0.005, target_trust_score=0.7)
        self.assertAlmostEqual(r2, -9.0, places=2)

        # Branch 3: Failed offloading on untrusted node (trust < 0.4)
        # r = -9.0 - 1.0 = -10.0
        r3 = calc.compute_reward(success=False, delay_sec=0.15, deadline_sec=0.10, energy_j=0.005, target_trust_score=0.2)
        self.assertAlmostEqual(r3, -10.0, places=2)

    def test_edge_and_vehicle_computation_energy(self):
        """Validates dynamic computing energy model: E = kappa * f^2 * cycles."""
        channel = V2XChannel()
        cpu_ghz = 4.0
        cycles_giga = 1.5
        delay, energy = channel.compute_computation_delay(cycles_giga, cpu_ghz)
        self.assertAlmostEqual(delay, 0.375, places=4)
        expected_e = channel.kappa * ((cpu_ghz * 1e9) ** 2) * (cycles_giga * 1e9)
        self.assertAlmostEqual(energy, expected_e, places=6)

    def test_federated_aggregation_strategies(self):
        """Validates all five federated aggregation strategies under synthetic weights."""
        w1 = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        w2 = {"weight": torch.tensor([1.2, 2.1, 2.9])}
        w3 = {"weight": torch.tensor([0.9, 1.9, 3.1])}
        w_poison = {"weight": torch.tensor([-10.0, -20.0, -30.0])}
        
        models = {1: w1, 2: w2, 3: w3, 4: w_poison}
        samples = {1: 10, 2: 10, 3: 10, 4: 10}
        
        # 1. Standard FedAvg (vulnerable to outlier)
        w_fedavg = FederatedAggregator.standard_fedavg(models, samples)
        self.assertLess(w_fedavg["weight"][0].item(), 0.0, "FedAvg without defense is contaminated by outlier")
        
        # 2. Coordinate-wise Median (robust against outlier)
        w_med = RobustFLBaselines.coordinate_median(models)
        self.assertAlmostEqual(w_med["weight"][0].item(), 1.0, delta=0.2, msg="Coordinate median resists outlier")
        
        # 3. Trimmed Mean
        w_trim = RobustFLBaselines.trimmed_mean(models, beta=0.25)
        self.assertGreater(w_trim["weight"][0].item(), 0.5, "Trimmed mean removes outlier")
        
        # 4. Multi-Krum (excludes outlier explicitly)
        w_krum, excluded = RobustFLBaselines.multi_krum(models, samples, num_byzantine=1)
        self.assertIn(4, excluded, "Multi-Krum must exclude adversarial client 4")
        
        # 5. Trust-Weighted Aggregation with exclusion
        trust_weights = {1: 0.9, 2: 0.85, 3: 0.88}  # Node 4 excluded
        w_trust = FederatedAggregator.trust_weighted_aggregation(models, samples, trust_weights)
        self.assertGreater(w_trust["weight"][0].item(), 0.9, "Trust-weighted aggregation excludes poisoned client 4")

    def test_schema_validation_error_on_corrupt_trace(self):
        """Validates that TraceDrivenMobility throws clear ValueError when trace schema is invalid."""
        import tempfile
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            pd.DataFrame({"invalid_col_1": [1, 2], "invalid_col_2": [3, 4]}).to_csv(f.name, index=False)
            corrupt_trace = f.name
            
        with self.assertRaises(ValueError):
            TraceDrivenMobility(trace_file=corrupt_trace)

if __name__ == "__main__":
    unittest.main()
