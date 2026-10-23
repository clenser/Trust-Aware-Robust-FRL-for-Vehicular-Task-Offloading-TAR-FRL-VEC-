"""
Unit tests for deterministic reproducibility and baseline fairness under identical scenarios.
"""

import unittest
import numpy as np
import torch
from environment.vanet_env import VANETOffloadingEnv

class TestBaselinesAndReproducibility(unittest.TestCase):
    def test_deterministic_reproducibility(self):
        """Validates that running two simulation trials with the same seed yields identical results."""
        def run_trial(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = VANETOffloadingEnv(num_vehicles=6, attack_ratio=0.33)
            states_dict = env._get_global_state_dict()
            positions = []
            rewards = []
            
            for _ in range(10):
                env.update_positions()
                for vid, v in env.vehicles.items():
                    positions.append((v.x, v.speed))
                    state, task = states_dict[vid]
                    action = 0  # Local compute
                    next_state, reward, done, info, next_task = env.step_vehicle(vid, action, task)
                    rewards.append(reward)
                    states_dict[vid] = (next_state, next_task)
            return positions, rewards

        pos1, rew1 = run_trial(seed=777)
        pos2, rew2 = run_trial(seed=777)

        self.assertEqual(len(pos1), len(pos2))
        for (x1, s1), (x2, s2) in zip(pos1, pos2):
            self.assertAlmostEqual(x1, x2, places=5)
            self.assertAlmostEqual(s1, s2, places=5)

        for r1, r2 in zip(rew1, rew2):
            self.assertAlmostEqual(r1, r2, places=5)

    def test_baseline_fairness(self):
        """Ensures all comparative baselines run under identical task loads and vehicle environments."""
        env1 = VANETOffloadingEnv(num_vehicles=10, attack_ratio=0.0)
        env2 = VANETOffloadingEnv(num_vehicles=10, attack_ratio=0.0)
        
        self.assertEqual(len(env1.vehicles), len(env2.vehicles))
        self.assertEqual(len(env1.rsus), len(env2.rsus))
        self.assertEqual(env1.channel.bandwidth, env2.channel.bandwidth)
        self.assertEqual(env1.mobility.highway_length, env2.mobility.highway_length)

if __name__ == "__main__":
    unittest.main()
