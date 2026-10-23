"""
Unit tests for environment mathematics, physical channel models, IDM mobility, and rewards.
"""

import unittest
import numpy as np
from environment.network import V2XChannel
from environment.mobility import HighwayMobility
from environment.vehicle import Vehicle
from rl.reward import RewardCalculator

class TestMathematicalModels(unittest.TestCase):
    def test_3gpp_channel_monotonicity(self):
        channel = V2XChannel(carrier_freq_ghz=5.9, bandwidth_mhz=20.0)
        np.random.seed(42)
        pl_near = channel.calculate_path_loss_db(distance_m=10.0, is_los=True)
        pl_far = channel.calculate_path_loss_db(distance_m=200.0, is_los=False)
        self.assertGreater(pl_far, pl_near, f"Path loss should increase with distance: {pl_near} vs {pl_far}")

    def test_data_rate_and_latency(self):
        channel = V2XChannel()
        rate_close = channel.calculate_data_rate_mbps(distance_m=20.0, max_range=300.0)
        rate_edge = channel.calculate_data_rate_mbps(distance_m=250.0, max_range=300.0)
        self.assertTrue(0.5 <= rate_close <= 54.0)
        self.assertGreaterEqual(rate_close, rate_edge)

        # Transmission delay test: 1.0 MB data
        delay_sec, energy_j = channel.compute_transmission_delay(data_size_mb=1.0, distance_m=50.0)
        self.assertGreater(delay_sec, 0.0)
        self.assertLess(delay_sec, 5.0)
        self.assertAlmostEqual(energy_j, channel.tx_power * delay_sec, places=4)

        # Out of range test
        out_delay, out_energy = channel.compute_transmission_delay(data_size_mb=1.0, distance_m=500.0, max_range=300.0)
        self.assertEqual(out_delay, float('inf'))

    def test_computation_delay_and_dynamic_energy(self):
        channel = V2XChannel()
        # 2.0 Giga-cycles on 4.0 GHz CPU
        delay, energy = channel.compute_computation_delay(cpu_cycles_giga=2.0, cpu_freq_ghz=4.0)
        self.assertAlmostEqual(delay, 0.5, places=5)  # 2.0 / 4.0 = 0.5 seconds
        expected_energy = channel.kappa * ((4.0 * 1e9) ** 2) * (2.0 * 1e9)
        self.assertAlmostEqual(energy, expected_energy, places=5)

    def test_idm_mobility_collision_free(self):
        np.random.seed(123)
        highway = HighwayMobility(highway_length=2400.0, num_lanes=5, dt=0.5)
        vehicles = {i: Vehicle(vehicle_id=i, x=i*80.0, y=0.0, speed=22.0, lane=1) for i in range(10)}
        highway.initialize_spaced_positions(vehicles)

        for _ in range(50):
            highway.step(vehicles)
            for v in vehicles.values():
                self.assertTrue(14.0 <= v.speed <= 32.0, f"Speed {v.speed} outside bounds")
                self.assertTrue(0.0 <= v.x < 2400.0, f"Position {v.x} outside highway loop")

    def test_reward_scaling(self):
        calc = RewardCalculator()
        # Successful execution under deadline, with a realistic energy value
        # for the corrected computation-energy model (a few mJ).
        r_success = calc.compute_reward(success=True, delay_sec=0.05, deadline_sec=0.10, energy_j=0.005, target_trust_score=0.9)
        self.assertGreater(r_success, 0.0, f"Successful offload should yield positive reward: {r_success}")

        # Failed execution (deadline violation)
        r_fail = calc.compute_reward(success=False, delay_sec=0.15, deadline_sec=0.10, energy_j=0.2, target_trust_score=0.8)
        self.assertLess(r_fail, 0.0, f"Failed offload should yield negative reward: {r_fail}")

        # Untrusted target penalty
        r_fail_untrusted = calc.compute_reward(success=False, delay_sec=0.15, deadline_sec=0.10, energy_j=0.2, target_trust_score=0.2)
        self.assertLess(r_fail_untrusted, r_fail, "Untrusted target failure should have additional penalty")

    def test_energy_term_is_not_capped(self):
        """Very different energy outcomes must remain distinguishable."""
        calc = RewardCalculator()
        low = calc.compute_reward(success=True, delay_sec=0.05, deadline_sec=0.10,
                                  energy_j=0.001, target_trust_score=0.9)
        high = calc.compute_reward(success=True, delay_sec=0.05, deadline_sec=0.10,
                                   energy_j=0.05, target_trust_score=0.9)
        self.assertGreater(low, high,
                           "a 50x more expensive task must yield a lower reward")

    def test_energy_normalisation_is_documented(self):
        """The reference energy must match the modelled task energy scale."""
        calc = RewardCalculator()
        self.assertAlmostEqual(calc.e_norm, 0.01, places=6)
        # A representative 0.22 Gcycle task on a 4 GHz core costs ~3.5 mJ, so
        # the reference energy puts a typical offload at ~0.35.
        channel = V2XChannel()
        _, energy = channel.compute_computation_delay(0.22, 4.0)
        self.assertTrue(2e-3 < energy < 6e-3,
                        f"representative task energy out of range: {energy}")

if __name__ == "__main__":
    unittest.main()
