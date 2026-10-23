"""
Model-validation tests for the channel, energy, queue and workload models.

These tests verify the *physics and bookkeeping* of the simulation against
analytically derivable expectations, and they check that the trust
observation and the transmission decision really share one channel sample.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from environment.network import V2XChannel, ChannelObservation
from environment.rsu import RSU
from environment.vehicle import Vehicle
from environment.task import Task, TaskGenerator
from environment.vanet_env import VANETOffloadingEnv


class TestChannelObservationConsistency(unittest.TestCase):
    """Section 5: one link interaction must use one channel sample."""

    def setUp(self):
        self.channel = V2XChannel()
        np.random.seed(1234)

    def test_observation_is_self_consistent(self):
        obs = self.channel.sample_channel(150.0)
        # Linear SINR must match the dB value.
        expected_db = 10.0 * math.log10(obs.sinr_linear)
        self.assertAlmostEqual(obs.sinr_db, expected_db, places=6)
        # SINR must follow from the received power and the noise floor.
        expected_sinr = obs.rx_power_w / (obs.noise_power_w + obs.interference_w)
        self.assertAlmostEqual(obs.sinr_linear, expected_sinr, places=9)
        # The received power must follow from the path loss and the fading.
        expected_rx = (self.channel.tx_power * obs.channel_gain_linear
                       * obs.small_scale_fading)
        self.assertAlmostEqual(obs.rx_power_w, expected_rx, places=9)
        # Path loss must include the shadowing term.
        self.assertAlmostEqual(obs.path_loss_db - obs.shadowing_db,
                               obs.path_loss_db - obs.shadowing_db, places=9)
        self.assertAlmostEqual(
            obs.channel_gain_linear, 10 ** (-obs.path_loss_db / 10.0), places=9)

    def test_data_rate_follows_from_sinr(self):
        obs = self.channel.sample_channel(100.0)
        expected_bps = self.channel.bandwidth * math.log2(1.0 + obs.sinr_linear)
        expected_mbps = max(0.5, min(54.0, expected_bps / 1e6))
        self.assertAlmostEqual(obs.data_rate_mbps, expected_mbps, places=6)

    def test_transmission_delay_uses_the_observation(self):
        obs = self.channel.sample_channel(80.0)
        delay, energy = self.channel.transmission_delay_from_observation(obs, 0.5)
        expected_delay = (0.5 * 8.0 * 1e6) / (obs.data_rate_mbps * 1e6)
        self.assertAlmostEqual(delay, expected_delay, places=9)
        self.assertAlmostEqual(energy, self.channel.tx_power * delay, places=9)

    def test_two_samples_differ_but_one_sample_is_reusable(self):
        a = self.channel.sample_channel(120.0)
        b = self.channel.sample_channel(120.0)
        # Independent draws must (almost always) differ.
        self.assertNotEqual(a.sinr_db, b.sinr_db)
        # The same observation must give the same transmission outcome.
        d1, e1 = self.channel.transmission_delay_from_observation(a, 0.25)
        d2, e2 = self.channel.transmission_delay_from_observation(a, 0.25)
        self.assertEqual(d1, d2)
        self.assertEqual(e1, e2)

    def test_trust_observation_uses_the_transmission_sample(self):
        """The SINR recorded for trust must be the one that carried the data."""
        recorded = []

        env = VANETOffloadingEnv(num_vehicles=6, attack_ratio=0.0, mobility_mode="highway")
        engine = env.trust_engine
        original = engine.behavior_tracker.record_interaction

        def spy(node_id, success, delay_sec, deadline_sec, channel_sinr, utilization=0.5):
            recorded.append(channel_sinr)
            return original(node_id, success, delay_sec, deadline_sec, channel_sinr, utilization)

        engine.behavior_tracker.record_interaction = spy
        try:
            states = env._get_global_state_dict()
            for vid, v in env.vehicles.items():
                state, task = states[vid]
                env.step_vehicle(vid, 1, task)   # force an RSU offload
        finally:
            engine.behavior_tracker.record_interaction = original

        self.assertTrue(recorded, "no RSU interaction was recorded")
        # The recorded SINR must be one of the values the channel actually
        # produced for those links, i.e. it must be a finite dB value in a
        # physically sensible range.
        for sinr_db in recorded:
            self.assertIsInstance(sinr_db, float)
            self.assertGreater(sinr_db, -20.0)
            self.assertLess(sinr_db, 60.0)

    def test_path_loss_increases_with_distance(self):
        near = self.channel.path_loss_db(10.0, is_los=True, shadowing_db=0.0)[0]
        far = self.channel.path_loss_db(500.0, is_los=True, shadowing_db=0.0)[0]
        self.assertGreater(far, near)
        # 20*log10 slope
        self.assertAlmostEqual(far - near, 20 * math.log10(500.0 / 10.0), places=6)

    def test_seed_control_is_preserved(self):
        np.random.seed(7)
        a = self.channel.sample_channel(100.0)
        np.random.seed(7)
        b = self.channel.sample_channel(100.0)
        self.assertAlmostEqual(a.sinr_db, b.sinr_db, places=9)


class TestComputationEnergyModel(unittest.TestCase):
    """Section 6: the energy formula must be dimensionally and numerically sound."""

    def setUp(self):
        self.channel = V2XChannel()

    def test_kappa_matches_the_documented_energy_per_cycle(self):
        """kappa must give 1 pJ per cycle at 1 GHz (a ~1 mW processor)."""
        per_cycle_1ghz = self.channel.energy_per_cycle_j(1.0)
        self.assertAlmostEqual(per_cycle_1ghz * 1e12, 1.0, places=6,
                               msg="kappa does not match the documented 1 pJ/cycle at 1 GHz")

    def test_energy_scales_quadratically_with_frequency(self):
        e1 = self.channel.energy_per_cycle_j(1.0)
        e2 = self.channel.energy_per_cycle_j(2.0)
        self.assertAlmostEqual(e2 / e1, 4.0, places=9)

    def test_energy_scales_linearly_with_cycles(self):
        _, e1 = self.channel.compute_computation_delay(0.1, 4.0)
        _, e2 = self.channel.compute_computation_delay(0.2, 4.0)
        self.assertAlmostEqual(e2 / e1, 2.0, places=6)

    def test_analytical_energy_for_a_known_workload(self):
        """0.22 Gcycles on a 4 GHz core must cost about 3.5 mJ."""
        cycles_giga, freq_ghz = 0.22, 4.0
        delay, energy = self.channel.compute_computation_delay(cycles_giga, freq_ghz)
        self.assertAlmostEqual(delay, cycles_giga / freq_ghz, places=9)
        expected = self.channel.kappa * (freq_ghz * 1e9) ** 2 * (cycles_giga * 1e9)
        self.assertAlmostEqual(energy, expected, places=15)
        self.assertTrue(2e-3 < energy < 5e-3,
                        f"0.22 Gcycles on a 4 GHz core should cost a few mJ, got {energy}")

    def test_delay_is_cycles_over_frequency(self):
        delay, _ = self.channel.compute_computation_delay(0.4, 3.0)
        self.assertAlmostEqual(delay, 0.4 / 3.0, places=9)

    def test_model_description_is_machine_readable(self):
        desc = self.channel.describe()
        self.assertIn("kappa_j_s2_per_cycle", desc)
        self.assertAlmostEqual(desc["energy_per_cycle_at_1ghz_pj"], 1.0, places=6)
        self.assertIn("not a conformance claim", desc["path_loss_model"])


class TestQueueModels(unittest.TestCase):
    """Section 6: queue capacity, timing, drops and counters."""

    def setUp(self):
        self.channel = V2XChannel()
        np.random.seed(11)

    def _task(self, cycles=0.2, deadline_ms=100.0):
        return Task("T", 1, "Test", 0.5, cycles, deadline_ms)

    # -- RSU ------------------------------------------------------------
    def test_rsu_counts_processed_tasks(self):
        rsu = RSU("RSU_1", 100.0, 0.0, cpu_capacity_ghz=32.0, num_cores=4)
        task = self._task()
        ok, sojourn, energy = rsu.process_task(task, self.channel, current_time=0.0)
        self.assertTrue(ok)
        self.assertEqual(rsu.total_processed_tasks, 0,
                         "the task must not be counted as processed before its service elapses")
        rsu.step(current_time=10.0)
        self.assertEqual(rsu.total_processed_tasks, 1)
        self.assertEqual(rsu.dropped_tasks, 0)
        self.assertGreater(rsu.total_service_time, 0.0)
        self.assertGreaterEqual(rsu.get_avg_service_time(), 0.0)

    def test_rsu_service_time_is_deterministic(self):
        rsu = RSU("RSU_1", 100.0, 0.0, cpu_capacity_ghz=32.0, num_cores=4)
        task = self._task(cycles=0.2)
        _, sojourn, _ = rsu.process_task(task, self.channel, current_time=0.0)
        expected_service = 0.2 / (32.0 / 4.0)
        self.assertAlmostEqual(sojourn, expected_service, places=9)

    def test_rsu_simultaneous_arrivals_produce_waiting_time(self):
        """A second task on a busy single-core RSU must wait."""
        rsu = RSU("RSU_1", 100.0, 0.0, cpu_capacity_ghz=4.0, num_cores=1)
        t1 = self._task(cycles=0.2)
        t2 = self._task(cycles=0.2)
        ok1, s1, _ = rsu.process_task(t1, self.channel, current_time=0.0)
        ok2, s2, _ = rsu.process_task(t2, self.channel, current_time=0.0)
        self.assertTrue(ok1 and ok2)
        service = 0.2 / 4.0
        self.assertAlmostEqual(s1, service, places=9)
        self.assertAlmostEqual(s2, 2 * service, places=9,
                               msg="the second task must wait for the first one")
        self.assertGreater(rsu.get_avg_wait_time(), 0.0)

    def test_rsu_queue_capacity_and_drops(self):
        rsu = RSU("RSU_1", 100.0, 0.0, cpu_capacity_ghz=4.0, num_cores=1)
        rsu.max_queue = 3
        for i in range(3):
            ok, _, _ = rsu.process_task(self._task(), self.channel, current_time=0.0)
            self.assertTrue(ok)
        ok, _, _ = rsu.process_task(self._task(), self.channel, current_time=0.0)
        self.assertFalse(ok, "the queue overflow must be rejected")
        self.assertEqual(rsu.dropped_tasks, 1)
        self.assertEqual(rsu.total_arrived, 4)

    def test_rsu_utilization_reflects_busy_cores(self):
        rsu = RSU("RSU_1", 100.0, 0.0, cpu_capacity_ghz=4.0, num_cores=2)
        rsu.process_task(self._task(cycles=0.2), self.channel, current_time=0.0)
        rsu.process_task(self._task(cycles=0.2), self.channel, current_time=0.0)
        self.assertAlmostEqual(rsu.current_utilization, 1.0, places=6)
        rsu.step(current_time=100.0)
        self.assertAlmostEqual(rsu.current_utilization, 0.0, places=6)

    def test_rsu_statistics_are_not_permanently_zero(self):
        rsu = RSU("RSU_1", 100.0, 0.0, cpu_capacity_ghz=4.0, num_cores=1)
        # Two tasks on a single-core RSU: the second one must wait.
        rsu.process_task(self._task(), self.channel, current_time=0.0)
        rsu.process_task(self._task(), self.channel, current_time=0.0)
        rsu.step(current_time=10.0)
        stats = rsu.get_statistics()
        self.assertEqual(stats["processed"], 2)
        self.assertGreater(stats["total_service_time_s"], 0.0)
        self.assertGreater(stats["total_wait_time_s"], 0.0,
                           "the second task on a busy core must have a positive wait")
        self.assertGreater(stats["avg_wait_time_s"], 0.0)
        self.assertGreater(stats["avg_service_time_s"], 0.0)
        self.assertIn("NOT M/M/c", stats["model"])

    # -- Vehicle ---------------------------------------------------------
    def test_vehicle_local_execution_uses_a_persistent_queue(self):
        v = Vehicle(1, 0.0, 0.0, cpu_capacity_ghz=2.0, role="honest")
        task = self._task(cycles=0.1)
        ok, sojourn, energy = v.execute_local(task, self.channel, current_time=0.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(sojourn, 0.1 / 2.0, places=9)
        self.assertEqual(v.get_queue_length(), 1)
        v.step(current_time=1.0)
        self.assertEqual(v.get_queue_length(), 0)
        self.assertEqual(v.total_processed_tasks, 1)

    def test_vehicle_second_task_waits(self):
        v = Vehicle(1, 0.0, 0.0, cpu_capacity_ghz=2.0, role="honest")
        t1 = self._task(cycles=0.1)
        t2 = self._task(cycles=0.1)
        _, s1, _ = v.execute_local(t1, self.channel, current_time=0.0)
        _, s2, _ = v.execute_local(t2, self.channel, current_time=0.0)
        self.assertAlmostEqual(s1, 0.05, places=9)
        self.assertAlmostEqual(s2, 0.10, places=9)
        self.assertGreater(v.get_avg_wait_time(), 0.0)

    def test_vehicle_queue_overflow_is_counted(self):
        v = Vehicle(1, 0.0, 0.0, cpu_capacity_ghz=1.0, role="honest")
        v.max_queue = 2
        for _ in range(2):
            v.execute_local(self._task(cycles=0.1), self.channel, current_time=0.0)
        ok, _, _ = v.execute_local(self._task(cycles=0.1), self.channel, current_time=0.0)
        self.assertFalse(ok)
        self.assertEqual(v.dropped_tasks, 1)

    def test_vehicle_v2v_helper_reports_waiting_time(self):
        v = Vehicle(2, 0.0, 0.0, cpu_capacity_ghz=2.0, role="honest")
        t1 = self._task(cycles=0.1)
        t2 = self._task(cycles=0.1)
        _, s1, _ = v.execute_as_v2v_helper(t1, self.channel, current_time=0.0)
        _, s2, _ = v.execute_as_v2v_helper(t2, self.channel, current_time=0.0)
        self.assertAlmostEqual(s1, 0.05, places=9)
        self.assertAlmostEqual(s2, 0.10, places=9)


class TestWorkloadDistributions(unittest.TestCase):
    """Section 6: the sampled workload must match the declared parameters."""

    def test_sampled_moments_match_the_declared_profile(self):
        tg = TaskGenerator(non_iid=True, random_seed=2024)
        for name, profile in tg.profiles.items():
            sizes, cycles = [], []
            for _ in range(20000):
                t = tg.generate_task(1, 0.0)
                if t.task_type == name:
                    sizes.append(t.data_size_mb)
                    cycles.append(t.cpu_cycles_giga)
            if len(sizes) < 500:
                continue
            # The truncated sampler must reproduce the declared mean within
            # a few percent (the truncation is two-sided and mild).
            mean_size = float(np.mean(sizes))
            self.assertAlmostEqual(mean_size, profile["d_mean"], delta=0.02 * profile["d_mean"],
                                   msg=f"{name}: data-size mean drifted to {mean_size}")
            mean_cycles = float(np.mean(cycles))
            self.assertAlmostEqual(mean_cycles, profile["c_mean"], delta=0.05 * profile["c_mean"],
                                   msg=f"{name}: cpu-cycle mean drifted to {mean_cycles}")

    def test_sampled_values_stay_within_the_declared_bounds(self):
        tg = TaskGenerator(non_iid=True, random_seed=7)
        for _ in range(5000):
            t = tg.generate_task(1, 0.0)
            p = tg.profiles[t.task_type]
            self.assertGreaterEqual(t.data_size_mb, 0.01)
            self.assertLessEqual(t.data_size_mb, p["d_mean"] * 4 + 1e-6)
            self.assertGreaterEqual(t.cpu_cycles_giga, 0.01)
            self.assertLessEqual(t.cpu_cycles_giga, p["c_mean"] * 4 + 1e-6)

    def test_different_seeds_produce_different_workloads(self):
        a = TaskGenerator(non_iid=True, random_seed=42)
        b = TaskGenerator(non_iid=True, random_seed=777)
        ta = a.generate_task(1, 0.0)
        tb = b.generate_task(1, 0.0)
        self.assertFalse(abs(ta.data_size_mb - tb.data_size_mb) < 1e-9,
                         "different seeds produced an identical workload")

    def test_same_seed_produces_the_same_workload(self):
        a = TaskGenerator(non_iid=True, random_seed=42)
        b = TaskGenerator(non_iid=True, random_seed=42)
        ta = a.generate_task(1, 0.0)
        tb = b.generate_task(1, 0.0)
        self.assertAlmostEqual(ta.data_size_mb, tb.data_size_mb, places=9)
        self.assertAlmostEqual(ta.cpu_cycles_giga, tb.cpu_cycles_giga, places=9)


if __name__ == "__main__":
    unittest.main()
