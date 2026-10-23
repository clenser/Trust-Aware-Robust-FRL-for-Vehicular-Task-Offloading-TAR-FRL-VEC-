"""
Tests for the version-2 environment semantics and the corrected FL lifecycle.

* the action names an RSU and the simulator honours it (no silent re-association),
* uplink bandwidth is shared among simultaneous uplinks,
* the reward does not favour offloading by construction,
* the vehicle service order is seeded and not biased towards low identifiers,
* the federated runner no longer crashes under partial participation, reports
  measured communication volume, and reports an honest per-round status,
* user-specified RSU placement is validated,
* console configuration changes are range-checked.
"""

import copy
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from environment.vanet_env import VANETOffloadingEnv, STATE_DIM, STATE_DIM_V21
from environment.network import V2XChannel
from environment.task import Task
from rl.reward import RewardCalculator
from trust import alt_detectors


def _env(**kw):
    np.random.seed(7)
    return VANETOffloadingEnv(num_vehicles=6, attack_ratio=0.0, mobility_mode="highway",
                              random_seed=7, **kw)


def _task():
    return Task("t", 1, "Cooperative_Perception", 0.45, 0.22, 160.0)


class TestRSUChoiceIsHonoured(unittest.TestCase):
    def test_out_of_coverage_choice_fails_without_energy_or_trust_penalty(self):
        env = _env()
        v = env.vehicles[1]
        rsu = env.rsus[2]
        v.x = 0.0                       # RSU_3 is at ~2460 m, far outside its range
        before = env.trust_engine.behavior_tracker.get_trust(rsu.rsu_id)
        _, _, _, info, _ = env.step_vehicle(1, 3, _task())
        self.assertFalse(info["success"])
        self.assertEqual(info["failure_reason"], "out_of_coverage")
        self.assertEqual(info["energy_j"], 0.0)
        self.assertEqual(env.trust_engine.behavior_tracker.get_trust(rsu.rsu_id), before)

    def test_in_coverage_choice_reaches_that_rsu(self):
        env = _env()
        v = env.vehicles[1]
        v.x, v.y = env.rsus[0].x, 0.0
        _, _, _, info, _ = env.step_vehicle(1, 1, _task())
        self.assertEqual(info["target"], "RSU_1")

    def test_legacy_mode_reassociates(self):
        cfg = load_config()
        cfg = copy.deepcopy(cfg)
        cfg["environment_model"]["honor_rsu_choice"] = False
        env = VANETOffloadingEnv(num_vehicles=6, attack_ratio=0.0, mobility_mode="highway",
                                 random_seed=7, config=cfg)
        v = env.vehicles[1]
        v.x, v.y = env.rsus[0].x, 0.0
        _, _, _, info, _ = env.step_vehicle(1, 3, _task())   # asks for RSU_3, out of range
        self.assertEqual(info["target"], "RSU_1")


class TestUplinkContention(unittest.TestCase):
    def test_full_ue_channel_until_cell_is_saturated(self):
        ch = V2XChannel()
        obs = ch.sample_channel(100.0)
        d1, _ = ch.transmission_delay_shared(obs, 0.45, 100.0, 1)
        d5, _ = ch.transmission_delay_shared(obs, 0.45, 100.0, 5)
        base, _ = ch.transmission_delay_from_observation(obs, 0.45)
        self.assertAlmostEqual(d1, base, places=9)
        self.assertAlmostEqual(d5, base, places=9)

    def test_delay_grows_beyond_saturation(self):
        ch = V2XChannel()
        # Strong link: the 54 Mbps clip does not bind once bandwidth shrinks.
        obs = ch.sample_channel(10.0)
        d5, _ = ch.transmission_delay_shared(obs, 0.45, 100.0, 5)
        d20, _ = ch.transmission_delay_shared(obs, 0.45, 100.0, 20)
        self.assertGreater(d20, d5)

    def test_energy_is_power_times_airtime(self):
        ch = V2XChannel()
        obs = ch.sample_channel(50.0)
        d, e = ch.transmission_delay_shared(obs, 0.45, 100.0, 12)
        self.assertAlmostEqual(e, ch.tx_power * d, places=12)


class TestRewardIsNotBiasedTowardsOffloading(unittest.TestCase):
    def test_default_trust_weight_is_zero(self):
        cfg = load_config()
        self.assertEqual(float(cfg["reward"]["lambda_trust"]), 0.0)
        env = _env()
        self.assertEqual(env.reward_calculator.lambda_trust, 0.0)

    def test_reward_is_independent_of_target_trust_when_weight_is_zero(self):
        rc = RewardCalculator(lambda_trust=0.0)
        r_hi = rc.compute_reward(True, 0.1, 0.16, 0.01, target_trust_score=1.0)
        r_lo = rc.compute_reward(True, 0.1, 0.16, 0.01, target_trust_score=0.1)
        self.assertEqual(r_hi, r_lo)

    def test_legacy_weight_still_available(self):
        rc = RewardCalculator(lambda_trust=2.5)
        self.assertGreater(rc.compute_reward(True, 0.1, 0.16, 0.01, 1.0),
                           rc.compute_reward(True, 0.1, 0.16, 0.01, 0.0))


class TestStateAndOrdering(unittest.TestCase):
    def test_state_has_rsu_load_features(self):
        env = _env()
        state, task = env._get_global_state_dict()[1]
        self.assertEqual(len(state), env.state_dim)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in state[14:17]))

    def test_default_state_is_v21_with_coverage_features(self):
        env = _env()
        self.assertEqual(env.state_dim, STATE_DIM_V21)
        v = env.vehicles[1]
        v.x, v.y = env.rsus[0].x, 0.0
        state, _ = env._get_global_state_dict()[1]
        self.assertEqual(state[17], 1.0)                      # in coverage of RSU_1
        self.assertEqual(state[18], 0.0)                      # RSU_2 is ~1000 m away
        self.assertLess(state[4], 0.5)                        # relative distance inside the boundary
        self.assertGreaterEqual(state[5], 0.5)                # outside it

    def test_v20_state_is_still_available(self):
        cfg = copy.deepcopy(load_config())
        cfg["environment_model"]["state_features"] = "v2.0"
        env = VANETOffloadingEnv(num_vehicles=6, attack_ratio=0.0, mobility_mode="highway",
                                 random_seed=7, config=cfg)
        self.assertEqual(env.state_dim, STATE_DIM)
        self.assertEqual(len(env._get_global_state_dict()[1][0]), STATE_DIM)

    def test_vehicle_order_is_a_seeded_permutation(self):
        a, b = _env(), _env()
        oa = [a.vehicle_order() for _ in range(5)]
        ob = [b.vehicle_order() for _ in range(5)]
        self.assertEqual(oa, ob)
        for o in oa:
            self.assertEqual(sorted(o), sorted(a.vehicles.keys()))
        self.assertGreater(len({tuple(o) for o in oa}), 1, "order never changes")


class TestRsuPlacementValidation(unittest.TestCase):
    def test_custom_positions_applied_on_highway(self):
        env = VANETOffloadingEnv(num_vehicles=6, mobility_mode="highway", random_seed=1,
                                 rsu_x_positions=[300, 1200, 2800])
        self.assertEqual([r.x for r in env.rsus], [300.0, 1200.0, 2800.0])

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            VANETOffloadingEnv(num_vehicles=6, mobility_mode="highway", random_seed=1,
                               rsu_x_positions=[300, 1200, 9000])

    def test_wrong_count_rejected(self):
        with self.assertRaises(ValueError):
            VANETOffloadingEnv(num_vehicles=6, mobility_mode="highway", random_seed=1,
                               rsu_x_positions=[300, 1200])

    def test_rejected_for_trace_replay(self):
        with self.assertRaises(ValueError):
            VANETOffloadingEnv(num_vehicles=6, mobility_mode="trace", random_seed=1,
                               rsu_x_positions=[100, 200, 300])

    def test_scene_description_matches_backend_numbers(self):
        env = _env()
        scene = env.scene_description()
        self.assertEqual(scene["road_length_m"], env.road_length)
        self.assertEqual([r["x"] for r in scene["rsus"]], [r.x for r in env.rsus])


class TestFederatedRunnerLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.experiment_runner import run_simulation_trial, EvaluationConfig
        cls.trial = staticmethod(run_simulation_trial)
        cls.ev = EvaluationConfig(episodes=1, steps_per_episode=3, seed_offset=500000)
        cfg = copy.deepcopy(load_config())
        cfg["federated_learning"]["local_epochs"] = 2
        cls.cfg = cfg

    def test_partial_participation_does_not_crash_and_records_uploads(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["federated_learning"]["participation_fraction"] = 0.5
        rec = self.trial(seed=3, config=cfg, eval_config=self.ev, num_vehicles=10,
                       num_rounds=2, steps_per_round=6, aggregation_strategy="fedavg")
        self.assertEqual(len(rec["training_rounds"]), 2)
        for r in rec["training_rounds"]:
            self.assertLessEqual(r["uploaded_updates"], 5 + 1)

    def test_communication_volume_is_measured_not_constant(self):
        rec = self.trial(seed=3, config=self.cfg, eval_config=self.ev, num_vehicles=10,
                       num_rounds=2, steps_per_round=6, aggregation_strategy="fedavg")
        up_down = sum(r["model_bytes"] * (r["uploaded_updates"] + 10)
                      for r in rec["training_rounds"] if r["status"] != "not_attempted_insufficient_updates")
        self.assertAlmostEqual(rec["comm_overhead_kb"], up_down / 1024.0, delta=0.2)
        self.assertNotEqual(rec["comm_overhead_kb"], 600.0)

    def test_round_status_is_one_of_the_honest_values(self):
        rec = self.trial(seed=3, config=self.cfg, eval_config=self.ev, num_vehicles=10,
                       num_rounds=2, steps_per_round=6, aggregation_strategy="trust_aware")
        allowed = {"model_updated", "model_unchanged", "no_model_produced",
                   "not_attempted_insufficient_updates"}
        for r in rec["training_rounds"]:
            self.assertIn(r["status"], allowed)
            self.assertNotEqual(r["status"], "success")

    def test_completed_latency_excludes_failures(self):
        rec = self.trial(seed=3, config=self.cfg, eval_config=self.ev, num_vehicles=10,
                       num_rounds=2, steps_per_round=6, aggregation_strategy="fedavg")
        ev = rec["evaluation_records"][0]
        self.assertEqual(len(ev["completed_latencies"]), sum(ev["successes"]))


class TestAlternativeDetectors(unittest.TestCase):
    def _round(self, seed=0, n=20, d=400, flipped=(0, 1, 2, 3)):
        rng = np.random.RandomState(seed)
        base = rng.randn(d)
        ups = {}
        for i in range(n):
            u = 0.6 * base + 0.8 * rng.randn(d)
            if i in flipped:
                u = -u
            ups[i] = u
        return ups

    def test_krum_and_cosine_find_sign_flippers(self):
        ups = self._round()
        mal = {0, 1, 2, 3}
        for name in ("krum", "cosine_median"):
            ex = alt_detectors.ALTERNATIVES[name](ups)
            s = alt_detectors.score(ex, ups.keys(), mal)
            self.assertGreaterEqual(s["tp"], 3, name)

    def test_no_detection_excludes_nobody(self):
        self.assertEqual(alt_detectors.no_detection(self._round()), set())

    def test_score_counts_add_up(self):
        s = alt_detectors.score({0, 5}, range(10), {0, 1})
        self.assertEqual(s, {"tp": 1, "fp": 1, "fn": 1, "tn": 7})


class TestConsoleConfigValidation(unittest.TestCase):
    def setUp(self):
        from fastapi import HTTPException
        from web.app import SimulationState
        self.HTTPException = HTTPException
        self.state = SimulationState()

    def tearDown(self):
        self.state.close(wait=False)

    def test_attack_ratio_range(self):
        for bad in (-0.1, 0.9, "abc", float("nan")):
            with self.assertRaises(self.HTTPException):
                self.state.set_config({"attack_ratio": bad})

    def test_vehicle_count_range(self):
        for bad in (2, 500):
            with self.assertRaises(self.HTTPException):
                self.state.set_config({"num_vehicles": bad})

    def test_alpha_and_tau_range(self):
        with self.assertRaises(self.HTTPException):
            self.state.set_config({"alpha": 1.5})
        with self.assertRaises(self.HTTPException):
            self.state.set_config({"exclusion_threshold": 0.0})

    def test_rsu_positions_rejected_for_trace_mode(self):
        with self.assertRaises(self.HTTPException):
            self.state.set_config({"mobility_mode": "trace", "rsu_x_positions": [100, 200, 300]})

    def test_response_distinguishes_live_and_new_session_parameters(self):
        out = self.state.set_config({"alpha": 0.3})
        self.assertIn("alpha", out["applied_live"])
        self.assertFalse(out["new_session_started"])
        out = self.state.set_config({"num_vehicles": 12})
        self.assertTrue(out["new_session_started"])

    def test_snapshot_exposes_scene_and_connectivity(self):
        snap = self.state.get_state_snapshot()
        self.assertIn("scene", snap)
        self.assertIn("connected_rsus", snap["vehicles"][0])


if __name__ == "__main__":
    unittest.main()
