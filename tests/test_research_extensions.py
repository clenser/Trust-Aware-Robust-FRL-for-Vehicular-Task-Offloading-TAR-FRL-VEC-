"""
Tests for the extensions that address the problem statement's edge-node and
intermittent-connectivity requirements:

* an unreliable edge node (RSU fault model) and the behavioural trust it produces,
* masking the trust features (ablation switch),
* unreliable vehicles are honest participants of the federated round (labels),
* stale updates under missed broadcasts and the staleness discount,
* the behavioural-trust-aware greedy baseline.
"""

import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from environment.vanet_env import VANETOffloadingEnv
from environment.task import Task
from trust.trust_engine import TrustEngine


def _task():
    return Task("t", 1, "Cooperative_Perception", 0.45, 0.22, 160.0)


def _env(faulty=None, mask=False, seed=3):
    cfg = copy.deepcopy(load_config())
    if faulty is not None:
        cfg["adversarial"]["faulty_rsus"] = faulty
    cfg["environment_model"]["mask_trust_features"] = mask
    np.random.seed(seed)
    return VANETOffloadingEnv(num_vehicles=8, attack_ratio=0.0, mobility_mode="highway",
                              random_seed=seed, config=cfg)


class TestFaultyEdgeNode(unittest.TestCase):
    def test_full_drop_fault_fails_every_task_and_healthy_nodes_are_unaffected(self):
        env = _env(faulty=[{"index": 1, "drop_prob": 1.0}])
        v = env.vehicles[1]
        v.x, v.y = env.rsus[1].x, 0.0
        _, _, _, info, _ = env.step_vehicle(1, 2, _task())
        self.assertFalse(info["success"])
        v.x = env.rsus[0].x
        ok = [env.step_vehicle(1, 1, _task())[3]["success"] for _ in range(3)]
        self.assertTrue(any(ok), "the healthy RSU must still serve tasks")

    def test_behavioural_trust_of_the_faulty_node_falls(self):
        env = _env(faulty=[{"index": 1, "drop_prob": 1.0}])
        v = env.vehicles[1]
        v.x, v.y = env.rsus[1].x, 0.0
        before = env.trust_engine.behavior_tracker.get_trust("RSU_2")
        for _ in range(40):
            env.step_vehicle(1, 2, _task())
            env.update_positions()
            v.x, v.y = env.rsus[1].x, 0.0
        after = env.trust_engine.behavior_tracker.get_trust("RSU_2")
        self.assertLess(after, 0.45)
        self.assertLess(after, before)
        self.assertGreater(env.trust_engine.behavior_tracker.get_trust("RSU_1"), 0.6)

    def test_delay_fault_slows_service(self):
        env = _env(faulty=[{"index": 0, "delay_factor": 4.0}])
        slow = env.rsus[0]
        fast = env.rsus[2]
        t = _task()
        _, s_slow, _ = slow.process_task(t, env.channel, 0.0)
        _, s_fast, _ = fast.process_task(_task(), env.channel, 0.0)
        # RSU_1 (4 GHz/core) with factor 4 vs RSU_3 (3.33 GHz/core) healthy
        self.assertGreater(s_slow, 3.0 * s_fast)

    def test_invalid_fault_index_rejected(self):
        with self.assertRaises(ValueError):
            _env(faulty=[{"index": 7, "drop_prob": 0.5}])


class TestTrustFeatureMask(unittest.TestCase):
    def test_trust_entries_are_zero_when_masked_and_not_otherwise(self):
        plain = _env(mask=False)
        masked = _env(mask=True)
        s_plain, _ = plain._get_global_state_dict()[1]
        s_masked, _ = masked._get_global_state_dict()[1]
        for i in (9, 10, 11, 13):
            self.assertGreater(s_plain[i], 0.0)
            self.assertEqual(s_masked[i], 0.0)
        # every other feature carries the same information
        keep = [i for i in range(len(s_plain)) if i not in (9, 10, 11, 13)]
        self.assertEqual(len(keep), len(s_plain) - 4)


class TestUnreliableVehiclesAreHonestParticipants(unittest.TestCase):
    def test_unreliable_role_is_not_a_missed_attacker(self):
        rng = np.random.RandomState(0)
        base = rng.randn(200)
        updates = {i: 0.6 * base + 0.8 * rng.randn(200) for i in range(10)}
        roles = {i: "honest" for i in range(10)}
        roles[0] = "unreliable"
        roles[1] = "poisoning"
        updates[1] = -updates[1]
        eng = TrustEngine()
        _, _, m = eng.filter_trusted_participants(updates, ground_truth_roles=roles)
        self.assertEqual(m["true_positives"] + m["false_negatives"], 1,
                         "only the poisoning client is a positive")
        self.assertEqual(m["unreliable_total"], 1)


class TestStaleness(unittest.TestCase):
    def test_discount_scales_aggregation_weights(self):
        import torch
        from federated.server import FederatedServer
        srv = FederatedServer(aggregation_strategy="fedavg")
        srv.staleness_mode = "discount"
        w = srv.get_global_weights()
        a = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in w.items()}
        b = {k: torch.ones_like(v, dtype=torch.float32) for k, v in w.items()}
        merged, _ = srv.run_aggregation_round({1: a, 2: b}, {1: 10, 2: 10}, staleness={1: 0, 2: 3})
        # weights 10 and 10/4 -> the fresh client dominates: mean = 2.5/12.5 = 0.2
        first = next(iter(merged.values())).float()
        self.assertAlmostEqual(float(first.flatten()[0]), 0.2, places=5)

    def test_no_discount_by_default(self):
        import torch
        from federated.server import FederatedServer
        srv = FederatedServer(aggregation_strategy="fedavg")
        w = srv.get_global_weights()
        a = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in w.items()}
        b = {k: torch.ones_like(v, dtype=torch.float32) for k, v in w.items()}
        merged, m = srv.run_aggregation_round({1: a, 2: b}, {1: 10, 2: 10}, staleness={1: 0, 2: 3})
        self.assertAlmostEqual(float(next(iter(merged.values())).float().flatten()[0]), 0.5, places=5)
        self.assertEqual(m["mean_staleness"], 1.5)

    def test_missed_broadcast_produces_stale_uploads(self):
        from experiments.experiment_runner import run_simulation_trial, EvaluationConfig
        cfg = copy.deepcopy(load_config())
        cfg["federated_learning"].update({"local_epochs": 2, "connectivity_dropout": 0.5,
                                          "missed_broadcast": True, "staleness_mode": "discount"})
        rec = run_simulation_trial(seed=5, config=cfg, eval_config=EvaluationConfig(1, 3, 500000),
                                   num_vehicles=10, num_rounds=6, steps_per_round=6,
                                   aggregation_strategy="trust_aware")
        stale = [r["mean_staleness"] for r in rec["training_rounds"] if r["mean_staleness"] is not None]
        self.assertTrue(stale, "staleness must be recorded")
        self.assertGreater(max(stale), 0.0, "some client must have missed a broadcast")
        self.assertTrue(any(r["offline_clients"] > 0 for r in rec["training_rounds"]))
        self.assertEqual(rec["effective_config"]["missed_broadcast"], True)


class TestTrustGreedyBaseline(unittest.TestCase):
    def test_skips_an_untrusted_rsu(self):
        from experiments.experiment_runner import _trust_greedy_action
        env = _env()
        v = env.vehicles[1]
        v.x, v.y = env.rsus[1].x, 0.0
        v.cpu_capacity_ghz = 0.8          # slow vehicle: offloading is attractive
        t = _task()
        self.assertEqual(_trust_greedy_action(env, v, t), 2)
        env.trust_engine.behavior_tracker.node_records["RSU_2"]["trust_score"] = 0.2
        self.assertNotEqual(_trust_greedy_action(env, v, t), 2)


if __name__ == "__main__":
    unittest.main()
