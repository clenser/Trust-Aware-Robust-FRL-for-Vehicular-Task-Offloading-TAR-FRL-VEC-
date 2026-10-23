"""Do-no-harm detector, vectorised features and the v2.2 delay-estimate state."""

import copy
import unittest

import numpy as np

from configs.config_loader import load_config
from environment.vanet_env import VANETOffloadingEnv
from federated.server import FederatedServer
from trust.dnh_detector import DoNoHarmDetector, loo_features


def _population(n=20, d=2000, n_bad=4, scale=2.5, seed=0, flip=True):
    rng = np.random.RandomState(seed)
    common = rng.randn(d)
    honest = [common * 0.5 + rng.randn(d) for _ in range(n - n_bad)]
    bad = [(-scale if flip else scale) * (common * 0.5 + rng.randn(d)) for _ in range(n_bad)]
    ups = {i: u for i, u in enumerate(honest + bad)}
    return ups, set(range(n - n_bad, n))


class TestFeatures(unittest.TestCase):
    def test_vectorised_loo_matches_a_loop(self):
        rng = np.random.RandomState(1)
        M = rng.randn(12, 500)
        ratio, cos, norms, live = loo_features(M)
        k = max(1, int(0.25 * 11))
        for i in range(12):
            others = np.delete(M, i, axis=0)
            s = np.sort(others, axis=0)
            ref = s[k:len(others) - k].mean(axis=0)
            expect = M[i] @ ref / (np.linalg.norm(M[i]) * np.linalg.norm(ref))
            self.assertAlmostEqual(cos[i], expect, places=9)

    def test_zero_update_is_not_live(self):
        M = np.random.RandomState(2).randn(8, 100)
        M[3] = 0.0
        ratio, cos, norms, live = loo_features(M)
        self.assertFalse(live[3])
        self.assertEqual(ratio[3], 0.0)


class TestDetector(unittest.TestCase):
    def test_flags_sign_flippers_and_spares_honest_clients(self):
        ups, bad = _population()
        det = DoNoHarmDetector()
        for _ in range(3):                      # evidence accumulates over rounds
            excluded, _ = det.decide(ups)
        self.assertEqual(excluded, bad)

    def test_all_honest_population_is_left_alone(self):
        ups, _ = _population(n_bad=0)
        det = DoNoHarmDetector()
        for _ in range(5):
            excluded, _ = det.decide(ups)
        self.assertEqual(excluded, set())

    def test_free_rider_is_excluded_by_the_absolute_norm_test(self):
        ups, _ = _population(n_bad=0)
        ups[0] = np.zeros_like(ups[0])
        excluded, ev = DoNoHarmDetector().decide(ups)
        self.assertIn(0, excluded)

    def test_exclusion_is_capped(self):
        ups, _ = _population(n=20, n_bad=14)    # more attackers than the honest-majority bound
        excluded, _ = DoNoHarmDetector(max_exclude_fraction=0.45).decide(ups)
        self.assertLessEqual(len(excluded), 9)

    def test_too_few_participants_excludes_nobody(self):
        ups, _ = _population(n=3, n_bad=1)
        self.assertEqual(DoNoHarmDetector().decide(ups)[0], set())


class TestServerStrategy(unittest.TestCase):
    def test_do_no_harm_round_runs_and_reports_confusion(self):
        srv = FederatedServer(state_dim=20, action_dim=5, aggregation_strategy="do_no_harm")
        g = srv.get_global_weights()
        import torch
        rng = np.random.RandomState(0)
        models, samples, roles = {}, {}, {}
        for cid in range(8):
            w = {k: v.clone() for k, v in g.items()}
            for k in w:
                w[k] = w[k] + torch.tensor(rng.randn(*w[k].shape) * 0.01, dtype=w[k].dtype)
            models[cid], samples[cid], roles[cid] = w, 10, "honest"
        new, metrics = srv.run_aggregation_round(models, samples, ground_truth_roles=roles)
        self.assertEqual(metrics["defense_type"], "do_no_harm_exclusion")
        self.assertTrue(metrics["aggregation_produced_model"])
        self.assertEqual(metrics["false_positives"], 0)


class TestStateV22(unittest.TestCase):
    def _env(self, version):
        cfg = copy.deepcopy(load_config())
        cfg["environment_model"]["state_features"] = version
        np.random.seed(3)
        return VANETOffloadingEnv(config=cfg, num_vehicles=10, attack_ratio=0.2, random_seed=3)

    def test_dimension_and_range(self):
        env = self._env("v2.2")
        self.assertEqual(env.state_dim, 25)
        sd = env._get_global_state_dict()
        for s, task in sd.values():
            self.assertEqual(len(s), 25)
            self.assertTrue(np.all(s[20:] >= 0.0) and np.all(s[20:] <= 1.0))

    def test_v22_prefix_equals_v21(self):
        a, b = self._env("v2.1"), self._env("v2.2")
        sa, sb = a._get_global_state_dict(), b._get_global_state_dict()
        for vid in sa:
            np.testing.assert_allclose(sa[vid][0], sb[vid][0][:20], atol=1e-6)

    def test_out_of_coverage_rsu_is_estimated_at_the_cap(self):
        env = self._env("v2.2")
        v = env.vehicles[1]
        task = env.task_generator.generate_task(1, 0.0)
        far = env.rsus[0]
        v.x = far.x + far.coverage_radius_m * 3
        est = env._delay_estimates(v, task, None)
        self.assertEqual(est[1], 1.0)           # clipped, scaled
        self.assertEqual(est[4], 1.0)           # no peer


if __name__ == "__main__":
    unittest.main()
