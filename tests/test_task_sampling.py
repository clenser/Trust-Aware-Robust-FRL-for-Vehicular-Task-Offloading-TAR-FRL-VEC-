"""The task generator's performance paths must not change what benchmarks see."""

import os
import sys
import unittest

import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.task import TaskGenerator
from environment.vanet_env import VANETOffloadingEnv
from configs.config_loader import load_config


class TestTaskGeneratorPaths(unittest.TestCase):
    def test_cached_task_mix_equals_the_original_formula(self):
        g = TaskGenerator(random_seed=5)
        for vid in (1, 2, 17, 40):
            ref = np.random.RandomState(int(vid) * 31 + 7).dirichlet(np.ones(len(g.types)) * 0.4)
            self.assertTrue(np.array_equal(ref, g.get_client_task_distribution(vid)))
            self.assertTrue(np.array_equal(ref, g.get_client_task_distribution(vid)))   # cached call

    def test_default_stream_is_deterministic(self):
        a, b = TaskGenerator(random_seed=9), TaskGenerator(random_seed=9)
        ta = [(t.task_type, t.data_size_mb, t.cpu_cycles_giga) for t in (a.generate_task(1) for _ in range(25))]
        tb = [(t.task_type, t.data_size_mb, t.cpu_cycles_giga) for t in (b.generate_task(1) for _ in range(25))]
        self.assertEqual(ta, tb)

    def test_fast_sampler_has_the_same_distribution_and_respects_bounds(self):
        f = TaskGenerator(random_seed=100, fast_sampling=True)
        s = TaskGenerator(random_seed=200)
        fx = [f.generate_task(1) for _ in range(2000)]
        sx = [s.generate_task(1) for _ in range(2000)]
        for attr in ("data_size_mb", "cpu_cycles_giga"):
            p = stats.ks_2samp([getattr(t, attr) for t in fx], [getattr(t, attr) for t in sx]).pvalue
            self.assertGreater(p, 0.01, attr)
        self.assertTrue(all(t.data_size_mb >= 0.01 and t.cpu_cycles_giga >= 0.01 for t in fx))

    def test_benchmarks_never_use_the_fast_sampler_by_default(self):
        cfg = load_config()
        self.assertFalse(cfg["task_profiles"].get("fast_sampling", False))
        env = VANETOffloadingEnv(num_vehicles=5, mobility_mode="highway", random_seed=1)
        self.assertFalse(env.task_generator.fast_sampling)


if __name__ == "__main__":
    unittest.main()
