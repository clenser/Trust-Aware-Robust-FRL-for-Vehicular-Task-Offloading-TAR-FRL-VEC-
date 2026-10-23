"""
Tests for the corrected experimental methodology.

These verify the properties the protocol must have, independently of any
particular result:

* the evaluation phase does not modify the learned parameters;
* every method is evaluated on the same episodes and task workloads;
* a metric with no observation is ``None`` (N/A), never a zero;
* the detection metrics are pooled from the underlying confusion counts;
* the paired comparison statistics are computed from the per-seed records;
* the effective configuration, the dataset checksums and the raw per-seed
  records are stored in the result record.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from experiments.experiment_runner import (
    EvaluationConfig, run_simulation_trial, run_heuristic_baseline,
    run_multiseed_experiment, compare_methods, summarize_metric,
    paired_comparison,
)
from experiments.experiment_runner import _pooled_from_counts


class TestEvaluationProtocol(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eval_config = EvaluationConfig(episodes=1, steps_per_episode=2)
        cls.learned = run_simulation_trial(
            seed=4242, attack_ratio=0.24, mobility_mode="highway",
            eval_config=cls.eval_config)
        cls.baseline = run_heuristic_baseline(
            "Local_Only", seed=4242, attack_ratio=0.24, mobility_mode="highway",
            eval_config=cls.eval_config)

    def test_evaluation_is_inert(self):
        self.assertTrue(self.learned["evaluation_was_inert"],
                        "the evaluation phase modified the learned parameters")

    def test_evaluation_records_the_episode_seeds(self):
        seeds = [r["episode_seed"] for r in self.learned["evaluation_records"]]
        self.assertEqual(seeds, [4242 + 500000])

    def test_evaluation_seeds_are_disjoint_from_training(self):
        train_seed = self.learned["seed"]
        for r in self.learned["evaluation_records"]:
            self.assertNotEqual(r["episode_seed"], train_seed)

    def test_learned_and_baseline_share_the_evaluation_protocol(self):
        """The paired comparison is only valid on identical episodes."""
        self.assertEqual(
            [r["episode_seed"] for r in self.learned["evaluation_records"]],
            [r["episode_seed"] for r in self.baseline["evaluation_records"]])
        self.assertEqual(self.learned["effective_config"]["evaluation"],
                         self.baseline["effective_config"]["evaluation"])

    def test_baseline_receives_the_same_mobility_mode(self):
        self.assertEqual(self.learned["effective_config"]["mobility_mode"],
                         self.baseline["effective_config"]["mobility_mode"])
        self.assertEqual(self.learned["datasets"]["mobility_mode"],
                         self.baseline["datasets"]["mobility_mode"])

    def test_metrics_are_present_when_observations_exist(self):
        self.assertGreater(self.learned["n_observations"], 0)
        self.assertIsNotNone(self.learned["avg_latency_ms"])
        self.assertIsNotNone(self.learned["completion_rate"])

    def test_effective_configuration_is_recorded(self):
        cfg = self.learned["effective_config"]
        for key in ("num_vehicles", "attack_ratio", "realised_malicious_clients",
                    "num_rounds", "steps_per_round", "local_epochs",
                    "local_gradient_steps_per_round", "participation_fraction",
                    "total_gradient_steps_per_client", "aggregation_strategy",
                    "alpha", "exclusion_tau", "mobility_mode", "seed",
                    "evaluation"):
            self.assertIn(key, cfg, "the effective configuration lacks '%s'" % key)
        # The realised malicious count must match the configured ratio.
        self.assertEqual(cfg["realised_malicious_clients"],
                         int(round(cfg["num_vehicles"] * cfg["attack_ratio"])))

    def test_dataset_checksums_are_recorded(self):
        for record in (self.learned, self.baseline):
            datasets = record["datasets"]
            self.assertIn("mobility_mode", datasets)
            self.assertIn("mobility_source", datasets)
            if datasets["mobility_mode"] != "highway":
                self.assertEqual(len(datasets["trace_sha256"]), 64)
            # The task-profile checksum is always recorded.
            self.assertEqual(len(record["task_profiles"]["task_profile_sha256"]), 64)

    def test_raw_per_seed_records_are_retained(self):
        self.assertTrue(self.learned["evaluation_records"])
        for record in self.learned["evaluation_records"]:
            self.assertIn("latencies", record)
            self.assertIn("successes", record)
            self.assertIn("energies", record)
        self.assertTrue(self.learned["training_rounds"] is not None)


class TestMetricValidity(unittest.TestCase):

    def test_missing_observations_are_none_not_zero(self):
        summary = summarize_metric([None, None, None])
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["std"])
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["ci95"])

    def test_summary_reports_the_sample_size(self):
        summary = summarize_metric([1.0, 2.0, 3.0, None])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 2.0, places=9)
        self.assertIsNotNone(summary["ci95"])

    def test_pooled_rates_come_from_the_counts(self):
        rates = _pooled_from_counts({"tp": 3, "fp": 1, "fn": 1, "tn": 15})
        self.assertAlmostEqual(rates["tpr"], 0.75, places=9)
        self.assertAlmostEqual(rates["fpr"], 0.0625, places=9)
        self.assertAlmostEqual(rates["fnr"], 0.25, places=9)
        self.assertAlmostEqual(rates["precision"], 0.75, places=9)

    def test_undefined_rates_are_none(self):
        rates = _pooled_from_counts({"tp": 0, "fp": 0, "fn": 0, "tn": 10})
        self.assertIsNone(rates["tpr"])
        self.assertIsNone(rates["fnr"])
        self.assertIsNone(rates["precision"])
        self.assertAlmostEqual(rates["fpr"], 0.0, places=9)

    def test_detection_metrics_are_not_reported_for_non_detectors(self):
        """A strategy without client detection must report N/A, not zero."""
        rec = run_multiseed_experiment([101], method="trust_no_exclusion")
        self.assertIsNone(rec["attack_detection_rate"])
        self.assertIsNone(rec["false_positive_rate"])
        self.assertIsNone(rec["false_negative_rate"])


class TestPairedStatistics(unittest.TestCase):

    def test_paired_difference_is_computed_from_the_paired_samples(self):
        a = [10.0, 12.0, 11.0, 13.0]
        b = [12.0, 13.0, 14.0, 15.0]
        result = paired_comparison(a, b)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mean_difference"], -2.0, places=9)
        self.assertIsNotNone(result["ci95"])
        self.assertIsNotNone(result["cohens_d"])
        self.assertLessEqual(result["ci95_low"], result["mean_difference"])
        self.assertGreaterEqual(result["ci95_high"], result["mean_difference"])

    def test_zero_variance_difference_has_no_effect_size(self):
        """A constant paired difference has an undefined standardised effect size."""
        result = paired_comparison([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["mean_difference"], -1.0, places=9)
        self.assertIsNone(result["cohens_d"])
        self.assertIsNone(result["p_value"])

    def test_paired_comparison_ignores_missing_samples(self):
        result = paired_comparison([1.0, None, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(result["n"], 2)

    def test_paired_comparison_without_data(self):
        result = paired_comparison([None], [None])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["mean_difference"])

    def test_compare_methods_aligns_by_seed(self):
        learned = run_multiseed_experiment([11, 22], method="trust_aware",
                                          mobility_mode="highway")
        baseline = run_multiseed_experiment([11, 22], baseline="Local_Only",
                                            mobility_mode="highway")
        comparison = compare_methods({"Proposed_TFRL": learned, "Local_Only": baseline})
        self.assertIn("Local_Only", comparison)
        self.assertEqual(comparison["Local_Only"]["latency_ms"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
