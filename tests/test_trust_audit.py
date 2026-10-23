"""
Tests for the trust-metric audit tool.

The audit recomputes every reported detection metric from the raw TP/FP/FN/TN
counts.  These tests check that:

* the tool agrees with the published results (no disagreement);
* an *undefined* metric (zero denominator) is reported as ``None``, while a
  *measured zero* is reported as ``0.0``;
* the confusion counts of the active run sum to the expected number of
  decisions.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.trust_audit import (
    audit_run, rates_from_counts, check_against_record, DENOMINATORS,
)
from web.app import _active_run_entry

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")


def _active_results_file():
    entry = _active_run_entry(RESULTS)
    if entry is None:
        raise unittest.SkipTest("no active, validated run in results/manifest.json")
    return os.path.join(RESULTS, entry["run_id"], "benchmark_results.json")


class TestRateArithmetic(unittest.TestCase):

    def test_undefined_metrics_are_none(self):
        r = rates_from_counts(0, 0, 0, 10)
        self.assertIsNone(r["tpr"])
        self.assertIsNone(r["fnr"])
        self.assertIsNone(r["precision"])
        self.assertIsNone(r["recall"])
        # A defined metric with a zero numerator is a measured zero.
        self.assertEqual(r["fpr"], 0.0)
        self.assertEqual(r["accuracy"], 1.0)

    def test_no_honest_clients_gives_no_fpr(self):
        r = rates_from_counts(5, 0, 0, 0)
        self.assertIsNone(r["fpr"])
        self.assertEqual(r["tpr"], 1.0)

    def test_denominators_are_documented(self):
        r = rates_from_counts(3, 1, 1, 5)
        self.assertEqual(r["n_decisions"], 10)
        self.assertEqual(r["n_malicious"], 4)
        self.assertEqual(r["n_honest"], 6)
        self.assertAlmostEqual(r["tpr"], 0.75)
        self.assertAlmostEqual(r["fpr"], 1 / 6.0)
        self.assertAlmostEqual(r["fnr"], 0.25)
        self.assertAlmostEqual(r["precision"], 0.75)
        self.assertAlmostEqual(r["accuracy"], 0.8)
        self.assertIn("tpr", DENOMINATORS)
        self.assertIn("FP + TN", DENOMINATORS["fpr"])


class TestAuditOfTheActiveRun(unittest.TestCase):

    def test_no_disagreement_with_the_published_counts(self):
        report = audit_run(_active_results_file())
        self.assertEqual(report["total_problems"], 0,
                         "published detection metrics disagree with the raw counts")
        self.assertGreater(len(report["rows"]), 0)

    def test_every_record_is_audited(self):
        report = audit_run(_active_results_file())
        sections = {r["section"] for r in report["rows"]}
        self.assertIn("experiment_a_baseline_comparison", sections)
        seeds = {str(r["seed"]) for r in report["rows"]}
        self.assertIn("pooled", seeds)
        self.assertTrue(any(s.isdigit() for s in seeds),
                        "the audit table must be per seed as well as per scenario")

    def test_check_against_record_detects_a_tampered_value(self):
        record = {
            "confusion_counts": {"tp": 10, "fp": 2, "fn": 0, "tn": 8},
            "attack_detection_rate": 0.9,       # wrong on purpose
            "false_positive_rate": 0.1,
            "false_negative_rate": 0.0,
            "detection_precision": 0.8333333333,
        }
        result = check_against_record(record)
        self.assertTrue(result["problems"])
        self.assertTrue(any("attack_detection_rate" in p for p in result["problems"]))


if __name__ == "__main__":
    unittest.main()
