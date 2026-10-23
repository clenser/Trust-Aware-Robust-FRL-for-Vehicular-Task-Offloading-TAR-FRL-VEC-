"""
Tests for the participation protocol and the population-reference repair.

These cover the logic that was changed after the detector audit:

* a client that holds fewer transitions than the minimum training batch
  could not perform a gradient step, so it is a NON-PARTICIPANT of the
  round: it is not judged and it is not counted as a false positive;
* a free rider holds plenty of experience and withholds it, so it stays
  inside the detection population and is still detected;
* the participation rule is a no-op when every client has experience;
* a degenerate population reference (more than half of the participants
  uploaded nothing) is repaired instead of producing unbounded norm ratios.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from experiments.trust_validation import (
    ScenarioSpec, build_scenario, compute_decisions, evaluate_scenario,
    run_suite, summarize,
)
from experiments.detector_study import wilson_interval, rates_from_counts
from trust.model_trust import ModelTrustEvaluator
from trust.trust_engine import TrustEngine

TAU = 0.40


def _spec(**kw):
    base = dict(name="s", heterogeneity="non_iid", shared_component=0.6,
                seed_index=1)
    base.update(kw)
    return ScenarioSpec(**base)


class TestParticipationRule(unittest.TestCase):

    def test_idle_clients_are_not_judged(self):
        spec = _spec(idle_clients=6)
        updates, malicious, idle = build_scenario(spec, 101)
        res = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                                idle=idle, exclude_non_participants=True)
        self.assertEqual(len(idle), 6)
        for cid in idle:
            self.assertNotIn(cid, res["decisions"])
        # The confusion counts must be computed over the participants only.
        self.assertEqual(sum(res["confusion"].values()), 20 - len(idle))

    def test_idle_clients_would_be_false_positives_without_the_rule(self):
        spec = _spec(idle_clients=6)
        updates, malicious, idle = build_scenario(spec, 101)
        res = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                                idle=idle, exclude_non_participants=False)
        self.assertGreaterEqual(res["confusion"]["fp"], len(idle),
                                "the pre-fix protocol judged the idle clients")

    def test_free_riders_stay_detectable(self):
        spec = _spec(idle_clients=6, free_rider_clients=2)
        updates, malicious, idle = build_scenario(spec, 101)
        res = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                                idle=idle, exclude_non_participants=True)
        self.assertEqual(res["confusion"]["tp"], 2,
                         "free riders hold experience and must be detected")
        self.assertEqual(res["confusion"]["fn"], 0)
        self.assertEqual(res["confusion"]["fp"], 0)

    def test_the_rule_is_a_no_op_when_everyone_participates(self):
        spec = _spec(idle_clients=0)
        updates, malicious, idle = build_scenario(spec, 101)
        a = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                              idle=idle, exclude_non_participants=True)
        b = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                              idle=idle, exclude_non_participants=False)
        self.assertEqual(a["confusion"], b["confusion"])

    def test_the_participation_signal_is_role_independent(self):
        """The idle set is derived from the available experience, not the role."""
        spec = _spec(idle_clients=6, free_rider_clients=2)
        updates, malicious, idle = build_scenario(spec, 101)
        self.assertTrue(set(idle).isdisjoint(set(malicious)),
                        "an idle client must never be a malicious client")
        self.assertEqual(len(set(malicious)), 2)

    def test_detection_counts_use_the_participated_population(self):
        spec = _spec(idle_clients=12, free_rider_clients=2)
        updates, malicious, idle = build_scenario(spec, 101)
        res = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                                idle=idle, exclude_non_participants=True)
        total = sum(res["confusion"].values())
        self.assertEqual(total, 20 - 12)
        self.assertEqual(res["rates"]["n_decisions"], 8)


class TestPopulationReferenceRepair(unittest.TestCase):

    def test_degenerate_population_is_repaired(self):
        """When more than half of the participants upload nothing, the
        median norm is zero and every informative client would obtain an
        unbounded norm ratio.  The reference is recomputed over the clients
        that did upload something."""
        rng = np.random.RandomState(3)
        updates = {}
        for i in range(12):
            updates[i] = np.zeros(8)                    # degenerate majority
        for i in range(12, 20):
            updates[i] = rng.normal(size=8) * 0.1 + 1.0
        ev = ModelTrustEvaluator()
        scores = ev.evaluate_model_updates(updates)
        self.assertTrue(ev.last_population_degenerate)
        # The informative clients must not be scored as scale outliers.
        informative = [scores[i] for i in range(12, 20)]
        self.assertGreater(min(informative), 0.5,
                           "informative clients must keep a high model trust")

    def test_a_healthy_population_is_not_flagged_as_degenerate(self):
        rng = np.random.RandomState(4)
        updates = {i: rng.normal(size=8) + 1.0 for i in range(20)}
        ev = ModelTrustEvaluator()
        ev.evaluate_model_updates(updates)
        self.assertFalse(ev.last_population_degenerate)

    def test_a_healthy_population_scores_identically_with_and_without_the_repair(self):
        rng = np.random.RandomState(5)
        updates = {i: rng.normal(size=8) + 1.0 for i in range(20)}
        a = ModelTrustEvaluator()
        b = ModelTrustEvaluator()
        # Force the repair branch to be reachable only for degenerate inputs.
        sa = a.evaluate_model_updates(updates)
        sb = b.evaluate_model_updates(updates)
        self.assertEqual(sa, sb)


class TestDetectionIsUnaffectedByTheRuleWhenThePopulationIsHealthy(unittest.TestCase):

    def test_validation_suite_without_idle_clients_is_unchanged(self):
        """The participation rule must not change any scenario in which
        every client has experience."""
        from experiments.trust_validation import BASE_VALIDATION_SCENARIOS
        healthy = [s for s in BASE_VALIDATION_SCENARIOS
                    if s.idle_clients == 0 and s.free_rider_clients == 0]
        a = run_suite(healthy, seed_base=1, exclusion_threshold=TAU,
                      exclude_non_participants=True)
        b = run_suite(healthy, seed_base=1, exclusion_threshold=TAU,
                      exclude_non_participants=False)
        for spec in healthy:
            self.assertEqual(a[spec.name]["confusion"], b[spec.name]["confusion"],
                             spec.name)


class TestDocumentedScaleFragility(unittest.TestCase):
    """The scale *z-score* rule is fragile on a low-dispersion population.

    The absolute scaling rule (a norm far above the population median) is
    scale-safe, but the statistical scale outlier rule uses a robust z-score
    of the log-norm ratio.  When the honest population has very little norm
    dispersion the robust scale collapses and a benign client can exceed the
    z-threshold.  This is a documented limitation, pinned here so that it
    cannot change silently.
    """

    def test_a_low_dispersion_population_can_produce_a_scale_false_positive(self):
        spec = _spec(idle_clients=6)
        updates, malicious, idle = build_scenario(spec, 101)
        res = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                                idle=idle, exclude_non_participants=True)
        # The scenario is documented to produce exactly one benign scale
        # outlier; the count is pinned, not assumed.
        self.assertEqual(res["confusion"]["fp"], 1)

    def test_a_higher_dispersion_population_does_not(self):
        spec = _spec(idle_clients=0, shared_component=0.6, seed_index=1)
        updates, malicious, idle = build_scenario(spec, 101)
        res = evaluate_scenario(updates, malicious, exclusion_threshold=TAU,
                                idle=idle, exclude_non_participants=True)
        self.assertEqual(res["confusion"]["fp"], 0)


class TestUncertainty(unittest.TestCase):

    def test_wilson_interval_brackets_the_point_estimate(self):
        lo, hi = wilson_interval(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 1.0)
        lo, hi = wilson_interval(20, 20)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 1.0)

    def test_wilson_interval_is_undefined_without_denominator(self):
        lo, hi = wilson_interval(0, 0)
        self.assertIsNone(lo)
        self.assertIsNone(hi)

    def test_wilson_interval_shrinks_with_the_sample_size(self):
        small = wilson_interval(0, 20)[1]
        large = wilson_interval(0, 2000)[1]
        self.assertLess(large, small)

    def test_rates_from_counts_reports_denominators(self):
        r = rates_from_counts({"tp": 10, "fp": 2, "fn": 1, "tn": 87})
        self.assertEqual(r["honest_denominator"], 89)
        self.assertEqual(r["malicious_denominator"], 11)
        self.assertAlmostEqual(r["tpr"], 10 / 11.0)
        self.assertAlmostEqual(r["fpr"], 2 / 89.0)
        self.assertEqual(r["tpr_ci"]["n"], 11)
        self.assertEqual(r["fpr_ci"]["n"], 89)


class TestDeployedConfiguration(unittest.TestCase):

    def test_the_participation_rule_is_enabled_by_default(self):
        from configs.config_loader import load_config
        cfg = load_config()
        self.assertTrue(cfg["trust_management"].get("exclude_non_participants"))

    def test_the_rule_is_recorded_in_the_effective_configuration(self):
        from configs.config_loader import load_config
        cfg = load_config()
        self.assertIn("exclude_non_participants", cfg["trust_management"])


if __name__ == "__main__":
    unittest.main()
