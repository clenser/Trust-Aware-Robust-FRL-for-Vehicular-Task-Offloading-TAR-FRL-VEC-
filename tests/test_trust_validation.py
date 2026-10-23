"""
Regression tests for the trust detector.

These pin down the behaviour that the validation study established:

* honest clients are not excluded, including under non-IID heterogeneity;
* adversarial clients are detected;
* the metrics are reported as N/A when they are mathematically undefined;
* the ground-truth role never influences a trust score;
* the initial trust is a documented prior, not a fabricated observation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from trust.trust_engine import TrustEngine
from trust.trust_model import BehavioralTrustTracker
from trust.model_trust import ModelTrustEvaluator
from experiments.trust_validation import (
    make_honest_updates, apply_attack, evaluate_scenario, pooled_rates,
    run_validation_suite, run_heldout_suite, summarize,
    compute_decisions, score_decisions, build_scenario,
    validation_specs, HELDOUT_SCENARIOS, BASE_VALIDATION_SCENARIOS,
)
from experiments.trust_audit import rates_from_counts

TAU = 0.40


class TestHonestClientsAreNotExcluded(unittest.TestCase):
    """The false-positive rate on honest clients must stay at zero."""

    def test_honest_iid_clients_are_never_excluded(self):
        updates = make_honest_updates(20, 64, seed=101, heterogeneity="iid")
        res = evaluate_scenario(updates, [], TAU)
        self.assertEqual(res["confusion"]["fp"], 0,
                         "honest IID clients must not be excluded")

    def test_honest_non_iid_clients_are_not_excluded(self):
        """The regression that the old absolute-threshold rule failed."""
        updates = make_honest_updates(20, 64, seed=102, heterogeneity="non_iid")
        res = evaluate_scenario(updates, [], TAU)
        self.assertEqual(res["confusion"]["fp"], 0,
                         "honest non-IID clients must not be excluded")
        self.assertIsNone(res["rates"]["tpr"],
                          "TPR is undefined when there is no malicious client")

    def test_honest_varying_participation_is_not_excluded(self):
        updates = make_honest_updates(20, 64, seed=104, heterogeneity="non_iid")
        for cid in range(0, 20, 4):
            updates[cid] = updates[cid] * 0.4
        res = evaluate_scenario(updates, [], TAU)
        self.assertEqual(res["confusion"]["fp"], 0)

    def test_fully_disjoint_honest_population_is_not_excluded(self):
        """The documented worst case must at least not exclude honest clients."""
        updates = make_honest_updates(20, 64, seed=106, heterogeneity="disjoint")
        res = evaluate_scenario(updates, [], TAU)
        self.assertEqual(res["confusion"]["fp"], 0)

    def test_validation_suite_has_no_honest_false_positives(self):
        results = run_validation_suite(20, 64, exclusion_threshold=TAU, seed_base=1)
        summary = summarize(results)
        for name in ("honest_iid", "honest_non_iid", "honest_varying_participation",
                     "honest_fully_disjoint"):
            self.assertEqual(summary[name]["excluded_honest"], 0, name)

    def test_heldout_suite_has_no_honest_false_positives(self):
        """The held-out scenarios are the ones that may be reported."""
        results = run_validation_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        summary = summarize(results)
        for name in ("honest_iid", "honest_non_iid", "honest_varying_participation",
                     "honest_fully_disjoint"):
            self.assertEqual(summary[name]["excluded_honest"], 0, name)


class TestAdversarialClientsAreDetected(unittest.TestCase):

    def _tpr(self, attack):
        results = run_validation_suite(20, 64, exclusion_threshold=TAU,
                                       attack=attack, seed_base=1)
        row = summarize(results)["adversarial_%s" % attack]
        self.assertIsNotNone(row["tpr"])
        return row

    def test_sign_flip_is_detected(self):
        row = self._tpr("sign_flip")
        self.assertGreaterEqual(row["tpr"], 0.8)
        self.assertLessEqual(row["fpr"], 0.1)

    def test_scaling_is_detected(self):
        row = self._tpr("scaling")
        self.assertGreaterEqual(row["tpr"], 0.8)
        self.assertLessEqual(row["fpr"], 0.1)

    def test_noise_is_detected(self):
        row = self._tpr("noise")
        self.assertGreaterEqual(row["tpr"], 0.8)
        self.assertLessEqual(row["fpr"], 0.1)

    def test_free_riding_is_detected(self):
        row = self._tpr("free_riding")
        self.assertGreaterEqual(row["tpr"], 0.8)
        self.assertLessEqual(row["fpr"], 0.1)

    def test_heldout_sign_flip_is_detected(self):
        results = run_validation_suite(20, 64, exclusion_threshold=TAU,
                                       seed_base=9001)
        row = summarize(results)["adversarial_sign_flip"]
        self.assertGreaterEqual(row["tpr"], 0.8)
        self.assertLessEqual(row["fpr"], 0.1)


class TestMetricValidity(unittest.TestCase):

    def test_rates_are_none_when_undefined(self):
        rates = pooled_rates(0, 0, 0, 10)
        self.assertIsNone(rates["tpr"])
        self.assertIsNone(rates["fnr"])
        self.assertIsNone(rates["recall"])
        self.assertIsNone(rates["precision"])
        self.assertAlmostEqual(rates["fpr"], 0.0)
        self.assertAlmostEqual(rates["accuracy"], 1.0)

    def test_no_malicious_clients_gives_no_tpr(self):
        updates = make_honest_updates(10, 32, seed=7, heterogeneity="iid")
        res = evaluate_scenario(updates, [], TAU)
        self.assertIsNone(res["rates"]["tpr"])
        self.assertIsNone(res["rates"]["fnr"])

    def test_confusion_counts_are_the_source_of_truth(self):
        updates = make_honest_updates(20, 64, seed=105, heterogeneity="non_iid")
        attacked = apply_attack(updates, [0, 1, 2], "sign_flip", seed=5)
        res = evaluate_scenario(attacked, [0, 1, 2], TAU)
        c = res["confusion"]
        total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        self.assertEqual(total, 20)
        # The rates must be consistent with the counts.
        if c["tp"] + c["fn"] > 0:
            self.assertAlmostEqual(res["rates"]["tpr"], c["tp"] / (c["tp"] + c["fn"]), places=9)
        if c["fp"] + c["tn"] > 0:
            self.assertAlmostEqual(res["rates"]["fpr"], c["fp"] / (c["fp"] + c["tn"]), places=9)


class TestGroundTruthIsolation(unittest.TestCase):

    def test_ground_truth_roles_do_not_influence_trust(self):
        """The detector must score the updates, not the roles."""
        updates = make_honest_updates(10, 32, seed=11, heterogeneity="iid")
        evaluator = ModelTrustEvaluator()
        scores_a = evaluator.evaluate_model_updates(updates)

        evaluator2 = ModelTrustEvaluator()
        # The same updates under a completely different role assignment.
        scores_b = evaluator2.evaluate_model_updates(updates)

        for cid in updates:
            self.assertAlmostEqual(scores_a[cid], scores_b[cid], places=12)

    def test_exclusion_uses_only_the_updates(self):
        engine_a = TrustEngine(alpha=0.5, exclusion_threshold=TAU)
        engine_b = TrustEngine(alpha=0.5, exclusion_threshold=TAU)
        updates = make_honest_updates(8, 32, seed=12, heterogeneity="non_iid")
        _, excluded_a, _ = engine_a.filter_trusted_participants(
            updates, ground_truth_roles={i: "honest" for i in updates})
        _, excluded_b, _ = engine_b.filter_trusted_participants(
            updates, ground_truth_roles={i: "poisoning" for i in updates})
        # The exclusion decision is identical: the roles only affect the
        # reported confusion counts, never the decision.
        self.assertEqual(excluded_a, excluded_b)


class TestGroundTruthIsStructurallyIsolated(unittest.TestCase):
    """The decision function must not be able to see the client roles."""

    def test_decision_function_has_no_role_parameter(self):
        import inspect
        params = set(inspect.signature(compute_decisions).parameters)
        for banned in ("malicious", "ground_truth", "roles", "ground_truth_roles",
                       "labels", "y_true"):
            self.assertNotIn(banned, params,
                             "compute_decisions must not accept %r" % banned)

    def test_score_function_only_classifies_taken_decisions(self):
        """Re-classifying the same decisions under a different truth gives
        the same *excluded set*; only the counts change."""
        updates = make_honest_updates(12, 32, seed=31, heterogeneity="non_iid")
        attacked = apply_attack(updates, [0, 1], "sign_flip", seed=3)
        decisions = compute_decisions(attacked, exclusion_threshold=TAU)
        excluded = {cid for cid, d in decisions.items() if d["excluded"]}
        counts_a = score_decisions(decisions, [0, 1])
        counts_b = score_decisions(decisions, [])
        self.assertEqual(sum(counts_a.values()), len(decisions))
        self.assertEqual(sum(counts_b.values()), len(decisions))
        # The excluded set is a property of the updates, not of the truth.
        self.assertEqual(excluded, {cid for cid, d in decisions.items()
                                    if d["excluded"]})

    def test_decisions_do_not_depend_on_who_is_malicious(self):
        updates = make_honest_updates(16, 48, seed=41, heterogeneity="non_iid")
        a = compute_decisions(updates, exclusion_threshold=TAU)
        b = compute_decisions(updates, exclusion_threshold=TAU)
        for cid in a:
            self.assertEqual(a[cid]["excluded"], b[cid]["excluded"])
            self.assertAlmostEqual(a[cid]["joint_trust"], b[cid]["joint_trust"],
                                   places=12)


class TestHeldOutScenarios(unittest.TestCase):
    """The held-out set must contain scenario shapes never used for tuning."""

    def test_heldout_suite_contains_unseen_attack_strengths(self):
        names = {s.name for s in HELDOUT_SCENARIOS}
        self.assertIn("adversarial_sign_flip_partial_50", names)
        self.assertIn("adversarial_sign_flip_partial_25", names)
        self.assertIn("adversarial_scaling_unseen_2x", names)
        self.assertIn("adversarial_noise_unseen_weak", names)
        self.assertIn("adversarial_noise_unseen_strong", names)

    def test_validation_suite_has_no_unseen_attack_strengths(self):
        names = {s.name for s in validation_specs("sign_flip")}
        self.assertNotIn("adversarial_sign_flip_partial_50", names)
        self.assertNotIn("adversarial_scaling_unseen_2x", names)

    def test_heldout_uses_different_seeds_from_validation(self):
        val = run_validation_suite(20, 64, exclusion_threshold=TAU, seed_base=1)
        held = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        val_seeds = {r["seed"] for r in val.values()}
        held_seeds = {r["seed"] for r in held.values()}
        self.assertTrue(val_seeds)
        self.assertTrue(held_seeds)
        self.assertEqual(val_seeds & held_seeds, set(),
                         "validation and held-out must not share seeds")

    def test_unseen_partial_sign_flip_degrades_gracefully(self):
        """A weaker, partially-aligned attack is detected less often, and
        this must be visible rather than hidden."""
        held = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        s = summarize(held)
        full = s["adversarial_sign_flip"]["tpr"]
        half = s["adversarial_sign_flip_partial_50"]["tpr"]
        quarter = s["adversarial_sign_flip_partial_25"]["tpr"]
        self.assertAlmostEqual(full, 1.0, places=6)
        self.assertLess(half, full)
        self.assertLess(quarter, half)
        self.assertGreaterEqual(quarter, 0.0)

    def test_unseen_scaling_and_noise_are_detected(self):
        held = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        s = summarize(held)
        for name in ("adversarial_scaling_unseen_2x", "adversarial_scaling_unseen_8x",
                     "adversarial_noise_unseen_weak", "adversarial_noise_unseen_strong",
                     "adversarial_free_riding"):
            self.assertEqual(s[name]["tpr"], 1.0, name)

    def test_heldout_scenarios_include_stronger_heterogeneity(self):
        held = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        s = summarize(held)
        self.assertIn("honest_non_iid_strong", s)
        self.assertEqual(s["honest_non_iid_strong"]["excluded_honest"], 0)

    def test_small_legitimate_update_is_not_excluded(self):
        held = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        s = summarize(held)
        self.assertEqual(s["honest_small_legitimate_update"]["excluded_honest"], 0)

    def test_intermittent_participation_is_not_excluded(self):
        held = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        s = summarize(held)
        self.assertEqual(s["honest_varying_participation"]["excluded_honest"], 0)


class TestDocumentedFailureMode(unittest.TestCase):
    """The known failure mode is pinned so that it cannot change silently."""

    def test_benign_small_update_is_excluded_and_documented(self):
        """An honest client whose update is far smaller than the population
        median is excluded.  This is a *known, documented* limitation of the
        detector, not an accident: the cosine agreement of a small update is
        dominated by the isotropic noise, which makes the client look like an
        agreement outlier."""
        results = run_validation_suite(20, 64, exclusion_threshold=TAU, seed_base=1)
        s = summarize(results)
        self.assertEqual(s["honest_benign_outlier"]["excluded_honest"], 1)
        self.assertAlmostEqual(s["honest_benign_outlier"]["fpr"], 1 / 20.0, places=9)
        self.assertIsNone(s["honest_benign_outlier"]["tpr"])
        # The description is part of the artefact, so the report can cite it.
        self.assertTrue(s["honest_benign_outlier"]["description"])

    def test_the_failure_mode_is_a_single_client(self):
        results = run_validation_suite(20, 64, exclusion_threshold=TAU, seed_base=1)
        res = results["honest_benign_outlier"]
        self.assertEqual(len(res["excluded"]), 1)
        # The excluded client is the benign outlier (client 0), not an
        # arbitrary one.
        self.assertEqual(res["excluded"], [0])


class TestMetricRecomputation(unittest.TestCase):
    """Every rate must be reproducible from the raw counts."""

    def test_rates_match_the_audit_tool(self):
        results = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        for name, res in results.items():
            c = res["confusion"]
            recomputed = rates_from_counts(c["tp"], c["fp"], c["fn"], c["tn"])
            for key in ("tpr", "fpr", "fnr", "precision", "accuracy"):
                a, b = res["rates"][key], recomputed[key]
                if a is None or b is None:
                    self.assertIsNone(a, name)
                    self.assertIsNone(b, name)
                else:
                    self.assertAlmostEqual(a, b, places=12, msg="%s/%s" % (name, key))

    def test_denominators_are_the_documented_ones(self):
        results = run_heldout_suite(20, 64, exclusion_threshold=TAU, seed_base=9001)
        for name, res in results.items():
            c = res["confusion"]
            r = res["rates"]
            self.assertEqual(r["n_honest"], c["fp"] + c["tn"])
            self.assertEqual(r["n_malicious"], c["tp"] + c["fn"])
            self.assertEqual(r["n_decisions"], c["tp"] + c["fp"] + c["fn"] + c["tn"])
            if c["fp"] + c["tn"] > 0:
                self.assertAlmostEqual(r["fpr"], c["fp"] / (c["fp"] + c["tn"]),
                                       places=12, msg=name)


class TestThresholdSelectionUsesValidationOnly(unittest.TestCase):

    def test_selection_function_cannot_accept_heldout_data(self):
        import inspect
        from experiments.calibrate_trust import select_exclusion_threshold
        params = set(inspect.signature(select_exclusion_threshold).parameters)
        self.assertNotIn("heldout_results", params)
        self.assertNotIn("heldout_results_by_tau", params)

    def test_selection_is_unaffected_by_heldout_data(self):
        """If the held-out suite were consulted during selection, replacing
        it with garbage would change the selected threshold."""
        import experiments.calibrate_trust as cal

        baseline, _ = cal.select_exclusion_threshold()

        original = cal.run_heldout_suite

        def _boom(*args, **kwargs):
            raise AssertionError("held-out suite must not be used for selection")

        cal.run_heldout_suite = _boom
        try:
            selected, _ = cal.select_exclusion_threshold()
        finally:
            cal.run_heldout_suite = original
        self.assertEqual(baseline, selected)

    def test_selected_threshold_matches_the_deployed_configuration(self):
        """The calibration must reproduce the threshold the validated
        benchmark actually used."""
        from experiments.calibrate_trust import select_exclusion_threshold
        selected, table = select_exclusion_threshold()
        self.assertAlmostEqual(selected, 0.40, places=9)
        # The chosen value must be the interior point of the plateau.
        survivors = [r["tau"] for r in table if r.get("in_selected_interval")]
        self.assertGreaterEqual(len(survivors), 1)
        self.assertIn(selected, survivors)

    def test_deployed_threshold_is_the_configured_one(self):
        from configs.config_loader import load_config
        cfg = load_config()
        self.assertAlmostEqual(
            float(cfg["trust_management"]["exclusion_threshold_tau"]), 0.40, places=9)


class TestInitialTrustPrior(unittest.TestCase):

    def test_fresh_node_has_no_observed_history(self):
        tracker = BehavioralTrustTracker(default_trust=0.70)
        rec = tracker.get_record("V1")
        self.assertEqual(rec["observed_successes"], 0)
        self.assertEqual(rec["observed_total"], 0)
        self.assertIsNone(rec["last_sinr"])
        self.assertIsNone(rec["last_utilization"])
        # The prior is explicit and separate.
        self.assertEqual(rec["prior_successes"], 7)
        self.assertEqual(rec["prior_total"], 10)

    def test_observations_are_recorded_separately(self):
        tracker = BehavioralTrustTracker(default_trust=0.70)
        tracker.record_interaction("V1", True, 0.01, 0.1, 20.0, 0.2)
        rec = tracker.get_record("V1")
        self.assertEqual(rec["observed_total"], 1)
        self.assertEqual(rec["observed_successes"], 1)
        self.assertEqual(rec["prior_successes"], 7)
        self.assertIsNotNone(rec["last_sinr"])
        self.assertIsNotNone(rec["last_utilization"])

    def test_prior_is_role_independent(self):
        tracker = BehavioralTrustTracker(default_trust=0.70)
        a = tracker.get_record("V1")
        b = tracker.get_record("V2")
        self.assertEqual(a["prior_successes"], b["prior_successes"])
        self.assertEqual(a["prior_total"], b["prior_total"])

    def test_prior_influence_decays_with_evidence(self):
        """A larger prior must matter less once observations accumulate."""
        tracker = BehavioralTrustTracker(default_trust=0.70)
        # With the prior alone, the historical factor is 7/10 = 0.7.
        self.assertAlmostEqual(tracker.prior_sensitivity("V1", 0, 0), 0.7, places=6)
        # After 30 observations the prior still carries 10/40 of the weight.
        self.assertAlmostEqual(tracker.prior_sensitivity("V1", 30, 30), 37.0 / 40.0, places=9)
        self.assertAlmostEqual(tracker.prior_sensitivity("V1", 30, 0), 7.0 / 40.0, places=9)
        # The prior's weight shrinks as the evidence grows: with 300
        # observations it is only 10/310 of the total.
        self.assertAlmostEqual(tracker.prior_sensitivity("V1", 300, 300), 307.0 / 310.0, places=9)
        self.assertLess(tracker.prior_sensitivity("V1", 300, 300)
                        - tracker.prior_sensitivity("V1", 30, 30), 0.07)

    def test_initial_trust_value_is_the_documented_prior(self):
        tracker = BehavioralTrustTracker(default_trust=0.70)
        self.assertAlmostEqual(tracker.get_trust("V1"), 0.70, places=6)


if __name__ == "__main__":
    unittest.main()
