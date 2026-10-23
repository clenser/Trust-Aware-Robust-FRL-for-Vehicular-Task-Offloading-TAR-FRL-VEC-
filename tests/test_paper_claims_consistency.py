"""
The headline numbers quoted in README.md and docs/PAPER.md must equal the numbers
recomputed from the active run.  If the run changes (or a number is edited by
hand) this test fails, so the prose cannot drift away from the evidence.
"""

import io
import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.claims_report import _headline_claims            # noqa: E402
from web.app import _active_run_entry                             # noqa: E402


def _read(rel):
    with io.open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


class TestProseMatchesEvidence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        entry = _active_run_entry(os.path.join(ROOT, "results"))
        if entry is None:
            raise unittest.SkipTest("no active run")
        run_dir = entry["directory"]
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(ROOT, run_dir)
        with io.open(os.path.join(run_dir, "benchmark_results.json"), encoding="utf-8") as f:
            cls.data = json.load(f)
        cls.A = cls.data["experiment_a_baseline_comparison"]
        cls.claims = {c["id"]: c for c in _headline_claims(cls.A)}
        cls.readme = _read("README.md")
        cls.paper = _read("docs/PAPER.md")

    def _fmt(self, x):
        return ("%+.2f" % x)

    def test_paired_differences_quoted_in_the_paper(self):
        for cid in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"):
            r = self.claims[cid]["res"]
            self.assertIn(self._fmt(r["mean_difference"]), self.paper, cid)
            self.assertIn(self._fmt(r["ci95_low"]), self.paper, cid)
            self.assertIn(self._fmt(r["ci95_high"]), self.paper, cid)

    def test_verdicts_quoted_in_the_paper(self):
        self.assertEqual(self.claims["H6"]["verdict"], "SUPPORTED")
        for cid in ("H1", "H2", "H3", "H4", "H5", "H7", "H8"):
            self.assertNotEqual(self.claims[cid]["verdict"], "SUPPORTED", cid)
        self.assertIn("**supported**", self.paper)

    def test_readme_headline_numbers(self):
        r1, r6 = self.claims["H1"]["res"], self.claims["H6"]["res"]
        self.assertIn("%.1f" % r1["mean_difference"], self.readme)
        self.assertIn("%.1f" % r6["mean_difference"], self.readme)
        self.assertIn("%.3f" % self.claims["H1"]["p_holm"], self.readme)

    def test_detector_rates_quoted(self):
        c = self.A["Proposed_TFRL"]["pooled_confusion_counts"]
        tpr = c["tp"] / (c["tp"] + c["fn"])
        fpr = c["fp"] / (c["fp"] + c["tn"])
        for text in (self.paper, self.readme):
            self.assertIn("%.3f" % tpr, text)
            self.assertIn("%.3f" % fpr, text)

    def test_method_completion_means_in_table_1(self):
        for name in ("FedAvg_Clean_Reference", "Oracle_Exclusion_Upper_Bound", "Robust_FL_MultiKrum",
                     "Nearest_RSU", "Proposed_TFRL", "Standard_FedAvg_RL", "Resource_Aware",
                     "Non_Federated_D3QN", "Local_Only"):
            self.assertIn("%.1f" % self.A[name]["completion_rate"]["mean"], self.paper, name)

    def test_unsupported_claims_are_not_made(self):
        for text in (self.readme, self.paper):
            self.assertNotIn("complete policy collapse", text.lower())
            self.assertNotIn("more energy-efficient than local", text.lower().replace(
                "trust-aware fl is more energy-efficient than local execution", ""))


if __name__ == "__main__":
    unittest.main()
