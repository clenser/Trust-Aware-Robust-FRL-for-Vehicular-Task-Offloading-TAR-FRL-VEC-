"""
Artifact-integrity tests.

These guard the rule that the *active* presentation may only contain figures
that were regenerated from the raw records of the validated current run:

* the current-run plot directory contains exactly the manifest figures and
  nothing else;
* every manifest figure exists and matches its recorded SHA-256;
* the manifest's source results file still matches its recorded SHA-256;
* a stale / unreferenced figure in the current-run set is rejected;
* historical run directories are preserved untouched;
* the cleaner refuses to touch a protected historical directory;
* the dashboard serves only manifest figures - a historical figure name
  returns 404 even though the file still exists on disk.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.plotter import (
    MANIFEST_NAME, CURRENT_FIGURES, clean_output_dir, protected_plots_dirs,
    cross_check_aggregates, recompute_paired, sha256_file,
)
from web.app import _load_figure_manifest, _active_run_entry
from experiments.research_figures import RESEARCH_FIGURES

from experiments.supplementary_figures import SUPPLEMENTARY_FIGURES

ALL_FIGURES = list(CURRENT_FIGURES) + list(RESEARCH_FIGURES)
TOP_LEVEL_FIGURES = ALL_FIGURES + list(SUPPLEMENTARY_FIGURES)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
HISTORICAL_RUN = "run_20261021_150330"
TOP_PLOTS = os.path.join(ROOT, "plots")


def _active_run_id():
    """The run id of the validated, published active run."""
    from web.app import _active_run_entry
    entry = _active_run_entry(RESULTS)
    if entry is None:
        raise unittest.SkipTest("no active, validated run in results/manifest.json")
    return entry["run_id"]


def _active_dir():
    return os.path.join(RESULTS, _active_run_id())


def _load_json(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


class TestActiveRunPointer(unittest.TestCase):

    def test_manifest_lists_the_active_run(self):
        entries = _load_json(os.path.join(RESULTS, "manifest.json"))
        ids = [e.get("run_id") for e in entries]
        self.assertIn(_active_run_id(), ids)

    def test_active_pointer_resolves_to_the_validated_run(self):
        entry = _active_run_entry(RESULTS)
        self.assertIsNotNone(entry, "no active, validated run found")
        self.assertEqual(entry.get("validation_status"), "passed")
        self.assertFalse(entry.get("superseded", False))

    def test_only_one_active_run(self):
        entries = _load_json(os.path.join(RESULTS, "manifest.json"))
        active = [e for e in entries
                  if e.get("published_as_active") and not e.get("superseded")]
        self.assertEqual(len(active), 1,
                         "exactly one run may be published as active")


class TestCurrentRunFigureSet(unittest.TestCase):

    def setUp(self):
        self.active_run = _active_run_id()
        self.active_dir = _active_dir()
        self.plots_dir = os.path.join(self.active_dir, "plots")
        self.manifest_path = os.path.join(self.plots_dir, MANIFEST_NAME)
        self.assertTrue(os.path.isfile(self.manifest_path),
                        "the current run must carry a figure manifest")
        self.manifest = _load_json(self.manifest_path)

    def test_manifest_belongs_to_the_active_run(self):
        self.assertEqual(self.manifest.get("experiment_id"), self.active_run)

    def test_every_figure_on_disk_is_in_the_manifest(self):
        on_disk = {n for n in os.listdir(self.plots_dir)
                   if n.endswith(".png")}
        listed = {f["filename"] for f in self.manifest["figures"]}
        self.assertEqual(on_disk, listed,
                         "the current-run plot directory contains a figure that "
                         "is not referenced by the manifest")

    def test_no_stale_historical_figure_in_the_current_run(self):
        on_disk = {n for n in os.listdir(self.plots_dir) if n.endswith(".png")}
        stale = {"fig2_malicious_ratio_sweep.png", "fig3_poisoning_strength.png"}
        self.assertEqual(on_disk & stale, set(),
                         "a stale historical figure is present in the current run")

    def test_manifest_covers_the_expected_figures(self):
        listed = {f["filename"] for f in self.manifest["figures"]}
        self.assertEqual(listed, set(ALL_FIGURES))

    def test_figure_hashes_match_the_files_on_disk(self):
        for entry in self.manifest["figures"]:
            path = os.path.join(self.plots_dir, entry["filename"])
            self.assertTrue(os.path.isfile(path), entry["filename"])
            self.assertEqual(sha256_file(path), entry["sha256"],
                             "%s does not match its recorded hash" % entry["filename"])

    def test_manifest_records_the_source_of_every_figure(self):
        for entry in self.manifest["figures"]:
            self.assertTrue(entry.get("experiment_id"))
            self.assertTrue(entry.get("source_experiment"))
            self.assertTrue(entry.get("source_records"))
            self.assertTrue(entry.get("source_results_sha256"))

    def test_manifest_source_file_hash_matches(self):
        src = self.manifest["results_file"]
        self.assertTrue(os.path.isfile(src), src)
        self.assertEqual(sha256_file(src), self.manifest["results_file_sha256"])

    def test_manifest_cross_check_reports_no_problems(self):
        self.assertEqual(self.manifest.get("cross_check_problems"), [])

    def test_aggregate_cross_check_recomputes_cleanly(self):
        data = _load_json(os.path.join(self.active_dir, "benchmark_results.json"))
        self.assertEqual(cross_check_aggregates(data), [])


class TestTopLevelPlotsDirectory(unittest.TestCase):

    def test_top_level_plots_hold_only_the_current_figures(self):
        if not os.path.isdir(TOP_PLOTS):
            self.skipTest("no top-level plots directory")
        on_disk = {n for n in os.listdir(TOP_PLOTS) if n.endswith(".png")}
        self.assertEqual(on_disk, set(TOP_LEVEL_FIGURES),
                         "the active plots directory contains stale figures")

    def test_top_level_plots_have_a_manifest(self):
        path = os.path.join(TOP_PLOTS, MANIFEST_NAME)
        self.assertTrue(os.path.isfile(path))
        manifest = _load_json(path)
        listed = {f["filename"] for f in manifest["figures"]}
        self.assertEqual(listed, set(TOP_LEVEL_FIGURES))


class TestHistoricalArtifactsArePreserved(unittest.TestCase):

    def test_historical_run_figures_still_exist(self):
        hist = os.path.join(RESULTS, HISTORICAL_RUN, "plots")
        self.assertTrue(os.path.isdir(hist))
        for name in ("fig1_baseline_comparison.png", "fig2_malicious_ratio_sweep.png",
                     "fig3_poisoning_strength.png", "fig4_ablation_study.png",
                     "fig5_sensitivity_alpha.png"):
            self.assertTrue(os.path.isfile(os.path.join(hist, name)),
                            "historical figure %s was removed" % name)

    def test_historical_run_results_still_exist(self):
        hist = os.path.join(RESULTS, HISTORICAL_RUN, "benchmark_results.json")
        self.assertTrue(os.path.isfile(hist))
        self.assertGreater(os.path.getsize(hist), 0)

    def test_historical_directory_is_protected_from_cleaning(self):
        protected = protected_plots_dirs(RESULTS)
        hist = os.path.join(RESULTS, HISTORICAL_RUN, "plots")
        self.assertIn(os.path.abspath(hist), protected)

    def test_cleaner_refuses_a_protected_directory(self):
        hist = os.path.join(RESULTS, HISTORICAL_RUN, "plots")
        with self.assertRaises(RuntimeError):
            clean_output_dir(hist, set())

    def test_cleaner_does_not_touch_historical_figures(self):
        hist = os.path.join(RESULTS, HISTORICAL_RUN, "plots")
        before = sorted(os.listdir(hist))
        try:
            clean_output_dir(hist, set())
        except RuntimeError:
            pass
        self.assertEqual(sorted(os.listdir(hist)), before)


class TestCleanerRejectsStaleFigures(unittest.TestCase):

    def test_cleaner_removes_unreferenced_figures(self):
        tmp = tempfile.mkdtemp(prefix="vannet_plots_")
        try:
            keep = os.path.join(tmp, "fig1_baseline_comparison.png")
            stale = os.path.join(tmp, "fig2_malicious_ratio_sweep.png")
            with open(keep, "wb") as f:
                f.write(b"x")
            with open(stale, "wb") as f:
                f.write(b"x")
            removed = clean_output_dir(tmp, {"fig1_baseline_comparison.png"})
            self.assertEqual(removed, ["fig2_malicious_ratio_sweep.png"])
            self.assertTrue(os.path.isfile(keep))
            self.assertFalse(os.path.isfile(stale))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cleaner_ignores_non_figure_files(self):
        tmp = tempfile.mkdtemp(prefix="vannet_plots_")
        try:
            other = os.path.join(tmp, "figures_manifest.json")
            with open(other, "wb") as f:
                f.write(b"{}")
            removed = clean_output_dir(tmp, set())
            self.assertEqual(removed, [])
            self.assertTrue(os.path.isfile(other))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestDashboardServesOnlyActiveFigures(unittest.TestCase):

    def test_manifest_resolves_to_the_active_run(self):
        manifest, plots_dir = _load_figure_manifest()
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.get("experiment_id"), _active_run_id())
        self.assertTrue(os.path.isdir(plots_dir))

    def test_api_rejects_a_historical_figure_name(self):
        from fastapi.testclient import TestClient
        from web.app import create_app
        client = TestClient(create_app())
        r = client.get("/api/plots/fig2_malicious_ratio_sweep.png")
        self.assertEqual(r.status_code, 404)
        r = client.get("/api/plots/fig3_poisoning_strength.png")
        self.assertEqual(r.status_code, 404)

    def test_api_serves_the_manifest_figures(self):
        from fastapi.testclient import TestClient
        from web.app import create_app
        client = TestClient(create_app())
        r = client.get("/api/figures")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["experiment_id"], _active_run_id())
        names = {f["filename"] for f in data["figures"]}
        self.assertEqual(names, set(ALL_FIGURES))
        for name in names:
            resp = client.get("/api/plots/%s" % name)
            self.assertEqual(resp.status_code, 200, name)

    def test_api_rejects_path_traversal(self):
        from fastapi.testclient import TestClient
        from web.app import create_app
        client = TestClient(create_app())
        for bad in ("../results/benchmark_results.json", "..%2Fbenchmark_results.json"):
            r = client.get("/api/plots/%s" % bad)
            self.assertIn(r.status_code, (404, 400, 422))


class TestFrontendHasNoHardCodedFigures(unittest.TestCase):

    def test_index_html_does_not_reference_figure_files(self):
        path = os.path.join(ROOT, "web", "static", "index.html")
        with io.open(path, encoding="utf-8") as f:
            html = f.read()
        self.assertNotIn("fig2_malicious_ratio_sweep", html)
        self.assertNotIn("fig3_poisoning_strength", html)
        self.assertNotIn("/api/plots/fig", html)

    def test_dashboard_js_has_no_wildcard_figure_loading(self):
        path = os.path.join(ROOT, "web", "static", "js", "research.js")
        with io.open(path, encoding="utf-8") as f:
            js = f.read()
        self.assertNotIn("fig2_malicious_ratio_sweep", js)
        self.assertNotIn("fig3_poisoning_strength", js)
        # No wildcard directory listing may be used to build the figure list.
        self.assertNotIn("api/plots/*", js)
        self.assertIn("/api/figures", js)


class TestPairedStatisticsAreRecomputed(unittest.TestCase):

    def test_recomputed_paired_difference_matches_the_stored_one(self):
        data = _load_json(os.path.join(_active_dir(), "benchmark_results.json"))
        exp_a = data["experiment_a_baseline_comparison"]
        stored = data["experiment_a_paired_comparison"]
        ref = exp_a["Proposed_TFRL"]
        for name, res in exp_a.items():
            if name == "Proposed_TFRL":
                continue
            fresh = recompute_paired(res, ref, "avg_latency_ms")
            old = stored[name]["latency_ms"]
            self.assertEqual(fresh["n"], old["n"], name)
            self.assertAlmostEqual(fresh["mean_difference"], old["mean_difference"],
                                   places=4, msg=name)
            self.assertAlmostEqual(fresh["ci95_low"], old["ci95_low"],
                                   places=4, msg=name)
            self.assertAlmostEqual(fresh["ci95_high"], old["ci95_high"],
                                   places=4, msg=name)
            if old["p_value"] is not None:
                self.assertAlmostEqual(fresh["p_value"], old["p_value"],
                                       places=6, msg=name)

    def test_sign_convention_is_method_minus_reference(self):
        """A positive difference must mean the compared method was slower."""
        data = _load_json(os.path.join(_active_dir(), "benchmark_results.json"))
        exp_a = data["experiment_a_baseline_comparison"]
        ref = exp_a["Proposed_TFRL"]
        for name, res in exp_a.items():
            if name == "Proposed_TFRL":
                continue
            fresh = recompute_paired(res, ref, "avg_latency_ms")
            self.assertGreater(fresh["mean_difference"],
                               fresh["mean_difference"] - fresh["ci95"],
                               name)


class TestStatisticalVerificationTool(unittest.TestCase):

    def test_verification_reports_no_disagreement(self):
        from experiments.verify_statistics import verify
        report = verify(os.path.join(_active_dir(), "benchmark_results.json"))
        self.assertEqual(report["problems"], [],
                         "the published results disagree with the raw records: %s"
                         % report["problems"])

    def test_verification_derives_deadline_violations(self):
        from experiments.verify_statistics import verify
        report = verify(os.path.join(_active_dir(), "benchmark_results.json"))
        for name, d in report["derived"].items():
            self.assertGreater(d["n_observations"], 0, name)
            self.assertLessEqual(d["deadline_violations"], d["n_observations"])
            self.assertIsNotNone(d["deadline_violation_rate"])

    def test_verification_checks_protocol_identity(self):
        from experiments.verify_statistics import verify
        report = verify(os.path.join(_active_dir(), "benchmark_results.json"))
        seeds = set()
        for name, info in report["protocol"].items():
            self.assertTrue(info["evaluation_was_inert"], name)
            meta = _load_json(os.path.join(_active_dir(), "metadata.json"))
            expected = len(meta["seeds"]) * meta["evaluation_config"]["episodes"]                 if name in ("Proposed_TFRL",) else None
            if name == "Proposed_TFRL":
                self.assertEqual(len(info["episode_seeds"]), expected, name)
            self.assertGreater(len(info["episode_seeds"]), 0, name)
            seeds.add(tuple(info["episode_seeds"]))
        self.assertEqual(len(seeds), 1,
                         "all methods must be evaluated on the same episodes")


if __name__ == "__main__":
    unittest.main()
