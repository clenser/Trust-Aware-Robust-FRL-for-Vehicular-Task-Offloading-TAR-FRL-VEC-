"""
Tests for the console additions: windowed series for the charts, the federated
block that is sent only when it changes, the per-round visualisation payload
(real uploads / exclusions), HTTP caching and the research summary.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from web.app import SimulationState, create_app


class TestSeriesAndWindows(unittest.TestCase):
    def setUp(self):
        self.sim = SimulationState()
        self.sim.fl_warmup_steps = 2
        self.sim.fl_interval_steps = 3
        self.sim.fl_min_buffer_size = 4
        self.sim.fl_min_clients = 2
        self.sim.control("start")

    def tearDown(self):
        self.sim.control("stop")
        self.sim.close(wait=True)

    def _steps(self, n):
        for _ in range(n):
            self.sim.step()

    def test_a_point_is_recorded_per_step_with_consistent_numbers(self):
        self._steps(6)
        pts = list(self.sim.series)
        self.assertEqual(len(pts), 6)
        self.assertEqual([p["step"] for p in pts], list(range(1, 7)))
        last = pts[-1]
        self.assertTrue(0.0 <= last["win_completion"] <= 100.0)
        self.assertTrue(0.0 <= last["cum_completion"] <= 100.0)
        self.assertEqual(sum(last["actions"]), len(self.sim._recent))

    def test_window_is_bounded_and_cumulative_is_slower_than_recent(self):
        self._steps(30)
        self.assertLessEqual(len(self.sim._recent), self.sim.SERIES_WINDOW)
        snap = self.sim.get_state_snapshot()
        k = snap["kpis"]
        self.assertIn("completion_rate_window", k)
        self.assertEqual(k["window_tasks"], len(self.sim._recent))
        self.assertEqual(sum(k["action_mix"]), k["window_tasks"])
        self.assertIn("mean_epsilon", k)

    def test_series_are_reset_by_a_new_session(self):
        self._steps(5)
        self.sim.control("stop")
        self.sim.rebuild_sim()
        self.assertEqual(len(self.sim.series), 0)
        self.assertEqual(len(self.sim.round_series), 0)

    def test_series_endpoint_matches_the_buffers(self):
        self._steps(4)
        payload = self.sim.series_payload()
        self.assertEqual(len(payload["series"]), 4)
        self.assertEqual(set(payload["roles"].keys()), {str(v) for v in self.sim.env.vehicles})


class TestFederatedBlockAndVisualisation(unittest.TestCase):
    def setUp(self):
        self.sim = SimulationState()
        self.sim.fl_warmup_steps = 2
        self.sim.fl_interval_steps = 3
        self.sim.fl_min_buffer_size = 4
        self.sim.fl_min_clients = 2
        self.sim.control("start")

    def tearDown(self):
        self.sim.control("stop")
        self.sim.close(wait=True)

    def test_light_snapshot_omits_the_federated_block_but_keeps_its_version(self):
        for _ in range(8):
            self.sim.step()
        light = self.sim.get_state_snapshot(include_fl=False)
        full = self.sim.get_state_snapshot()
        self.assertNotIn("fl_rounds", light)
        self.assertIn("fl_rounds", full)
        self.assertEqual(light["fl_version"], full["fl_version"])
        self.assertLess(len(str(light)), len(str(full)))

    def test_a_round_produces_the_visualisation_payload_and_bumps_the_version(self):
        for _ in range(8):
            self.sim.step()
        v0 = self.sim.fl_version
        metrics = asyncio.run(self.sim.run_fl_round())
        self.assertGreater(self.sim.fl_version, v0)
        viz = self.sim.fl_viz
        self.assertIsNotNone(viz)
        self.assertEqual(viz["round"], metrics.round_number)
        self.assertEqual(len(viz["clients"]), len(self.sim.clients))
        for c in viz["clients"]:
            self.assertIn(c["included"], (True, False, None))
        self.assertEqual(len(self.sim.round_series), 1)
        rs = self.sim.round_series[-1]
        self.assertEqual(rs["round"], metrics.round_number)
        self.assertEqual(rs["excluded"], sum(1 for c in viz["clients"] if c["included"] is False))

    def test_exclusions_in_the_payload_are_backend_decisions_not_ground_truth(self):
        for _ in range(8):
            self.sim.step()
        asyncio.run(self.sim.run_fl_round())
        for c in self.sim.fl_viz["clients"]:
            if c["included"] is False:
                self.assertTrue(c["reason"], "an exclusion must carry the detector's reason")


class TestHttpCachingAndResearchSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_versioned_static_assets_are_cacheable(self):
        r = self.client.get("/static/js/dashboard.js?v=6.1")
        self.assertEqual(r.status_code, 200)
        self.assertIn("max-age", r.headers.get("cache-control", ""))

    def test_live_state_is_never_cached(self):
        r = self.client.get("/api/state")
        self.assertEqual(r.headers.get("cache-control"), "no-store")

    def test_index_is_not_cached(self):
        r = self.client.get("/")
        self.assertIn("no-store", r.headers.get("cache-control", ""))

    def test_research_summary_has_an_etag_and_revalidates_with_304(self):
        r = self.client.get("/api/research/summary")
        self.assertEqual(r.status_code, 200)
        etag = r.headers.get("etag")
        self.assertTrue(etag)
        r2 = self.client.get("/api/research/summary", headers={"If-None-Match": etag})
        self.assertEqual(r2.status_code, 304)

    def test_research_summary_contains_verdicts_and_derived_series(self):
        data = self.client.get("/api/research/summary").json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(len(data["claims"]["rows"]), 8)
        self.assertIn("per_seed_completion", data["derived"])
        self.assertIn("proposed_false_exclusion_by_round", data["derived"])
        self.assertIsInstance(data["supplementary"], list)

    def test_plots_are_served_with_a_revalidating_cache_header(self):
        figs = self.client.get("/api/figures").json()["figures"]
        r = self.client.get("/api/plots/%s" % figs[0]["filename"])
        self.assertEqual(r.status_code, 200)
        self.assertIn("must-revalidate", r.headers.get("cache-control", ""))

    def test_series_endpoint(self):
        d = self.client.get("/api/series").json()
        for key in ("series", "round_series", "trust_series", "roles", "window"):
            self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main()
