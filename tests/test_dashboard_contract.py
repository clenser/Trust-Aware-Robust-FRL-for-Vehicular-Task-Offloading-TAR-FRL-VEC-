"""
Static contract tests for the dashboard front-end.

Browser automation (Playwright / Selenium) is not available in this
environment, so these tests verify the DOM contract that the JavaScript
relies on:

* every element id referenced by ``dashboard.js`` exists in ``index.html``,
* every required console panel is present,
* the manual FL trigger button is gone,
* the JavaScript no longer contains the stub renderers,
* the JavaScript does not treat a missing simulation-state value as "running",
* the legacy conflated attack flag is no longer used.
"""

import os
import re
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HTML_PATH = os.path.join(ROOT, "web", "static", "index.html")
JS_PATH = os.path.join(ROOT, "web", "static", "js", "dashboard.js")
VIEW_JS_PATH = os.path.join(ROOT, "web", "static", "js", "sim_view.js")
RESEARCH_JS_PATH = os.path.join(ROOT, "web", "static", "js", "research.js")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestDashboardDomContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.html = read(HTML_PATH)
        cls.js = read(JS_PATH)
        cls.view_js = read(VIEW_JS_PATH)
        cls.research_js = read(RESEARCH_JS_PATH)
        cls.html_ids = set(re.findall(r'id="([^"]+)"', cls.html))

    # ------------------------------------------------------------------
    # Elements referenced by the JavaScript must exist
    # ------------------------------------------------------------------
    def test_all_referenced_element_ids_exist(self):
        referenced = set(re.findall(r"getElementById\(\s*'([^']+)'\s*\)", self.js))
        referenced |= set(re.findall(r"setText\(\s*'([^']+)'", self.js))
        referenced |= set(re.findall(r"querySelector\('#([^']+)'\)", self.js))

        missing = sorted(referenced - self.html_ids)
        self.assertEqual(missing, [],
                         f"dashboard.js references element ids that do not exist in index.html: {missing}")

    # ------------------------------------------------------------------
    # Required console panels
    # ------------------------------------------------------------------
    def test_required_fl_panels_present(self):
        required = [
            # simulation status
            "kpi-latency", "kpi-completion", "kpi-round", "kpi-attackers",
            "kpi-simstate", "connection-status",
            # error banner
            "error-banner",
            # session identity
            "session-id", "session-started", "session-steps",
            # compact live FL indicator
            "fl-status-indicator", "fl-stage-detail", "viewport", "inspector",
            "event-feed", "view-sim", "view-research", "research-root",
            "input-vehicles", "input-attack-ratio", "btn-apply-config", "config-message",
            # FL round status panel
            "fl-stage", "fl-round-no", "fl-status", "fl-elapsed", "fl-strategy",
            "fl-model-version", "fl-eligible", "fl-trained", "fl-accepted",
            "fl-excluded", "fl-readiness-reason", "fl-exclusion-list",
            # client participation table
            "client-table", "client-table-body",
            # round history
            "fl-round-history",
            # research / security metrics
            "m-completed", "m-failed", "m-dropped", "m-latency", "m-energy",
            "m-tpr", "m-fpr", "m-precision", "m-fnr", "m-simtime", "m-eval-note",
            # controls
            "btn-start", "btn-pause", "btn-stop", "select-mobility",
            "mobility-source",
        ]
        missing = [i for i in required if i not in self.html_ids]
        self.assertEqual(missing, [], f"Missing console elements: {missing}")

    # ------------------------------------------------------------------
    # The optional 3D view must not be able to kill the console
    # ------------------------------------------------------------------
    def test_scene_initialisation_is_isolated(self):
        """A 3D-renderer failure must not abort the console initialisation.

        This is the regression guard for the "dashboard renders but clicking
        START does nothing" failure: an exception in the 3D constructor used
        to abort the whole DOMContentLoaded handler, so the WebSocket was
        never opened and the control handlers were never attached.
        """
        handler = re.search(
            r"addEventListener\('DOMContentLoaded',\s*\(\)\s*=>\s*\{", self.js)
        self.assertIsNotNone(handler, "DOMContentLoaded handler not found")

        body = self.js[handler.end():handler.end() + 2500]
        # The scene construction must be inside a try/catch.
        scene_at = body.find("VANETRoadView")
        self.assertGreater(scene_at, -1, "the road viewport is not constructed")
        try_at = body.rfind("try", 0, scene_at)
        self.assertGreater(try_at, -1,
                           "the 3D scene construction is not protected by try/catch")
        catch_at = body.find("catch", scene_at)
        self.assertGreater(catch_at, -1, "the 3D scene construction has no catch block")

        # The critical wiring must come after the isolated section.
        for fn in ("connectWebSocket()", "setupEventListeners()", "fetchInitialState()"):
            self.assertIn(fn, body, f"{fn} is missing from the init handler")
            self.assertGreater(body.find(fn), catch_at,
                               f"{fn} must run after the optional 3D init")

    def test_charts_initialisation_is_isolated(self):
        self.assertIn("initCharts();", self.js)
        # The chart construction must be guarded by a try/catch.
        idx = self.js.find("initCharts();")
        window = self.js[max(0, idx - 200):idx + 200]
        self.assertIn("try", window, "initCharts() is not protected by try/catch")
        self.assertIn("catch", window, "initCharts() has no catch block")

    def test_error_banner_is_sticky_after_failed_request(self):
        """A failed control request must remain visible."""
        self.assertIn("error-banner", self.html)
        self.assertIn("showBanner", self.js)
        self.assertIn("bannerSticky", self.js)
        # The control failure path must request a sticky banner.
        self.assertIn("'error', true)", self.js,
                      "a failed control request must show a sticky error")
        # And a successful action must be able to clear it.
        self.assertIn("clearBanner(true)", self.js)

    def test_api_errors_are_not_silently_swallowed(self):
        """The control path must surface failures instead of only logging them."""
        self.assertNotIn("console.error('Control error:', e);\n    }\n  }", self.js)
        self.assertIn("failed:", self.js)

    def test_control_button_state_machine_present(self):
        for token in ("'STOPPED'", "'RUNNING'", "'PAUSED'", "'ERROR'"):
            self.assertIn(token, self.js, f"the button state machine lacks {token}")

    def test_backend_control_contract(self):
        """The /api/control response must expose the authoritative state."""
        app_py = read(os.path.join(ROOT, "web", "app.py"))
        self.assertIn('result["kpis"]', app_py)
        self.assertIn('result["session"]', app_py)
        # The lifecycle states must be defined on the backend.
        for state in ("SIM_STOPPED", "SIM_RUNNING", "SIM_PAUSED", "SIM_ERROR"):
            self.assertIn(state, app_py, f"the backend does not define {state}")
        # Invalid transitions must produce a 409.
        self.assertIn("409", app_py)

    def test_background_tasks_owned_by_lifespan(self):
        app_py = read(os.path.join(ROOT, "web", "app.py"))
        self.assertIn("asynccontextmanager", app_py)
        self.assertIn("_simulation_loop(state)", app_py)
        self.assertIn("_broadcast_loop(state)", app_py)
        # The WebSocket handler must not advance the simulation.
        ws = app_py[app_py.find("async def websocket_telemetry"):]
        ws = ws[:2000]
        self.assertNotIn("sim.step()", ws,
                         "the WebSocket handler must not advance the simulation")
        self.assertNotIn("state.step()", ws,
                         "the WebSocket handler must not advance the simulation")

    def test_worker_pool_is_managed_and_shut_down(self):
        """Local training must run on a managed pool that is shut down."""
        app_py = read(os.path.join(ROOT, "web", "app.py"))
        self.assertIn("ThreadPoolExecutor", app_py)
        self.assertIn("shutdown_workers", app_py)
        # The default asyncio executor must not be used for the FL workers:
        # its shutdown would block the event loop.
        self.assertNotIn("asyncio.to_thread(", app_py)

    def test_static_assets_are_cache_busted(self):
        """The page must reference versioned assets so a stale bundle is not used."""
        self.assertRegex(self.html, r"dashboard\.js\?v=\d")
        self.assertRegex(self.html, r"sim_view\.js\?v=\d")
        self.assertRegex(self.html, r"research\.js\?v=\d")
        self.assertRegex(self.html, r"style\.css\?v=\d")

    def test_index_is_not_cached(self):
        app_py = read(os.path.join(ROOT, "web", "app.py"))
        self.assertIn("no-store", app_py)

    # ------------------------------------------------------------------
    # The manual FL trigger must be gone
    # ------------------------------------------------------------------
    def test_manual_fl_trigger_button_removed(self):
        self.assertNotIn("btn-trigger-fl", self.html,
                         "The manual FL trigger button must be removed entirely")
        self.assertNotIn("btn-trigger-fl", self.js,
                         "The manual FL trigger handler must be removed entirely")
        self.assertNotIn("TRIGGER FL", self.html)
        self.assertNotIn("/api/trigger_round", self.js,
                         "The frontend must not call a manual FL trigger endpoint")

    def test_backend_exposes_no_manual_trigger_endpoint(self):
        app_py = read(os.path.join(ROOT, "web", "app.py"))
        self.assertNotIn("/api/trigger_round", app_py,
                         "The manual FL trigger endpoint must be removed")

    # ------------------------------------------------------------------
    # The stub renderers must be implemented
    # ------------------------------------------------------------------
    def test_renderers_are_implemented(self):
        for fn in ("updateFLRoundsDisplay", "updateClientParticipationDisplay"):
            m = re.search(r"function\s+%s\s*\([^)]*\)\s*\{" % fn, self.js)
            self.assertIsNotNone(m, f"{fn} is missing")
            body = self.js[m.end():]
            # Find the matching closing brace of the function body.
            depth = 1
            out = []
            for ch in body:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                out.append(ch)
            body = "".join(out)
            self.assertGreater(len(body.strip()), 40,
                               f"{fn} appears to be a stub (body too short)")
            self.assertNotIn("contains only a comment", body)

    # ------------------------------------------------------------------
    # Simulation state must come from the backend, never be inferred
    # ------------------------------------------------------------------
    def test_missing_state_is_not_treated_as_running(self):
        """A missing is_running value must never be read as 'running'."""
        # The old buggy pattern was: kpis.is_running !== false  -> true when missing
        self.assertNotIn("is_running !== false", self.js,
                         "A missing is_running value must not be treated as running")
        self.assertNotIn("!== false", self.js)

    def test_start_button_derives_from_backend_state(self):
        """The START button must be disabled from the authoritative backend flag."""
        m = re.search(r"const startBtn\s*=\s*document\.getElementById\('btn-start'\);", self.js)
        self.assertIsNotNone(m, "START button wiring not found")
        tail = self.js[m.end():m.end() + 700]
        self.assertIn("simState", tail,
                      "The START button state must be derived from the backend sim_state")
        self.assertIn("disabled", tail)
        # The state itself must come from the backend snapshot, never guessed.
        self.assertIn("k.sim_state", self.js)

    # ------------------------------------------------------------------
    # Security telemetry must keep the concepts separate
    # ------------------------------------------------------------------
    def test_legacy_attack_flag_is_unused(self):
        self.assertNotIn("ev.is_attack", self.js)
        self.assertNotIn("is_attack:", self.js)
        self.assertNotIn("ev.is_attack", self.view_js)
        self.assertIn("detector_alert", self.js)
        self.assertIn("ground_truth_malicious", self.js)
        self.assertIn("task_failed", self.js)

    def test_viewport_separates_detector_output_from_ground_truth(self):
        """Vehicle colour comes from the detector's trust score; the ground-truth
        role is only drawn as an explicit, opt-in evaluation overlay."""
        self.assertIn("trust_score", self.view_js)
        self.assertIn("showGroundTruth", self.view_js)
        self.assertNotIn("ev.role === 'poisoning'", self.view_js)
        # The overlay is off by default.
        self.assertIn("this.showGroundTruth = false", self.view_js)

    def test_viewport_uses_backend_geometry(self):
        """Road length, lanes and RSU coverage are drawn from the backend scene."""
        for token in ("scene.road_length_m", "scene.num_lanes", "coverage_radius", "x_min_m"):
            self.assertIn(token, self.view_js)

    def test_research_view_reads_the_active_run(self):
        self.assertIn("/api/research/summary", self.research_js)
        self.assertIn("/api/figures", self.research_js)
        # No benchmark number may be embedded in the view.
        self.assertNotRegex(self.research_js, r"81\.9\d")

    # ------------------------------------------------------------------
    # Mobility selector must be initialised from the backend
    # ------------------------------------------------------------------
    def test_mobility_selector_synced_from_backend(self):
        m = re.search(r"const mobilitySel\s*=\s*document\.getElementById\('select-mobility'\);", self.js)
        self.assertIsNotNone(m, "Mobility selector wiring not found")
        tail = self.js[m.end():m.end() + 500]
        self.assertIn("mobility_mode", tail,
                      "The mobility selector must be synced from the backend configuration")

    def test_mobility_options_cover_all_backend_modes(self):
        for mode in ("highway", "trace", "authentic_trace"):
            self.assertIn(f'value="{mode}"', self.html,
                          f"The mobility selector is missing the '{mode}' option")

    # ------------------------------------------------------------------
    # Missing observations must render as N/A
    # ------------------------------------------------------------------
    def test_null_metrics_render_as_not_available(self):
        self.assertIn("function fmt(", self.js)
        self.assertIn("'N/A'", self.js)
        # Latency / completion must not be hard-coded to 0.
        self.assertNotIn("kpis.avg_latency_ms} ms", self.js)
        self.assertNotIn("kpis.completion_rate}%", self.js)


if __name__ == "__main__":
    unittest.main()
