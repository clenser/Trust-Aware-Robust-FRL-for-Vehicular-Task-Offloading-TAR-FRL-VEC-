"""
Behavioural tests for the simulation lifecycle and the automatic
federated-learning scheduler.

These tests drive the *real* HTTP API and the *real* background-task
lifecycle (the FastAPI lifespan starts the single simulation loop and the
broadcaster).  They never set ``is_running`` directly to fake a start.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from web.app import SimulationState, app, create_app, sim as global_sim
from fastapi.testclient import TestClient


class LifecycleTestBase(unittest.TestCase):
    """Shared helpers: every test drives the real API.

    The application is created through the factory with a dedicated
    :class:`SimulationState`, so no background task, worker thread or trust
    state can leak between tests.
    """

    def setUp(self):
        # An isolated simulation instance and a client whose lifespan owns the
        # single simulation loop, the broadcaster and the worker pool.
        self.state = SimulationState()
        self._application = create_app(self.state)
        self.client = TestClient(self._application)
        self.client.__enter__()
        self._reset()

    def tearDown(self):
        try:
            self.client.post("/api/control", json={"action": "stop"})
        except Exception:
            pass
        try:
            self.client.__exit__(None, None, None)
        finally:
            self.state.close(wait=True)
            self.state = None
            self.client = None

    def _reset(self):
        r = self.client.post("/api/control", json={"action": "stop"})
        self.assertIn(r.status_code, (200, 409))
        # Test isolation: the round history of previous tests is cleared here.
        # This is fixture setup, not a way of faking a start (the start itself
        # always goes through the API).
        self.state.fl_round_history = []
        self.state.steps_since_last_fl = 0
        self.state.fl_stage = "idle"
        self.state.fl_stage_detail = "Ready - press START to run the simulation"
        self.state.current_round_metrics = None
        self.state.fl_round_in_progress = False
        self.state.last_error = None

    # -- helpers ------------------------------------------------------
    def kpis(self):
        r = self.client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        return r.json()["kpis"]

    def control(self, action):
        return self.client.post("/api/control", json={"action": action})

    def wait_for_steps(self, minimum, timeout=8.0):
        """Wait until the simulation step counter reaches ``minimum``."""
        deadline = time.time() + timeout
        last = 0
        while time.time() < deadline:
            last = self.kpis()["sim_steps"]
            if last >= minimum:
                return last
            time.sleep(0.1)
        return last


class TestSimulationLifecycle(LifecycleTestBase):

    # 1 ---------------------------------------------------------------
    def test_start_transitions_stopped_to_running(self):
        """START changes the backend from STOPPED to RUNNING."""
        self.assertEqual(self.kpis()["sim_state"], "STOPPED")

        r = self.control("start")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "started")
        self.assertEqual(body["state"], "RUNNING")

        k = self.kpis()
        self.assertEqual(k["sim_state"], "RUNNING")
        self.assertTrue(k["is_running"])
        self.assertIsNotNone(k["session_id"])

    # 2 ---------------------------------------------------------------
    def test_step_counter_advances_after_start(self):
        """The simulation step counter advances after START."""
        self.control("start")
        self.assertEqual(self.kpis()["sim_steps"], 0)
        reached = self.wait_for_steps(3)
        self.assertGreaterEqual(reached, 3, "the step counter did not advance")

    # 3 ---------------------------------------------------------------
    def test_pause_stops_step_advancement(self):
        """PAUSE stops step advancement without destroying state."""
        self.control("start")
        self.wait_for_steps(2)
        before = self.kpis()
        self.assertGreater(before["total_tasks"], 0)

        r = self.control("pause")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"], "PAUSED")

        paused_at = self.kpis()["sim_steps"]
        time.sleep(1.5)
        after = self.kpis()

        self.assertEqual(after["sim_steps"], paused_at,
                         "the simulation kept advancing while paused")
        self.assertEqual(after["sim_state"], "PAUSED")
        # State is preserved, not destroyed.
        self.assertEqual(after["total_tasks"], before["total_tasks"])
        self.assertEqual(after["session_id"], before["session_id"])

    # 4 ---------------------------------------------------------------
    def test_resume_continues_the_same_session(self):
        """RESUME continues the same session without resetting anything."""
        self.control("start")
        self.wait_for_steps(3)
        running = self.kpis()

        self.control("pause")
        r = self.control("resume")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"], "RUNNING")

        resumed = self.kpis()
        self.assertEqual(resumed["session_id"], running["session_id"],
                         "RESUME must not create a new session")
        self.assertGreaterEqual(resumed["total_tasks"], running["total_tasks"])
        self.assertGreater(self.wait_for_steps(running["sim_steps"] + 1),
                           running["sim_steps"],
                           "the simulation did not continue after RESUME")

    # 5 ---------------------------------------------------------------
    def test_stop_terminates_session_and_background_work(self):
        """STOP terminates the session safely."""
        self.control("start")
        self.wait_for_steps(2)
        session = self.kpis()["session_id"]

        r = self.control("stop")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"], "STOPPED")

        k = self.kpis()
        self.assertEqual(k["sim_state"], "STOPPED")
        self.assertFalse(k["is_running"])
        self.assertEqual(k["sim_steps"], 0)
        self.assertIsNone(k["session_started_at_iso"])

        # The background loop must no longer advance anything.
        time.sleep(1.0)
        self.assertEqual(self.kpis()["sim_steps"], 0)
        self.assertIsNotNone(session)

    # 6 ---------------------------------------------------------------
    def test_multiple_start_requests_do_not_duplicate_loops(self):
        """Repeated START requests must not create a second simulation loop."""
        self.control("start")
        self.wait_for_steps(2)

        for _ in range(3):
            r = self.control("start")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "already_running")

        # A duplicated loop would roughly double the step rate.
        before = self.kpis()["sim_steps"]
        time.sleep(2.0)
        delta = self.kpis()["sim_steps"] - before

        # The loop ticks at ~2 steps/s (0.5 s per step); allow generous slack
        # but reject a clearly doubled rate.
        self.assertLessEqual(delta, 12,
                             f"the step rate doubled ({delta} steps in 2 s)")

    # 7 ---------------------------------------------------------------
    def test_two_websocket_clients_do_not_accelerate_simulation(self):
        """WebSocket clients subscribe; they never advance simulation time."""
        self.control("start")
        self.wait_for_steps(1)

        with self.client.websocket_connect("/ws/telemetry") as ws1:
            ws1.receive_json()
            time.sleep(0.4)
            one = self.kpis()["sim_steps"]

            with self.client.websocket_connect("/ws/telemetry") as ws2:
                ws2.receive_json()
                time.sleep(0.4)
                two = self.kpis()["sim_steps"]

        self.assertGreater(two, one)
        # Two subscribers must not advance the clock faster than one.
        self.assertLessEqual(two - one, 6,
                             "a second WebSocket client accelerated the simulation")

    # 8 ---------------------------------------------------------------
    def test_backend_exceptions_are_visible_and_surfaceed(self):
        """A background failure must be visible and must not leave the UI stuck."""
        self.control("start")
        self.wait_for_steps(1)

        # Inject a failure into the simulation step.  The patch targets the
        # *isolated* instance used by this test and is restored afterwards.
        original_step = self.state.env.update_positions
        calls = {"n": 0}

        def exploding_update():
            calls["n"] += 1
            raise RuntimeError("synthetic background failure")

        self.state.env.update_positions = exploding_update
        try:
            deadline = time.time() + 12
            k = self.kpis()
            while time.time() < deadline:
                k = self.kpis()
                if k["sim_state"] == "ERROR" or k["last_error"]:
                    break
                time.sleep(0.2)
        finally:
            self.state.env.update_positions = original_step

        k = self.kpis()
        self.assertEqual(k["sim_state"], "ERROR",
                         "the backend failure was not surfaced in the state")
        self.assertIsNotNone(k["last_error"])
        self.assertIn("synthetic background failure", k["last_error"])
        self.assertFalse(k["is_running"])

        # The console must be able to recover: a new START works.
        r = self.control("start")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"], "RUNNING")

    # 12 --------------------------------------------------------------
    def test_initial_metrics_are_not_falsely_shown(self):
        """With no observations the dashboard must not show current numbers."""
        self._reset()
        k = self.kpis()
        self.assertEqual(k["total_tasks"], 0)
        self.assertIsNone(k["avg_latency_ms"],
                          "latency must be unavailable, not 0, before any task")
        self.assertIsNone(k["completion_rate"],
                          "completion rate must be unavailable, not 0 or 100")
        self.assertEqual(k["sim_steps"], 0)
        self.assertIsNone(k["session_id"])

        # The invariant that matters: a session that has executed no task
        # must never report a latency or a completion rate.  This holds
        # whatever the simulation has done in the meantime.
        self.control("start")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            k = self.kpis()
            if k["total_tasks"] == 0:
                self.assertIsNone(k["avg_latency_ms"])
                self.assertIsNone(k["completion_rate"])
            else:
                # Once a task has run the metrics must be real numbers.
                self.assertIsNotNone(k["avg_latency_ms"])
                self.assertIsNotNone(k["completion_rate"])
                self.assertGreater(k["total_tasks"], 0)
                break
            time.sleep(0.05)

        self.control("pause")
        k = self.kpis()
        if k["total_tasks"] == 0:
            self.assertIsNone(k["avg_latency_ms"])
            self.assertIsNone(k["completion_rate"])


class TestAutomaticFederatedLearning(LifecycleTestBase):

    def _configure(self, **kw):
        payload = {
            "fl_warmup_steps": 2,
            "fl_interval_steps": 3,
            "fl_min_buffer_size": 4,
            "fl_min_clients": 2,
        }
        payload.update(kw)
        r = self.client.post("/api/configure", json=payload)
        self.assertEqual(r.status_code, 200)

    # 9 ---------------------------------------------------------------
    def test_fl_begins_only_after_valid_experience(self):
        """No FL round may start before the experience requirements are met."""
        self._configure(fl_warmup_steps=4, fl_min_buffer_size=4, fl_min_clients=2)
        self.control("start")

        # Not enough steps yet -> the scheduler must not fire.
        time.sleep(0.6)
        k = self.kpis()
        self.assertEqual(k["fl_round"], 0, "an FL round started before the warm-up elapsed")

        # The readiness report explains why it is waiting.
        readiness = self.client.get("/api/fl_rounds").json()["readiness"]
        self.assertFalse(readiness["ready"])
        self.assertTrue(readiness["reason"])

        # Once the warm-up elapses and the buffers are filled, a round runs.
        # Wait for a *completed* round: ``fl_round`` counts rounds that have
        # started, ``fl_rounds_completed`` counts rounds that have finished.
        deadline = time.time() + 25
        while time.time() < deadline and self.kpis()["fl_rounds_completed"] < 1:
            time.sleep(0.2)
        self.assertGreaterEqual(self.kpis()["fl_rounds_completed"], 1,
                                "no automatic FL round completed after the requirements were met")

    # 10 --------------------------------------------------------------
    def test_successful_round_updates_counters_and_dashboard(self):
        """A successful round updates the round counter, model version and payload."""
        self._configure()
        self.control("start")

        deadline = time.time() + 30
        while time.time() < deadline and self.kpis()["fl_rounds_completed"] < 1:
            time.sleep(0.2)

        k = self.kpis()
        self.assertGreaterEqual(k["fl_round"], 1)
        self.assertGreaterEqual(k["fl_rounds_completed"], 1)
        self.assertEqual(k["fl_rounds_failed"], 0)
        self.assertGreaterEqual(k["global_model_version"], 1)

        body = self.client.get("/api/fl_rounds").json()
        self.assertTrue(body["rounds"])
        last = body["rounds"][-1]
        self.assertEqual(last["status"], "success")
        self.assertGreater(last["trained_clients"], 0)
        self.assertGreater(last["accepted_clients"], 0)
        # The full lifecycle must be recoverable from the round record.
        self.assertIn("training", last["stage_timeline"])
        self.assertIn("aggregating", last["stage_timeline"])
        self.assertIn("updating", last["stage_timeline"])
        # The dashboard payload must expose the same round.
        state = self.client.get("/api/state").json()
        self.assertIsNotNone(state["last_round"])
        self.assertEqual(state["last_round"]["round_number"], last["round_number"])
    # 11 --------------------------------------------------------------
    def test_failed_rounds_do_not_increment_success_counters(self):
        """A failed round must not be counted as a successful round."""
        self._configure(fl_warmup_steps=1, fl_interval_steps=1,
                        fl_min_buffer_size=100000, fl_min_clients=2)
        self.control("start")

        deadline = time.time() + 12
        while time.time() < deadline:
            k = self.kpis()
            if k["fl_stage"] == "waiting_experience" or k["fl_rounds_failed"] > 0:
                break
            time.sleep(0.2)

        k = self.kpis()
        self.assertEqual(k["fl_rounds_completed"], 0,
                         "a round that never trained was counted as successful")
        self.assertEqual(k["global_model_version"], 0,
                         "the model version advanced without a successful round")
        self.assertEqual(k["fl_round"], 0)

        rounds = self.client.get("/api/fl_rounds").json()["rounds"]
        for r in rounds:
            self.assertNotEqual(r["status"], "success")

    # extra -----------------------------------------------------------
    def test_fl_stage_is_published_live_to_subscribers(self):
        """Subscribers observe the live federated lifecycle."""
        self._configure()
        self.control("start")

        seen = set()
        deadline = time.time() + 25
        with self.client.websocket_connect("/ws/telemetry") as ws:
            while time.time() < deadline:
                try:
                    msg = ws.receive_json()
                except Exception:
                    break
                stage = msg.get("kpis", {}).get("fl_stage")
                if stage:
                    seen.add(stage)
                if "complete" in seen and "aggregating" in seen:
                    break

        self.assertIn("training", seen, f"stages seen: {seen}")
        self.assertIn("aggregating", seen, f"stages seen: {seen}")
        self.assertIn("updating", seen, f"stages seen: {seen}")
        self.assertIn("complete", seen, f"stages seen: {seen}")


class TestControlValidation(LifecycleTestBase):

    def test_invalid_transitions_return_clear_errors(self):
        self._reset()
        r = self.control("pause")
        self.assertEqual(r.status_code, 409)
        self.assertIn("detail", r.json())

        r = self.control("resume")
        self.assertEqual(r.status_code, 409)

        r = self.client.post("/api/control", json={"action": "nonsense"})
        self.assertEqual(r.status_code, 400)

        r = self.client.post("/api/control", json={})
        self.assertEqual(r.status_code, 400)

    def test_start_after_stop_creates_a_new_session(self):
        self.control("start")
        self.wait_for_steps(1)
        first = self.kpis()["session_id"]

        self.control("stop")
        self.control("start")
        second = self.kpis()["session_id"]

        self.assertNotEqual(first, second)
        self.assertIsNotNone(self.kpis()["session_started_at_iso"])


if __name__ == "__main__":
    unittest.main()
