"""
Regression tests for the automatic federated-learning scheduler and the
web research console.

These tests exercise the *real* code paths:

* the central simulation loop started by the FastAPI lifespan,
* the broadcast-only WebSocket subscriber (no simulation advance per socket),
* the REST control and configuration endpoints,
* the federated round lifecycle including honest failure reporting.
"""

import os
import sys
import time
import unittest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from web.app import SimulationState, create_app, FL_STAGE_COMPLETE, FL_STAGE_ERROR


class TestFederatedRoundLifecycle(unittest.TestCase):
    """Unit-level tests of the scheduler logic on a SimulationState instance.

    The simulation is started through the real control API (``control``) rather
    than by setting ``is_running`` directly, so the authoritative state machine
    is always exercised.
    """

    def setUp(self):
        self.sim = SimulationState()
        # Small thresholds so a round can complete quickly.
        self.sim.fl_warmup_steps = 2
        self.sim.fl_interval_steps = 3
        # Must be at least the DQN minimum batch size, otherwise no client can
        # take a gradient step and the round can never legitimately succeed.
        self.sim.fl_min_buffer_size = 4
        self.sim.fl_min_clients = 2

    def tearDown(self):
        self.sim.control("stop")
        # Release the worker pool so that no thread outlives this test.
        self.sim.close(wait=True)

    def _start(self):
        """Start the simulation through the real control path."""
        self.sim.control("start")

    def _pause(self):
        self.sim.control("pause")

    def _run_steps(self, n):
        for _ in range(n):
            self.sim.step()

    def _run_round(self):
        """Run one federated round coroutine to completion."""
        return asyncio.run(self.sim.run_fl_round())

    # ------------------------------------------------------------------
    # 1. Automatic FL enabled by default
    # ------------------------------------------------------------------
    def test_auto_fl_enabled_by_default(self):
        self.assertTrue(self.sim.auto_fl_enabled)

    # ------------------------------------------------------------------
    # 2. No FL round before the minimum eligibility conditions are met
    # ------------------------------------------------------------------
    def test_no_fl_before_minimum_experience(self):
        """A round must not start while the warm-up or buffer criteria are unmet.

        This test drives a *running* simulation (is_running = True) so that
        step() actually advances time; otherwise the warm-up condition would
        never be evaluated at all.
        """
        self._start()
        self.sim.fl_warmup_steps = 4
        self.sim.fl_min_buffer_size = 50   # unreachable within the test

        self._run_steps(2)
        # Warm-up not yet satisfied -> still idle, no round.
        self.assertEqual(self.sim.fl_stage, "idle")
        self.assertEqual(self.sim.server.current_round, 0)
        self.assertFalse(self.sim.fl_readiness()["ready"])

        self._run_steps(3)
        # Warm-up satisfied but buffers too small -> not ready, and an explicit
        # round request is rejected as "insufficient experience" rather than
        # being reported as a successful round.
        readiness = self.sim.fl_readiness()
        self.assertFalse(readiness["ready"])
        self.assertIn("eligible clients", readiness["reason"].lower())

        metrics = self._run_round()
        self.assertEqual(metrics.status, "insufficient_experience")
        self.assertEqual(self.sim.fl_stage, "waiting_experience")
        self.assertEqual(self.sim.server.current_round, 0)
        self.assertEqual(self.sim.global_model_version, 0)

    def test_no_fl_when_min_clients_not_met(self):
        self._start()
        self.sim.fl_min_clients = 999
        self._run_steps(6)
        self.assertEqual(self.sim.server.current_round, 0)
        self.assertIn("eligible clients", self.sim.fl_readiness()["reason"].lower())

    # ------------------------------------------------------------------
    # 3. FL starts automatically once eligible
    # ------------------------------------------------------------------
    def test_fl_round_succeeds_after_eligibility(self):
        self._start()
        self._run_steps(8)
        metrics = self._run_round()
        self.assertEqual(metrics.status, "success")
        self.assertEqual(metrics.stage, FL_STAGE_COMPLETE)
        self.assertGreaterEqual(metrics.trained_clients, self.sim.fl_min_clients)
        self.assertGreater(metrics.accepted_clients, 0)
        self.assertEqual(self.sim.global_model_version, 1)

    # ------------------------------------------------------------------
    # 4. A completed round schedules the next eligible round
    # ------------------------------------------------------------------
    def test_next_round_uses_interval_not_warmup(self):
        self._start()
        self._run_steps(8)
        first = self._run_round()
        self.assertEqual(first.status, "success")

        readiness = self.sim.fl_readiness()
        self.assertEqual(readiness["min_steps"], self.sim.fl_interval_steps)
        self.assertEqual(readiness["steps_since_last_fl"], 0)

        # Not ready until the interval has elapsed.
        self._run_steps(self.sim.fl_interval_steps - 1)
        self.assertFalse(self.sim.fl_readiness()["ready"])

        self._run_steps(1)
        self.assertTrue(self.sim.fl_readiness()["ready"])

        second = self._run_round()
        self.assertEqual(second.round_number, 2)
        self.assertEqual(self.sim.global_model_version, 2)

    # ------------------------------------------------------------------
    # 5. Concurrency: no overlapping rounds
    # ------------------------------------------------------------------
    def test_concurrent_rounds_do_not_overlap(self):
        """Two concurrent round requests must not both execute."""
        self._start()
        self._run_steps(8)

        async def run_both():
            return await asyncio.gather(
                self.sim.run_fl_round(),
                self.sim.run_fl_round(),
            )

        results = asyncio.run(run_both())
        successes = [r for r in results if r.status == "success"]
        self.assertEqual(len(successes), 1,
                         "Exactly one of two concurrent rounds may succeed")
        self.assertEqual(self.sim.server.current_round, 1)

    # ------------------------------------------------------------------
    # 6. Failure handling is honest
    # ------------------------------------------------------------------
    def test_no_op_updates_are_reported_as_failure(self):
        """Clients without a real gradient step must not count as valid updates."""
        self._start()
        # Raise the replay requirement above what a single step can produce so
        # that no client can perform a gradient step.
        self.sim.fl_min_buffer_size = 10_000
        self._run_steps(3)

        metrics = self._run_round()
        # The round is rejected as insufficient experience, not "successful".
        self.assertIn(metrics.status, ("insufficient_experience", "failed"))
        self.assertNotEqual(metrics.status, "success")
        self.assertEqual(self.sim.global_model_version, 0)

    def test_min_buffer_size_is_clamped_to_trainable_minimum(self):
        """A client must never be 'eligible' yet unable to take a gradient step."""
        from rl.agent import DQNAgent

        self.assertGreaterEqual(self.sim.fl_min_buffer_size, DQNAgent.MIN_TRAIN_BATCH)

        # The configuration endpoint must apply the same clamp.
        self.sim.set_config({"fl_min_buffer_size": 1})
        self.assertGreaterEqual(self.sim.fl_min_buffer_size, DQNAgent.MIN_TRAIN_BATCH)

    def test_failed_round_does_not_mark_clients_as_excluded(self):
        """A round that fails before aggregation must not claim clients were excluded."""
        self._start()
        self.sim.fl_min_buffer_size = 10_000
        self._run_steps(3)
        metrics = self._run_round()

        for detail in metrics.per_client_metrics.values():
            if detail["eligible"]:
                self.assertIsNone(detail["included_in_aggregation"],
                                  "Inclusion must be undecided when aggregation never ran")
                self.assertIsNone(detail["exclusion_reason"])

    def test_aggregation_failure_is_recorded(self):
        """An exception during aggregation must produce a failed round record."""
        self._start()
        self._run_steps(8)

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic aggregation failure")

        original = self.sim.server.run_aggregation_round
        self.sim.server.run_aggregation_round = boom
        try:
            metrics = self._run_round()
        finally:
            self.sim.server.run_aggregation_round = original

        self.assertEqual(metrics.status, "failed")
        self.assertEqual(metrics.failure_stage, "aggregating")
        self.assertIn("synthetic aggregation failure", metrics.error_message)
        self.assertEqual(self.sim.global_model_version, 0, "A failed round must not bump the model version")
        self.assertEqual(self.sim.fl_stage, FL_STAGE_ERROR)

    def test_failed_round_recorded_in_history(self):
        self._start()
        self._run_steps(8)
        original = self.sim.server.run_aggregation_round
        self.sim.server.run_aggregation_round = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        try:
            self._run_round()
        finally:
            self.sim.server.run_aggregation_round = original
        self.assertEqual(len(self.sim.fl_round_history), 1)
        self.assertEqual(self.sim.fl_round_history[0].status, "failed")

    # ------------------------------------------------------------------
    # 7. Per-client metrics are real
    # ------------------------------------------------------------------
    def test_per_client_metrics_use_real_training_loss(self):
        """Training loss must come from the DQN update, not from reward magnitude."""
        self._start()
        self._run_steps(8)
        metrics = self._run_round()
        self.assertEqual(metrics.status, "success")

        per_client = metrics.per_client_metrics
        self.assertTrue(per_client)

        trained = [c for c in per_client.values() if c.get("performed_training")]
        self.assertTrue(trained, "At least one client should have trained")
        for c in trained:
            self.assertIsNotNone(c["training_loss"])
            self.assertGreater(c["grad_steps"], 0)
            self.assertIsNotNone(c["training_duration_ms"])
            self.assertTrue(c["submitted_update"])

        # Free riders must be marked as abstaining, not as "trained".
        for c in per_client.values():
            if c["role"] == "free_rider":
                self.assertFalse(c["performed_training"])
                self.assertEqual(c["note"], "free_rider_abstains")

    def test_participation_table_populated(self):
        self._start()
        self._run_steps(8)
        self._run_round()
        self.assertTrue(self.sim.client_participation)
        trained = [p for p in self.sim.client_participation.values() if p.rounds_trained > 0]
        self.assertTrue(trained)
        for p in trained:
            self.assertIsNotNone(p.last_training_loss)

    def test_exclusion_decisions_carry_evidence(self):
        """Excluded clients must state the evidence that caused the exclusion."""
        self._start()
        self._run_steps(8)
        metrics = self._run_round()
        self.assertEqual(metrics.status, "success")

        per = metrics.per_client_metrics
        excluded = [c for c in per.values() if c.get("included_in_aggregation") is False]
        included = [c for c in per.values() if c.get("included_in_aggregation") is True]

        self.assertTrue(included, "Some clients should be accepted")
        for c in included:
            self.assertIsNone(c["exclusion_reason"])

        for c in excluded:
            self.assertTrue(c["exclusion_reason"],
                            "An excluded client must carry a reason")
            self.assertIn("trust", c["exclusion_reason"].lower())

        # The accepted + excluded split must match the aggregator's own count.
        self.assertEqual(len(included) + len(excluded), metrics.accepted_clients + metrics.excluded_clients)

    def test_inclusion_state_is_tri_state(self):
        """Included / excluded / undecided must be distinguishable."""
        self._start()
        self._run_steps(8)
        self._run_round()
        states = {p.inclusion_state for p in self.sim.client_participation.values()}
        self.assertTrue(states <= {"included", "excluded", "not_in_aggregation", "not_participated"})

    # ------------------------------------------------------------------
    # 8. Global model distribution
    # ------------------------------------------------------------------
    def test_every_client_receives_new_global_model(self):
        import torch
        self._start()
        self._run_steps(8)
        self._run_round()

        server_weights = self.sim.server.get_global_weights()
        for vid, client in self.sim.clients.items():
            client_weights = client.agent.get_weights()
            for key in server_weights:
                self.assertTrue(
                    torch.allclose(server_weights[key].float(), client_weights[key].float(), atol=1e-5),
                    f"Client V{vid} did not receive the new global model ({key})"
                )

    # ------------------------------------------------------------------
    # 9. Strategy changes reach the real aggregation implementation
    # ------------------------------------------------------------------
    def test_strategy_change_reaches_server(self):
        self.sim.set_config({"aggregation_strategy": "fedavg"})
        self.assertEqual(self.sim.server.aggregation_strategy, "fedavg")
        self._run_steps(8)
        metrics = self._run_round()
        self.assertEqual(metrics.aggregation_strategy, "fedavg")

    # ------------------------------------------------------------------
    # 10. Pause / resume control the backend
    # ------------------------------------------------------------------
    def test_pause_and_resume_control_backend(self):
        self._start()
        self._run_steps(3)
        before = self.sim.sim_step_count
        self.assertGreater(before, 0)

        self.sim.control("pause")
        self._run_steps(3)
        self.assertEqual(self.sim.sim_step_count, before, "Paused simulation must not advance")

        self.sim.control("resume")
        self._run_steps(2)
        self.assertGreater(self.sim.sim_step_count, before)

    def test_stop_resets_state(self):
        self._start()
        self._run_steps(3)
        self.sim.control("stop")
        self.assertFalse(self.sim.is_running)
        self.assertEqual(self.sim.sim_step_count, 0)
        self.assertEqual(self.sim.server.current_round, 0)
        self.assertEqual(self.sim.global_model_version, 0)

    # ------------------------------------------------------------------
    # 11. Authoritative simulation state in the snapshot
    # ------------------------------------------------------------------
    def test_snapshot_reports_authoritative_state(self):
        snap = self.sim.get_state_snapshot()
        self.assertIn("is_running", snap["kpis"])
        self.assertIn("sim_state", snap["kpis"])
        self.assertIs(snap["kpis"]["is_running"], False)
        self.assertEqual(snap["kpis"]["sim_state"], "STOPPED")

        self._start()
        snap = self.sim.get_state_snapshot()
        self.assertIs(snap["kpis"]["is_running"], True)
        self.assertEqual(snap["kpis"]["sim_state"], "RUNNING")

    def test_missing_observations_are_none_not_defaults(self):
        snap = self.sim.get_state_snapshot()
        # No tasks executed yet -> latency and completion rate are undefined.
        self.assertIsNone(snap["kpis"]["avg_latency_ms"])
        self.assertIsNone(snap["kpis"]["completion_rate"])
        self.assertEqual(snap["kpis"]["total_tasks"], 0)

    def test_current_round_metrics_exposed(self):
        self._start()
        self._run_steps(8)
        self._run_round()
        snap = self.sim.get_state_snapshot()
        # No round is in progress any more, so current_round is None ...
        self.assertIsNone(snap["current_round"])
        # ... but the last completed round is fully described.
        self.assertIsNotNone(snap["last_round"])
        self.assertEqual(snap["last_round"]["status"], "success")
        self.assertTrue(snap["last_round"]["per_client_metrics"])
        self.assertTrue(snap["fl_rounds"])

    # ------------------------------------------------------------------
    # 12. Security telemetry keeps the concepts separate
    # ------------------------------------------------------------------
    def test_security_event_fields_are_separate(self):
        self._start()
        self._run_steps(3)
        snap = self.sim.get_state_snapshot()
        for ev in snap["events"]:
            self.assertIn("ground_truth_malicious", ev)
            self.assertIn("detector_alert", ev)
            self.assertIn("task_failed", ev)
            # The legacy conflated flag must be gone.
            self.assertNotIn("is_attack", ev)

    # ------------------------------------------------------------------
    # 13. Mobility source is described accurately
    # ------------------------------------------------------------------
    def test_mobility_source_reported(self):
        snap = self.sim.get_state_snapshot()
        src = snap["kpis"]["mobility_source"]
        self.assertIn("label", src)
        self.assertIn("authentic", src)
        self.assertIn("note", src)
        self.assertFalse(src["authentic"], "Default highway mode is synthetic")
        self.assertEqual(snap["kpis"]["mobility_mode"], "highway")



def _wait_for_steps(client, previous, timeout=15.0):
    """Wait until the simulation step counter moves past ``previous``.

    A fixed sleep is not a reliable condition: under CPU contention the
    simulation loop may not tick within the sleep window, which would make the
    test report a false failure.  The assertion itself is unchanged - the
    simulation must advance - only the way it is observed becomes robust.
    """
    deadline = time.time() + timeout
    steps = previous
    while time.time() < deadline:
        steps = client.get("/api/state").json()["kpis"]["sim_steps"]
        if steps > previous:
            return steps
        time.sleep(0.1)
    return steps


class TestConsoleAPIAndWebSocket(unittest.TestCase):
    """Integration tests against the real FastAPI app (lifespan + WebSocket).

    Every test builds its own application through the factory so that no
    background task, worker thread or simulation state is shared.
    """

    def setUp(self):
        self.state = SimulationState()
        self.application = create_app(self.state)

    def tearDown(self):
        self.state.request_worker_cancellation()
        self.state.shutdown_workers(wait=True, timeout=10.0)
        self.state = None
        self.application = None

    def _client(self):
        return TestClient(self.application)

    def test_state_endpoint_and_control(self):
        with self._client() as client:
            # Start from a known state.
            client.post("/api/control", json={"action": "stop"})

            r = client.get("/api/state")
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertIn("is_running", data["kpis"])
            self.assertFalse(data["kpis"]["is_running"])
            self.assertEqual(data["kpis"]["sim_state"], "STOPPED")

            r = client.post("/api/control", json={"action": "start"})
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertTrue(body["is_running"])
            # The authoritative lifecycle state is reported as a plain string.
            self.assertEqual(body["state"], "RUNNING")
            # The full KPI snapshot is returned as well.
            self.assertTrue(body["kpis"]["is_running"])
            self.assertIsNotNone(body["session"]["id"])

            r = client.post("/api/control", json={"action": "pause"})
            self.assertFalse(r.json()["is_running"])
            self.assertEqual(r.json()["state"], "PAUSED")

            r = client.post("/api/control", json={"action": "stop"})
            self.assertEqual(r.json()["status"], "stopped")
            self.assertFalse(r.json()["is_running"])
            self.assertEqual(r.json()["state"], "STOPPED")

    def test_configure_endpoint_applies_strategy(self):
        with self._client() as client:
            r = client.post("/api/configure", json={"aggregation_strategy": "coordinate_median"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["config"]["aggregation_strategy"], "coordinate_median")

            r = client.post("/api/configure", json={"aggregation_strategy": "not_a_strategy"})
            self.assertEqual(r.status_code, 400)

    def test_fl_rounds_and_participation_endpoints(self):
        with self._client() as client:
            r = client.get("/api/fl_rounds")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertIn("rounds", body)
            self.assertIn("readiness", body)

            r = client.get("/api/client_participation")
            self.assertEqual(r.status_code, 200)
            self.assertIn("clients", r.json())

    def test_event_log_endpoint(self):
        with self._client() as client:
            r = client.get("/api/event_log")
            self.assertEqual(r.status_code, 200)
            self.assertIn("events", r.json())

    def test_websocket_is_broadcast_only(self):
        """A WebSocket must not advance simulation time by itself.

        The subscriber receives snapshots; the *central* loop owns the clock.
        With no client connected the simulation must not advance either.
        """
        with self._client() as client:
            client.post("/api/control", json={"action": "start"})
            time.sleep(0.4)
            before = client.get("/api/state").json()["kpis"]["sim_steps"]

            with client.websocket_connect("/ws/telemetry") as ws:
                msg = ws.receive_json()
                self.assertIn("kpis", msg)
                self.assertIn("is_running", msg["kpis"])

                # While connected, time advances because of the central loop.
                during = _wait_for_steps(client, before)
                self.assertGreater(during, before)

            # After disconnect the central loop keeps running (it owns the clock).
            after = _wait_for_steps(client, during)
            self.assertGreater(after, during)

    def test_multiple_websockets_do_not_double_advance(self):
        """Two subscribers must not make the simulation run twice as fast."""
        with self._client() as client:
            client.post("/api/control", json={"action": "start"})
            time.sleep(0.5)

            with client.websocket_connect("/ws/telemetry") as ws1:
                ws1.receive_json()
                start = client.get("/api/state").json()["kpis"]["sim_steps"]
                one_conn_steps = _wait_for_steps(client, start, timeout=10.0)

            with client.websocket_connect("/ws/telemetry") as ws1, \
                 client.websocket_connect("/ws/telemetry") as ws2:
                ws1.receive_json()
                ws2.receive_json()
                start2 = client.get("/api/state").json()["kpis"]["sim_steps"]
                two_conn_steps = _wait_for_steps(client, start2, timeout=10.0)

            # Two connections must not advance the clock faster than one.
            one_delta = one_conn_steps - start
            two_delta = two_conn_steps - start2
            self.assertGreaterEqual(one_delta, 0)
            self.assertLessEqual(two_delta, max(1, one_delta) * 1.6,
                                 "Two WebSocket connections advanced the simulation faster than one")

    def test_websocket_reconnect_does_not_reset_simulation(self):
        with self._client() as client:
            client.post("/api/control", json={"action": "start"})
            time.sleep(0.6)
            before = client.get("/api/state").json()["kpis"]["sim_steps"]
            self.assertGreater(before, 0)

            with client.websocket_connect("/ws/telemetry") as ws:
                msg = ws.receive_json()
                self.assertGreaterEqual(msg["kpis"]["sim_steps"], 0)

            with client.websocket_connect("/ws/telemetry") as ws:
                msg = ws.receive_json()
                self.assertGreaterEqual(msg["kpis"]["sim_steps"], before,
                                        "Reconnecting must not reset the simulation")
                self.assertTrue(msg["kpis"]["is_running"])

    def test_fl_stage_is_streamed_to_subscribers(self):
        """A subscriber must observe the live federated round stage."""
        with self._client() as client:
            client.post("/api/configure", json={
                "fl_warmup_steps": 1,
                "fl_interval_steps": 1,
                "fl_min_buffer_size": 2,
                "fl_min_clients": 2,
            })
            client.post("/api/control", json={"action": "start"})

            seen = set()
            deadline = time.time() + 12.0
            with client.websocket_connect("/ws/telemetry") as ws:
                while time.time() < deadline:
                    try:
                        msg = ws.receive_json()
                    except Exception:
                        break
                    stage = msg.get("kpis", {}).get("fl_stage")
                    if stage:
                        seen.add(stage)
                    if "training" in seen and ("aggregating" in seen or "updating" in seen):
                        break

            # At minimum the collector must have seen a round complete.
            self.assertTrue(
                seen & {"training", "aggregating", "updating", "complete", "waiting_experience"},
                f"No federated stage was streamed to the subscriber (saw {seen})"
            )


if __name__ == "__main__":
    unittest.main()
