"""
Regression tests for the three P0 findings.

1. The authentic trace replay must follow the *recorded* trajectories of
   several source vehicles across several timestamps, and must handle the
   arrivals and departures that the source fleet actually exhibits.
2. A federated round in which every client is excluded must be reported as a
   failure and must not advance the global model version.
3. The federated round task must be owned by the application lifecycle, and a
   round that is overtaken by a rebuild/stop must not mutate the new session.
"""

import asyncio
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch

from environment.mobility import TraceDrivenMobility
from environment.vanet_env import VANETOffloadingEnv
from web.app import SimulationState, create_app
from fastapi.testclient import TestClient

TRACE = "data/traces/ngsim_us101_authentic.csv"


def _trace_available():
    return os.path.exists(TRACE)


class TestAuthenticTraceReplay(unittest.TestCase):
    """P0-1: the simulated fleet must follow the recorded source trajectories."""

    @classmethod
    def setUpClass(cls):
        if not _trace_available():
            raise unittest.SkipTest("The processed authentic trace is not present.")
        cls.df = pd.read_csv(TRACE)
        cls.src = cls.df.set_index(["time_sec", "vehicle_id"])

    def test_simulated_fleet_follows_source_trajectories(self):
        """Positions and speeds must match the source records exactly."""
        env = VANETOffloadingEnv(num_vehicles=25, attack_ratio=0.0,
                                 mobility_mode="authentic_trace")
        mob = env.mobility

        checked = 0
        mismatches = []
        for _ in range(40):
            env.update_positions()
            t = round(mob.current_time_sec, 1)
            for sim_vid, src_vid in mob.active_slot_bindings.items():
                key = (t, src_vid)
                if key not in self.src.index:
                    continue
                row = self.src.loc[key]
                v = env.vehicles[sim_vid]
                dx = abs(float(row["pos_x"]) - v.x)
                dv = abs(float(row["speed_mps"]) - v.speed)
                checked += 1
                if dx > 0.05 or dv > 0.05:
                    mismatches.append((t, sim_vid, src_vid, round(dx, 3), round(dv, 3)))

        self.assertGreater(checked, 20,
                           "too few source records were replayed to be meaningful")
        self.assertEqual(mismatches, [],
                         "the simulated vehicles deviate from the recorded trajectories")

    def test_multiple_vehicles_are_replayed(self):
        """More than one source vehicle must be followed at the same time."""
        env = VANETOffloadingEnv(num_vehicles=25, attack_ratio=0.0,
                                 mobility_mode="authentic_trace")
        mob = env.mobility

        max_concurrent = 0
        distinct_sources = set()
        for _ in range(120):
            env.update_positions()
            bindings = mob.active_slot_bindings
            max_concurrent = max(max_concurrent, len(bindings))
            distinct_sources |= set(bindings.values())

        self.assertGreater(max_concurrent, 1,
                           "only one source vehicle was ever followed")
        self.assertGreater(len(distinct_sources), 5,
                           "the replay did not follow a realistic number of source vehicles")

    def test_vehicle_arrivals_and_departures_are_handled(self):
        """Slots must be released on departure and re-bound on arrival."""
        env = VANETOffloadingEnv(num_vehicles=25, attack_ratio=0.0,
                                 mobility_mode="authentic_trace")
        mob = env.mobility

        arrivals = 0
        departures = 0
        previous = set()
        for _ in range(120):
            env.update_positions()
            current = set(mob.active_slot_bindings.values())
            arrivals += len(current - previous)
            departures += len(previous - current)
            previous = current

        self.assertGreater(arrivals, 0, "no vehicle arrival was observed")
        self.assertGreater(departures, 0, "no vehicle departure was observed")

    def test_source_identity_is_preserved(self):
        """A source vehicle must keep the same slot for the whole of its stay.

        A slot may legitimately be re-bound to a *different* source vehicle
        once its previous source has departed the observation window, but a
        source vehicle that is still present must never be moved onto another
        slot (that would swap trajectories between vehicles).
        """
        env = VANETOffloadingEnv(num_vehicles=25, attack_ratio=0.0,
                                 mobility_mode="authentic_trace")
        mob = env.mobility

        source_to_slot = {}
        slot_to_source = {}
        for _ in range(60):
            env.update_positions()
            bindings = mob.active_slot_bindings
            present = set(bindings.values())

            # 1. A source that is still present keeps its slot.
            for src_vid, slot in mob.active_source_bindings.items():
                if src_vid in source_to_slot:
                    self.assertEqual(source_to_slot[src_vid], slot,
                                     "a source vehicle was moved onto another slot")
                else:
                    source_to_slot[src_vid] = slot

            # 2. A slot may only change its source after the previous source
            #    has departed.
            for slot, src_vid in bindings.items():
                previous = slot_to_source.get(slot)
                if previous is not None and previous != src_vid:
                    self.assertNotIn(previous, present,
                                     "a slot was re-bound while its previous "
                                     "source vehicle was still present")
                slot_to_source[slot] = src_vid

        self.assertTrue(source_to_slot)

    def test_absent_vehicles_do_not_extrapolate(self):
        """A vehicle without a source trajectory must not invent one."""
        env = VANETOffloadingEnv(num_vehicles=25, attack_ratio=0.0,
                                 mobility_mode="authentic_trace")
        mob = env.mobility

        for _ in range(10):
            env.update_positions()

        unbound = [vid for vid in env.vehicles
                   if vid not in mob.active_slot_bindings]
        self.assertTrue(unbound, "expected some slots to be unbound early on")
        for vid in unbound:
            v = env.vehicles[vid]
            self.assertFalse(v.active, "an unbound vehicle must be marked inactive")
            self.assertEqual(v.speed, 0.0, "an unbound vehicle must not move")

    def test_absent_vehicles_do_not_produce_training_transitions(self):
        """An absent vehicle must not contribute a task transition."""
        env = VANETOffloadingEnv(num_vehicles=25, attack_ratio=0.0,
                                 mobility_mode="authentic_trace")
        states = env._get_global_state_dict()
        vid = next(iter(states))
        state, task = states[vid]

        v = env.vehicles[vid]
        v.active = False
        next_state, reward, done, info, next_task = env.step_vehicle(vid, 0, task)
        self.assertFalse(info["present"])
        self.assertEqual(info["target"], "not_present")
        self.assertEqual(reward, 0.0)



class TestTraceEndPolicy(unittest.TestCase):
    """The end of the recording must be handled explicitly, not by wrapping."""

    def _mobility(self, policy="stop"):
        return TraceDrivenMobility(
            trace_file="data/synthetic_traces/synthetic_ngsim_highway.csv",
            dt=0.5, road_length=2400.0, trace_end_policy=policy)

    def test_default_policy_does_not_wrap(self):
        mob = self._mobility()
        self.assertEqual(mob.trace_end_policy, "stop")
        # Past the end of the recording the frame must be empty, not the
        # beginning of the recording again.
        matched, frame = mob._match_frame(mob.max_time + 10.0)
        self.assertEqual(frame, {})
        self.assertTrue(mob.trace_exhausted)

    def test_inside_the_recording_the_frame_is_populated(self):
        mob = self._mobility()
        matched, frame = mob._match_frame(5.0)
        self.assertTrue(frame)
        self.assertFalse(mob.trace_exhausted)

    def test_loop_policy_is_opt_in(self):
        mob = self._mobility("loop")
        self.assertEqual(mob.trace_end_policy, "loop")
        matched, frame = mob._match_frame(mob.max_time + 10.0)
        self.assertTrue(frame, "the loop policy must replay the recording")
        self.assertFalse(mob.trace_exhausted)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            self._mobility("wrap-around")

    def test_policy_is_reported_in_the_geometry(self):
        self.assertEqual(self._mobility().geometry["trace_end_policy"], "stop")

class TestAggregationAcceptanceChecks(unittest.TestCase):
    """P0-2: a round with no accepted update must not be a success."""

    def setUp(self):
        self.sim = SimulationState()
        self.sim.fl_warmup_steps = 2
        self.sim.fl_interval_steps = 3
        self.sim.fl_min_buffer_size = 4
        self.sim.fl_min_clients = 2
        self.sim.control("start")
        for _ in range(8):
            self.sim.step()

    def tearDown(self):
        self.sim.control("stop")
        # Release the worker pool so that no thread outlives this test.
        self.sim.close(wait=True)

    def test_round_with_every_client_excluded_fails(self):
        """Excluding every client must produce a failed round."""
        # Raise the exclusion threshold above every achievable trust score so
        # that the trust filter rejects every client.
        self.sim.server.trust_engine.exclusion_threshold = 1.01
        self.sim.exclusion_threshold = 1.01

        metrics = asyncio.run(self.sim.run_fl_round())

        self.assertNotEqual(metrics.status, "success",
                            "a round in which every client was excluded was reported as a success")
        self.assertEqual(metrics.status, "failed")
        self.assertEqual(metrics.accepted_clients, 0)
        self.assertGreater(metrics.excluded_clients, 0)
        self.assertIsNotNone(metrics.error_message)

    def test_failed_aggregation_does_not_advance_the_model(self):
        """A failed round must not increment the model version."""
        version_before = self.sim.global_model_version
        weights_before = {k: v.clone() for k, v in self.sim.server.get_global_weights().items()}

        self.sim.server.trust_engine.exclusion_threshold = 1.01
        self.sim.exclusion_threshold = 1.01
        asyncio.run(self.sim.run_fl_round())

        self.assertEqual(self.sim.global_model_version, version_before)
        after = self.sim.server.get_global_weights()
        for k in weights_before:
            self.assertTrue(torch.allclose(weights_before[k].float(), after[k].float(), atol=1e-7),
                            "the global model changed during a failed round")

    def test_failed_aggregation_is_recorded_as_failed(self):
        self.sim.server.trust_engine.exclusion_threshold = 1.01
        self.sim.exclusion_threshold = 1.01
        asyncio.run(self.sim.run_fl_round())

        self.assertTrue(self.sim.fl_round_history)
        last = self.sim.fl_round_history[-1]
        self.assertEqual(last.status, "failed")
        self.assertIn("accepted", (last.error_message or "").lower())

    def test_server_reports_that_no_model_was_produced(self):
        """The aggregator must state explicitly that it produced no model."""
        self.sim.server.trust_engine.exclusion_threshold = 1.01
        self.sim.exclusion_threshold = 1.01
        metrics = asyncio.run(self.sim.run_fl_round())
        agg = metrics.aggregation_metrics
        self.assertIn("aggregation_produced_model", agg)
        self.assertFalse(agg["aggregation_produced_model"])

    def test_successful_round_reports_a_changed_model(self):
        """A genuine round must report that the global model changed."""
        self.sim.server.trust_engine.exclusion_threshold = 0.40
        self.sim.exclusion_threshold = 0.40
        metrics = asyncio.run(self.sim.run_fl_round())
        self.assertEqual(metrics.status, "success", metrics.error_message)
        self.assertGreater(metrics.accepted_clients, 0)
        self.assertTrue(metrics.aggregation_metrics.get("aggregation_produced_model"))
        self.assertTrue(metrics.aggregation_metrics.get("model_changed"),
                        "a successful round must have changed the global model")


class TestFederatedTaskLifecycle(unittest.TestCase):
    """P0-3: the FL task must be owned by the lifecycle and must not outlive it."""

    def test_fl_task_handle_is_retained(self):
        """The scheduler must keep a handle on the round task."""
        from web import app as webapp
        source = open(webapp.__file__, encoding="utf-8").read()
        self.assertIn("state._fl_task = asyncio.create_task", source)
        self.assertIn("state._fl_task", source.split("async def _broadcast_loop")[0])

    def test_round_is_superseded_when_the_simulation_is_rebuilt(self):
        """A rebuild during training must invalidate the in-flight round."""
        sim = SimulationState()
        sim.fl_warmup_steps = 2
        sim.fl_interval_steps = 3
        sim.fl_min_buffer_size = 4
        sim.fl_min_clients = 2
        sim.control("start")
        for _ in range(8):
            sim.step()

        generation = sim._generation

        async def run_and_rebuild():
            task = asyncio.create_task(sim.run_fl_round())
            # Rebuild while the round is in flight.
            await asyncio.sleep(0.0)
            sim.rebuild_sim(keep_round_history=True)
            return await task

        metrics = asyncio.run(run_and_rebuild())

        self.assertNotEqual(sim._generation, generation,
                            "the rebuild did not advance the generation")
        self.assertEqual(metrics.status, "superseded",
                         "a stale round was not marked as superseded")
        self.assertEqual(metrics.failure_stage, "lifecycle")

    def test_superseded_round_does_not_mutate_the_new_session(self):
        """A stale round must not change the model version or the clients."""
        sim = SimulationState()
        sim.fl_warmup_steps = 2
        sim.fl_interval_steps = 3
        sim.fl_min_buffer_size = 4
        sim.fl_min_clients = 2
        sim.control("start")
        for _ in range(8):
            sim.step()

        version_before = sim.global_model_version
        clients_before = {vid: id(c) for vid, c in sim.clients.items()}

        async def run_and_stop():
            task = asyncio.create_task(sim.run_fl_round())
            await asyncio.sleep(0.0)
            sim.control("stop")
            return await task

        metrics = asyncio.run(run_and_stop())

        self.assertEqual(metrics.status, "superseded")
        self.assertEqual(sim.global_model_version, version_before,
                         "the stale round advanced the model version")
        self.assertEqual(sim.state, "STOPPED")

    def test_rebuild_during_training_does_not_raise(self):
        """Rebuilding while the worker thread runs must not corrupt anything."""
        sim = SimulationState()
        sim.fl_warmup_steps = 2
        sim.fl_interval_steps = 3
        sim.fl_min_buffer_size = 4
        sim.fl_min_clients = 2
        sim.control("start")
        for _ in range(8):
            sim.step()

        async def run_and_rebuild_twice():
            task = asyncio.create_task(sim.run_fl_round())
            await asyncio.sleep(0.0)
            sim.rebuild_sim(keep_round_history=True)
            await asyncio.sleep(0.01)
            sim.rebuild_sim(keep_round_history=True)
            return await task

        metrics = asyncio.run(run_and_rebuild_twice())
        self.assertIn(metrics.status, ("superseded", "failed"))
        # The simulation must still be usable afterwards.
        sim.rebuild_sim(keep_round_history=True)
        self.assertTrue(sim.clients)

    def test_lifespan_cancels_and_awaits_the_fl_task(self):
        """The application lifespan must cancel and await the round task."""
        from web import app as webapp
        source = open(webapp.__file__, encoding="utf-8").read()
        lifespan = source.split("async def lifespan")[1].split("def create_app")[0]
        self.assertIn("state._fl_task", lifespan)
        self.assertIn("task.cancel()", lifespan)
        self.assertIn("await task", lifespan)
        self.assertIn("shutdown_workers", lifespan)

    def test_no_orphan_round_task_after_shutdown(self):
        """After the client context exits, no round task may remain pending."""
        state = SimulationState()
        application = create_app(state)
        with TestClient(application) as client:
            client.post("/api/control", json={"action": "start"})
            time.sleep(1.5)
        # The lifespan has shut down; the simulation must be stopped and the
        # worker pool closed.
        self.assertFalse(state.is_running)
        self.assertIsNone(state._fl_task)
        state.shutdown_workers(wait=True, timeout=10.0)


if __name__ == "__main__":
    unittest.main()
