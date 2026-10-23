"""
FastAPI Backend Server & Simulation Controller for the VANET Research Console.

Architecture
------------
* ONE authoritative :class:`SimulationState` instance.
* ONE central simulation loop (started in the FastAPI lifespan).  WebSocket
  connections are *broadcast-only* subscribers: they never advance simulation
  time, so N browsers still produce exactly one simulation timeline.
* ONE federated-learning coordinator.  It publishes every stage transition
  (``waiting_experience`` -> ``training`` -> ``aggregating`` -> ``updating``
  -> ``complete``) while the round is running, so the dashboard can display
  the live lifecycle instead of only the end state.  Expensive local training
  runs in a worker thread so the asyncio event loop stays responsive.
* Model weights are guarded by a dedicated lock and local training operates on
  *detached copies* of the client networks, so gradient updates can never race
  with action selection.

Telemetry honesty
-----------------
Ground-truth roles, detector alerts and task failures are reported as three
separate fields.  No attack alert is ever synthesised, and metrics that
cannot be computed are reported as ``None`` rather than as a placeholder.
"""

import os
import sys
import copy
import json
import asyncio
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request, Response

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.vanet_env import VANETOffloadingEnv
from federated.client import FederatedClient
from federated.server import FederatedServer
from configs.config_loader import load_config
from rl.agent import DQNAgent

# A client needs at least this many transitions before a gradient step is
# possible at all; below it the client is a non-participant of the round.
MIN_TRAIN_BATCH = int(DQNAgent.MIN_TRAIN_BATCH)


# Stages of the federated round lifecycle.
FL_STAGE_IDLE = "idle"
FL_STAGE_WAITING = "waiting_experience"
FL_STAGE_TRAINING = "training"
FL_STAGE_AGGREGATING = "aggregating"
FL_STAGE_UPDATING = "updating"
FL_STAGE_COMPLETE = "complete"
FL_STAGE_ERROR = "error"

# Authoritative simulation lifecycle states.
SIM_STOPPED = "STOPPED"
SIM_RUNNING = "RUNNING"
SIM_PAUSED = "PAUSED"
SIM_ERROR = "ERROR"

_VALID_SIM_STATES = (SIM_STOPPED, SIM_RUNNING, SIM_PAUSED, SIM_ERROR)

_ACTIVE_STAGES = (FL_STAGE_TRAINING, FL_STAGE_AGGREGATING, FL_STAGE_UPDATING)


@dataclass
class ClientRoundMetrics:
    """Per-client detail for a single federated round."""
    client_id: int
    role: str
    replay_samples: int = 0
    eligible: bool = False
    submitted_update: bool = False
    performed_training: bool = False
    grad_steps: int = 0
    training_loss: Optional[float] = None
    training_duration_ms: Optional[float] = None
    # Tri-state: None = the aggregator did not decide (round failed earlier,
    # or the client never reached the aggregation stage),
    # True = accepted, False = explicitly excluded.
    included_in_aggregation: Optional[bool] = None
    exclusion_reason: Optional[str] = None
    note: Optional[str] = None


@dataclass
class FLRoundMetrics:
    """Metrics for one federated round (in progress or completed)."""
    round_number: int
    status: str  # in_progress | success | failed | insufficient_experience
    stage: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    eligible_clients: int = 0
    participating_clients: int = 0
    trained_clients: int = 0
    submitted_updates: int = 0
    accepted_clients: int = 0
    excluded_clients: int = 0
    aggregation_strategy: str = "unknown"
    global_model_version: int = 0
    session_id: Optional[str] = None
    # Ordered list of the lifecycle stages this round passed through, so the
    # full progression is recoverable even if a subscriber missed a frame.
    stage_timeline: List[str] = field(default_factory=list)
    per_client_metrics: Dict[str, Any] = field(default_factory=dict)
    aggregation_metrics: Dict[str, Any] = field(default_factory=dict)
    trust_scores_before: Dict[str, float] = field(default_factory=dict)
    trust_scores_after: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    failure_stage: Optional[str] = None


@dataclass
class ClientParticipation:
    """Aggregated participation record for a client across the session."""
    client_id: int
    role: str
    samples_contributed: int = 0
    last_training_loss: Optional[float] = None
    last_training_duration_ms: Optional[float] = None
    total_grad_steps: int = 0
    rounds_participated: int = 0
    rounds_trained: int = 0
    rounds_submitted: int = 0
    rounds_included: int = 0
    rounds_excluded: int = 0
    last_round_participated: int = 0
    inclusion_state: str = "not_participated"  # included | excluded | not_participated
    exclusion_reason: Optional[str] = None


class SimulationState:
    """Thread-safe simulation state with an automatic federated-learning scheduler."""

    def __init__(self):
        # _lock      -> guards simulation state (vehicles, counters, config)
        # _fl_lock   -> guards the federated round orchestration
        # _model_lock-> guards client network weights (reads and writes)
        # _trust_lock-> guards the trust engine (shared by step() and aggregation)
        self._lock = threading.RLock()
        self._fl_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._trust_lock = threading.RLock()

        cfg = load_config()
        cfg_mob = cfg.get("mobility", {})
        cfg_adv = cfg.get("adversarial", {})
        cfg_fl = cfg.get("federated_learning", {})
        cfg_trust = cfg.get("trust_management", {})

        self.num_vehicles = cfg_mob.get("num_vehicles", 25)
        self.attack_ratio = cfg_adv.get("attack_ratio", 0.24)
        self.mobility_mode = cfg_mob.get("scenario", "highway")
        self.rsu_x_positions: Optional[List[float]] = None
        self.aggregation_strategy = cfg_fl.get("aggregation_strategy", "trust_aware")
        self.alpha = cfg_trust.get("alpha", 0.50)
        self.exclusion_threshold = cfg_trust.get("exclusion_threshold_tau", 0.40)

        # --- Simulation control ---
        self.state = SIM_STOPPED
        self.is_running = False
        self.sim_speed = 1.0
        self.started_at: Optional[float] = None
        self.sim_step_count = 0
        self.last_error: Optional[str] = None

        # --- Session identity ---
        self.session_id: Optional[str] = None
        self.session_sequence = 0
        self.session_started_at: Optional[float] = None

        # --- Statistics ---
        self.total_completed = 0
        self.total_failed = 0
        self.total_dropped = 0
        self.recent_latencies: List[float] = []
        self.active_events: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self._init_series()

        # --- Federated learning configuration ---
        self.auto_fl_enabled = True
        # The minimum buffer must be at least the DQN minimum batch size,
        # otherwise a client can be "eligible" yet be unable to take a single
        # gradient step.  Clamp it so the readiness check and the training
        # step always agree.
        self.fl_min_buffer_size = max(
            int(cfg_fl.get("min_buffer_size", 10)), DQNAgent.MIN_TRAIN_BATCH
        )
        self.fl_min_clients = int(cfg_fl.get("min_clients", 2))
        self.fl_warmup_steps = int(cfg_fl.get("steps_per_round", 10))
        self.fl_interval_steps = int(cfg_fl.get("fl_interval_steps", 20))
        self.fl_local_epochs = int(cfg_fl.get("local_epochs", 2))

        # --- Federated learning runtime state ---
        self.fl_stage = FL_STAGE_IDLE
        self.fl_stage_detail = ""
        self.fl_stage_updated_at = time.time()
        self.steps_since_last_fl = 0
        self.fl_round_history: List[FLRoundMetrics] = []
        self.current_round_metrics: Optional[FLRoundMetrics] = None
        self.global_model_version = 0
        self.client_participation: Dict[int, ClientParticipation] = {}
        self.fl_round_in_progress = False

        # --- WebSocket subscriber registry ---
        self._subscribers: Set[WebSocket] = set()
        self._ws_versions: Dict[int, int] = {}
        self._subscribers_lock = threading.Lock()

        # --- Background task handles (owned by the application lifespan) ---
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._broadcast_event: Optional[asyncio.Event] = None
        self._simulation_task: Optional[asyncio.Task] = None
        self._fl_scheduler_task: Optional[asyncio.Task] = None
        # The handle of the federated round that is currently in flight.  It
        # is owned by the application lifecycle so that shutdown can cancel
        # and await it instead of leaving an orphan task behind.
        self._fl_task: Optional[asyncio.Task] = None
        # Incremented whenever the simulation is rebuilt or stopped.  A round
        # that was started before the change is "stale" and must not mutate
        # the new session.
        self._generation = 0

        # --- Managed worker lifecycle -----------------------------------
        # Local training runs on a *dedicated* thread pool that this object
        # owns.  Using the asyncio default executor would make the event loop
        # shutdown wait for those threads, which is a well known source of
        # hangs in test suites.  A dedicated pool with cooperative
        # cancellation keeps the shutdown bounded and observable.
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="vanet-fl-worker")
        self._executor_shutdown = False
        # Cooperative cancellation token for the training worker.  The worker
        # checks it between clients and aborts promptly when it is set.
        self._cancel_token = threading.Event()

        self.rebuild_sim()

    # ------------------------------------------------------------------
    # Time series for the charts
    # ------------------------------------------------------------------
    #: Window (in tasks) of the "recent" completion rate and latency.
    SERIES_WINDOW = 200

    def _init_series(self):
        """Ring buffers behind the live charts.  They hold only values the
        simulation actually produced; nothing is interpolated or invented."""
        self._recent = deque(maxlen=self.SERIES_WINDOW)     # (success, latency_ms | None, action)
        self.series = deque(maxlen=720)                     # one point per simulation step
        self.round_series = deque(maxlen=200)               # one point per federated round
        self.trust_series = deque(maxlen=120)               # joint trust per vehicle, every 4 steps
        self.fl_version = 0                                 # bumped on every FL state change
        self.fl_viz = None                                  # latest round, for the viewport

    def _record_series_point(self):
        """Append the per-step point (called once per simulation step)."""
        recent = list(self._recent)
        n = len(recent)
        win_c = (sum(r[0] for r in recent) / n * 100.0) if n else None
        lats = [r[1] for r in recent if r[0] and r[1] is not None]
        total = self.total_completed + self.total_failed
        counts = [0] * 5
        for _, _, a in recent:
            counts[int(a)] += 1
        point = {
            "step": self.sim_step_count,
            "t": round(float(getattr(self.env, "current_time_sec", 0.0)), 2),
            "win_completion": round(win_c, 2) if win_c is not None else None,
            "cum_completion": round(self.total_completed / total * 100.0, 2) if total else None,
            "win_latency": round(float(np.mean(lats)), 1) if lats else None,
            "actions": counts,
            "round": self.server.current_round,
        }
        self.series.append(point)
        if self.sim_step_count % 4 == 0:
            scores = self.env.trust_engine.get_all_trust_scores()
            self.trust_series.append({
                "step": self.sim_step_count,
                "trust": {str(vid): round(float(scores.get("V%d" % vid, 0.7)), 3)
                          for vid in self.env.vehicles}})
        return point

    def series_payload(self):
        with self._lock:
            return {"series": list(self.series), "round_series": list(self.round_series),
                    "trust_series": list(self.trust_series), "window": self.SERIES_WINDOW,
                    "roles": {str(v.vehicle_id): v.role for v in self.env.vehicles.values()}}

    # ------------------------------------------------------------------
    # Managed worker lifecycle
    # ------------------------------------------------------------------
    def request_worker_cancellation(self):
        """Ask any in-flight worker to stop at the next safe point."""
        self._cancel_token.set()

    def clear_worker_cancellation(self):
        """Allow a new worker to run."""
        self._cancel_token.clear()

    def worker_cancelled(self) -> bool:
        return self._cancel_token.is_set()

    async def run_in_worker(self, func, *args):
        """Run ``func`` on the managed executor and await the result.

        If the awaiting task is cancelled the underlying thread cannot be
        interrupted, so the cancellation token is set to let the worker abort
        at its next checkpoint.
        """
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, func, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self.request_worker_cancellation()
            raise

    def shutdown_workers(self, wait: bool = False, timeout: float = 5.0):
        """Shut the worker pool down without blocking the event loop.

        Threads that are already running cannot be interrupted, so the
        cancellation token is set first; the cooperative checkpoints in the
        training phase let them finish quickly.
        """
        if self._executor_shutdown:
            return
        self._executor_shutdown = True
        self.request_worker_cancellation()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def close(self, wait: bool = True):
        """Release every resource held by this simulation instance.

        Call this when an instance is no longer needed (tests, or a
        long-running server that is being replaced) so that no worker thread
        outlives it.
        """
        self.request_worker_cancellation()
        self._cancel_fl_task()
        self.shutdown_workers(wait=wait)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def rebuild_sim(self, keep_round_history: bool = False):
        """Rebuild the simulation environment and reset the live counters.

        The federated round history is only cleared when explicitly requested;
        a new session keeps the previous rounds (they are tagged with their
        session id) so that the console still shows what happened earlier.
        Offline benchmark artefacts under ``results/`` are never touched.
        """
        with self._lock:
            # Any round that is currently in flight belongs to the previous
            # generation and must not mutate the new session.
            self._generation += 1
            # The console uses the fast task sampler (same distribution, different
            # random stream); benchmarks never do.
            live_cfg = copy.deepcopy(load_config())
            live_cfg.setdefault("task_profiles", {})["fast_sampling"] = True
            self.env = VANETOffloadingEnv(
                config=live_cfg,
                num_vehicles=self.num_vehicles,
                attack_ratio=self.attack_ratio,
                mobility_mode=self.mobility_mode,
                rsu_x_positions=(self.rsu_x_positions
                                 if self.mobility_mode == "highway" else None),
            )
            self.server = FederatedServer(
                state_dim=self.env.state_dim,
                action_dim=5,
                aggregation_strategy=self.aggregation_strategy,
                alpha=self.alpha,
                exclusion_threshold=self.exclusion_threshold,
                trust_engine=self.env.trust_engine,
            )
            self.clients = {}
            self.client_participation = {}
            for vid, v in self.env.vehicles.items():
                self.clients[vid] = FederatedClient(vehicle_id=vid, state_dim=self.env.state_dim, action_dim=5, role=v.role)
                self.client_participation[vid] = ClientParticipation(client_id=vid, role=v.role)
            self.states_dict = self.env._get_global_state_dict()
            # --- live metrics of the session are reset ---
            self.total_completed = 0
            self.total_failed = 0
            self.total_dropped = 0
            self.recent_latencies = []
            self.active_events = []
            self._init_series()
            self.steps_since_last_fl = 0
            self.fl_stage = FL_STAGE_IDLE
            self.fl_stage_detail = "Ready - press START to run the simulation"
            self.current_round_metrics = None
            self.global_model_version = 0
            self.sim_step_count = 0
            if not keep_round_history:
                self.fl_round_history = []
            self._synchronize_clients()

    def _synchronize_clients(self):
        """Distribute the server's global weights to every client."""
        global_w = self.server.get_global_weights()
        with self._model_lock:
            for client in self.clients.values():
                client.update_from_global(global_w)
        self.clients_synchronized = True

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------
    def _log_event(self, category: str, message: str, **payload):
        """Append a structured event to the backend event log."""
        entry = {
            "ts": time.time(),
            "category": category,   # task | fl_stage | fl_round | security | error | system
            "message": message,
        }
        entry.update(payload)
        self.event_log.append(entry)
        if len(self.event_log) > 300:
            del self.event_log[:100]

    def _set_stage(self, stage: str, detail: str = ""):
        """Publish a federated-lifecycle stage change (visible to subscribers).

        The stage change is pushed to the subscribers immediately instead of
        waiting for the next periodic broadcast: the ``aggregating`` and
        ``updating`` stages can each last only a few milliseconds, so a slow
        broadcast cadence would hide them completely.
        """
        self.fl_stage = stage
        self.fl_stage_detail = detail
        self.fl_stage_updated_at = time.time()
        self.fl_version += 1
        # Record the progression inside the in-progress round as well, so the
        # complete lifecycle is recoverable from the round record.
        current = self.current_round_metrics
        if current is not None and (not current.stage_timeline or current.stage_timeline[-1] != stage):
            current.stage_timeline.append(stage)
        self._log_event("fl_stage", detail or stage, stage=stage)
        self._request_immediate_broadcast()

    def _request_immediate_broadcast(self):
        """Ask the broadcaster to publish a snapshot right away."""
        loop = self._loop
        event = self._broadcast_event
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            # The loop is closed (application shutting down).
            pass

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------
    def step(self) -> Dict[str, Any]:
        """Advance the simulation by exactly one step."""
        with self._lock:
            if self.state != SIM_RUNNING:
                return self.get_state_snapshot()

            self.env.update_positions()
            self.active_events = []
            self.sim_step_count += 1

            for vid in self.env.vehicle_order():
                client = self.clients[vid]
                state, task = self.states_dict[vid]

                # Action selection reads the network weights -> guard with the model lock.
                with self._model_lock:
                    raw_action = client.agent.select_action(state, explore=(self.server.current_round < 5))
                action = int(raw_action)

                next_state, reward, done, info, next_task = self.env.step_vehicle(vid, action, task)

                # A vehicle that is absent from the trace observation window
                # produces no transition and must not train on invented data.
                if not info.get("present", True):
                    continue

                client.agent.memory.push(state, action, reward, next_state, done)
                self.states_dict[vid] = (next_state, next_task)

                # The live latency statistic is over COMPLETED tasks only: a
                # failed task has no meaningful end-to-end latency (the
                # environment books a penalty value for it).
                self._recent.append((1 if info["success"] else 0,
                                     info["delay_ms"] if info["success"] else None, int(action)))
                if info["success"]:
                    self.recent_latencies.append(info["delay_ms"])
                    if len(self.recent_latencies) > 200:
                        self.recent_latencies.pop(0)

                if info["success"]:
                    self.total_completed += 1
                else:
                    self.total_failed += 1

                # Record genuine offloading events with separated telemetry concepts.
                if action > 0 and info["target"] not in ["None", "Local", ""] and not info["target"].endswith("_Local"):
                    v = self.env.vehicles[vid]
                    ground_truth_malicious = v.role != "honest"
                    task_failed = not info["success"]

                    # Detector alert: the trust engine actually flagged the target
                    # node as below the exclusion threshold. This is evidence of a
                    # detector decision, not an inference from the ground-truth role.
                    target_id = info["target"]
                    with self._trust_lock:
                        target_trust = self.env.trust_engine.behavior_tracker.get_trust(target_id)
                    detector_alert = bool(target_trust < self.exclusion_threshold)

                    event = {
                        "vid": vid,
                        "source_pos": {"x": v.x, "y": v.y},
                        "action": action,
                        "target": info["target"],
                        "success": info["success"],
                        "delay_ms": info["delay_ms"],
                        "task_type": task.task_type,
                        "reward": info["reward"],
                        "energy_j": info.get("energy_j", 0.0),
                        "role": v.role,
                        "ground_truth_malicious": ground_truth_malicious,
                        "detector_alert": detector_alert,
                        "task_failed": task_failed,
                        "target_trust": round(float(target_trust), 3),
                    }
                    self.active_events.append(event)

            self.steps_since_last_fl += 1
            self._record_series_point()
            return self.get_state_snapshot(include_fl=False)

    # ------------------------------------------------------------------
    # Federated learning readiness
    # ------------------------------------------------------------------
    def _min_steps_for_next_round(self) -> int:
        """Warm-up applies to the first round; the interval applies afterwards."""
        if self.server.current_round == 0:
            return self.fl_warmup_steps
        return self.fl_interval_steps

    def eligible_client_ids(self) -> List[int]:
        """Clients whose replay buffer holds enough transitions to train on."""
        return [
            vid for vid, c in self.clients.items()
            if len(c.agent.memory) >= self.fl_min_buffer_size
        ]

    def fl_readiness(self) -> Dict[str, Any]:
        """Structured readiness assessment for the next automatic round."""
        eligible = self.eligible_client_ids()
        min_steps = self._min_steps_for_next_round()
        steps_ok = self.steps_since_last_fl >= min_steps
        clients_ok = len(eligible) >= self.fl_min_clients

        if not self.auto_fl_enabled:
            reason = "Automatic federated rounds are disabled"
        elif self.fl_round_in_progress:
            reason = "A federated round is already in progress"
        elif not steps_ok:
            reason = "Waiting for %d/%d simulation steps since the last round" % (
                self.steps_since_last_fl, min_steps)
        elif not clients_ok:
            reason = "Waiting for eligible clients: %d/%d have >= %d replay samples" % (
                len(eligible), self.fl_min_clients, self.fl_min_buffer_size)
        else:
            reason = "Ready to start a new federated round"

        return {
            "ready": bool(self.auto_fl_enabled and steps_ok and clients_ok and not self.fl_round_in_progress),
            "eligible_clients": len(eligible),
            "min_clients": self.fl_min_clients,
            "min_buffer_size": self.fl_min_buffer_size,
            "steps_since_last_fl": self.steps_since_last_fl,
            "min_steps": min_steps,
            "warmup_steps": self.fl_warmup_steps,
            "interval_steps": self.fl_interval_steps,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Federated round execution
    # ------------------------------------------------------------------
    async def run_fl_round(self) -> FLRoundMetrics:
        """Execute one federated round, publishing every stage as it happens.

        Returns the round metrics.  A round is only marked ``success`` when the
        minimum number of eligible clients produced *valid* updates, the
        aggregation completed and the global model was distributed.  Any failure
        is recorded honestly with the failing stage and error message.

        A request that arrives while a round is already running is rejected
        rather than queued, so two triggers can never execute overlapping
        rounds.
        """
        with self._lock:
            if self.fl_round_in_progress:
                return FLRoundMetrics(
                    round_number=self.server.current_round + 1,
                    status="failed",
                    stage=FL_STAGE_IDLE,
                    start_time=time.time(),
                    end_time=time.time(),
                    failure_stage="scheduling",
                    error_message="A federated round is already running",
                )
            self.fl_round_in_progress = True
            generation = self._generation
            # A new round starts with a clean cancellation token.
            self.clear_worker_cancellation()

        try:
            return await self._run_fl_round_locked(generation)
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            return FLRoundMetrics(
                round_number=self.server.current_round + 1,
                status="failed",
                stage=FL_STAGE_ERROR,
                start_time=time.time(),
                end_time=time.time(),
                failure_stage="internal",
                error_message="Unexpected error: %s: %s" % (type(exc).__name__, exc),
            )
        finally:
            with self._lock:
                self.fl_round_in_progress = False
    def _is_stale(self, generation: int) -> bool:
        """True when the simulation was rebuilt or stopped after the round began."""
        with self._lock:
            return self._generation != generation

    def _superseded_round(self, round_num: int, start_time: float, generation: int) -> FLRoundMetrics:
        """Build the record for a round that was overtaken by a rebuild/stop."""
        metrics = FLRoundMetrics(
            round_number=round_num,
            status="superseded",
            stage=FL_STAGE_ERROR,
            start_time=start_time,
            end_time=time.time(),
            failure_stage="lifecycle",
            error_message=(
                "The round was superseded: the simulation was rebuilt or stopped "
                "while it was running (generation %d -> %d)."
                % (generation, self._generation)
            ),
        )
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        self._log_event("fl_round", metrics.error_message, round=round_num, status="superseded")
        return metrics

    async def _run_fl_round_locked(self, generation: int) -> FLRoundMetrics:
        round_num = self.server.current_round + 1
        start_time = time.time()
        strategy = self.server.aggregation_strategy  # the strategy actually in use
        # Capture the server this round belongs to.  If the simulation is
        # rebuilt while the round runs, the aggregation and the distribution
        # must still act on *this* server and never on the new one.
        round_server = self.server

        metrics = FLRoundMetrics(
            round_number=round_num,
            status="in_progress",
            stage=FL_STAGE_WAITING,
            start_time=start_time,
            aggregation_strategy=strategy,
            global_model_version=self.global_model_version,
        )
        self.current_round_metrics = metrics
        self.fl_round_in_progress = True

        try:
            # ---- Stage: waiting for experience -------------------------
            self._set_stage(FL_STAGE_WAITING, "Checking client replay buffers")
            eligible = self.eligible_client_ids()
            metrics.eligible_clients = len(eligible)
            metrics.per_client_metrics = {
                str(vid): ClientRoundMetrics(
                    client_id=vid,
                    role=self.clients[vid].role,
                    replay_samples=len(self.clients[vid].agent.memory),
                    eligible=vid in eligible,
                ).__dict__ for vid in self.clients
            }

            if len(eligible) < self.fl_min_clients:
                metrics.status = "insufficient_experience"
                metrics.failure_stage = FL_STAGE_WAITING
                metrics.error_message = (
                    "Only %d client(s) hold >= %d replay samples; %d required."
                    % (len(eligible), self.fl_min_buffer_size, self.fl_min_clients)
                )
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_WAITING, metrics.error_message)
                self._log_event("fl_round", metrics.error_message,
                                round=round_num, status=metrics.status)
                self._finalize_round(metrics)
                return metrics

            # ---- Stage: local training ---------------------------------
            self._set_stage(FL_STAGE_TRAINING,
                            "Local training on %d client(s)" % len(eligible))
            metrics.stage = FL_STAGE_TRAINING
            with self._trust_lock:
                metrics.trust_scores_before = self.env.trust_engine.get_all_trust_scores()

            try:
                train_result = await self.run_in_worker(self._training_phase, eligible, round_num, generation)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - report honestly
                metrics.status = "failed"
                metrics.failure_stage = FL_STAGE_TRAINING
                metrics.error_message = "Local training raised %s: %s" % (type(exc).__name__, exc)
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_ERROR, metrics.error_message)
                self._log_event("error", metrics.error_message, round=round_num, stage=FL_STAGE_TRAINING)
                self._finalize_round(metrics)
                return metrics

            if train_result.get("cancelled"):
                superseded = self._superseded_round(round_num, start_time, generation)
                superseded.error_message = (
                    "The local training worker was cancelled (the simulation was "
                    "stopped or rebuilt while the round was running)."
                )
                self._set_stage(FL_STAGE_ERROR, superseded.error_message)
                self.current_round_metrics = None
                return superseded

            # Merge per-client training results into the round record.
            for vid_str, detail in train_result["per_client"].items():
                base = metrics.per_client_metrics.setdefault(vid_str, {})
                base.update(detail)
            metrics.per_client_metrics = train_result["per_client"]
            metrics.trained_clients = train_result["trained_clients"]
            metrics.submitted_updates = train_result["submitted_updates"]
            metrics.participating_clients = train_result["participating_clients"]

            # The simulation may have been rebuilt or stopped while the
            # training worker was running.  In that case the round belongs to
            # a previous generation and must not touch the new session.
            if self._is_stale(generation):
                superseded = self._superseded_round(round_num, start_time, generation)
                self._set_stage(FL_STAGE_ERROR, superseded.error_message)
                self.current_round_metrics = None
                return superseded

            # A round is only viable if enough clients produced *valid* updates.
            if train_result["valid_updates"] < self.fl_min_clients:
                metrics.status = "failed"
                metrics.failure_stage = FL_STAGE_TRAINING
                metrics.error_message = (
                    "Only %d client(s) produced valid trained updates; %d required. "
                    "Clients without a gradient step are not counted."
                    % (train_result["valid_updates"], self.fl_min_clients)
                )
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_ERROR, metrics.error_message)
                self._log_event("error", metrics.error_message, round=round_num, stage=FL_STAGE_TRAINING)
                self._finalize_round(metrics)
                return metrics

            # ---- Stage: aggregation ------------------------------------
            # Re-check immediately before the aggregation, because the
            # aggregation mutates the server's global weights.  The server
            # reference is captured so that a rebuild during the round can
            # never redirect the aggregation into the new session.
            if self._is_stale(generation):
                superseded = self._superseded_round(round_num, start_time, generation)
                self._set_stage(FL_STAGE_ERROR, superseded.error_message)
                self.current_round_metrics = None
                return superseded

            self._set_stage(FL_STAGE_AGGREGATING,
                            "Aggregating %d update(s) with strategy '%s'"
                            % (train_result["valid_updates"], strategy))
            metrics.stage = FL_STAGE_AGGREGATING
            try:
                new_global, agg_metrics = await self.run_in_worker(
                    self._aggregation_phase,
                    train_result["client_models"],
                    train_result["client_samples"],
                    round_server,
                )
            except Exception as exc:  # noqa: BLE001
                metrics.status = "failed"
                metrics.failure_stage = FL_STAGE_AGGREGATING
                metrics.error_message = "Aggregation raised %s: %s" % (type(exc).__name__, exc)
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_ERROR, metrics.error_message)
                self._log_event("error", metrics.error_message, round=round_num, stage=FL_STAGE_AGGREGATING)
                self._finalize_round(metrics)
                return metrics

            metrics.aggregation_metrics = agg_metrics

            # Record the actual inclusion / exclusion decision per client, then
            # derive the accepted / excluded counts from that decision so the
            # numbers always agree with the per-client table.
            excluded_ids = self._resolve_excluded_ids(agg_metrics, train_result["valid_client_ids"])
            accepted_ids = set(train_result["valid_client_ids"]) - excluded_ids

            for vid_str, detail in metrics.per_client_metrics.items():
                vid = int(vid_str)
                if vid in excluded_ids:
                    detail["included_in_aggregation"] = False
                    detail["exclusion_reason"] = self._exclusion_reason(agg_metrics, vid)
                elif vid in accepted_ids:
                    detail["included_in_aggregation"] = True
                    detail["exclusion_reason"] = None
                else:
                    # The client never reached the aggregation stage.
                    detail["included_in_aggregation"] = None
                    detail["exclusion_reason"] = None

            metrics.excluded_clients = len(excluded_ids)
            metrics.accepted_clients = len(accepted_ids)

            # ---- Post-aggregation acceptance check --------------------
            # A round is only viable when the aggregation actually accepted
            # enough updates AND produced a genuinely new global model.
            # Otherwise the round must fail honestly instead of distributing
            # the unchanged previous model and being counted as a success.
            produced_model = bool(agg_metrics.get("aggregation_produced_model", True))
            if not produced_model:
                metrics.status = "failed"
                metrics.failure_stage = FL_STAGE_AGGREGATING
                metrics.error_message = (
                    "The aggregation produced no new global model: %s"
                    % (agg_metrics.get("aggregation_note") or "no client update was accepted")
                )
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_ERROR, metrics.error_message)
                self._log_event("error", metrics.error_message, round=round_num, stage=FL_STAGE_AGGREGATING)
                self._finalize_round(metrics)
                return metrics

            if metrics.accepted_clients < self.fl_min_clients:
                metrics.status = "failed"
                metrics.failure_stage = FL_STAGE_AGGREGATING
                metrics.error_message = (
                    "Only %d client update(s) were accepted by the '%s' strategy; "
                    "%d required. The remaining %d client(s) were excluded."
                    % (metrics.accepted_clients, strategy, self.fl_min_clients,
                       metrics.excluded_clients)
                )
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_ERROR, metrics.error_message)
                self._log_event("error", metrics.error_message, round=round_num, stage=FL_STAGE_AGGREGATING)
                self._finalize_round(metrics)
                return metrics

            if agg_metrics.get("model_changed") is False:
                metrics.status = "failed"
                metrics.failure_stage = FL_STAGE_AGGREGATING
                metrics.error_message = (
                    "The aggregated global model is numerically identical to the "
                    "previous one; the round produced no learning effect."
                )
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_ERROR, metrics.error_message)
                self._log_event("error", metrics.error_message, round=round_num, stage=FL_STAGE_AGGREGATING)
                self._finalize_round(metrics)
                return metrics

            # ---- Stage: distributing the global model ------------------
            # Re-check immediately before mutating the shared model state.
            if self._is_stale(generation):
                superseded = self._superseded_round(round_num, start_time, generation)
                self._set_stage(FL_STAGE_ERROR, superseded.error_message)
                self.current_round_metrics = None
                return superseded

            self._set_stage(FL_STAGE_UPDATING,
                            "Distributing global model v%d to %d client(s)"
                            % (self.global_model_version + 1, len(self.clients)))
            metrics.stage = FL_STAGE_UPDATING
            try:
                await self.run_in_worker(self._distribution_phase, new_global, round_server)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                metrics.status = "failed"
                metrics.failure_stage = FL_STAGE_UPDATING
                metrics.error_message = "Model distribution raised %s: %s" % (type(exc).__name__, exc)
                metrics.end_time = time.time()
                metrics.duration_ms = (metrics.end_time - start_time) * 1000
                self._set_stage(FL_STAGE_ERROR, metrics.error_message)
                self._log_event("error", metrics.error_message, round=round_num, stage=FL_STAGE_UPDATING)
                self._finalize_round(metrics)
                return metrics

            # ---- Round complete ----------------------------------------
            self.global_model_version += 1
            metrics.global_model_version = self.global_model_version
            self.steps_since_last_fl = 0
            self.clients_synchronized = True

            with self._trust_lock:
                metrics.trust_scores_after = self.env.trust_engine.get_all_trust_scores()

            metrics.status = "success"
            metrics.stage = FL_STAGE_COMPLETE
            metrics.end_time = time.time()
            metrics.duration_ms = (metrics.end_time - start_time) * 1000
            self._set_stage(
                FL_STAGE_COMPLETE,
                "Round %d complete: %d trained, %d accepted, %d excluded (%.0f ms)"
                % (round_num, metrics.trained_clients, metrics.accepted_clients,
                   metrics.excluded_clients, metrics.duration_ms)
            )
            self._log_event(
                "fl_round",
                "Round %d completed: %d trained / %d accepted / %d excluded"
                % (round_num, metrics.trained_clients, metrics.accepted_clients, metrics.excluded_clients),
                round=round_num,
                status="success",
                duration_ms=round(metrics.duration_ms, 2),
            )
            self._finalize_round(metrics)
            return metrics

        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            metrics.status = "failed"
            metrics.failure_stage = metrics.stage
            metrics.error_message = "Unexpected error: %s: %s" % (type(exc).__name__, exc)
            metrics.end_time = time.time()
            metrics.duration_ms = (metrics.end_time - start_time) * 1000
            self._set_stage(FL_STAGE_ERROR, metrics.error_message)
            self._log_event("error", metrics.error_message, round=round_num)
            self._finalize_round(metrics)
            return metrics

    # ------------------------------------------------------------------
    # Federated round phases (run in worker threads)
    # ------------------------------------------------------------------
    def _training_phase(self, eligible: List[int], round_num: int, generation: int) -> Dict[str, Any]:
        """Perform local training on detached copies and prepare uploads.

        Runs in a worker thread.  Because the training happens on *copies* of
        the client networks, the simulation thread can keep selecting actions
        with the live networks without racing.

        The client and participation mappings are snapshotted first: the
        simulation may be rebuilt or stopped while this worker runs, and
        iterating over a dictionary that is being replaced would otherwise
        raise.
        """
        clients_snapshot = dict(self.clients)
        participation_snapshot = dict(self.client_participation)

        per_client: Dict[str, Any] = {}
        client_models: Dict[int, Any] = {}
        client_samples: Dict[int, int] = {}
        valid_client_ids: List[int] = []
        trained_clients = 0
        submitted_updates = 0
        valid_updates = 0
        cancelled = False

        global_w = self.server.get_global_weights()
        ground_truth = {vid: c.role for vid, c in clients_snapshot.items()}

        for vid in clients_snapshot:
            # Cooperative cancellation: the worker aborts at the next client
            # boundary when it has been asked to stop, or when the
            # simulation has been rebuilt underneath it.  Both checks are
            # race-free (the generation counter is monotonic).
            if self.worker_cancelled() or self._generation != generation:
                cancelled = True
                break

            client = clients_snapshot[vid]
            part = participation_snapshot.get(vid)
            detail = ClientRoundMetrics(
                client_id=vid,
                role=client.role,
                replay_samples=len(client.agent.memory),
                eligible=vid in eligible,
            )

            if vid in eligible and part is not None:
                part.rounds_participated += 1
                part.last_round_participated = round_num

                train_start = time.time()
                outcome = client.local_train_detached(epochs=self.fl_local_epochs)
                detail.training_duration_ms = round((time.time() - train_start) * 1000.0, 3)
                detail.grad_steps = int(outcome.get("grad_steps", 0))
                detail.performed_training = bool(outcome.get("trained", False))
                detail.training_loss = outcome.get("loss")
                detail.note = outcome.get("skipped_reason")
                detail.replay_samples = int(outcome.get("samples", detail.replay_samples))

                if detail.performed_training:
                    trained_clients += 1
                    part.rounds_trained += 1
                    part.total_grad_steps += detail.grad_steps
                    part.last_training_loss = detail.training_loss
                    part.last_training_duration_ms = detail.training_duration_ms

                    upload = client.get_upload_weights(global_w, local_weights=outcome.get("weights"))
                    client_models[vid] = upload
                    client_samples[vid] = max(1, client.sample_count or detail.replay_samples)
                    detail.submitted_update = True
                    submitted_updates += 1
                    valid_client_ids.append(vid)
                    valid_updates += 1
                    part.rounds_submitted += 1
                elif detail.replay_samples >= MIN_TRAIN_BATCH:
                    # The client held enough experience to train but withheld
                    # it (free riding).  Its upload is the unchanged model, and
                    # the trust engine must be able to judge that signature -
                    # exactly as in the offline benchmark.  The participation
                    # decision uses the client's available experience, never
                    # its role.
                    upload = client.get_upload_weights(global_w, local_weights=outcome.get("weights"))
                    client_models[vid] = upload
                    client_samples[vid] = max(1, client.sample_count or detail.replay_samples)
                    detail.submitted_update = True
                    submitted_updates += 1
                    valid_client_ids.append(vid)
                    valid_updates += 1
                    part.rounds_submitted += 1
                else:
                    # The client never held enough experience to train, so it
                    # could not produce an update at all: it is a
                    # NON-PARTICIPANT of this round, not a malicious updater.
                    detail.submitted_update = False
                    detail.note = detail.note or "no_valid_update"

            per_client[str(vid)] = detail.__dict__

        return {
            "per_client": per_client,
            "client_models": client_models,
            "client_samples": client_samples,
            "valid_client_ids": valid_client_ids,
            "trained_clients": trained_clients,
            "submitted_updates": submitted_updates,
            "valid_updates": valid_updates,
            "participating_clients": len(eligible),
            "ground_truth": ground_truth,
            "cancelled": cancelled,
        }

    def _aggregation_phase(self, client_models: Dict[int, Any], client_samples: Dict[int, int],
                           server=None):
        """Run the server-side aggregation (runs in a worker thread).

        The server instance is passed in explicitly so that a round which was
        started before a rebuild can never aggregate into the *new* server.
        """
        target_server = server if server is not None else self.server
        ground_truth = {vid: c.role for vid, c in self.clients.items()}
        with self._trust_lock:
            return target_server.run_aggregation_round(
                client_models, client_samples, ground_truth_roles=ground_truth
            )

    def _distribution_phase(self, new_global, server=None):
        """Push the new global model to every client (runs in a worker thread)."""
        target_server = server if server is not None else self.server
        with self._model_lock:
            for client in self.clients.values():
                client.update_from_global(new_global)
            # Keep the captured server's weights in sync with what was just
            # distributed, so a stale round cannot leave the two inconsistent.
            if target_server is not None:
                target_server.global_weights = new_global
                try:
                    target_server.global_network.load_state_dict(new_global)
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _resolve_excluded_ids(self, agg_metrics: Dict[str, Any], valid_client_ids: List[int]) -> Set[int]:
        """Determine which clients the aggregator actually excluded."""
        excluded: Set[int] = set()

        explicit = agg_metrics.get("excluded_clients")
        if explicit:
            for cid in explicit:
                excluded.add(int(cid))
            return excluded

        # The trust-aware path reports a count; recover the identities by
        # comparing against the exclusion threshold on the recorded scores.
        count = int(agg_metrics.get("excluded_count", 0) or 0)
        if count <= 0:
            return excluded

        scored = []
        for vid in valid_client_ids:
            canon = f"V{vid}"
            score = self.env.trust_engine.final_trust_scores.get(canon)
            if score is None:
                score = self.env.trust_engine.behavior_tracker.get_trust(canon)
            scored.append((float(score), vid))
        scored.sort()
        for _score, vid in scored[:count]:
            excluded.add(vid)
        return excluded

    def _exclusion_reason(self, agg_metrics: Dict[str, Any], vid: int) -> str:
        """Explain why a client was excluded, using the aggregator's own evidence."""
        canon = f"V{vid}"
        score = self.env.trust_engine.final_trust_scores.get(canon)
        if score is None:
            score = self.env.trust_engine.behavior_tracker.get_trust(canon)
        if score is not None and float(score) < self.exclusion_threshold:
            return "Joint trust %.3f below exclusion threshold %.2f" % (float(score), self.exclusion_threshold)
        if agg_metrics.get("defense_type", "").startswith("multi_krum"):
            return "Excluded by Multi-Krum (outlier distance)"
        return "Excluded by aggregation strategy"

    def _finalize_round(self, metrics: FLRoundMetrics):
        """Store the round record and reset the in-progress flag."""
        metrics.session_id = self.session_id
        self.fl_round_history.append(metrics)
        if len(self.fl_round_history) > 50:
            del self.fl_round_history[0]
        self.current_round_metrics = None
        self.fl_round_in_progress = False
        self.fl_version += 1

        # Round-level series and the payload that lets the viewport draw this
        # round's uploads and exclusions (only real decisions of this round).
        losses = [d.get("training_loss") for d in metrics.per_client_metrics.values()
                  if d.get("performed_training") and d.get("training_loss") is not None]
        agg = metrics.aggregation_metrics or {}
        trust_now = self.env.trust_engine.get_all_trust_scores()
        self.round_series.append({
            "round": metrics.round_number, "status": metrics.status,
            "loss": round(float(np.mean(losses)), 4) if losses else None,
            "trained": metrics.trained_clients, "accepted": metrics.accepted_clients,
            "excluded": metrics.excluded_clients,
            "tpr": agg.get("attack_detection_rate"), "fpr": agg.get("false_positive_rate"),
            "duration_ms": round(float(metrics.duration_ms), 1),
            "mean_trust": round(float(np.mean(list(trust_now.values()))), 3) if trust_now else None,
            "sim_step": self.sim_step_count,
        })
        self.fl_viz = {
            "round": metrics.round_number, "status": metrics.status, "sim_step": self.sim_step_count,
            "clients": [{
                "vid": int(vid), "trained": bool(d.get("performed_training")),
                "submitted": bool(d.get("submitted_update")),
                "included": d.get("included_in_aggregation"),
                "reason": d.get("exclusion_reason"),
                # ground truth, shown only as an evaluation overlay
                "ground_truth_role": d.get("role")} for vid, d in metrics.per_client_metrics.items()],
        }

        # Update the per-client participation summary.
        for vid_str, detail in metrics.per_client_metrics.items():
            vid = int(vid_str)
            part = self.client_participation.get(vid)
            if part is None:
                continue
            if detail.get("performed_training"):
                part.last_training_loss = detail.get("training_loss")
                part.last_training_duration_ms = detail.get("training_duration_ms")
            included = detail.get("included_in_aggregation")
            if included is True:
                part.rounds_included += 1
                part.inclusion_state = "included"
                part.exclusion_reason = None
            elif included is False:
                part.rounds_excluded += 1
                part.inclusion_state = "excluded"
                part.exclusion_reason = detail.get("exclusion_reason")
            elif detail.get("eligible"):
                # The round failed before the aggregation decision was made.
                part.inclusion_state = "not_in_aggregation"
                part.exclusion_reason = None
            else:
                part.inclusion_state = "not_participated"

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------
    def get_fl_block(self) -> Dict[str, Any]:
        """The heavy federated-learning part of the snapshot.

        It only changes when a round or a stage changes (``fl_version``), so the
        broadcaster sends it once per change instead of twice per second.
        """
        with self._lock:
            current_round = asdict(self.current_round_metrics) if self.current_round_metrics else None
            last_round = asdict(self.fl_round_history[-1]) if self.fl_round_history else None
            return {
                "fl_version": self.fl_version,
                "fl_rounds": [asdict(r) for r in self.fl_round_history[-10:]],
                "current_round": current_round,
                "last_round": last_round,
                "fl_readiness": self.fl_readiness(),
                "client_participation": {str(k): asdict(v) for k, v in self.client_participation.items()},
            }

    def get_state_snapshot(self, include_fl: bool = True) -> Dict[str, Any]:
        """Thread-safe snapshot consumed by the REST API and the broadcaster.

        ``include_fl=False`` omits the heavy federated block (see
        :meth:`get_fl_block`); REST callers always get the full snapshot.
        """
        with self._lock:
            trust_scores = self.env.trust_engine.get_all_trust_scores()

            v_data = []
            for vid, v in self.env.vehicles.items():
                v_dict = v.to_dict()
                node_key = f"V{vid}"
                score = trust_scores.get(node_key, getattr(v, "trust_score", 0.5))
                v_dict["trust_score"] = round(float(score), 2)
                v_dict["connected_rsus"] = self.env.connected_rsus(v)
                v_data.append(v_dict)

            rsu_data = [r.to_dict() for r in self.env.rsus]
            for r in rsu_data:
                r["trust_score"] = trust_scores.get(r["rsu_id"], 0.5)

            avg_lat = float(np.mean(self.recent_latencies[-50:])) if self.recent_latencies else None
            total = self.total_completed + self.total_failed
            comp_rate = (self.total_completed / total * 100.0) if total > 0 else None

            readiness = self.fl_readiness()
            elapsed = (time.time() - self.started_at) if self.started_at else 0.0

            win_recent = list(self._recent)
            win_comp = (sum(r[0] for r in win_recent) / len(win_recent) * 100.0) if win_recent else None
            action_mix = [0] * 5
            for _, _, a_ in win_recent:
                action_mix[int(a_)] += 1
            epsilons = [c.agent.epsilon for c in self.clients.values()]

            snap = {
                "vehicles": v_data,
                "rsus": rsu_data,
                "scene": self.env.scene_description(),
                "traffic_lights": {
                    "phase": getattr(self.env.mobility, "light_phase", 0),
                    "timer": round(getattr(self.env.mobility, "light_timer", 0.0), 1),
                    "coords": getattr(self.env.mobility, "coords", [-300.0, -100.0, 100.0, 300.0]),
                },
                "events": self.active_events[-15:],
                "event_log": self.event_log[-60:],
                "kpis": {
                    # Authoritative simulation state - never inferred by the client.
                    "is_running": bool(self.is_running),
                    "sim_state": self.state,
                    "session_id": self.session_id,
                    "session_sequence": self.session_sequence,
                    "session_started_at": self.session_started_at,
                    "session_started_at_iso": (
                        datetime.fromtimestamp(self.session_started_at, timezone.utc).isoformat()
                        if self.session_started_at else None
                    ),
                    "last_error": self.last_error,
                    "sim_steps": self.sim_step_count,
                    "elapsed_wall_sec": round(elapsed, 1),
                    "avg_latency_ms": round(avg_lat, 1) if avg_lat is not None else None,
                    "avg_latency_basis": "completed tasks, last 200",
                    "completion_rate": round(comp_rate, 1) if comp_rate is not None else None,
                    "completion_rate_window": round(win_comp, 1) if win_comp is not None else None,
                    "window_tasks": len(win_recent),
                    "action_mix": action_mix,
                    "mean_epsilon": round(float(np.mean(epsilons)), 3) if epsilons else None,
                    "exploring": bool(self.server.current_round < 5),
                    "total_tasks": total,
                    "completed_tasks": self.total_completed,
                    "failed_tasks": self.total_failed,
                    "dropped_tasks": self.total_dropped,
                    "fl_round": self.server.current_round,
                    "fl_rounds_completed": len([
                        r for r in self.fl_round_history
                        if r.status == "success" and r.session_id == self.session_id
                    ]),
                    "fl_rounds_failed": len([
                        r for r in self.fl_round_history
                        if r.status == "failed" and r.session_id == self.session_id
                    ]),
                    "fl_rounds_total_completed": len([
                        r for r in self.fl_round_history if r.status == "success"
                    ]),
                    "active_vehicles": len(self.env.vehicles),
                    "sim_speed": self.sim_speed,
                    "configured_vehicles": self.num_vehicles,
                    "configured_attack_ratio": self.attack_ratio,
                    "attackers": sum(1 for v in self.env.vehicles.values() if v.role != "honest"),
                    "strategy": self.server.aggregation_strategy,
                    "fl_stage": self.fl_stage,
                    "fl_stage_detail": self.fl_stage_detail,
                    "fl_stage_age_sec": round(time.time() - self.fl_stage_updated_at, 2),
                    "steps_since_fl": self.steps_since_last_fl,
                    "warmup_steps": self.fl_warmup_steps,
                    "fl_interval_steps": self.fl_interval_steps,
                    "auto_fl_enabled": self.auto_fl_enabled,
                    "next_round_ready": readiness["ready"],
                    "next_round_reason": readiness["reason"],
                    "eligible_clients": readiness["eligible_clients"],
                    "min_clients": self.fl_min_clients,
                    "min_buffer_size": self.fl_min_buffer_size,
                    "total_clients": len(self.clients),
                    "global_model_version": self.global_model_version,
                    "clients_synchronized": self.clients_synchronized,
                    "simulation_time_sec": getattr(self.env, "current_time_sec", 0.0),
                    "mobility_mode": self.mobility_mode,
                    "mobility_source": self.describe_mobility(),
                },
                "trust_scores": trust_scores,
                "fl_version": self.fl_version,
                "fl_viz": self.fl_viz,
                "point": self.series[-1] if self.series else None,
            }
            if include_fl:
                snap.update(self.get_fl_block())
            return snap

    def describe_mobility(self) -> Dict[str, Any]:
        """Describe the mobility source actually in use (never over-claim)."""
        mode = self.mobility_mode
        if mode == "highway":
            return {
                "mode": "highway",
                "label": "Synthetic Highway (IDM car-following)",
                "dataset": "none (procedural)",
                "authentic": False,
                "note": "Vehicles are simulated with the Intelligent Driver Model.",
            }

        trace_file = getattr(self.env.mobility, "trace_file", "") or ""
        normalised = trace_file.replace("\\", "/")
        geom = getattr(self.env, "trace_geometry", None)

        if "authentic" in normalised or "source/us101_raw" in normalised:
            label = "Authentic FHWA NGSIM US-101 trace replay"
            note = "Replaying the processed official US-101 trajectory export."
        elif "synthetic_traces" in normalised:
            label = "Calibrated synthetic NGSIM-style trace replay"
            note = ("Replaying a synthetic trace calibrated against NGSIM statistics. "
                    "This is NOT the authentic FHWA dataset.")
        else:
            label = "Trace replay"
            note = "Replaying a trajectory trace."

        result = {
            "mode": mode,
            "label": label,
            "dataset": trace_file,
            "authentic": label.startswith("Authentic"),
            "note": note,
        }
        if geom:
            result["geometry"] = geom
        return result

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _number(value, name: str) -> float:
        """Parse a finite number or reject the request with a clear message."""
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{name} must be a number")
        if x != x or x in (float("inf"), float("-inf")):
            raise HTTPException(400, f"{name} must be finite")
        return x

    def set_config(self, config: dict) -> Dict[str, Any]:
        """Update configuration with validation.  Returns the applied values."""
        with self._lock:
            needs_rebuild = False
            updated: Dict[str, Any] = {}

            # ---- Parameters that define a NEW experiment (rebuild the session) ----
            if "attack_ratio" in config:
                ratio = self._number(config["attack_ratio"], "attack_ratio")
                if not 0.0 <= ratio <= 0.5:
                    raise HTTPException(400, "attack_ratio must be within [0.0, 0.5]; "
                                             "above 0.5 the population-relative detector has no honest majority")
                self.attack_ratio = ratio
                updated["attack_ratio"] = self.attack_ratio
                needs_rebuild = True

            if "num_vehicles" in config:
                n = int(self._number(config["num_vehicles"], "num_vehicles"))
                if not 5 <= n <= 60:
                    raise HTTPException(400, "num_vehicles must be within [5, 60]")
                self.num_vehicles = n
                updated["num_vehicles"] = n
                needs_rebuild = True

            if "rsu_x_positions" in config:
                pos = config["rsu_x_positions"]
                if pos is None:
                    self.rsu_x_positions = None
                else:
                    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                        raise HTTPException(400, "rsu_x_positions must be a list of 3 numbers (metres)")
                    pos = [self._number(x, "rsu_x_positions") for x in pos]
                    if config.get("mobility_mode", self.mobility_mode) != "highway":
                        raise HTTPException(400, "RSU positions can only be set for the procedural highway; "
                                                 "trace replay fixes the RSUs inside the observed road segment")
                    if any(x < 0 or x > 3200 for x in pos):
                        raise HTTPException(400, "RSU positions must lie within [0, 3200] m")
                    self.rsu_x_positions = pos
                updated["rsu_x_positions"] = self.rsu_x_positions
                needs_rebuild = True

            if "aggregation_strategy" in config:
                strategy = str(config["aggregation_strategy"])
                valid = ["trust_aware", "fedavg", "coordinate_median", "trimmed_mean",
                         "multi_krum", "exclusion_no_weighting", "trust_no_exclusion"]
                if strategy not in valid:
                    raise HTTPException(400, f"Invalid strategy. Must be one of: {valid}")
                self.aggregation_strategy = strategy
                # Apply to the *actual* aggregation implementation.
                self.server.aggregation_strategy = strategy
                updated["aggregation_strategy"] = strategy

            if "alpha" in config:
                a_val = self._number(config["alpha"], "alpha")
                if not 0.0 <= a_val <= 1.0:
                    raise HTTPException(400, "alpha must be within [0, 1]")
                self.alpha = a_val
                self.server.trust_engine.alpha = self.alpha
                updated["alpha"] = self.alpha

            if "exclusion_threshold" in config:
                tau_val = self._number(config["exclusion_threshold"], "exclusion_threshold")
                if not 0.05 <= tau_val <= 0.95:
                    raise HTTPException(400, "exclusion_threshold must be within [0.05, 0.95]")
                self.exclusion_threshold = tau_val
                self.server.trust_engine.exclusion_threshold = self.exclusion_threshold
                updated["exclusion_threshold"] = self.exclusion_threshold

            if "mobility_mode" in config:
                mode = str(config["mobility_mode"])
                if mode not in ["highway", "trace", "authentic_trace"]:
                    raise HTTPException(400, "mobility_mode must be 'highway', 'trace' or 'authentic_trace'")
                self.mobility_mode = mode
                updated["mobility_mode"] = mode
                needs_rebuild = True

            if "sim_speed" in config:
                speed = self._number(config["sim_speed"], "sim_speed")
                if not 0.1 <= speed <= 20.0:
                    raise HTTPException(400, "sim_speed must be within [0.1, 20]")
                self.sim_speed = speed
                updated["sim_speed"] = self.sim_speed

            if "auto_fl_enabled" in config:
                self.auto_fl_enabled = bool(config["auto_fl_enabled"])
                updated["auto_fl_enabled"] = self.auto_fl_enabled

            if "fl_warmup_steps" in config:
                self.fl_warmup_steps = max(1, int(config["fl_warmup_steps"]))
                updated["fl_warmup_steps"] = self.fl_warmup_steps

            if "fl_interval_steps" in config:
                self.fl_interval_steps = max(1, int(config["fl_interval_steps"]))
                updated["fl_interval_steps"] = self.fl_interval_steps

            if "fl_min_buffer_size" in config:
                self.fl_min_buffer_size = max(
                    int(config["fl_min_buffer_size"]), DQNAgent.MIN_TRAIN_BATCH
                )
                updated["fl_min_buffer_size"] = self.fl_min_buffer_size

            if "fl_min_clients" in config:
                self.fl_min_clients = max(1, int(config["fl_min_clients"]))
                updated["fl_min_clients"] = self.fl_min_clients

            if config.get("rebuild", False) or needs_rebuild:
                self.rebuild_sim()
                updated["rebuilt"] = True

            live = [k for k in updated if k not in ("rebuilt",)]
            return {"status": "updated", "config": updated,
                    "applied_live": [k for k in live if k not in
                                     ("attack_ratio", "num_vehicles", "rsu_x_positions", "mobility_mode")],
                    "new_session_started": bool(updated.get("rebuilt"))}

    # ------------------------------------------------------------------
    # Control (explicit state machine)
    # ------------------------------------------------------------------
    def control(self, action: str) -> Dict[str, Any]:
        """Start / pause / resume / stop the simulation.

        The simulation has an explicit state machine:

        ``STOPPED`` -> ``RUNNING`` -> ``PAUSED`` -> ``RUNNING``
        ``RUNNING`` / ``PAUSED`` -> ``STOPPED``  (ends the session)
        any state            -> ``ERROR``       (background failure)

        Repeated START requests are idempotent and never create a second
        simulation loop: the loop is owned by the application lifespan.
        """
        action = str(action).lower().strip()
        with self._lock:
            if action == "start":
                if self.state == SIM_RUNNING:
                    # Idempotent: the loop already exists.
                    self._log_event("system", "START ignored - already running")
                    return self._control_result("already_running")
                if self.state == SIM_PAUSED:
                    # START from PAUSED continues the existing session.
                    self.state = SIM_RUNNING
                    self.is_running = True
                    self._log_event("system", "Simulation resumed via START")
                    return self._control_result("resumed")
                # STOPPED -> new session
                self._start_new_session()
                self._log_event("system", "Simulation started (new session %s)" % self.session_id)
                return self._control_result("started")

            if action == "resume":
                if self.state == SIM_RUNNING:
                    self._log_event("system", "RESUME ignored - already running")
                    return self._control_result("already_running")
                if self.state != SIM_PAUSED:
                    raise HTTPException(
                        409,
                        "Cannot RESUME from state '%s'. Use START to begin a new session." % self.state,
                    )
                self.state = SIM_RUNNING
                self.is_running = True
                self._log_event("system", "Simulation resumed")
                return self._control_result("resumed")

            if action == "pause":
                if self.state != SIM_RUNNING:
                    raise HTTPException(
                        409,
                        "Cannot PAUSE from state '%s'. Only a RUNNING simulation can be paused." % self.state,
                    )
                self.state = SIM_PAUSED
                self.is_running = False
                self._log_event("system", "Simulation paused")
                return self._control_result("paused")

            if action == "stop":
                self.state = SIM_STOPPED
                self.is_running = False
                # Cancel any in-flight round and ask its worker to abort at the
                # next checkpoint, so that a stopped session cannot be
                # modified by a round that is still finishing.
                self.request_worker_cancellation()
                self._cancel_fl_task()
                self._end_session()
                self._log_event("system", "Simulation stopped")
                return self._control_result("stopped")

            raise HTTPException(
                400,
                "Unknown action '%s'. Use 'start', 'pause', 'resume' or 'stop'." % action,
            )

    def _start_new_session(self):
        """Begin a fresh session: reset live state and start the clock."""
        self.session_sequence += 1
        self.session_id = "sess-%d-%s" % (self.session_sequence, uuid.uuid4().hex[:8])
        self.last_error = None
        # Live simulation state is reset for the new session; the federated
        # round history of previous sessions is kept (each round is tagged with
        # its session id) and the offline benchmark artefacts are never touched.
        self.rebuild_sim(keep_round_history=True)
        self.session_started_at = time.time()
        self.started_at = self.session_started_at
        self.state = SIM_RUNNING
        self.is_running = True

    def _end_session(self):
        """Stop the session and reset the live counters."""
        self.started_at = None
        self.session_started_at = None
        # The session identifier is cleared because no session is active any
        # more.  The rounds that were executed during it keep their session
        # tag in the history.
        self.session_id = None
        self.rebuild_sim(keep_round_history=True)

    def _control_result(self, status: str) -> Dict[str, Any]:
        return {
            "status": status,
            "state": self.state,
            "is_running": self.is_running,
            "session_id": self.session_id,
            "session_started_at": self.session_started_at,
        }

    def mark_error(self, message: str):
        """Record a background failure and surface it in the console."""
        with self._lock:
            self.state = SIM_ERROR
            self.is_running = False
            self.last_error = str(message)
            self._log_event("error", "Simulation entered ERROR state: %s" % message)

    def _cancel_fl_task(self):
        """Cancel the in-flight federated round task, if any.

        The task is cancelled but not awaited here: the awaiting happens in
        the event loop, which owns the task.  The worker thread is asked to
        abort cooperatively at the same time.
        """
        task = self._fl_task
        if task is not None and not task.done():
            task.cancel()
        self._fl_task = None


    # ------------------------------------------------------------------
    # Subscriber registry (WebSocket broadcast-only clients)
    # ------------------------------------------------------------------
    def add_subscriber(self, ws: WebSocket):
        with self._subscribers_lock:
            self._subscribers.add(ws)

    def remove_subscriber(self, ws: WebSocket):
        with self._subscribers_lock:
            self._subscribers.discard(ws)
        self._ws_versions.pop(id(ws), None)

    def subscriber_count(self) -> int:
        with self._subscribers_lock:
            return len(self._subscribers)


# ----------------------------------------------------------------------
# Module level singletons
# ----------------------------------------------------------------------
sim = SimulationState()


async def _simulation_loop(state: "SimulationState"):
    """THE single simulation timeline.

    Advances the simulation and schedules automatic federated rounds.  Nothing
    else in the application may advance simulation time.  Any unexpected
    failure moves the simulation into the ERROR state so that the dashboard can
    surface it instead of appearing frozen.
    """
    consecutive_errors = 0
    while True:
        try:
            if state.state == SIM_RUNNING:
                state.step()

                # Schedule an automatic federated round when the requirements
                # are met.  The task handle is retained so that the
                # application lifecycle can cancel and await it on shutdown,
                # and so that a second round cannot be started while one is
                # in flight.
                readiness = state.fl_readiness()
                if readiness["ready"] and not state.fl_round_in_progress:
                    if state._fl_task is None or state._fl_task.done():
                        state._fl_task = asyncio.create_task(
                            state.run_fl_round(), name="vanet-fl-round")
                    else:
                        state._log_event(
                            "fl_round",
                            "A federated round is already in flight; not starting another one.")

            consecutive_errors = 0
            interval = 0.5 / max(0.2, state.sim_speed)
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            message = "Simulation loop error: %s: %s" % (type(exc).__name__, exc)
            state._log_event("error", message)
            if consecutive_errors >= 3:
                # Repeated failures: stop pretending the simulation is alive.
                state.mark_error(message)
                consecutive_errors = 0
            await asyncio.sleep(0.5)


async def _broadcast_loop(state: "SimulationState"):
    """Push state snapshots to all subscribers.

    The loop waits on an event so that a federated-lifecycle stage change is
    published immediately, and otherwise falls back to the periodic cadence.
    """
    while True:
        try:
            if state.subscriber_count() > 0:
                payload = state.get_state_snapshot(include_fl=False)
                fl_block = None
                dead = []
                for ws in list(state._subscribers):
                    try:
                        frame = payload
                        if state._ws_versions.get(id(ws)) != state.fl_version:
                            # This subscriber has not seen the current federated
                            # state yet: send the heavy block once, then only
                            # lightweight frames until it changes again.
                            if fl_block is None:
                                fl_block = state.get_fl_block()
                            frame = dict(payload, **fl_block)
                            state._ws_versions[id(ws)] = fl_block["fl_version"]
                        await ws.send_json(frame)
                    except Exception:  # noqa: BLE001
                        dead.append(ws)
                for ws in dead:
                    state.remove_subscriber(ws)

            # Wait for the next periodic tick or an immediate stage change.
            timeout = 0.5 / max(0.2, state.sim_speed)
            event = state._broadcast_event
            if event is not None:
                try:
                    await asyncio.wait_for(event.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if event is not None:
                        event.clear()
            else:
                await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the single simulation loop and the broadcaster.

    The background tasks are owned by the application lifecycle: exactly one
    simulation loop and one broadcaster are created here, so repeated START
    requests can never spawn a second loop and a browser reconnect cannot
    reset the simulation.

    The worker pool that runs local training is shut down here as well, so
    that no thread survives the application.
    """
    state: "SimulationState" = app.state.sim
    state._loop = asyncio.get_running_loop()
    state._broadcast_event = asyncio.Event()
    sim_task = None
    broadcast_task = None
    try:
        sim_task = asyncio.create_task(_simulation_loop(state), name="vanet-simulation-loop")
        broadcast_task = asyncio.create_task(_broadcast_loop(state), name="vanet-broadcaster")
        print("VANET simulation console started (single authoritative simulation loop).")
    except Exception as exc:  # noqa: BLE001
        state.mark_error("Failed to start background tasks: %s: %s" % (type(exc).__name__, exc))
        raise
    try:
        yield
    finally:
        # Cancel and await every background task, including a federated round
        # that may still be training.
        state.request_worker_cancellation()
        state._cancel_fl_task()
        for task in (state._fl_task, sim_task, broadcast_task):
            if task is not None:
                task.cancel()
        for task in (state._fl_task, sim_task, broadcast_task):
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        state._fl_task = None
        # Shut the dedicated worker pool down.  ``wait`` is False so the
        # event loop is never blocked by a running thread; the cooperative
        # cancellation token lets the worker finish promptly.
        state.shutdown_workers(wait=False)
        state.state = SIM_STOPPED
        state.is_running = False
        print("VANET simulation console stopped.")


# ----------------------------------------------------------------------
# Publication figures
# ----------------------------------------------------------------------
# The dashboard may only ever show figures that belong to the *validated,
# active* experiment run.  The figure manifest is the single source of
# truth: a file that is not listed there is not served, and no directory
# scan or wildcard is used, so a historical figure can never enter the
# active presentation.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _active_run_entry(results_dir: str):
    """The manifest entry of the active, validated run (or ``None``)."""
    manifest_path = os.path.join(results_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return None
    candidates = [e for e in (entries or [])
                  if isinstance(e, dict)
                  and e.get("published_as_active")
                  and not e.get("superseded")
                  and e.get("validation_status") == "passed"]
    if not candidates:
        return None
    return candidates[-1]


def _load_figure_manifest(results_dir: Optional[str] = None):
    """``(manifest, plots_dir)`` of the active run, or ``(None, None)``.

    The active run's own figure directory is preferred; the top-level
    ``plots/`` directory is the fallback for a repository that has not
    produced a versioned run yet.
    """
    results_dir = results_dir or os.path.join(_PROJECT_ROOT, "results")
    candidates = []
    entry = _active_run_entry(results_dir)
    if entry:
        run_dir = entry.get("directory") or ""
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(_PROJECT_ROOT, run_dir)
        candidates.append(os.path.join(run_dir, "plots", "figures_manifest.json"))
    candidates.append(os.path.join(_PROJECT_ROOT, "plots", "figures_manifest.json"))
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        return manifest, os.path.dirname(path)
    return None, None


# ----------------------------------------------------------------------
# Research summary (compact view of the active run)
# ----------------------------------------------------------------------
_SUMMARY_CACHE: Dict[str, Any] = {}


def _strip_raw(node):
    """Recursively drop the bulky per-seed raw records."""
    if isinstance(node, dict):
        return {k: _strip_raw(v) for k, v in node.items()
                if k not in ("per_seed_results", "evaluation_records", "training_rounds")}
    return node


def _supplementary_runs(results_dir: str, parent_run_id: str):
    """Validated supplementary runs that supplement the active run (read-only)."""
    manifest_path = os.path.join(results_dir, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for e in entries or []:
        if (isinstance(e, dict) and e.get("kind") == "supplementary"
                and e.get("supplements") == parent_run_id and e.get("validation_status") == "passed"):
            d = e.get("directory") or ""
            if not os.path.isabs(d):
                d = os.path.join(_PROJECT_ROOT, d)
            if os.path.isfile(os.path.join(d, "benchmark_results.json")) and os.path.isfile(os.path.join(d, "metadata.json")):
                out.append((e, d))
    return out


def _per_seed(cell, key="completion_rate"):
    return {str(r["seed"]): r.get(key) for r in cell.get("per_seed_results", [])}


def _derived(results):
    """Compact series computed from the raw records (kept out of the stripped copy)."""
    out = {}
    A = results.get("experiment_a_baseline_comparison") or {}
    out["per_seed_completion"] = {m: _per_seed(v) for m, v in A.items() if isinstance(v, dict) and "per_seed_results" in v}
    P = A.get("Proposed_TFRL")
    if P:
        n_rounds = max((len(r.get("training_rounds", [])) for r in P["per_seed_results"]), default=0)
        fp = [0] * n_rounds
        hon = [0] * n_rounds
        loss = [[] for _ in range(n_rounds)]
        for r in P["per_seed_results"]:
            for i, t in enumerate(r.get("training_rounds", [])):
                fp[i] += t.get("false_positives") or 0
                hon[i] += (t.get("false_positives") or 0) + (t.get("true_negatives") or 0)
                if t.get("mean_train_loss") is not None:
                    loss[i].append(t["mean_train_loss"])
        out["proposed_false_exclusion_by_round"] = [round(100.0 * f / h, 2) if h else None for f, h in zip(fp, hon)]
        out["proposed_train_loss_by_round"] = [round(float(np.mean(x)), 4) if x else None for x in loss]
    return out


def _supplementary_claims(sres, parent_results):
    """Paired claims of the supplementary run (same rule and Holm correction as the headline claims)."""
    from experiments.claims_report import paired_cells, holm, verdict
    L = sres.get("experiment_l_faulty_edge_node") or {}
    M = sres.get("experiment_m_intermittent_connectivity") or {}
    A = parent_results.get("experiment_a_baseline_comparison") or {}
    spec = [
        ("S1", "Behavioural trust protects offloading from a faulty edge node (policy with trust features minus the same policy with them masked)",
         L.get("learned_with_trust_features"), L.get("learned_trust_features_masked"), +1),
        ("S2", "...and the learned policy beats Nearest-RSU when one RSU is faulty",
         L.get("learned_with_trust_features"), L.get("Nearest_RSU_faulty"), +1),
        ("S3", "...and beats the greedy heuristic that also uses behavioural trust",
         L.get("learned_with_trust_features"), L.get("Trust_Greedy_faulty"), +1),
        ("S4", "A faulty edge node lowers the learned policy's completion (vs the no-fault main-run cell, same seeds)",
         L.get("learned_with_trust_features"), A.get("Proposed_TFRL"), -1),
        ("S5", "A staleness discount recovers completion under intermittent connectivity",
         M.get("trust_aware_staleness_discount"), M.get("trust_aware_no_staleness_handling"), +1),
        ("S6", "Trust-aware aggregation beats FedAvg under intermittent connectivity",
         M.get("trust_aware_no_staleness_handling"), M.get("fedavg_stale_uploads"), +1),
    ]
    rows = []
    for cid, text, a, b, direction in spec:
        if not a or not b or "per_seed_results" not in a or "per_seed_results" not in b:
            continue
        res = paired_cells(a, b)
        rows.append({"id": cid, "text": text, "res": res, "direction": direction})
    adj = holm([r["res"]["p_value"] for r in rows])
    out = []
    for r, ph in zip(rows, adj):
        v = verdict(r["res"], r["direction"])
        if v == "SUPPORTED" and ph is not None and ph >= 0.05:
            v = "NOT ESTABLISHED (fails Holm correction)"
        out.append({"id": r["id"], "text": r["text"], "verdict": v, "mean_difference": r["res"]["mean_difference"],
                    "ci95_low": r["res"]["ci95_low"], "ci95_high": r["res"]["ci95_high"], "n": r["res"]["n"],
                    "p_value": r["res"]["p_value"], "p_holm": ph, "mde_80": r["res"].get("mde_80")})
    return out


def _unreliable_summary(sres):
    """Detection counts and the exclusion rate of unreliable-but-honest vehicles (K2)."""
    out = {}
    for name, cell in (sres.get("experiment_k2_mixed_threat_relabelled") or {}).items():
        if "per_seed_results" not in cell:
            continue
        ut = ue = 0
        for r in cell["per_seed_results"]:
            for t in r.get("training_rounds", []):
                ut += t.get("unreliable_total") or 0
                ue += t.get("unreliable_excluded") or 0
        out[name] = {"unreliable_decisions": ut, "unreliable_excluded": ue,
                     "confusion": cell.get("pooled_confusion_counts")}
    return out


def _research_summary() -> Dict[str, Any]:
    import hashlib
    results_dir = os.path.join(_PROJECT_ROOT, "results")
    entry = _active_run_entry(results_dir)
    if entry is None:
        return {"status": "not_ready", "reason": "no validated active run in results/manifest.json"}
    run_dir = entry.get("directory") or ""
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(_PROJECT_ROOT, run_dir)
    res_path = os.path.join(run_dir, "benchmark_results.json")
    meta_path = os.path.join(run_dir, "metadata.json")
    if not (os.path.isfile(res_path) and os.path.isfile(meta_path)):
        return {"status": "not_ready", "reason": "active run artefacts are missing"}
    supp = _supplementary_runs(results_dir, entry.get("run_id"))
    key = (res_path, os.path.getmtime(res_path), os.path.getmtime(meta_path),
           tuple((d, os.path.getmtime(os.path.join(d, "benchmark_results.json"))) for _, d in supp))
    if _SUMMARY_CACHE.get("key") == key:
        return _SUMMARY_CACHE["value"]
    with open(res_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    claims = None
    try:
        # The verdicts are computed by the same code that writes the claims
        # report, from the raw per-seed records, so the console cannot drift.
        from experiments.claims_report import _headline_claims, VERDICT_RULE
        rows = _headline_claims(results["experiment_a_baseline_comparison"])
        claims = {
            "rule": VERDICT_RULE,
            "rows": [{
                "id": r["id"], "text": r["text"], "note": r["note"], "verdict": r["verdict"],
                "mean_difference": r["res"]["mean_difference"], "ci95_low": r["res"]["ci95_low"],
                "ci95_high": r["res"]["ci95_high"], "n": r["res"]["n"], "p_value": r["res"]["p_value"],
                "p_holm": r["p_holm"], "mde_80": r["res"].get("mde_80")} for r in rows]}
    except Exception as exc:  # noqa: BLE001 - reported to the client, never hidden
        claims = {"error": "%s: %s" % (type(exc).__name__, exc)}
    supplementary = []
    for e, d in supp:
        with open(os.path.join(d, "benchmark_results.json"), "r", encoding="utf-8") as f:
            sres = json.load(f)
        with open(os.path.join(d, "metadata.json"), "r", encoding="utf-8") as f:
            smeta = json.load(f)
        supplementary.append({
            "run_id": e["run_id"], "metadata": {k: smeta.get(k) for k in (
                "run_id", "timestamp_utc", "seeds", "evaluation_config", "training_protocol", "faulty_rsu",
                "connectivity", "limitations", "reference_cells", "validation_status", "supplements")},
            "results": _strip_raw(sres),
            "claims": _supplementary_claims(sres, results),
            "unreliable_summary": _unreliable_summary(sres),
            "per_seed_completion": {g: {n: _per_seed(c) for n, c in grp.items() if isinstance(c, dict) and "per_seed_results" in c}
                                    for g, grp in sres.items()},
            "parent_per_seed_completion": _derived(results)["per_seed_completion"]})
    value = {
        "status": "ok",
        "etag": hashlib.sha256(repr(key).encode()).hexdigest()[:16],
        "claims": claims,
        "run_id": entry.get("run_id"),
        "metadata": {k: metadata.get(k) for k in (
            "run_id", "run_kind", "timestamp_utc", "seeds", "sweep_seeds",
            "evaluation_config", "dataset_provenance", "code_revision",
            "environment_versions", "attack_settings", "provenance_disclosure",
            "validation_status", "validation_problems", "environment_model",
            "training_protocol", "limitations")},
        "derived": _derived(results),
        "supplementary": supplementary,
        "results": _strip_raw(results),
    }
    _SUMMARY_CACHE["key"] = key
    _SUMMARY_CACHE["value"] = value
    return value


def create_app(state: Optional["SimulationState"] = None) -> FastAPI:
    """Create the console application.

    A dedicated :class:`SimulationState` can be supplied so that tests (and
    any future multi-tenant deployment) get a fully isolated simulation
    instead of sharing the module-level instance.
    """
    application = FastAPI(
        title="Trust-Aware VANET Simulation & Research Console",
        lifespan=lifespan,
    )
    application.state.sim = state if state is not None else SimulationState()

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.add_middleware(GZipMiddleware, minimum_size=1024)

    @application.middleware("http")
    async def cache_headers(request: Request, call_next):
        """Caching policy.

        * ``/static`` files are requested with a version query string, so they may
          be cached for an hour; a new version string forces a refetch.
        * ``/api/plots`` figures are cached but must be revalidated (their name
          can point at a different run later).
        * the research summary is revalidated with an ETag (304 when unchanged).
        * live state, control and configuration are never cached.
        """
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") and "v=" in (request.url.query or ""):
            response.headers["Cache-Control"] = "public, max-age=3600"
        elif path.startswith("/api/plots/"):
            response.headers.setdefault("Cache-Control", "public, max-age=300, must-revalidate")
        elif path.startswith("/api/") and "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    application.mount("/static", StaticFiles(directory=static_dir), name="static")
    _register_routes(application)
    return application


static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)


def _register_routes(application: FastAPI):
    """Register the console routes against the application-scoped simulation."""

    def _sim() -> "SimulationState":
        return application.state.sim

    @application.get("/health")
    def get_health():
        state = _sim()
        return {"status": "ok", "state": state.state, "is_running": state.is_running}

    @application.get("/")
    def get_index():
        """Serve the console shell.

        ``Cache-Control: no-store`` guarantees that a browser always receives
        the current markup (and therefore the current asset version query
        strings) instead of a stale cached copy that would reference old
        JavaScript.
        """
        response = FileResponse(os.path.join(static_dir, "index.html"))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @application.get("/api/state")
    def get_state():
        return _sim().get_state_snapshot()

    @application.post("/api/control")
    def control_simulation(action: dict):
        """Unified control endpoint: start / pause / resume / stop.

        Response contract:
            status  - "started" | "resumed" | "paused" | "stopped" | "already_running"
            state   - the authoritative lifecycle state (STOPPED/RUNNING/PAUSED/ERROR)
            kpis    - the full KPI snapshot for the current session
            session - the session identifier

        Invalid transitions produce a 409 with an explanatory message rather
        than a silent no-op.
        """
        act = str(action.get("action", "")).strip()
        if not act:
            raise HTTPException(400, "Missing 'action'. Use 'start', 'pause', 'resume' or 'stop'.")
        state = _sim()
        result = state.control(act)
        snapshot = state.get_state_snapshot()
        result["kpis"] = snapshot["kpis"]
        result["session"] = {
            "id": snapshot["kpis"].get("session_id"),
            "started_at": snapshot["kpis"].get("session_started_at"),
            "started_at_iso": snapshot["kpis"].get("session_started_at_iso"),
        }
        return result

    @application.post("/api/configure")
    async def set_config(config: dict):
        # Rebuilding a session can parse a large trajectory file.  It runs on
        # the managed worker pool so the event loop (telemetry, other clients)
        # stays responsive meanwhile.
        state = _sim()
        return await state.run_in_worker(state.set_config, config)

    @application.get("/api/fl_rounds")
    def get_fl_rounds():
        """Full federated-round history including per-client detail."""
        state = _sim()
        return {
            "rounds": [asdict(r) for r in state.fl_round_history],
            "current": asdict(state.current_round_metrics) if state.current_round_metrics else None,
            "readiness": state.fl_readiness(),
        }

    @application.get("/api/client_participation")
    def get_client_participation():
        """Per-client participation, training loss and inclusion/exclusion state."""
        state = _sim()
        return {
            "clients": {str(k): asdict(v) for k, v in state.client_participation.items()},
            "trust_scores": state.env.trust_engine.get_all_trust_scores(),
            "exclusion_threshold": state.exclusion_threshold,
            "alpha": state.alpha,
        }

    @application.get("/api/event_log")
    def get_event_log():
        """Structured backend event log."""
        return {"events": _sim().event_log[-200:]}

    @application.get("/api/benchmark_data")
    def get_benchmark_results():
        path = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {"status": "not_ready"}

    @application.get("/api/series")
    def get_series():
        """Back-fill for the live charts (a page that connects late)."""
        return _sim().series_payload()

    @application.get("/api/research/summary")
    def get_research_summary(request: Request):
        """Compact, provenance-tagged summary of the ACTIVE run's results.

        Everything is read from the active run's artefacts; nothing is
        hard-coded.  The per-seed raw records are dropped here (they stay on
        disk and in ``/api/benchmark_data``) to keep the payload small.
        """
        value = _research_summary()
        etag = '"%s"' % value.get("etag", "none")
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        return JSONResponse(content=value, headers={"ETag": etag, "Cache-Control": "no-cache"})

    @application.get("/api/figures")
    def get_figures():
        """The publication figures of the validated active run.

        The front end renders whatever this endpoint returns; it never
        hard-codes a figure name, so a stale or renamed figure can not be
        requested at all.
        """
        manifest, plots_dir = _load_figure_manifest()
        if manifest is None:
            return {"status": "not_ready", "figures": []}
        figures = []
        for entry in manifest.get("figures", []):
            figures.append({
                "filename": entry.get("filename"),
                "title": entry.get("title"),
                "description": entry.get("description"),
                "experiment_id": entry.get("experiment_id"),
                "source_experiment": entry.get("source_experiment"),
                "source_records": entry.get("source_records"),
                "sha256": entry.get("sha256"),
            })
        return {
            "status": "ok",
            "experiment_id": manifest.get("experiment_id"),
            "generated_utc": manifest.get("generated_utc"),
            "results_file": manifest.get("results_file"),
            "results_file_sha256": manifest.get("results_file_sha256"),
            "cross_check_problems": manifest.get("cross_check_problems"),
            "plots_dir": plots_dir,
            "figures": figures,
        }

    @application.get("/api/plots/{filename}")
    def get_plot_image(filename: str):
        """Serve a figure of the active run.

        Only files listed in the active run's figure manifest are served.
        Everything else - including a historical figure that still exists
        on disk - returns 404.
        """
        if os.path.basename(filename) != filename:
            return JSONResponse(status_code=404,
                                content={"message": "Plot not found"})
        manifest, plots_dir = _load_figure_manifest()
        if manifest is not None:
            allowed = {e.get("filename") for e in manifest.get("figures", [])}
            if filename not in allowed:
                return JSONResponse(status_code=404,
                                    content={"message": "Plot not part of the active run"})
            path = os.path.join(plots_dir, filename)
            if os.path.isfile(path):
                return FileResponse(path)
            return JSONResponse(status_code=404,
                                content={"message": "Plot file missing"})
        # No manifest at all: fall back to the top-level directory, but only
        # for a plain figure file name.
        path = os.path.join(_PROJECT_ROOT, "plots", filename)
        if os.path.isfile(path) and filename.startswith("fig"):
            return FileResponse(path)
        return JSONResponse(status_code=404, content={"message": "Plot not found"})

    @application.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        """Subscribe to state updates.

        This handler NEVER advances the simulation.  The central simulation
        loop owns simulation time; the socket only receives broadcast
        snapshots.
        """
        state = _sim()
        await websocket.accept()
        state.add_subscriber(websocket)
        try:
            # Send an immediate snapshot so a reconnecting browser is up to date.
            await websocket.send_json(state.get_state_snapshot())
            while True:
                # Keep the connection open; the broadcaster pushes updates.
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            print("WebSocket telemetry exception:", exc)
        finally:
            state.remove_subscriber(websocket)


# The module-level default application and its authoritative simulation.
sim = SimulationState()
app = create_app(sim)
