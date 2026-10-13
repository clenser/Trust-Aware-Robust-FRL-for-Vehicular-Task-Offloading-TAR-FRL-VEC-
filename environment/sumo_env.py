"""Microscopic SUMO Vehicular Edge Computing Environment (SumoVehicularEdgeEnv).

Integrates Eclipse SUMO microscopic traffic with the TAR-FRL-VEC task offloading,
RSU association, handover tracking, and trust-aware reinforcement learning logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from specs import formulas as F
from trust.behavioural import BehaviouralTrustTable, Evidence
from environment.config import EnvConfig, EpisodeSummary, MEAN_GAP_S
from environment.edge_node import EdgeNode
from environment.network import link_available, link_rate_hz
from environment.task import Task, TaskGenerator
from environment.vehicle import Vehicle
from environment.vanet_env import Candidate, StepResult, InvalidActionError, VehicularEdgeEnv
from environment.sumo_adapter import (
    SumoTrafficSimulator,
    SumoScenarioConfig,
    SumoVehicleState,
    is_sumo_available,
    get_sumo_status,
)


@dataclass
class HandoverEvent:
    """Record of a vehicle transitioning between roadside units."""
    step_idx: int
    vid: int
    sumo_vid: str
    x_m: float
    time_s: float
    from_rsu: Optional[int]
    to_rsu: Optional[int]
    event_type: str  # "associate", "handover", "disconnect", "vehicle_departure"
    dist_to_prev_rsu_m: Optional[float] = None
    dist_to_new_rsu_m: Optional[float] = None
    reason: str = "unspecified"


class SumoVehicularEdgeEnv:
    """Microscopic vehicular edge environment driven by Eclipse SUMO and TraCI.

    Maintains full behavioral compatibility with VehicularEdgeEnv:
    - Observation space: 7 + 3*K
    - Action space: 0 (local) .. K (top-k offload candidates)
    - Trust management & evidence recording
    - Dynamic RSU associations, coverage boundaries, and handovers
    """

    def __init__(
        self,
        config: EnvConfig,
        sumo_config: Optional[SumoScenarioConfig] = None,
        use_sumo: bool = True,
    ) -> None:
        config.validate()
        self.cfg = config
        self.use_sumo = use_sumo and is_sumo_available()
        
        self.sumo_cfg = sumo_config or SumoScenarioConfig(
            road_length_m=config.road_length_m,
            num_lanes=3,
            density=config.mobility,  # "low", "medium", "high"
            seed=config.seed,
            step_length_s=0.1,
            gui=False,
        )

        self._rng = np.random.default_rng(config.seed)
        self._tasks = TaskGenerator(self._rng, config.task_load, MEAN_GAP_S[config.task_load])

        self.vehicles: list[Vehicle] = []
        self.edges: list[EdgeNode] = []
        self.now_s: float = 0.0
        self._task_counter: int = 0
        self._requester_ptr: int = 0
        self.pending: Optional[Task] = None
        self._cand_cache: Optional[list[Candidate]] = None
        self._cand_task_id: int = -1
        self.last_evidence: Optional[Evidence] = None
        self.last_evidence_task_id: int = -1

        # RSU tracking and handovers
        self.vehicle_associations: dict[int, Optional[int]] = {}  # vid -> connected eid
        self.handovers: list[HandoverEvent] = []
        self._sumo_sim: Optional[SumoTrafficSimulator] = None
        self._sumo_states: dict[int, SumoVehicleState] = {}
        self._int_to_sumo_id: dict[int, str] = {}
        self._sumo_to_int_id: dict[str, int] = {}
        self._next_vehicle_id: int = 0

        # Trust table
        self.trust = BehaviouralTrustTable(
            n_vehicles=max(config.num_vehicles * 20, 500),
            n_edges=config.num_edge_nodes,
            beta=tuple(config.trust_beta),
            lam=config.trust_lambda,
            rho=config.trust_rho,
            prior=config.trust_prior,
        )

    # ------------------------------------------------------------------ Setup
    def reset(self, seed: Optional[int] = None) -> tuple[list[float], list[bool]]:
        """Initialize or reset microscopic simulation scenario."""
        if seed is not None:
            self.cfg.seed = seed
            self.sumo_cfg.seed = seed

        self._rng = np.random.default_rng(self.cfg.seed)
        self._tasks = TaskGenerator(self._rng, self.cfg.task_load, MEAN_GAP_S[self.cfg.task_load])
        self.now_s = 0.0
        self._task_counter = 0
        self._requester_ptr = 0
        self.pending = None
        self._cand_cache = None
        self._cand_task_id = -1
        self.last_evidence = None
        self.last_evidence_task_id = -1
        self.vehicle_associations.clear()
        self.handovers.clear()
        self._sumo_states.clear()
        self._int_to_sumo_id.clear()
        self._sumo_to_int_id.clear()
        self._next_vehicle_id = 0

        # Setup Roadside Units (RSUs) spaced along the highway corridor
        self.edges = []
        n_edges = max(1, self.cfg.num_edge_nodes)
        edge_spacing = self.cfg.road_length_m / (n_edges + 1)
        for i in range(n_edges):
            x_pos = float((i + 1) * edge_spacing)
            edge_freq = float(self._rng.uniform(6e9, 10e9))
            self.edges.append(
                EdgeNode(
                    eid=i,
                    x_m=x_pos,
                    cpu_freq_hz=edge_freq,
                    available=True,
                )
            )

        # Initialize vehicles
        self.vehicles = []

        if self.use_sumo:
            if self._sumo_sim:
                self._sumo_sim.close()
            self._sumo_sim = SumoTrafficSimulator(self.sumo_cfg)
            self._sumo_sim.start(gui=False, seed=self.cfg.seed)

            # Warm up SUMO for 5-10 seconds to populate highway with real traffic
            warmup_steps = 50
            self._sumo_sim.step(warmup_steps)
            self._sync_sumo_vehicles()
        else:
            # Synthetic fallback initialization
            for i in range(self.cfg.num_vehicles):
                cpu_ghz = self.cfg.vehicle_cpu_ghz if getattr(self.cfg, "vehicle_cpu_ghz", None) else float(self._rng.uniform(1.2, 2.8))
                v = Vehicle(
                    vid=i,
                    x_m=float(self._rng.uniform(0, self.cfg.road_length_m)),
                    speed_mps=float(self._rng.uniform(20.0, 33.3)),
                    cpu_freq_hz=float(cpu_ghz * 1e9),
                    utilisation=float(self._rng.uniform(0.1, 0.6)),
                    bandwidth_hz=self.cfg.bandwidth_hz,
                    tx_power_w=self.cfg.tx_power_w,
                )
                self.vehicles.append(v)
                self._update_vehicle_association(v.vid, v.x_m)

        # Ensure at least one vehicle exists
        if not self.vehicles:
            cpu_ghz = self.cfg.vehicle_cpu_ghz if getattr(self.cfg, "vehicle_cpu_ghz", None) else float(self._rng.uniform(1.2, 2.8))
            v_fallback = Vehicle(
                vid=0,
                x_m=100.0,
                speed_mps=28.0,
                cpu_freq_hz=float(cpu_ghz * 1e9),
                utilisation=0.2,
                bandwidth_hz=self.cfg.bandwidth_hz,
                tx_power_w=self.cfg.tx_power_w,
            )
            self.vehicles.append(v_fallback)

        self.pending = self._next_task()
        return self.current_observation()

    def _sync_sumo_vehicles(self) -> None:
        """Synchronize active SUMO microscopic vehicle states with research vehicles."""
        if not self._sumo_sim:
            return

        active_sumo = self._sumo_sim.get_active_vehicles()
        current_sumo_ids = {s.vid for s in active_sumo}

        # Index existing vehicles by int id
        veh_map = {v.vid: v for v in self.vehicles}
        updated_vehicles: list[Vehicle] = []

        for s in active_sumo:
            if s.vid not in self._sumo_to_int_id:
                int_id = self._next_vehicle_id
                self._next_vehicle_id += 1
                self._sumo_to_int_id[s.vid] = int_id
                self._int_to_sumo_id[int_id] = s.vid
            else:
                int_id = self._sumo_to_int_id[s.vid]

            self._sumo_states[int_id] = s

            if int_id in veh_map:
                v = veh_map[int_id]
                v.x_m = s.x_m
                v.speed_mps = s.speed_mps
                updated_vehicles.append(v)
            else:
                # Spawn new vehicle with heterogeneous computational capacity
                cpu_ghz = self.cfg.vehicle_cpu_ghz if getattr(self.cfg, "vehicle_cpu_ghz", None) else float(self._rng.uniform(1.2, 2.8))
                v_new = Vehicle(
                    vid=int_id,
                    x_m=s.x_m,
                    speed_mps=s.speed_mps,
                    cpu_freq_hz=float(cpu_ghz * 1e9),
                    utilisation=float(self._rng.uniform(0.1, 0.5)),
                    bandwidth_hz=self.cfg.bandwidth_hz,
                    tx_power_w=self.cfg.tx_power_w,
                )
                updated_vehicles.append(v_new)

            # Update association and check for handover events
            self._update_vehicle_association(int_id, s.x_m)

        # Detect departed vehicles that exited the simulation network
        departed_vids = [v.vid for v in self.vehicles if self._int_to_sumo_id.get(v.vid) not in current_sumo_ids]
        for d_vid in departed_vids:
            prev_rsu = self.vehicle_associations.get(d_vid)
            if prev_rsu is not None:
                sumo_name = self._int_to_sumo_id.get(d_vid, f"v_{d_vid}")
                prev_x = veh_map[d_vid].x_m if d_vid in veh_map else self.cfg.road_length_m
                dist_prev = abs(prev_x - self.edges[prev_rsu].x_m) if prev_rsu < len(self.edges) else None
                self.handovers.append(HandoverEvent(
                    step_idx=self._task_counter,
                    vid=d_vid,
                    sumo_vid=sumo_name,
                    x_m=prev_x,
                    time_s=self.now_s,
                    from_rsu=prev_rsu,
                    to_rsu=None,
                    event_type="vehicle_departure",
                    dist_to_prev_rsu_m=dist_prev,
                    dist_to_new_rsu_m=None,
                    reason="vehicle_departed_network",
                ))
                self.vehicle_associations[d_vid] = None

        self.vehicles = updated_vehicles

    def _update_vehicle_association(self, vid: int, x_m: float) -> None:
        """Detect RSU coverage association, loss of coverage, and RSU handovers."""
        prev_rsu = self.vehicle_associations.get(vid)
        current_rsu = None
        min_dist = float("inf")

        for edge in self.edges:
            dist = abs(x_m - edge.x_m)
            if dist <= self.cfg.comm_range_m and dist < min_dist:
                current_rsu = edge.eid
                min_dist = dist

        self.vehicle_associations[vid] = current_rsu
        sumo_name = self._int_to_sumo_id.get(vid, f"v_{vid}")

        dist_prev = abs(x_m - self.edges[prev_rsu].x_m) if (prev_rsu is not None and prev_rsu < len(self.edges)) else None
        dist_curr = abs(x_m - self.edges[current_rsu].x_m) if (current_rsu is not None and current_rsu < len(self.edges)) else None

        if prev_rsu is None and current_rsu is not None:
            self.handovers.append(HandoverEvent(
                step_idx=self._task_counter,
                vid=vid,
                sumo_vid=sumo_name,
                x_m=x_m,
                time_s=self.now_s,
                from_rsu=None,
                to_rsu=current_rsu,
                event_type="associate",
                dist_to_prev_rsu_m=None,
                dist_to_new_rsu_m=dist_curr,
                reason="initial_rsu_coverage",
            ))
        elif prev_rsu is not None and current_rsu is not None and prev_rsu != current_rsu:
            self.handovers.append(HandoverEvent(
                step_idx=self._task_counter,
                vid=vid,
                sumo_vid=sumo_name,
                x_m=x_m,
                time_s=self.now_s,
                from_rsu=prev_rsu,
                to_rsu=current_rsu,
                event_type="handover",
                dist_to_prev_rsu_m=dist_prev,
                dist_to_new_rsu_m=dist_curr,
                reason="closer_rsu_in_overlap",
            ))
        elif prev_rsu is not None and current_rsu is None:
            self.handovers.append(HandoverEvent(
                step_idx=self._task_counter,
                vid=vid,
                sumo_vid=sumo_name,
                x_m=x_m,
                time_s=self.now_s,
                from_rsu=prev_rsu,
                to_rsu=None,
                event_type="disconnect",
                dist_to_prev_rsu_m=dist_prev,
                dist_to_new_rsu_m=None,
                reason="out_of_coverage",
            ))

    def _next_task(self) -> Task:
        if not self.vehicles:
            raise RuntimeError("No vehicles available in the simulation to request tasks.")
        req_id = self.vehicles[self._requester_ptr % len(self.vehicles)].vid
        self._requester_ptr += 1
        req_veh = next((v for v in self.vehicles if v.vid == req_id), self.vehicles[0])
        task = self._tasks.next(self._task_counter, req_veh.vid, self.now_s, req_veh.cpu_freq_hz)
        self._task_counter += 1
        return task

    # -------------------------------------------------------------- Candidates
    def candidates(self) -> list[Candidate]:
        """Discovers eligible offloading targets (covered RSUs + nearby V2V peers)."""
        assert self.pending is not None, "call reset() first"
        t = self.pending
        if self._cand_cache is not None and self._cand_task_id == t.task_id:
            return self._cand_cache

        req = next((v for v in self.vehicles if v.vid == t.requester_id), None)
        if not req:
            req = self.vehicles[0]
            t.requester_id = req.vid

        cands: list[Candidate] = []

        # 1. Edge nodes within wireless communication range (R_comm = 300m)
        for e in self.edges:
            dist = abs(req.x_m - e.x_m)
            if dist <= self.cfg.comm_range_m and e.available:
                cands.append(Candidate(kind="edge", node_id=e.eid, dist_m=dist))

        # 2. Peer vehicles within V2V communication range (R_v2v = 150m)
        v2v_range = min(self.cfg.comm_range_m, 150.0)
        for v in self.vehicles:
            if v.vid == req.vid or not v.available:
                continue
            dist = abs(req.x_m - v.x_m)
            if dist <= v2v_range:
                cands.append(Candidate(kind="vehicle", node_id=v.vid, dist_m=dist))

        # Sort candidates: RSUs first by distance, then vehicles by distance
        cands.sort(key=lambda c: (0 if c.kind == "edge" else 1, c.dist_m))
        cands = cands[:self.cfg.top_k]

        self._cand_cache = cands
        self._cand_task_id = t.task_id
        return cands

    def current_observation(self) -> tuple[list[float], list[bool]]:
        """Observation vector (7 + 3*K) and valid action mask."""
        assert self.pending is not None, "call reset() first"
        t = self.pending
        c = self.cfg
        req = next((v for v in self.vehicles if v.vid == t.requester_id), self.vehicles[0])

        obs: list[float] = [
            F.minmax_norm(t.data_bits, 0.0, c.data_bits_max),
            F.minmax_norm(t.cycles, 0.0, c.cycles_max),
            F.minmax_norm(t.deadline_s, 0.0, c.deadline_max_s),
            F.minmax_norm(float(t.priority), 1.0, 3.0),
            F.minmax_norm(req.utilisation, 0.0, 1.0),
            F.minmax_norm(req.queue_cycles, 0.0, c.queue_cycles_max),
        ]

        cands = self.candidates()
        for cand in cands:
            node = self.edges[cand.node_id] if cand.kind == "edge" else next((v for v in self.vehicles if v.vid == cand.node_id), req)
            q = node.queue_cycles
            obs += [
                F.minmax_norm(cand.dist_m, 0.0, c.comm_range_m),
                self.trust.get(cand.kind, cand.node_id),
                F.minmax_norm(q, 0.0, c.queue_cycles_max),
            ]

        for _ in range(c.top_k - len(cands)):
            obs += [0.0, 0.0, 0.0]

        obs.append(F.minmax_norm(len(cands), 0.0, c.top_k))
        mask = [True] + [i < len(cands) for i in range(c.top_k)]

        assert len(obs) == c.obs_dim, f"obs {len(obs)} != declared {c.obs_dim}"
        assert len(mask) == c.action_dim, f"mask {len(mask)} != declared {c.action_dim}"
        return obs, mask

    def _advance_dynamics(self, dt_s: float) -> None:
        """Advance microscopic traffic dynamics, drain queues, and update utilization."""
        if self.use_sumo and self._sumo_sim:
            steps = max(1, int(round(dt_s / self.sumo_cfg.step_length_s)))
            self._sumo_sim.step(steps)
            self._sync_sumo_vehicles()
        else:
            # Synthetic mobility update
            for v in self.vehicles:
                noise = float(self._rng.uniform(-0.5, 0.5)) * dt_s
                v.x_m = (v.x_m + v.speed_mps * dt_s + noise) % self.cfg.road_length_m
                self._update_vehicle_association(v.vid, v.x_m)

        for v in self.vehicles:
            v.drain_queue(dt_s)
            v.utilisation = float(np.clip(v.utilisation + self._rng.normal(0, 0.03), 0.0, 0.8))

        for e in self.edges:
            e.drain_queue(dt_s)

    def step(self, action: int) -> StepResult:
        """Execute task offloading decision under microscopic traffic conditions."""
        assert self.pending is not None, "call reset() first"
        c = self.cfg
        t = self.pending
        cands = self.candidates()

        if not 0 <= action <= c.top_k:
            raise InvalidActionError(f"action {action} out of range 0..{c.top_k}")
        if action > 0 and action > len(cands):
            raise InvalidActionError(f"action {action} targets ineligible candidate")

        dt = max(0.05, t.arrival_s - self.now_s)
        req = next((v for v in self.vehicles if v.vid == t.requester_id), self.vehicles[0])

        # Capture decision-time state
        if action == 0:
            dec_node = req
            dec_freq = dec_node.effective_freq_hz
            dest_trust = self.trust.get("vehicle", req.vid)
        else:
            cand = cands[action - 1]
            dec_node = self.edges[cand.node_id] if cand.kind == "edge" else next((v for v in self.vehicles if v.vid == cand.node_id), req)
            dec_freq = dec_node.cpu_freq_hz if cand.kind == "edge" else dec_node.effective_freq_hz  # type: ignore[attr-defined]
            dest_trust = self.trust.get(cand.kind, cand.node_id)

        expected_exe = t.cycles / dec_freq + dec_node.queue_cycles / dec_freq

        # Advance microscopic traffic
        self._advance_dynamics(dt)
        self.now_s = t.arrival_s

        # Compute offload outcome
        handover_penalty_s = 0.0
        path_ok = True

        if action == 0:
            dest_kind, dest_id = "local", req.vid
            tx_s = 0.0
            exe_s = F.exec_latency(t.cycles, req.effective_freq_hz, req.queue_cycles / req.effective_freq_hz)
            energy = F.task_energy(0.0, 0.0, c.kappa, t.cycles, req.cpu_freq_hz)
            req.enqueue(t.cycles)
            path_ok = True
        else:
            cand = cands[action - 1]
            node = self.edges[cand.node_id] if cand.kind == "edge" else next((v for v in self.vehicles if v.vid == cand.node_id), req)
            dest_kind, dest_id = cand.kind, cand.node_id
            freq = node.cpu_freq_hz if cand.kind == "edge" else node.effective_freq_hz  # type: ignore[attr-defined]

            # Shannon channel link transmission latency
            tx_s = F.transmission_latency(
                t.data_bits,
                link_rate_hz(cand.dist_m, req.bandwidth_hz, req.tx_power_w, c.pl_k0, c.pl_gamma, c.noise_w),
                c.handshake_s,
            )

            # RSU Handover impact
            if cand.kind == "edge":
                curr_assoc = self.vehicle_associations.get(req.vid)
                if curr_assoc != cand.node_id and curr_assoc is not None:
                    handover_penalty_s = 0.05  # Handover signalling delay
                # Verify vehicle did not exit coverage boundary
                curr_dist = abs(req.x_m - node.x_m)
                if curr_dist > c.comm_range_m:
                    path_ok = False  # Link dropped due to vehicular movement out of range

            tx_s += handover_penalty_s
            exe_s = F.exec_latency(t.cycles, freq, node.queue_cycles / freq)
            energy = F.task_energy(req.tx_power_w, tx_s, c.kappa, t.cycles, node.cpu_freq_hz)
            node.enqueue(t.cycles)

            if path_ok:
                path_ok = bool(node.available and link_available(self._rng, c.link_reliability))

        latency = F.total_latency(tx_s, exe_s)
        outcome = F.task_outcome(latency, t.deadline_s, path_ok)

        # Update behavioural trust table
        key_kind = "vehicle" if dest_kind == "local" else dest_kind
        evidence = Evidence(key_kind, dest_id, outcome["success"], path_ok, latency, t.deadline_s, expected_exe, exe_s)
        self.trust.observe(t.task_id, evidence)
        self.last_evidence = evidence
        self.last_evidence_task_id = t.task_id
        self.trust.age_all(t.task_id, except_key=(key_kind, dest_id))

        # Advance to next task
        self.pending = self._next_task()
        next_obs, next_mask = self.current_observation()

        return StepResult(
            task_id=t.task_id,
            action=action,
            dest_kind=dest_kind,
            dest_id=dest_id,
            latency_s=latency,
            energy_j=energy,
            success=outcome["success"],
            failed=outcome["failed"],
            deadline_missed=outcome["deadline_missed"],
            eligible_count=len(cands),
            dest_trust=dest_trust,
            obs=next_obs,
            mask=next_mask,
        )

    def close(self) -> None:
        """Safely terminate SUMO simulation session."""
        if self._sumo_sim:
            self._sumo_sim.close()
            self._sumo_sim = None
