"""Vehicular edge task-offloading environment (SPEC §§1-4, EQ-1–EQ-10).

Public API used by later phases:
    env = VehicularEdgeEnv(config)   # or VehicularEdgeEnv.from_yaml(path)
    env.reset(seed)                  # -> first (obs, mask)
    obs, mask = env.current_observation()  # obs length cfg.obs_dim = 7+3K (SPEC §4)
    result = env.step(action)        # StepResult with obs/mask for the NEXT task
    summary = env.run_episode(policy)  # policy(obs, mask) -> action in 0..K

Conventions (frozen from Phase 1): action 0 = local execution, 1..K = offload
to the j-th candidate (EQ-10). Invalid actions raise InvalidActionError; use
the mask. The env never computes rewards (Phase 4 owns the adapter in rl/);
per-candidate behavioural trust (Phase 5, trust/) fills the obs trust slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from specs import formulas as F
from trust.behavioural import BehaviouralTrustTable, Evidence
from environment.config import EnvConfig, EpisodeSummary, MEAN_GAP_S, from_yaml
from environment.edge_node import EdgeNode
from environment.mobility import MobilityModel
from environment.network import link_available, link_rate_hz
from environment.task import Task, TaskGenerator
from environment.vehicle import Vehicle


class InvalidActionError(ValueError):
    """Raised when an action is out of range or masked (ineligible destination)."""


@dataclass
class Candidate:
    """One offload destination offered for the current task."""

    kind: str  # "vehicle" | "edge"
    node_id: int
    dist_m: float


@dataclass
class StepResult:
    """Everything a later phase needs about one executed task (SPEC §4/§11)."""

    task_id: int
    action: int
    dest_kind: str  # "local" | "vehicle" | "edge"
    dest_id: int
    latency_s: float
    energy_j: float
    success: bool
    failed: bool
    deadline_missed: bool
    eligible_count: int
    dest_trust: float = 0.5  # SPEC EQ-11 destination trust at decision time (Phase 5+)
    obs: list[float] = field(default_factory=list)  # EQ-9, for the NEXT task
    mask: list[bool] = field(default_factory=list)  # EQ-10, for the NEXT task


class VehicularEdgeEnv:
    """Seeded synthetic vehicular-edge environment (SPEC U3 resolutions in README)."""

    def __init__(self, config: EnvConfig) -> None:
        config.validate()
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)
        self._mobility = MobilityModel(self._rng, config.road_length_m, config.mobility)
        self._tasks = TaskGenerator(self._rng, config.task_load, MEAN_GAP_S[config.task_load])
        self.vehicles: list[Vehicle] = []
        self.edges: list[EdgeNode] = []
        self.now_s: float = 0.0
        self._task_counter: int = 0
        self._requester_ptr: int = 0
        self.pending: Task | None = None
        self._cand_cache: list[Candidate] | None = None
        self._cand_task_id: int = -1
        self.trust = BehaviouralTrustTable(config.num_vehicles, config.num_edge_nodes,
                                           tuple(config.trust_beta), config.trust_lambda,
                                           config.trust_rho, config.trust_prior)

    @classmethod
    def from_yaml(cls, path: str) -> "VehicularEdgeEnv":
        return cls(from_yaml(path))

    def set_trust_lambda(self, lam: float) -> None:
        """Set effective trust smoothing coefficient and persist to config."""
        self.cfg.trust_lambda = float(lam)
        if hasattr(self, "trust") and self.trust is not None:
            self.trust.lam = float(lam)

    def get_effective_trust_lambda(self) -> float:
        """Single source of truth for runtime trust smoothing coefficient."""
        if hasattr(self, "trust") and self.trust is not None:
            return float(self.trust.lam)
        return float(self.cfg.trust_lambda)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None) -> tuple[list[float], list[bool]]:
        """(Re)build the scenario. Same seed -> identical stream (tested)."""
        if seed is not None:
            self.cfg.seed = seed
        self._rng = np.random.default_rng(self.cfg.seed)
        self._mobility = MobilityModel(self._rng, self.cfg.road_length_m, self.cfg.mobility)
        self._tasks = TaskGenerator(self._rng, self.cfg.task_load, MEAN_GAP_S[self.cfg.task_load])
        self.vehicles = [
            Vehicle(
                vid=i,
                x_m=float(self._rng.uniform(0, self.cfg.road_length_m)),
                speed_mps=self._mobility.sample_speed(),
                cpu_freq_hz=float(self.cfg.vehicle_cpu_ghz * 1e9) if getattr(self.cfg, "vehicle_cpu_ghz", None) is not None else float(self._rng.uniform(1e9, 3e9)),
                utilisation=float(self._rng.uniform(0.0, 0.7)),
                bandwidth_hz=self.cfg.bandwidth_hz,
                tx_power_w=self.cfg.tx_power_w,
            )
            for i in range(self.cfg.num_vehicles)
        ]
        spacing = self.cfg.road_length_m / max(1, self.cfg.num_edge_nodes)
        self.edges = [
            EdgeNode(
                eid=j,
                x_m=(j + 0.5) * spacing + float(self._rng.uniform(-50.0, 50.0)),
                cpu_freq_hz=float(self._rng.uniform(5e9, 10e9)),
            )
            for j in range(self.cfg.num_edge_nodes)
        ]
        for node in self.vehicles + self.edges:  # type: ignore[attr-defined]
            node.validate()
        self.now_s = 0.0
        self._task_counter = 0
        self._requester_ptr = 0
        self.trust = BehaviouralTrustTable(self.cfg.num_vehicles, self.cfg.num_edge_nodes,
                                           tuple(self.cfg.trust_beta), self.cfg.trust_lambda,
                                           self.cfg.trust_rho, self.cfg.trust_prior)
        self.pending = self._next_task()
        return self.current_observation()

    # ------------------------------------------------------------- observation
    def _wrapped_dist(self, a: float, b: float) -> float:
        raw = abs(a - b) % self.cfg.road_length_m
        return min(raw, self.cfg.road_length_m - raw)

    def candidates(self) -> list[Candidate]:
        """Eligible destinations for the pending task (EQ-2), nearest-first, top-K.

        Cached per task: the set shown in the observation is the set `step`
        executes against, so masks stay consistent within a decision step.
        """
        assert self.pending is not None, "call reset() first"
        if self._cand_cache is not None and self._cand_task_id == self.pending.task_id:
            return self._cand_cache
        req = self.vehicles[self.pending.requester_id]
        out: list[Candidate] = []
        for v in self.vehicles:
            if v.vid == req.vid or not v.available:
                continue
            d = self._wrapped_dist(req.x_m, v.x_m)
            if d <= self.cfg.comm_range_m and link_available(self._rng, self.cfg.link_reliability):
                out.append(Candidate("vehicle", v.vid, d))
        for e in self.edges:
            if not e.available:
                continue
            d = self._wrapped_dist(req.x_m, e.x_m)
            if d <= self.cfg.comm_range_m and link_available(self._rng, self.cfg.link_reliability):
                out.append(Candidate("edge", e.eid, d))
        out.sort(key=lambda c: c.dist_m)
        self._cand_cache = out[: self.cfg.top_k]
        self._cand_task_id = self.pending.task_id
        return self._cand_cache

    def current_observation(self) -> tuple[list[float], list[bool]]:
        """EQ-9 state vector (length cfg.obs_dim = 7+3K, all in [0,1]) + EQ-10 mask.

        Component order (SPEC §4): task (d, c, tau, p) | local (u, q) |
        per-candidate (dist_norm, trust, load) x K, zero-padded | eligible_count/K.
        """
        assert self.pending is not None, "call reset() first"
        c = self.cfg
        t = self.pending
        req = self.vehicles[t.requester_id]
        cands = self.candidates()
        obs = [
            F.minmax_norm(t.data_bits, 0.0, c.data_bits_max),
            F.minmax_norm(t.cycles, 0.0, c.cycles_max),
            F.minmax_norm(t.deadline_s, 0.0, c.deadline_max_s),
            F.minmax_norm(t.priority, 1.0, 3.0),
            F.minmax_norm(req.utilisation, 0.0, 1.0),
            F.minmax_norm(req.queue_cycles, 0.0, c.queue_cycles_max),
        ]
        for cand in cands:
            node = self._node_of(cand)
            q = node.queue_cycles
            obs += [
                F.minmax_norm(cand.dist_m, 0.0, c.comm_range_m),
                self.trust.get(cand.kind, cand.node_id),  # per-candidate behavioural trust (Phase 5)
                F.minmax_norm(q, 0.0, c.queue_cycles_max),
            ]
        for _ in range(c.top_k - len(cands)):
            obs += [0.0, 0.0, 0.0]
        obs.append(F.minmax_norm(len(cands), 0.0, c.top_k))
        mask = [True] + [i < len(cands) for i in range(c.top_k)]
        assert len(obs) == c.obs_dim, f"obs {len(obs)} != declared {c.obs_dim}"
        assert len(mask) == c.action_dim, f"mask {len(mask)} != declared {c.action_dim}"
        return obs, mask

    # ------------------------------------------------------------------- step
    def _node_of(self, cand: Candidate):
        return self.vehicles[cand.node_id] if cand.kind == "vehicle" else self.edges[cand.node_id]

    def _next_task(self) -> Task:
        req_id = self._requester_ptr % self.cfg.num_vehicles
        self._requester_ptr += 1
        task = self._tasks.next(self._task_counter, req_id, self.now_s, self.vehicles[req_id].cpu_freq_hz)
        self._task_counter += 1
        return task

    def _advance_dynamics(self, dt_s: float) -> None:
        """Mobility (EQ-1), queue draining, utilisation drift, availability flaps."""
        self._mobility.step(self.vehicles, dt_s)
        for v in self.vehicles:
            v.drain_queue(dt_s)
            v.utilisation = float(np.clip(v.utilisation + self._rng.normal(0, 0.05), 0.0, 0.8))
            if self._rng.random() < 0.02:
                v.available = not v.available
        for e in self.edges:
            e.drain_queue(dt_s)
            if self._rng.random() < 0.02:
                e.available = not e.available

    def step(self, action: int) -> StepResult:
        """Execute the pending task via `action`; returns outcome + next obs/mask."""
        assert self.pending is not None, "call reset() first"
        c = self.cfg
        t = self.pending
        cands = self.candidates()
        if not 0 <= action <= c.top_k:
            raise InvalidActionError(f"action {action} out of range 0..{c.top_k}")
        if action > 0 and action > len(cands):
            raise InvalidActionError(f"action {action} targets ineligible/padded candidate")

        dt = max(0.0, t.arrival_s - self.now_s)
        req = self.vehicles[t.requester_id]
        # Decision-time observables for the trust evidence (before dynamics run).
        if action == 0:
            _dec_node = req
            _dec_freq = _dec_node.effective_freq_hz
            _dest_trust = self.trust.get("vehicle", req.vid)
        else:
            _dec_cand = cands[action - 1]
            _dec_node = self._node_of(_dec_cand)
            _dec_freq = (_dec_node.cpu_freq_hz if _dec_cand.kind == "edge"
                         else _dec_node.effective_freq_hz)  # type: ignore[attr-defined]
            _dest_trust = self.trust.get(_dec_cand.kind, _dec_cand.node_id)
        _expected_exe = t.cycles / _dec_freq + _dec_node.queue_cycles / _dec_freq
        self._advance_dynamics(dt)
        self.now_s = t.arrival_s

        if action == 0:
            dest_kind, dest_id = "local", req.vid
            tx_s = 0.0
            exe_s = F.exec_latency(t.cycles, req.effective_freq_hz, req.queue_cycles / req.effective_freq_hz)
            energy = F.task_energy(0.0, 0.0, c.kappa, t.cycles, req.cpu_freq_hz)
            req.enqueue(t.cycles)
            path_ok = True
        else:
            cand = cands[action - 1]
            node = self._node_of(cand)
            dest_kind, dest_id = cand.kind, cand.node_id
            freq = node.cpu_freq_hz if cand.kind == "edge" else node.effective_freq_hz  # type: ignore[attr-defined]
            tx_s = F.transmission_latency(
                t.data_bits,
                link_rate_hz(cand.dist_m, req.bandwidth_hz, req.tx_power_w, c.pl_k0, c.pl_gamma, c.noise_w),
                c.handshake_s,
            )
            exe_s = F.exec_latency(t.cycles, freq, node.queue_cycles / freq)
            energy = F.task_energy(req.tx_power_w, tx_s, c.kappa, t.cycles, node.cpu_freq_hz)
            node.enqueue(t.cycles)
            # Return path must survive for the result to count (EQ-8, synthetic).
            path_ok = bool(node.available and link_available(self._rng, c.link_reliability))

        latency = F.total_latency(tx_s, exe_s)
        outcome = F.task_outcome(latency, t.deadline_s, path_ok)
        # Phase 5: behavioural-trust update from this execution, then neutral
        # ageing for every node NOT observed this step (SPEC EQ-13).
        # Local execution updates the requester's own vehicle entry.
        key_kind = "vehicle" if dest_kind == "local" else dest_kind
        self.trust.observe(t.task_id, Evidence(key_kind, dest_id, outcome["success"], path_ok,
                                               latency, t.deadline_s, _expected_exe, exe_s))
        self.trust.age_all(t.task_id, except_key=(key_kind, dest_id))
        self.pending = self._next_task()
        obs, mask = self.current_observation()
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
            dest_trust=_dest_trust,
            obs=obs,
            mask=mask,
        )

    def run_episode(self, policy, max_tasks: int | None = None) -> tuple[list[StepResult], EpisodeSummary]:
        """Run tasks with policy(obs, mask) -> action; returns per-task results + summary."""
        n = max_tasks or self.cfg.num_tasks
        results: list[StepResult] = []
        for _ in range(n):
            obs, mask = self.current_observation()
            results.append(self.step(int(policy(obs, mask))))
        lat = [r.latency_s for r in results]
        ene = [r.energy_j for r in results]
        summary = EpisodeSummary(
            num_tasks=len(results),
            successes=sum(r.success for r in results),
            deadline_misses=sum(r.deadline_missed for r in results),
            failures=sum(r.failed for r in results),
            mean_latency_s=float(np.mean(lat)) if lat else 0.0,
            mean_energy_j=float(np.mean(ene)) if ene else 0.0,
            offload_fraction=float(np.mean([r.action > 0 for r in results])) if results else 0.0,
        )
        return results, summary
