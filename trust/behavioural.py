"""Behavioural trust table (SPEC §6, EQ-12/EQ-13).

Standalone: no dependency on the environment, DQN, or federation code, so
Phase 6+ can reuse it. All scores live in [0, 1]; every update is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from specs import formulas as F

DEFAULT_BETA = (0.4, 0.25, 0.2, 0.15)  # success, timeliness, resources, reliability


@dataclass
class Evidence:
    """One observed task execution on a node (only env-exposed quantities)."""

    kind: str  # "vehicle" | "edge"
    node_id: int
    success: bool  # verified task success (SPEC s1)
    path_ok: bool  # result returned (False => communication/path failure)
    latency_s: float  # measured total latency
    deadline_s: float  # task deadline
    expected_exe_s: float  # queue-advertised expectation at decision time
    actual_exe_s: float  # realised execution part (excl. transmission)


@dataclass
class TrustRecord:
    """Auditable account of one trust change (SPEC §6 logging)."""

    task_id: int
    kind: str
    node_id: int
    event: str  # "success" | "deadline_miss" | "path_failure" | "idle_age"
    s1: float = float("nan")
    s2: float = float("nan")
    s3: float = float("nan")
    s4: float = float("nan")
    t_before: float = float("nan")
    t_after: float = float("nan")


class BehaviouralTrustTable:
    """Per-node behavioural trust with temporal smoothing and neutral ageing."""

    def __init__(self, n_vehicles: int, n_edges: int, beta: tuple = DEFAULT_BETA,
                 lam: float = 0.9, rho: float = 0.99, prior: float = 0.5) -> None:
        if abs(sum(beta) - 1.0) > 1e-9:
            raise ValueError("beta weights must sum to 1")
        if not 0.0 <= lam < 1.0:
            raise ValueError("lam must be in [0, 1)")
        self.beta = beta
        self.lam = lam
        self.rho = rho
        self.prior = F.clamp01(prior)
        self.n_vehicles = n_vehicles
        self.n_edges = n_edges
        self._t: dict[tuple[str, int], float] = {}
        self._last_s2: dict[tuple[str, int], float] = {}
        self._aged_rounds: dict[tuple[str, int], int] = {}
        self.history: list[TrustRecord] = []
        self.reset()

    def _check(self, kind: str, node_id: int) -> tuple[str, int]:
        if kind == "vehicle" and 0 <= node_id < self.n_vehicles:
            return (kind, node_id)
        if kind == "edge" and 0 <= node_id < self.n_edges:
            return (kind, node_id)
        raise ValueError(f"unknown node {(kind, node_id)}")

    def reset(self) -> None:
        """All nodes back to the prior; history cleared (called on env reset)."""
        self._t = {(k, i): self.prior
                   for k in ("vehicle", "edge")
                   for i in range(self.n_vehicles if k == "vehicle" else self.n_edges)}
        self._last_s2 = {}
        self._aged_rounds = {key: 0 for key in self._t}
        self.history = []

    def get(self, kind: str, node_id: int) -> float:
        return self._t[self._check(kind, node_id)]

    def snapshot(self) -> dict[tuple[str, int], float]:
        return dict(self._t)

    def observe(self, task_id: int, ev: Evidence) -> TrustRecord:
        """EQ-12/EQ-13 update from one execution. Never invents missing data:

        - s1: verified success, 1/0 (factual even on path failure).
        - s2: timeliness clip(1 - L/tau) when the result returned; otherwise the
          node's last s2 (missing, NOT negative) — documented resolution.
        - s3: advertised-vs-observed resource match = expected/actual execution.
        - s4: communication reliability, 1 iff the return path survived.
        """
        key = self._check(ev.kind, ev.node_id)
        for name, val in (("latency_s", ev.latency_s), ("deadline_s", ev.deadline_s),
                          ("expected_exe_s", ev.expected_exe_s), ("actual_exe_s", ev.actual_exe_s)):
            if not math.isfinite(val):
                raise ValueError(f"non-finite evidence {name}={val} for node {key}: "
                                 "refusing to score NaN/inf as trust (would poison silently)")
            if val < 0:
                raise ValueError(f"negative evidence {name}={val} for node {key}")
        if ev.deadline_s <= 0:
            raise ValueError(f"non-positive deadline {ev.deadline_s} for node {key}")
        s1 = 1.0 if ev.success else 0.0
        if ev.path_ok:
            s2 = F.clamp01(1.0 - ev.latency_s / ev.deadline_s)
            self._last_s2[key] = s2
        else:
            s2 = self._last_s2.get(key, self.prior)  # missing evidence carried, not punished
        s3 = F.clamp01(ev.expected_exe_s / ev.actual_exe_s) if ev.actual_exe_s > 0 else 1.0
        s4 = 1.0 if ev.path_ok else 0.0
        t_beh = F.behavioural_trust(
            {"success": s1, "timeliness": s2, "resources": s3, "reliability": s4},
            {"success": self.beta[0], "timeliness": self.beta[1],
             "resources": self.beta[2], "reliability": self.beta[3]})
        before = self._t[key]
        after = F.temporal_update(before, t_beh, self.lam)
        self._t[key] = after
        event = "success" if ev.success else ("path_failure" if not ev.path_ok else "deadline_miss")
        rec = TrustRecord(task_id, ev.kind, ev.node_id, event, s1, s2, s3, s4, before, after)
        self.history.append(rec)
        return rec

    def retract_observation(self, task_id: int) -> TrustRecord | None:
        """Remove the latest non-idle observation record for a task and restore
        the node's trust to its pre-observation value.

        Support hook for fault-injection wrappers (Phase 8/10): the wrapper lets
        the base environment record its honest observation, then retracts it and
        re-observes once with faulty evidence. This preserves exactly-one-
        observation semantics per task (no double temporal smoothing) and keeps
        the audit history truthful. Returns the retracted record, or None if
        no observation record exists for the task.
        Note: the per-node carried timeliness (_last_s2) is intentionally left
        in place, since it derives from the same task attempt's measured latency.
        """
        for idx in range(len(self.history) - 1, -1, -1):
            rec = self.history[idx]
            if rec.task_id == task_id and rec.event != "idle_age":
                del self.history[idx]
                self._t[(rec.kind, rec.node_id)] = rec.t_before
                return rec
        return None

    def age_all(self, task_id: int, except_key: tuple[str, int] | None = None) -> None:
        """EQ-13 disconnect rule: non-observed nodes age neutrally toward the
        prior. A lost link or idle node is never scored as misbehaviour."""
        for key in self._t:
            if key == except_key:
                continue
            before = self._t[key]
            after = F.idle_ageing(before, self.prior, self.rho)
            self._t[key] = after
            self._aged_rounds[key] += 1
            self.history.append(TrustRecord(task_id, key[0], key[1], "idle_age",
                                            t_before=before, t_after=after))
