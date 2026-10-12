"""Scenario C: Unreliable Execution and Intermittent Connectivity.

Fault model (documented scope): corruption of an unreliable CLIENT's local
training experience. The wrapper sits around one client's whole environment,
so an injected fault perturbs whatever task that client executes next —
its observed outcome, reward signal, and behavioural-trust update — regardless
of which destination served it. This does NOT simulate destination-specific
unreliable edge execution; no claim of that form is made anywhere downstream.
Unreliable clients are honest participants (is_malicious=False); their updates
are ordinary FedAvg inputs, and detectors must not treat them as adversaries.

Simulates:
- Execution failures: task fails on node (success=0, path_ok=1) -> observed failure.
- Delay / deadline misses: task delayed beyond deadline (success=0, deadline_miss=1, path_ok=1).
- Intermittent connectivity: return link drops (path_ok=0) -> unobserved disconnect.
  Per SPEC §6 and EQ-13, a lost link carries timeliness neutrally and is NEVER automatic malice.

Trust is updated through the authentic `BehaviouralTrustTable.observe()` path
exactly once per task: the base environment's honest observation is retracted
(trust restored) and replaced by the faulty-evidence observation.

Event precedence for the applied outcome (counters still record every sampled
event independently, so drops+fails+delays may overlap while
tasks_with_any_fault counts the union): drop > fail > delay.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from environment import StepResult, VehicularEdgeEnv
from specs import formulas as F
from trust import Evidence



@dataclass
class UnreliableProfile:
    """Configurable failure, delay, and drop rates for an unreliable participant."""
    fail_prob: float = 0.0     # Probability of task execution failure
    delay_prob: float = 0.0    # Probability of excessive execution delay (deadline miss)
    delay_factor: float = 3.0  # Latency multiplier when delay occurs
    drop_prob: float = 0.0     # Probability of return path disconnection (path_ok=False)
    name: str = "unreliable_profile"


class UnreliableEnvWrapper:
    """Wraps VehicularEdgeEnv to inject controlled unreliability into task outcomes.

    Delegates all other calls to the wrapped env.
    Crucially, trust is updated through `env.trust.observe(...)` using legitimate evidence,
    distinguishing between observed failures, deadline misses, and unobserved disconnects.
    """

    def __init__(
        self,
        base_env: VehicularEdgeEnv,
        profile: UnreliableProfile,
        seed: int = 1234,
    ) -> None:
        self.env = base_env
        self.profile = profile
        self._fault_rng = np.random.default_rng(seed)
        self.injected_tasks_exposed: int = 0
        self.tasks_with_any_fault: int = 0
        self.injected_drops: int = 0
        self.injected_fails: int = 0
        self.injected_delays: int = 0

    def set_trust_lambda(self, lam: float) -> None:
        """Propagate trust smoothing coefficient to base environment."""
        if hasattr(self.env, "set_trust_lambda"):
            self.env.set_trust_lambda(lam)
        else:
            self.env.cfg.trust_lambda = float(lam)
            if hasattr(self.env, "trust") and self.env.trust is not None:
                self.env.trust.lam = float(lam)

    def get_effective_trust_lambda(self) -> float:
        """Single source of truth for runtime trust smoothing coefficient."""
        if hasattr(self.env, "get_effective_trust_lambda"):
            return self.env.get_effective_trust_lambda()
        return float(getattr(getattr(self.env, "trust", None), "lam", self.env.cfg.trust_lambda))

    def get_fault_stats(self) -> dict[str, int]:
        return {
            "tasks_exposed": self.injected_tasks_exposed,
            "tasks_with_any_fault": self.tasks_with_any_fault,
            "injected_drops": self.injected_drops,
            "injected_fails": self.injected_fails,
            "injected_delays": self.injected_delays,
        }

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, seed: int | None = None) -> tuple[list[float], list[bool]]:
        if seed is not None:
            self._fault_rng = np.random.default_rng(seed + 888)
        # Note: lifetime fault counters persist across resets within the trial
        return self.env.reset(seed)

    def step(self, action: int) -> StepResult:
        # Resolve the true destination BEFORE the base step consumes anything:
        # action 0 runs on the requester's own vehicle, else candidates[action-1].
        # (candidates are cached per task, so this is the exact set step() uses.)
        pre_cands = self.env.candidates()
        if action == 0:
            req_id = self.env.pending.requester_id if self.env.pending is not None else 0
            dest_kind, dest_id = "vehicle", req_id
        else:
            _cand = pre_cands[action - 1]
            dest_kind, dest_id = _cand.kind, _cand.node_id
        current_task = self.env.pending
        deadline = current_task.deadline_s if current_task is not None else self.env.cfg.deadline_max_s

        # First execute normally through environment to compute base physical quantities
        res = self.env.step(action)
        self.injected_tasks_exposed += 1

        # Independent Bernoulli sampling for each fault mode
        inject_drop = self._fault_rng.random() < self.profile.drop_prob
        inject_fail = self._fault_rng.random() < self.profile.fail_prob
        inject_delay = self._fault_rng.random() < self.profile.delay_prob

        has_fault = inject_drop or inject_fail or inject_delay
        if has_fault:
            self.tasks_with_any_fault += 1

        if inject_drop:
            self.injected_drops += 1
        if inject_fail:
            self.injected_fails += 1
        if inject_delay:
            self.injected_delays += 1

        if not has_fault:
            return res

        # In base env, path_ok is True iff (res.success or res.failed or res.deadline_missed)
        base_path_ok = bool(res.success or res.failed or res.deadline_missed)
        new_path_ok = False if inject_drop else base_path_ok
        new_latency = (res.latency_s * self.profile.delay_factor) if inject_delay else res.latency_s
        new_energy = res.energy_j

        # Outcome logic with documented precedence: drop > fail > delay
        if inject_drop:
            # Disconnect with no observed outcome on return link (path_ok=False)
            new_success = False
            new_failed = True
            new_deadline_missed = False
        elif inject_fail:
            # Observed execution failure
            new_success = False
            new_failed = True
            new_deadline_missed = False
        elif inject_delay:
            # Observed deadline miss
            new_success = False
            new_failed = False
            new_deadline_missed = True
        else:
            new_success = res.success
            new_failed = res.failed
            new_deadline_missed = res.deadline_missed

        # Retract the base environment's honest observation (trust restored to
        # pre-task value) so this task is observed exactly once, then re-observe
        # with the faulty evidence attributed to the TRUE destination.
        base_ev = getattr(self.env, "last_evidence", None)
        expected_exe = base_ev.expected_exe_s if base_ev is not None else max(new_latency, 1e-9)
        self.env.trust.retract_observation(res.task_id)

        ev = Evidence(
            kind=dest_kind,
            node_id=dest_id,
            success=new_success,
            path_ok=new_path_ok,
            latency_s=new_latency,
            deadline_s=deadline,
            expected_exe_s=expected_exe,
            actual_exe_s=new_latency,
        )
        self.env.trust.observe(res.task_id, ev)

        dest_trust = self.env.trust.get(dest_kind, dest_id)
        import dataclasses
        return dataclasses.replace(
            res,
            latency_s=new_latency,
            energy_j=new_energy,
            success=new_success,
            failed=new_failed,
            deadline_missed=new_deadline_missed,
            dest_trust=dest_trust,
        )


