"""Scenario C: Unreliable Execution and Intermittent Connectivity.

Simulates:
- Execution failures: task fails on node (success=0, path_ok=1) -> observed failure.
- Delay / deadline misses: task delayed beyond deadline (success=0, deadline_miss=1, path_ok=1).
- Intermittent connectivity: return link drops (path_ok=0) -> unobserved disconnect.
  Per SPEC §6 and EQ-13, a lost link carries timeliness neutrally and is NEVER automatic malice.

Updates behavioural trust through the authentic `BehaviouralTrustTable.observe()` path.
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

        # If dropped or failed or delayed beyond deadline
        last_rec = self.env.trust.history[-1] if self.env.trust.history else None

        dest_kind = last_rec.kind if last_rec else ("vehicle" if action == 0 else "edge")
        dest_id = last_rec.node_id if last_rec else 0

        # Outcome logic
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

        # Pop previous trust record and re-observe with faulty evidence
        if self.env.trust.history:
            self.env.trust.history.pop()

        ev = Evidence(
            kind=dest_kind,
            node_id=dest_id,
            success=new_success,
            path_ok=new_path_ok,
            latency_s=new_latency,
            deadline_s=deadline,
            expected_exe_s=last_rec.s3 if last_rec else 1.0,
            actual_exe_s=new_latency,
        )
        self.env.trust.observe(self.env._task_counter, ev)

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


