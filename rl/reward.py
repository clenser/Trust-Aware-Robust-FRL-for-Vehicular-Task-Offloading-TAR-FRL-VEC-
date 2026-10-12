"""Phase 4 reward adapter: SPEC EQ-11 with explicit provisional normalizers.

EQ-11 needs normalised latency/energy, whose bounds Phase 1 left unspecified.
Provisional, explicitly documented choice (same as the smoke scripts):
LAT_NORM_S = 10 s, ENE_NORM_J = 10 J, destination trust = neutral 0.5
(trust scoring arrives in Phase 5). Final calibration belongs to Phase 4
tuning / Phase 10 sweeps — these values must NOT be silently tuned to make
DQN beat baselines. Non-finite inputs raise loudly instead of propagating NaN.
"""

from __future__ import annotations

import math

from specs import formulas as F

LAT_NORM_S = 10.0
ENE_NORM_J = 10.0
NEUTRAL_TRUST = 0.5
REWARD_CAVEAT = (
    "EQ-11 defaults; latency/energy norms provisional (10s, 10J); "
    "destination trust neutral 0.5 pending Phase 5."
)


def task_reward(latency_s: float, energy_j: float, success: bool,
                failed: bool, deadline_missed: bool) -> float:
    """Compute one task's reward. Raises on non-finite latency/energy."""
    for name, val in (("latency_s", latency_s), ("energy_j", energy_j)):
        if not math.isfinite(val):
            raise ValueError(f"non-finite {name}: {val}")
    if latency_s < 0 or energy_j < 0:
        raise ValueError(f"negative latency/energy: {latency_s}, {energy_j}")
    return F.compute_reward(
        F.clamp01(latency_s / LAT_NORM_S),
        F.clamp01(energy_j / ENE_NORM_J),
        success, failed, deadline_missed, NEUTRAL_TRUST,
    )
