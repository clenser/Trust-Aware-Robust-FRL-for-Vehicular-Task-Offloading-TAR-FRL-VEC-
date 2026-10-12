"""Phase 3 baseline offloading policies (no learning; Phases 4+ own learned agents).

Protocol: every policy is `policy(obs, mask, ctx) -> int` where `ctx` is a dict
with at least `"candidates"` (the env's current Candidate list, nearest-first)
and optionally `"env"` (for observable node state). Policies must only use
information available at decision time — no future task data, no hidden flags.
Any candidate-less situation falls back to action 0 (documented per policy).
"""

from __future__ import annotations

import numpy as np

# Resource-aware scoring weights (provisional, documented in baselines/README.md).
W_QUEUE = 0.5
W_FREQ = 0.3
W_DIST = 0.2
FREQ_REF_HZ = 10e9  # normalisation for advertised CPU capacity (max edge freq)


def ensure_valid(action: int, mask: list[bool]) -> int:
    """Return `action` if the mask allows it, else safe fallback 0 (local)."""
    if 0 <= action < len(mask) and mask[action]:
        return action
    return 0


def local_only(obs: list[float], mask: list[bool], ctx: dict | None = None) -> int:
    """Always execute locally (action 0). Mask-independent; never invalid."""
    return 0


def make_random_valid(seed: int):
    """Factory: uniform choice among mask-valid actions from a seeded stream."""

    rng = np.random.default_rng(seed)

    def policy(obs: list[float], mask: list[bool], ctx: dict | None = None) -> int:
        valid = [a for a, ok in enumerate(mask) if ok]
        return int(rng.choice(valid))

    policy.__name__ = f"random_valid_seed{seed}"
    return policy


def nearest_edge(obs: list[float], mask: list[bool], ctx: dict | None = None) -> int:
    """Offload to the nearest eligible edge (RSU) node; fallback local.

    Deterministic tie-break: (distance ascending, edge id ascending).
    Action index = position in the env's nearest-first candidate list + 1.
    """
    cands = (ctx or {}).get("candidates", [])
    best_key = None
    best_pos = -1
    for pos, cand in enumerate(cands):
        if cand.kind != "edge":
            continue
        key = (cand.dist_m, cand.node_id)
        if best_key is None or key < best_key:
            best_key = key
            best_pos = pos
    if best_pos < 0:
        return 0
    return ensure_valid(best_pos + 1, mask)


def candidate_score(queue_cycles: float, queue_max: float, freq_hz: float, dist_m: float, comm_range_m: float) -> float:
    """Transparent scoring rule: free capacity + compute power + proximity."""
    free = 1.0 - min(1.0, queue_cycles / queue_max)
    power = min(1.0, freq_hz / FREQ_REF_HZ)
    prox = 1.0 - min(1.0, dist_m / comm_range_m)
    return W_QUEUE * free + W_FREQ * power + W_DIST * prox


def resource_aware(obs: list[float], mask: list[bool], ctx: dict | None = None) -> int:
    """Offload to the eligible candidate with the best observable score.

    Uses only decision-time observables: advertised queue, advertised CPU freq,
    distance (all visible via ctx). Deterministic tie-break:
    (score descending, distance ascending, node id ascending).
    Falls back to local when no candidate is eligible.
    """
    ctx = ctx or {}
    cands = ctx.get("candidates", [])
    env = ctx.get("env")
    if not cands or env is None:
        return 0
    cfg = env.cfg
    best_key = None
    best_pos = -1
    for pos, cand in enumerate(cands):
        node = env.vehicles[cand.node_id] if cand.kind == "vehicle" else env.edges[cand.node_id]
        score = candidate_score(node.queue_cycles, cfg.queue_cycles_max, node.cpu_freq_hz, cand.dist_m, cfg.comm_range_m)
        key = (-score, cand.dist_m, cand.node_id)
        if best_key is None or key < best_key:
            best_key = key
            best_pos = pos
    if best_pos < 0:
        return 0
    return ensure_valid(best_pos + 1, mask)


POLICIES = ("local_only", "random_valid", "nearest_edge", "resource_aware")


def make_policy(name: str, seed: int):
    """Build a named policy; random_valid is seeded per evaluation seed."""
    if name == "local_only":
        return local_only
    if name == "random_valid":
        return make_random_valid(seed)
    if name == "nearest_edge":
        return nearest_edge
    if name == "resource_aware":
        return resource_aware
    if name == "trust_aware":
        return trust_aware
    raise ValueError(f"unknown policy: {name}")


def trust_aware(obs: list[float], mask: list[bool], ctx: dict | None = None) -> int:
    """Phase 5 exploratory variant: pick the eligible candidate with the highest
    behavioural trust (env.trust table). Deterministic tie-break:
    (trust descending, distance ascending, node id ascending).
    Falls back to local when no candidate is eligible OR when the best trust
    is below env.cfg.trust_threshold (documented low-trust rule).
    Does not modify the four Phase 3 baselines; exploratory, not robustness proof.
    """
    ctx = ctx or {}
    cands = ctx.get("candidates", [])
    env = ctx.get("env")
    if not cands or env is None:
        return 0
    threshold = env.cfg.trust_threshold
    best_key = None
    best_pos = -1
    for pos, cand in enumerate(cands):
        trust = env.trust.get(cand.kind, cand.node_id)
        key = (-trust, cand.dist_m, cand.node_id)
        if best_key is None or key < best_key:
            best_key = key
            best_pos = pos
    if best_pos < 0:
        return 0
    best_cand = cands[best_pos]
    if env.trust.get(best_cand.kind, best_cand.node_id) < threshold:
        return 0
    return ensure_valid(best_pos + 1, mask)
