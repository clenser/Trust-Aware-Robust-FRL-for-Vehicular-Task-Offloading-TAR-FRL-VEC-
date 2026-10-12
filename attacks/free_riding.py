"""Scenario B: Free-Riding Attacks (Zero-Update, Copy-Update, Stale-Replay).

Implements:
- B1: Zero-update submission: client submits unchanged previous global model (Delta = 0).
- B2: Copy-update: client duplicates another participating client's update.
- B3: Stale-replay: client replays an update trained in an earlier round with historical round metadata.
"""

from __future__ import annotations

import copy
import torch


def apply_zero_update(reference_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Zero-update attack (B1): Client submits theta = theta_prev without local training."""
    return {k: v.cpu().clone() for k, v in reference_state.items()}


def apply_copy_update(target_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy-update attack (B2): Client submits a duplicated copy of another client's update."""
    return {k: v.cpu().clone() for k, v in target_state.items()}


def create_stale_update(
    historical_update: dict,
    current_round: int = 2,
    staleness_age: int | None = None,
) -> dict:
    """Stale-replay attack (B3): Replays a previous update with historical round metadata.

    Preserves `round_trained = historical_update['round_trained']` so the server
    computes the actual age `current_round - round_trained = staleness_age`.
    """
    stale_u = copy.deepcopy(historical_update)
    if "round_trained" in historical_update and staleness_age is None:
        origin_round = int(historical_update["round_trained"])
        stale_u["round_trained"] = origin_round
        stale_u["staleness"] = max(0, current_round - origin_round)
    elif staleness_age is not None:
        stale_u["round_trained"] = max(0, current_round - staleness_age)
        stale_u["staleness"] = staleness_age
    else:
        origin_round = int(stale_u.get("round_trained", 0))
        stale_u["round_trained"] = origin_round
        stale_u["staleness"] = max(0, current_round - origin_round)
    return stale_u
