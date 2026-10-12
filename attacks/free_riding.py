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
) -> dict:
    """Stale-replay attack (B3): replays a previous update with its genuine origin.

    Provenance rule: the originating `round_trained` is ALWAYS preserved and the
    actual staleness is computed as `current_round - origin_round`. Origin
    metadata is never rewritten to manufacture a target age. If the caller
    needs a specific age but no matching update exists, it must select an
    explicitly documented fallback itself (see AttackPipeline) and report the
    actual resulting age.
    """
    stale_u = copy.deepcopy(historical_update)
    origin_round = int(stale_u.get("round_trained", 0))
    stale_u["round_trained"] = origin_round
    stale_u["staleness"] = max(0, current_round - origin_round)
    return stale_u
