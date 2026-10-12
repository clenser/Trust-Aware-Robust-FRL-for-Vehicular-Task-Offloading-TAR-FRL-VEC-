"""Trust-aware robust federated aggregation (SPEC §8, EQ-22 and EQ-23).

Supports:
- EQ-21: Standard FedAvg (sample-count-weighted).
- EQ-22: Trust-weighted aggregation (normalized by participating trust scores).
- EQ-23: Trust-aware aggregation with threshold-based exclusion and fallback model retention.

Design rules:
- State dict validation matches fedavg.py (key match, shape match, dtype match, finite check).
- When exclusion removes all clients (S == empty) or total trust <= 0, the server safely
  retains the last known good global model (fallback) without error or NaN creation.
- Deterministic iteration order for numerical reproducibility.
- Pure parameter updates; clients send parameters + metadata only.
"""

from __future__ import annotations

import copy
import torch

from federated.fedavg import validate_update
from specs import formulas as F

DEFAULT_TAU_EXC: float = 0.30


def trust_weighted_aggregate(
    states: list[dict[str, torch.Tensor]],
    trust_scores: list[float],
    template: dict[str, torch.Tensor],
    client_ids: list[str | int],
    threshold: float = 0.0,
    fallback_state: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str | int], list[float]]:
    """Trust-weighted parameter average with optional exclusion threshold (EQ-22 / EQ-23).

    Parameters:
    - states: list of client state dicts.
    - trust_scores: list of client trust scores in [0, 1] (e.g. T_final).
    - template: template model state dict for shape and type validation.
    - client_ids: list of client identifiers matching `states` order.
    - threshold: minimum trust score tau_exc required for inclusion in aggregation.
    - fallback_state: model state dict to retain if no clients meet threshold.

    Returns:
    - (new_state, accepted_client_ids, normalized_weights)

    Behavior:
    - Validates all client states against template.
    - Filters client set: S = {i : trust_scores[i] >= threshold}.
    - If S is empty or sum_{i in S} trust_scores[i] <= 0:
        - If fallback_state is provided, returns (copy of fallback_state, [], []).
        - If fallback_state is None, returns (copy of template, [], []).
    - Otherwise:
        - Computes normalized trust weights: w_i = trust_scores[i] / sum_{j in S} trust_scores[j].
        - Computes theta_new = sum_{i in S} w_i * theta_i in float64, converted to template dtype.
        - Returns (theta_new, accepted_ids, weights).
    """
    if len(states) != len(trust_scores) or len(states) != len(client_ids):
        raise ValueError(
            f"Length mismatch: states={len(states)}, trusts={len(trust_scores)}, ids={len(client_ids)}"
        )

    # Validate each participating state
    for s, cid in zip(states, client_ids):
        validate_update(s, template, str(cid))

    # Identify eligible clients S = {i : trust_scores[i] >= threshold}
    eligible_indices: list[int] = []
    for idx, t in enumerate(trust_scores):
        clamped_t = F.clamp01(float(t))
        if clamped_t >= threshold:
            eligible_indices.append(idx)

    # Calculate total eligible trust
    eligible_trust_sum = sum(F.clamp01(float(trust_scores[i])) for i in eligible_indices)

    # Fallback condition (EQ-23 fallback): no eligible clients or non-positive total trust
    if not eligible_indices or eligible_trust_sum <= 0.0:
        base_fallback = fallback_state if fallback_state is not None else template
        fallback_copy = {k: v.cpu().clone() for k, v in base_fallback.items()}
        return fallback_copy, [], []

    # Calculate normalized weights w_i
    accepted_ids = [client_ids[i] for i in eligible_indices]
    normalized_weights = [
        F.clamp01(float(trust_scores[i])) / eligible_trust_sum for i in eligible_indices
    ]

    # Aggregate parameters in float64
    out: dict[str, torch.Tensor] = {}
    for k in template:
        acc = torch.zeros_like(template[k], dtype=torch.float64)
        for idx_pos, orig_idx in enumerate(eligible_indices):
            w = normalized_weights[idx_pos]
            acc += states[orig_idx][k].to(torch.float64) * w
        out[k] = acc.to(template[k].dtype)

    return out, accepted_ids, normalized_weights
