"""Standard sample-count-weighted FedAvg (SPEC EQ-21; Phase 6 only).

theta_global = sum_i (n_i / N) * theta_i, N = sum of client weights.
Trust weighting, exclusion, and anomaly detection arrive in Phase 7.
"""

from __future__ import annotations

import torch


def validate_update(state: dict[str, torch.Tensor], template: dict[str, torch.Tensor], client_id: str) -> None:
    """Reject malformed updates: key mismatch, shape/dtype mismatch, non-finite."""
    if set(state.keys()) != set(template.keys()):
        raise ValueError(f"client {client_id}: parameter key mismatch "
                         f"(extra={set(state) - set(template)}, missing={set(template) - set(state)})")
    for k in template:
        if state[k].shape != template[k].shape:
            raise ValueError(f"client {client_id}: shape mismatch on {k}: "
                             f"{tuple(state[k].shape)} != {tuple(template[k].shape)}")
        if state[k].dtype != template[k].dtype:
            raise ValueError(f"client {client_id}: dtype mismatch on {k}")
        if not torch.isfinite(state[k]).all():
            raise ValueError(f"client {client_id}: non-finite values in {k}")


def fedavg_aggregate(states: list[dict[str, torch.Tensor]], weights: list[float],
                     template: dict[str, torch.Tensor], client_ids: list[str]) -> dict[str, torch.Tensor]:
    """Weighted parameter average. Deterministic for fixed inputs (summation in
    client order). Raises on empty participation or zero total weight."""
    if not states:
        raise ValueError("no participating clients: cannot aggregate empty update set")
    if len(states) != len(weights) or len(states) != len(client_ids):
        raise ValueError("states/weights/client_ids length mismatch")
    if any(w < 0 for w in weights):
        raise ValueError("aggregation weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("total aggregation weight is zero: global model unchanged")
    for s, cid in zip(states, client_ids):
        validate_update(s, template, cid)
    out: dict[str, torch.Tensor] = {}
    for k in template:
        acc = torch.zeros_like(template[k], dtype=torch.float64)
        for s, w in zip(states, weights):
            acc += s[k].to(torch.float64) * (w / total)
        out[k] = acc.to(template[k].dtype)
    return out


def payload_bytes(state: dict[str, torch.Tensor]) -> int:
    """Analytical upload/download payload per model transfer (fp32, no compression).

    Simulated estimate, NOT measured network performance: params * 4 bytes.
    """
    return sum(t.numel() * t.element_size() for t in state.values())
