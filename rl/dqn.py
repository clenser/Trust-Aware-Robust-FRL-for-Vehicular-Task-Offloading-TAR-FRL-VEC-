"""Q-network and masked Bellman-target helpers (SPEC EQ-9/EQ-10 dimensions)."""

from __future__ import annotations

import torch
import torch.nn as nn

NEG_INF = -1e9


def build_q_network(obs_dim: int, action_dim: int, hidden: list[int]) -> nn.Sequential:
    """MLP Q(s,·): obs_dim inputs, one output per discrete action (ReLU hidden)."""
    if obs_dim < 1 or action_dim < 1 or not hidden or any(h < 1 for h in hidden):
        raise ValueError("invalid network dimensions")
    layers: list[nn.Module] = []
    prev = obs_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, action_dim))
    return nn.Sequential(*layers)


def masked_greedy(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax over mask-valid actions only; raises if a row has no valid action."""
    if q_values.dim() == 1:
        q_values = q_values.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if not mask.any(dim=1).all():
        raise ValueError("mask row with no valid action")
    masked = torch.where(mask, q_values, torch.full_like(q_values, NEG_INF))
    out = masked.argmax(dim=1)
    return out[0] if squeeze else out


def masked_max_next_q(target_net: nn.Module, next_obs: torch.Tensor, next_mask: torch.Tensor) -> torch.Tensor:
    """max_{a' valid} Q_target(s', a'); rows without valid actions yield 0.0."""
    with torch.no_grad():
        q = target_net(next_obs)
        masked = torch.where(next_mask, q, torch.full_like(q, NEG_INF))
        best, _ = masked.max(dim=1)
        return torch.where(next_mask.any(dim=1), best, torch.zeros_like(best))
