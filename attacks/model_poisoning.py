"""Scenario A: Model Poisoning Attacks (Sign-Flip and Perturbation).

Implements:
- A1: Sign-flip poisoning: Delta_attack = -s * Delta_honest
- A2: Random perturbation poisoning: Delta_attack = Delta_honest + epsilon * z
"""

from __future__ import annotations

import copy
import math
import numpy as np
import torch


def apply_sign_flip_attack(
    honest_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    strength: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Sign-flip attack (A1): Delta_attack = -s * Delta_honest.

    theta_attack = theta_prev + Delta_attack = theta_prev - s * (theta_honest - theta_prev).
    """
    if strength <= 0.0:
        raise ValueError(f"Sign-flip strength s must be positive, got {strength}")

    attacked_state: dict[str, torch.Tensor] = {}
    for k in reference_state:
        if k not in honest_state:
            raise ValueError(f"Missing parameter key {k} in honest update")
        t_ref = reference_state[k]
        t_honest = honest_state[k]
        if torch.is_floating_point(t_honest):
            delta_honest = t_honest.to(torch.float64) - t_ref.to(torch.float64)
            delta_attack = -strength * delta_honest
            attacked_state[k] = (t_ref.to(torch.float64) + delta_attack).to(t_honest.dtype)
        else:
            attacked_state[k] = t_honest.clone()
    return attacked_state


def apply_perturbation_attack(
    honest_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Perturbation attack (A2): Delta_attack = Delta_honest + epsilon * u.

    where u is a normalized unit random vector sampled from N(0, 1) and
    epsilon = strength * ||Delta_honest|| (relative perturbation) if ||Delta_honest|| > 0,
    or strength * 1.0 as fallback when ||Delta_honest|| == 0.
    """
    if strength <= 0.0:
        raise ValueError(f"Perturbation strength must be positive, got {strength}")

    if rng is None:
        rng = np.random.default_rng(seed)

    keys = sorted(reference_state.keys())
    # Flatten honest delta
    deltas_list: list[torch.Tensor] = []
    shapes: list[tuple[str, torch.Size, torch.dtype]] = []
    for k in keys:
        t_ref = reference_state[k]
        t_honest = honest_state[k]
        if torch.is_floating_point(t_honest):
            d = (t_honest.to(torch.float64) - t_ref.to(torch.float64)).reshape(-1)
            deltas_list.append(d)
            shapes.append((k, t_honest.shape, t_honest.dtype))

    if not deltas_list:
        return copy.deepcopy(honest_state)

    flat_delta = torch.cat(deltas_list, dim=0)
    honest_norm = torch.linalg.norm(flat_delta).item()

    # Generate random gaussian perturbation vector
    z_np = rng.standard_normal(size=flat_delta.numel())
    z_tensor = torch.from_numpy(z_np).to(torch.float64)
    z_norm = torch.linalg.norm(z_tensor).item()

    if z_norm > 0.0 and math.isfinite(z_norm):
        u = z_tensor / z_norm
    else:
        u = torch.zeros_like(flat_delta)

    # Relative perturbation norm if nonzero, else absolute fallback
    scale = (strength * honest_norm) if honest_norm > 1e-9 else float(strength)
    flat_attack = flat_delta + scale * u

    # Reconstruct state dict
    attacked_state: dict[str, torch.Tensor] = {}
    offset = 0
    for k, shape, dtype in shapes:
        numel = shape.numel()
        param_delta = flat_attack[offset : offset + numel].reshape(shape)
        attacked_state[k] = (reference_state[k].to(torch.float64) + param_delta).to(dtype)
        offset += numel

    # Copy non-floating point buffers as-is
    for k in reference_state:
        if k not in attacked_state:
            attacked_state[k] = honest_state[k].clone()

    return attacked_state
