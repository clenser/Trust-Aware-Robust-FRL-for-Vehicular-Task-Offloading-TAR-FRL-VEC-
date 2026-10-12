"""Model-update trust module (SPEC §7, EQ-14 through EQ-20).

Calculates multi-dimensional update trust indicators for client model updates:
- EQ-14: Norm plausibility (m1)
- EQ-15: Directional agreement (m2)
- EQ-16: Staleness trust (m3)
- EQ-17: Copy / zero flags (m4)
- EQ-18: Local progress proxy (m5)
- EQ-19: Composite model update trust (T_model)
- EQ-20: Combined multi-signal trust (T_final = alpha * T_beh + (1 - alpha) * T_model)

Design constraints:
- Input state dicts are converted to 1D flat floating-point parameter vectors.
- All indicators and final scores are clamped to [0.0, 1.0].
- Pure, deterministic, inspectable evaluations with detailed indicator reporting.
- Does not assume malice for honest non-IID variance or edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import torch

from specs import formulas as F

DEFAULT_GAMMA: dict[str, float] = {
    "norm": 0.25,
    "direction": 0.30,
    "staleness": 0.15,
    "originality": 0.20,
    "progress": 0.10,
}

DEFAULT_ALPHA: float = 0.60
DEFAULT_CLIP_C: float = 3.0
DEFAULT_TAU_EXC: float = 0.30
DEFAULT_EPS_TOL: float = 1e-7



def flatten_state_dict(
    state: dict[str, torch.Tensor],
    keys: list[str] | None = None,
) -> torch.Tensor:
    """Flatten floating-point tensors in a state dict into a 1D float64 tensor.

    Deterministic key ordering is guaranteed by sorting keys if not provided.
    Non-floating point buffers (e.g. integer step counters) are excluded.
    """
    if keys is None:
        keys = sorted(state.keys())
    tensors: list[torch.Tensor] = []
    for k in keys:
        if k not in state:
            raise ValueError(f"Missing key in state dict: {k}")
        t = state[k]
        if torch.is_floating_point(t):
            tensors.append(t.detach().cpu().to(torch.float64).reshape(-1))
    if not tensors:
        raise ValueError("State dict contains no floating-point parameters to flatten.")
    return torch.cat(tensors, dim=0)


def compute_param_delta(
    client_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute Delta_i = theta_i - theta_ref as a 1D flat float64 tensor."""
    keys = sorted(reference_state.keys())
    flat_client = flatten_state_dict(client_state, keys)
    flat_ref = flatten_state_dict(reference_state, keys)
    if flat_client.shape != flat_ref.shape:
        raise ValueError(f"Parameter shape mismatch: {flat_client.shape} != {flat_ref.shape}")
    return flat_client - flat_ref


def compute_norm_plausibility(
    delta_norm: float,
    median_norm: float,
    clip_c: float = DEFAULT_CLIP_C,
) -> float:
    """EQ-14: Evaluate parameter-delta norm against median peer norm.

    m1 = 1 if delta_norm <= clip_c * median_norm, else (clip_c * median_norm) / delta_norm.
    Special cases:
    - If median_norm <= 0:
        - If delta_norm == 0: 1.0 (consistent with median).
        - If delta_norm > 0: 0.0 (non-zero update when peer median is zero).
    - If delta_norm <= 0: 1.0.
    """
    if clip_c <= 0:
        raise ValueError("clip_c must be positive")
    if delta_norm < 0:
        raise ValueError("delta_norm cannot be negative")

    if median_norm <= 0:
        return 1.0 if delta_norm <= 0 else 0.0

    return F.norm_plausibility(delta_norm, median_norm, clip_c=clip_c)


def compute_directional_agreement(
    client_delta: torch.Tensor,
    reference_delta: torch.Tensor,
) -> float:
    """EQ-15: Cosine agreement mapped from [-1, 1] to [0, 1].

    m2 = (1 + cos(Delta_i, Delta_bar)) / 2.
    If either norm is zero or non-finite:
    - If both are zero: exact match -> cosine = 1.0 -> m2 = 1.0.
    - If one is zero: undefined -> neutral cosine = 0.0 -> m2 = 0.5.
    """
    c_64 = client_delta.detach().to(torch.float64)
    r_64 = reference_delta.detach().to(torch.float64)
    norm_c = torch.linalg.norm(c_64).item()
    norm_r = torch.linalg.norm(r_64).item()

    if norm_c == 0.0 and norm_r == 0.0:
        return 1.0
    if norm_c == 0.0 or norm_r == 0.0:
        return 0.5

    dot = torch.dot(c_64, r_64).item()
    denom = norm_c * norm_r
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.5

    cos = max(-1.0, min(1.0, dot / denom))
    return F.directional_agreement(cos)



def compute_staleness_trust(staleness_rounds: int) -> float:
    """EQ-16: 1 / (1 + staleness_rounds)."""
    return F.staleness_trust(max(0, staleness_rounds))


def compute_copy_zero_flag(
    client_delta: torch.Tensor,
    peer_deltas: list[torch.Tensor],
    tol: float = DEFAULT_EPS_TOL,
) -> float:
    """EQ-17: Copy and zero detection.

    m4 = 0.0 if:
    - ||Delta_i|| < tol (zero update / no local learning step).
    - ||Delta_i - Delta_j|| < tol for any peer j != i (bit-identical / copy).
    Otherwise 1.0.
    """
    norm_i = torch.linalg.norm(client_delta).item()
    if norm_i < tol:
        return 0.0

    for other_delta in peer_deltas:
        diff_norm = torch.linalg.norm(client_delta - other_delta).item()
        if diff_norm < tol:
            return 0.0

    return 1.0


def compute_local_progress(
    progress_val: float | None = None,
) -> float:
    """EQ-18: Local progress proxy score in [0, 1].

    If not provided (None or NaN), returns neutral prior 0.5.
    If provided, clamps to [0, 1].
    """
    if progress_val is None or not math.isfinite(progress_val):
        return 0.5
    return F.clamp01(float(progress_val))


@dataclass
class UpdateTrustReport:
    """Full breakdown of model update trust indicators and scores for an update."""
    client_id: int | str
    delta_norm: float
    median_norm: float
    m1_norm: float
    m2_direction: float
    m3_staleness: float
    m4_originality: float
    m5_progress: float
    t_model: float
    t_beh: float
    t_final: float
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "delta_norm": round(self.delta_norm, 6),
            "median_norm": round(self.median_norm, 6),
            "m1_norm": round(self.m1_norm, 4),
            "m2_direction": round(self.m2_direction, 4),
            "m3_staleness": round(self.m3_staleness, 4),
            "m4_originality": round(self.m4_originality, 4),
            "m5_progress": round(self.m5_progress, 4),
            "t_model": round(self.t_model, 4),
            "t_beh": round(self.t_beh, 4),
            "t_final": round(self.t_final, 4),
        }


def evaluate_model_updates(
    updates: list[dict],
    reference_state: dict[str, torch.Tensor],
    behavioural_trusts: dict[int | str, float] | None = None,
    weights: dict[str, float] | None = None,
    alpha: float = DEFAULT_ALPHA,
    clip_c: float = DEFAULT_CLIP_C,
    tol: float = DEFAULT_EPS_TOL,
) -> list[UpdateTrustReport]:
    """Evaluate EQ-14 through EQ-20 for a batch of participating client updates.

    Parameters:
    - updates: list of client update dicts containing:
        - "client_id": int | str
        - "state": dict[str, torch.Tensor]
        - "staleness": int (current_round - round_trained)
        - optional "progress": float (local progress proxy in [0, 1])
    - reference_state: state dict of the reference model (e.g. current global model)
    - behavioural_trusts: mapping of client_id -> T_beh in [0, 1] (defaults to 0.5 prior)
    - weights: indicator weights gamma (norm, direction, staleness, originality, progress)
    - alpha: combination weight for EQ-20 T_final = alpha * T_beh + (1 - alpha) * T_model
    - clip_c: multiplier for median norm plausibility bound
    - tol: float tolerance for zero / copy detection

    Returns:
    - list of UpdateTrustReport objects, one per update in the same order.
    """
    if not updates:
        return []

    if weights is None:
        weights = DEFAULT_GAMMA

    if behavioural_trusts is None:
        behavioural_trusts = {}

    # 1. Compute parameter deltas
    keys = sorted(reference_state.keys())
    deltas: list[torch.Tensor] = []
    norms: list[float] = []

    for u in updates:
        delta = compute_param_delta(u["state"], reference_state)
        deltas.append(delta)
        norms.append(torch.linalg.norm(delta).item())

    # 2. Reference norms and reference direction
    # Median norm of participating updates
    median_norm = float(np.median(norms)) if len(norms) > 0 else 0.0

    # Peer aggregate delta Delta_bar = (1 / |P|) * sum(Delta_j)
    mean_delta = torch.stack(deltas, dim=0).mean(dim=0)

    # 3. Evaluate each client
    reports: list[UpdateTrustReport] = []

    for i, u in enumerate(updates):
        cid = u["client_id"]
        d_i = deltas[i]
        norm_i = norms[i]

        # m1: Norm plausibility (EQ-14)
        m1 = compute_norm_plausibility(norm_i, median_norm, clip_c=clip_c)

        # m2: Directional agreement (EQ-15)
        # Reference direction is the peer aggregate mean_delta
        m2 = compute_directional_agreement(d_i, mean_delta)

        # m3: Staleness (EQ-16)
        staleness = int(u.get("staleness", 0))
        m3 = compute_staleness_trust(staleness)

        # m4: Copy / zero flags (EQ-17)
        other_deltas = [deltas[j] for j in range(len(deltas)) if j != i]
        m4 = compute_copy_zero_flag(d_i, other_deltas, tol=tol)

        # m5: Local progress proxy (EQ-18)
        prog_val = u.get("progress", None)
        m5 = compute_local_progress(prog_val)

        # EQ-19: T_model
        indicators = {
            "norm": m1,
            "direction": m2,
            "staleness": m3,
            "originality": m4,
            "progress": m5,
        }
        t_model = F.model_update_trust(indicators, weights)

        # EQ-20: T_final
        t_beh = F.clamp01(float(behavioural_trusts.get(cid, 0.5)))
        t_final = F.combine_trust(t_beh, t_model, alpha=alpha)

        rep = UpdateTrustReport(
            client_id=cid,
            delta_norm=norm_i,
            median_norm=median_norm,
            m1_norm=m1,
            m2_direction=m2,
            m3_staleness=m3,
            m4_originality=m4,
            m5_progress=m5,
            t_model=t_model,
            t_beh=t_beh,
            t_final=t_final,
            details={
                "staleness_rounds": staleness,
                "progress_raw": prog_val if prog_val is not None else float("nan"),
            },
        )
        reports.append(rep)

    return reports
