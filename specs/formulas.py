"""Phase 1 executable mathematical helpers (specification-level only).

Each function implements one numbered equation from docs/PHASE1_SPEC.md
(equation numbers in docstrings). Pure Python, no dependencies, no simulator /
RL / federated / attack / dashboard logic. Behavioural trust dynamics,
DQN training, and federated orchestration belong to Phases 2-8.
"""

from __future__ import annotations

import math


def clamp01(x: float) -> float:
    """EQ-0: clamp any value into [0, 1]."""
    return max(0.0, min(1.0, x))


def shannon_rate(bandwidth_hz: float, snr: float) -> float:
    """EQ-3: channel rate r = B * log2(1 + SNR) in bits/s."""
    if bandwidth_hz <= 0 or snr < 0:
        raise ValueError("bandwidth must be > 0 and SNR >= 0")
    return bandwidth_hz * math.log2(1.0 + snr)


def transmission_latency(data_bits: float, rate_bps: float, handshake_s: float = 0.0) -> float:
    """EQ-4: L_tx = d / r + handshake delta."""
    if data_bits < 0 or rate_bps <= 0 or handshake_s < 0:
        raise ValueError("invalid transmission parameters")
    return data_bits / rate_bps + handshake_s


def exec_latency(cycles: float, freq_hz: float, queue_wait_s: float = 0.0) -> float:
    """EQ-5: L_exe = c / f + queue wait."""
    if cycles < 0 or freq_hz <= 0 or queue_wait_s < 0:
        raise ValueError("invalid execution parameters")
    return cycles / freq_hz + queue_wait_s


def total_latency(tx_s: float, exe_s: float) -> float:
    """EQ-6: total latency is the sum of transmission and execution parts."""
    return tx_s + exe_s


def task_energy(tx_power_w: float, tx_s: float, kappa: float, cycles: float, freq_hz: float) -> float:
    """EQ-7: E = P_tx * L_tx + kappa * c * f^2."""
    if tx_power_w < 0 or tx_s < 0 or kappa < 0 or cycles < 0 or freq_hz <= 0:
        raise ValueError("invalid energy parameters")
    return tx_power_w * tx_s + kappa * cycles * freq_hz**2


def task_outcome(latency_s: float, deadline_s: float, path_available: bool) -> dict:
    """EQ-8: success iff within deadline and return path available."""
    success = bool(path_available and latency_s <= deadline_s)
    missed = bool(path_available and latency_s > deadline_s)
    failed = bool(not path_available)
    return {"success": success, "deadline_missed": missed, "failed": failed}


def minmax_norm(x: float, lo: float, hi: float) -> float:
    """EQ-9 helper: normalise x into [0, 1] given documented bounds."""
    if hi <= lo:
        raise ValueError("upper bound must exceed lower bound")
    return clamp01((x - lo) / (hi - lo))


def compute_reward(
    latency_n: float,
    energy_n: float,
    success: bool,
    failed: bool,
    deadline_missed: bool,
    trust_dest: float,
    w_l: float = 1.0,
    w_e: float = 0.3,
    w_s: float = 2.0,
    c_fail: float = 3.0,
    c_dl: float = 2.0,
    w_t: float = 0.5,
) -> float:
    """EQ-11: reward with fixed signs (latency/energy/failure penalised)."""
    return (
        -(w_l * latency_n + w_e * energy_n)
        + w_s * float(success)
        - c_fail * float(failed)
        - c_dl * float(deadline_missed)
        + w_t * clamp01(trust_dest)
    )


def behavioural_trust(signals: dict, weights: dict | None = None) -> float:
    """EQ-12: weighted mean of [0, 1] signals (success/timeliness/resources/reliability)."""
    if weights is None:
        weights = {"success": 0.4, "timeliness": 0.25, "resources": 0.2, "reliability": 0.15}
    if set(signals) != set(weights):
        raise ValueError("signals and weights must share the same keys")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1")
    return clamp01(sum(weights[k] * clamp01(signals[k]) for k in weights))


def temporal_update(prev: float, obs: float, lam: float = 0.9) -> float:
    """EQ-13: T(t) = lam * T(t-1) + (1 - lam) * observation."""
    if not 0.0 <= lam < 1.0:
        raise ValueError("smoothing lam must be in [0, 1)")
    return clamp01(lam * clamp01(prev) + (1.0 - lam) * clamp01(obs))


def idle_ageing(prev: float, prior: float = 0.5, rho: float = 0.99) -> float:
    """EQ-13 disconnect rule: no observation -> age neutrally toward prior, never malice."""
    if not 0.0 < rho <= 1.0:
        raise ValueError("ageing rho must be in (0, 1]")
    return clamp01(rho * clamp01(prev) + (1.0 - rho) * clamp01(prior))


def norm_plausibility(norm: float, median_norm: float, clip_c: float = 3.0) -> float:
    """EQ-14: 1 if within C * median norm, else proportionally down-weighted."""
    if norm < 0 or median_norm <= 0 or clip_c <= 0:
        raise ValueError("invalid norm parameters")
    if norm <= clip_c * median_norm:
        return 1.0
    return clamp01(clip_c * median_norm / norm)


def directional_agreement(cosine: float) -> float:
    """EQ-15: map cosine similarity [-1, 1] to trust [0, 1]."""
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("cosine must be in [-1, 1]")
    return (1.0 + cosine) / 2.0


def staleness_trust(age_rounds: int) -> float:
    """EQ-16: 1 / (1 + age); fresh updates score 1."""
    if age_rounds < 0:
        raise ValueError("age must be >= 0")
    return 1.0 / (1.0 + age_rounds)


def copy_zero_flag(is_zero: bool, is_copy: bool) -> float:
    """EQ-17: 0 for zero/copied updates, else 1."""
    return 0.0 if (is_zero or is_copy) else 1.0


def model_update_trust(indicators: dict, weights: dict | None = None) -> float:
    """EQ-19: weighted mean of [0, 1] update-quality indicators."""
    if weights is None:
        weights = {"norm": 0.25, "direction": 0.3, "staleness": 0.15, "originality": 0.2, "progress": 0.1}
    if set(indicators) != set(weights):
        raise ValueError("indicators and weights must share the same keys")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1")
    return clamp01(sum(weights[k] * clamp01(indicators[k]) for k in weights))


def combine_trust(t_beh: float, t_model: float, alpha: float = 0.6) -> float:
    """EQ-20 (binding): T_final = alpha * T_beh + (1 - alpha) * T_model."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return clamp01(alpha * clamp01(t_beh) + (1.0 - alpha) * clamp01(t_model))


def fedavg(thetas: list[list[float]], counts: list[float]) -> list[float]:
    """EQ-21: sample-weighted parameter average."""
    if not thetas or len(thetas) != len(counts) or any(c < 0 for c in counts):
        raise ValueError("need one non-negative count per client update")
    total = sum(counts)
    if total <= 0:
        raise ValueError("total sample count must be > 0")
    dim = len(thetas[0])
    if any(len(v) != dim for v in thetas):
        raise ValueError("all updates must share dimensionality")
    return [sum(v[d] * c for v, c in zip(thetas, counts)) / total for d in range(dim)]


def trust_weighted_aggregate(
    thetas: list[list[float]],
    trusts: list[float],
    threshold: float = 0.3,
    fallback: list[float] | None = None,
) -> tuple[list[float], list[int]]:
    """EQ-22/23: trust-weighted mean over clients above threshold.

    Returns (aggregated, eligible_indices). Empty eligibility -> returns the
    fallback (last known good global model) unchanged; never divides by zero.
    """
    if len(thetas) != len(trusts):
        raise ValueError("need one trust score per client update")
    eligible = [i for i, t in enumerate(trusts) if clamp01(t) >= threshold]
    if not eligible:
        if fallback is None:
            raise ValueError("no eligible clients and no fallback model provided")
        return list(fallback), []
    total = sum(clamp01(trusts[i]) for i in eligible)
    dim = len(thetas[0])
    agg = [sum(thetas[i][d] * clamp01(trusts[i]) for i in eligible) / total for d in range(dim)]
    return agg, eligible
