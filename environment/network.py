"""Network model (SPEC EQ-2/EQ-3): path loss, Shannon rate, link eligibility.

Resolution of SPEC U3: SNR follows a power-law path loss
    snr(d) = tx_power_w * pl_k0 / ((d + 1)^pl_gamma * noise_w),
floored to [snr_min, ...] for numerical safety. This is a synthetic,
documented stand-in — not a calibrated fading/MAC model (per A4).
"""

from __future__ import annotations

from specs.formulas import shannon_rate

SNR_MIN = 1e-3  # floor keeps rates finite at long range


def snr_linear(
    dist_m: float,
    tx_power_w: float,
    pl_k0: float,
    pl_gamma: float,
    noise_w: float,
) -> float:
    """Distance-based SNR (linear, not dB). d+1 avoids the d=0 singularity."""
    if dist_m < 0 or tx_power_w <= 0 or noise_w <= 0 or pl_gamma <= 0:
        raise ValueError("invalid path-loss parameters")
    snr = tx_power_w * pl_k0 / (((dist_m + 1.0) ** pl_gamma) * noise_w)
    return max(SNR_MIN, snr)


def link_rate_hz(
    dist_m: float,
    bandwidth_hz: float,
    tx_power_w: float,
    pl_k0: float,
    pl_gamma: float,
    noise_w: float,
) -> float:
    """EQ-3: Shannon rate for a link of length dist_m."""
    return shannon_rate(bandwidth_hz, snr_linear(dist_m, tx_power_w, pl_k0, pl_gamma, noise_w))


def link_available(rng, reliability: float) -> bool:
    """Intermittent-connectivity draw: link up with probability `reliability`."""
    return bool(rng.random() < reliability)
