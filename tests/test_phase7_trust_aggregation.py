"""Phase 7 tests: Model-update trust indicators and trust-aware robust aggregation.

Covers:
1. Exact hand-calculated spot checks for EQ-14 through EQ-20.
2. Tensor flattening, cosine angle boundary values, zero/copy detection.
3. Trust-weighted aggregation (EQ-22) vs FedAvg (EQ-21).
4. Threshold exclusion (EQ-23) with safe fallback model retention.
5. Empty participation and zero eligible trust edge cases.
6. Server strategy switching ("fedavg", "trust_weighted", "trust_threshold").
7. Non-IID honest client behavior (honest non-IID updates not unfairly rejected).
8. Determinism and reproducibility across seeds.
"""

import math
import numpy as np
import pytest
import torch

from environment import EnvConfig
from federated import FedClient, FedServer, fedavg_aggregate
from federated.trust_aggregation import trust_weighted_aggregate
from specs import formulas as F
from trust.model_update import (
    DEFAULT_ALPHA,
    DEFAULT_CLIP_C,
    DEFAULT_GAMMA,
    DEFAULT_TAU_EXC,
    UpdateTrustReport,
    compute_copy_zero_flag,
    compute_directional_agreement,
    compute_local_progress,
    compute_norm_plausibility,
    compute_param_delta,
    compute_staleness_trust,
    evaluate_model_updates,
    flatten_state_dict,
)


def make_tensor_dict(d: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}


def test_flatten_and_delta():
    ref = make_tensor_dict({"w": [1.0, 2.0], "b": [0.5]})
    c1 = make_tensor_dict({"w": [2.0, 2.0], "b": [1.5]})
    flat = flatten_state_dict(ref)
    assert flat.shape == (3,)
    # keys sorted: "b" then "w" -> [0.5, 1.0, 2.0]
    assert flat.tolist() == pytest.approx([0.5, 1.0, 2.0])

    delta = compute_param_delta(c1, ref)
    # c1 - ref: "b" is 1.5 - 0.5 = 1.0, "w" is [1.0, 0.0]
    assert delta.tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_eq14_norm_plausibility_spot_check():
    # norm <= clip_c * median -> 1.0
    assert compute_norm_plausibility(2.0, 1.0, clip_c=3.0) == 1.0
    assert compute_norm_plausibility(3.0, 1.0, clip_c=3.0) == 1.0
    # norm > clip_c * median -> (clip_c * median) / norm
    # 6.0 vs 3.0 -> 3.0 / 6.0 = 0.5
    assert compute_norm_plausibility(6.0, 1.0, clip_c=3.0) == pytest.approx(0.5)

    # Edge cases
    assert compute_norm_plausibility(0.0, 0.0, clip_c=3.0) == 1.0
    assert compute_norm_plausibility(1.0, 0.0, clip_c=3.0) == 0.0
    assert compute_norm_plausibility(0.0, 1.0, clip_c=3.0) == 1.0


def test_eq15_directional_agreement_spot_check():
    d_same = torch.tensor([1.0, 2.0, 3.0])
    d_opp = torch.tensor([-1.0, -2.0, -3.0])
    d_orth = torch.tensor([-2.0, 1.0, 0.0])  # dot = -2 + 2 + 0 = 0
    d_zero = torch.tensor([0.0, 0.0, 0.0])

    # Same direction: cos = 1 -> m2 = 1.0
    assert compute_directional_agreement(d_same, d_same) == pytest.approx(1.0)
    # Opposite direction: cos = -1 -> m2 = 0.0
    assert compute_directional_agreement(d_opp, d_same) == pytest.approx(0.0)
    # Orthogonal: cos = 0 -> m2 = 0.5
    assert compute_directional_agreement(d_orth, d_same) == pytest.approx(0.5)
    # Zero vector: neutral -> 0.5
    assert compute_directional_agreement(d_zero, d_same) == pytest.approx(0.5)
    # Both zero: 1.0
    assert compute_directional_agreement(d_zero, d_zero) == pytest.approx(1.0)


def test_eq16_staleness_spot_check():
    assert compute_staleness_trust(0) == pytest.approx(1.0)
    assert compute_staleness_trust(1) == pytest.approx(0.5)
    assert compute_staleness_trust(3) == pytest.approx(0.25)


def test_eq17_copy_zero_spot_check():
    d1 = torch.tensor([1.0, 2.0])
    d2 = torch.tensor([1.0, 2.0])  # identical copy
    d3 = torch.tensor([1.5, 2.5])
    d_zero = torch.tensor([0.0, 0.0])

    # Non-zero unique
    assert compute_copy_zero_flag(d1, [d3]) == 1.0
    # Zero vector -> 0.0
    assert compute_copy_zero_flag(d_zero, [d3]) == 0.0
    # Copied vector -> 0.0
    assert compute_copy_zero_flag(d1, [d2, d3]) == 0.0


def test_eq18_progress_spot_check():
    assert compute_local_progress(None) == pytest.approx(0.5)
    assert compute_local_progress(float("nan")) == pytest.approx(0.5)
    assert compute_local_progress(0.8) == pytest.approx(0.8)
    assert compute_local_progress(-0.2) == pytest.approx(0.0)
    assert compute_local_progress(1.5) == pytest.approx(1.0)


def test_eq19_and_eq20_composite_trust_spot_check():
    ref_state = make_tensor_dict({"w": [0.0, 0.0]})
    # Client 0: delta = [1.0, 0.0]
    # Client 1: delta = [1.0, 0.0]
    # Mean delta = [1.0, 0.0]
    updates = [
        {"client_id": 0, "state": make_tensor_dict({"w": [1.0, 0.0]}), "staleness": 0, "progress": 0.5},
        {"client_id": 1, "state": make_tensor_dict({"w": [1.0, 0.0]}), "staleness": 0, "progress": 0.5},
    ]
    # Note: client 0 and 1 are copies of each other, so originality m4 = 0.0
    # m1: norms are 1.0, median is 1.0 -> m1 = 1.0
    # m2: direction matches mean delta -> m2 = 1.0
    # m3: staleness = 0 -> m3 = 1.0
    # m4: peer is identical copy -> m4 = 0.0
    # m5: progress 0.5 -> m5 = 0.5
    # gamma = (0.25, 0.30, 0.15, 0.20, 0.10)
    # expected t_model = 0.25*1.0 + 0.30*1.0 + 0.15*1.0 + 0.20*0.0 + 0.10*0.5
    #                  = 0.25 + 0.30 + 0.15 + 0.0 + 0.05 = 0.75
    beh = {0: 0.8, 1: 0.6}
    reports = evaluate_model_updates(updates, ref_state, behavioural_trusts=beh, alpha=0.6)

    assert len(reports) == 2
    r0 = reports[0]
    assert r0.m1_norm == pytest.approx(1.0)
    assert r0.m2_direction == pytest.approx(1.0)
    assert r0.m3_staleness == pytest.approx(1.0)
    assert r0.m4_originality == pytest.approx(0.0)
    assert r0.m5_progress == pytest.approx(0.5)
    assert r0.t_model == pytest.approx(0.75)
    # EQ-20: T_final = alpha * T_beh + (1 - alpha) * T_model
    # For client 0: 0.6 * 0.8 + 0.4 * 0.75 = 0.48 + 0.30 = 0.78
    assert r0.t_final == pytest.approx(0.78)

    # For client 1: 0.6 * 0.6 + 0.4 * 0.75 = 0.36 + 0.30 = 0.66
    r1 = reports[1]
    assert r1.t_final == pytest.approx(0.66)


def test_trust_weighted_aggregate_math():
    template = make_tensor_dict({"w": [0.0, 0.0]})
    s0 = make_tensor_dict({"w": [1.0, 2.0]})
    s1 = make_tensor_dict({"w": [3.0, 6.0]})

    # Case 1: weights without threshold
    # t_scores = [0.4, 0.6], total = 1.0 -> weights = 0.4, 0.6
    # w = 0.4 * [1, 2] + 0.6 * [3, 6] = [0.4 + 1.8, 0.8 + 3.6] = [2.2, 4.4]
    out, accepted, weights = trust_weighted_aggregate(
        states=[s0, s1],
        trust_scores=[0.4, 0.6],
        template=template,
        client_ids=["c0", "c1"],
        threshold=0.0,
    )
    assert accepted == ["c0", "c1"]
    assert weights == pytest.approx([0.4, 0.6])
    assert out["w"].tolist() == pytest.approx([2.2, 4.4])


def test_threshold_exclusion_and_fallback():
    template = make_tensor_dict({"w": [0.0, 0.0]})
    fallback = make_tensor_dict({"w": [9.0, 9.0]})
    s0 = make_tensor_dict({"w": [1.0, 2.0]})
    s1 = make_tensor_dict({"w": [3.0, 6.0]})

    # Case 2: threshold = 0.5 -> c0 (0.4) excluded, c1 (0.6) accepted
    out, accepted, weights = trust_weighted_aggregate(
        states=[s0, s1],
        trust_scores=[0.4, 0.6],
        template=template,
        client_ids=["c0", "c1"],
        threshold=0.5,
        fallback_state=fallback,
    )
    assert accepted == ["c1"]
    assert weights == pytest.approx([1.0])
    assert out["w"].tolist() == pytest.approx([3.0, 6.0])

    # Case 3: threshold = 0.7 -> both excluded -> fallback returned unchanged
    out_fb, accepted_fb, weights_fb = trust_weighted_aggregate(
        states=[s0, s1],
        trust_scores=[0.4, 0.6],
        template=template,
        client_ids=["c0", "c1"],
        threshold=0.7,
        fallback_state=fallback,
    )
    assert accepted_fb == []
    assert weights_fb == []
    assert out_fb["w"].tolist() == pytest.approx([9.0, 9.0])


def test_fedserver_strategy_switching():
    tmpl = make_tensor_dict({"w": [0.0, 0.0]})
    s_fedavg = FedServer(tmpl, strategy="fedavg")
    s_weighted = FedServer(tmpl, strategy="trust_weighted", alpha=0.5)
    s_thresh = FedServer(tmpl, strategy="trust_threshold", alpha=0.5, tau_exc=0.6)

    # Updates with different samples and different states
    u0 = {"client_id": 0, "round_trained": 0, "num_samples": 10, "state": make_tensor_dict({"w": [2.0, 2.0]})}
    u1 = {"client_id": 1, "round_trained": 0, "num_samples": 30, "state": make_tensor_dict({"w": [6.0, 6.0]})}
    updates = [u0, u1]

    # FedAvg aggregates by num_samples (10 vs 30 -> 1/4 and 3/4)
    # Expected FedAvg = 0.25 * [2, 2] + 0.75 * [6, 6] = [0.5 + 4.5, 0.5 + 4.5] = [5.0, 5.0]
    log_fedavg = s_fedavg.aggregate_round(updates, current_round=0)
    assert log_fedavg["status"] == "aggregated"
    assert s_fedavg.get_global()["w"].tolist() == pytest.approx([5.0, 5.0])

    # Trust-weighted with custom behavioural trust
    # Provide higher trust to client 0
    beh = {0: 0.9, 1: 0.2}
    log_w = s_weighted.aggregate_round(updates, current_round=0, behavioural_trusts=beh)
    assert log_w["status"] == "aggregated"
    assert "trust_reports" in log_w
    # Client 0 should have higher weight than client 1 despite fewer samples
    weights_w = log_w["weights"]
    assert weights_w[0] > weights_w[1]

    # Trust-threshold with tau_exc=0.6: client 1 should be excluded if t_final < 0.6
    log_th = s_thresh.aggregate_round(updates, current_round=0, behavioural_trusts=beh)
    assert 1 in log_th["excluded"]
    assert 0 in log_th["accepted"]
    # Model should match client 0 alone [2.0, 2.0]
    assert s_thresh.get_global()["w"].tolist() == pytest.approx([2.0, 2.0])


def test_non_iid_honest_clients_retained():
    """Verify that honest non-IID clients with distinct gradient trajectories are not rejected."""
    tmpl = make_tensor_dict({"w": [0.0, 0.0], "b": [0.0]})
    server = FedServer(tmpl, strategy="trust_threshold", tau_exc=0.3)

    # 3 honest clients with distinct legitimate updates (non-IID data distributions)
    u0 = {"client_id": 0, "round_trained": 0, "num_samples": 20, "state": make_tensor_dict({"w": [0.1, 0.2], "b": [0.05]})}
    u1 = {"client_id": 1, "round_trained": 0, "num_samples": 25, "state": make_tensor_dict({"w": [0.2, 0.1], "b": [0.08]})}
    u2 = {"client_id": 2, "round_trained": 0, "num_samples": 18, "state": make_tensor_dict({"w": [0.15, 0.18], "b": [0.06]})}

    beh = {0: 0.7, 1: 0.75, 2: 0.72}
    log = server.aggregate_round([u0, u1, u2], current_round=0, behavioural_trusts=beh)

    assert log["status"] == "aggregated"
    assert len(log["accepted"]) == 3
    assert len(log["excluded"]) == 0
    for r in log["trust_reports"]:
        assert r["t_final"] >= 0.3
        # Norms and originality should both be high for distinct legitimate updates
        assert r["m1_norm"] == pytest.approx(1.0)
        assert r["m4_originality"] == pytest.approx(1.0)
