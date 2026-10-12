"""Unit tests for Phase 8 attack components (Sign-flip, Perturbation, Free-riding, and Label Isolation)."""

import math
import numpy as np
import pytest
import torch

from attacks.base import AttackType, ClientScenarioLabel, ScenarioAssignment
from attacks.free_riding import apply_copy_update, apply_zero_update, create_stale_update
from attacks.model_poisoning import apply_perturbation_attack, apply_sign_flip_attack
from attacks.pipeline import AttackPipeline
from trust.model_update import compute_copy_zero_flag, compute_directional_agreement, compute_staleness_trust


def make_tensor_dict(d: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}


def test_sign_flip_exact_math():
    ref = make_tensor_dict({"w": [1.0, 2.0], "b": [0.5]})
    honest = make_tensor_dict({"w": [1.5, 3.0], "b": [1.0]})  # Delta = [0.5, 1.0], [0.5]

    # With strength s = 1.0 -> Delta_attack = -Delta_honest = [-0.5, -1.0], [-0.5]
    # theta_attack = ref + Delta_attack = [0.5, 1.0], [0.0]
    attack_s1 = apply_sign_flip_attack(honest, ref, strength=1.0)
    assert attack_s1["w"].tolist() == pytest.approx([0.5, 1.0])
    assert attack_s1["b"].tolist() == pytest.approx([0.0])

    # Directional agreement should be 0.0 (anti-aligned)
    delta_h = torch.tensor([0.5, 1.0, 0.5])
    delta_a = torch.tensor([-0.5, -1.0, -0.5])
    m2 = compute_directional_agreement(delta_a, delta_h)
    assert m2 == pytest.approx(0.0)

    # With strength s = 2.0 -> Delta_attack = -2 * Delta_honest = [-1.0, -2.0], [-1.0]
    # theta_attack = [0.0, 0.0], [-0.5]
    attack_s2 = apply_sign_flip_attack(honest, ref, strength=2.0)
    assert attack_s2["w"].tolist() == pytest.approx([0.0, 0.0])
    assert attack_s2["b"].tolist() == pytest.approx([-0.5])


def test_perturbation_reproducibility_and_norm():
    ref = make_tensor_dict({"w": [1.0, 2.0]})
    honest = make_tensor_dict({"w": [2.0, 4.0]})  # Delta = [1.0, 2.0], norm = sqrt(5) ~= 2.236

    rng1 = np.random.default_rng(100)
    rng2 = np.random.default_rng(100)

    p1 = apply_perturbation_attack(honest, ref, strength=1.5, rng=rng1)
    p2 = apply_perturbation_attack(honest, ref, strength=1.5, rng=rng2)

    # Deterministic given seed
    assert torch.equal(p1["w"], p2["w"])

    # Perturbed state differs from honest
    assert not torch.equal(p1["w"], honest["w"])

    # Zero honest delta fallback
    honest_zero = make_tensor_dict({"w": [1.0, 2.0]})
    rng_zero = np.random.default_rng(200)
    p_zero = apply_perturbation_attack(honest_zero, ref, strength=1.0, rng=rng_zero)
    # Attack delta norm should be approximately 1.0
    delta_norm = torch.linalg.norm(p_zero["w"] - ref["w"]).item()
    assert delta_norm == pytest.approx(1.0, abs=1e-5)


def test_zero_update_flagged_by_originality():
    ref = make_tensor_dict({"w": [1.0, 2.0]})
    zero_state = apply_zero_update(ref)
    assert torch.equal(zero_state["w"], ref["w"])

    delta = zero_state["w"] - ref["w"]
    m4 = compute_copy_zero_flag(delta, [])
    assert m4 == 0.0  # Zero update flagged!


def test_copy_update_flagged_by_originality():
    ref = make_tensor_dict({"w": [0.0, 0.0]})
    peer_honest = make_tensor_dict({"w": [1.5, 2.5]})
    copied_state = apply_copy_update(peer_honest)

    delta_c = copied_state["w"] - ref["w"]
    delta_p = peer_honest["w"] - ref["w"]

    # When compared against peer delta, copy flag triggers
    m4 = compute_copy_zero_flag(delta_c, [delta_p])
    assert m4 == 0.0  # Copy update flagged!


def test_stale_replay_staleness_calculation():
    hist_u = {
        "client_id": 0,
        "round_trained": 0,
        "num_samples": 10,
        "staleness": 0,
        "state": make_tensor_dict({"w": [1.0, 1.0]}),
    }
    # Replay in round 3 with staleness_age 2 -> round_trained should be 1
    stale_u = create_stale_update(hist_u, staleness_age=2, current_round=3)
    assert stale_u["round_trained"] == 1
    assert stale_u["staleness"] == 2

    # Staleness score EQ-16: 1 / (1 + 2) = 1/3
    m3 = compute_staleness_trust(stale_u["staleness"])
    assert m3 == pytest.approx(1.0 / 3.0)


def test_ground_truth_label_isolation():
    """Verify ground-truth malicious labels are held by ScenarioAssignment and NEVER in client update dict."""
    scen = ScenarioAssignment.create(
        num_clients=4,
        malicious_fraction=0.5,
        attack_type=AttackType.SIGN_FLIP,
        seed=42,
    )
    assert len(scen.malicious_client_ids) == 2

    pipe = AttackPipeline(scen, seed=42)
    ref = make_tensor_dict({"w": [0.0, 0.0]})
    honest_u = {
        "client_id": scen.malicious_client_ids[0],
        "round_trained": 0,
        "num_samples": 10,
        "staleness": 0,
        "state": make_tensor_dict({"w": [1.0, 2.0]}),
    }

    attacked_u, audit = pipe.transform_update(honest_u, ref, current_round=0)

    # Server-bound attacked_u must NOT contain any label or attack keys
    forbidden_keys = {"is_malicious", "attack_type", "attack_strength", "attack_applied"}
    assert set(attacked_u.keys()).isdisjoint(forbidden_keys)

    # Only audit dict holds ground truth
    assert audit["is_malicious"] is True
    assert audit["attack_type"] == "sign_flip"
