"""Integration tests for Phase 8: Attack pipeline with FedServer aggregation and strategy response."""

import pytest
import torch

from attacks.base import AttackType, ScenarioAssignment
from attacks.pipeline import AttackPipeline
from federated.server import FedServer


def make_tensor_dict(d: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}


def test_sign_flip_integration_with_trust_threshold():
    """Verify that a sign-flip update receives low directional agreement and is excluded or down-weighted."""
    tmpl = make_tensor_dict({"w": [0.0, 0.0]})
    server_thresh = FedServer(tmpl, strategy="trust_threshold", tau_exc=0.45, alpha=0.5)
    server_fedavg = FedServer(tmpl, strategy="fedavg")


    # 3 clients: c0 and c1 honest (moving in direction +[1, 1]), c2 sign-flipped (moving in direction -[2, 2])
    scen = ScenarioAssignment.create(num_clients=3, malicious_fraction=0.33, attack_type=AttackType.SIGN_FLIP, seed=42)
    # Force c2 to be the malicious one
    scen.malicious_client_ids = [2]
    scen.client_labels[0].is_malicious = False
    scen.client_labels[1].is_malicious = False
    scen.client_labels[2].is_malicious = True
    scen.client_labels[2].attack_type = AttackType.SIGN_FLIP
    scen.client_labels[2].attack_strength = 1.0

    pipe = AttackPipeline(scen, seed=42)

    # Updates
    u0 = {"client_id": 0, "round_trained": 0, "num_samples": 10, "state": make_tensor_dict({"w": [1.0, 1.0]})}
    u1 = {"client_id": 1, "round_trained": 0, "num_samples": 10, "state": make_tensor_dict({"w": [1.2, 1.2]})}
    # Honest update for c2 was [1.0, 1.0]
    u2_honest = {"client_id": 2, "round_trained": 0, "num_samples": 10, "state": make_tensor_dict({"w": [1.0, 1.0]})}

    # Transform updates through attack pipeline
    u0_att, _ = pipe.transform_update(u0, tmpl, 0)
    u1_att, _ = pipe.transform_update(u1, tmpl, 0)
    u2_att, audit = pipe.transform_update(u2_honest, tmpl, 0)

    assert audit["is_malicious"] is True
    assert audit["attack_applied"] is True
    # Sign-flipped state: ref is [0, 0], honest delta was [1.0, 1.0], attacked is [0, 0] - 1.0 * [1.0, 1.0] = [-1.0, -1.0]
    assert u2_att["state"]["w"].tolist() == pytest.approx([-1.0, -1.0])

    updates = [u0_att, u1_att, u2_att]

    # Under FedAvg: c2 is blindly accepted and poisons the model
    log_fedavg = server_fedavg.aggregate_round(updates, current_round=0)
    assert 2 in log_fedavg["accepted"]
    # Model pulled down: (1.0 + 1.2 - 1.0) / 3 = 0.4
    assert server_fedavg.get_global()["w"].tolist() == pytest.approx([0.4, 0.4])

    # Under trust_threshold: c2 has negative cosine with peer aggregate, lower trust
    # Provide neutral behavioural trust
    beh = {0: 0.8, 1: 0.8, 2: 0.2}
    log_thresh = server_thresh.aggregate_round(updates, current_round=0, behavioural_trusts=beh)
    rep2 = next(r for r in log_thresh["trust_reports"] if r["client_id"] == 2)
    # Directional agreement m2 for c2 is 0.0 (anti-aligned with mean delta [0.4, 0.4])
    assert rep2["m2_direction"] == pytest.approx(0.0)
    # Client 2 excluded
    assert 2 in log_thresh["excluded"]
    assert 2 not in log_thresh["accepted"]
    # Global model aggregated over c0 and c1 only: (1.0 + 1.2) / 2 = 1.1
    assert server_thresh.get_global()["w"].tolist() == pytest.approx([1.1, 1.1])


def test_non_finite_update_rejection_safety():

    """Verify that an adversarial non-finite update (NaN/Inf) does not corrupt the global model."""
    tmpl = make_tensor_dict({"w": [1.0, 1.0]})
    server = FedServer(tmpl, strategy="trust_threshold")

    # Attacked update with NaN
    u_nan = {
        "client_id": 0,
        "round_trained": 0,
        "num_samples": 10,
        "state": make_tensor_dict({"w": [float("nan"), 1.0]}),
    }
    log = server.aggregate_round([u_nan], current_round=0)
    assert log["status"].startswith("skipped")
    # Global model preserved without NaN contamination
    assert torch.equal(server.get_global()["w"], tmpl["w"])
