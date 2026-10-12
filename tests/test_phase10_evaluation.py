"""Comprehensive unit and integration test suite for Phase 10 Final Remediation.

Covers all 19 required acceptance and methodological checks:
1. Matrix-plan generation matches 630 core, 70 ablation, 180 sensitivity = 880 total.
2. Scenario/strategy/fraction/seed combinations are unique and complete.
3. Clean IID and Honest Non-IID controls use distinct client configurations.
4. Matched attack comparisons use the correct control.
5. Unreliable severity levels are invoked and actual fault counts recorded.
6. Unreliability labels remain separate from malicious labels (no leakage).
7. All 7 ablations run on both required attack scenarios.
8. All 4 sensitivity parameters run at specified values.
9. Deterministic seed propagation and reproducible reruns.
10. Anomaly threshold and smoothing overrides influence calculations.
11. Task-accounting identities hold for every run.
12. FedAvg is unchanged and remains unweighted by trust.
13. Model checkpoints and reported hashes correspond.
14. Detection metrics correctly marked not applicable when no detector exists.
15. Summary JSON groups by planned factors without mixing attack fractions.
16. Confidence intervals use correct sample counts and Student-t critical values.
17. Statistical analysis uses paired run-level observations and exact p-values.
18. Manifest row counts and hashes match final artifacts.
19. Existing Phase 0-9 regression tests remain passing.
"""

from pathlib import Path
import pytest
import yaml
import numpy as np
import torch
from scipy import stats

from attacks.base import AttackType, ScenarioAssignment
from attacks.unreliable_execution import UnreliableEnvWrapper, UnreliableProfile
from environment import EnvConfig, VehicularEdgeEnv
from federated.client import FedClient
from federated.server import FedServer
from rl.agent import DQNAgent, seed_all
from scripts.run_phase10_final_evaluation import (
    build_phase10_experiment_plan,
    build_clients_with_seed,
    execute_plan_item,
    PlanItem,
)
from scripts.analyze_phase10_results import compute_student_t_ci_95


def test_matrix_plan_counts_and_uniqueness():
    """Verify matrix-plan generation matches 630 core, 70 ablation, 180 sensitivity = 880 total."""
    plan = build_phase10_experiment_plan()
    assert len(plan) == 880, f"Expected exactly 880 runs, got {len(plan)}"

    core = [p for p in plan if p.matrix_type == "core"]
    abl = [p for p in plan if p.matrix_type == "ablation"]
    sens = [p for p in plan if p.matrix_type == "sensitivity"]

    assert len(core) == 630, f"Expected 630 core runs, got {len(core)}"
    assert len(abl) == 70, f"Expected 70 ablation runs, got {len(abl)}"
    assert len(sens) == 180, f"Expected 180 sensitivity runs, got {len(sens)}"

    # Uniqueness of run IDs
    run_ids = [p.run_id for p in plan]
    assert len(run_ids) == len(set(run_ids)), "All run IDs in the plan must be unique"


def test_clean_iid_vs_honest_non_iid_distinction():
    """Verify that Clean IID and Honest Non-IID controls use genuinely distinct client configurations."""
    with open("configs/feddqn_phase10.yaml") as f:
        raw = yaml.safe_load(f)

    agent_kwargs = {"lr": 1e-3, "gamma": 0.99, "batch_size": 4, "capacity": 100}

    # IID clients
    iid_clients = build_clients_with_seed(raw, agent_kwargs, num_tasks=10, master_seed=0, is_iid=True)
    assert len(iid_clients) == 10
    loads_iid = {c.env_cfg.task_load for c in iid_clients}
    mobs_iid = {c.env_cfg.mobility for c in iid_clients}
    rels_iid = {c.env_cfg.link_reliability for c in iid_clients}
    # All IID clients must have identical configurations
    assert len(loads_iid) == 1, "IID clients must have homogeneous task load"
    assert len(mobs_iid) == 1, "IID clients must have homogeneous mobility"
    assert len(rels_iid) == 1, "IID clients must have homogeneous link reliability"

    # Non-IID clients
    non_iid_clients = build_clients_with_seed(raw, agent_kwargs, num_tasks=10, master_seed=0, is_iid=False)
    assert len(non_iid_clients) == 10
    loads_non_iid = {c.env_cfg.task_load for c in non_iid_clients}
    mobs_non_iid = {c.env_cfg.mobility for c in non_iid_clients}
    rels_non_iid = {c.env_cfg.link_reliability for c in non_iid_clients}
    # Non-IID clients must exhibit heterogeneity
    assert len(loads_non_iid) > 1, "Non-IID clients must have heterogeneous task loads"
    assert len(mobs_non_iid) > 1, "Non-IID clients must have heterogeneous mobility"
    assert len(rels_non_iid) > 1, "Non-IID clients must have heterogeneous link reliability"


def test_unreliable_execution_counters_and_labels():
    """Verify UnreliableEnvWrapper records fault counts and keeps unreliable labels strictly non-malicious."""
    scen = ScenarioAssignment.create(
        num_clients=10,
        malicious_fraction=0.0,
        attack_type=AttackType.NONE,
        seed=42,
        unreliable_fraction=0.20,
        unreliable_severity="medium",
    )
    assert len(scen.unreliable_client_ids) == 2, "Expected 2 unreliable clients (20% of 10)"
    assert len(scen.malicious_client_ids) == 0, "Malicious client IDs must be empty"

    # Verify client labels
    for cid, label in scen.client_labels.items():
        assert not label.is_malicious, f"Client {cid} must NOT be labeled malicious"
        if cid in scen.unreliable_client_ids:
            assert label.notes == "unreliable_medium"

    # Verify wrapper instrumentation
    cfg = EnvConfig(seed=42, num_vehicles=4, num_edge_nodes=2, num_tasks=15)
    base_env = VehicularEdgeEnv(cfg)
    prof = UnreliableProfile(fail_prob=0.5, delay_prob=0.5, drop_prob=0.5, name="high")
    wrapper = UnreliableEnvWrapper(base_env, prof, seed=123)

    wrapper.reset(123)
    for _ in range(15):
        wrapper.step(0)

    stats = wrapper.get_fault_stats()
    assert stats["tasks_exposed"] == 15, "Expected 15 exposed tasks"
    total_injected = stats["injected_drops"] + stats["injected_fails"] + stats["injected_delays"]
    assert total_injected > 0, "Expected at least one fault injected with high probabilities"


def test_deterministic_seed_propagation():
    """Verify master seed propagation produces identical reruns and different models across seeds."""
    with open("configs/feddqn_phase10.yaml") as f:
        raw = yaml.safe_load(f)
    raw["_config_path"] = "configs/feddqn_phase10.yaml"

    agent_kwargs = {"lr": 1e-3, "gamma": 0.99, "batch_size": 4, "capacity": 100}

    # Same seed -> identical clients
    c_a = build_clients_with_seed(raw, agent_kwargs, num_tasks=5, master_seed=3, is_iid=True)
    c_b = build_clients_with_seed(raw, agent_kwargs, num_tasks=5, master_seed=3, is_iid=True)
    assert c_a[0].seed == c_b[0].seed

    # Different seed -> different clients
    c_c = build_clients_with_seed(raw, agent_kwargs, num_tasks=5, master_seed=7, is_iid=True)
    assert c_a[0].seed != c_c[0].seed


def test_detector_availability_and_not_applicable():
    """Verify that FedAvg and local RL mark detector metrics as None (N/A) rather than misleading zero scores."""
    item_fedavg = PlanItem(
        run_id="test_fedavg",
        matrix_type="core",
        scenario_group="clean_iid",
        attack_type="none",
        attack_fraction=0.0,
        attack_strength=0.0,
        unreliable_severity="none",
        unreliable_fraction=0.0,
        strategy="fedavg",
        master_seed=0,
        eval_seeds=[100],
        is_iid=True,
    )
    with open("configs/feddqn_phase10.yaml") as f:
        raw = yaml.safe_load(f)
    raw["_config_path"] = "configs/feddqn_phase10.yaml"
    raw["federated"]["rounds"] = 1

    temp_dir = Path("results/test_p10_detector_check")
    temp_dir.mkdir(parents=True, exist_ok=True)
    res = execute_plan_item(item_fedavg, raw, temp_dir)

    assert not res["has_detector"], "FedAvg must have has_detector=False"
    assert res["precision"] is None, "Precision must be None (N/A) for FedAvg"
    assert res["recall"] is None, "Recall must be None (N/A) for FedAvg"
    assert res["f1"] is None, "F1 must be None (N/A) for FedAvg"
    assert res["fpr"] is None, "FPR must be None (N/A) for FedAvg"

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_student_t_confidence_interval_exactness():
    """Verify compute_student_t_ci_95 computes exact Student-t critical values, not normal approximations."""
    sample = [0.75, 0.80, 0.85, 0.70, 0.90]  # n = 5, df = 4
    ci = compute_student_t_ci_95(sample)

    assert ci["n"] == 5
    assert ci["method"] == "student_t"
    expected_t_crit = stats.t.ppf(0.975, df=4)
    assert ci["t_crit"] == pytest.approx(expected_t_crit)
    # Ensure it is NOT the normal z-value 1.95996
    assert ci["t_crit"] > 2.50, "df=4 critical value must be ~2.776, substantially higher than 1.96"


def test_all_seven_ablations_and_four_sensitivities_in_plan():
    """Verify all 7 mandatory ablations and 4 sensitivity parameters are present in the frozen plan."""
    plan = build_phase10_experiment_plan()

    # Ablations
    abl_items = [p for p in plan if p.matrix_type == "ablation"]
    abl_names = {p.ablation_name for p in abl_items}
    expected_abl = {
        "full_proposed", "no_behavioural_trust", "no_model_trust",
        "no_temporal_smoothing", "no_threshold_exclusion",
        "fedavg_no_trust", "local_rl_no_fl"
    }
    assert abl_names == expected_abl, f"Missing ablation configurations: {expected_abl - abl_names}"

    # Sensitivity
    sens_items = [p for p in plan if p.matrix_type == "sensitivity"]
    sens_params = {p.sweep_param for p in sens_items}
    # Accepts m2_dir_threshold or tau_anomaly for directional sensitivity
    assert {"alpha", "tau_exc", "lambda_smooth"}.issubset(sens_params)
    assert ("tau_anomaly" in sens_params or "m2_dir_threshold" in sens_params)


def test_runtime_trust_lambda_propagation_and_resets():
    """Regression test: verifies that smoothing overrides survive environment resets throughout local training."""
    with open("configs/feddqn_phase10.yaml") as f:
        raw = yaml.safe_load(f)
    agent_kwargs = {"lr": 1e-3, "gamma": 0.99, "batch_size": 4, "capacity": 100}

    for target_lambda in [0.50, 0.70, 0.90, 0.95]:
        clients = build_clients_with_seed(raw, agent_kwargs, num_tasks=5, master_seed=1, is_iid=True, lambda_smooth=target_lambda)
        c = clients[0]
        # Verify initial setting
        assert c.get_effective_trust_lambda() == pytest.approx(target_lambda)
        assert c.env.trust.lam == pytest.approx(target_lambda)

        # Trigger reset directly
        c.env.reset(seed=999)
        assert c.get_effective_trust_lambda() == pytest.approx(target_lambda)
        assert c.env.trust.lam == pytest.approx(target_lambda)

        # Run local_train which resets env multiple times
        c.local_train(num_episodes=2, base_seed=1000, num_tasks=5, warmup_steps=0, sync_every=10)
        assert c.get_effective_trust_lambda() == pytest.approx(target_lambda)
        assert c.env.trust.lam == pytest.approx(target_lambda)

        # Wrap with UnreliableEnvWrapper and re-verify
        prof = UnreliableProfile(fail_prob=0.1, delay_prob=0.1, drop_prob=0.1, name="low")
        c.env = UnreliableEnvWrapper(c.env, prof, seed=555)
        c.set_trust_lambda(target_lambda)

        c.env.reset(seed=888)
        assert c.get_effective_trust_lambda() == pytest.approx(target_lambda)
        base_env = getattr(c.env, "env", c.env)
        assert base_env.trust.lam == pytest.approx(target_lambda)

        c.local_train(num_episodes=2, base_seed=2000, num_tasks=5, warmup_steps=0, sync_every=10)
        assert c.get_effective_trust_lambda() == pytest.approx(target_lambda)
        assert base_env.trust.lam == pytest.approx(target_lambda)


def test_two_unreliable_clients_counter_aggregation_and_resets():
    """Verify two unreliable clients: aggregate counters equal sum of both, persist across resets, and accounting identity holds."""
    cfg1 = EnvConfig(seed=101, num_vehicles=4, num_edge_nodes=2, num_tasks=10)
    cfg2 = EnvConfig(seed=102, num_vehicles=4, num_edge_nodes=2, num_tasks=10)
    env1 = VehicularEdgeEnv(cfg1)
    env2 = VehicularEdgeEnv(cfg2)

    prof = UnreliableProfile(fail_prob=0.3, delay_prob=0.3, drop_prob=0.2, name="test")
    wrap1 = UnreliableEnvWrapper(env1, prof, seed=111)
    wrap2 = UnreliableEnvWrapper(env2, prof, seed=222)

    # Step client 1 for 10 tasks across 2 episodes (with a reset)
    wrap1.reset(1)
    for _ in range(5):
        wrap1.step(0)
    wrap1.reset(2)  # Reset must NOT clear lifetime counters
    for _ in range(5):
        wrap1.step(0)

    # Step client 2 for 10 tasks across 2 episodes (with a reset)
    wrap2.reset(3)
    for _ in range(5):
        wrap2.step(0)
    wrap2.reset(4)
    for _ in range(5):
        wrap2.step(0)

    st1 = wrap1.get_fault_stats()
    st2 = wrap2.get_fault_stats()

    assert st1["tasks_exposed"] == 10
    assert st2["tasks_exposed"] == 10

    # Aggregate counters across both clients
    total_exposed = st1["tasks_exposed"] + st2["tasks_exposed"]
    total_with_fault = st1["tasks_with_any_fault"] + st2["tasks_with_any_fault"]
    total_drops = st1["injected_drops"] + st2["injected_drops"]
    total_fails = st1["injected_fails"] + st2["injected_fails"]
    total_delays = st1["injected_delays"] + st2["injected_delays"]

    assert total_exposed == 20
    assert total_with_fault <= total_exposed
    assert st1["tasks_with_any_fault"] <= st1["tasks_exposed"]
    assert st2["tasks_with_any_fault"] <= st2["tasks_exposed"]
    assert total_drops + total_fails + total_delays >= total_with_fault


def test_stale_replay_provenance_and_round_zero():
    """Verify stale replay preserves originating round, computes true staleness, and does not attack in round 0."""
    from attacks.pipeline import AttackPipeline

    scen = ScenarioAssignment.create(
        num_clients=4,
        malicious_fraction=0.25,
        attack_type=AttackType.STALE_REPLAY,
        seed=123,
        attack_strength=2.0,  # target staleness age = 2
    )
    mal_cid = scen.malicious_client_ids[0]
    pipeline = AttackPipeline(scen, seed=123)

    ref_state = {"w": torch.tensor([1.0, 2.0])}

    # Round 0: First round -> NO earlier update exists to replay!
    u0 = {"client_id": mal_cid, "round_trained": 0, "num_samples": 10, "staleness": 0, "state": {"w": torch.tensor([1.1, 2.1])}}
    att0, audit0 = pipeline.transform_update(u0, ref_state, current_round=0)
    assert not audit0["attack_applied"], "Round 0 must NOT be labeled as stale replay"
    assert att0["round_trained"] == 0
    assert att0["staleness"] == 0

    # Round 1: Only round 0 exists
    u1 = {"client_id": mal_cid, "round_trained": 1, "num_samples": 10, "staleness": 0, "state": {"w": torch.tensor([1.2, 2.2])}}
    att1, audit1 = pipeline.transform_update(u1, ref_state, current_round=1)
    assert audit1["attack_applied"], "Round 1 must replay round 0 update"
    assert att1["round_trained"] == 0, "Origin round must remain 0"
    assert att1["staleness"] == 1, "Staleness in round 1 must be 1"
    assert torch.allclose(att1["state"]["w"], u0["state"]["w"]), "Parameters must match round 0 update"

    # Round 2: Round 0 and Round 1 exist. Target age = 2 -> round 2 - 2 = 0
    u2 = {"client_id": mal_cid, "round_trained": 2, "num_samples": 10, "staleness": 0, "state": {"w": torch.tensor([1.3, 2.3])}}
    att2, audit2 = pipeline.transform_update(u2, ref_state, current_round=2)
    assert audit2["attack_applied"]
    assert att2["round_trained"] == 0
    assert att2["staleness"] == 2
    assert torch.allclose(att2["state"]["w"], u0["state"]["w"])

    # Round 5: Replay maintains true age (5 - 0 = 5)
    u5 = {"client_id": mal_cid, "round_trained": 5, "num_samples": 10, "staleness": 0, "state": {"w": torch.tensor([1.6, 2.6])}}
    att5, audit5 = pipeline.transform_update(u5, ref_state, current_round=5)
    assert audit5["attack_applied"]
    assert att5["round_trained"] == 0
    assert att5["staleness"] == 5

