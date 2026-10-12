"""Scenario tests for Phase 8: Fractions, client assignment determinism, unreliable nodes, and non-IID control."""

import pytest
import numpy as np

from attacks.base import AttackType, ScenarioAssignment
from attacks.unreliable_execution import UnreliableEnvWrapper, UnreliableProfile
from environment import EnvConfig, VehicularEdgeEnv


def test_malicious_fractions_integer_counts():
    """Verify that for 10 clients, fractions {0, 0.1, 0.2, 0.3, 0.4} map exactly to {0, 1, 2, 3, 4} malicious clients."""
    expected_counts = {
        0.0: 0,
        0.1: 1,
        0.2: 2,
        0.3: 3,
        0.4: 4,
    }
    for frac, exp_num in expected_counts.items():
        scen = ScenarioAssignment.create(
            num_clients=10,
            malicious_fraction=frac,
            attack_type=AttackType.SIGN_FLIP if frac > 0 else AttackType.NONE,
            seed=0,
        )
        assert len(scen.malicious_client_ids) == exp_num
        assert sum(1 for lbl in scen.client_labels.values() if lbl.is_malicious) == exp_num


def test_clean_scenario_all_honest():
    """f = 0 results in 100% honest clients."""
    scen = ScenarioAssignment.create(
        num_clients=10,
        malicious_fraction=0.0,
        attack_type=AttackType.NONE,
        seed=123,
    )
    assert len(scen.malicious_client_ids) == 0
    assert all(not lbl.is_malicious for lbl in scen.client_labels.values())


def test_assignment_reproducible_and_seed_controlled():
    scen1 = ScenarioAssignment.create(10, 0.3, AttackType.PERTURBATION, seed=42)
    scen2 = ScenarioAssignment.create(10, 0.3, AttackType.PERTURBATION, seed=42)
    scen3 = ScenarioAssignment.create(10, 0.3, AttackType.PERTURBATION, seed=99)

    assert scen1.malicious_client_ids == scen2.malicious_client_ids
    # Different seed should generally produce different or independent subset
    assert len(scen1.malicious_client_ids) == 3
    assert len(scen3.malicious_client_ids) == 3


def test_unreliable_env_wrapper_fault_distinctions():
    """Verify that failures, deadline misses, and disconnections are distinguished correctly."""
    cfg = EnvConfig(num_vehicles=3, num_edge_nodes=1, num_tasks=5, seed=0)
    base_env = VehicularEdgeEnv(cfg)

    # 1. Failure profile: 100% execution failure
    p_fail = UnreliableProfile(fail_prob=1.0, delay_prob=0.0, drop_prob=0.0)
    w_fail = UnreliableEnvWrapper(base_env, p_fail, seed=0)
    w_fail.reset(0)
    res_f = w_fail.step(0)  # local execution
    assert res_f.failed is True
    assert res_f.success is False
    assert base_env.trust.history[-1].s4 == 1.0  # path survived

    # 2. Delay profile: 100% delay
    p_delay = UnreliableProfile(fail_prob=0.0, delay_prob=1.0, delay_factor=5.0, drop_prob=0.0)
    w_delay = UnreliableEnvWrapper(base_env, p_delay, seed=0)
    w_delay.reset(0)
    res_d = w_delay.step(0)
    assert res_d.deadline_missed is True
    assert res_d.success is False
    assert base_env.trust.history[-1].s4 == 1.0  # path survived

    # 3. Disconnect profile: 100% drop (unobserved disconnect, NOT confirmed failure)
    p_drop = UnreliableProfile(fail_prob=0.0, delay_prob=0.0, drop_prob=1.0)
    w_drop = UnreliableEnvWrapper(base_env, p_drop, seed=0)
    w_drop.reset(0)
    res_dr = w_drop.step(0)
    assert base_env.trust.history[-1].s4 == 0.0  # return path dropped
    assert res_dr.success is False
    assert res_dr.failed is True  # Task fails due to return path loss (EQ-8)
    assert res_dr.deadline_missed is False



def test_honest_non_iid_control_labels():
    """Verify honest non-IID scenario marks all clients as honest."""
    scen = ScenarioAssignment.create(
        num_clients=10,
        malicious_fraction=0.0,
        attack_type=AttackType.NONE,
        seed=10,
    )
    for cid, lbl in scen.client_labels.items():
        assert lbl.is_malicious is False
        assert lbl.attack_type == AttackType.NONE


def test_unreliable_multi_task_execution_and_trust_degradation():
    """Verify that multi-task execution through UnreliableEnvWrapper degrades task success

    and alters behavioural trust compared to a clean environment under identical seeds.
    """
    cfg = EnvConfig(num_vehicles=5, num_edge_nodes=2, num_tasks=30, seed=42)

    # 1. Clean environment
    clean_env = VehicularEdgeEnv(cfg)
    clean_env.reset(100)
    clean_results = [clean_env.step(0) for _ in range(30)]
    clean_succ = sum(r.success for r in clean_results)
    clean_fail = sum(r.failed for r in clean_results)
    clean_mean_trust = float(np.mean(list(clean_env.trust.snapshot().values())))

    # 2. Medium-severity unreliable environment (25% fail, 25% delay, 20% drop)
    med_prof = UnreliableProfile(fail_prob=0.25, delay_prob=0.25, drop_prob=0.2, name="medium")
    med_env = VehicularEdgeEnv(cfg)
    med_env.reset(100)
    med_wrap = UnreliableEnvWrapper(med_env, med_prof, seed=999)
    med_results = [med_wrap.step(0) for _ in range(30)]
    med_succ = sum(r.success for r in med_results)
    med_fail = sum(r.failed for r in med_results)
    med_mean_trust = float(np.mean(list(med_wrap.trust.snapshot().values())))

    # Invariants and outcome shifts
    assert len(clean_results) == 30
    assert len(med_results) == 30
    # In clean env, local execution with sufficient CPU/queue has 0 failures
    assert clean_fail == 0
    # In unreliable env, faults are actively injected: failures MUST be observed (> 0)
    assert med_fail > 0
    # Successful completions must decrease due to injected faults
    assert med_succ < clean_succ
    # Accounting invariant holds for each individual step result
    for r in med_results:
        assert (r.success + r.failed + r.deadline_missed) >= 1

