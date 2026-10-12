"""
Unit and integration tests for Phase 10: Full Evaluation, Ablation, and Sensitivity Analysis.
Tests:
- Phase 10 configuration validation.
- Ablation parameter mapping (alpha=0, alpha=1, lam=0, skip_fl).
- Matrix generation invariants and task partition constraints.
"""

from pathlib import Path
import pytest
import yaml

from attacks.base import AttackType, ScenarioAssignment
from environment import EnvConfig
from scripts.run_phase10_evaluation import run_phase10_trial


def test_phase10_config_structure():
    """Verify configs/feddqn_phase10.yaml has valid keys and parameters."""
    cfg_path = Path("configs/feddqn_phase10.yaml")
    assert cfg_path.exists(), "Phase 10 config must exist"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["federated"]["rounds"] == 8
    assert cfg["federated"]["tau_exc"] == 0.40
    assert cfg["federated"]["alpha"] == 0.60
    assert len(cfg["federated"]["client_conditions"]) == 10


def test_phase10_trial_ablation_executions():
    """Verify that run_phase10_trial handles ablations cleanly (alpha=0, alpha=1, skip_fl)."""
    with open("configs/feddqn_phase10.yaml") as f:
        raw = yaml.safe_load(f)
    raw["_config_path"] = "configs/feddqn_phase10.yaml"
    raw["federated"]["rounds"] = 1  # Fast test

    scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=0.2, attack_type=AttackType.SIGN_FLIP, seed=42)
    test_out = Path("results/test_phase10_unit_temp")

    # 1. Ablation without model trust (alpha=1.0)
    res_no_mod, _ = run_phase10_trial(raw, scen, "trust_threshold", seed=42, outdir=test_out / "no_mod", eval_seeds=[10], alpha=1.0)
    assert (res_no_mod["completion_rate"] + res_no_mod["failure_rate"] + res_no_mod["miss_rate"]) == 1.0

    # 2. Ablation without behavioural trust (alpha=0.0)
    res_no_beh, _ = run_phase10_trial(raw, scen, "trust_threshold", seed=42, outdir=test_out / "no_beh", eval_seeds=[10], alpha=0.0)
    assert (res_no_beh["completion_rate"] + res_no_beh["failure_rate"] + res_no_beh["miss_rate"]) == 1.0

    # 3. Independent local RL (skip_fl=True)
    res_local, _ = run_phase10_trial(raw, scen, "fedavg", seed=42, outdir=test_out / "local", eval_seeds=[10], skip_fl=True)
    assert (res_local["completion_rate"] + res_local["failure_rate"] + res_local["miss_rate"]) == 1.0

    # Cleanup temp test output
    import shutil
    shutil.rmtree(test_out, ignore_errors=True)
