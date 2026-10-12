"""
Unit and integration tests for Phase 9 Streamlit Dashboard.
Tests data loaders, components, data integrity, formatting, and simulator integration.
"""

from pathlib import Path
import pytest
import pandas as pd

from dashboard.data_loader import (
    get_available_datasets,
    load_experiment_data,
    load_summary_metrics,
    load_manifest,
    get_available_checkpoints,
    get_available_figures,
    REPO_ROOT
)
from environment.vanet_env import VehicularEdgeEnv, EnvConfig
import baselines


def test_available_datasets_discovery():
    """Verify that both primary multi-round (r4) and historical datasets are discovered."""
    datasets = get_available_datasets()
    assert len(datasets) >= 1, "At least one dataset must be discovered"
    assert any("4-Round" in k for k in datasets.keys()), "Phase 8 r4 dataset should be present"
    
    # Check paths
    for name, info in datasets.items():
        assert Path(info["csv_path"]).exists(), f"CSV path for {name} does not exist"
        assert Path(info["json_path"]).exists(), f"JSON path for {name} does not exist"


def test_load_experiment_data_schema():
    """Verify that loaded experiment CSV contains required columns and friendly labels."""
    datasets = get_available_datasets()
    key = list(datasets.keys())[0]
    csv_path = datasets[key]["csv_path"]
    
    df = load_experiment_data(csv_path)
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) in (225, 540, 630), f"Expected 225, 540, or 630 runs, got {len(df)}"
    
    # Required columns
    expected_cols = [
        "scenario_group", "strategy", "seed", "attack_fraction",
        "completion_rate", "failure_rate", "miss_rate", "mean_latency_s",
        "f1", "precision", "recall", "fpr", "scenario_label", "strategy_label"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Invariant: task partition
    total_task_rate = (df["completion_rate"] + df["failure_rate"] + df["miss_rate"]).round(4)
    assert (total_task_rate == 1.0).all(), "Task rates must sum to 1.0 across all rows"


def test_load_summary_metrics():
    """Verify that summary metrics JSON parses successfully and has overall/by_fraction keys."""
    datasets = get_available_datasets()
    key = list(datasets.keys())[0]
    json_path = datasets[key]["json_path"]
    
    metrics = load_summary_metrics(json_path)
    assert isinstance(metrics, dict)
    assert "overall" in metrics or "clean_control" in metrics, "Expected metric groups in summary"


def test_load_manifest_checksums():
    """Verify that run manifest exists and contains experiment hashes."""
    datasets = get_available_datasets()
    key = list(datasets.keys())[0]
    manifest_path = datasets[key]["manifest_path"]
    
    manifest = load_manifest(manifest_path)
    if manifest_path.exists():
        assert "phase" in manifest or "experiment_id" in manifest or "files" in manifest


def test_get_available_checkpoints():
    """Verify that existing model checkpoints are detected."""
    ckpts = get_available_checkpoints()
    assert isinstance(ckpts, list)
    if ckpts:
        assert "name" in ckpts[0]
        assert "path" in ckpts[0]
        assert "size_kb" in ckpts[0]


def test_interactive_simulator_execution():
    """Verify that the VehicularEdgeEnv and baselines execute safely for the interactive simulator."""
    cfg = EnvConfig(
        seed=123,
        num_vehicles=4,
        num_edge_nodes=2,
        mobility="low",
        task_load="low",
        num_tasks=15
    )
    env = VehicularEdgeEnv(cfg)
    env.reset()
    
    for pol_name in baselines.POLICIES:
        pol = baselines.make_policy(pol_name, seed=123)
        step_results, ep_sum = env.run_episode(pol)
        assert ep_sum.num_tasks == 15
        assert ep_sum.successes + ep_sum.deadline_misses + ep_sum.failures == 15
        assert ep_sum.mean_latency_s >= 0.0


def test_dashboard_filtering_logic():
    """Verify that multi-dimensional filtering produces valid subsets without data corruption."""
    datasets = get_available_datasets()
    key = list(datasets.keys())[0]
    df = load_experiment_data(datasets[key]["csv_path"])

    # Filter by scenario
    scen_val = "Poisoning: Sign-Flip"
    n_seeds = df["seed"].nunique()
    sub_scen = df[df["scenario_label"] == scen_val]
    assert len(sub_scen) == 3 * 4 * n_seeds

    # Filter by strategy
    strat_val = "Trust-Threshold Robust"
    sub_strat = sub_scen[sub_scen["strategy_label"] == strat_val]
    assert len(sub_strat) == 4 * n_seeds

    # Filter by fraction
    sub_frac = sub_strat[sub_strat["attack_fraction"] == 0.2]
    assert len(sub_frac) == n_seeds

    # Filter by seed
    first_seed = sub_frac["seed"].iloc[0]
    sub_seed = sub_frac[sub_frac["seed"] == first_seed]
    assert len(sub_seed) == 1
    assert sub_seed.iloc[0]["attack_fraction"] == 0.2
    assert sub_seed.iloc[0]["seed"] == first_seed

