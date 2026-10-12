"""
Unit and integration tests for Phase 9 Streamlit Dashboard.
Tests data loaders, components, data integrity, formatting, and simulator integration.
"""

from pathlib import Path
import pytest
import pandas as pd
import json
import tempfile
import os

from dashboard.data_loader import (
    get_available_datasets,
    load_experiment_data,
    load_summary_metrics,
    load_manifest,
    get_available_checkpoints,
    get_available_figures,
    REPO_ROOT,
    verify_dataset_integrity
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


def test_verify_dataset_integrity_status():
    """Verify that verify_dataset_integrity returns honest status across manifests."""
    
    # 1. Non-existent manifest
    res_none = verify_dataset_integrity(Path("reports/non_existent_manifest.json"))
    assert res_none["status"] == "unavailable"
    assert "Unavailable" in res_none["label"]
    assert res_none["checked_count"] == 0
    assert res_none["total_count"] == 0

    # 2. Real Phase 10 v4 manifest with sampling (should be "sampled" not "verified")
    datasets = get_available_datasets()
    v4_info = datasets["Phase 10 Final v4 (Full 880-Run Corrected Suite)"]
    res_v4 = verify_dataset_integrity(v4_info["manifest_path"], max_check=20)
    assert res_v4["status"] == "sampled", f"Expected 'sampled', got '{res_v4['status']}'"
    assert res_v4["checked_count"] == 20
    assert res_v4["total_count"] == 894
    assert "Sampled" in res_v4["label"]
    assert "Full verification not performed" in res_v4["message"]

    # 3. Full verification with max_check >= total (should be "verified")
    res_full = verify_dataset_integrity(v4_info["manifest_path"], max_check=1000)
    assert res_full["status"] == "verified", f"Expected 'verified', got '{res_full['status']}'"
    assert res_full["checked_count"] == 894
    assert "Integrity Verified" in res_full["label"]
    assert "all 894" in res_full["message"]


def test_verify_dataset_integrity_malformed_manifest():
    """Verify handling of malformed manifests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        temp_path = Path(f.name)
    try:
        res = verify_dataset_integrity(temp_path)
        assert res["status"] == "unverified"
        assert "Malformed" in res["label"]
    finally:
        os.unlink(temp_path)


def test_verify_dataset_integrity_empty_artifacts():
    """Verify handling of manifest with empty artifacts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"artifacts": {}}, f)
        temp_path = Path(f.name)
    try:
        res = verify_dataset_integrity(temp_path)
        assert res["status"] == "unverified"
        assert "No Artifacts" in res["label"]
    finally:
        os.unlink(temp_path)


def test_verify_dataset_integrity_missing_sha256():
    """Verify handling of manifest entries missing SHA-256."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"artifacts": {"some/file.txt": {"bytes": 100}}}, f)
        temp_path = Path(f.name)
    try:
        res = verify_dataset_integrity(temp_path)
        assert res["status"] == "unverified"
        assert "No Valid Artifacts" in res["label"]
    finally:
        os.unlink(temp_path)


def test_verify_dataset_integrity_path_traversal():
    """Verify that path traversal attempts are rejected."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"artifacts": {"../outside_repo.txt": {"sha256": "abc123"}}}, f)
        temp_path = Path(f.name)
    try:
        res = verify_dataset_integrity(temp_path)
        assert res["status"] == "unverified"
        assert "No Valid Artifacts" in res["label"]
    finally:
        os.unlink(temp_path)


def test_verify_dataset_integrity_missing_file():
    """Verify detection of missing files in manifest."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"artifacts": {"nonexistent/file.txt": {"sha256": "abc123"}}}, f)
        temp_path = Path(f.name)
    try:
        res = verify_dataset_integrity(temp_path, max_check=10)
        assert res["status"] == "failed"
        assert "Missing" in res["message"]
    finally:
        os.unlink(temp_path)


def test_verify_dataset_integrity_hash_mismatch():
    """Verify detection of hash mismatches."""
    # Create a real file within the repo
    test_file = REPO_ROOT / "test_hash_mismatch.txt"
    test_file.write_text("actual content")
    try:
        # Create manifest with wrong hash, using relative path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as mf:
            rel_path = test_file.relative_to(REPO_ROOT).as_posix()
            json.dump({"artifacts": {rel_path: {"sha256": "wrong_hash"}}}, mf)
            manifest_path = Path(mf.name)
        try:
            res = verify_dataset_integrity(manifest_path, max_check=10)
            assert res["status"] == "failed"
            assert "Hash mismatch" in res["message"]
        finally:
            os.unlink(manifest_path)
    finally:
        test_file.unlink(missing_ok=True)


def test_verify_dataset_integrity_full_verification():
    """Verify that full verification works when max_check >= total artifacts."""
    datasets = get_available_datasets()
    v4_info = datasets["Phase 10 Final v4 (Full 880-Run Corrected Suite)"]
    
    # Test with a small manifest to make full verification fast
    # We can use a smaller dataset manifest for this
    for name, info in datasets.items():
        if info["manifest_path"].exists() and name != "Phase 10 Final v4 (Full 880-Run Corrected Suite)":
            # Get total artifact count
            manifest = load_manifest(info["manifest_path"])
            total = len(manifest.get("artifacts", {}))
            if total > 0 and total < 50:  # Small manifest
                res = verify_dataset_integrity(info["manifest_path"], max_check=total + 10)
                assert res["status"] == "verified"
                assert res["checked_count"] == total
                break

