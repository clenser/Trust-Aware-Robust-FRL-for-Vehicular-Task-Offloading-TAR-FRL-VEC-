"""
Data loading and caching module for the VANET Research Dashboard.
Provides safe, cached ingestion of experimental records, metrics,
manifests, and figures with graceful fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_available_datasets() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available result datasets and their paths."""
    datasets = {}

    # Phase 10 Final v4 (Full 880-Run Corrected Suite - Primary)
    p10_v4_csv = REPO_ROOT / "results" / "phase10_final_v4" / "phase10_experiment_runs.csv"
    p10_v4_json = REPO_ROOT / "results" / "phase10_final_v4" / "phase10_summary_metrics.json"
    if p10_v4_csv.exists():
        datasets["Phase 10 Final v4 (Full 880-Run Corrected Suite)"] = {
            "key": "phase10_final_v4",
            "csv_path": p10_v4_csv,
            "ablation_csv_path": REPO_ROOT / "results" / "phase10_final_v4" / "phase10_ablation_runs.csv",
            "sensitivity_csv_path": REPO_ROOT / "results" / "phase10_final_v4" / "phase10_sensitivity_runs.csv",
            "json_path": p10_v4_json,
            "manifest_path": REPO_ROOT / "reports" / "PHASE10_FINAL_V4_RUN_MANIFEST.json",
            "figures_dir": REPO_ROOT / "reports" / "figures" / "phase10_final_v4",
            "rounds": 8,
            "core_rows": 630,
            "ablation_rows": 70,
            "sensitivity_rows": 180,
            "description": "Corrected full matrix suite (630 core runs, 70 ablations, 180 sensitivity sweeps, 10+5 seeds, 8 rounds)."
        }

    # Phase 10 Final v3 (Historical - 880 Runs, superseded parameter values)
    p10_v3_csv = REPO_ROOT / "results" / "phase10_final_v3" / "phase10_experiment_runs.csv"
    p10_v3_json = REPO_ROOT / "results" / "phase10_final_v3" / "phase10_summary_metrics.json"
    if p10_v3_csv.exists():
        datasets["Phase 10 Final v3 (Historical 880-Run Suite)"] = {
            "key": "phase10_final_v3",
            "csv_path": p10_v3_csv,
            "ablation_csv_path": REPO_ROOT / "results" / "phase10_final_v3" / "phase10_ablation_runs.csv",
            "sensitivity_csv_path": REPO_ROOT / "results" / "phase10_final_v3" / "phase10_sensitivity_runs.csv",
            "json_path": p10_v3_json,
            "manifest_path": REPO_ROOT / "reports" / "PHASE10_FINAL_V3_RUN_MANIFEST.json",
            "figures_dir": REPO_ROOT / "reports" / "figures" / "phase10_final_v3",
            "rounds": 8,
            "core_rows": 630,
            "ablation_rows": 70,
            "sensitivity_rows": 180,
            "description": "Historical 880-run suite (superseded: alpha=0.60/tau_exc=0.40 core values)."
        }

    # Phase 10 Final v2 (Historical Reconciled - 880 Runs)
    p10_final_csv = REPO_ROOT / "results" / "phase10_final_v2" / "phase10_experiment_runs.csv"
    p10_final_json = REPO_ROOT / "results" / "phase10_final_v2" / "phase10_summary_metrics.json"
    if p10_final_csv.exists():
        datasets["Phase 10 Final v2 (Historical Reconciled Suite)"] = {
            "key": "phase10_final_v2",
            "csv_path": p10_final_csv,
            "ablation_csv_path": REPO_ROOT / "results" / "phase10_final_v2" / "phase10_ablation_runs.csv",
            "sensitivity_csv_path": REPO_ROOT / "results" / "phase10_final_v2" / "phase10_sensitivity_runs.csv",
            "json_path": p10_final_json,
            "manifest_path": REPO_ROOT / "reports" / "PHASE10_FINAL_RUN_MANIFEST.json",
            "figures_dir": REPO_ROOT / "reports" / "figures" / "phase10_final_v2",
            "rounds": 8,
            "description": "Historical reconciled matrix (630 core runs, 70 ablations, 180 sensitivity runs, 10 seeds, 8 rounds)."
        }

    # Phase 10 (Superseded Preliminary - 540 Core Runs)
    p10_csv = REPO_ROOT / "results" / "phase10" / "phase10_experiment_runs.csv"
    p10_json = REPO_ROOT / "results" / "phase10" / "phase10_summary_metrics.json"
    if p10_csv.exists():
        datasets["Phase 10 (Superseded Preliminary - 540 Core Runs)"] = {
            "key": "phase10",
            "csv_path": p10_csv,
            "json_path": p10_json,
            "manifest_path": REPO_ROOT / "reports" / "PHASE10_RUN_MANIFEST.json",
            "figures_dir": REPO_ROOT / "reports" / "figures" / "phase10",
            "rounds": 8,
            "description": "Preliminary evaluation suite (540 core runs, 35 ablations, 55 sensitivity runs, 10 seeds, 8 rounds)."
        }

    # Phase 8 r4 (4 rounds - primary)
    p8_r4_csv = REPO_ROOT / "results" / "phase8_r4" / "phase8_experiment_runs.csv"
    p8_r4_json = REPO_ROOT / "results" / "phase8_r4" / "phase8_summary_metrics.json"
    if p8_r4_csv.exists():
        datasets["Phase 8 (4-Round Horizon)"] = {
            "key": "phase8_r4",
            "csv_path": p8_r4_csv,
            "json_path": p8_r4_json,
            "manifest_path": REPO_ROOT / "reports" / "PHASE8_R4_RUN_MANIFEST.json",
            "figures_dir": REPO_ROOT / "reports" / "figures" / "phase8_r4",
            "rounds": 4,
            "description": "Multi-round horizon benchmark (225 runs, 10 scenarios, 3 strategies, 3 seeds)."
        }

    # Phase 8 historical (2 rounds)
    p8_csv = REPO_ROOT / "results" / "phase8" / "phase8_experiment_runs.csv"
    p8_json = REPO_ROOT / "results" / "phase8" / "phase8_summary_metrics.json"
    if p8_csv.exists():
        datasets["Phase 8 Historical (2-Round Horizon)"] = {
            "key": "phase8",
            "csv_path": p8_csv,
            "json_path": p8_json,
            "manifest_path": REPO_ROOT / "reports" / "PHASE8_RUN_MANIFEST.json",
            "figures_dir": REPO_ROOT / "reports" / "figures" / "phase8",
            "rounds": 2,
            "description": "Historical 2-round baseline benchmark (225 runs, 10 scenarios, 3 strategies, 3 seeds)."
        }

    return datasets


def load_experiment_data(csv_path: Path) -> pd.DataFrame:
    """Load experiment runs CSV and compute friendly presentation labels."""
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)

    # Add display-friendly names if not present
    scenario_display_map = {
        "clean_control": "Clean Control (Baseline)",
        "clean_iid": "Clean IID Control (Homogeneous)",
        "honest_non_iid": "Honest Non-IID Control (Heterogeneous)",
        "poisoning_sign_flip": "Poisoning: Sign-Flip",
        "poisoning_perturbation": "Poisoning: Noise Perturbation",
        "freeriding_zero_update": "Free-Riding: Zero-Update",
        "freeriding_copy_update": "Free-Riding: Copy-Update",
        "freeriding_stale_replay": "Free-Riding: Stale-Replay",
        "unreliable_low": "Unreliable Execution (Low: 10% fault)",
        "unreliable_medium": "Unreliable Execution (Medium: 25% fault)",
        "unreliable_high": "Unreliable Execution (High: 50% fault)",
    }
    strategy_display_map = {
        "fedavg": "FedAvg (Unweighted)",
        "trust_weighted": "Trust-Weighted FedAvg",
        "trust_threshold": "Trust-Threshold Robust"
    }

    if "scenario_group" in df.columns:
        df["scenario_label"] = df["scenario_group"].map(scenario_display_map).fillna(df["scenario_group"])
    if "strategy" in df.columns:
        df["strategy_label"] = df["strategy"].map(strategy_display_map).fillna(df["strategy"])

    return df


def load_summary_metrics(json_path: Path) -> Dict[str, Any]:
    """Load structured summary metrics JSON."""
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load run manifest JSON."""
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_dataset_integrity(manifest_path: Path, max_check: int = 50) -> Dict[str, Any]:
    """Verify integrity of a dataset manifest.
    
    Returns a status dict:
      - status: 'verified', 'unverified', 'failed', 'unavailable'
      - label: human-readable label
      - checked_count: number of artifacts hashed
      - total_count: total artifacts in manifest
      - message: explanation
    """
    if not manifest_path or not manifest_path.exists():
        return {
            "status": "unavailable",
            "label": "Manifest Unavailable ⚠️",
            "checked_count": 0,
            "total_count": 0,
            "message": "No manifest file found for this dataset."
        }
    
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})
        if not artifacts:
            return {
                "status": "unverified",
                "label": "Manifest Loaded (No Artifacts Listed)",
                "checked_count": 0,
                "total_count": 0,
                "message": "Manifest is loaded but contains no artifact hashes."
            }
        
        # Verify sample/subset or all artifacts
        import hashlib
        checked = 0
        mismatches = []
        for rel_path, meta in list(artifacts.items())[:max_check]:
            target = REPO_ROOT / rel_path
            if not target.exists():
                mismatches.append(f"Missing: {rel_path}")
                break
            h = hashlib.sha256()
            with open(target, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != meta.get("sha256"):
                mismatches.append(f"Hash mismatch: {rel_path}")
                break
            checked += 1
            
        if mismatches:
            return {
                "status": "failed",
                "label": "Verification Failed ❌",
                "checked_count": checked,
                "total_count": len(artifacts),
                "message": mismatches[0]
            }
        
        return {
            "status": "verified",
            "label": f"Integrity Verified ✅ ({checked}/{len(artifacts)} sampled)",
            "checked_count": checked,
            "total_count": len(artifacts),
            "message": f"Successfully verified SHA-256 hashes for {checked} artifacts against manifest."
        }
    except Exception as e:
        return {
            "status": "failed",
            "label": "Verification Error ⚠️",
            "checked_count": 0,
            "total_count": 0,
            "message": str(e)
        }


def load_config_yaml(config_path: Path) -> Dict[str, Any]:
    """Load configuration YAML."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_available_checkpoints() -> List[Dict[str, Any]]:
    """Scan and return all available model checkpoints and metadata."""
    ckpt_dir = REPO_ROOT / "checkpoints"
    results = []
    if not ckpt_dir.exists():
        return results

    for p in sorted(ckpt_dir.glob("*.pt")):
        stat = p.stat()
        results.append({
            "name": p.name,
            "path": str(p),
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": pd.to_datetime(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
        })
    return results


def get_available_figures(figures_dir: Path) -> List[Path]:
    """Return all PNG figures in the specified figure directory."""
    if not figures_dir.exists():
        return []
    return sorted(list(figures_dir.glob("*.png")))
