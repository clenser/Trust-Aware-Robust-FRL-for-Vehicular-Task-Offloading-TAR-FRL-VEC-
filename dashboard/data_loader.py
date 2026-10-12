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
    
    # Phase 10 (8 rounds, 10 seeds - full evaluation)
    p10_csv = REPO_ROOT / "results" / "phase10" / "phase10_experiment_runs.csv"
    p10_json = REPO_ROOT / "results" / "phase10" / "phase10_summary_metrics.json"
    if p10_csv.exists():
        datasets["Phase 10 (Full Evaluation - 10 Seeds, 8 Rounds)"] = {
            "key": "phase10",
            "csv_path": p10_csv,
            "json_path": p10_json,
            "manifest_path": REPO_ROOT / "reports" / "PHASE10_RUN_MANIFEST.json",
            "figures_dir": REPO_ROOT / "reports" / "figures" / "phase10",
            "rounds": 8,
            "description": "Full evaluation suite (540 core runs, 35 ablations, 55 sensitivity runs, 10 seeds, 8 rounds)."
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
        "poisoning_sign_flip": "Poisoning: Sign-Flip",
        "poisoning_perturbation": "Poisoning: Noise Perturbation",
        "freeriding_zero_update": "Free-Riding: Zero-Update",
        "freeriding_copy_update": "Free-Riding: Copy-Update",
        "freeriding_stale_replay": "Free-Riding: Stale-Replay",
        "unreliable_low": "Unreliable Execution (p=0.2)",
        "unreliable_medium": "Unreliable Execution (p=0.4)",
        "unreliable_high": "Unreliable Execution (p=0.7)",
        "honest_non_iid": "Honest Non-IID Heterogeneity"
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
