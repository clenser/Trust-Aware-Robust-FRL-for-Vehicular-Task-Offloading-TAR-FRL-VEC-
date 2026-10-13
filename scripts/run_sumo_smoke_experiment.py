"""Standalone SUMO Microscopic Traffic Smoke Experiment Runner.

Executes a small, reproducible benchmark comparing offloading policies under
realistic microscopic highway traffic dynamics using Eclipse SUMO and TraCI.
Persists results into `results/sumo_smoke/` without touching historical Phase 10 artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from environment.config import EnvConfig
from environment.sumo_adapter import is_sumo_available, get_sumo_status
from dashboard.simulator_engine import run_traced_episode

OUTPUT_DIR = REPO_ROOT / "results" / "sumo_smoke"


def run_sumo_smoke_experiment(
    n_tasks: int = 60,
    seed: int = 42,
    density: str = "medium",
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run comparative evaluation of offloading policies under microscopic SUMO traffic."""
    if not is_sumo_available():
        status = get_sumo_status()
        print(f"ERROR: Eclipse SUMO or TraCI not available: {status['instructions']}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    policies_to_evaluate = [
        "resource_aware",
        "nearest_edge",
        "random_valid",
        "local_only",
    ]

    # Include trained FedDQN global model if available
    fed_ckpt = REPO_ROOT / "checkpoints" / "feddqn_default" / "global_final.pt"
    if fed_ckpt.exists():
        policies_to_evaluate.append("feddqn_global_final")

    print(f"=== Starting Microscopic SUMO Traffic Smoke Experiment ===")
    print(f"Policies: {policies_to_evaluate}")
    print(f"Tasks per policy: {n_tasks} | PRNG Seed: {seed} | Traffic Density: {density}")

    cfg = EnvConfig(
        seed=seed,
        num_vehicles=6,
        num_edge_nodes=2,
        mobility=density,
        task_load="medium",
        num_tasks=n_tasks,
        road_length_m=1200.0,
        comm_range_m=250.0,
    )

    summary_rows = []
    all_handovers = []

    for pol in policies_to_evaluate:
        print(f"Running policy: {pol}...")
        snapshots, summary = run_traced_episode(
            cfg=cfg,
            policy_name=pol,
            max_tasks=n_tasks,
            mobility_mode="sumo",
            strict_policy=True,
        )

        row = {
            "policy": pol,
            "policy_description": summary.get("policy_description", pol),
            "total_tasks": summary["total_tasks"],
            "successes": summary["successes"],
            "deadline_misses": summary["deadline_misses"],
            "failures": summary["failures"],
            "success_rate": summary["success_rate"],
            "mean_latency_s": summary["mean_latency_s"],
            "mean_energy_j": summary["mean_energy_j"],
            "offload_fraction": summary["offload_fraction"],
            "total_handovers": summary.get("total_handovers", 0),
            "total_associations": summary.get("total_associations", 0),
            "total_disconnections": summary.get("total_disconnections", 0),
            "total_departures": summary.get("total_departures", 0),
            "policy_exceptions": summary.get("policy_exceptions", 0),
            "invalid_actions": summary.get("invalid_actions", 0),
            "mobility_engine": summary.get("mobility_engine", "SUMO"),
        }
        summary_rows.append(row)

        for h in summary.get("handover_events", []):
            h_copy = dict(h)
            h_copy["policy"] = pol
            all_handovers.append(h_copy)

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "sumo_smoke_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to: {summary_csv}")

    # Save handovers CSV
    df_handovers = pd.DataFrame(all_handovers)
    handovers_csv = output_dir / "sumo_smoke_handovers.csv"
    df_handovers.to_csv(handovers_csv, index=False)
    print(f"Saved handover logs to: {handovers_csv}")

    # Save full JSON report
    report = {
        "run_id": f"sumo_smoke_s{seed}_{density}",
        "seed": seed,
        "traffic_density": density,
        "n_tasks": n_tasks,
        "road_length_m": cfg.road_length_m,
        "num_edge_nodes": cfg.num_edge_nodes,
        "summary": summary_rows,
        "handover_events_total": len(all_handovers),
        "true_handovers_total": sum(1 for h in all_handovers if h.get("type") == "handover"),
        "associations_total": sum(1 for h in all_handovers if h.get("type") == "associate"),
        "disconnections_total": sum(1 for h in all_handovers if h.get("type") == "disconnect"),
        "handover_count": sum(1 for h in all_handovers if h.get("type") == "handover"),
        "status": "PASS",
    }
    report_json = output_dir / "sumo_smoke_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved experiment report to: {report_json}")
    print("=== SUMO Smoke Experiment Completed Successfully ===")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SUMO microscopic traffic smoke experiment.")
    parser.add_argument("--tasks", type=int, default=60, help="Number of tasks per policy.")
    parser.add_argument("--seed", type=int, default=42, help="Random PRNG seed.")
    parser.add_argument("--density", type=str, default="medium", choices=["low", "medium", "high"])
    args = parser.parse_args()

    run_sumo_smoke_experiment(n_tasks=args.tasks, seed=args.seed, density=args.density)
