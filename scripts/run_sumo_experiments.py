"""Multi-Seed SUMO Microscopic Traffic Experiment Matrix Runner.

Executes a multi-seed, statistically aggregated benchmark comparing vehicular
task offloading policies under microscopic highway traffic dynamics using Eclipse SUMO and TraCI.
Persists all results into `results/sumo_experiments/` while leaving historical Phase 10 artifacts untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from environment.config import EnvConfig
from environment.sumo_adapter import is_sumo_available, get_sumo_status
from dashboard.simulator_engine import run_traced_episode

OUTPUT_DIR = REPO_ROOT / "results" / "sumo_experiments"


def run_experiments_matrix(
    seeds: Sequence[int] = (42, 101, 2024),
    densities: Sequence[str] = ("medium",),
    n_tasks: int = 60,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run full factorial policy evaluation across multiple random seeds and traffic densities."""
    if not is_sumo_available():
        status = get_sumo_status()
        print(f"ERROR: Eclipse SUMO or TraCI not available: {status['instructions']}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    policies = [
        "resource_aware",
        "nearest_edge",
        "random_valid",
        "local_only",
    ]

    fed_ckpt = REPO_ROOT / "checkpoints" / "feddqn_default" / "global_final.pt"
    if fed_ckpt.exists():
        policies.append("feddqn_global_final")

    print("======================================================================")
    print("=== TAR-FRL-VEC: Multi-Seed SUMO Microscopic Traffic Benchmark     ===")
    print(f"Policies: {policies}")
    print(f"Seeds: {list(seeds)} | Densities: {list(densities)} | Tasks: {n_tasks}")
    print(f"Output Directory: {output_dir}")
    print("======================================================================")

    detailed_rows = []
    all_handovers = []

    for density in densities:
        for seed in seeds:
            print(f"\n--- Running Seed {seed} | Traffic Density: {density} ---")
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

            for pol in policies:
                print(f"  Executing policy: {pol:<22} (seed={seed}, density={density})...", end="", flush=True)
                snapshots, summary = run_traced_episode(
                    cfg=cfg,
                    policy_name=pol,
                    max_tasks=n_tasks,
                    mobility_mode="sumo",
                    strict_policy=True,
                )
                print(f" Done! (Success: {summary['success_rate']*100:.1f}%, Latency: {summary['mean_latency_s']:.2f}s)")

                row = {
                    "policy": pol,
                    "seed": seed,
                    "density": density,
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
                detailed_rows.append(row)

                for h in summary.get("handover_events", []):
                    h_copy = dict(h)
                    h_copy["policy"] = pol
                    h_copy["seed"] = seed
                    h_copy["density"] = density
                    all_handovers.append(h_copy)

    # 1. Save detailed runs CSV
    df_detailed = pd.DataFrame(detailed_rows)
    detailed_csv = output_dir / "sumo_experiments_detailed.csv"
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"\n[OK] Saved detailed metrics: {detailed_csv}")

    # 2. Save handover events CSV
    df_handovers = pd.DataFrame(all_handovers)
    handovers_csv = output_dir / "sumo_experiments_handovers.csv"
    df_handovers.to_csv(handovers_csv, index=False)
    print(f"[OK] Saved handover events: {handovers_csv}")

    # 3. Compute statistical aggregation (mean and std across seeds)
    metric_cols = ["success_rate", "mean_latency_s", "mean_energy_j", "offload_fraction", "total_handovers", "total_associations"]
    agg_df = df_detailed.groupby(["density", "policy"])[metric_cols].agg(["mean", "std"]).reset_index()
    # Flatten multi-index columns
    agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns]
    agg_csv = output_dir / "sumo_experiments_aggregate.csv"
    agg_df.to_csv(agg_csv, index=False)
    print(f"[OK] Saved statistical aggregate metrics: {agg_csv}")

    # 4. Save JSON summary report
    report = {
        "benchmark": "sumo_microscopic_traffic_matrix",
        "seeds": list(seeds),
        "densities": list(densities),
        "n_tasks_per_run": n_tasks,
        "total_runs": len(detailed_rows),
        "total_handover_events": len(all_handovers),
        "true_handovers_total": sum(1 for h in all_handovers if h.get("type") == "handover"),
        "detailed_runs": detailed_rows,
        "status": "PASS",
    }
    report_json = output_dir / "sumo_experiments_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved benchmark report: {report_json}")

    print("\n=== Aggregate Benchmark Results ===")
    print(agg_df.to_string(index=False))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed SUMO microscopic traffic experiment matrix.")
    parser.add_argument("--seeds", type=str, default="42,101,2024", help="Comma-separated random seeds.")
    parser.add_argument("--densities", type=str, default="medium", help="Comma-separated densities (low,medium,high).")
    parser.add_argument("--tasks", type=int, default=60, help="Number of tasks per run.")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR), help="Output directory path.")
    args = parser.parse_args()

    parsed_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    parsed_densities = [d.strip() for d in args.densities.split(",") if d.strip()]
    out_path = Path(args.out)

    run_experiments_matrix(seeds=parsed_seeds, densities=parsed_densities, n_tasks=args.tasks, output_dir=out_path)
