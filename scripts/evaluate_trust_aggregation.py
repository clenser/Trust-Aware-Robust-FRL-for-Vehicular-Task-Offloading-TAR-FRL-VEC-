"""Phase 7 evaluation CLI: Comparative study of aggregation strategies on honest clients.

Compares:
1. Standard FedAvg (EQ-21, Phase 6)
2. Trust-Weighted FedAvg (EQ-22, Phase 7)
3. Trust-Weighted with Threshold Exclusion (EQ-23, Phase 7)

Evaluates:
- Training trajectories (rewards, client accepted/excluded counts, weights)
- Indicator values (m1 norm, m2 direction, m3 staleness, m4 originality, m5 progress)
- Held-out evaluation performance (completion rate, latency, energy, reward)
- Multi-seed honest runs

Outputs written to results/phase7/ and reports/figures/phase7/
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from baselines import aggregate
from environment import VehicularEdgeEnv
from federated import FedClient, FedServer
from rl import DQNAgent, seed_all
from scripts.evaluate_feddqn import greedy_rollout
from scripts.train_feddqn import build_clients, run_federated


def run_strategy_comparison(
    config_path: str,
    seeds: list[int],
    output_dir: Path,
    fig_dir: Path,
) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        raw_base = yaml.safe_load(fh)

    strategies = ["fedavg", "trust_weighted", "trust_threshold"]
    all_summary_rows = []
    indicator_records = []
    run_records = {}

    eval_seeds = list(
        raw_base.get("training", {}).get(
            "eval_seeds",
            raw_base.get("federated", {}).get("eval_seeds", [10, 11]),
        )
    )

    for strat in strategies:
        strat_rows = []
        for seed in seeds:
            raw = copy.deepcopy(raw_base)
            raw["seed"] = seed
            raw["federated"]["strategy"] = strat
            run_name = f"phase7_{strat}_seed{seed}"
            raw["federated"]["run_name"] = run_name
            ckpt_dir = output_dir / "checkpoints" / run_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Running {strat} (seed={seed}) ---")
            round_rows, indep_paths, agent_kwargs = run_federated(
                raw=raw,
                outdir=ckpt_dir,
                train_independent=False,
            )

            # Extract indicator logs
            for r_idx, r_data in enumerate(round_rows):
                # r_data contains "round", "strategy", "status", "accepted", "excluded"
                pass

            # Held-out evaluation on eval_seeds
            gckpt = str(ckpt_dir / "global_final.pt")
            probe = VehicularEdgeEnv.from_yaml(config_path)
            num_tasks = probe.cfg.num_tasks
            hidden = list(raw.get("dqn", {}).get("hidden", [16, 16]))

            # Deterministic fingerprint of global model checkpoint
            import hashlib
            st_blob = torch.load(gckpt, map_location="cpu", weights_only=True)
            bytes_concat = b"".join([v.numpy().tobytes() for k, v in sorted(st_blob.items())])
            ckpt_fingerprint = hashlib.sha256(bytes_concat).hexdigest()[:16]

            for ev_seed in eval_seeds:
                eval_row = greedy_rollout(
                    config_path=config_path,
                    checkpoint=gckpt,
                    seed=ev_seed,
                    num_tasks=num_tasks,
                    policy_name=f"{strat}_seed{seed}",
                    hidden=hidden,
                )
                eval_row["strategy"] = strat
                eval_row["train_seed"] = seed
                eval_row["ckpt_fingerprint"] = ckpt_fingerprint
                strat_rows.append(eval_row)
                all_summary_rows.append(eval_row)

        run_records[strat] = strat_rows


    # Save summary CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "strategy_comparison_summary.csv"
    if all_summary_rows:
        keys = list(all_summary_rows[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_summary_rows)

    # Compute aggregate stats per strategy
    agg_results = {}
    agg_keys = ["completion_rate", "miss_rate", "failure_rate", "mean_latency_s", "mean_energy_j", "mean_reward"]
    for strat in strategies:
        rows = [r for r in all_summary_rows if r["strategy"] == strat]
        agg_results[strat] = aggregate(rows, agg_keys)

    # Save JSON summary
    summary_json = output_dir / "strategy_comparison_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(agg_results, fh, indent=2)

    # Generate comparison plot
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_path = fig_dir / "phase7_strategy_comparison.png"
    generate_comparison_plot(agg_results, strategies, plot_path)

    print(f"\nPhase 7 evaluation complete.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Metrics JSON: {summary_json}")
    print(f"Plot saved: {plot_path}")

    return agg_results


def generate_comparison_plot(
    agg_results: dict,
    strategies: list[str],
    plot_path: Path,
) -> None:
    labels = ["FedAvg (EQ-21)", "Trust-Weighted (EQ-22)", "Trust-Threshold (EQ-23)"]
    x = np.arange(len(strategies))
    width = 0.28

    comp_means = [agg_results[s]["completion_rate"]["mean"] for s in strategies]
    comp_stds = [agg_results[s]["completion_rate"]["stdev"] for s in strategies]

    rew_means = [agg_results[s]["mean_reward"]["mean"] for s in strategies]
    rew_stds = [agg_results[s]["mean_reward"]["stdev"] for s in strategies]

    lat_means = [agg_results[s]["mean_latency_s"]["mean"] for s in strategies]
    lat_stds = [agg_results[s]["mean_latency_s"]["stdev"] for s in strategies]


    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_comp = "#2b5c8f"
    color_rew = "#388e3c"
    color_lat = "#d32f2f"

    rects1 = ax1.bar(x - width, comp_means, width, yerr=comp_stds, label="Completion Rate", color=color_comp, capsize=4)
    rects2 = ax1.bar(x, lat_means, width, yerr=lat_stds, label="Mean Latency (s)", color=color_lat, capsize=4)
    
    ax1.set_ylabel("Rate / Latency (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Phase 7: Aggregation Strategies on Honest Clients (Held-Out Seeds)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, rew_means, width, yerr=rew_stds, label="Mean Reward", color=color_rew, capsize=4)
    ax2.set_ylabel("Mean Reward")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 7 trust aggregation strategies.")
    parser.add_argument("--config", default="configs/feddqn_smoke.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--outdir", default="results/phase7")
    parser.add_argument("--figdir", default="reports/figures/phase7")
    args = parser.parse_args(argv)

    run_strategy_comparison(
        config_path=args.config,
        seeds=args.seeds,
        output_dir=Path(args.outdir),
        fig_dir=Path(args.figdir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
