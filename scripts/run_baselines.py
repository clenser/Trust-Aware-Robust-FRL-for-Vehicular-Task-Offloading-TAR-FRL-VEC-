"""Phase 3 evaluation CLI: every baseline x every seed, same config, saved outputs.

Usage:
  python scripts/run_baselines.py --config configs/smoke.yaml --seeds 0,1,2 --out results/baselines_smoke

Writes <out>.csv (one row per policy x seed) and <out>.json (rows + aggregates
+ run metadata), prints the aggregate table. Reward uses SPEC EQ-11 defaults
with neutral trust 0.5 and provisional norms (10 s, 10 J) — illustrative until
Phase 4 calibrates the reward (same caveat as scripts/run_smoke.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import POLICIES, aggregate, episode_metrics, make_policy  # noqa: E402
from environment import VehicularEdgeEnv  # noqa: E402
from specs import formulas as F  # noqa: E402

LAT_NORM_S = 10.0
ENE_NORM_J = 10.0
REWARD_NOTE = (
    "EQ-11 defaults, neutral trust 0.5, provisional norms (10s, 10J); "
    "illustrative only, calibrated in Phase 4."
)
AGG_KEYS = ["completion_rate", "miss_rate", "failure_rate", "mean_latency_s",
            "mean_energy_j", "offload_fraction", "mean_reward"]


def run_policy(config_path: str, policy_name: str, seed: int, num_tasks: int) -> dict:
    """One policy x one seed: fresh env, identical configured workload start."""
    env = VehicularEdgeEnv.from_yaml(config_path)
    env.cfg.num_tasks = num_tasks
    env.reset(seed)
    policy = make_policy(policy_name, seed)
    results = []
    for _ in range(num_tasks):
        obs, mask = env.current_observation()
        ctx = {"candidates": env.candidates(), "env": env}
        results.append(env.step(int(policy(obs, mask, ctx))))
    rewards = [
        F.compute_reward(
            F.clamp01(r.latency_s / LAT_NORM_S),
            F.clamp01(r.energy_j / ENE_NORM_J),
            r.success, r.failed, r.deadline_missed, F.clamp01(r.dest_trust),
        )
        for r in results
    ]
    row = episode_metrics(
        len(results), sum(r.success for r in results),
        sum(r.deadline_missed for r in results), sum(r.failed for r in results),
        [r.latency_s for r in results], [r.energy_j for r in results],
        sum(r.action > 0 for r in results), rewards,
    )
    row.update({"config": config_path, "policy": policy_name, "seed": seed})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate baseline offloading policies.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    policies = [p.strip() for p in args.policies.split(",") if p.strip() != ""]
    for p in policies:
        try:
            make_policy(p, 0)
        except ValueError:
            print(f"unknown policy: {p}", file=sys.stderr)
            return 2
    probe = VehicularEdgeEnv.from_yaml(args.config)
    num_tasks = args.tasks or probe.cfg.num_tasks

    rows: list[dict] = []
    for name in policies:
        for seed in seeds:
            rows.append(run_policy(args.config, name, seed, num_tasks))

    by_policy = {}
    for name in policies:
        prows = [r for r in rows if r["policy"] == name]
        by_policy[name] = aggregate(prows, AGG_KEYS)

    out = args.out or f"results/baselines_{Path(args.config).stem}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out + ".csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config", "policy", "seed", "num_tasks",
                                                "successes", "deadline_misses", "failures",
                                                "completion_rate", "miss_rate", "failure_rate",
                                                "mean_latency_s", "mean_energy_j",
                                                "offload_fraction", "mean_reward"])
        writer.writeheader()
        writer.writerows(rows)
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"reward_note": REWARD_NOTE, "seeds": seeds, "num_tasks": num_tasks,
                   "rows": rows, "aggregate": by_policy}, fh, indent=2)

    print(f"{'policy':<14}{'compl±sd':<16}{'lat(s)±sd':<16}{'ene(J)±sd':<16}{'offload':<10}{'reward±sd'}")
    for name in policies:
        a = by_policy[name]
        print(f"{name:<14}"
              f"{a['completion_rate']['mean']:.3f}±{a['completion_rate']['stdev']:.3f}   "
              f"{a['mean_latency_s']['mean']:.2f}±{a['mean_latency_s']['stdev']:.2f}   "
              f"{a['mean_energy_j']['mean']:.2f}±{a['mean_energy_j']['stdev']:.2f}   "
              f"{a['offload_fraction']['mean']:.2f}     "
              f"{a['mean_reward']['mean']:.3f}±{a['mean_reward']['stdev']:.3f}")
    print(f"saved: {out}.csv, {out}.json  ({REWARD_NOTE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
