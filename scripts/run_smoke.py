"""Phase 2 CLI smoke simulation: random-valid policy over configs/smoke.yaml.

Usage:  python scripts/run_smoke.py [--config configs/smoke.yaml] [--seed 0] [--tasks 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment import VehicularEdgeEnv  # noqa: E402
from specs import formulas as F  # noqa: E402

# EQ-11 illustration only: latency/energy normalisation bounds are UNSPECIFIED in
# Phase 1 (see environment/README.md "Known spec ambiguity"). Provisional,
# explicitly documented choice for the smoke printout — NOT a tuned reward.
LAT_NORM_S = 10.0
ENE_NORM_J = 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a smoke offloading episode.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tasks", type=int, default=None)
    args = parser.parse_args(argv)

    env = VehicularEdgeEnv.from_yaml(args.config)
    if args.tasks is not None:
        env.cfg.num_tasks = args.tasks
    seed = args.seed if args.seed is not None else env.cfg.seed
    env.reset(seed)

    rng = env._rng  # shared seeded stream; smoke policy only picks valid actions

    def policy(obs, mask):
        valid = [a for a, ok in enumerate(mask) if ok]
        return int(rng.choice(valid))

    results, summary = env.run_episode(policy)
    rewards = [
        F.compute_reward(
            F.clamp01(r.latency_s / LAT_NORM_S),
            F.clamp01(r.energy_j / ENE_NORM_J),
            r.success,
            r.failed,
            r.deadline_missed,
            env.cfg.neutral_trust,
        )
        for r in results
    ]
    print(f"config={args.config} seed={seed} tasks={summary.num_tasks}")
    print(f"success={summary.successes} missed={summary.deadline_misses} failed={summary.failures}")
    print(f"completion_rate={summary.successes / max(1, summary.num_tasks):.3f}")
    print(f"mean_latency_s={summary.mean_latency_s:.4f} mean_energy_j={summary.mean_energy_j:.4f}")
    print(f"offload_fraction={summary.offload_fraction:.3f} mean_reward*={sum(rewards) / len(rewards):.4f}")
    print("* reward uses EQ-11 defaults with neutral trust 0.5 and provisional norms "
          f"({LAT_NORM_S}s, {ENE_NORM_J}J); illustrative only, calibrated in Phase 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
