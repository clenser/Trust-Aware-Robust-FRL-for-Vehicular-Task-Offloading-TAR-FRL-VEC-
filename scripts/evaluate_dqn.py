"""Phase 4 evaluation CLI: greedy DQN vs all four baselines on held-out seeds.

Usage: python scripts/evaluate_dqn.py --config configs/dqn_default.yaml [--checkpoint checkpoints/dqn_default/checkpoint_final.pt]
Outputs (results/dqn_eval_<run_name>.* + reports/figures/dqn_<run_name>_*.png):
per-seed CSV/JSON rows for dqn + local_only/random_valid/nearest_edge/
resource_aware on identical eval seeds, aggregate table, and plots drawn from
the saved logs only.

Fairness caveat (Phase 3): per-link Bernoulli draws are consumed on demand, so
stochastic trajectories can diverge across policies after differing actions;
comparisons share config+seeds but are not perfectly paired trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from baselines import POLICIES, aggregate, episode_metrics, make_policy  # noqa: E402
from environment import VehicularEdgeEnv  # noqa: E402
from rl import DQNAgent, seed_all, task_reward  # noqa: E402
from rl.reward import REWARD_CAVEAT  # noqa: E402
from scripts.run_baselines import run_policy as run_baseline_policy  # noqa: E402

AGG_KEYS = ["completion_rate", "miss_rate", "failure_rate", "mean_latency_s",
            "mean_energy_j", "offload_fraction", "mean_reward"]
NAMES = ["dqn"] + list(POLICIES)


def eval_dqn(config_path: str, checkpoint: str, seed: int, num_tasks: int, hidden: list[int] | None = None) -> dict:
    """Greedy (epsilon=0) held-out rollout of a trained checkpoint."""
    seed_all(10_000 + seed)
    env = VehicularEdgeEnv.from_yaml(config_path)
    env.cfg.num_tasks = num_tasks
    env.reset(seed)
    kwargs = {"eps_start": 0.0, "eps_end": 0.0, "seed": 10_000 + seed}
    if hidden:
        kwargs["hidden"] = hidden
    agent = DQNAgent.load(checkpoint, **kwargs)
    agent.epsilon = 0.0  # evaluation never explores, regardless of stored epsilon
    results = []
    for i in range(num_tasks):
        obs, mask = env.current_observation()
        action = agent.act(obs, mask)
        assert mask[action], f"DQN selected masked action {action}"
        res = env.step(action)
        results.append(res)
    rewards = [task_reward(r.latency_s, r.energy_j, r.success, r.failed, r.deadline_missed, r.dest_trust)
               for r in results]
    row = episode_metrics(len(results), sum(r.success for r in results),
                          sum(r.deadline_missed for r in results), sum(r.failed for r in results),
                          [r.latency_s for r in results], [r.energy_j for r in results],
                          sum(r.action > 0 for r in results), rewards)
    row.update({"config": config_path, "policy": "dqn", "seed": seed,
                "checkpoint": checkpoint, "epsilon": 0.0})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DQN vs baselines.")
    parser.add_argument("--config", default="configs/dqn_default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    run_name = str(raw.get("training", {}).get("run_name", "dqn_run"))
    eval_seeds = list(raw.get("training", {}).get("eval_seeds", [10, 11, 12]))
    probe = VehicularEdgeEnv.from_yaml(args.config)
    num_tasks = probe.cfg.num_tasks
    hidden = list(raw.get("dqn", {}).get("hidden", [64, 64]))
    ckpt = args.checkpoint or f"checkpoints/{run_name}/checkpoint_final.pt"

    rows: list[dict] = []
    for seed in eval_seeds:
        rows.append(eval_dqn(args.config, ckpt, seed, num_tasks, hidden=hidden))
        for name in POLICIES:
            rows.append(run_baseline_policy(args.config, name, seed, num_tasks))
    by_policy = {n: aggregate([r for r in rows if r["policy"] == n], AGG_KEYS) for n in NAMES}

    out = args.out or f"results/dqn_eval_{run_name}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "policy", "seed", "num_tasks", "successes", "deadline_misses",
              "failures", "completion_rate", "miss_rate", "failure_rate",
              "mean_latency_s", "mean_energy_j", "offload_fraction", "mean_reward"]
    with open(out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"reward_note": REWARD_CAVEAT, "checkpoint": ckpt, "eval_seeds": eval_seeds,
                   "fairness_caveat": ("shared config+seeds; trajectories may diverge "
                                       "across policies (Phase 3 on-demand-draw caveat)"),
                   "rows": rows, "aggregate": by_policy}, fh, indent=2)

    # Plots from saved data only.
    figdir = Path("reports/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    labels = NAMES
    means = [by_policy[n]["completion_rate"]["mean"] for n in labels]
    sds = [by_policy[n]["completion_rate"]["stdev"] for n in labels]
    plt.figure()
    plt.bar(labels, means, yerr=sds, capsize=4)
    plt.ylabel("completion rate")
    plt.title(f"Held-out completion by policy ({run_name}, seeds {eval_seeds})")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figdir / f"dqn_{run_name}_completion.png")
    plt.close()

    # Training curve from the saved episode log (real run data only).
    import csv as _csv
    import math as _math
    log_path = Path("checkpoints") / run_name / "episode_log.csv"
    if log_path.is_file():
        with open(log_path, encoding="utf-8") as fh:
            log = list(_csv.DictReader(fh))
        eps = [int(r["episode"]) for r in log]
        rew = [float(r["mean_reward"]) for r in log]
        loss = [float(r["mean_loss"]) if r["mean_loss"] not in ("", "nan") else float("nan")
                for r in log]
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("episode")
        ax1.set_ylabel("mean episode reward")
        ax1.plot(eps, rew, marker="o", label="reward")
        ax1.tick_params(axis="x", labelrotation=0)
        ax2 = ax1.twinx()
        ax2.set_ylabel("mean TD loss")
        ax2.plot(eps, [l for l in loss], marker="s", color="tab:orange", label="loss")
        plt.title(f"Training curve ({run_name}; NaN loss = warm-up, no updates yet)")
        fig.tight_layout()
        plt.savefig(figdir / f"dqn_{run_name}_training.png")
        plt.close()

    print(f"{'policy':<14}{'compl±sd':<16}{'lat(s)±sd':<16}{'reward±sd'}")
    for n in NAMES:
        a = by_policy[n]
        print(f"{n:<14}{a['completion_rate']['mean']:.3f}±{a['completion_rate']['stdev']:.3f}   "
              f"{a['mean_latency_s']['mean']:.2f}±{a['mean_latency_s']['stdev']:.2f}   "
              f"{a['mean_reward']['mean']:.3f}±{a['mean_reward']['stdev']:.3f}")
    print(f"saved: {out}.csv/.json, {figdir}/dqn_{run_name}_completion.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
