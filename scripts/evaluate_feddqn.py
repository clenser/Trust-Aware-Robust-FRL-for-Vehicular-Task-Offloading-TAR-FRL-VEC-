"""Phase 6 evaluation CLI: federated-global vs independent-local vs single DQN vs baselines.

Usage: python scripts/evaluate_feddqn.py --config configs/feddqn_default.yaml
Evaluates on held-out seeds (never training seeds):
  feddqn_global : greedy rollout of checkpoints/<run>/global_final.pt
  local_alone_N : greedy rollout of each independent model (mean reported)
  single_dqn    : Phase 4 checkpoints/dqn_default/checkpoint_final.pt (skipped if absent)
  4 Phase 3 baselines on the same seeds.
Outputs results/fedeval_<run>.* + reports/figures/fedeval_<run>_completion.png,
all from actual runs. Fairness caveat (Phase 3 on-demand draws) recorded.
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

from baselines import POLICIES, aggregate, episode_metrics  # noqa: E402
from environment import VehicularEdgeEnv  # noqa: E402
from rl import DQNAgent, seed_all, task_reward  # noqa: E402
from scripts.run_baselines import run_policy as run_baseline_policy  # noqa: E402

AGG_KEYS = ["completion_rate", "miss_rate", "failure_rate", "mean_latency_s",
            "mean_energy_j", "offload_fraction", "mean_reward"]


def greedy_rollout(config_path: str, checkpoint: str, seed: int, num_tasks: int,
                   policy_name: str, hidden: list[int]) -> dict:
    seed_all(20_000 + seed)
    env = VehicularEdgeEnv.from_yaml(config_path)
    env.cfg.num_tasks = num_tasks
    env.reset(seed)
    # Federated checkpoints store raw global parameters; Phase 4 checkpoints
    # store agent dicts with a "q_state" entry. Support both, documented.
    import torch as _torch
    blob = _torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "q_state" in blob:
        agent = DQNAgent.load(checkpoint, eps_start=0.0, eps_end=0.0, seed=20_000 + seed)
    else:
        agent = DQNAgent(obs_dim=env.cfg.obs_dim, action_dim=env.cfg.action_dim,
                         hidden=hidden, seed=20_000 + seed)
        agent.q_net.load_state_dict(blob)
        agent.sync_target()
    agent.epsilon = 0.0
    results = []
    for _ in range(num_tasks):
        obs, mask = env.current_observation()
        action = agent.act(obs, mask)
        assert mask[action], f"{policy_name} selected masked action {action}"
        results.append(env.step(action))
    rewards = [task_reward(r.latency_s, r.energy_j, r.success, r.failed, r.deadline_missed)
               for r in results]
    row = episode_metrics(len(results), sum(r.success for r in results),
                          sum(r.deadline_missed for r in results), sum(r.failed for r in results),
                          [r.latency_s for r in results], [r.energy_j for r in results],
                          sum(r.action > 0 for r in results), rewards)
    row.update({"config": config_path, "policy": policy_name, "seed": seed,
                "checkpoint": checkpoint, "epsilon": 0.0})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate federated DQN vs alternatives.")
    parser.add_argument("--config", default="configs/feddqn_default.yaml")
    parser.add_argument("--global-ckpt", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    run_name = str(raw.get("federated", {}).get("run_name", "feddqn_run"))
    eval_seeds = list(raw.get("training", {}).get("eval_seeds",
                                                  raw.get("federated", {}).get("eval_seeds", [10, 11, 12])))
    probe = VehicularEdgeEnv.from_yaml(args.config)
    num_tasks = probe.cfg.num_tasks
    gckpt = args.global_ckpt or f"checkpoints/{run_name}/global_final.pt"
    names = ["feddqn_global", "local_alone_mean", "single_dqn"] + list(POLICIES)

    rows: list[dict] = []
    hidden = list(raw.get("dqn", {}).get("hidden", [64, 64]))
    for seed in eval_seeds:
        rows.append(greedy_rollout(args.config, gckpt, seed, num_tasks, "feddqn_global", hidden))
        ipaths = sorted((Path("checkpoints") / run_name / "independent").glob("client*_alone.pt"))
        local_rows = [greedy_rollout(args.config, str(p), seed, num_tasks, f"local_alone_{p.stem}", hidden)
                      for p in ipaths]
        rows += local_rows
        if local_rows:  # mean over independent clients, per seed
            mean_row = {"config": args.config, "policy": "local_alone_mean", "seed": seed,
                        "num_tasks": num_tasks}
            for k in ["successes", "deadline_misses", "failures", "completion_rate", "miss_rate",
                      "failure_rate", "mean_latency_s", "mean_energy_j", "offload_fraction",
                      "mean_reward"]:
                mean_row[k] = sum(r[k] for r in local_rows) / len(local_rows)
            rows.append(mean_row)
        single = Path("checkpoints/dqn_default/checkpoint_final.pt")
        if single.is_file():
            with open("configs/dqn_default.yaml", encoding="utf-8") as _fh:
                _single_hidden = list(yaml.safe_load(_fh).get("dqn", {}).get("hidden", [64, 64]))
            rows.append(greedy_rollout("configs/dqn_default.yaml", str(single), seed,
                                       VehicularEdgeEnv.from_yaml("configs/dqn_default.yaml").cfg.num_tasks,
                                       "single_dqn", _single_hidden))
        for name in POLICIES:
            rows.append(run_baseline_policy(args.config, name, seed, num_tasks))

    present = [n for n in names if any(r["policy"] == n for r in rows)]
    by_policy = {n: aggregate([r for r in rows if r["policy"] == n], AGG_KEYS) for n in present}

    out = args.out or f"results/fedeval_{run_name}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "policy", "seed", "num_tasks", "successes", "deadline_misses",
              "failures", "completion_rate", "miss_rate", "failure_rate",
              "mean_latency_s", "mean_energy_j", "offload_fraction", "mean_reward"]
    with open(out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"global_checkpoint": gckpt, "eval_seeds": eval_seeds,
                   "fairness_caveat": "shared config+seeds; trajectories may diverge "
                                      "(Phase 3 on-demand-draw caveat)",
                   "rows": rows, "aggregate": by_policy}, fh, indent=2)

    figdir = Path("reports/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    means = [by_policy[n]["completion_rate"]["mean"] for n in present]
    sds = [by_policy[n]["completion_rate"]["stdev"] for n in present]
    plt.figure(figsize=(9, 4))
    plt.bar(present, means, yerr=sds, capsize=4)
    plt.ylabel("completion rate")
    plt.title(f"Held-out completion ({run_name}, seeds {eval_seeds})")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figdir / f"fedeval_{run_name}_completion.png")
    plt.close()

    print(f"{'policy':<16}{'compl±sd':<16}{'lat(s)±sd':<16}{'reward±sd'}")
    for n in present:
        a = by_policy[n]
        print(f"{n:<16}{a['completion_rate']['mean']:.3f}±{a['completion_rate']['stdev']:.3f}   "
              f"{a['mean_latency_s']['mean']:.2f}±{a['mean_latency_s']['stdev']:.2f}   "
              f"{a['mean_reward']['mean']:.3f}±{a['mean_reward']['stdev']:.3f}")
    print(f"saved: {out}.csv/.json, {figdir}/fedeval_{run_name}_completion.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
