"""Phase 4 training CLI: DQN learns offloading from real env transitions.

Usage: python scripts/train_dqn.py --config configs/dqn_smoke.yaml
Outputs (checkpoints/<run_name>/): episode_log.csv, run_meta.json,
checkpoint_ep{N}.pt + checkpoint_final.pt.

Each episode uses env seed train_seed_base + episode (deterministic curriculum,
disjoint from held-out eval seeds). Transitions are (obs, mask, a, r, next
obs, next mask, done); the last task of each episode is terminal (no bootstrap).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from baselines import episode_metrics  # noqa: E402
from environment import EnvConfig, VehicularEdgeEnv, from_yaml  # noqa: E402
from rl import DQNAgent, Transition, seed_all, task_reward  # noqa: E402
from rl.reward import REWARD_CAVEAT  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train DQN for task offloading.")
    parser.add_argument("--config", default="configs/dqn_smoke.yaml")
    args = parser.parse_args(argv)
    raw = load_cfg(args.config)
    dqn_cfg = raw.get("dqn", {})
    tr_cfg = raw.get("training", {})

    seed_all(int(raw.get("seed", 0)))
    episodes = int(tr_cfg.get("episodes", 3))
    seed_base = int(tr_cfg.get("train_seed_base", 1000))
    run_name = str(tr_cfg.get("run_name", "dqn_run"))
    outdir = Path("checkpoints") / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    probe = VehicularEdgeEnv.from_yaml(args.config)
    agent = DQNAgent(obs_dim=probe.cfg.obs_dim, action_dim=probe.cfg.action_dim,
                     hidden=list(dqn_cfg.get("hidden", [64, 64])), lr=float(dqn_cfg.get("lr", 1e-3)),
                     gamma=float(dqn_cfg.get("gamma", 0.99)), batch_size=int(dqn_cfg.get("batch_size", 32)),
                     capacity=int(dqn_cfg.get("capacity", 10000)), eps_start=float(dqn_cfg.get("eps_start", 1.0)),
                     eps_end=float(dqn_cfg.get("eps_end", 0.05)),
                     eps_decay_episodes=int(dqn_cfg.get("eps_decay_episodes", 20)),
                     max_grad_norm=float(dqn_cfg.get("max_grad_norm", 10.0)), seed=int(raw.get("seed", 0)))
    warmup = int(dqn_cfg.get("warmup_steps", 500))
    sync_every = int(dqn_cfg.get("target_sync_steps", 200))
    ckpt_every = int(tr_cfg.get("checkpoint_every", 5))

    total_transitions = 0
    log_rows: list[dict] = []
    for ep in range(episodes):
        env = VehicularEdgeEnv.from_yaml(args.config)
        env.reset(seed_base + ep)
        agent.update_epsilon(ep)
        ep_rewards: list[float] = []
        ep_losses: list[float] = []
        succ = miss = fail = off = 0
        lat: list[float] = []
        ene: list[float] = []
        n_tasks = env.cfg.num_tasks
        for step_i in range(n_tasks):
            obs, mask = env.current_observation()
            action = agent.act(obs, mask)
            res = env.step(action)
            reward = task_reward(res.latency_s, res.energy_j, res.success, res.failed, res.deadline_missed, res.dest_trust)
            done = step_i == n_tasks - 1
            agent.buffer.push(Transition(obs, mask, action, reward, res.obs, res.mask, done))
            total_transitions += 1
            ep_rewards.append(reward)
            succ += res.success
            miss += res.deadline_missed
            fail += res.failed
            off += res.action > 0
            lat.append(res.latency_s)
            ene.append(res.energy_j)
            if total_transitions > warmup and len(agent.buffer) >= agent.batch_size:
                ep_losses.append(agent.train_step())
                if agent.train_steps % sync_every == 0:
                    agent.sync_target()
        m = episode_metrics(n_tasks, succ, miss, fail, lat, ene, off, ep_rewards)
        row = {"episode": ep, "epsilon": round(agent.epsilon, 4),
               "mean_loss": (sum(ep_losses) / len(ep_losses)) if ep_losses else float("nan"),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}
        log_rows.append(row)
        print(f"ep={ep} eps={agent.epsilon:.3f} R={m['mean_reward']:.3f} "
              f"compl={m['completion_rate']:.3f} loss={row['mean_loss']:.4f} offload={m['offload_fraction']:.2f}")
        if (ep + 1) % ckpt_every == 0 or ep == episodes - 1:
            agent.save(outdir / f"checkpoint_ep{ep + 1}.pt")
    agent.save(outdir / "checkpoint_final.pt")

    with open(outdir / "episode_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    meta = {"config_file": args.config, "config": raw, "reward_caveat": REWARD_CAVEAT,
            "train_seeds": [seed_base + e for e in range(episodes)],
            "total_transitions": total_transitions, "train_steps": agent.train_steps,
            "final_checkpoint": str(outdir / "checkpoint_final.pt"),
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": numpy.__version__}}
    with open(outdir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved: {outdir}/episode_log.csv, run_meta.json, checkpoints ({REWARD_CAVEAT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
