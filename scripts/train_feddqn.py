"""Phase 6 federated training CLI: FedAvg rounds over simulated DQN clients.

Usage: python scripts/train_feddqn.py --config configs/feddqn_smoke.yaml
Outputs (checkpoints/<run_name>/): rounds_log.csv, run_meta.json,
global_round{R}.pt + global_final.pt, independent/ subdir (if enabled).

Workflow per round: seeded client selection -> optional seeded dropout ->
local training (private env + private buffer) -> parameter-only updates ->
FedAvg -> global distribution (online AND target nets). Behavioural trust is
logged per client for analysis only; it never weights updates (Phase 7).
This is a synthetic single-process simulation, not deployed V2X communication.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from federated import FedClient, FedServer  # noqa: E402
from rl import DQNAgent, seed_all  # noqa: E402


def build_clients(raw: dict, agent_kwargs: dict, num_tasks: int):
    from environment.config import EnvConfig
    base = EnvConfig.from_raw(raw)
    fed = raw.get("federated", {})
    conds = fed.get("client_conditions", [{}])
    clients = []
    for i in range(int(fed.get("num_clients", 2))):
        cfg = copy.deepcopy(base)
        for k, v in conds[i % len(conds)].items():
            if k == "task_load":
                cfg.task_load = str(v)
            elif k == "mobility":
                cfg.mobility = str(v)
            elif k == "link_reliability":
                cfg.link_reliability = float(v)
            else:
                raise ValueError(f"unknown client override: {k}")
        cfg.num_tasks = num_tasks
        cfg.validate()
        clients.append(FedClient(i, cfg, agent_kwargs, seed=int(raw.get("seed", 0)) + 100 * i))
    return clients


def run_federated(raw: dict, outdir: Path, train_independent: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    fed = raw.get("federated", {})

    dqn_cfg = raw.get("dqn", {})
    rounds = int(fed.get("rounds", 1))
    local_eps = int(fed.get("local_episodes", 1))
    num_tasks = int(raw.get("scenario", {}).get("num_steps", 25))
    frac = float(fed.get("participation_fraction", 1.0))
    dropout = float(fed.get("dropout_prob", 0.0))
    agent_kwargs = dict(hidden=list(dqn_cfg.get("hidden", [64, 64])), lr=float(dqn_cfg.get("lr", 1e-3)),
                        gamma=float(dqn_cfg.get("gamma", 0.99)), batch_size=int(dqn_cfg.get("batch_size", 32)),
                        capacity=int(dqn_cfg.get("capacity", 10000)), eps_start=float(dqn_cfg.get("eps_start", 1.0)),
                        eps_end=float(dqn_cfg.get("eps_end", 0.05)),
                        eps_decay_episodes=int(dqn_cfg.get("eps_decay_episodes", 12)),
                        max_grad_norm=float(dqn_cfg.get("max_grad_norm", 10.0)))
    warmup = int(dqn_cfg.get("warmup_steps", 200))
    sync_every = int(dqn_cfg.get("target_sync_steps", 100))

    strategy = str(fed.get("strategy", raw.get("strategy", "fedavg")))
    alpha = float(fed.get("alpha", raw.get("alpha", 0.6)))
    tau_exc = float(fed.get("tau_exc", raw.get("tau_exc", 0.3)))
    clip_c = float(fed.get("clip_c", raw.get("clip_c", 3.0)))

    seed_all(int(raw.get("seed", 0)))
    server_rng = np.random.default_rng(int(raw.get("seed", 0)) + 7)
    clients = build_clients(raw, agent_kwargs, num_tasks)
    template = DQNAgent(obs_dim=clients[0].env_cfg.obs_dim, action_dim=clients[0].env_cfg.action_dim,
                        seed=int(raw.get("seed", 0)), **agent_kwargs)
    server = FedServer(template.q_net.state_dict(), strategy=strategy, alpha=alpha,
                       tau_exc=tau_exc, clip_c=clip_c)
    for c in clients:  # round 0 starts from the common init
        c.install_global(server.get_global())

    round_rows: list[dict] = []
    for r in range(rounds):
        selected = server.select_clients([c.client_id for c in clients], frac, server_rng)
        updates, train_rewards = [], {}
        for c in clients:
            if c.client_id not in selected:
                continue
            if server_rng.random() < dropout:
                train_rewards[c.client_id] = "dropped"
                continue
            out = c.local_train(local_eps, base_seed=5000 + r * 100 + c.client_id * 10,
                                num_tasks=num_tasks, warmup_steps=warmup, sync_every=sync_every)
            updates.append(c.make_update(round_trained=r))
            train_rewards[c.client_id] = round(out["metrics"]["mean_reward"], 4)
        
        # Collect client behavioural trust summary for server
        beh_trust = {c.client_id: c.trust_summary()["mean_trust"] for c in clients}
        log = server.aggregate_round(updates, current_round=r, behavioural_trusts=beh_trust)
        for c in clients:  # distribute to ALL clients (incl. non-participants)
            c.install_global(server.get_global())
        trust = {c.client_id: round(c.trust_summary()["mean_trust"], 4) for c in clients}
        row = {"round": r, "selected": selected, "train_rewards": train_rewards,
               "mean_trust": trust, **{k: log[k] for k in
               ("strategy", "status", "participants", "accepted", "excluded", "weights",
                "staleness", "upload_bytes", "download_bytes_per_client", "agg_time_s", "trust_reports")}}

        round_rows.append(row)
        print(f"round={r} strategy={log['strategy']} status={log['status']} "
              f"accepted={log['accepted']} excluded={log['excluded']} "
              f"rewards={train_rewards} trust={trust}")
        torch.save(server.get_global(), outdir / f"global_round{r}.pt")
    torch.save(server.get_global(), outdir / "global_final.pt")

    if round_rows:
        with open(outdir / "rounds_log.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(round_rows[0].keys()))
            w.writeheader()
            w.writerows(round_rows)


    indep_paths: list[str] = []
    if train_independent:  # same budget, no server: isolated local learners
        idir = outdir / "independent"
        idir.mkdir(exist_ok=True)
        for c in clients:
            iso = FedClient(900 + c.client_id, c.env_cfg, agent_kwargs, seed=c.seed + 5000)
            iso.local_train(rounds * local_eps, base_seed=5000 + c.client_id * 10,
                            num_tasks=num_tasks, warmup_steps=warmup, sync_every=sync_every)
            p = idir / f"client{c.client_id}_alone.pt"
            torch.save({k: v.cpu() for k, v in iso.agent.q_net.state_dict().items()}, p)
            indep_paths.append(str(p))
    return round_rows, indep_paths, agent_kwargs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train federated DQN (FedAvg).")
    parser.add_argument("--config", default="configs/feddqn_smoke.yaml")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    run_name = str(raw.get("federated", {}).get("run_name", "feddqn_run"))
    outdir = Path("checkpoints") / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    round_rows, indep_paths, agent_kwargs = run_federated(
        raw, outdir, bool(raw.get("federated", {}).get("train_independent", True)))
    with open(outdir / "rounds_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(round_rows[0].keys()))
        w.writeheader()
        w.writerows(round_rows)
    meta = {"config_file": args.config, "config": raw, "agent_kwargs": agent_kwargs,
            "final_global": str(outdir / "global_final.pt"),
            "independent_models": indep_paths,
            "note": "single-process synthetic FL simulation; payloads analytical (4B/param)",
            "versions": {"python": platform.python_version(), "torch": torch.__version__}}
    with open(outdir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved: {outdir}/rounds_log.csv, run_meta.json, globals, independent={len(indep_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
