"""Phase 5 trust logging CLI: snapshots, event history, trust-over-time plot.

Usage: python scripts/log_trust.py --config configs/smoke.yaml --seeds 0,1,2 --policy resource_aware --out results/trust_smoke
Writes <out>_snapshots.csv (per-task per-node trust), <out>_events.csv (every
trust update with its evidence), <out>.json (summary), and
reports/figures/trust_<out-stem>.png (trust over time, seed 0).
All values come from the actual run. Synthetic env, no attack scenarios yet.
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

from baselines import make_policy  # noqa: E402
from environment import VehicularEdgeEnv  # noqa: E402


def run_seed(config_path: str, policy_name: str, seed: int, num_tasks: int):
    env = VehicularEdgeEnv.from_yaml(config_path)
    env.cfg.num_tasks = num_tasks
    env.reset(seed)
    policy = make_policy(policy_name, seed)
    snaps, succ = [], 0
    for _ in range(num_tasks):
        obs, mask = env.current_observation()
        res = env.step(int(policy(obs, mask, {"candidates": env.candidates(), "env": env})))
        succ += res.success
        for (kind, nid), trust in sorted(env.trust.snapshot().items()):
            snaps.append({"seed": seed, "task_id": res.task_id, "kind": kind,
                          "node_id": nid, "trust": round(trust, 4)})
    events = [{"seed": seed, "task_id": r.task_id, "kind": r.kind, "node_id": r.node_id,
               "event": r.event, "s1": r.s1, "s2": r.s2, "s3": r.s3, "s4": r.s4,
               "t_before": round(r.t_before, 4), "t_after": round(r.t_after, 4)}
              for r in env.trust.history]
    return snaps, events, succ


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log behavioural trust over episodes.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--policy", default="resource_aware")
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    probe = VehicularEdgeEnv.from_yaml(args.config)
    num_tasks = args.tasks or probe.cfg.num_tasks

    all_snaps, all_events, compl = [], [], {}
    for seed in seeds:
        snaps, events, succ = run_seed(args.config, args.policy, seed, num_tasks)
        all_snaps += snaps
        all_events += events
        compl[seed] = succ / num_tasks

    out = args.out or f"results/trust_{Path(args.config).stem}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out + "_snapshots.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "task_id", "kind", "node_id", "trust"])
        w.writeheader()
        w.writerows(all_snaps)
    with open(out + "_events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "task_id", "kind", "node_id", "event",
                                           "s1", "s2", "s3", "s4", "t_before", "t_after"])
        w.writeheader()
        w.writerows(all_events)
    kinds = {}
    for e in all_events:
        kinds[e["event"]] = kinds.get(e["event"], 0) + 1
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"config": args.config, "policy": args.policy, "seeds": seeds,
                   "completion": compl, "event_counts": kinds,
                   "note": "synthetic env; no attack scenarios in Phase 5"}, fh, indent=2)

    figdir = Path("reports/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    seed0 = seeds[0]
    series: dict[tuple, list] = {}
    for s in all_snaps:
        if s["seed"] == seed0:
            series.setdefault((s["kind"], s["node_id"]), []).append((s["task_id"], s["trust"]))
    plt.figure()
    for (kind, nid), pts in sorted(series.items()):
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker=".", markersize=3, label=f"{kind[0]}{nid}")
    plt.xlabel("task")
    plt.ylabel("behavioural trust")
    plt.ylim(0, 1)
    plt.legend(ncol=4, fontsize="small")
    plt.title(f"Behavioural trust over time ({args.policy}, seed {seed0}; synthetic, no attacks)")
    plt.tight_layout()
    figpath = figdir / f"trust_{Path(out).stem}.png"
    plt.savefig(figpath)
    plt.close()

    print(f"completion={compl} events={kinds}")
    print(f"saved: {out}_snapshots.csv, {out}_events.csv, {out}.json, {figpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
