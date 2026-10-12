"""Phase 8 Experiment Runner: Multi-scenario, multi-seed research evaluation.

Evaluates:
- Scenario A: Model Poisoning (sign-flip, perturbation) across attack fractions f in {0.1, 0.2, 0.3, 0.4}.
- Scenario B: Free-Riding (zero-update, copy-update, stale-replay) across attack fractions.
- Scenario C: Unreliable Execution (low, medium, high failure/delay/drop profiles).
- Scenario D: Honest Non-IID Control (f = 0.0 with diverse client conditions).

Compares:
- FedAvg (EQ-21)
- Trust-Weighted FedAvg (EQ-22)
- Trust-Weighted with Threshold Exclusion (EQ-23)

Outputs saved to results/phase8/ and reports/figures/phase8/
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from attacks.base import AttackType, ScenarioAssignment
from attacks.metrics import (
    compute_confidence_interval_95,
    compute_confusion_matrix,
    compute_degradation,
    compute_detection_metrics,
    validate_task_accounting,
)
from attacks.pipeline import AttackPipeline
from attacks.unreliable_execution import UnreliableEnvWrapper, UnreliableProfile
from baselines import aggregate
from environment import EnvConfig, VehicularEdgeEnv
from federated import FedClient, FedServer
from rl import DQNAgent, seed_all
from scripts.evaluate_feddqn import greedy_rollout
from scripts.train_feddqn import build_clients


def run_single_phase8_trial(
    raw_base: dict,
    scenario_assignment: ScenarioAssignment,
    strategy: str,
    seed: int,
    outdir: Path,
    eval_seeds: list[int],
    unreliable_profile: UnreliableProfile | None = None,
) -> tuple[dict, list[dict]]:
    """Execute one federated training run under a given scenario, strategy, and seed."""
    outdir.mkdir(parents=True, exist_ok=True)
    raw = copy.deepcopy(raw_base)
    raw["seed"] = seed
    fed = raw.get("federated", {})
    dqn_cfg = raw.get("dqn", {})

    rounds = int(fed.get("rounds", 2))
    local_eps = int(fed.get("local_episodes", 1))
    num_tasks = int(raw.get("scenario", {}).get("num_steps", 10))
    frac = float(fed.get("participation_fraction", 1.0))
    alpha = float(fed.get("alpha", 0.6))
    tau_exc = float(fed.get("tau_exc", 0.3))
    clip_c = float(fed.get("clip_c", 3.0))

    agent_kwargs = dict(
        hidden=list(dqn_cfg.get("hidden", [16, 16])),
        lr=float(dqn_cfg.get("lr", 1e-3)),
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        batch_size=int(dqn_cfg.get("batch_size", 4)),
        capacity=int(dqn_cfg.get("capacity", 1000)),
        eps_start=float(dqn_cfg.get("eps_start", 1.0)),
        eps_end=float(dqn_cfg.get("eps_end", 0.1)),
        eps_decay_episodes=int(dqn_cfg.get("eps_decay_episodes", 6)),
        max_grad_norm=float(dqn_cfg.get("max_grad_norm", 10.0)),
    )
    warmup = int(dqn_cfg.get("warmup_steps", 0))
    sync_every = int(dqn_cfg.get("target_sync_steps", 20))

    seed_all(seed)
    server_rng = np.random.default_rng(seed + 7)
    clients = build_clients(raw, agent_kwargs, num_tasks)

    # Wrap unreliable clients if applicable
    if unreliable_profile is not None:
        for c in clients:
            if c.client_id in scenario_assignment.malicious_client_ids:
                c.env = UnreliableEnvWrapper(c.env, unreliable_profile, seed=seed + c.client_id)

    template = DQNAgent(
        obs_dim=clients[0].env_cfg.obs_dim,
        action_dim=clients[0].env_cfg.action_dim,
        seed=seed,
        **agent_kwargs,
    )
    server = FedServer(
        template.q_net.state_dict(),
        strategy=strategy,
        alpha=alpha,
        tau_exc=tau_exc,
        clip_c=clip_c,
    )
    for c in clients:
        c.install_global(server.get_global())

    pipeline = AttackPipeline(scenario_assignment, seed=seed)
    round_records = []
    detection_history = {"y_true": [], "y_pred": [], "excluded": []}

    for r in range(rounds):
        selected = server.select_clients([c.client_id for c in clients], frac, server_rng)
        honest_updates = []
        train_rewards = {}

        for c in clients:
            if c.client_id not in selected:
                continue
            out = c.local_train(
                local_eps,
                base_seed=5000 + r * 100 + c.client_id * 10,
                num_tasks=num_tasks,
                warmup_steps=warmup,
                sync_every=sync_every,
            )
            honest_updates.append(c.make_update(round_trained=r))
            train_rewards[c.client_id] = round(out["metrics"]["mean_reward"], 4)

        # Apply attacks
        attacked_updates = []
        round_audits = []
        for u in honest_updates:
            att_u, audit = pipeline.transform_update(
                update=u,
                reference_state=server.global_state,
                current_round=r,
                peer_updates=honest_updates,
            )
            attacked_updates.append(att_u)
            round_audits.append(audit)

        # Server aggregates
        beh_trust = {c.client_id: c.trust_summary()["mean_trust"] for c in clients}
        log = server.aggregate_round(attacked_updates, current_round=r, behavioural_trusts=beh_trust)

        for c in clients:
            c.install_global(server.get_global())

        trust = {c.client_id: round(c.trust_summary()["mean_trust"], 4) for c in clients}

        # Configurable detection thresholds (server-side only, independent of ground truth)
        det_cfg = raw.get("detector", {})
        tau_anomaly = float(det_cfg.get("tau_anomaly", 0.50))
        m2_dir_thresh = float(det_cfg.get("m2_dir_threshold", 0.45))
        m4_orig_thresh = float(det_cfg.get("m4_orig_threshold", 0.0))
        m3_stale_thresh = float(det_cfg.get("m3_stale_threshold", 0.5))

        # Track round detections:
        # A participant is predicted as anomalous if flagged by trust indicators
        for u in attacked_updates:
            cid = u["client_id"]
            is_mal = scenario_assignment.client_labels[cid].is_malicious
            rep = next((item for item in log.get("trust_reports", []) if item["client_id"] == cid), None)
            is_excluded = cid in log.get("excluded", [])
            if rep is not None:
                is_anomaly = (
                    is_excluded
                    or (rep["m2_direction"] < m2_dir_thresh)
                    or (rep["m4_originality"] <= m4_orig_thresh)
                    or (rep["m3_staleness"] <= m3_stale_thresh)
                    or (rep["t_final"] < tau_anomaly)
                )
                detected = is_anomaly
            else:
                # Baseline FedAvg: no trust evaluation or anomaly detection
                detected = is_excluded
            detection_history["y_true"].append(is_mal)
            detection_history["y_pred"].append(detected)
            detection_history["excluded"].append(is_excluded)

        r_row = {
            "round": r,
            "strategy": strategy,
            "status": log["status"],
            "accepted": log["accepted"],
            "excluded": log["excluded"],
            "rewards": train_rewards,
            "trust": trust,
        }
        round_records.append(r_row)
        torch.save(server.get_global(), outdir / f"global_round{r}.pt")

    final_ckpt = outdir / "global_final.pt"
    torch.save(server.get_global(), final_ckpt)

    # Compute model fingerprint
    blob = server.get_global()
    fp_bytes = b"".join([v.numpy().tobytes() for k, v in sorted(blob.items())])
    ckpt_hash = hashlib.sha256(fp_bytes).hexdigest()[:16]

    # Compute detection metrics over all evaluated updates
    det_metrics = compute_detection_metrics(detection_history["y_true"], detection_history["y_pred"])
    total_eval_updates = len(detection_history["excluded"])
    exclusion_rate = float(sum(detection_history["excluded"])) / float(total_eval_updates) if total_eval_updates > 0 else 0.0

    # Held-out evaluation
    eval_rows = []
    hidden = list(raw.get("dqn", {}).get("hidden", [16, 16]))
    for ev_seed in eval_seeds:
        row = greedy_rollout(
            config_path=raw["_config_path"],
            checkpoint=str(final_ckpt),
            seed=ev_seed,
            num_tasks=num_tasks,
            policy_name=f"{strategy}_seed{seed}",
            hidden=hidden,
        )
        validate_task_accounting(row["successes"], row["deadline_misses"], row["failures"], num_tasks)
        eval_rows.append(row)

    # Aggregate held-out evaluations
    comp_mean = float(np.mean([r["completion_rate"] for r in eval_rows]))
    lat_mean = float(np.mean([r["mean_latency_s"] for r in eval_rows]))
    ene_mean = float(np.mean([r["mean_energy_j"] for r in eval_rows]))
    rew_mean = float(np.mean([r["mean_reward"] for r in eval_rows]))
    fail_mean = float(np.mean([r["failure_rate"] for r in eval_rows]))
    miss_mean = float(np.mean([r["miss_rate"] for r in eval_rows]))

    trial_summary = {
        "scenario_group": "clean_control",
        "strategy": strategy,
        "seed": seed,
        "rounds": rounds,
        "attack_type": scenario_assignment.attack_type.value,
        "attack_fraction": scenario_assignment.malicious_fraction,
        "attack_strength": scenario_assignment.attack_strength,
        "num_clients": scenario_assignment.num_clients,
        "num_malicious": len(scenario_assignment.malicious_client_ids),
        "malicious_ids": str(scenario_assignment.malicious_client_ids),
        "ckpt_hash": ckpt_hash,
        "exclusion_rate": exclusion_rate,
        "completion_rate": comp_mean,
        "failure_rate": fail_mean,
        "miss_rate": miss_mean,
        "mean_latency_s": lat_mean,
        "mean_energy_j": ene_mean,
        "mean_reward": rew_mean,
        "degradation_completion": 0.0,
        "degradation_reward": 0.0,
        **det_metrics,
    }


    return trial_summary, eval_rows


def run_phase8_matrix(
    config_path: str,
    seeds: list[int],
    outdir: Path,
    figdir: Path,
    smoke_only: bool = False,
) -> dict[str, Any]:
    """Orchestrates the Phase 8 research experimental matrix across scenarios and strategies."""
    with open(config_path, encoding="utf-8") as fh:
        raw_base = yaml.safe_load(fh)
    raw_base["_config_path"] = config_path

    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    strategies = ["fedavg", "trust_weighted", "trust_threshold"]
    fractions = [0.1, 0.2, 0.3, 0.4] if not smoke_only else [0.2]
    eval_seeds = list(raw_base.get("federated", {}).get("eval_seeds", [10, 11]))

    all_trials: list[dict] = []
    clean_baselines: dict[str, dict[int, dict]] = {s: {} for s in strategies}

    print("=" * 70)
    print("PHASE 8 RESEARCH EXPERIMENT: ADVERSARIAL AND DYNAMIC SCENARIOS")
    print(f"Config: {config_path} | Seeds: {seeds} | Strategies: {strategies}")
    print("=" * 70)

    # 1. Clean Control (f = 0.0)
    print("\n>>> Scenario 1: Clean Control (f = 0.0)")
    for strat in strategies:
        for seed in seeds:
            scen = ScenarioAssignment.create(
                num_clients=10,
                malicious_fraction=0.0,
                attack_type=AttackType.NONE,
                seed=seed,
            )
            run_dir = outdir / "checkpoints" / f"clean_{strat}_seed{seed}"
            res, _ = run_single_phase8_trial(raw_base, scen, strat, seed, run_dir, eval_seeds)
            res["scenario_group"] = "clean_control"
            all_trials.append(res)
            clean_baselines[strat][seed] = res
            print(f"[{strat}] seed={seed} | completion={res['completion_rate']:.3f} reward={res['mean_reward']:.3f}")

    # 2. Scenario A: Model Poisoning (Sign-Flip & Perturbation)
    print("\n>>> Scenario A: Model Poisoning")
    for attack_type in [AttackType.SIGN_FLIP, AttackType.PERTURBATION]:
        strength = 1.0 if attack_type == AttackType.SIGN_FLIP else 2.0
        for f in fractions:
            for strat in strategies:
                for seed in seeds:
                    scen = ScenarioAssignment.create(
                        num_clients=10,
                        malicious_fraction=f,
                        attack_type=attack_type,
                        seed=seed,
                        attack_strength=strength,
                    )
                    run_dir = outdir / "checkpoints" / f"{attack_type.value}_f{f}_{strat}_seed{seed}"
                    res, _ = run_single_phase8_trial(raw_base, scen, strat, seed, run_dir, eval_seeds)
                    res["scenario_group"] = f"poisoning_{attack_type.value}"

                    # Compute degradation vs clean baseline
                    clean_res = clean_baselines[strat][seed]
                    res["degradation_completion"] = compute_degradation(
                        clean_res["completion_rate"], res["completion_rate"], higher_is_better=True
                    )
                    res["degradation_reward"] = compute_degradation(
                        clean_res["mean_reward"], res["mean_reward"], higher_is_better=True
                    )
                    all_trials.append(res)
                    print(
                        f"[{attack_type.value} f={f} {strat}] seed={seed} | "
                        f"F1={res['f1']:.3f} (P={res['precision']:.2f}, R={res['recall']:.2f}) | "
                        f"comp={res['completion_rate']:.3f} (deg={res['degradation_completion']:.3f})"
                    )

    # 3. Scenario B: Free-Riding (Zero, Copy, Stale)
    print("\n>>> Scenario B: Free-Riding")
    for attack_type in [AttackType.ZERO_UPDATE, AttackType.COPY_UPDATE, AttackType.STALE_REPLAY]:
        for f in (fractions if not smoke_only else [0.2]):
            for strat in strategies:
                for seed in seeds:
                    scen = ScenarioAssignment.create(
                        num_clients=10,
                        malicious_fraction=f,
                        attack_type=attack_type,
                        seed=seed,
                        attack_strength=2.0 if attack_type == AttackType.STALE_REPLAY else 1.0,
                    )
                    run_dir = outdir / "checkpoints" / f"{attack_type.value}_f{f}_{strat}_seed{seed}"
                    res, _ = run_single_phase8_trial(raw_base, scen, strat, seed, run_dir, eval_seeds)
                    res["scenario_group"] = f"freeriding_{attack_type.value}"

                    clean_res = clean_baselines[strat][seed]
                    res["degradation_completion"] = compute_degradation(
                        clean_res["completion_rate"], res["completion_rate"], higher_is_better=True
                    )
                    res["degradation_reward"] = compute_degradation(
                        clean_res["mean_reward"], res["mean_reward"], higher_is_better=True
                    )
                    all_trials.append(res)
                    print(
                        f"[{attack_type.value} f={f} {strat}] seed={seed} | "
                        f"F1={res['f1']:.3f} (P={res['precision']:.2f}, R={res['recall']:.2f})"
                    )

    # 4. Scenario C: Unreliable Execution
    print("\n>>> Scenario C: Unreliable Execution Profiles")
    unrel_profiles = {
        "low": UnreliableProfile(fail_prob=0.1, delay_prob=0.1, drop_prob=0.1, name="low"),
        "medium": UnreliableProfile(fail_prob=0.25, delay_prob=0.25, drop_prob=0.2, name="medium"),
        "high": UnreliableProfile(fail_prob=0.5, delay_prob=0.4, drop_prob=0.3, name="high"),
    }
    for level, prof in unrel_profiles.items():
        if smoke_only and level != "medium":
            continue
        for strat in strategies:
            for seed in seeds:
                scen = ScenarioAssignment.create(
                    num_clients=10,
                    malicious_fraction=0.2,  # 2 unreliable nodes
                    attack_type=AttackType.UNRELIABLE,
                    seed=seed,
                )
                run_dir = outdir / "checkpoints" / f"unreliable_{level}_{strat}_seed{seed}"
                res, _ = run_single_phase8_trial(
                    raw_base, scen, strat, seed, run_dir, eval_seeds, unreliable_profile=prof
                )
                res["scenario_group"] = f"unreliable_{level}"
                clean_res = clean_baselines[strat][seed]
                res["degradation_completion"] = compute_degradation(
                    clean_res["completion_rate"], res["completion_rate"], higher_is_better=True
                )
                res["degradation_reward"] = compute_degradation(
                    clean_res["mean_reward"], res["mean_reward"], higher_is_better=True
                )
                all_trials.append(res)
                print(f"[unreliable-{level} {strat}] seed={seed} | comp={res['completion_rate']:.3f}")

    # 5. Scenario D: Honest Non-IID Control
    print("\n>>> Scenario D: Honest Non-IID Control")
    for strat in strategies:
        for seed in seeds:
            scen = ScenarioAssignment.create(
                num_clients=10,
                malicious_fraction=0.0,
                attack_type=AttackType.NONE,
                seed=seed + 555,
            )
            run_dir = outdir / "checkpoints" / f"honest_non_iid_{strat}_seed{seed}"
            res, _ = run_single_phase8_trial(raw_base, scen, strat, seed, run_dir, eval_seeds)
            res["scenario_group"] = "honest_non_iid"
            all_trials.append(res)
            print(f"[honest_non_iid {strat}] seed={seed} | FP={res['fp']} FPR={res['fpr']:.3f}")

    # Save detailed per-trial CSV
    csv_path = outdir / "phase8_experiment_runs.csv"
    if all_trials:
        fieldnames = list(all_trials[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trials)
        print(f"\nSaved raw trial results to {csv_path}")

    # Compute summary aggregations
    summary_metrics = compute_matrix_summary(all_trials)
    json_path = outdir / "phase8_summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary_metrics, fh, indent=2)
    print(f"Saved summary metrics to {json_path}")

    # Generate research figures
    generate_phase8_figures(all_trials, figdir)
    print(f"Generated research figures in {figdir}")

    return summary_metrics


def compute_matrix_summary(all_trials: list[dict]) -> dict[str, Any]:
    """Group results multi-dimensionally:
    1. By (scenario_group, strategy) overall.
    2. By (scenario_group, attack_fraction, strategy) to observe fraction sensitivity.
    """
    metrics_to_stat = [
        "completion_rate",
        "failure_rate",
        "mean_latency_s",
        "mean_reward",
        "exclusion_rate",
        "precision",
        "recall",
        "f1",
        "fpr",
    ]

    # 1. Overall by scenario group and strategy
    groups_overall: dict[str, dict[str, list[dict]]] = {}
    # 2. Detailed by scenario group, attack fraction, and strategy
    groups_by_fraction: dict[str, dict[str, dict[str, list[dict]]]] = {}

    for t in all_trials:
        grp = t.get("scenario_group", "unknown")
        strat = t["strategy"]
        frac_val = float(t.get("attack_fraction", 0.0))
        frac_str = f"f_{frac_val:.1f}"

        # Overall
        if grp not in groups_overall:
            groups_overall[grp] = {}
        if strat not in groups_overall[grp]:
            groups_overall[grp][strat] = []
        groups_overall[grp][strat].append(t)

        # By fraction
        if grp not in groups_by_fraction:
            groups_by_fraction[grp] = {}
        if frac_str not in groups_by_fraction[grp]:
            groups_by_fraction[grp][frac_str] = {}
        if strat not in groups_by_fraction[grp][frac_str]:
            groups_by_fraction[grp][frac_str][strat] = []
        groups_by_fraction[grp][frac_str][strat].append(t)

    summary: dict[str, Any] = {}
    for grp, strats in groups_overall.items():
        summary[grp] = {}
        for strat, trial_list in strats.items():
            summary[grp][strat] = {
                "n_trials": len(trial_list),
            }
            for m in metrics_to_stat:
                vals = [float(item[m]) for item in trial_list if m in item]
                summary[grp][strat][m] = compute_confidence_interval_95(vals)

    # Detailed fraction breakdowns
    fraction_summary: dict[str, Any] = {}
    for grp, fractions in groups_by_fraction.items():
        fraction_summary[grp] = {}
        for frac_str, strats in fractions.items():
            fraction_summary[grp][frac_str] = {}
            for strat, trial_list in strats.items():
                fraction_summary[grp][frac_str][strat] = {
                    "n_trials": len(trial_list),
                }
                for m in metrics_to_stat:
                    vals = [float(item[m]) for item in trial_list if m in item]
                    fraction_summary[grp][frac_str][strat][m] = compute_confidence_interval_95(vals)

    return {
        "overall": summary,
        "by_fraction": fraction_summary,
    }


def generate_phase8_figures(all_trials: list[dict], figdir: Path) -> None:
    """Generate publication-ready comparison figures from experimental trial data."""
    # 1. Detection F1 vs Attack Fraction for Poisoning & Free-Riding
    strategies = ["fedavg", "trust_weighted", "trust_threshold"]
    fractions = [0.1, 0.2, 0.3, 0.4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Poisoning F1
    for strat in strategies:
        f1_means = []
        f1_errs = []
        for f in fractions:
            subset = [
                t
                for t in all_trials
                if t.get("scenario_group") == "poisoning_sign_flip"
                and t["strategy"] == strat
                and t["attack_fraction"] == f
            ]
            if subset:
                vals = [t["f1"] for t in subset]
                ci = compute_confidence_interval_95(vals)
                f1_means.append(ci["mean"])
                f1_errs.append(ci["ci95_half"])
            else:
                f1_means.append(0.0)
                f1_errs.append(0.0)
        ax1.errorbar(fractions, f1_means, yerr=f1_errs, label=strat, marker="o", capsize=4)

    ax1.set_xlabel("Malicious Client Fraction (f)")
    ax1.set_ylabel("Detection F1 Score")
    ax1.set_title("Attack Detection F1 vs Fraction (Sign-Flip Poisoning)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Free-riding F1 (Zero update)
    for strat in strategies:
        f1_means = []
        f1_errs = []
        for f in fractions:
            subset = [
                t
                for t in all_trials
                if t.get("scenario_group") == "freeriding_zero_update"
                and t["strategy"] == strat
                and t["attack_fraction"] == f
            ]
            if subset:
                vals = [t["f1"] for t in subset]
                ci = compute_confidence_interval_95(vals)
                f1_means.append(ci["mean"])
                f1_errs.append(ci["ci95_half"])
            else:
                f1_means.append(0.0)
                f1_errs.append(0.0)
        ax2.errorbar(fractions, f1_means, yerr=f1_errs, label=strat, marker="s", capsize=4)

    ax2.set_xlabel("Malicious Client Fraction (f)")
    ax2.set_ylabel("Detection F1 Score")
    ax2.set_title("Attack Detection F1 vs Fraction (Zero-Update Free-Riding)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(figdir / "detection_f1_vs_fraction.png", dpi=150)
    plt.close(fig)

    # 2. Completion Rate vs Fraction under Poisoning
    fig, ax = plt.subplots(figsize=(7, 5))
    for strat in strategies:
        comp_means = []
        comp_errs = []
        for f in fractions:
            subset = [
                t
                for t in all_trials
                if t.get("scenario_group") == "poisoning_sign_flip"
                and t["strategy"] == strat
                and t["attack_fraction"] == f
            ]
            if subset:
                vals = [t["completion_rate"] for t in subset]
                ci = compute_confidence_interval_95(vals)
                comp_means.append(ci["mean"])
                comp_errs.append(ci["ci95_half"])
            else:
                comp_means.append(0.0)
                comp_errs.append(0.0)
        ax.errorbar(fractions, comp_means, yerr=comp_errs, label=strat, marker="^", capsize=4)

    ax.set_xlabel("Malicious Client Fraction (f)")
    ax.set_ylabel("Task Completion Rate")
    ax.set_title("Global Model Task Completion Under Sign-Flip Poisoning")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "completion_vs_fraction.png", dpi=150)
    plt.close(fig)

    # 3. Honest Non-IID False Positive Rate Analysis
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fpr_vals = []
    strat_labels = []
    for strat in strategies:
        subset = [t for t in all_trials if t.get("scenario_group") == "honest_non_iid" and t["strategy"] == strat]
        fpr_mean = np.mean([t["fpr"] for t in subset]) if subset else 0.0
        fpr_vals.append(fpr_mean)
        strat_labels.append(strat)

    ax.bar(strat_labels, fpr_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.4)
    ax.set_ylabel("False Positive Rate (FPR)")
    ax.set_ylim(0.0, 0.25)
    ax.set_title("Honest Non-IID Control: False Positive Rates")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figdir / "honest_non_iid_fpr.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 8 adversarial and dynamic scenario experiments.")
    parser.add_argument("--config", default="configs/feddqn_phase8.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--outdir", default="results/phase8")
    parser.add_argument("--figdir", default="reports/figures/phase8")
    parser.add_argument("--smoke", action="store_true", help="Run small smoke subset for quick pipeline test")
    args = parser.parse_args(argv)

    run_phase8_matrix(
        config_path=args.config,
        seeds=args.seeds,
        outdir=Path(args.outdir),
        figdir=Path(args.figdir),
        smoke_only=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
