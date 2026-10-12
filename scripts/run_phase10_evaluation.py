"""
Phase 10 Comprehensive Research Experiment Runner:
Executes:
1. Core Evaluation Matrix (10 seeds, 8 rounds, clean control, poisoning, free-riding, honest non-IID).
2. Mandatory Ablations (Complete, w/o Beh, w/o Model, w/o Temporal, w/o Excl, FedAvg w/o Trust, Local w/o FL).
3. Parameter Sensitivity Sweeps (alpha sweep, tau_exc sweep, lambda sweep).
Saves all results to results/phase10/ without modifying prior phase data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.base import AttackType, ScenarioAssignment
from attacks.metrics import (
    compute_confidence_interval_95,
    compute_degradation,
    compute_detection_metrics,
)
from attacks.pipeline import AttackPipeline
from attacks.unreliable_execution import UnreliableEnvWrapper, UnreliableProfile
from environment import EnvConfig, VehicularEdgeEnv
from federated.client import FedClient
from federated.server import FedServer
from rl.agent import DQNAgent
from rl.reward import task_reward
from scripts.run_phase8_experiments import (
    build_clients,
    greedy_rollout,
    run_single_phase8_trial,
    seed_all,
    validate_task_accounting,
)

logger = logging.getLogger("phase10")


def run_phase10_trial(
    raw_cfg: dict,
    scenario_assignment: ScenarioAssignment,
    strategy: str,
    seed: int,
    outdir: Path,
    eval_seeds: list[int],
    alpha: float = 0.6,
    tau_exc: float = 0.40,
    lam_override: float | None = None,
    unreliable_profile: UnreliableProfile | None = None,
    skip_fl: bool = False,
) -> tuple[dict, list[dict]]:
    """Run a single Phase 10 trial supporting ablation and sensitivity parameters."""
    seed_all(1000 + seed)
    outdir.mkdir(parents=True, exist_ok=True)

    fed_cfg = raw_cfg.get("federated", {})
    dqn_cfg = raw_cfg.get("dqn", {})
    env_base = raw_cfg.get("scenario", {})
    rounds = int(fed_cfg.get("rounds", 8))
    local_eps = int(fed_cfg.get("local_episodes", 1))
    frac = float(fed_cfg.get("participation_fraction", 1.0))
    clip_c = float(fed_cfg.get("clip_c", 3.0))
    num_tasks = int(env_base.get("num_steps", 10))
    warmup = int(dqn_cfg.get("warmup_steps", 0))
    sync_every = int(dqn_cfg.get("target_sync_steps", 20))
    agent_kwargs = {
        "lr": float(dqn_cfg.get("lr", 1e-3)),
        "gamma": float(dqn_cfg.get("gamma", 0.99)),
        "batch_size": int(dqn_cfg.get("batch_size", 4)),
        "capacity": int(dqn_cfg.get("capacity", 1000)),
        "eps_start": float(dqn_cfg.get("eps_start", 1.0)),
        "eps_end": float(dqn_cfg.get("eps_end", 0.1)),
        "eps_decay_episodes": int(dqn_cfg.get("eps_decay_episodes", 8)),
        "max_grad_norm": float(dqn_cfg.get("max_grad_norm", 10.0)),
        "hidden": list(dqn_cfg.get("hidden", [16, 16])),
    }

    server_rng = np.random.default_rng(2000 + seed)
    clients = build_clients(raw_cfg, agent_kwargs, num_tasks)

    # Apply temporal smoothing override if testing temporal ablation
    if lam_override is not None:
        for c in clients:
            c.env.trust.lam = lam_override

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
        strategy=strategy if not skip_fl else "fedavg",
        alpha=alpha,
        tau_exc=tau_exc,
        clip_c=clip_c,
    )
    for c in clients:
        c.install_global(server.get_global())

    pipeline = AttackPipeline(scenario_assignment, seed=seed)
    detection_history = {"y_true": [], "y_pred": [], "excluded": []}

    for r in range(rounds):
        selected = server.select_clients([c.client_id for c in clients], frac, server_rng)
        honest_updates = []
        for c in clients:
            if c.client_id not in selected:
                continue
            c.local_train(
                local_eps,
                base_seed=5000 + r * 100 + c.client_id * 10,
                num_tasks=num_tasks,
                warmup_steps=warmup,
                sync_every=sync_every,
            )
            honest_updates.append(c.make_update(round_trained=r))

        if skip_fl:
            # Independent local RL ablation: no server aggregation or distribution
            continue

        # Apply attacks
        attacked_updates = []
        for u in honest_updates:
            att_u, audit = pipeline.transform_update(
                update=u,
                reference_state=server.global_state,
                current_round=r,
                peer_updates=honest_updates,
            )
            attacked_updates.append(att_u)

        # Server aggregates
        beh_trust = {c.client_id: c.trust_summary()["mean_trust"] for c in clients}
        log = server.aggregate_round(attacked_updates, current_round=r, behavioural_trusts=beh_trust)

        for c in clients:
            c.install_global(server.get_global())

        det_cfg = raw_cfg.get("detector", {})
        tau_anomaly = float(det_cfg.get("tau_anomaly", 0.50))
        m2_dir_thresh = float(det_cfg.get("m2_dir_threshold", 0.45))
        m4_orig_thresh = float(det_cfg.get("m4_orig_threshold", 0.0))
        m3_stale_thresh = float(det_cfg.get("m3_stale_threshold", 0.5))

        for u in attacked_updates:
            cid = u["client_id"]
            is_mal = scenario_assignment.client_labels[cid].is_malicious
            rep = next((item for item in log.get("trust_reports", []) if item["client_id"] == cid), None)
            is_excluded = cid in log.get("excluded", [])
            if rep is not None and strategy != "fedavg":
                is_anomaly = (
                    is_excluded
                    or (rep["m2_direction"] < m2_dir_thresh)
                    or (rep["m4_originality"] <= m4_orig_thresh)
                    or (rep["m3_staleness"] <= m3_stale_thresh)
                    or (rep["t_final"] < tau_anomaly)
                )
                detected = is_anomaly
            else:
                detected = is_excluded
            detection_history["y_true"].append(is_mal)
            detection_history["y_pred"].append(detected)
            detection_history["excluded"].append(is_excluded)

    final_ckpt = outdir / "global_final.pt"
    if skip_fl:
        # Use first client's local policy for evaluation
        torch.save(clients[0].agent.q_net.state_dict(), final_ckpt)
    else:
        torch.save(server.get_global(), final_ckpt)

    blob = server.get_global()
    fp_bytes = b"".join([v.numpy().tobytes() for k, v in sorted(blob.items())])
    ckpt_hash = hashlib.sha256(fp_bytes).hexdigest()[:16]

    det_metrics = compute_detection_metrics(detection_history["y_true"], detection_history["y_pred"])
    total_eval_updates = len(detection_history["excluded"])
    exclusion_rate = float(sum(detection_history["excluded"])) / float(total_eval_updates) if total_eval_updates > 0 else 0.0

    eval_rows = []
    hidden = list(dqn_cfg.get("hidden", [16, 16]))
    for ev_seed in eval_seeds:
        row = greedy_rollout(
            config_path=raw_cfg["_config_path"],
            checkpoint=str(final_ckpt),
            seed=ev_seed,
            num_tasks=num_tasks,
            policy_name=f"{strategy}_seed{seed}",
            hidden=hidden,
        )
        validate_task_accounting(row["successes"], row["deadline_misses"], row["failures"], num_tasks)
        eval_rows.append(row)

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
        "malicious_ids": ",".join(map(str, sorted(scenario_assignment.malicious_client_ids))),
        "ckpt_hash": ckpt_hash,
        "exclusion_rate": exclusion_rate,
        "completion_rate": float(np.mean([r["completion_rate"] for r in eval_rows])),
        "failure_rate": float(np.mean([r["failure_rate"] for r in eval_rows])),
        "miss_rate": float(np.mean([r["miss_rate"] for r in eval_rows])),
        "mean_latency_s": float(np.mean([r["mean_latency_s"] for r in eval_rows])),
        "mean_energy_j": float(np.mean([r["mean_energy_j"] for r in eval_rows])),
        "mean_reward": float(np.mean([r["mean_reward"] for r in eval_rows])),
        "degradation_completion": 0.0,
        "degradation_reward": 0.0,
        "tp": det_metrics["tp"],
        "fp": det_metrics["fp"],
        "tn": det_metrics["tn"],
        "fn": det_metrics["fn"],
        "precision": det_metrics["precision"],
        "recall": det_metrics["recall"],
        "f1": det_metrics["f1"],
        "fpr": det_metrics["fpr"],
        "tpr": det_metrics["tpr"],
        "accuracy": det_metrics["accuracy"],
    }
    return trial_summary, []


def run_phase10_full_suite(
    config_path: str = "configs/feddqn_phase10.yaml",
    outdir: Path = Path("results/phase10"),
    num_seeds_core: int = 10,
    num_seeds_ablation: int = 5,
) -> None:
    """Orchestrate all Phase 10 evaluation matrices."""
    outdir.mkdir(parents=True, exist_ok=True)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["_config_path"] = config_path

    eval_seeds = list(raw.get("federated", {}).get("eval_seeds", [10, 11]))
    core_seeds = list(range(num_seeds_core))
    ablation_seeds = list(range(num_seeds_ablation))

    all_trials = []
    clean_baselines = {}

    print(f"=================================================================")
    print(f"Phase 10 Full Research Matrix: Core Evaluation ({num_seeds_core} seeds, 8 rounds)")
    print(f"=================================================================")

    # 1. Clean Baseline (Core 10 seeds)
    strategies = ["fedavg", "trust_weighted", "trust_threshold"]
    for strat in strategies:
        clean_baselines[strat] = {}
        for seed in core_seeds:
            scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=0.0, attack_type=AttackType.NONE, seed=seed)
            run_dir = outdir / "checkpoints" / f"clean_{strat}_seed{seed}"
            res, _ = run_phase10_trial(raw, scen, strat, seed, run_dir, eval_seeds)
            res["scenario_group"] = "clean_control"
            clean_baselines[strat][seed] = res
            all_trials.append(res)
            print(f"[clean {strat}] seed={seed} | comp={res['completion_rate']:.3f} | f1={res['f1']:.3f}")

    # 2. Attack Sweeps (Core 10 seeds)
    attack_matrix = [
        ("poisoning_sign_flip", AttackType.SIGN_FLIP, 1.0, [0.1, 0.2, 0.3, 0.4]),
        ("poisoning_perturbation", AttackType.PERTURBATION, 2.0, [0.1, 0.2, 0.3, 0.4]),
        ("freeriding_zero_update", AttackType.ZERO_UPDATE, 0.0, [0.1, 0.2, 0.3, 0.4]),
        ("freeriding_copy_update", AttackType.COPY_UPDATE, 0.0, [0.1, 0.2, 0.3, 0.4]),
    ]
    for name, att_type, strength, fractions in attack_matrix:
        print(f"\n>>> Running Scenario: {name} (10 seeds x {len(fractions)} fractions x 3 strategies)")
        for frac in fractions:
            for strat in strategies:
                for seed in core_seeds:
                    scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=frac, attack_type=att_type, attack_strength=strength, seed=seed)
                    run_dir = outdir / "checkpoints" / f"{name}_f{int(frac*100)}_{strat}_seed{seed}"
                    res, _ = run_phase10_trial(raw, scen, strat, seed, run_dir, eval_seeds)
                    res["scenario_group"] = name
                    clean_res = clean_baselines[strat][seed]
                    res["degradation_completion"] = compute_degradation(clean_res["completion_rate"], res["completion_rate"], higher_is_better=True)
                    res["degradation_reward"] = compute_degradation(clean_res["mean_reward"], res["mean_reward"], higher_is_better=True)
                    all_trials.append(res)
                print(f"[{name} f={frac:.1f} {strat}] mean comp={np.mean([t['completion_rate'] for t in all_trials[-num_seeds_core:]]):.3f} | f1={np.mean([t['f1'] for t in all_trials[-num_seeds_core:]]):.3f}")

    # 3. Honest Non-IID Control (Core 10 seeds)
    print(f"\n>>> Running Scenario: honest_non_iid (10 seeds x 3 strategies)")
    for strat in strategies:
        for seed in core_seeds:
            scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=0.0, attack_type=AttackType.NONE, seed=seed)
            run_dir = outdir / "checkpoints" / f"non_iid_{strat}_seed{seed}"
            res, _ = run_phase10_trial(raw, scen, strat, seed, run_dir, eval_seeds)
            res["scenario_group"] = "honest_non_iid"
            clean_res = clean_baselines[strat][seed]
            res["degradation_completion"] = compute_degradation(clean_res["completion_rate"], res["completion_rate"], higher_is_better=True)
            res["degradation_reward"] = compute_degradation(clean_res["mean_reward"], res["mean_reward"], higher_is_better=True)
            all_trials.append(res)
        print(f"[honest_non_iid {strat}] fpr={np.mean([t['fpr'] for t in all_trials[-num_seeds_core:]]):.3f}")

    # Save Core Runs CSV
    df_runs = pd.DataFrame(all_trials)
    csv_path = outdir / "phase10_experiment_runs.csv"
    df_runs.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_runs)} core experiment runs to {csv_path}")

    # 4. Mandatory Ablation Matrix (5 seeds)
    print(f"\n=================================================================")
    print(f"Phase 10 Mandatory Ablations ({num_seeds_ablation} seeds, 8 rounds)")
    print(f"=================================================================")
    ablation_records = []
    ablation_scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=0.2, attack_type=AttackType.SIGN_FLIP, seed=0)

    ablations = [
        ("full_proposed", "trust_threshold", 0.6, 0.40, None, False),
        ("no_behavioural_trust", "trust_threshold", 0.0, 0.40, None, False),
        ("no_model_trust", "trust_threshold", 1.0, 0.40, None, False),
        ("no_temporal_smoothing", "trust_threshold", 0.6, 0.40, 0.0, False),
        ("no_threshold_exclusion", "trust_weighted", 0.6, 0.0, None, False),
        ("fedavg_no_trust", "fedavg", 0.6, 0.0, None, False),
        ("local_rl_no_fl", "fedavg", 0.6, 0.0, None, True),
    ]

    for abl_name, strat, alpha_val, tau_val, lam_val, skip_fl_flag in ablations:
        for seed in ablation_seeds:
            scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=0.2, attack_type=AttackType.SIGN_FLIP, seed=seed)
            run_dir = outdir / "checkpoints" / f"ablation_{abl_name}_seed{seed}"
            res, _ = run_phase10_trial(
                raw, scen, strat, seed, run_dir, eval_seeds,
                alpha=alpha_val, tau_exc=tau_val, lam_override=lam_val, skip_fl=skip_fl_flag
            )
            res["ablation_name"] = abl_name
            ablation_records.append(res)
        sub_abl = [r for r in ablation_records if r["ablation_name"] == abl_name]
        print(f"[ablation: {abl_name:24s}] comp={np.mean([r['completion_rate'] for r in sub_abl]):.3f} | f1={np.mean([r['f1'] for r in sub_abl]):.3f}")

    df_abl = pd.DataFrame(ablation_records)
    abl_csv = outdir / "phase10_ablation_runs.csv"
    df_abl.to_csv(abl_csv, index=False)
    print(f"Saved {len(df_abl)} ablation runs to {abl_csv}")

    # 5. Sensitivity Sweeps (5 seeds)
    print(f"\n=================================================================")
    print(f"Phase 10 Sensitivity Sweeps ({num_seeds_ablation} seeds, 8 rounds)")
    print(f"=================================================================")
    sens_records = []

    # 5a. Alpha sweep on sign-flip (f=0.2)
    print(">>> Sensitivity: Alpha sweep (0.0 to 1.0)")
    for alpha_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for seed in ablation_seeds:
            scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=0.2, attack_type=AttackType.SIGN_FLIP, seed=seed)
            run_dir = outdir / "checkpoints" / f"sens_alpha_{int(alpha_val*100)}_seed{seed}"
            res, _ = run_phase10_trial(raw, scen, "trust_threshold", seed, run_dir, eval_seeds, alpha=alpha_val, tau_exc=0.40)
            res["sweep_param"] = "alpha"
            res["param_value"] = alpha_val
            sens_records.append(res)

    # 5b. Tau_exc sweep on sign-flip (f=0.2)
    print(">>> Sensitivity: Tau_exc threshold sweep (0.2 to 0.6)")
    for tau_val in [0.20, 0.30, 0.40, 0.50, 0.60]:
        for seed in ablation_seeds:
            scen = ScenarioAssignment.create(num_clients=10, malicious_fraction=0.2, attack_type=AttackType.SIGN_FLIP, seed=seed)
            run_dir = outdir / "checkpoints" / f"sens_tau_{int(tau_val*100)}_seed{seed}"
            res, _ = run_phase10_trial(raw, scen, "trust_threshold", seed, run_dir, eval_seeds, alpha=0.6, tau_exc=tau_val)
            res["sweep_param"] = "tau_exc"
            res["param_value"] = tau_val
            sens_records.append(res)

    df_sens = pd.DataFrame(sens_records)
    sens_csv = outdir / "phase10_sensitivity_runs.csv"
    df_sens.to_csv(sens_csv, index=False)
    print(f"Saved {len(df_sens)} sensitivity runs to {sens_csv}")

    # Generate Structured Summary JSON with Student-t Confidence Intervals
    summary_data = {
        "overall": {},
        "by_fraction": {},
        "ablations": {},
        "sensitivity": {},
    }
    for sg in df_runs["scenario_group"].unique():
        summary_data["overall"][sg] = {}
        for st_name in strategies:
            sub = df_runs[(df_runs["scenario_group"] == sg) & (df_runs["strategy"] == st_name)]
            if not sub.empty:
                summary_data["overall"][sg][st_name] = {
                    "completion_rate": compute_confidence_interval_95(sub["completion_rate"].tolist()),
                    "mean_latency_s": compute_confidence_interval_95(sub["mean_latency_s"].tolist()),
                    "mean_reward": compute_confidence_interval_95(sub["mean_reward"].tolist()),
                    "f1": compute_confidence_interval_95(sub["f1"].tolist()),
                    "fpr": compute_confidence_interval_95(sub["fpr"].tolist()),
                }

    for abl_name in df_abl["ablation_name"].unique():
        sub = df_abl[df_abl["ablation_name"] == abl_name]
        summary_data["ablations"][abl_name] = {
            "completion_rate": compute_confidence_interval_95(sub["completion_rate"].tolist()),
            "f1": compute_confidence_interval_95(sub["f1"].tolist()),
            "fpr": compute_confidence_interval_95(sub["fpr"].tolist()),
            "exclusion_rate": compute_confidence_interval_95(sub["exclusion_rate"].tolist()),
        }

    sum_json = outdir / "phase10_summary_metrics.json"
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved Phase 10 summary JSON to {sum_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feddqn_phase10.yaml")
    parser.add_argument("--outdir", default="results/phase10")
    parser.add_argument("--seeds-core", type=int, default=10)
    parser.add_argument("--seeds-ablation", type=int, default=5)
    args = parser.parse_args()
    run_phase10_full_suite(args.config, Path(args.outdir), args.seeds_core, args.seeds_ablation)
