"""Phase 10 Final Remediation: Full Matrix Orchestration and Execution.

Implements the frozen plan from reports/PHASE10_FINAL_PLAN.md:
- Matrix A: Core evaluation (630 runs across Clean IID, Honest Non-IID, Core Attacks, and Unreliable Execution)
- Matrix B: Mandatory ablations (70 runs across 7 configurations x 2 scenarios x 5 seeds)
- Matrix C: Sensitivity analysis (180 runs across 4 parameters x 2 scenarios x 5 seeds)
Total: 880 experimental runs.

Ensures:
- Deep-copied configurations per trial
- Deterministic seed derivation across model, client envs, exploration, and attacks
- Held-out evaluation seeds independent from training seeds
- Unreliable execution tracking separate from malicious attack detection
- Strict task accounting and ground-truth isolation
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys
import time
from typing import Any

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
    compute_confusion_matrix,
    compute_degradation,
    compute_detection_metrics,
    validate_task_accounting,
)
from attacks.pipeline import AttackPipeline
from attacks.unreliable_execution import UnreliableEnvWrapper, UnreliableProfile
from environment import EnvConfig, VehicularEdgeEnv
from federated.client import FedClient
from federated.server import FedServer
from rl.agent import DQNAgent, seed_all
from scripts.evaluate_feddqn import greedy_rollout

logger = logging.getLogger("phase10_final")


@dataclass
class PlanItem:
    run_id: str
    matrix_type: str  # core, ablation, sensitivity
    scenario_group: str
    attack_type: str
    attack_fraction: float
    attack_strength: float
    unreliable_severity: str
    unreliable_fraction: float
    strategy: str
    master_seed: int
    eval_seeds: list[int]
    is_iid: bool
    ablation_name: str = "none"
    sweep_param: str = "none"
    param_value: float = 0.0
    alpha: float = 0.60
    tau_exc: float = 0.40
    lambda_smooth: float | None = None
    tau_anomaly: float = 0.50
    skip_fl: bool = False


def build_phase10_experiment_plan(
    eval_seeds: list[int] | None = None,
    num_seeds_core: int = 10,
    num_seeds_secondary: int = 5,
) -> list[PlanItem]:
    """Explicitly enumerate every configuration in the Phase 10 frozen plan."""
    if eval_seeds is None:
        eval_seeds = [100, 101, 102, 103, 104]

    plan: list[PlanItem] = []
    strategies = ["fedavg", "trust_weighted", "trust_threshold"]
    core_seeds = list(range(num_seeds_core))
    sec_seeds = list(range(num_seeds_secondary))

    # ==========================================
    # Matrix A: Core Evaluation (630 runs)
    # ==========================================

    # A1. Clean IID Control (30 runs)
    for strat in strategies:
        for seed in core_seeds:
            plan.append(PlanItem(
                run_id=f"core_clean_iid_{strat}_seed{seed}",
                matrix_type="core",
                scenario_group="clean_iid",
                attack_type="none",
                attack_fraction=0.0,
                attack_strength=0.0,
                unreliable_severity="none",
                unreliable_fraction=0.0,
                strategy=strat,
                master_seed=seed,
                eval_seeds=eval_seeds,
                is_iid=True,
            ))

    # A2. Honest Non-IID Control (30 runs)
    for strat in strategies:
        for seed in core_seeds:
            plan.append(PlanItem(
                run_id=f"core_honest_non_iid_{strat}_seed{seed}",
                matrix_type="core",
                scenario_group="honest_non_iid",
                attack_type="none",
                attack_fraction=0.0,
                attack_strength=0.0,
                unreliable_severity="none",
                unreliable_fraction=0.0,
                strategy=strat,
                master_seed=seed,
                eval_seeds=eval_seeds,
                is_iid=False,
            ))

    # A3. Core Attack Sweeps (480 runs)
    core_attacks = [
        ("poisoning_sign_flip", "sign_flip", 1.0, [0.1, 0.2, 0.3, 0.4]),
        ("poisoning_perturbation", "perturbation", 2.0, [0.1, 0.2, 0.3, 0.4]),
        ("freeriding_zero_update", "zero_update", 0.0, [0.1, 0.2, 0.3, 0.4]),
        ("freeriding_copy_update", "copy_update", 0.0, [0.1, 0.2, 0.3, 0.4]),
    ]
    for sg_name, att_type, strength, fractions in core_attacks:
        for frac in fractions:
            for strat in strategies:
                for seed in core_seeds:
                    plan.append(PlanItem(
                        run_id=f"core_{sg_name}_f{int(frac*100)}_{strat}_seed{seed}",
                        matrix_type="core",
                        scenario_group=sg_name,
                        attack_type=att_type,
                        attack_fraction=frac,
                        attack_strength=strength,
                        unreliable_severity="none",
                        unreliable_fraction=0.0,
                        strategy=strat,
                        master_seed=seed,
                        eval_seeds=eval_seeds,
                        is_iid=False,
                    ))

    # A4. Unreliable Execution (90 runs)
    severities = ["low", "medium", "high"]
    for sev in severities:
        for strat in strategies:
            for seed in core_seeds:
                plan.append(PlanItem(
                    run_id=f"core_unreliable_{sev}_{strat}_seed{seed}",
                    matrix_type="core",
                    scenario_group=f"unreliable_{sev}",
                    attack_type="none",
                    attack_fraction=0.0,
                    attack_strength=0.0,
                    unreliable_severity=sev,
                    unreliable_fraction=0.20,
                    strategy=strat,
                    master_seed=seed,
                    eval_seeds=eval_seeds,
                    is_iid=False,
                ))

    # ==========================================
    # Matrix B: Mandatory Ablations (70 runs)
    # ==========================================
    ablation_specs = [
        ("full_proposed", "trust_threshold", 0.60, 0.40, None, 0.50, False),
        ("no_behavioural_trust", "trust_threshold", 0.00, 0.40, None, 0.50, False),
        ("no_model_trust", "trust_threshold", 1.00, 0.40, None, 0.50, False),
        ("no_temporal_smoothing", "trust_threshold", 0.60, 0.40, 0.0, 0.50, False),
        ("no_threshold_exclusion", "trust_weighted", 0.60, 0.00, None, 0.50, False),
        ("fedavg_no_trust", "fedavg", 0.60, 0.00, None, 0.50, False),
        ("local_rl_no_fl", "fedavg", 0.60, 0.00, None, 0.50, True),
    ]
    ablation_scenarios = [
        ("poisoning_sign_flip", "sign_flip", 1.0),
        ("freeriding_zero_update", "zero_update", 0.0),
    ]
    for abl_name, strat, a_val, tau_val, lam_val, tau_anom, skip_fl_flag in ablation_specs:
        for sc_name, att_type, strength in ablation_scenarios:
            for seed in sec_seeds:
                plan.append(PlanItem(
                    run_id=f"abl_{abl_name}_{sc_name}_seed{seed}",
                    matrix_type="ablation",
                    scenario_group=sc_name,
                    attack_type=att_type,
                    attack_fraction=0.20,
                    attack_strength=strength,
                    unreliable_severity="none",
                    unreliable_fraction=0.0,
                    strategy=strat,
                    master_seed=seed,
                    eval_seeds=eval_seeds,
                    is_iid=False,
                    ablation_name=abl_name,
                    alpha=a_val,
                    tau_exc=tau_val,
                    lambda_smooth=lam_val,
                    tau_anomaly=tau_anom,
                    skip_fl=skip_fl_flag,
                ))

    # ==========================================
    # Matrix C: Sensitivity Analysis (180 runs)
    # ==========================================
    sens_scenarios = [
        ("poisoning_sign_flip", "sign_flip", 1.0),
        ("freeriding_stale_replay", "stale_replay", 0.0),
    ]
    # Sweep 1: Alpha (6 values)
    for a_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for sc_name, att_type, strength in sens_scenarios:
            for seed in sec_seeds:
                plan.append(PlanItem(
                    run_id=f"sens_alpha_{int(a_val*100)}_{sc_name}_seed{seed}",
                    matrix_type="sensitivity",
                    scenario_group=sc_name,
                    attack_type=att_type,
                    attack_fraction=0.20,
                    attack_strength=strength,
                    unreliable_severity="none",
                    unreliable_fraction=0.0,
                    strategy="trust_threshold",
                    master_seed=seed,
                    eval_seeds=eval_seeds,
                    is_iid=False,
                    sweep_param="alpha",
                    param_value=a_val,
                    alpha=a_val,
                    tau_exc=0.40,
                ))

    # Sweep 2: Tau Exclusion (5 values)
    for tau_val in [0.20, 0.30, 0.40, 0.50, 0.60]:
        for sc_name, att_type, strength in sens_scenarios:
            for seed in sec_seeds:
                plan.append(PlanItem(
                    run_id=f"sens_tau_exc_{int(tau_val*100)}_{sc_name}_seed{seed}",
                    matrix_type="sensitivity",
                    scenario_group=sc_name,
                    attack_type=att_type,
                    attack_fraction=0.20,
                    attack_strength=strength,
                    unreliable_severity="none",
                    unreliable_fraction=0.0,
                    strategy="trust_threshold",
                    master_seed=seed,
                    eval_seeds=eval_seeds,
                    is_iid=False,
                    sweep_param="tau_exc",
                    param_value=tau_val,
                    alpha=0.60,
                    tau_exc=tau_val,
                ))

    # Sweep 3: Temporal Smoothing Lambda (4 values)
    for lam_val in [0.50, 0.70, 0.90, 0.95]:
        for sc_name, att_type, strength in sens_scenarios:
            for seed in sec_seeds:
                plan.append(PlanItem(
                    run_id=f"sens_lambda_{int(lam_val*100)}_{sc_name}_seed{seed}",
                    matrix_type="sensitivity",
                    scenario_group=sc_name,
                    attack_type=att_type,
                    attack_fraction=0.20,
                    attack_strength=strength,
                    unreliable_severity="none",
                    unreliable_fraction=0.0,
                    strategy="trust_threshold",
                    master_seed=seed,
                    eval_seeds=eval_seeds,
                    is_iid=False,
                    sweep_param="lambda_smooth",
                    param_value=lam_val,
                    alpha=0.60,
                    tau_exc=0.40,
                    lambda_smooth=lam_val,
                ))

    # Sweep 4: Anomaly Threshold (3 values)
    for anom_val in [0.40, 0.50, 0.60]:
        for sc_name, att_type, strength in sens_scenarios:
            for seed in sec_seeds:
                plan.append(PlanItem(
                    run_id=f"sens_anomaly_{int(anom_val*100)}_{sc_name}_seed{seed}",
                    matrix_type="sensitivity",
                    scenario_group=sc_name,
                    attack_type=att_type,
                    attack_fraction=0.20,
                    attack_strength=strength,
                    unreliable_severity="none",
                    unreliable_fraction=0.0,
                    strategy="trust_threshold",
                    master_seed=seed,
                    eval_seeds=eval_seeds,
                    is_iid=False,
                    sweep_param="tau_anomaly",
                    param_value=anom_val,
                    alpha=0.60,
                    tau_exc=0.40,
                    tau_anomaly=anom_val,
                ))

    return plan


def build_clients_with_seed(
    raw: dict,
    agent_kwargs: dict,
    num_tasks: int,
    master_seed: int,
    is_iid: bool = False,
) -> list[FedClient]:
    """Build client environments with genuine master seed derivation and IID/Non-IID control."""
    from environment.config import EnvConfig
    base = EnvConfig.from_raw(raw)
    fed = raw.get("federated", {})
    conds = fed.get("client_conditions", [{}])
    num_clients = int(fed.get("num_clients", 10))
    clients = []

    for i in range(num_clients):
        cfg = copy.deepcopy(base)
        if not is_iid:
            for k, v in conds[i % len(conds)].items():
                if k == "task_load":
                    cfg.task_load = str(v)
                elif k == "mobility":
                    cfg.mobility = str(v)
                elif k == "link_reliability":
                    cfg.link_reliability = float(v)
                else:
                    raise ValueError(f"unknown client override: {k}")
        else:
            # Clean IID baseline: homogeneous medium load, medium mobility, 0.90 reliability
            cfg.task_load = "medium"
            cfg.mobility = "medium"
            cfg.link_reliability = 0.90

        cfg.num_tasks = num_tasks
        cfg.validate()

        # Seed propagation: unique, deterministic client environment and exploration seeds
        client_env_seed = 20000 + master_seed * 100 + i
        client_agent_kwargs = copy.deepcopy(agent_kwargs)
        if "seed" in client_agent_kwargs:
            del client_agent_kwargs["seed"]

        c = FedClient(i, cfg, client_agent_kwargs, seed=client_env_seed)
        clients.append(c)

    return clients


def execute_plan_item(
    item: PlanItem,
    raw_cfg_base: dict,
    checkpoints_dir: Path,
) -> dict[str, Any]:
    """Execute a single plan item deterministically."""
    # Master seed isolation
    seed_all(10000 + item.master_seed)

    raw_cfg = copy.deepcopy(raw_cfg_base)
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
    hidden = list(dqn_cfg.get("hidden", [16, 16]))

    agent_kwargs = {
        "lr": float(dqn_cfg.get("lr", 1e-3)),
        "gamma": float(dqn_cfg.get("gamma", 0.99)),
        "batch_size": int(dqn_cfg.get("batch_size", 4)),
        "capacity": int(dqn_cfg.get("capacity", 1000)),
        "eps_start": float(dqn_cfg.get("eps_start", 1.0)),
        "eps_end": float(dqn_cfg.get("eps_end", 0.1)),
        "eps_decay_episodes": int(dqn_cfg.get("eps_decay_episodes", 8)),
        "max_grad_norm": float(dqn_cfg.get("max_grad_norm", 10.0)),
        "hidden": hidden,
    }

    # Deterministic client construction with master seed derivation
    clients = build_clients_with_seed(raw_cfg, agent_kwargs, num_tasks, item.master_seed, is_iid=item.is_iid)

    # Apply temporal smoothing override if specified
    if item.lambda_smooth is not None:
        for c in clients:
            c.env.trust.lam = item.lambda_smooth

    # Build scenario assignment
    attack_enum = AttackType(item.attack_type)
    scen = ScenarioAssignment.create(
        num_clients=len(clients),
        malicious_fraction=item.attack_fraction,
        attack_type=attack_enum,
        seed=item.master_seed,
        attack_strength=item.attack_strength,
        unreliable_fraction=item.unreliable_fraction,
        unreliable_severity=item.unreliable_severity,
    )

    # Wrap unreliable clients if specified
    unreliable_profile = None
    if item.unreliable_fraction > 0.0 and item.unreliable_severity != "none":
        profiles = {
            "low": UnreliableProfile(fail_prob=0.10, delay_prob=0.10, drop_prob=0.10, name="low"),
            "medium": UnreliableProfile(fail_prob=0.25, delay_prob=0.25, drop_prob=0.20, name="medium"),
            "high": UnreliableProfile(fail_prob=0.50, delay_prob=0.40, drop_prob=0.30, name="high"),
        }
        unreliable_profile = profiles.get(item.unreliable_severity, profiles["medium"])
        for c in clients:
            if c.client_id in scen.unreliable_client_ids:
                c.env = UnreliableEnvWrapper(
                    c.env,
                    unreliable_profile,
                    seed=60000 + item.master_seed * 100 + c.client_id,
                )

    # Model template
    template_seed = 10000 + item.master_seed
    template = DQNAgent(
        obs_dim=clients[0].env_cfg.obs_dim,
        action_dim=clients[0].env_cfg.action_dim,
        seed=template_seed,
        **agent_kwargs,
    )

    server = FedServer(
        template.q_net.state_dict(),
        strategy=item.strategy if not item.skip_fl else "fedavg",
        alpha=item.alpha,
        tau_exc=item.tau_exc,
        clip_c=clip_c,
    )

    # Attack pipeline
    pipeline = AttackPipeline(scenario=scen, seed=item.master_seed)

    # Execution tracking
    detection_history = {"y_true": [], "y_pred": [], "excluded": []}
    has_detector = (item.strategy != "fedavg" and not item.skip_fl)

    # Anomaly thresholds
    m2_dir_thresh = 0.0
    m4_orig_thresh = 0.50
    m3_stale_thresh = 0.50
    tau_anomaly = item.tau_anomaly

    # Training-time unreliability metrics
    injected_tasks_exposed = 0
    injected_drops = 0
    injected_fails = 0
    injected_delays = 0

    server_rng = np.random.default_rng(40000 + item.master_seed)

    for r in range(rounds):
        # Sample clients
        k = max(1, int(round(frac * len(clients))))
        selected_indices = sorted(server_rng.choice(len(clients), size=k, replace=False).tolist())

        # Train local
        honest_updates = []
        for c in clients:
            if c.client_id not in selected_indices:
                continue
            c.local_train(
                local_eps,
                base_seed=5000 + item.master_seed * 1000 + r * 100 + c.client_id * 10,
                num_tasks=num_tasks,
                warmup_steps=warmup,
                sync_every=sync_every,
            )
            honest_updates.append(c.make_update(round_trained=r))

        # Ingest faults counters from unreliable wrappers
        for c in clients:
            if isinstance(c.env, UnreliableEnvWrapper):
                stats = c.env.get_fault_stats()
                injected_tasks_exposed = stats["tasks_exposed"]
                injected_drops = stats["injected_drops"]
                injected_fails = stats["injected_fails"]
                injected_delays = stats["injected_delays"]

        # If skip_fl (local RL ablation), do not aggregate or distribute
        if item.skip_fl:
            continue

        # Attacks
        attacked_updates = []
        for u in honest_updates:
            att_u, audit = pipeline.transform_update(
                update=u,
                reference_state=server.global_state,
                current_round=r,
                peer_updates=honest_updates,
            )
            attacked_updates.append(att_u)

        # Server aggregation
        beh_trust = {c.client_id: c.trust_summary()["mean_trust"] for c in clients}
        log = server.aggregate_round(attacked_updates, current_round=r, behavioural_trusts=beh_trust)

        for c in clients:
            c.install_global(server.get_global())

        # Decision & detection accounting
        for u in attacked_updates:
            cid = u["client_id"]
            is_mal = scen.client_labels[cid].is_malicious
            rep = next((item for item in log.get("trust_reports", []) if item["client_id"] == cid), None)
            is_excluded = cid in log.get("excluded", [])

            if rep is not None and has_detector:
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

    # Save final model checkpoint
    ckpt_dir = checkpoints_dir / item.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / "global_final.pt"

    if item.skip_fl:
        final_state = clients[0].agent.q_net.state_dict()
    else:
        final_state = server.get_global()
    torch.save(final_state, final_ckpt)

    # Checkpoint SHA256 parameter hash
    fp_bytes = b"".join([v.numpy().tobytes() for k, v in sorted(final_state.items())])
    ckpt_hash = hashlib.sha256(fp_bytes).hexdigest()[:16]

    # Clean held-out evaluation across eval seeds
    eval_rows = []
    config_path = raw_cfg_base["_config_path"]
    for ev_seed in item.eval_seeds:
        row = greedy_rollout(
            config_path=config_path,
            checkpoint=str(final_ckpt),
            seed=ev_seed,
            num_tasks=num_tasks,
            policy_name=f"{item.strategy}_seed{item.master_seed}",
            hidden=hidden,
        )
        validate_task_accounting(row["successes"], row["deadline_misses"], row["failures"], num_tasks)
        eval_rows.append(row)

    # Detection metrics computation
    if has_detector:
        det = compute_detection_metrics(detection_history["y_true"], detection_history["y_pred"])
        precision_val = det["precision"]
        recall_val = det["recall"]
        f1_val = det["f1"]
        fpr_val = det["fpr"]
        tp_val = det["tp"]
        fp_val = det["fp"]
        tn_val = det["tn"]
        fn_val = det["fn"]
    else:
        precision_val = None
        recall_val = None
        f1_val = None
        fpr_val = None
        tp_val = 0
        fp_val = 0
        tn_val = len(detection_history["y_true"])
        fn_val = 0

    total_eval_updates = len(detection_history["excluded"])
    exclusion_rate = float(sum(detection_history["excluded"])) / float(total_eval_updates) if total_eval_updates > 0 else 0.0

    return {
        "run_id": item.run_id,
        "matrix_type": item.matrix_type,
        "scenario_group": item.scenario_group,
        "attack_type": item.attack_type,
        "attack_fraction": item.attack_fraction,
        "attack_strength": item.attack_strength,
        "unreliable_severity": item.unreliable_severity,
        "unreliable_fraction": item.unreliable_fraction,
        "strategy": item.strategy,
        "seed": item.master_seed,
        "is_iid": item.is_iid,
        "ablation_name": item.ablation_name,
        "sweep_param": item.sweep_param,
        "param_value": item.param_value,
        "alpha": item.alpha,
        "tau_exc": item.tau_exc,
        "tau_anomaly": item.tau_anomaly,
        "lambda_smooth": item.lambda_smooth if item.lambda_smooth is not None else 0.70,
        "skip_fl": item.skip_fl,
        "has_detector": has_detector,
        "rounds": rounds,
        "ckpt_hash": ckpt_hash,
        "completion_rate": float(np.mean([r["completion_rate"] for r in eval_rows])),
        "failure_rate": float(np.mean([r["failure_rate"] for r in eval_rows])),
        "miss_rate": float(np.mean([r["miss_rate"] for r in eval_rows])),
        "mean_latency_s": float(np.mean([r["mean_latency_s"] for r in eval_rows])),
        "mean_energy_j": float(np.mean([r["mean_energy_j"] for r in eval_rows])),
        "mean_reward": float(np.mean([r["mean_reward"] for r in eval_rows])),
        "exclusion_rate": exclusion_rate,
        "tp": tp_val,
        "fp": fp_val,
        "tn": tn_val,
        "fn": fn_val,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "fpr": fpr_val,
        "injected_tasks_exposed": injected_tasks_exposed,
        "injected_drops": injected_drops,
        "injected_fails": injected_fails,
        "injected_delays": injected_delays,
    }


def _worker_wrapper(args: tuple[PlanItem, dict, Path]) -> dict[str, Any]:
    item, raw_cfg_base, checkpoints_dir = args
    return execute_plan_item(item, raw_cfg_base, checkpoints_dir)


def run_full_phase10_matrix(
    config_path: str = "configs/feddqn_phase10.yaml",
    outdir: Path = Path("results/phase10_final_v2"),
    num_seeds_core: int = 10,
    num_seeds_secondary: int = 5,
    num_workers: int = 4,
) -> None:
    """Execute the full 880-run matrix in strict stages."""
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = outdir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        raw_cfg_base = yaml.safe_load(f)
    raw_cfg_base["_config_path"] = config_path

    # Generate plan
    eval_seeds = [100, 101, 102, 103, 104]
    plan = build_phase10_experiment_plan(
        eval_seeds=eval_seeds,
        num_seeds_core=num_seeds_core,
        num_seeds_secondary=num_seeds_secondary,
    )

    core_count = sum(1 for p in plan if p.matrix_type == "core")
    abl_count = sum(1 for p in plan if p.matrix_type == "ablation")
    sens_count = sum(1 for p in plan if p.matrix_type == "sensitivity")
    total_count = len(plan)

    print("=================================================================")
    print(f"Phase 10 Remediation: Executing Complete Matrix ({total_count} runs)")
    print(f"  Matrix A (Core):        {core_count} runs")
    print(f"  Matrix B (Ablations):    {abl_count} runs")
    print(f"  Matrix C (Sensitivity): {sens_count} runs")
    print(f"  Parallel workers:       {num_workers}")
    print("=================================================================")

    start_time = time.perf_counter()
    all_results = []

    if num_workers > 1:
        worker_args = [(item, raw_cfg_base, checkpoints_dir) for item in plan]
        with Pool(processes=num_workers) as pool:
            for idx, res in enumerate(pool.imap(_worker_wrapper, worker_args)):
                all_results.append(res)
                if (idx + 1) % 50 == 0 or (idx + 1) == total_count:
                    elapsed = time.perf_counter() - start_time
                    rate = (idx + 1) / elapsed
                    remaining = (total_count - (idx + 1)) / rate
                    print(f"[{idx+1:4d}/{total_count}] ({res['matrix_type']:11s}) {res['run_id']:38s} | comp={res['completion_rate']:.3f} | elapsed={elapsed/60:.1f}m | eta={remaining/60:.1f}m")
    else:
        for idx, item in enumerate(plan):
            t0 = time.perf_counter()
            res = execute_plan_item(item, raw_cfg_base, checkpoints_dir)
            all_results.append(res)
            dt = time.perf_counter() - t0
            if (idx + 1) % 50 == 0 or (idx + 1) == total_count:
                elapsed = time.perf_counter() - start_time
                rate = (idx + 1) / elapsed
                remaining = (total_count - (idx + 1)) / rate
                print(f"[{idx+1:4d}/{total_count}] ({item.matrix_type:11s}) {item.run_id:38s} | comp={res['completion_rate']:.3f} | {dt:.2f}s | elapsed={elapsed/60:.1f}m | eta={remaining/60:.1f}m")

    # Match paired controls to compute degradation
    # Non-IID attacks match against honest_non_iid for same strat and seed
    # IID clean matches against clean_iid
    df_all = pd.DataFrame(all_results)

    # Compute degradation
    non_iid_controls = df_all[df_all["scenario_group"] == "honest_non_iid"].set_index(["strategy", "seed"])
    iid_controls = df_all[df_all["scenario_group"] == "clean_iid"].set_index(["strategy", "seed"])

    deg_comp = []
    deg_rew = []
    for _, row in df_all.iterrows():
        strat = row["strategy"]
        seed = row["seed"]
        comp = row["completion_rate"]
        rew = row["mean_reward"]

        if row["is_iid"]:
            ctrl = iid_controls.loc[(strat, seed)] if (strat, seed) in iid_controls.index else None
        else:
            ctrl = non_iid_controls.loc[(strat, seed)] if (strat, seed) in non_iid_controls.index else None

        if ctrl is not None:
            c_deg = compute_degradation(ctrl["completion_rate"], comp, higher_is_better=True)
            r_deg = compute_degradation(ctrl["mean_reward"], rew, higher_is_better=True)
        else:
            c_deg = 0.0
            r_deg = 0.0
        deg_comp.append(c_deg)
        deg_rew.append(r_deg)

    df_all["degradation_completion"] = deg_comp
    df_all["degradation_reward"] = deg_rew

    # Split into dedicated datasets
    df_core = df_all[df_all["matrix_type"] == "core"]
    df_abl = df_all[df_all["matrix_type"] == "ablation"]
    df_sens = df_all[df_all["matrix_type"] == "sensitivity"]

    # Save CSVs
    core_csv = outdir / "phase10_experiment_runs.csv"
    abl_csv = outdir / "phase10_ablation_runs.csv"
    sens_csv = outdir / "phase10_sensitivity_runs.csv"

    df_core.to_csv(core_csv, index=False)
    df_abl.to_csv(abl_csv, index=False)
    df_sens.to_csv(sens_csv, index=False)

    print(f"\n=================================================================")
    print(f"Matrix Execution Complete! Total elapsed: {(time.perf_counter()-start_time)/60:.2f} minutes")
    print(f"  Saved Core runs:        {len(df_core)} rows to {core_csv}")
    print(f"  Saved Ablation runs:    {len(df_abl)} rows to {abl_csv}")
    print(f"  Saved Sensitivity runs: {len(df_sens)} rows to {sens_csv}")
    print("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feddqn_phase10.yaml")
    parser.add_argument("--outdir", default="results/phase10_final_v2")
    parser.add_argument("--seeds-core", type=int, default=10)
    parser.add_argument("--seeds-sec", type=int, default=5)
    args = parser.parse_args()

    run_full_phase10_matrix(
        config_path=args.config,
        outdir=Path(args.outdir),
        num_seeds_core=args.seeds_core,
        num_seeds_secondary=args.seeds_sec,
    )
