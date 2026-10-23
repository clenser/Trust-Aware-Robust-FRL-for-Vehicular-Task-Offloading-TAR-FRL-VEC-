"""
Publication experimentation suite for trust-aware federated reinforcement
learning in VANETs.

Experimental protocol
---------------------
* **Training** and **evaluation** are separate phases with separate
  environments, separate seeds and separate task workloads.
* **Evaluation** uses an explicit :class:`EvaluationConfig` (a number of
  independent episodes, a step budget per episode and a fixed seed
  derivation).  It is *not* derived from the training step count any more.
* **Evaluation is inert**: no exploration, no replay-buffer writes, no
  gradient steps and no parameter changes.  The learned parameters are hashed
  before and after the evaluation phase and the hash must not change.
* **Paired evaluation**: for a given seed every method is evaluated on the
  *same* evaluation scenarios and the same task workloads, so the comparison
  is paired and the per-seed differences are meaningful.
* **Statistics**: mean, sample standard deviation, sample count and an exact
  Student's *t* 95 % confidence interval are reported for every metric, and
  the method comparison reports the paired mean difference, its confidence
  interval, a paired *t* test and Cohen's *d* for the paired samples.
* **Missing observations are reported as ``null``** (N/A), never as zero.
* **Detection metrics** are aggregated from the underlying confusion counts,
  not from an average of per-round rates.

Provenance
----------
Every artefact records the fully resolved configuration, the dataset paths and
SHA-256 checksums that the run actually loaded, the task-profile checksum,
the code revision, the seeds, the attack settings, the per-seed raw records
and the statistical summaries.
"""

import sys
import os
import json
import time
import copy
import random
import hashlib
import platform
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from configs.config_loader import load_config
from trust.trust_engine import TrustEngine
from environment.vanet_env import VANETOffloadingEnv
from federated.client import FederatedClient
from federated.server import FederatedServer
from baselines.heuristics import ResourceAwareBaseline
from rl.agent import DQNAgent

# The minimum number of transitions a client must hold before a gradient step
# is possible at all.  A client below this threshold cannot produce an update,
# so it is a non-participant of the round rather than a candidate updater.
MIN_TRAIN_BATCH = int(DQNAgent.MIN_TRAIN_BATCH)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def set_all_seeds(seed):
    """Deterministic seeding across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_confidence_interval_95(samples):
    """Exact 95 % CI half-width using Student's t."""
    values = [v for v in samples if v is not None]
    n = len(values)
    if n <= 1:
        return 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    std_err = float(np.std(values, ddof=1) / np.sqrt(n))
    return float(t_crit * std_err)


def summarize_metric(samples):
    """Mean / std / n / 95 % CI for a metric, ignoring missing observations.

    Returns ``None`` for the mean when there is no observation at all, so a
    missing measurement is never reported as zero.
    """
    values = [v for v in samples if v is not None]
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0, "ci95": None,
                "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half = compute_confidence_interval_95(values)
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "n": n,
        "ci95": round(half, 6),
        "ci95_low": round(mean - half, 6),
        "ci95_high": round(mean + half, 6),
    }


def paired_comparison(a_samples, b_samples):
    """Paired difference between two methods, evaluated seed by seed.

    ``a_samples`` and ``b_samples`` must be aligned by seed; entries that are
    ``None`` in either list drop that seed from the comparison.

    Sign convention: the reported difference is always ``a - b``.  In
    :func:`compare_methods` the *compared* method is ``a`` and the reference
    method (Proposed_TFRL) is ``b``, so a positive latency difference means
    the compared method was slower than the reference.
    """
    a_aligned, b_aligned = [], []
    for a, b in zip(a_samples, b_samples):
        if a is None or b is None:
            continue
        a_aligned.append(a)
        b_aligned.append(b)
    n = len(a_aligned)
    if n == 0:
        return {"n": 0, "mean_difference": None, "ci95": None,
                "t_statistic": None, "p_value": None, "cohens_d": None}
    diffs = np.asarray(a_aligned, dtype=float) - np.asarray(b_aligned, dtype=float)
    mean_diff = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    half = compute_confidence_interval_95(diffs.tolist())
    if n > 1 and sd > 0:
        t_stat, p_value = stats.ttest_rel(
            np.asarray(a_aligned, dtype=float),
            np.asarray(b_aligned, dtype=float))
        t_stat = float(t_stat)
        p_value = float(p_value)
        # Cohen's d for paired samples (dz).
        cohen_d = float(mean_diff / sd) if sd > 0 else None
    else:
        t_stat, p_value, cohen_d = None, None, None
    return {
        "n": n,
        "mean_difference": round(mean_diff, 6),
        "std_difference": round(sd, 6),
        "ci95": round(half, 6),
        "ci95_low": round(mean_diff - half, 6),
        "ci95_high": round(mean_diff + half, 6),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohen_d,
        "significant_at_0_05": (p_value is not None and p_value < 0.05),
    }


def get_file_sha256(filepath):
    """SHA-256 of a file, or ``None`` when it does not exist."""
    if not filepath:
        return None
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_dict_bytes(state):
    """Wire size of a model update (fp32 payload of every tensor)."""
    return int(sum(int(t.numel()) * 4 for t in state.values()))


def _hash_state_dict(state):
    """Order-independent hash of a parameter set (for inertness checks)."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode())
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    """Explicit, independent evaluation protocol."""
    episodes: int = 3
    steps_per_episode: int = 20
    seed_offset: int = 500_000

    def episode_seeds(self, base_seed: int) -> List[int]:
        """Deterministic, independent seeds for the evaluation episodes."""
        return [base_seed + self.seed_offset + i for i in range(self.episodes)]

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------
# Trial execution
# ----------------------------------------------------------------------
def _resolve_datasets(config, mobility_mode):
    """The dataset files a mobility mode actually loads, with checksums."""
    cfg_mob = config.get("mobility", {})
    if mobility_mode == "highway":
        return {
            "mobility_mode": "highway",
            "mobility_source": "procedural IDM highway (no trajectory file)",
            "trace_path": None,
            "trace_sha256": None,
            "authentic": False,
        }
    if mobility_mode == "authentic_trace":
        path = cfg_mob.get("authentic_trace_dataset",
                           "data/traces/ngsim_us101_authentic.csv")
    else:
        path = cfg_mob.get("trace_dataset",
                           "data/synthetic_traces/synthetic_ngsim_highway.csv")
    absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    return {
        "mobility_mode": mobility_mode,
        "mobility_source": ("authentic FHWA NGSIM US-101 processed trace"
                             if mobility_mode == "authentic_trace"
                             else "calibrated synthetic NGSIM-style trace"),
        "trace_path": path,
        "trace_sha256": get_file_sha256(absolute),
        "trace_bytes": os.path.getsize(absolute) if os.path.exists(absolute) else None,
        "authentic": mobility_mode == "authentic_trace",
    }


def _resolve_task_profiles(config):
    cfg_task = config.get("task_profiles", {})
    path = cfg_task.get("catalog_csv",
                        "data/synthetic_traces/synthetic_v2x_task_profiles.csv")
    absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    return {"task_profile_path": path, "task_profile_sha256": get_file_sha256(absolute)}


def _make_root_trainer(config, num_vehicles, mobility_mode, seed, state_dim, agent_kwargs,
                       local_epochs, steps_per_round):
    """Trusted root-data trainer for the FLTrust-style baseline.

    The server owns a separate, attacker-free probe environment (same configuration,
    different seed) driven by a simple heuristic policy.  Every round it collects
    ``steps_per_round`` steps of transitions, trains a copy of the current global model
    on them for ``local_epochs`` gradient steps and returns the result; the difference to
    the global model is the *root update*.  This is the trusted root dataset FLTrust
    requires; the vehicles' own data are never used.
    """
    root_env = VANETOffloadingEnv(num_vehicles=num_vehicles, attack_ratio=0.0, mobility_mode=mobility_mode,
                                  config=config, random_seed=int(seed) + 777)
    agent = DQNAgent(state_dim=state_dim, action_dim=5, **agent_kwargs)
    probe = DQNAgent(state_dim=state_dim, action_dim=5, **agent_kwargs)     # only used to evaluate models
    states = root_env._get_global_state_dict()

    def loss(weights, n=512):
        """Mean squared error between Q(s, a) and the observed reward on the trusted root data
        (a bandit-style value error; lower = the model explains the trusted outcomes better)."""
        if len(agent.memory) < 8:
            return None
        import torch
        st, ac, rw, _, _ = agent.memory.sample(min(n, len(agent.memory)))
        probe.set_weights(weights)
        with torch.no_grad():
            q = probe.policy_net(st.to(probe.device)).gather(1, ac.unsqueeze(1).to(probe.device)).squeeze(1)
            return float(((q - rw.to(probe.device)) ** 2).mean().item())

    def trainer(global_weights):
        for _ in range(steps_per_round):
            root_env.update_positions()
            for vid in root_env.vehicle_order():
                v = root_env.vehicles[vid]
                state, task = states[vid]
                action = _greedy_delay_action(root_env, v, task)
                ns, r, done, info, nt = root_env.step_vehicle(vid, action, task)
                if not info.get("present", True):
                    continue
                agent.memory.push(state, action, r, ns, done)
                states[vid] = (ns, nt)
        agent.set_weights(global_weights)
        for _ in range(local_epochs):
            agent.update()
        return agent.get_weights()

    trainer.loss = loss
    return trainer


def run_simulation_trial(num_vehicles=None,
                          attack_ratio=None,
                          num_rounds=None,
                          steps_per_round=None,
                          eval_config: Optional[EvaluationConfig] = None,
                          aggregation_strategy=None,
                          alpha=None,
                          exclusion_tau=None,
                          poison_strength=None,
                          mobility_mode=None,
                          is_federated=True,
                          seed=None,
                          config=None,
                          poison_type=None,
                          attack_probability=None,
                          shadow=False):
    """Run one controlled trial: training phase, then an inert evaluation phase.

    Returns a record with the raw per-episode evaluation observations, the
    training diagnostics and the fully resolved configuration.
    """
    if config is None:
        config = load_config()

    cfg_sys = config.get("system", {})
    cfg_mob = config.get("mobility", {})
    cfg_adv = config.get("adversarial", {})
    cfg_fl = config.get("federated_learning", {})
    cfg_trust = config.get("trust_management", {})

    num_vehicles = num_vehicles if num_vehicles is not None else cfg_mob.get("num_vehicles", 25)
    attack_ratio = attack_ratio if attack_ratio is not None else cfg_adv.get("attack_ratio", 0.24)
    num_rounds = num_rounds if num_rounds is not None else cfg_fl.get("num_rounds", 10)
    steps_per_round = steps_per_round if steps_per_round is not None else cfg_fl.get("steps_per_round", 10)
    aggregation_strategy = aggregation_strategy if aggregation_strategy is not None else cfg_fl.get("aggregation_strategy", "trust_aware")
    alpha = alpha if alpha is not None else cfg_trust.get("alpha", 0.50)
    exclusion_tau = exclusion_tau if exclusion_tau is not None else cfg_trust.get("exclusion_threshold_tau", 0.40)
    poison_strength = poison_strength if poison_strength is not None else cfg_adv.get("poison_strength", 1.5)
    mobility_mode = mobility_mode if mobility_mode is not None else cfg_mob.get("scenario", "highway")
    seed = seed if seed is not None else cfg_sys.get("random_seed", 42)
    poison_type = poison_type if poison_type is not None else cfg_adv.get("poison_type", "sign_flip")
    attack_probability = float(attack_probability if attack_probability is not None
                               else cfg_adv.get("attack_probability", 1.0))
    if eval_config is None:
        eval_config = EvaluationConfig(
            episodes=cfg_fl.get("eval_episodes", 3),
            steps_per_episode=cfg_fl.get("eval_steps", 20),
            seed_offset=cfg_fl.get("eval_seed_offset", 500_000),
        )

    datasets = _resolve_datasets(config, mobility_mode)
    task_profiles = _resolve_task_profiles(config)
    agent_kwargs = dict(lr=float(cfg_fl.get("learning_rate", 1e-3)),
                        gamma=float(cfg_fl.get("gamma_discount", 0.95)),
                        batch_size=int(cfg_fl.get("batch_size", 32)))

    # ---------------- Phase 1: training ----------------
    set_all_seeds(seed)

    trust_engine = TrustEngine(alpha=alpha, exclusion_threshold=exclusion_tau, config=config)
    env_train = VANETOffloadingEnv(
        num_vehicles=num_vehicles,
        attack_ratio=attack_ratio,
        mobility_mode=mobility_mode,
        trust_engine=trust_engine,
        alpha=alpha,
        exclusion_tau=exclusion_tau,
        config=config,
        random_seed=seed,
    )
    server = FederatedServer(
        state_dim=env_train.state_dim,
        action_dim=5,
        aggregation_strategy=aggregation_strategy,
        alpha=alpha,
        exclusion_threshold=exclusion_tau,
        trust_engine=trust_engine,
        config=config,
    )
    server.shadow_detectors = bool(shadow)
    if aggregation_strategy in ("fltrust", "impact_validated") or os.environ.get("VANNET_ROOT") == "1":
        server.root_trainer = _make_root_trainer(
            config, num_vehicles, mobility_mode, seed, env_train.state_dim, agent_kwargs,
            int(cfg_fl.get("local_epochs", 16)), int(steps_per_round))
    server.staleness_mode = str(cfg_fl.get("staleness_mode", "none"))
    missed_broadcast = bool(cfg_fl.get("missed_broadcast", False))
    connectivity_dropout = float(cfg_fl.get("connectivity_dropout", 0.0))
    shadow_confusion = {}

    clients = {}
    init_gw = server.get_global_weights()
    for vid, v in env_train.vehicles.items():
        v.poison_strength = poison_strength
        client = FederatedClient(
            vehicle_id=vid, state_dim=env_train.state_dim, action_dim=5, role=v.role,
            poison_strength=poison_strength,
            poison_type=poison_type,
            attack_probability=attack_probability,
            agent_kwargs=agent_kwargs,
        )
        if is_federated:
            client.update_from_global(init_gw)
        clients[vid] = client

    comm_up_bytes = 0
    comm_down_bytes = 0
    detection_rates = []
    fpr_rates = []
    fnr_rates = []
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    round_records = []

    states_dict = env_train._get_global_state_dict()
    for r in range(num_rounds):
        for s in range(steps_per_round):
            env_train.update_positions()
            for vid in env_train.vehicle_order():
                client = clients[vid]
                state, task = states_dict[vid]
                action = client.agent.select_action(state, explore=(r < num_rounds * 0.4))
                next_state, reward, done, info, next_task = env_train.step_vehicle(vid, action, task)
                if not info.get("present", True):
                    continue
                client.agent.memory.push(state, action, reward, next_state, done)
                states_dict[vid] = (next_state, next_task)

        client_models = {}
        client_samples = {}
        ground_truth_roles = {vid: c.role for vid, c in clients.items()}
        global_w = server.get_global_weights()

        # Federated participation: a fraction of the clients trains and uploads
        # in each round.  The default is 1.0 (every client participates), but
        # the fraction is configurable and recorded in the run metadata.
        local_epochs = cfg_fl.get("local_epochs", 16)
        participation = float(cfg_fl.get("participation_fraction", 1.0))
        exclude_non_participants = bool(
            cfg_trust.get("exclude_non_participants", True))
        non_participants: List[int] = []
        participants = list(clients.keys())
        if participation < 1.0:
            n_part = max(int(cfg_fl.get("min_clients", 2)),
                         int(round(len(clients) * participation)))
            participants = sorted(np.random.RandomState(seed * 7919 + r).choice(
                len(clients), size=n_part, replace=False).tolist())
            participants = [list(clients.keys())[i] for i in participants]

        # Optional connectivity gating: a vehicle can only upload when it is
        # present and inside the coverage of at least one RSU at aggregation
        # time.  This models intermittent connectivity explicitly.
        offline: List[int] = []
        if connectivity_dropout > 0.0:
            # Intermittent connectivity independent of coverage (cellular
            # interruption, tunnel, handover failure): each client is offline
            # in a round with this probability (seeded, reproducible).
            drop_rng = np.random.RandomState(seed * 104729 + r * 31 + 13)
            gone = [vid for vid in participants if drop_rng.rand() < connectivity_dropout]
            offline.extend(int(v_) for v_ in gone)
            participants = [vid for vid in participants if vid not in set(gone)]
        if bool(cfg_fl.get("require_connectivity", False)):
            connected = []
            for vid in participants:
                v = env_train.vehicles[vid]
                if getattr(v, "active", True) and any(
                        rr.get_distance_to(v.x, v.y) <= rr.coverage_radius_m
                        for rr in env_train.rsus):
                    connected.append(vid)
                else:
                    offline.append(int(vid))
            participants = connected

        grad_steps_round = 0
        losses_round = []
        for vid in participants:
            client = clients[vid]
            steps_before = client.agent.training_steps
            mean_loss = client.local_train(epochs=local_epochs)
            did = client.agent.training_steps - steps_before
            grad_steps_round += did
            if did > 0:
                losses_round.append(mean_loss)
            # Participation rule (identical to the live console, see
            # ``web/app.py``): a client that holds fewer transitions than the
            # minimum training batch could not perform a gradient step, so
            # its "upload" is the global model itself.  Such a client is a
            # **non-participant** of this round, not a malicious updater -
            # there is nothing about it to detect, and its zero update would
            # only dilute the aggregate.  The decision uses the client's
            # available experience, which is a role-independent participation
            # statistic; the ground-truth role is never consulted.
            if (exclude_non_participants
                    and client.available_samples < MIN_TRAIN_BATCH):
                non_participants.append(int(vid))
                continue
            client_models[vid] = client.get_upload_weights(global_w)
            client_samples[vid] = max(1, client.sample_count)

        if is_federated:
            model_bytes = _state_dict_bytes(global_w)
            n_uploads = len(client_models)
            stale = ({vid: server.current_round - clients[vid].model_version for vid in client_models}
                     if missed_broadcast else None)
            if n_uploads >= int(cfg_fl.get("min_clients", 2)):
                new_global, metrics = server.run_aggregation_round(
                    client_models, client_samples, ground_truth_roles=ground_truth_roles,
                    staleness=stale)
                attempted = True
            else:
                # Too few valid updates: no aggregation is attempted and the
                # global model stays as it was.
                new_global, metrics = None, {
                    "round": server.current_round, "total_clients": n_uploads,
                    "excluded_count": 0, "excluded_clients": [],
                    "accepted_clients": [],
                    "aggregation_produced_model": False,
                    "aggregation_note": "fewer than min_clients valid updates"}
                attempted = False
            for metric_key, count_key in (
                    ("true_positives", "tp"),
                    ("false_positives", "fp"),
                    ("true_negatives", "tn"),
                    ("false_negatives", "fn")):
                if metrics.get(metric_key) is not None:
                    confusion[count_key] += int(metrics[metric_key])
            if metrics.get("attack_detection_rate") is not None:
                detection_rates.append(metrics["attack_detection_rate"])
            if metrics.get("false_positive_rate") is not None:
                fpr_rates.append(metrics["false_positive_rate"])
            if metrics.get("false_negative_rate") is not None:
                fnr_rates.append(metrics["false_negative_rate"])
            produced = bool(metrics.get("aggregation_produced_model"))
            changed = bool(metrics.get("model_changed"))
            accepted_ids = metrics.get("accepted_clients")
            n_accepted = (len(accepted_ids) if accepted_ids
                          else (n_uploads if produced else 0))
            if not attempted:
                status = "not_attempted_insufficient_updates"
            elif not produced:
                status = "no_model_produced"
            elif not changed:
                status = "model_unchanged"
            else:
                status = "model_updated"
            comm_up_bytes += model_bytes * n_uploads
            comm_down_bytes += model_bytes * len(clients) if produced else 0
            round_records.append({
                "round": metrics.get("round"),
                "status": status,
                "attempted": attempted,
                "uploaded_updates": n_uploads,
                "accepted_updates": n_accepted,
                "rejected_updates": max(0, n_uploads - n_accepted) if produced else n_uploads,
                "eligible_clients": metrics.get("total_clients"),
                "excluded_count": metrics.get("excluded_count"),
                "true_positives": metrics.get("true_positives"),
                "false_positives": metrics.get("false_positives"),
                "true_negatives": metrics.get("true_negatives"),
                "false_negatives": metrics.get("false_negatives"),
                "non_participants": len(non_participants),
                "offline_clients": len(offline),
                "mean_staleness": metrics.get("mean_staleness"),
                "unreliable_total": metrics.get("unreliable_total"),
                "unreliable_excluded": metrics.get("unreliable_excluded"),
                "aggregation_time_ms": metrics.get("aggregation_time_ms"),
                "model_bytes": model_bytes,
                "gradient_steps": int(grad_steps_round),
                "mean_train_loss": (float(np.mean(losses_round)) if losses_round else None),
            })
            for name, c in (metrics.get("shadow_detectors") or {}).items():
                tot = shadow_confusion.setdefault(name, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
                for k, v_ in c.items():
                    tot[k] += int(v_)
            if new_global is not None:
                offline_set = set(offline) if missed_broadcast else set()
                for vid_, client in clients.items():
                    if vid_ in offline_set:
                        continue          # missed the broadcast: keeps its old model
                    client.update_from_global(new_global, version=server.current_round)

    # ---------------- Phase 2: inert evaluation ----------------
    # The learned parameters must not change during the evaluation.
    params_before = {vid: _hash_state_dict(c.agent.get_weights())
                     for vid, c in clients.items()}

    eval_records = []
    for episode_seed in eval_config.episode_seeds(seed):
        set_all_seeds(episode_seed)
        env_eval = VANETOffloadingEnv(
            num_vehicles=num_vehicles,
            attack_ratio=attack_ratio,
            mobility_mode=mobility_mode,
            config=config,
            random_seed=episode_seed,
        )
        eval_states = env_eval._get_global_state_dict()
        latencies, successes, energies, rewards = [], [], [], []
        completed_lat, fail_reasons = [], {}
        action_counts = np.zeros(5, dtype=int)
        for _ in range(eval_config.steps_per_episode):
            env_eval.update_positions()
            for vid in env_eval.vehicle_order():
                state, task = eval_states[vid]
                if vid in clients:
                    # Greedy, no exploration, no buffer write, no gradient.
                    with torch.no_grad():
                        q = clients[vid].agent.policy_net(
                            torch.FloatTensor(state).unsqueeze(0))
                    action = int(q.argmax(dim=-1).item())
                else:
                    action = 0
                action_counts[action] += 1
                next_state, r_val, done, info, next_task = env_eval.step_vehicle(vid, action, task)
                if not info.get("present", True):
                    continue
                latencies.append(info["delay_ms"])
                successes.append(1 if info["success"] else 0)
                if info["success"]:
                    completed_lat.append(info["delay_ms"])
                else:
                    fr = info.get("failure_reason") or "unknown"
                    fail_reasons[fr] = fail_reasons.get(fr, 0) + 1
                rewards.append(r_val)
                energies.append(info.get("energy_j", 0.0))
                eval_states[vid] = (next_state, next_task)
        eval_records.append({
            "episode_seed": episode_seed,
            "n_observations": len(latencies),
            "latencies": latencies,
            "completed_latencies": completed_lat,
            "failure_reasons": fail_reasons,
            "successes": successes,
            "energies": energies,
            "rewards": rewards,
            "action_histogram": action_counts.tolist(),
        })

    params_after = {vid: _hash_state_dict(c.agent.get_weights())
                    for vid, c in clients.items()}
    evaluation_was_inert = (params_before == params_after)

    all_completed = [x for r in eval_records for x in r["completed_latencies"]]
    all_latencies = [x for r in eval_records for x in r["latencies"]]
    all_successes = [x for r in eval_records for x in r["successes"]]
    all_energies = [x for r in eval_records for x in r["energies"]]
    all_rewards = [x for r in eval_records for x in r["rewards"]]

    n_obs = len(all_latencies)
    mean_latency = float(np.mean(all_latencies)) if all_latencies else None
    mean_completed_latency = float(np.mean(all_completed)) if all_completed else None
    completion = (float(np.mean(all_successes)) * 100.0) if all_successes else None
    mean_energy = float(np.mean(all_energies)) if all_energies else None
    mean_reward = float(np.mean(all_rewards)) if all_rewards else None

    # Detection metrics pooled from the confusion counts.
    pooled = _pooled_from_counts(confusion)
    # Measured federated traffic: fp32 model payloads actually uploaded by the
    # clients that participated plus the broadcasts of the aggregated model.
    comm_overhead_kb = (comm_up_bytes + comm_down_bytes) / 1024.0 if is_federated else 0.0

    effective_config = {
        "num_vehicles": num_vehicles,
        "attack_ratio": attack_ratio,
        "realised_malicious_clients": int(round(num_vehicles * attack_ratio)),
        "num_rounds": num_rounds,
        "steps_per_round": steps_per_round,
        "local_epochs": cfg_fl.get("local_epochs", 16),
        "local_gradient_steps_per_round": cfg_fl.get("local_epochs", 16),
        "participation_fraction": float(cfg_fl.get("participation_fraction", 1.0)),
        "total_gradient_steps_per_client": int(cfg_fl.get("local_epochs", 16))
            * int(num_rounds)
            * float(cfg_fl.get("participation_fraction", 1.0)),
        "aggregation_strategy": aggregation_strategy,
        "alpha": alpha,
        "exclusion_tau": exclusion_tau,
        "poison_strength": poison_strength,
        "poison_type": poison_type,
        "attack_probability": attack_probability,
        "mobility_mode": mobility_mode,
        "is_federated": is_federated,
        "seed": seed,
        "participation_fraction_effective": float(cfg_fl.get("participation_fraction", 1.0)),
        "require_connectivity": bool(cfg_fl.get("require_connectivity", False)),
        "connectivity_dropout": connectivity_dropout,
        "missed_broadcast": missed_broadcast,
        "staleness_mode": str(cfg_fl.get("staleness_mode", "none")),
        "faulty_rsus": list(cfg_adv.get("faulty_rsus", []) or []),
        "environment_model": dict(config.get("environment_model", {})),
        "reward": dict(config.get("reward", {})),
        "task_heterogeneity": {"non_iid": config.get("task_profiles", {}).get("non_iid", True),
                               "dirichlet_alpha": config.get("task_profiles", {}).get("dirichlet_alpha", 0.4)},
        "unreliable_helper_ratio": cfg_adv.get("unreliable_helper_ratio", 0.0),
        "evaluation": eval_config.to_dict(),
        "learning_rate": cfg_fl.get("learning_rate", 0.001),
        "batch_size": cfg_fl.get("batch_size", 32),
        "gamma_discount": cfg_fl.get("gamma_discount", 0.95),
    }

    return {
        "seed": seed,
        "avg_latency_ms": None if mean_latency is None else round(mean_latency, 3),
        "avg_completed_latency_ms": None if mean_completed_latency is None else round(mean_completed_latency, 3),
        "completion_rate": None if completion is None else round(completion, 4),
        "failure_rate": None if completion is None else round(100.0 - completion, 4),
        "avg_reward": None if mean_reward is None else round(mean_reward, 4),
        "avg_energy_j": None if mean_energy is None else round(mean_energy, 6),
        "n_observations": n_obs,
        "attack_detection_rate": pooled["tpr"],
        "false_positive_rate": pooled["fpr"],
        "false_negative_rate": pooled["fnr"],
        "detection_precision": pooled["precision"],
        "confusion_counts": confusion,
        "shadow_confusion": shadow_confusion,
        "comm_overhead_kb": round(comm_overhead_kb, 1),
        "evaluation_was_inert": evaluation_was_inert,
        "evaluation_records": eval_records,
        "training_rounds": round_records,
        "effective_config": effective_config,
        "datasets": datasets,
        "task_profiles": task_profiles,
    }


def _pooled_from_counts(c):
    """Pooled detection rates from the underlying confusion counts."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    return {
        "tpr": (tp / (tp + fn)) if (tp + fn) > 0 else None,
        "fpr": (fp / (fp + tn)) if (fp + tn) > 0 else None,
        "fnr": (fn / (tp + fn)) if (tp + fn) > 0 else None,
        "precision": (tp / (tp + fp)) if (tp + fp) > 0 else None,
        "recall": (tp / (tp + fn)) if (tp + fn) > 0 else None,
        "accuracy": ((tp + tn) / (tp + fp + fn + tn)) if (tp + fp + fn + tn) > 0 else None,
    }


# ----------------------------------------------------------------------
# Heuristic baselines - identical evaluation protocol
# ----------------------------------------------------------------------
def _greedy_delay_action(env, v, task):
    """Greedy minimum-estimated-delay heuristic (local or a reachable RSU).

    Uses only information a vehicle could observe: its own CPU and queue, the
    RSUs' advertised capacity and queue length, and a nominal 20 Mbps uplink.
    It ignores trust, V2V peers and future contention.
    """
    best_action, best_est = 0, (task.cpu_cycles_giga / max(1e-6, v.cpu_capacity_ghz)
                                 + v.queue_length * task.cpu_cycles_giga / max(1e-6, v.cpu_capacity_ghz))
    for i, r in enumerate(env.rsus[:3]):
        if r.get_distance_to(v.x, v.y) > r.coverage_radius_m:
            continue
        per_core = r.cpu_capacity_ghz / max(1, r.num_cores)
        service = task.cpu_cycles_giga / per_core
        wait = (r.queue_length / max(1, r.num_cores)) * service
        est = task.data_size_mb * 8.0 / 20.0 + wait + service
        if est < best_est:
            best_action, best_est = i + 1, est
    return best_action


def _trust_greedy_action(env, v, task, min_trust=0.5):
    """Greedy minimum-estimated-delay heuristic that skips RSUs whose behavioural
    trust is below ``min_trust`` (a fair non-learning user of behavioural trust)."""
    best_action, best_est = 0, (task.cpu_cycles_giga / max(1e-6, v.cpu_capacity_ghz)
                                 + v.queue_length * task.cpu_cycles_giga / max(1e-6, v.cpu_capacity_ghz))
    for i, r in enumerate(env.rsus[:3]):
        if r.get_distance_to(v.x, v.y) > r.coverage_radius_m:
            continue
        if env.trust_engine.behavior_tracker.get_trust(r.rsu_id) < min_trust:
            continue
        per_core = r.cpu_capacity_ghz / max(1, r.num_cores)
        service = task.cpu_cycles_giga / per_core
        wait = (r.queue_length / max(1, r.num_cores)) * service
        est = task.data_size_mb * 8.0 / 20.0 + wait + service
        if est < best_est:
            best_action, best_est = i + 1, est
    return best_action


def _v2v_aware_action(env, v, task, use_trust, min_trust=0.5):
    """Greedy minimum-estimated-delay heuristic that also considers the V2V peer.

    ``use_trust`` False  -> trust-blind: RSUs and the peer are judged by estimated
                            delay only (the peer is whichever the environment's
                            selection rule returns).
    ``use_trust`` True   -> candidates with behavioural trust below ``min_trust``
                            are skipped (RSUs and the peer alike).
    The peer delay estimate uses a nominal 50 Mbps V2V link (measured in this channel model: ~54 Mbps) and the peer's
    advertised CPU and queue; it cannot see that a peer sabotages tasks.
    """
    if use_trust:
        action = _trust_greedy_action(env, v, task, min_trust)
    else:
        action = _greedy_delay_action(env, v, task)
    cur = {0: task.cpu_cycles_giga / max(1e-6, v.cpu_capacity_ghz) * (1 + v.queue_length)}
    if action in (1, 2, 3):
        r = env.rsus[action - 1]
        per_core = r.cpu_capacity_ghz / max(1, r.num_cores)
        service = task.cpu_cycles_giga / per_core
        cur_est = task.data_size_mb * 8.0 / 20.0 + (r.queue_length / max(1, r.num_cores)) * service + service
    else:
        cur_est = cur[0]
    peer, _ = env.select_v2v_peer(v)
    if peer is None:
        return action
    if use_trust and env.trust_engine.behavior_tracker.get_trust(f"V{peer.vehicle_id}") < min_trust:
        return action
    est = task.data_size_mb * 8.0 / 50.0 + (peer.queue_length + 1) * task.cpu_cycles_giga / max(1e-6, peer.cpu_capacity_ghz)
    return 4 if est < cur_est else action


def run_heuristic_baseline(baseline_name, num_vehicles=None, attack_ratio=None,
                           mobility_mode=None, seed=None, config=None,
                           eval_config: Optional[EvaluationConfig] = None):
    """Evaluate a heuristic baseline under exactly the same protocol.

    The baseline receives the *same* mobility mode, the same evaluation
    episodes, the same seeds and the same task workloads as the learned
    methods, so the comparison is paired.
    """
    if config is None:
        config = load_config()
    cfg_sys = config.get("system", {})
    cfg_mob = config.get("mobility", {})
    cfg_adv = config.get("adversarial", {})
    cfg_fl = config.get("federated_learning", {})

    num_vehicles = num_vehicles if num_vehicles is not None else cfg_mob.get("num_vehicles", 25)
    attack_ratio = attack_ratio if attack_ratio is not None else cfg_adv.get("attack_ratio", 0.24)
    mobility_mode = mobility_mode if mobility_mode is not None else cfg_mob.get("scenario", "highway")
    seed = seed if seed is not None else cfg_sys.get("random_seed", 42)
    if eval_config is None:
        eval_config = EvaluationConfig(
            episodes=cfg_fl.get("eval_episodes", 3),
            steps_per_episode=cfg_fl.get("eval_steps", 20),
            seed_offset=cfg_fl.get("eval_seed_offset", 500_000),
        )

    datasets = _resolve_datasets(config, mobility_mode)
    task_profiles = _resolve_task_profiles(config)

    eval_records = []
    for episode_seed in eval_config.episode_seeds(seed):
        set_all_seeds(episode_seed)
        env_eval = VANETOffloadingEnv(
            num_vehicles=num_vehicles,
            attack_ratio=attack_ratio,
            mobility_mode=mobility_mode,
            config=config,
            random_seed=episode_seed,
        )
        states_dict = env_eval._get_global_state_dict()
        latencies, successes, energies, rewards = [], [], [], []
        completed_lat, fail_reasons = [], {}
        for _ in range(eval_config.steps_per_episode):
            env_eval.update_positions()
            for vid in env_eval.vehicle_order():
                v = env_eval.vehicles[vid]
                state, task = states_dict[vid]
                if baseline_name == "Local_Only":
                    action = 0
                elif baseline_name == "Nearest_RSU":
                    # Nearest RSU among those whose coverage contains the vehicle.
                    dists = [r.get_distance_to(v.x, v.y) for r in env_eval.rsus]
                    reachable = [i for i, r in enumerate(env_eval.rsus)
                                 if dists[i] <= r.coverage_radius_m]
                    pool = reachable if reachable else range(len(dists))
                    action = int(min(pool, key=lambda i: dists[i]) + 1)
                elif baseline_name == "Random_Valid":
                    action = int(np.random.randint(0, 5))
                elif baseline_name == "Resource_Aware":
                    action = _greedy_delay_action(env_eval, v, task)
                elif baseline_name == "Trust_Greedy":
                    action = _trust_greedy_action(env_eval, v, task)
                elif baseline_name == "Greedy_V2V":
                    action = _v2v_aware_action(env_eval, v, task, use_trust=False)
                elif baseline_name == "Trust_V2V":
                    action = _v2v_aware_action(env_eval, v, task, use_trust=True)
                else:
                    raise ValueError(baseline_name)
                next_state, r_val, done, info, next_task = env_eval.step_vehicle(vid, action, task)
                if not info.get("present", True):
                    continue
                latencies.append(info["delay_ms"])
                successes.append(1 if info["success"] else 0)
                if info["success"]:
                    completed_lat.append(info["delay_ms"])
                else:
                    fr = info.get("failure_reason") or "unknown"
                    fail_reasons[fr] = fail_reasons.get(fr, 0) + 1
                rewards.append(r_val)
                energies.append(info.get("energy_j", 0.0))
                states_dict[vid] = (next_state, next_task)
        eval_records.append({
            "episode_seed": episode_seed,
            "n_observations": len(latencies),
            "latencies": latencies,
            "completed_latencies": completed_lat,
            "failure_reasons": fail_reasons,
            "successes": successes,
            "energies": energies,
            "rewards": rewards,
        })

    all_completed = [x for r in eval_records for x in r["completed_latencies"]]
    all_latencies = [x for r in eval_records for x in r["latencies"]]
    all_successes = [x for r in eval_records for x in r["successes"]]
    all_energies = [x for r in eval_records for x in r["energies"]]
    all_rewards = [x for r in eval_records for x in r["rewards"]]

    mean_latency = float(np.mean(all_latencies)) if all_latencies else None
    mean_completed_latency = float(np.mean(all_completed)) if all_completed else None
    completion = (float(np.mean(all_successes)) * 100.0) if all_successes else None
    mean_energy = float(np.mean(all_energies)) if all_energies else None
    mean_reward = float(np.mean(all_rewards)) if all_rewards else None

    return {
        "seed": seed,
        "avg_latency_ms": None if mean_latency is None else round(mean_latency, 3),
        "avg_completed_latency_ms": None if mean_completed_latency is None else round(mean_completed_latency, 3),
        "completion_rate": None if completion is None else round(completion, 4),
        "failure_rate": None if completion is None else round(100.0 - completion, 4),
        "avg_reward": None if mean_reward is None else round(mean_reward, 4),
        "avg_energy_j": None if mean_energy is None else round(mean_energy, 6),
        "n_observations": len(all_latencies),
        "attack_detection_rate": None,
        "false_positive_rate": None,
        "false_negative_rate": None,
        "detection_precision": None,
        "confusion_counts": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "comm_overhead_kb": 0.0,
        "evaluation_was_inert": True,
        "evaluation_records": eval_records,
        "training_rounds": [],
        "effective_config": {
            "num_vehicles": num_vehicles,
            "attack_ratio": attack_ratio,
            "realised_malicious_clients": int(round(num_vehicles * attack_ratio)),
            "mobility_mode": mobility_mode,
            "seed": seed,
            "evaluation": eval_config.to_dict(),
            "baseline": baseline_name,
            "is_federated": False,
        },
        "datasets": datasets,
        "task_profiles": task_profiles,
    }


# ----------------------------------------------------------------------
# Multi-seed orchestration
# ----------------------------------------------------------------------
def _resolve_jobs(n_tasks):
    """Number of worker processes (``VANNET_JOBS`` env var, default 1)."""
    try:
        jobs = int(os.environ.get("VANNET_JOBS", "1"))
    except ValueError:
        jobs = 1
    return max(1, min(jobs, n_tasks))


_POOL = {"pool": None, "jobs": 0}


def _get_pool(jobs):
    if _POOL["pool"] is None or _POOL["jobs"] != jobs:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        if _POOL["pool"] is not None:
            _POOL["pool"].shutdown()
        _POOL["pool"] = ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn"),
                                            initializer=_worker_init)
        _POOL["jobs"] = jobs
    return _POOL["pool"]


def _worker_init():
    torch.set_num_threads(1)


def _trial_worker(task):
    """Run one seed of one method (module-level so it can be pickled)."""
    s, baseline, method, config, eval_config, kwargs = task
    if baseline is not None:
        return run_heuristic_baseline(baseline, seed=s, config=config,
                                      eval_config=eval_config, **kwargs)
    return run_simulation_trial(seed=s, config=config, eval_config=eval_config,
                                aggregation_strategy=method, **kwargs)


def run_multiseed_experiment(seeds, method="trust_aware", baseline=None, **kwargs):
    """Run a method (or a baseline) across seeds and summarise the results.

    The per-seed raw records are retained so that every aggregate can be
    recomputed and audited.
    """
    config = kwargs.pop("config", None) or load_config()
    eval_config = kwargs.pop("eval_config", None) or EvaluationConfig(
        episodes=config.get("federated_learning", {}).get("eval_episodes", 3),
        steps_per_episode=config.get("federated_learning", {}).get("eval_steps", 20),
        seed_offset=config.get("federated_learning", {}).get("eval_seed_offset", 500_000),
    )

    jobs = _resolve_jobs(len(seeds))
    tasks = [(s, baseline, method, config, eval_config, kwargs) for s in seeds]
    if jobs > 1:
        # One process per seed, from a persistent pool (spawning workers and
        # importing torch costs ~25 s, so the pool is created once).  Every
        # trial reseeds all libraries at its start, so the per-seed records
        # are identical to a sequential run.
        per_seed = list(_get_pool(jobs).map(_trial_worker, tasks))
    else:
        per_seed = [_trial_worker(t) for t in tasks]

    lat = [r["avg_latency_ms"] for r in per_seed]
    comp = [r["completion_rate"] for r in per_seed]
    rew = [r["avg_reward"] for r in per_seed]
    eng = [r["avg_energy_j"] for r in per_seed]
    obs = [r["n_observations"] for r in per_seed]

    total_c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in per_seed:
        for k in total_c:
            total_c[k] += r["confusion_counts"][k]
    pooled = _pooled_from_counts(total_c)
    shadow_total = {}
    for r in per_seed:
        for name, c in (r.get("shadow_confusion") or {}).items():
            tot = shadow_total.setdefault(name, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            for k in tot:
                tot[k] += c[k]

    return {
        "method": baseline if baseline is not None else method,
        "is_baseline": baseline is not None,
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "avg_latency_ms": summarize_metric(lat),
        "avg_completed_latency_ms": summarize_metric(
            [r.get("avg_completed_latency_ms") for r in per_seed]),
        "completion_rate": summarize_metric(comp),
        "failure_rate": summarize_metric([None if c is None else 100.0 - c for c in comp]),
        "avg_reward": summarize_metric(rew),
        "avg_energy_j": summarize_metric(eng),
        "observations_per_seed": summarize_metric(obs),
        "attack_detection_rate": pooled["tpr"],
        "false_positive_rate": pooled["fpr"],
        "false_negative_rate": pooled["fnr"],
        "detection_precision": pooled["precision"],
        "pooled_confusion_counts": total_c,
        "pooled_shadow_confusion": shadow_total,
        "comm_overhead_kb": summarize_metric([r["comm_overhead_kb"] for r in per_seed]),
        "evaluation_was_inert": all(r["evaluation_was_inert"] for r in per_seed),
        "per_seed_results": per_seed,
    }


def compare_methods(results_by_method, reference="Proposed_TFRL"):
    """Paired comparison of every method against the reference method."""
    if reference not in results_by_method:
        return {}
    ref = results_by_method[reference]
    ref_by_seed = {r["seed"]: r for r in ref["per_seed_results"]}
    out = {}
    for name, res in results_by_method.items():
        if name == reference:
            continue
        pairs = []
        for r in res["per_seed_results"]:
            ref_r = ref_by_seed.get(r["seed"])
            if ref_r is None:
                continue
            pairs.append((r, ref_r))
        out[name] = {
            "latency_ms": paired_comparison(
                [p[0]["avg_latency_ms"] for p in pairs],
                [p[1]["avg_latency_ms"] for p in pairs]),
            "completed_latency_ms": paired_comparison(
                [p[0].get("avg_completed_latency_ms") for p in pairs],
                [p[1].get("avg_completed_latency_ms") for p in pairs]),
            "completion_rate": paired_comparison(
                [p[0]["completion_rate"] for p in pairs],
                [p[1]["completion_rate"] for p in pairs]),
            "reward": paired_comparison(
                [p[0]["avg_reward"] for p in pairs],
                [p[1]["avg_reward"] for p in pairs]),
            "energy_j": paired_comparison(
                [p[0]["avg_energy_j"] for p in pairs],
                [p[1]["avg_energy_j"] for p in pairs]),
        }
    return out


# ----------------------------------------------------------------------
# Source revision
# ----------------------------------------------------------------------
def source_revision():
    """A best-effort identifier of the code revision that produced a run."""
    head = os.path.join(PROJECT_ROOT, ".git")
    if os.path.isdir(head):
        try:
            import subprocess
            rev = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            return {"vcs": "git", "commit": rev.decode().strip()}
        except Exception:
            pass
    # No VCS: fingerprint the source files that determine the results.
    parts = []
    for rel in ("web/app.py", "environment/vanet_env.py", "environment/network.py",
                "environment/task.py", "environment/rsu.py", "environment/mobility.py",
                "federated/client.py", "federated/server.py",
                "trust/trust_engine.py", "trust/trust_model.py", "trust/model_trust.py",
                "rl/agent.py", "rl/reward.py", "baselines/heuristics.py",
                "attacks/poisoning.py", "attacks/free_riding.py",
                "trust/alt_detectors.py", "experiments/experiment_runner.py",
                "configs/default_config.yaml"):
        digest = get_file_sha256(os.path.join(PROJECT_ROOT, rel))
        if digest:
            parts.append("%s:%s" % (rel, digest))
    combined = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return {"vcs": "content-hash", "content_sha256": combined, "files": parts}

# ----------------------------------------------------------------------
# Suite orchestration
# ----------------------------------------------------------------------
# The main comparison uses at least 10 independent seeds so that the
# confidence intervals and the paired tests have a usable sample size.
DEFAULT_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111,
                 1212, 1313, 1414, 1515]


def _environment_versions():
    """Versions of the software that produced a run."""
    import scipy
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _validate_results(results, expected_seed_count):
    """Check that a run is internally consistent before it may be published.

    Returns ``(ok, problems)``.
    """
    problems = []
    for name, res in results.items():
        if res.get("num_seeds", 0) < expected_seed_count:
            problems.append("%s: only %d seeds (expected %d)"
                            % (name, res.get("num_seeds", 0), expected_seed_count))
        if not res.get("evaluation_was_inert", False):
            problems.append("%s: the evaluation phase modified the learned parameters" % name)
        for r in res.get("per_seed_results", []):
            if r.get("n_observations", 0) <= 0:
                problems.append("%s (seed %s): no evaluation observation was recorded"
                                % (name, r.get("seed")))
                break
        # Every aggregate must be recomputable from the raw records.
        lat = [r["avg_latency_ms"] for r in res.get("per_seed_results", [])
               if r.get("avg_latency_ms") is not None]
        if lat:
            mean = res.get("avg_latency_ms", {}).get("mean")
            if mean is None or abs(mean - float(np.mean(lat))) > 1e-6:
                problems.append("%s: the latency aggregate does not match the raw records" % name)
    return (len(problems) == 0), problems


def run_experiment_suite(save_dir="results", seeds=None, quick=False, label=None, resume_from=None):
    """Run the corrected benchmark suite and write a new versioned artefact set.

    The existing artefacts under ``results/`` are never modified.  A new
    ``run_<timestamp>_<label>`` directory is created, and the active pointer
    (``results/benchmark_results.json``) is only replaced once the new run
    has passed validation.
    """
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    if quick:
        seeds = seeds[:2]
    # Parallel seeds: one worker process per seed (results are identical to a
    # sequential run because every trial reseeds all libraries).
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(6, (os.cpu_count() or 2)))))
    # The main comparison uses the full seed budget; the parameter sweeps use
    # a smaller one so that the suite stays within a sane compute budget.  The
    # seed count of every experiment is recorded in its result record.
    sweep_seeds = seeds[:6] if len(seeds) > 6 else seeds

    config = load_config()
    if quick:
        # Smoke configuration: exercises every code path in a few minutes.  A
        # quick run is never published as the active result set.
        config["federated_learning"].update({"num_rounds": 2, "steps_per_round": 4,
                                             "local_epochs": 2, "eval_episodes": 1,
                                             "eval_steps": 4})
    eval_config = EvaluationConfig(
        episodes=config.get("federated_learning", {}).get("eval_episodes", 3),
        steps_per_episode=config.get("federated_learning", {}).get("eval_steps", 20),
        seed_offset=config.get("federated_learning", {}).get("eval_seed_offset", 500_000),
    )

    attack_ratio = config.get("adversarial", {}).get("attack_ratio", 0.24)
    mobility_mode = config.get("mobility", {}).get("scenario", "highway")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    label = label or ("quick" if quick else "full")
    versioned_dir = os.path.join(save_dir, "run_%s_%s" % (timestamp, label))
    if resume_from:
        # Continue inside the interrupted run's own directory: finished cells
        # are reloaded from its partial_results.json and never recomputed.
        versioned_dir = resume_from
    os.makedirs(versioned_dir, exist_ok=True)

    print("=" * 78)
    print("CORRECTED BENCHMARK SUITE")
    print("=" * 78)
    print("seeds            : %d (%s)" % (len(seeds), seeds))
    print("evaluation       : %d episodes x %d steps, seed offset %d"
          % (eval_config.episodes, eval_config.steps_per_episode, eval_config.seed_offset))
    print("mobility mode    : %s" % mobility_mode)
    print("attack ratio     : %s" % attack_ratio)
    print("output directory : %s" % versioned_dir)
    print("")

    all_results = {}
    partial_path = os.path.join(versioned_dir, "partial_results.json")
    if resume_from:
        with open(os.path.join(resume_from, "partial_results.json")) as f:
            all_results = json.load(f)
        print("resuming: %d finished experiment groups loaded" % len(all_results))

    def checkpoint():
        with open(partial_path, "w") as f:
            json.dump(all_results, f)

    def group(exp_key):
        return all_results.setdefault(exp_key, {})

    def run(exp_key, name, seeds_, cfg=None, **kw):
        """Run one (experiment, scenario) cell unless it was already finished."""
        g = group(exp_key)
        if name in g and isinstance(g[name], dict) and g[name].get("per_seed_results"):
            print("    %s / %s (resumed)" % (exp_key, name))
            return g[name]
        print("    %s / %s ..." % (exp_key, name), flush=True)
        t0 = time.time()
        g[name] = run_multiseed_experiment(seeds_, config=cfg or config,
                                           eval_config=eval_config, **kw)
        print("      done in %.0f s" % (time.time() - t0), flush=True)
        checkpoint()
        return g[name]

    def variant(**sections):
        cfg = copy.deepcopy(config)
        for sec, kv in sections.items():
            cfg.setdefault(sec, {}).update(kv)
        return cfg

    common = dict(attack_ratio=attack_ratio, mobility_mode=mobility_mode)

    # ---------------- Experiment A: baseline comparison ----------------
    print("[A] Baseline comparison (paired evaluation)")
    baselines = ["Local_Only", "Random_Valid", "Nearest_RSU", "Resource_Aware"]
    methods = {
        "Proposed_TFRL": dict(method="trust_aware", shadow=True),
        "Standard_FedAvg_RL": dict(method="fedavg"),
        "Robust_FL_Median": dict(method="coordinate_median"),
        "Robust_FL_TrimmedMean": dict(method="trimmed_mean"),
        "Robust_FL_MultiKrum": dict(method="multi_krum"),
        "Non_Federated_D3QN": dict(method="trust_aware", is_federated=False),
        # Reference rows that frame the downstream question:
        "FedAvg_Clean_Reference": dict(method="fedavg", attack_ratio=0.0),
        "Oracle_Exclusion_Upper_Bound": dict(method="oracle_exclusion"),
    }
    for name, kwargs in methods.items():
        kw = dict(common)
        kw.update(kwargs)
        run("experiment_a_baseline_comparison", name, seeds, **kw)
    for name in baselines:
        run("experiment_a_baseline_comparison", name, seeds, baseline=name, **common)
    comparison = all_results["experiment_a_baseline_comparison"]
    all_results["experiment_a_paired_comparison"] = compare_methods(comparison)
    # Downstream impact of the attack: everything against the clean reference.
    all_results["experiment_a_paired_vs_clean_reference"] = compare_methods(
        comparison, reference="FedAvg_Clean_Reference")
    checkpoint()

    # Cells that are identical to a main-comparison cell (same method, same
    # configuration, same seeds) are REUSED rather than recomputed; the
    # reused record is the very same object, so nothing is duplicated or
    # re-drawn.  Sweeps run on the first ``len(sweep_seeds)`` seeds, and the
    # paired tests pair by seed.
    proposed = comparison["Proposed_TFRL"]
    fedavg_ref = comparison["Standard_FedAvg_RL"]

    # ---------------- Experiment B: mobility sweep ----------------
    print("[B] Mobility sweep")
    group("experiment_b_mobility_sweep")["highway"] = proposed
    group("experiment_b2_mobility_fedavg")["highway"] = fedavg_ref
    for mode in ("authentic_trace",):
        for tag, kwargs in (("experiment_b_mobility_sweep", dict(method="trust_aware")),
                            ("experiment_b2_mobility_fedavg", dict(method="fedavg"))):
            try:
                run(tag, mode, sweep_seeds, attack_ratio=attack_ratio, mobility_mode=mode, **kwargs)
            except Exception as exc:
                print("    %s SKIPPED: %s: %s" % (mode, type(exc).__name__, exc))
                group(tag)[mode] = {"status": "skipped", "reason": str(exc)}

    # ---------------- Experiment C: attack-ratio sweep ----------------
    print("[C] Malicious-client ratio sweep")
    group("experiment_c_attack_ratio_sweep")[str(attack_ratio)] = proposed
    group("experiment_c2_attack_ratio_fedavg")[str(attack_ratio)] = fedavg_ref
    for ratio in (0.0, 0.4):
        for tag, kwargs in (("experiment_c_attack_ratio_sweep", dict(method="trust_aware", shadow=True)),
                            ("experiment_c2_attack_ratio_fedavg", dict(method="fedavg"))):
            run(tag, str(ratio), sweep_seeds, attack_ratio=ratio, mobility_mode=mobility_mode, **kwargs)

    # ---------------- Experiment D: ablation ----------------
    print("[D] Ablation study")
    group("experiment_d_ablation")["Full_Proposed"] = proposed
    group("experiment_d_ablation")["FedAvg_RL_No_Trust"] = fedavg_ref
    ablations = {
        "Without_Behavioral_Trust": dict(method="trust_aware", alpha=0.0),
        "Without_Model_Trust": dict(method="trust_aware", alpha=1.0),
        "Without_Exclusion": dict(method="trust_no_exclusion"),
        "Without_Trust_Weighting": dict(method="exclusion_no_weighting"),
    }
    for name, kwargs in ablations.items():
        run("experiment_d_ablation", name, sweep_seeds, **dict(common, **kwargs))
    ablation_results = all_results["experiment_d_ablation"]
    all_results["experiment_d_paired_comparison"] = compare_methods(ablation_results)

    # ---------------- Experiment E: alpha sensitivity ----------------
    print("[E] Alpha sensitivity")
    group("experiment_e_alpha_sensitivity")["0.5"] = proposed
    group("experiment_e_alpha_sensitivity")["0.0"] = ablation_results["Without_Behavioral_Trust"]
    group("experiment_e_alpha_sensitivity")["1.0"] = ablation_results["Without_Model_Trust"]
    for alpha in (0.2, 0.8):
        run("experiment_e_alpha_sensitivity", str(alpha), sweep_seeds,
            method="trust_aware", alpha=alpha, **common)

    # ---------------- Experiment F: attack families ----------------
    print("[F] Attack families (downstream impact and detection)")
    attacks = {
        "sign_flip_partial_25": dict(poison_type="partial_sign_flip", poison_strength=0.25),
        "stealth_sign_flip_norm_matched": dict(poison_type="stealth_sign_flip", poison_strength=1.0),
        "intermittent_sign_flip_p0.5": dict(poison_type="sign_flip", poison_strength=1.5,
                                           attack_probability=0.5),
    }
    group("experiment_f_attack_families")["sign_flip_full/trust_aware"] = proposed
    group("experiment_f_attack_families")["sign_flip_full/fedavg"] = fedavg_ref
    for atk, akw in attacks.items():
        for mname, mkw in (("trust_aware", dict(method="trust_aware", shadow=True)),
                           ("fedavg", dict(method="fedavg"))):
            run("experiment_f_attack_families", "%s/%s" % (atk, mname), sweep_seeds,
                **dict(common, **dict(akw, **mkw)))

    # ---------------- Experiment G: participation and connectivity ----------------
    print("[G] Partial participation and connectivity gating")
    for tag, cfg_g in (("participation_0.5", variant(federated_learning={"participation_fraction": 0.5})),
                       ("connectivity_gated", variant(federated_learning={"require_connectivity": True}))):
        for mname, mkw in (("trust_aware", dict(method="trust_aware")), ("fedavg", dict(method="fedavg"))):
            run("experiment_g_participation", "%s/%s" % (tag, mname), sweep_seeds, cfg=cfg_g,
                **dict(common, **mkw))

    # ---------------- Experiment H: client heterogeneity ----------------
    print("[H] Task-distribution heterogeneity")
    for tag, cfg_h in (("dirichlet_0.1_extreme", variant(task_profiles={"dirichlet_alpha": 0.1})),):
        for mname, mkw in (("trust_aware", dict(method="trust_aware")), ("fedavg", dict(method="fedavg"))):
            run("experiment_h_heterogeneity", "%s/%s" % (tag, mname), sweep_seeds, cfg=cfg_h,
                **dict(common, **mkw))

    # ---------------- Experiment I: reward sensitivity ----------------
    print("[I] Reward: legacy trust bonus (lambda_trust = 2.5)")
    cfg_i = variant(reward={"lambda_trust": 2.5})
    for mname, mkw in (("trust_aware", dict(method="trust_aware")), ("fedavg", dict(method="fedavg"))):
        run("experiment_i_reward_trust_bonus", mname, sweep_seeds, cfg=cfg_i, **dict(common, **mkw))
    for b in ("Local_Only", "Nearest_RSU"):
        run("experiment_i_reward_trust_bonus", b, sweep_seeds, cfg=cfg_i, baseline=b, **common)

    # ---------------- Experiment J: environment version 1 ----------------
    print("[J] Environment version 1 (silent RSU re-association, uncontended uplink, trust bonus)")
    cfg_j = variant(environment_model={"honor_rsu_choice": False, "uplink_contention": False},
                    reward={"lambda_trust": 2.5})
    for mname, mkw in (("trust_aware", dict(method="trust_aware")), ("fedavg", dict(method="fedavg"))):
        run("experiment_j_environment_v1", mname, sweep_seeds, cfg=cfg_j, **dict(common, **mkw))
    for b in ("Local_Only", "Nearest_RSU", "Resource_Aware"):
        run("experiment_j_environment_v1", b, sweep_seeds, cfg=cfg_j, baseline=b, **common)

    # ---------------- Experiment K: mixed threat (model + behavioural) ----------------
    print("[K] Mixed threat: poisoning clients plus task-dropping helpers")
    cfg_k = variant(adversarial={"unreliable_helper_ratio": 0.16})
    for name, kw in (("fedavg", dict(method="fedavg")),
                     ("trust_aware_alpha_0.0_model_only", dict(method="trust_aware", alpha=0.0)),
                     ("trust_aware_alpha_0.5_joint", dict(method="trust_aware", alpha=0.5)),
                     ("trust_aware_alpha_1.0_behaviour_only", dict(method="trust_aware", alpha=1.0))):
        run("experiment_k_mixed_threat", name, sweep_seeds, cfg=cfg_k, **dict(common, **kw))

    mobility_sweep = group("experiment_b_mobility_sweep")
    ratio_sweep = group("experiment_c_attack_ratio_sweep")
    alpha_sweep = group("experiment_e_alpha_sensitivity")

    # ---------------- Provenance ----------------
    dataset_provenance = {}
    for mode in ("highway", "trace", "authentic_trace"):
        dataset_provenance[mode] = _resolve_datasets(config, mode)
    dataset_provenance["task_profiles"] = _resolve_task_profiles(config)

    metadata = {
        "run_id": os.path.basename(versioned_dir),
        "run_kind": "quick" if quick else "full",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": seeds,
        "num_seeds": len(seeds),
        "sweep_seeds": sweep_seeds,
        "num_sweep_seeds": len(sweep_seeds),
        "evaluation_config": eval_config.to_dict(),
        "dataset_provenance": dataset_provenance,
        "code_revision": source_revision(),
        "environment_versions": _environment_versions(),
        "attack_settings": {
            "attack_ratio": attack_ratio,
            "realised_malicious_clients": int(round(
                config.get("mobility", {}).get("num_vehicles", 25) * attack_ratio)),
            "poison_strength": config.get("adversarial", {}).get("poison_strength", 1.5),
            "poison_type": config.get("adversarial", {}).get("poison_type", "sign_flip"),
        },
        "mobility_mode": mobility_mode,
        "training_protocol": {
            "num_rounds": config["federated_learning"].get("num_rounds"),
            "steps_per_round": config["federated_learning"].get("steps_per_round"),
            "local_epochs_gradient_steps_per_round": config["federated_learning"].get("local_epochs"),
            "initialisation": "federated methods: all clients start from the server's single global model; "
                              "Non_Federated_D3QN: independent random initialisation per client",
            "client_data": "each client trains only on its own replay buffer of its own vehicle's transitions "
                           "(non-IID task mixture, Dirichlet concentration %s)"
                           % config.get("task_profiles", {}).get("dirichlet_alpha", 0.4),
            "exploration": "epsilon-greedy during the first 40% of rounds, greedy afterwards",
            "evaluation": "separate environments and seeds, greedy policy of the client's own (post-sync) "
                          "network, no buffer writes, no gradient steps; parameter hash checked before/after",
            "agent": "dueling double DQN, Huber loss, Adam, gamma/learning rate from federated_learning config",
        },
        "environment_model": dict(config.get("environment_model", {}),
                                  reward=config.get("reward", {}),
                                  cell_bandwidth_mhz=config.get("network", {}).get("cell_bandwidth_mhz")),
        "limitations": [
            "Tasks, channel, queues, rewards and attacks are simulated; only the optional mobility replay is measured data.",
            "Edge queues are deterministic multi-core servers, not M/M/c; tasks of one slot arrive together in a seeded random order.",
            "Attackers are scripted and non-adaptive except the norm-matched stealth sign flip.",
            "Evaluation horizons are short (episodes x steps given in evaluation_config); results do not describe long-run behaviour.",
            "Sweeps use fewer seeds than the main comparison (see sweep_seeds).",
        ],
        "provenance_disclosure": {
            "task_profiles": "3GPP TR 36.885 derived, scaled by 0.1; modelling assumption",
            "rl_environment": "simulated 3GPP-style V2X channel, deterministic edge queues, modelled reward",
            "eval_isolation": "independent episodes, separate seeds, greedy policy, no buffer writes, no gradients",
            "participation_rule": (
                "clients holding fewer transitions than the minimum training batch "
                "(4) could not perform a gradient step; they are non-participants "
                "of the round and are not judged. The rule uses the client's "
                "available experience, never the ground-truth role. Identical to "
                "the live console (web/app.py)."
                if config.get("trust_management", {}).get("exclude_non_participants", True)
                else "every participant is judged, including clients that uploaded "
                     "no usable update (pre-fix protocol)"),
            "mobility": ("procedural IDM" if mobility_mode == "highway"
                         else "trace replay (see dataset_provenance for the exact file and checksum)"),
        },
        "resumed_from_partial_results": bool(resume_from),
        "validation_status": "pending",
    }

    ok, problems = _validate_results(comparison, len(seeds))
    for key, grp in all_results.items():
        if key == "experiment_a_baseline_comparison" or "paired" in key:
            continue
        runnable = {n: v for n, v in grp.items() if isinstance(v, dict) and "per_seed_results" in v}
        sweep_ok, sweep_problems = _validate_results(runnable, len(sweep_seeds))
        if not sweep_ok:
            problems.extend("%s: %s" % (key, p_) for p_ in sweep_problems)
    metadata["validation_status"] = "passed" if ok else "failed"
    metadata["validation_problems"] = problems
    if problems:
        print("")
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print("  - %s" % p)

    # ---------------- Write the artefacts ----------------
    with open(os.path.join(versioned_dir, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(versioned_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(versioned_dir, "effective_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # The active pointer is only updated when the run validated *and* it was a
    # full run.  A diagnostic (reduced-seed) run must never be published as
    # the active result set.
    active_updated = False
    if ok and not quick:
        with open(os.path.join(save_dir, "benchmark_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        with open(os.path.join(save_dir, "benchmark_results_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        active_updated = True
    elif ok and quick:
        print("")
        print("This was a diagnostic run: the active results pointer was NOT updated.")
    else:
        print("")
        print("The active results pointer was NOT updated because validation failed.")

    manifest_path = os.path.join(save_dir, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = []

    if active_updated:
        # Publishing a new run demotes the previously published one.  The
        # historical artefacts stay on disk untouched; only the *active*
        # pointer moves, so exactly one run is ever published as active.
        for entry in manifest:
            if not isinstance(entry, dict):
                continue
            if entry.get("published_as_active"):
                entry["published_as_active"] = False
                entry["superseded"] = True
                entry["superseded_by"] = metadata["run_id"]
                entry.setdefault(
                    "superseded_reason",
                    "Published before %s; its artefacts are preserved but it is "
                    "no longer the active result set." % metadata["run_id"])

    manifest.append({
        "run_id": metadata["run_id"],
        "directory": versioned_dir,
        "timestamp_utc": metadata["timestamp_utc"],
        "num_seeds": len(seeds),
        "seeds": seeds,
        "validation_status": metadata["validation_status"],
        "published_as_active": active_updated,
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Publication figures -------------------------------------------
    # The figures are recomputed from the executable outputs of *this* run.
    figures = []
    try:
        from experiments.plotter import generate_all_publication_plots
        import tempfile
        # 1. the versioned run keeps its own figures
        versioned_plots = os.path.join(versioned_dir, "plots")
        os.makedirs(versioned_plots, exist_ok=True)
        figures = generate_all_publication_plots(
            os.path.join(versioned_dir, "benchmark_results.json"),
            versioned_plots,
            os.path.join(versioned_dir, "metadata.json"),
            clean=True)
        from experiments.research_figures import generate as _research_figures
        figures = figures + _research_figures(
            os.path.join(versioned_dir, "benchmark_results.json"), versioned_plots,
            os.path.join(versioned_dir, "metadata.json"))
        # 2. the published pointer keeps a copy of the active figures
        if active_updated:
            generate_all_publication_plots(
                os.path.join(save_dir, "benchmark_results.json"),
                os.path.join(PROJECT_ROOT, "plots"),
                os.path.join(save_dir, "benchmark_results_metadata.json"),
                clean=True)
            _research_figures(os.path.join(save_dir, "benchmark_results.json"),
                              os.path.join(PROJECT_ROOT, "plots"),
                              os.path.join(save_dir, "benchmark_results_metadata.json"))
    except Exception as exc:  # noqa: BLE001
        print("Figure generation note: %s: %s" % (type(exc).__name__, exc))

    metadata["figures"] = figures
    with open(os.path.join(versioned_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    if active_updated:
        with open(os.path.join(save_dir, "benchmark_results_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print("")
    print("Artefacts written to : %s" % versioned_dir)
    print("Figures generated    : %d" % len(figures))
    print("Validation           : %s" % metadata["validation_status"])
    print("Active pointer updated: %s" % active_updated)
    return {"results": all_results, "metadata": metadata, "directory": versioned_dir}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="run only the first three seeds (diagnostic)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--resume", default=None,
                        help="run directory of an interrupted run; finished cells are reused")
    args = parser.parse_args()
    run_experiment_suite(seeds=args.seeds, quick=args.quick, label=args.label,
                         resume_from=args.resume)
