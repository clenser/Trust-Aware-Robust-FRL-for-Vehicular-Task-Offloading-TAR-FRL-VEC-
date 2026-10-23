"""
Supplementary experiments for the problem statement's edge-node and
intermittent-connectivity requirements.

The active benchmark (``results/manifest.json``) is never modified.  This suite
writes its OWN versioned run directory, with ``run_kind: supplementary`` and a
manifest entry that names the run it supplements, so provenance stays intact.
It uses the same seeds (the parent's sweep seeds) and the same training and
evaluation protocol, so its cells can be compared with the parent's cells by
seed.

Experiments
-----------
L  Faulty edge node.  One RSU (RSU_2) silently drops 60 % of the tasks it is
   given.  Does *behavioural* trust help the offloading decision?  The learned
   policy with trust features is compared with the same policy with the trust
   features masked, and with heuristics that do and do not use behavioural trust.
M  Intermittent connectivity.  Each client is offline in a round with
   probability 0.4 and then also misses the model broadcast, so its next upload
   is stale.  FedAvg, trust-aware aggregation, and trust-aware aggregation with a
   staleness discount are compared.
K2 Mixed threat with correct labels.  Task-dropping vehicles are honest
   participants of the federated round; only poisoning and free-riding clients are
   positives for the model-update detector.

    python -m experiments.supplementary_suite                 # ~1 h with 6 workers
    python -m experiments.supplementary_suite --quick         # smoke test
    python -m experiments.supplementary_suite --resume results/run_<...>
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from experiments import experiment_runner as er

RESULTS = os.path.join(er.PROJECT_ROOT, "results")

FAULTY_RSU = [{"index": 1, "drop_prob": 0.6, "delay_factor": 1.0}]


def _parent_run():
    from web.app import _active_run_entry
    entry = _active_run_entry(RESULTS)
    if entry is None:
        raise SystemExit("no active validated run to supplement")
    directory = entry.get("directory") or ""
    if not os.path.isabs(directory):
        directory = os.path.join(er.PROJECT_ROOT, directory)
    with open(os.path.join(directory, "metadata.json")) as f:
        meta = json.load(f)
    return entry, meta


def run(quick=False, resume_from=None, label="supplementary_edge_node_staleness"):
    parent_entry, parent_meta = _parent_run()
    seeds = list(parent_meta["sweep_seeds"])
    config = load_config()
    if quick:
        seeds = seeds[:2]
        config["federated_learning"].update({"num_rounds": 2, "steps_per_round": 4, "local_epochs": 2,
                                             "eval_episodes": 1, "eval_steps": 4})
    os.environ.setdefault("VANNET_JOBS", str(max(1, min(6, (os.cpu_count() or 2)))))
    eval_config = er.EvaluationConfig(
        episodes=config["federated_learning"].get("eval_episodes", 5),
        steps_per_episode=config["federated_learning"].get("eval_steps", 30),
        seed_offset=config["federated_learning"].get("eval_seed_offset", 500_000))
    attack_ratio = config["adversarial"].get("attack_ratio", 0.24)
    mobility_mode = config["mobility"].get("scenario", "highway")
    common = dict(attack_ratio=attack_ratio, mobility_mode=mobility_mode)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = resume_from or os.path.join(RESULTS, "run_%s_%s" % (stamp, label if not quick else "supp_quick"))
    os.makedirs(run_dir, exist_ok=True)
    partial = os.path.join(run_dir, "partial_results.json")
    results = {}
    if resume_from and os.path.exists(partial):
        with open(partial) as f:
            results = json.load(f)
        print("resuming: %d groups loaded" % len(results))

    def variant(**sections):
        cfg = copy.deepcopy(config)
        for sec, kv in sections.items():
            cfg.setdefault(sec, {}).update(kv)
        return cfg

    def cell(group, name, cfg=None, **kw):
        g = results.setdefault(group, {})
        if name in g and g[name].get("per_seed_results"):
            print("    %s / %s (resumed)" % (group, name))
            return
        print("    %s / %s ..." % (group, name), flush=True)
        t0 = time.time()
        g[name] = er.run_multiseed_experiment(seeds, config=cfg or config, eval_config=eval_config, **kw)
        print("      done in %.0f s" % (time.time() - t0), flush=True)
        with open(partial, "w") as f:
            json.dump(results, f)

    # ---------------- L: faulty edge node ----------------
    print("[L] Faulty edge node (RSU_2 drops 60 percent of tasks)")
    cfg_f = variant(adversarial={"faulty_rsus": FAULTY_RSU})
    cfg_fm = variant(adversarial={"faulty_rsus": FAULTY_RSU}, environment_model={"mask_trust_features": True})
    cfg_m = variant(environment_model={"mask_trust_features": True})
    g = "experiment_l_faulty_edge_node"
    cell(g, "learned_with_trust_features", cfg_f, method="trust_aware", **common)
    cell(g, "learned_trust_features_masked", cfg_fm, method="trust_aware", **common)
    cell(g, "learned_masked_no_fault", cfg_m, method="trust_aware", **common)
    for b in ("Local_Only", "Nearest_RSU", "Resource_Aware", "Trust_Greedy"):
        cell(g, b + "_faulty", cfg_f, baseline=b, **common)
    cell(g, "Trust_Greedy_no_fault", config, baseline="Trust_Greedy", **common)

    # ---------------- M: intermittent connectivity + stale updates ----------------
    print("[M] Intermittent connectivity (40 percent offline per round, missed broadcasts)")
    fl_m = {"connectivity_dropout": 0.4, "missed_broadcast": True}
    cfg_m1 = variant(federated_learning=dict(fl_m, staleness_mode="none"))
    cfg_m2 = variant(federated_learning=dict(fl_m, staleness_mode="discount"))
    g = "experiment_m_intermittent_connectivity"
    cell(g, "fedavg_stale_uploads", cfg_m1, method="fedavg", **common)
    cell(g, "trust_aware_no_staleness_handling", cfg_m1, method="trust_aware", **common)
    cell(g, "trust_aware_staleness_discount", cfg_m2, method="trust_aware", **common)

    # ---------------- K2: mixed threat, correct labels ----------------
    print("[K2] Mixed threat with correct labels")
    cfg_k = variant(adversarial={"unreliable_helper_ratio": 0.16})
    g = "experiment_k2_mixed_threat_relabelled"
    cell(g, "trust_aware_alpha_0.5_joint", cfg_k, method="trust_aware", alpha=0.5, **common)
    cell(g, "trust_aware_alpha_0.0_model_only", cfg_k, method="trust_aware", alpha=0.0, **common)

    # ---------------- validation, metadata, manifest ----------------
    problems = []
    for grp in results.values():
        ok, p = er._validate_results({k: v for k, v in grp.items() if "per_seed_results" in v}, len(seeds))
        problems.extend(p)
    metadata = {
        "run_id": os.path.basename(run_dir),
        "run_kind": "supplementary" if not quick else "quick",
        "supplements": parent_entry["run_id"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": seeds,
        "evaluation_config": eval_config.to_dict(),
        "training_protocol": {k: config["federated_learning"].get(k) for k in
                              ("num_rounds", "steps_per_round", "local_epochs")},
        "environment_model": dict(config.get("environment_model", {}), reward=config.get("reward", {})),
        "faulty_rsu": FAULTY_RSU,
        "connectivity": fl_m,
        "code_revision": er.source_revision(),
        "environment_versions": er._environment_versions(),
        "reference_cells": ("Cells of the supplemented run with the same seeds (Proposed_TFRL, Standard_FedAvg_RL, "
                            "Nearest_RSU, Resource_Aware, Local_Only) are the no-fault / no-dropout references."),
        "limitations": [
            "The edge-node fault is injected (a scripted 60 % drop rate on one RSU), not measured.",
            "Connectivity dropout is an independent Bernoulli process, not a mobility-derived outage model.",
            "Behavioural trust of RSUs starts from the neutral prior in every evaluation episode and adapts within it.",
            "Six seeds; exploratory, not multiplicity-corrected.",
        ],
        "validation_status": "passed" if not problems else "failed",
        "validation_problems": problems,
    }
    with open(os.path.join(run_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(run_dir, "effective_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    manifest_path = os.path.join(RESULTS, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest = [e for e in manifest if e.get("run_id") != metadata["run_id"]]
    if not quick:
        manifest.append({
            "run_id": metadata["run_id"], "directory": os.path.relpath(run_dir, er.PROJECT_ROOT),
            "timestamp_utc": metadata["timestamp_utc"], "num_seeds": len(seeds), "seeds": seeds,
            "validation_status": metadata["validation_status"], "published_as_active": False,
            "kind": "supplementary", "supplements": parent_entry["run_id"],
            "note": "Supplementary experiments (faulty edge node, intermittent connectivity, relabelled mixed threat) "
                    "for the active run; the active run itself is unchanged."})
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    print("Artefacts written to %s; validation %s" % (run_dir, metadata["validation_status"]))
    return run_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    run(quick=a.quick, resume_from=a.resume)
