"""
Extended experiments (data-plane attacks, V2V, empirical edge calibration).

Motivation.  In the active benchmark the model-level attackers behave honestly
as V2V helpers, so the *behavioural* tier of the trust framework has nothing to
detect and the benchmark (correctly) finds it adds no measurable value.  Whether
that conclusion survives an attacker that also misbehaves in the data plane is a
separate question, answered here.  Like the supplementary suite, this suite never
touches the active run; it writes its own versioned directory with
``run_kind: extended`` and reuses the parent's seeds and protocol.

Experiments
-----------
P  Data-plane attack and V2V.  Poisoning vehicles sabotage the V2V tasks they
   serve (70 % drop, 3x delay) and free riders refuse 80 % of them.  Heuristics
   that consider the V2V peer are compared with and without behavioural trust,
   and with nearest vs trust-aware peer selection.
Q  Learned policy under the data-plane attack: with trust features and with
   them masked.
R  Empirical edge calibration (Alibaba edge-platform trace): do the heuristic
   ranking and the learned-policy result survive co-tenant load and measured
   hand-over delay/loss?

    python -m experiments.extended_suite
    python -m experiments.extended_suite --quick
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from experiments import experiment_runner as er
from experiments.supplementary_suite import _parent_run, RESULTS

def run(quick=False, resume_from=None, label="extended_data_plane_edge_calibration"):
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
    run_dir = resume_from or os.path.join(RESULTS, "run_%s_%s" % (stamp, label if not quick else "ext_quick"))
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
            if sec == "adversarial_dp":
                cfg["adversarial"].setdefault("data_plane", {}).update(kv)
            else:
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

    DP = dict(enabled=True)
    # 30 % of the vehicles are compute-rich helpers (4-8 GHz, e.g. trucks with an
    # on-board server).  Without them a peer is never faster than the requester
    # and V2V offloading cannot pay off (see environment_spec).
    RICH = {"vehicle_compute": {"rich_fraction": 0.3, "rich_cpu_ghz": [4.0, 8.0]}}

    def v2v_cfg(attacked, selection):
        return variant(mobility=RICH, adversarial_dp=dict(enabled=attacked),
                       network={"v2v_peer_selection": selection})

    # ---------------- P: data-plane attack and V2V ----------------
    print("[P] Data-plane attack: malicious / selfish V2V helpers (30 % compute-rich helpers)")
    g = "experiment_p_data_plane_v2v"
    for tag, attacked in (("attacked", True), ("clean", False)):
        cell(g, "Resource_Aware_no_v2v_" + tag, v2v_cfg(attacked, "nearest"), baseline="Resource_Aware", **common)
        cell(g, "Greedy_V2V_nearest_" + tag, v2v_cfg(attacked, "nearest"), baseline="Greedy_V2V", **common)
        cell(g, "Trust_V2V_nearest_" + tag, v2v_cfg(attacked, "nearest"), baseline="Trust_V2V", **common)
        cell(g, "Greedy_V2V_trustpeer_" + tag, v2v_cfg(attacked, "trust_aware"), baseline="Greedy_V2V", **common)
        cell(g, "Trust_V2V_trustpeer_" + tag, v2v_cfg(attacked, "trust_aware"), baseline="Trust_V2V", **common)

    # ---------------- Q: learned policy under the data-plane attack ----------------
    print("[Q] Learned policy under the data-plane attack")
    g = "experiment_q_learned_data_plane"
    cfg_q = variant(mobility=RICH, adversarial_dp=DP, network={"v2v_peer_selection": "trust_aware"})
    cfg_qm = variant(mobility=RICH, adversarial_dp=DP, network={"v2v_peer_selection": "trust_aware"},
                     environment_model={"mask_trust_features": True})
    cell(g, "learned_with_trust_features", cfg_q, method="trust_aware", **common)
    cell(g, "learned_trust_features_masked", cfg_qm, method="trust_aware", **common)

    # ---------------- R: empirical edge calibration ----------------
    print("[R] Empirical edge calibration (Alibaba edge-platform trace), trough and peak hour")
    g = "experiment_r_edge_calibration"
    for hour, tag in ((5.0, "trough_05h"), (20.0, "peak_20h")):
        cfg_r = variant(edge_calibration={"enabled": True, "start_hour": hour})
        for b in ("Nearest_RSU", "Resource_Aware", "Local_Only"):
            cell(g, "%s_%s" % (b, tag), cfg_r, baseline=b, **common)
        cell(g, "learned_" + tag, cfg_r, method="trust_aware", **common)

    # ---------------- validation, metadata, manifest ----------------
    problems = []
    for grp in results.values():
        ok, p = er._validate_results({k: v for k, v in grp.items() if "per_seed_results" in v}, len(seeds))
        problems.extend(p)
    metadata = {
        "run_id": os.path.basename(run_dir),
        "run_kind": "extended" if not quick else "quick",
        "supplements": parent_entry["run_id"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": seeds,
        "evaluation_config": eval_config.to_dict(),
        "training_protocol": {k: config["federated_learning"].get(k) for k in
                              ("num_rounds", "steps_per_round", "local_epochs")},
        "environment_model": dict(config.get("environment_model", {}), reward=config.get("reward", {})),
        "data_plane_attack": dict(malicious_drop_prob=0.7, malicious_delay_factor=3.0, selfish_drop_prob=0.8),
        "edge_calibration": config.get("edge_calibration"),
        "compute_rich_helpers": {"rich_fraction": 0.3, "rich_cpu_ghz": [4.0, 8.0]},
        "code_revision": er.source_revision(),
        "environment_versions": er._environment_versions(),
        "reference_cells": "Cells of the supplemented run with the same seeds are the reference.",
        "limitations": [
            "The data-plane attack is a scripted behaviour (fixed drop probability and delay factor), not a measured one.",
            "The edge trace calibrates RSU co-tenant load and hand-over delay only; it contains no vehicular data.",
            "Edge capacity and background load use damped, clipped samples of the measured distributions.",
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
            "kind": "extended", "supplements": parent_entry["run_id"],
            "note": "Extended experiments (data-plane attack and V2V, learned policy under it, empirical edge calibration) for the active run; the active run itself is unchanged."})
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
