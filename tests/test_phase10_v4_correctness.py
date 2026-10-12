"""Phase 10 v4 correctness regression tests.

Covers the v4 acceptance gates: parameter propagation/precedence, exact
resource configs, fault accounting and destination attribution, stale-replay
provenance, matrix completeness, Holm statistics, manifest verification, and
dashboard v4 visibility. All assertions check effective/applied values.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from attacks.base import AttackType, ScenarioAssignment
from attacks.pipeline import AttackPipeline
from attacks.unreliable_execution import UnreliableEnvWrapper, UnreliableProfile
from environment import EnvConfig, VehicularEdgeEnv
from federated.server import FedServer
from rl.agent import DQNAgent
from scripts.analyze_phase10_results import analyze_phase10_results, holm_adjust_pvalues
from scripts.run_phase10_final_evaluation import (
    CORE_DEFAULTS,
    PlanItem,
    assert_plan_counts,
    build_clients_with_seed,
    build_phase10_experiment_plan,
    execute_plan_item,
    resolve_detector_params,
)


def load_raw():
    with open("configs/feddqn_phase10.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def blank_item(**over):
    base = dict(run_id="t", matrix_type="core", scenario_group="clean_iid",
                attack_type="none", attack_fraction=0.0, attack_strength=0.0,
                unreliable_severity="none", unreliable_fraction=0.0,
                strategy="trust_threshold", master_seed=0, eval_seeds=[100], is_iid=True)
    base.update(over)
    return PlanItem(**base)


# ---------------------------------------------------------------- parameters
def test_core_params_resolve_to_fixed_values():
    raw = load_raw()
    resolved = resolve_detector_params(raw, blank_item())
    assert resolved == {
        "alpha": 0.50, "tau_exc": 0.35, "trust_lambda": 0.70,
        "m2_dir_threshold": 0.45, "m3_stale_threshold": 0.50,
        "m4_orig_threshold": 0.00, "tau_anomaly": 0.50,
    }
    assert CORE_DEFAULTS["rounds"] == 8 and CORE_DEFAULTS["num_clients"] == 10


def test_precedence_override_beats_yaml_beats_core():
    raw = load_raw()
    assert resolve_detector_params(raw, blank_item(alpha=0.10))["alpha"] == 0.10
    raw2 = dict(raw)
    raw2["detector"] = dict(raw["detector"], tau_anomaly=0.90)
    assert resolve_detector_params(raw2, blank_item())["tau_anomaly"] == 0.90
    raw3 = dict(raw)
    raw3["detector"] = {}
    raw3["federated"] = dict(raw["federated"])
    del raw3["federated"]["alpha"]
    assert resolve_detector_params(raw3, blank_item())["alpha"] == CORE_DEFAULTS["alpha"]


def test_server_receives_resolved_params():
    tmpl = DQNAgent(obs_dim=13, action_dim=4, hidden=[8], seed=0).q_net.state_dict()
    server = FedServer(tmpl, strategy="trust_threshold", alpha=0.50, tau_exc=0.35)
    assert server.alpha == 0.50 and server.tau_exc == 0.35 and server.strategy == "trust_threshold"


def test_plan_core_uses_fixed_values_and_ablation_values():
    plan = build_phase10_experiment_plan()
    core = [p for p in plan if p.matrix_type == "core"]
    assert all(p.alpha is None and p.tau_exc is None for p in core)
    abls = {(p.ablation_name): p for p in plan if p.matrix_type == "ablation"}
    assert abls["full_proposed"].alpha == 0.50 and abls["full_proposed"].tau_exc == 0.35
    assert abls["no_behavioural_trust"].alpha == 0.00
    assert abls["no_model_trust"].alpha == 1.00
    assert abls["no_temporal_smoothing"].lambda_smooth == 0.0
    no_thr = abls["no_threshold_exclusion"]
    assert no_thr.strategy == "trust_threshold" and no_thr.tau_exc == 0.00
    sens = [p for p in plan if p.matrix_type == "sensitivity"]
    for p in sens:
        if p.sweep_param == "alpha":
            assert p.alpha == p.param_value and p.tau_exc is None
        elif p.sweep_param == "tau_exc":
            assert p.tau_exc == p.param_value and p.alpha is None
        elif p.sweep_param == "lambda_smooth":
            assert p.lambda_smooth == p.param_value and p.alpha is None
        elif p.sweep_param == "m2_dir_threshold":
            assert p.m2_dir_threshold == p.param_value and p.alpha is None


def test_assert_plan_counts_and_rejection():
    plan = build_phase10_experiment_plan()
    counts = assert_plan_counts(plan)
    assert counts == {"core": 630, "ablation": 70, "sensitivity": 180, "total": 880}
    with pytest.raises(AssertionError):
        assert_plan_counts(plan[:-1])
    dup = plan + [plan[0]]
    with pytest.raises(AssertionError):
        assert_plan_counts(dup)


def test_end_to_end_core_resolution_records_effective_values(tmp_path):
    raw = load_raw()
    raw["_config_path"] = "configs/feddqn_phase10.yaml"
    raw["federated"] = dict(raw["federated"], rounds=1, num_clients=2)
    item = blank_item(run_id="v4_core_check", strategy="trust_threshold")
    res = execute_plan_item(item, raw, tmp_path)
    assert res["alpha"] == 0.50 and res["tau_exc"] == 0.35
    assert res["lambda_smooth"] == 0.70
    assert res["runtime_trust_lambda"] == 0.70
    assert res["effective_trust_lambda"] == 0.70
    assert res["m2_dir_threshold"] == 0.45 and res["tau_anomaly"] == 0.50


def test_end_to_end_ablation_lambda_zero(tmp_path):
    raw = load_raw()
    raw["_config_path"] = "configs/feddqn_phase10.yaml"
    raw["federated"] = dict(raw["federated"], rounds=1, num_clients=2)
    item = blank_item(run_id="v4_abl_check", strategy="trust_threshold",
                      matrix_type="ablation", ablation_name="no_temporal_smoothing",
                      lambda_smooth=0.0)
    res = execute_plan_item(item, raw, tmp_path)
    assert res["lambda_smooth"] == 0.0
    assert res["effective_trust_lambda"] == 0.0
    assert res["alpha"] == 0.50  # untouched params stay at core


# ------------------------------------------------------------------- wrapper
def test_wrapper_attributes_fault_to_true_destination():
    cfg = EnvConfig(seed=5, num_vehicles=4, num_edge_nodes=1, num_tasks=6, link_reliability=1.0)
    env = VehicularEdgeEnv(cfg)
    prof = UnreliableProfile(fail_prob=1.0, delay_prob=0.0, drop_prob=0.0, name="always_fail")
    w = UnreliableEnvWrapper(env, prof, seed=11)
    w.reset(0)
    for _ in range(6):  # action 0 = local: faulty record must target the requester vehicle
        obs, mask = w.current_observation()
        req_before = w.env.pending.requester_id
        res = w.step(0)
        assert res.failed and not res.success
        own_records = [r for r in w.env.trust.history
                       if r.task_id == res.task_id and r.event != "idle_age"]
        assert len(own_records) == 1, "exactly one observation per task"
        assert (own_records[0].kind, own_records[0].node_id) == ("vehicle", req_before)
        assert own_records[0].event == "deadline_miss" or own_records[0].s1 == 0.0
    assert w.get_fault_stats()["injected_fails"] == 6
    assert w.get_fault_stats()["tasks_with_any_fault"] == 6
    for v in w.env.trust.snapshot().values():
        assert np.isfinite(v)


def test_wrapper_precedence_overlap_and_union():
    cfg = EnvConfig(seed=6, num_vehicles=4, num_edge_nodes=1, num_tasks=8)
    env = VehicularEdgeEnv(cfg)
    prof = UnreliableProfile(fail_prob=1.0, delay_prob=1.0, drop_prob=1.0, name="all")
    w = UnreliableEnvWrapper(env, prof, seed=12)
    w.reset(0)
    for _ in range(8):
        obs, mask = w.current_observation()
        res = w.step(0)
        # Precedence drop > fail > delay: drop wins the outcome
        assert res.failed and not res.success and not res.deadline_missed
    st = w.get_fault_stats()
    assert st["tasks_exposed"] == 8 and st["tasks_with_any_fault"] == 8  # union, not sum
    assert st["injected_drops"] == 8 and st["injected_fails"] == 8 and st["injected_delays"] == 8


def test_wrapper_zero_exposure_counters():
    cfg = EnvConfig(seed=7, num_vehicles=4, num_edge_nodes=1, num_tasks=5)
    w = UnreliableEnvWrapper(VehicularEdgeEnv(cfg), UnreliableProfile(), seed=13)
    assert w.get_fault_stats() == {"tasks_exposed": 0, "tasks_with_any_fault": 0,
                                   "injected_drops": 0, "injected_fails": 0, "injected_delays": 0}


def test_wrapper_outcome_counter_consistency():
    cfg = EnvConfig(seed=8, num_vehicles=4, num_edge_nodes=1, num_tasks=10)
    env = VehicularEdgeEnv(cfg)
    w = UnreliableEnvWrapper(env, UnreliableProfile(fail_prob=1.0), seed=14)
    w.reset(0)
    n_fail = 0
    for _ in range(10):
        obs, mask = w.current_observation()
        res = w.step(0)
        n_fail += res.failed
    st = w.get_fault_stats()
    assert n_fail == 10 == st["injected_fails"] == st["tasks_with_any_fault"]


def test_retract_observation_restores_trust():
    from trust import BehaviouralTrustTable, Evidence
    t = BehaviouralTrustTable(2, 1)
    before = t.get("edge", 0)
    rec = t.observe(0, Evidence("edge", 0, True, True, 1.0, 5.0, 0.8, 0.8))
    assert t.get("edge", 0) != before
    out = t.retract_observation(0)
    assert out is rec
    assert t.get("edge", 0) == before
    assert all(r.event == "idle_age" or r.task_id != 0 for r in t.history)
    assert t.retract_observation(999) is None


def test_observe_rejects_nonfinite_evidence():
    from trust import BehaviouralTrustTable, Evidence
    t = BehaviouralTrustTable(2, 1)
    with pytest.raises(ValueError):
        t.observe(0, Evidence("edge", 0, False, False, 1.0, 5.0, float("nan"), 1.0))


# -------------------------------------------------------------------- stale
def test_create_stale_update_preserves_origin():
    from attacks.free_riding import create_stale_update
    hist = {"client_id": 0, "round_trained": 1, "num_samples": 10, "staleness": 0,
            "state": {"w": torch.tensor([1.0])}}
    out = create_stale_update(hist, current_round=5)
    assert out["round_trained"] == 1 and out["staleness"] == 4
    assert torch.equal(out["state"]["w"], hist["state"]["w"])


def test_stale_fallback_when_requested_age_unavailable():
    scen = ScenarioAssignment.create(num_clients=4, malicious_fraction=0.25,
                                     attack_type=AttackType.STALE_REPLAY, seed=123,
                                     attack_strength=10.0)
    mal_cid = scen.malicious_client_ids[0]
    pipe = AttackPipeline(scen, seed=123)
    ref = {"w": torch.tensor([1.0, 2.0])}
    for r in range(3):  # build history rounds 0..2 (target age 10 never matches)
        u = {"client_id": mal_cid, "round_trained": r, "num_samples": 10, "staleness": 0,
             "state": {"w": torch.tensor([1.0 + r, 2.0 + r])}}
        att, audit = pipe.transform_update(u, ref, current_round=r)
    # Simulate a history gap: drop the round-0 update, replay at round 3
    pipe.history_updates[mal_cid] = [h for h in pipe.history_updates[mal_cid]
                                     if h["round_trained"] != 0]
    u3 = {"client_id": mal_cid, "round_trained": 3, "num_samples": 10, "staleness": 0,
          "state": {"w": torch.tensor([9.0, 9.0])}}
    att3, audit3 = pipe.transform_update(u3, ref, current_round=3)
    assert audit3["attack_applied"]
    # Fallback = earliest available (round 1); actual age 3-1=2 reported, origin kept
    assert att3["round_trained"] == 1
    assert att3["staleness"] == 2
    assert audit3["origin_round"] == 1 and audit3["staleness_age"] == 2


# ------------------------------------------------------------------- stats
def test_holm_adjusted_pvalues_hand_computed():
    # Sorted: 0.01(rank0, x3), 0.04(rank1, x2), 0.05(rank2, x1).
    res = holm_adjust_pvalues([("a", 0.01), ("b", 0.04), ("c", 0.05)])
    assert res["a"]["holm_adjusted_pvalue"] == pytest.approx(0.03)
    assert res["b"]["holm_adjusted_pvalue"] == pytest.approx(0.08)
    assert res["c"]["holm_adjusted_pvalue"] == pytest.approx(0.08)  # monotone enforced
    assert res["a"]["holm_adjusted_alpha"] == pytest.approx(0.05 / 3)
    assert res["a"]["holm_decision"] == "reject"
    assert res["b"]["holm_decision"] == "fail_to_reject"
    assert res["c"]["holm_decision"] == "fail_to_reject"
    res2 = holm_adjust_pvalues([("x", 0.40), ("y", 0.50)])
    assert res2["x"]["holm_adjusted_pvalue"] == pytest.approx(0.80)
    assert all(v["holm_decision"] == "fail_to_reject" for v in res2.values())


def _write_minimal_csvs(tmpdir: Path):
    cols = ["run_id", "matrix_type", "scenario_group", "attack_type", "attack_fraction",
            "strategy", "seed", "alpha", "tau_exc", "completion_rate", "failure_rate",
            "miss_rate", "mean_latency_s", "mean_energy_j", "mean_reward", "exclusion_rate",
            "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn",
            "tasks_exposed", "injected_tasks_exposed", "tasks_with_any_fault",
            "injected_tasks_with_any_fault", "injected_drops", "injected_fails", "injected_delays",
            "lambda_smooth", "runtime_trust_lambda", "effective_trust_lambda"]
    core_rows = []
    for strat in ("fedavg", "trust_threshold"):
        for seed in (0, 1):
            core_rows.append(["c_%s_%d" % (strat, seed), "core", "freeriding_zero_update",
                              "zero_update", 0.40, strat, seed, 0.5, 0.35,
                              0.80, 0.0, 0.20, 3.0, 20.0, 0.5, 0.0,
                              1.0, 1.0, 1.0, 0.0, 8, 0, 2, 0,
                              0, 0, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7])
    for strat in ("fedavg", "trust_threshold"):
        for seed in (0, 1):
            for scen, frac in (("poisoning_sign_flip", 0.20), ("poisoning_sign_flip", 0.40),
                               ("freeriding_zero_update", 0.20)):
                core_rows.append([f"c_{strat}_{scen}_{frac}_{seed}", "core", scen,
                                  "x", frac, strat, seed, 0.5, 0.35,
                                  0.80, 0.0, 0.20, 3.0, 20.0, 0.5, 0.0,
                                  1.0, 1.0, 1.0, 0.0, 8, 0, 2, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7])
    pd.DataFrame(core_rows, columns=cols).to_csv(tmpdir / "phase10_experiment_runs.csv", index=False)
    abl_cols = cols + ["ablation_name"]
    abl_rows = [[f"a_{a}_{s}_{seed}", "ablation", "poisoning_sign_flip", "sign_flip", 0.20,
                 "trust_threshold", seed, 0.5, 0.35, 0.80, 0.0, 0.20, 3.0, 20.0, 0.5, 0.0,
                 1.0, 1.0, 1.0, 0.0, 8, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7, a]
                for a in ("full_proposed",) for seed in (0, 1) for s in range(1)]
    pd.DataFrame(abl_rows, columns=abl_cols).to_csv(tmpdir / "phase10_ablation_runs.csv", index=False)
    sens_cols = cols + ["sweep_param", "param_value"]
    sens_rows = [[f"s_{seed}", "sensitivity", "poisoning_sign_flip", "sign_flip", 0.20,
                  "trust_threshold", seed, 0.5, 0.35, 0.80, 0.0, 0.20, 3.0, 20.0, 0.5, 0.0,
                  1.0, 1.0, 1.0, 0.0, 8, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0.7, 0.7,
                  "alpha", 0.5] for seed in (0, 1)]
    pd.DataFrame(sens_rows, columns=sens_cols).to_csv(tmpdir / "phase10_sensitivity_runs.csv", index=False)


def test_zero_variance_reports_undefined_not_evidence(tmp_path):
    _write_minimal_csvs(tmp_path)
    rep = analyze_phase10_results(tmp_path, tmp_path / "summary.json")
    h1 = rep["primary_hypotheses"]["H1_zero_update_f40"]
    assert h1["status"] == "undefined_zero_variance"
    assert h1["t_statistic"] is None and h1["p_value_twotailed"] is None
    assert h1["cohen_dz"] is None
    assert h1["statistically_significant_alpha_005"] is False
    # Stored JSON must be strict-valid (no NaN literals)
    import json as _json
    raw_text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    parsed = _json.loads(raw_text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    assert parsed["primary_hypotheses"]["H1_zero_update_f40"]["p_value_twotailed"] is None
    for key in ("H_zero_f20", "H_signflip_f20", "H_signflip_f40"):
        sec = rep["secondary_hypotheses"][key]
        assert sec["status"] == "undefined_zero_variance"
        assert "holm_adjusted_pvalue" not in sec  # undefined tests excluded from Holm family


# ----------------------------------------------------------------- manifest
def _write_manifest_fixture(root: Path):
    import hashlib as _hl
    import json as _json
    from scripts.verify_phase10_manifest import summarize_matrices
    res = root / "res"
    res.mkdir(parents=True, exist_ok=True)
    _write_minimal_csvs(res)
    extra = res / "extra.pt"
    extra.write_bytes(b"model-bytes")
    matrices = summarize_matrices(res)
    manifest = {
        "artifacts": {
            "res/extra.pt": {"sha256": _hl.sha256(b"model-bytes").hexdigest(), "bytes": 11},
        },
        "matrix_counts": {k: matrices[k] for k in
                          ("core_rows", "ablation_rows", "sensitivity_rows", "total_rows")},
    }
    for name in ("phase10_experiment_runs.csv", "phase10_ablation_runs.csv", "phase10_sensitivity_runs.csv"):
        p = res / name
        manifest["artifacts"][f"res/{name}"] = {"sha256": _hl.sha256(p.read_bytes()).hexdigest(),
                                               "bytes": p.stat().st_size}
    man_path = root / "manifest.json"
    man_path.write_text(_json.dumps(manifest), encoding="utf-8")
    return man_path


def test_manifest_verify_pass_tamper_and_missing(tmp_path):
    from scripts.verify_phase10_manifest import check_manifest
    man_path = _write_manifest_fixture(tmp_path)
    assert check_manifest(man_path, repo_root=tmp_path, results_subdir="res") == []
    (tmp_path / "res" / "extra.pt").write_bytes(b"tampered!")
    failures = check_manifest(man_path, repo_root=tmp_path, results_subdir="res")
    assert any("hash mismatch" in f for f in failures)
    (tmp_path / "res" / "extra.pt").unlink()
    failures = check_manifest(man_path, repo_root=tmp_path, results_subdir="res")
    assert any("missing" in f for f in failures)


# ---------------------------------------------------------------- dashboard
def test_dashboard_v4_text_and_visibility():
    from dashboard import data_loader as _dl
    import inspect as _inspect
    src_loader = _inspect.getsource(_dl.get_available_datasets)
    assert "phase10_final_v4" in src_loader
    assert "630" in src_loader and "70" in src_loader and "180" in src_loader
    app_src = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "4 rounds) and historical" not in app_src
    assert "8-round" in app_src
    assert "0.35" in app_src  # v4 core exclusion threshold, not 0.50
    assert "no bounded-FPR guarantee is claimed" in app_src
    assert "Ablation Results" in app_src and "Sensitivity Results" in app_src
    comp_src = Path("dashboard/components.py").read_text(encoding="utf-8")
    assert "after four federated rounds" not in comp_src
