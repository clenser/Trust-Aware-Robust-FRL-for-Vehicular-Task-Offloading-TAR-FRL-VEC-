"""Data-plane attacks, V2V peer selection, edge calibration and trace integrity."""

import copy
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from configs.config_loader import load_config
from environment.vanet_env import VANETOffloadingEnv
from environment.edge_calibration import EdgeCalibration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _env(**mods):
    np.random.seed(7)
    cfg = copy.deepcopy(load_config())
    for sec, kv in mods.items():
        cfg.setdefault(sec, {})
        for k, v in kv.items():
            if isinstance(v, dict):
                cfg[sec].setdefault(k, {}).update(v)
            else:
                cfg[sec][k] = v
    return VANETOffloadingEnv(config=cfg, num_vehicles=25, attack_ratio=0.3, random_seed=7)


def _v2v_stats(env, steps=120):
    sd = env._get_global_state_dict()
    stats = {}
    for _ in range(steps):
        env.update_positions()
        for vid in env.vehicle_order():
            s, task = sd[vid]
            ns, r, d, info, nt = env.step_vehicle(vid, 4, task)
            sd[vid] = (ns, nt)
            if info["target"].startswith("V"):
                role = env.vehicles[int(info["target"][1:])].role
                a = stats.setdefault(role, [0, 0])
                a[0] += 1
                a[1] += int(info["success"])
    return stats


class TestDataPlaneAttack(unittest.TestCase):
    def test_default_attackers_are_honest_helpers(self):
        env = _env()
        self.assertFalse(env.data_plane_enabled)
        self.assertTrue(all(v.helper_drop_prob is None for v in env.vehicles.values()))

    def test_enabled_attackers_sabotage_helper_tasks(self):
        env = _env(adversarial={"data_plane": {"enabled": True}})
        stats = _v2v_stats(env)
        honest = stats["honest"][1] / stats["honest"][0]
        poison = stats["poisoning"][1] / max(1, stats["poisoning"][0])
        self.assertLess(poison, honest * 0.5)
        self.assertGreater(sum(v.helper_tasks_sabotaged for v in env.vehicles.values()), 0)

    def test_inactive_vehicle_is_never_a_peer(self):
        env = _env()
        v = env.vehicles[1]
        for p in env.vehicles.values():
            p.active = False
        self.assertEqual(env.select_v2v_peer(v), (None, 999.0))

    def test_trust_aware_selection_avoids_a_distrusted_peer(self):
        env = _env(network={"v2v_peer_selection": "trust_aware"})
        v = next(x for x in env.vehicles.values() if len(env._v2v_candidates(x)) >= 2)
        cands = env._v2v_candidates(v)
        bad = min(cands, key=lambda c: c[1])[0]            # the nearest peer
        rec = env.trust_engine.behavior_tracker.node_records["V%d" % bad.vehicle_id]
        rec["trust_score"] = 0.0
        rec["prior_successes"], rec["prior_total"] = 0.0, 50.0
        peer, _ = env.select_v2v_peer(v)
        self.assertNotEqual(peer.vehicle_id, bad.vehicle_id)

    def test_invalid_selection_rejected(self):
        with self.assertRaises(ValueError):
            _env(network={"v2v_peer_selection": "bogus"})


class TestComputeRichHelpers(unittest.TestCase):
    def test_off_by_default_and_on_when_asked(self):
        self.assertTrue(all(v.cpu_capacity_ghz <= 2.0 for v in _env().vehicles.values()))
        env = _env(mobility={"vehicle_compute": {"rich_fraction": 0.4, "rich_cpu_ghz": [4.0, 8.0]}})
        rich = [v for v in env.vehicles.values() if v.cpu_capacity_ghz >= 4.0]
        self.assertEqual(len(rich), 10)


class TestEdgeCalibration(unittest.TestCase):
    def test_inert_by_default(self):
        env = _env()
        self.assertIsNone(env.edge_cal)
        self.assertTrue(all(r.background_share == 0.0 for r in env.rsus))

    def test_enabled_loads_rsus_with_diurnal_profile(self):
        env = _env(edge_calibration={"enabled": True, "start_hour": 5.0})
        trough = [r.background_share for r in env.rsus]
        env2 = _env(edge_calibration={"enabled": True, "start_hour": 20.0})
        peak = [r.background_share for r in env2.rsus]
        self.assertTrue(all(0 < t < p for t, p in zip(trough, peak)))

    def test_background_load_slows_service(self):
        env = _env()
        r = env.rsus[0]
        task = env.task_generator.generate_task(1, 0.0)
        base, _ = r._compute_service_time(task, env.channel)
        r.background_share = 0.5
        loaded, _ = r._compute_service_time(task, env.channel)
        self.assertAlmostEqual(loaded, base * 2.0, places=6)

    def test_summary_is_committed_and_scoped(self):
        with open(os.path.join(ROOT, "data", "traces", "edge_platform_summary.json"), encoding="utf-8") as fh:
            s = json.load(fh)
        self.assertIn("NO vehicular", s["scope"])
        self.assertEqual(len(s["bandwidth_three_months"]["normalised_load_by_hour"]), 24)
        self.assertTrue(all(np.isfinite(s["bandwidth_three_months"]["normalised_load_by_hour"])))

    def test_missing_summary_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            EdgeCalibration.from_config({"edge_calibration": {"enabled": True, "summary_path": "nope.json"}})


class TestTraceIntegrity(unittest.TestCase):
    def test_modified_trace_is_rejected(self):
        from environment.mobility import TraceDrivenMobility
        src = os.path.join(ROOT, "data", "synthetic_traces", "synthetic_ngsim_highway.csv")
        tmp = tempfile.mkdtemp()
        try:
            dst = os.path.join(tmp, "t.csv")
            shutil.copy(src, dst)
            with open(os.path.join(tmp, "t.provenance.json"), "w") as fh:
                json.dump({"output": {"sha256": "0" * 64}}, fh)
            with self.assertRaises(ValueError):
                TraceDrivenMobility(trace_file=dst)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
