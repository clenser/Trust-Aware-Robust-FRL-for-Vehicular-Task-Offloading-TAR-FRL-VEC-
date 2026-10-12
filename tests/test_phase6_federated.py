"""Phase 6 tests: FedAvg arithmetic/validation, client isolation, server rounds.

No trust weighting, attacks, or dashboard (later phases).
"""

import numpy as np
import pytest
import torch

from environment import EnvConfig
from federated import ALLOWED_UPDATE_KEYS, FedClient, FedServer, fedavg_aggregate, payload_bytes, validate_update
from rl import DQNAgent


def sd(vals: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in vals.items()}


def template() -> dict:
    return sd({"l1": [1.0, 2.0], "l2": [3.0]})


def test_fedavg_weighting_known_values():
    a = sd({"l1": [0.0, 0.0], "l2": [0.0]})
    b = sd({"l1": [2.0, 4.0], "l2": [6.0]})
    out = fedavg_aggregate([a, b], [1.0, 3.0], template(), ["c0", "c1"])
    assert out["l1"].tolist() == pytest.approx([1.5, 3.0])
    assert out["l2"].tolist() == pytest.approx([4.5])


def test_fedavg_unequal_counts_single_client():
    a = sd({"l1": [5.0, 5.0], "l2": [5.0]})
    out = fedavg_aggregate([a], [7.0], template(), ["c0"])
    assert out["l1"].tolist() == pytest.approx([5.0, 5.0])


def test_fedavg_rejects_malformed():
    ok = sd({"l1": [1.0, 2.0], "l2": [3.0]})
    with pytest.raises(ValueError):  # key mismatch
        fedavg_aggregate([sd({"l1": [1.0, 2.0]})], [1.0], template(), ["c0"])
    with pytest.raises(ValueError):  # shape mismatch
        fedavg_aggregate([sd({"l1": [1.0], "l2": [3.0]})], [1.0], template(), ["c0"])
    with pytest.raises(ValueError):  # non-finite
        fedavg_aggregate([sd({"l1": [float("nan"), 2.0], "l2": [3.0]})], [1.0], template(), ["c0"])
    with pytest.raises(ValueError):  # empty participation
        fedavg_aggregate([], [], template(), [])
    with pytest.raises(ValueError):  # zero total weight
        fedavg_aggregate([ok, ok], [0.0, 0.0], template(), ["c0", "c1"])
    with pytest.raises(ValueError):  # negative weight
        fedavg_aggregate([ok], [-1.0], template(), ["c0"])


def test_fedavg_deterministic():
    a = sd({"l1": [0.3, -1.2], "l2": [2.2]})
    b = sd({"l1": [1.1, 0.4], "l2": [-0.5]})
    o1 = fedavg_aggregate([a, b], [2.0, 5.0], template(), ["c0", "c1"])
    o2 = fedavg_aggregate([a, b], [2.0, 5.0], template(), ["c0", "c1"])
    for k in o1:
        assert torch.equal(o1[k], o2[k])


def test_payload_bytes_counts_params():
    assert payload_bytes(template()) == 3 * 4


def tiny_client(cid: int = 0, **over) -> FedClient:
    params = dict(seed=0, num_vehicles=3, num_edge_nodes=1, num_tasks=6)
    params.update(over)
    cfg = EnvConfig(**params)
    return FedClient(cid, cfg, dict(hidden=[8, 8], batch_size=2, capacity=50), seed=cid)


def test_client_update_contains_no_raw_experience():
    c = tiny_client()
    c.local_train(1, base_seed=100, num_tasks=6, warmup_steps=100, sync_every=10)
    u = c.make_update(round_trained=0)
    assert set(u.keys()) == ALLOWED_UPDATE_KEYS
    assert u["num_samples"] == 6 and u["round_trained"] == 0 and u["staleness"] == 0
    assert all(isinstance(v, torch.Tensor) for v in u["state"].values())  # params only


def test_client_buffers_isolated_and_local():
    a, b = tiny_client(0), tiny_client(1)
    a.local_train(1, base_seed=100, num_tasks=6, warmup_steps=100, sync_every=10)
    assert len(a.agent.buffer) == 6 and len(b.agent.buffer) == 0
    assert a.agent.buffer is not b.agent.buffer


def test_install_global_sets_online_and_target():
    c = tiny_client()
    ref = DQNAgent(obs_dim=c.env_cfg.obs_dim, action_dim=c.env_cfg.action_dim,
                   hidden=[8, 8], batch_size=2, capacity=50, seed=99)
    c.install_global(ref.q_net.state_dict())
    for p, q in zip(c.agent.q_net.parameters(), ref.q_net.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(c.agent.target_net.parameters(), ref.q_net.parameters()):
        assert torch.equal(p, q)


def test_server_selection_dropout_and_skip():
    tmpl = DQNAgent(obs_dim=13, action_dim=4, hidden=[8], seed=0).q_net.state_dict()
    server = FedServer(tmpl)
    rng = np.random.default_rng(0)
    sel = server.select_clients([0, 1, 2, 3], 0.5, rng)
    assert len(sel) == 2 and sel == sorted(sel)
    assert server.select_clients([0, 1, 2], 0.1, rng)  # at least one
    log = server.aggregate_round([], current_round=0)  # zero participation
    assert log["status"].startswith("skipped")
    for k in server.get_global():  # last good model preserved exactly
        assert torch.equal(server.get_global()[k], server.template[k])
    with pytest.raises(ValueError):
        server.select_clients([0, 1], 0.0, rng)


def test_stale_update_accepted_and_logged():
    c = tiny_client()
    tmpl = DQNAgent(obs_dim=c.env_cfg.obs_dim, action_dim=c.env_cfg.action_dim,
                    hidden=[8, 8], batch_size=2, capacity=50, seed=5).q_net.state_dict()
    server = FedServer(tmpl)
    c.local_train(1, base_seed=100, num_tasks=6, warmup_steps=100, sync_every=10)
    u = c.make_update(round_trained=0)
    log = server.aggregate_round([u], current_round=2)
    assert log["status"] == "aggregated" and log["staleness"] == [2]  # accepted, age recorded
    assert log["upload_bytes"] > 0 and log["download_bytes_per_client"] > 0


def test_global_changes_and_distributes():
    c0, c1 = tiny_client(0), tiny_client(1)
    tmpl = DQNAgent(obs_dim=c0.env_cfg.obs_dim, action_dim=c0.env_cfg.action_dim,
                    hidden=[8, 8], batch_size=2, capacity=50, seed=5).q_net.state_dict()
    server = FedServer(tmpl)
    before = [v.clone() for v in server.get_global().values()]
    for c in (c0, c1):
        c.local_train(1, base_seed=200 + c.client_id, num_tasks=6, warmup_steps=0, sync_every=5)
    server.aggregate_round([c0.make_update(0), c1.make_update(0)], current_round=0)
    after = list(server.get_global().values())
    assert any(not torch.equal(b, a) for a, b in zip(before, after))  # global moved
    for c in (c0, c1):  # distribution reaches clients
        c.install_global(server.get_global())
        for p, q in zip(c.agent.q_net.parameters(), server.get_global().values()):
            assert torch.equal(p, q)


def test_federated_rounds_reproducible():
    def run():
        rng = np.random.default_rng(3)
        clients = [tiny_client(i) for i in range(2)]
        tmpl = DQNAgent(obs_dim=clients[0].env_cfg.obs_dim, action_dim=clients[0].env_cfg.action_dim,
                        hidden=[8, 8], batch_size=2, capacity=50, seed=5).q_net.state_dict()
        server = FedServer(tmpl)
        for c in clients:
            c.install_global(server.get_global())
        for r in range(2):
            sel = server.select_clients([0, 1], 1.0, rng)
            ups = []
            for c in clients:
                if c.client_id in sel:
                    c.local_train(1, base_seed=300 + r * 10 + c.client_id, num_tasks=6,
                                  warmup_steps=0, sync_every=5)
                    ups.append(c.make_update(r))
            server.aggregate_round(ups, current_round=r)
            for c in clients:
                c.install_global(server.get_global())
        return server.get_global()
    g1, g2 = run(), run()
    for k in g1:
        assert torch.equal(g1[k], g2[k])


def test_trust_preserved_and_unweighted():
    c = tiny_client()
    c.local_train(1, base_seed=100, num_tasks=6, warmup_steps=100, sync_every=10)
    summary = c.trust_summary()
    assert set(summary) == {"mean_trust", "min_trust", "max_trust", "nodes"}
    assert all(0.0 <= summary[k] <= 1.0 for k in ("mean_trust", "min_trust", "max_trust"))
    u = c.make_update(0)  # update carries no trust weight/exclusion signal
    assert "trust" not in str(list(u.keys())).lower()
