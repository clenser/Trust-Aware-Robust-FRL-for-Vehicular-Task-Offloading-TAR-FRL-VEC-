# Phase 6 — Federated DQN with standard FedAvg

> Single-process **synthetic simulation** of federated learning, not deployed
> V2X communication. No trust weighting, no attack handling (Phase 7/8).

## Architecture

```text
FedServer (server.py)      global state, seeded selection, FedAvg, round log
FedClient (client.py)      private VehicularEdgeEnv + private DQNAgent/buffer
fedavg_aggregate (fedavg.py)  EQ-21 sample-count weighting + strict validation
```

Round workflow: select (seeded subset, ≥1) → seeded dropout → local training
on private experience → updates `{client_id, round_trained, num_samples,
staleness, state}` (parameters + scalars ONLY) → validate → FedAvg →
distribute to **all** clients (online AND target nets reload global; local
target-sync schedule continues in-round). Empty round → global unchanged,
logged `skipped`.

## Key definitions

- `n_i` (EQ-21 weight) = valid local transitions collected by client `i` **this round**.
- Staleness = `current_round − round_trained`; accepted and logged, never
  down-weighted (Phase 7 material). Malformed updates raise; zero total weight
  raises; both tested.
- Behavioural trust is logged per client/round (`mean_trust`) for analysis
  only — it never touches weights or participation.
- Heterogeneity is synthetic per-client overrides (`task_load`, `mobility`,
  `link_reliability`) + per-client seeds, all recorded in `run_meta.json`.

## Communication overhead (analytical estimates, NOT measurements)

Per transfer payload = 4 bytes × parameter count (`payload_bytes`). Logged per
round: total upload bytes + download bytes per client. Aggregation wall time is
measured local compute, not network time.

## Commands

```powershell
.venv\Scripts\python.exe scripts\train_feddqn.py --config configs\feddqn_smoke.yaml
.venv\Scripts\python.exe scripts\train_feddqn.py --config configs\feddqn_default.yaml
.venv\Scripts\python.exe scripts\evaluate_feddqn.py --config configs\feddqn_default.yaml
```
