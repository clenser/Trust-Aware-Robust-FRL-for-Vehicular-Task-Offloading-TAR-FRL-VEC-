# Trust-Aware Federated Reinforcement Learning for Secure Task Offloading in VANETs

> **Project stage: Phase 0 — Repository Audit and Research Plan.**
> This is a **synthetic Python simulation** of vehicular-edge task offloading.
> It is **not** a packet-level network simulator and does **not** require
> SUMO, NS-3, OMNeT++, real-world data collection, or pretrained vision models.

## Goal (summary)

Simulate vehicles offloading compute tasks (described by data size, CPU
cycles, deadline, priority) to neighbouring vehicles or roadside edge (RSU)
nodes. Learn offloading decisions with DQN (required starting algorithm),
track participant trust, train local RL agents, aggregate updates
federatively, and test resilience against malicious / unreliable participants.

Candidate contribution (hypothesis, **not yet proven**): a dynamic trust model
combining behavioural trust + model-update trust with trust-aware robust
aggregation:

```text
T_final(i,t) = alpha * T_behaviour(i,t) + (1 - alpha) * T_model(i,t),  all in [0, 1]
```

See `docs/PLAN.md` for the phased plan, `docs/ASSUMPTIONS.md` for the
assumptions register, and `AGENTS.md` for the binding project contract.

## Research questions

- RQ1: How can trust be dynamically estimated under changing behaviour, connectivity, resources?
- RQ2: Does behavioural + model-update trust detect malicious participants better than behavioural trust alone?
- RQ3: Does trust-aware offloading improve latency, completion, reliability?
- RQ4: How does performance degrade as malicious-client fraction grows (0/10/20/30/40%)?
- RQ5: Does the method hold under mobility, intermittent connectivity, heterogeneous resources, non-IID experience?

No novelty is claimed at this stage. The gap is phrased as a question pending
Phase 1 literature evidence.

## Repository layout (target, built incrementally)

```text
configs/        smoke.yaml, default.yaml, experiments.yaml
docs/           PLAN.md, ASSUMPTIONS.md
environment/    (Phase 2) vehicle/edge/task/mobility/network models
rl/             (Phase 4) DQN, replay buffer, reward
trust/          (Phase 5, 7) behavioural / model-update / temporal trust
federated/      (Phase 6, 7) clients, server, FedAvg, trust aggregation
attacks/        (Phase 8) poisoning, free-riding, unreliable nodes
baselines/      (Phase 3) local, random, nearest, resource-aware
experiments/    (Phase 10) runner, metrics, ablation, sensitivity
dashboard/      (Phase 9) Streamlit app (calls tested core, no duplicated logic)
results/        raw experiment outputs (git-ignored except .gitkeep)
reports/        figures/, tables/ regenerated from results
scripts/        CLI entry points
tests/          shared / smoke tests
```

Only the Phase 0 skeleton exists so far. Phase directories are created as
placeholders with `.gitkeep` and no algorithm code.

## Quick start (Phase 0)

```powershell
# 1. Create and activate a virtual environment (Python 3.11+ recommended)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate the config skeleton
python scripts\validate_configs.py --configs configs

# 4. Run the Phase 0 skeleton checks
pytest -q
```

> **Environment note (updated 2026-10-11):** Python 3.11.9 is installed and
> the project `.venv` runs on 3.11 (see validation-fix report). The system
> Python 3.10.11 remains present but is no longer used for validation.

## Configurations

| File                      | Purpose                                              |
|---------------------------|------------------------------------------------------|
| `configs/smoke.yaml`      | Tiny, fast smoke-test config (laptop / Colab safe)   |
| `configs/default.yaml`    | Default small-scale dev config                       |
| `configs/experiments.yaml`| Template matrix for Phase 10 sweeps (not executed)   |

All configs are validated by `scripts/validate_configs.py` and covered by
`tests/test_phase0_skeleton.py`. Random seeds are recorded in every config;
no experiment has been run yet.

## Status — progress so far (Phases 0–6 complete, Phase 7 needs approval)

- [x] **Phase 0** — audit, plan, assumptions register, README outline, config skeleton (`configs/`, `docs/PLAN.md`, `docs/ASSUMPTIONS.md`).
- [x] **Phase 1** — 40-paper verified literature DB (`reports/papers.csv`, `gap_matrix.md`), system-model math spec (`docs/PHASE1_SPEC.md`) and executable formula helpers (`specs/`) with tests.
- [x] **Phase 2** — reproducible synthetic VANET env (`environment/`, obs dim `n = 7 + 3K` verified, `scripts/run_smoke.py`); project `.venv` on Python 3.11.9.
- [x] **Phase 3** — four baselines (local/random/nearest/resource-aware) + multi-seed CSV/JSON evaluation (`scripts/run_baselines.py`).
- [x] **Phase 4** — single-agent DQN (`rl/`, train/eval scripts, `checkpoints/dqn_default/`); held-out 0.753±0.050, level with nearest_edge — no superiority claimed.
- [x] **Phase 5** — dynamic behavioural trust (`trust/`, EQ-12/13, disconnect-neutral ageing) wired into env obs; exploratory `trust_aware` variant; trust logs + plots.
- [x] **Phase 6** — federated DQN with standard FedAvg (`federated/` client/server, `scripts/train_feddqn.py`, `scripts/evaluate_feddqn.py`); held-out global 0.707±0.101 vs independent-local 0.738±0.080 — no superiority claimed.
- [ ] Phase 7: model-update trust and robust aggregation (needs approval)
- [ ] Phases 8–11: not started

### Smoke test (Phase 2)

```powershell
.venv\Scripts\python.exe scripts\run_smoke.py --config configs\smoke.yaml   # random-valid policy, 50 tasks
```

## Contributing / agent contract

All coding agents must read `AGENTS.md` and the active prompt in
`PHASE_PROMPTS.md`, work on **one phase only**, preserve unrelated files,
and never fabricate tests, results, citations, or security claims.
