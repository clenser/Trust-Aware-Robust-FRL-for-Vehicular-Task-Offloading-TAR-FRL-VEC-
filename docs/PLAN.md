# Phase 0 — Repository Audit and Research Plan
# Trust-Aware Federated RL for Secure Task Offloading in VANETs

> Scope: Phase 0 only. No simulation, DQN, federated learning, trust, or
> dashboard code. This document is a plan, not a result.

## 0.1 Repository audit (verified 2026-10-11, UTC)

| Item | Actual finding |
|---|---|
| Working directory | `C:\Users\srbar\Desktop\VANNET` |
| Files present | `AGENTS.md`, `PHASE_PROMPTS.md` only |
| Git status | **Not a git repository** (`fatal: not a git repository`). No branch, log, or remote. No commits exist. |
| Python | **3.10.11** (`C:\Users\srbar\AppData\Local\Programs\Python\Python310\python.exe`). Below the `AGENTS.md` requirement of Python 3.11+. |
| Installed (relevant) | numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.8, torch 2.7.1+cu118, scipy 1.15.3, PyYAML 6.0.3, pytest 8.3.3 |
| Missing | **streamlit not installed** (confirmed `pip show streamlit` → not found). No project `requirements.txt` / `pyproject.toml` / `configs/` / source packages / tests existed before Phase 0. |
| Source files / tests | None existed. Nothing to regress. |

Evidence commands are listed in §0.6. No test, benchmark, or security result
is claimed beyond these directly observed outputs.

## 0.2 Binding scope restatement (from AGENTS.md)

1. **Synthetic Python simulation only.** Task offloading is modelled with
   abstract task descriptors (data size, CPU cycles, deadline, priority),
   mobility, connectivity range, CPU/bandwidth/queue heterogeneity.
2. **Explicitly out of scope for the minimum project:** SUMO, NS-3, OMNeT++,
   manual real-world dataset collection, training/downloading computer-vision
   or pretrained perception models.
3. **DQN is the required starting RL algorithm.** SAC, blockchain, digital
   twins, 6G, SDN, complex consensus, multi-agent architectures are not to be
   started unless Phase 1 literature review justifies them.
4. **Stack:** Python 3.11+, NumPy, Pandas, PyTorch, Matplotlib, pytest;
   custom federated-learning code first, Flower only if it materially helps;
   Streamlit dashboard kept independent from simulation/algorithm logic.
5. **Laptop/Colab runnable.** Small smoke configs vs. larger experiment configs.
   No secrets, keys, or paid services.
6. **Honesty rules:** never claim an unrun test/plot/benchmark/security
   guarantee; never fabricate citations, numbers, or novelty.

## 0.3 Research questions (from AGENTS.md §3, unchanged)

- **RQ1.** How can trust be dynamically estimated when participant behaviour,
  connectivity, and available resources change?
- **RQ2.** Does combining behavioural trust with model-update trust identify
  malicious participants better than behavioural trust alone?
- **RQ3.** Does trust-aware offloading improve task latency, completion, and reliability?
- **RQ4.** How does performance change as the proportion of malicious
  participants increases (0/10/20/30/40%)?
- **RQ5.** Does the method remain useful under mobility, intermittent
  connectivity, heterogeneous resources, and non-IID local experience?

## 0.4 Dependency & environment strategy

- **Python version gap:** audit machine has 3.10.11; project requires 3.11+.
  Action: document upgrade step (`py -3.11 -m venv .venv` or Colab runtime),
  do **not** silently downgrade the requirement. Phase 2+ code must target
  3.11+ syntax and be re-verified there.
- **Pinned minimums** live in `requirements.txt` (install floor) and
  `pyproject.toml` (project metadata + pytest config). Pins are deliberately
  lower-bound (`>=`) rather than exact, to stay laptop/Colab friendly; exact
  lockfile is deferred to Phase 11 reproducibility work.
- **Streamlit** is listed in `requirements.txt` but not installed in the audit
  env. It is only needed from Phase 9; installing it now is optional. Do not
  add Flower, SUMO, NS-3, OMNeT++, or vision-model dependencies without a
  Phase 1 justification.
- **Verification:** `pip list` output inspected (§0.1); `pytest -q` and
  `python scripts/validate_configs.py` are the Phase 0 checks.

## 0.5 Phased implementation plan (one phase at a time, gates per AGENTS.md §9)

| Phase | Deliverables | Gate (abridged) |
|---|---|---|
| **0 (this phase)** | Audit, plan, assumptions register, README outline, dependency strategy, config skeleton | Scope matches AGENTS.md; constraints acknowledged; no novelty claim |
| 1 | Literature workflow, paper-tracking table (real papers only), gap matrix, system/threat model, finite DQN spec, trust/reward equations, hypotheses | Every paper traceable; gap provisional; action space finite |
| 2 | Synthetic env: vehicle/edge/task/mobility/network/queue/execution, seeded generators, unit tests, CLI smoke | Same seed → repeatable; range/resource/deadline effects; no RL yet |
| 3 | Baselines (local/random/nearest/resource-aware), shared metrics, CSV/JSON export, tests | Valid actions incl. no-eligible-node; reproducible metrics |
| 4 | Single-agent DQN + replay + target net + epsilon-greedy, checkpoints, real learning curves | Shape/range tests; smoke run completes; no convergence claim without evidence |
| 5 | Behavioural trust + temporal update + eligibility/ranking integration, logs, tests | Bounds [0,1]; disconnect ≠ malicious |
| 6 | Federated DQN: local training, FedAvg coordinator, dropout/staleness handling | No raw replay transfer; global model updates |
| 7 | Model-update trust + combined trust + trust-weighted aggregation + threshold exclusion + fallback | Poison/free-rider fixtures; honest non-IID not auto-rejected; decisions logged |
| 8 | Attack/dynamic scenarios: poisoning, free-riding, unreliable exec, intermittent participation, non-IID, 0–40% sweeps | Reproducible, labelled ground truth |
| 9 | Streamlit dashboard on tested core; labelled demo vs. real data | Controls drive real params; charts from real outputs |
| 10 | Scripted matrix, multi-seed, ablations, sensitivity, CIs | Everything regenerable from saved config + raw results |
| 11 | Final report, architecture figure, pseudocode, reproducibility checklist | Every claim maps to code/source/measurement |

**Rule:** do not start Phase 1 until the user approves Phase 0.

## 0.6 Checks executed (actual outcomes)

1. `git status / branch / log / remote` → all fail with
   `fatal: not a git repository`. Outcome: no git history to preserve or regress.
2. `python --version` → `Python 3.10.11`; `pip list` filtered →
   numpy/pandas/matplotlib/pytest/PyYAML/scipy/torch present, streamlit absent.
3. `Get-ChildItem -Force -Recurse` → only `AGENTS.md`, `PHASE_PROMPTS.md`.
4. After scaffolding: `python scripts/validate_configs.py --configs configs`
   and `pytest -q` — see Phase 0 report in the agent's final message for exact
   outputs (not pre-filled here).

## 0.7 Risks & limitations

- Python version below requirement — must upgrade before Phase 2+.
- No git history — recommend `git init` + first commit after user reviews diff
  (not done silently in a way that hides changes; see manual checklist).
- No literature evidence yet — novelty hypothesis is explicitly unconfirmed.
- Config values in `configs/` are placeholder ranges for interface design, not
  tuned experimental choices; tuning belongs to Phases 1–2 and 10.
