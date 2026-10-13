# Trust-Aware Federated Reinforcement Learning for Secure Task Offloading in VANETs

> **Project stage: Phase 11 — Final Research Report, Reproducibility, and Project Delivery Complete.**
> This is a **synthetic Python simulation** of vehicular-edge task offloading.
> It is **not** a packet-level network simulator and does **not** require
> SUMO, NS-3, OMNeT++, real-world data collection, or pretrained vision models.

---

## 1. Project Goal & Research Summary

Simulate vehicles offloading computationally intensive tasks (defined by data size, CPU cycles, deadline, and priority) to neighboring vehicles (V2V) or roadside edge servers (V2I/RSU). Decentralized offloading decisions are learned using local Deep Q-Networks (DQN). The coordinator aggregates model updates federatively while maintaining a dual-tier dynamic trust model to defend against adversarial participants and execution unreliability:

```text
T_final(i, t) = alpha * T_behaviour(i, t) + (1 - alpha) * T_model(i, t),   all in [0, 1]
```

- **Behavioural Trust ($T_{\text{beh}}$):** Tracks peer-observed task success, response timeliness, advertised resource match, and communication reliability, augmented by neutral disconnect decay under mobility-induced channel dropouts.
- **Model-Update Trust ($T_{\text{model}}$):** Evaluates parameter norm plausibility, directional cosine consensus agreement, update staleness, and identical/zero free-riding without requiring a server root dataset.
- **Robust Aggregation:** Trust-weighted averaging with an exclusion threshold ($\tau_{\text{exc}} = 0.35$) and defensive fallback protocol.

---

## 2. Key Scientific Findings & Honest Disclosures

Across an 880-run experimental matrix (630 core, 70 ablation, 180 sensitivity sweeps across 10 random seeds) evaluated over 8 federated rounds:
1. **Primary Task Completion Comparison ($f = 0.40$ Zero-Update Free-Riding):**
   - FedAvg task completion: $0.748 \pm 0.052$
   - Trust-Threshold task completion: $0.754 \pm 0.046$
   - Mean paired difference: $+0.0060$ (95% CI: $[-0.0193, +0.0313]$)
   - Paired Student's $t$-test: $t(9) = 0.537, p = 0.604$ (Two-tailed, **fail to reject null hypothesis**).
   - *Scientific Interpretation:* Under an 8-round federated horizon, the trust mechanism does not confer statistically significant task completion superiority over FedAvg.
2. **Adversary Detection and Isolation:**
   - Free-riding zero updates are completely detected and isolated: $F_1 = 1.000, \text{FPR} = 0.000$.
   - Honest non-IID participants are never falsely penalized: $\text{FPR} = 0.000$ across all seeds.
   - The primary research contribution is **Byzantine adversary isolation and model aggregation protection** without task offloading degradation.
3. **Simulation Scope:**
   - The environment is a discrete-time synthetic Python simulation; results are not claimed as field-validated on physical vehicles or packet-level simulators.

---

## 3. Quick Start & Reproducibility Guide

### Step 1: Environment Setup (Python 3.11+)
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Smoke Test (Phase 2 Environment)
```powershell
python scripts/run_smoke.py --config configs/smoke.yaml
```
*Expected result:* 50 synthetic offloading tasks execute cleanly under a random-valid policy with zero errors.

### Step 3: Run Full Test Suite (178 Tests)
```powershell
pytest -q
```
*Expected result:* `178 passed in ~14s` with exit code `0`.

### Step 4: Verify Experimental Artifact Manifest (894 Artifacts)
```powershell
python scripts/verify_phase10_manifest.py --check
```
*Expected result:* `MANIFEST CHECK PASSED: 894 artifacts re-hashed, row counts stable.` (validates SHA-256 hashes for all 880 PyTorch checkpoints, 3 CSVs, 1 summary JSON, 6 figures, and 4 tables).

### Step 5: Launch Streamlit Research Dashboard
```powershell
streamlit run dashboard/app.py
```
*Expected result:* Web browser opens at `http://localhost:8501`. Default dataset is `Phase 10 Final v4 (Full 880-Run Corrected Suite)`.

---

## 4. Key Artifacts & Documentation Index

| Deliverable | File Path | Description |
|---|---|---|
| **Final Research Report** | [`reports/FINAL_RESEARCH_REPORT.md`](reports/FINAL_RESEARCH_REPORT.md) | Comprehensive 14-section research paper with architecture, pseudocode, results, and discussion |
| **SUMO Mobility Spec** | [`docs/SUMO_MICROSCOPIC_MOBILITY_SPEC.md`](docs/SUMO_MICROSCOPIC_MOBILITY_SPEC.md) | Microscopic traffic simulation architecture, TraCI integration, and RSU handover model |
| **Phase 10 Closeout Report** | [`reports/PHASE10_FINAL_V4_CLOSEOUT.md`](reports/PHASE10_FINAL_V4_CLOSEOUT.md) | Final verification audit and gate compliance report |
| **Run Manifest** | [`reports/PHASE10_FINAL_V4_RUN_MANIFEST.json`](reports/PHASE10_FINAL_V4_RUN_MANIFEST.json) | SHA-256 cryptographic hashes for all 894 experimental outputs |
| **Summary Metrics JSON** | [`results/phase10_final_v4/phase10_summary_metrics.json`](results/phase10_final_v4/phase10_summary_metrics.json) | Machine-readable metrics, hypothesis tests, and Holm-Bonferroni corrections |
| **Core Experiment CSV** | [`results/phase10_final_v4/phase10_experiment_runs.csv`](results/phase10_final_v4/phase10_experiment_runs.csv) | 630 core scenario runs across 10 random seeds |
| **Ablation CSV** | [`results/phase10_final_v4/phase10_ablation_runs.csv`](results/phase10_final_v4/phase10_ablation_runs.csv) | 70 runs across 7 ablation configurations |
| **Sensitivity CSV** | [`results/phase10_final_v4/phase10_sensitivity_runs.csv`](results/phase10_final_v4/phase10_sensitivity_runs.csv) | 180 hyperparameter sweep runs |
| **Publication Figures** | [`reports/figures/phase10_final_v4/`](reports/figures/phase10_final_v4/) | 6 publication-ready PNG charts generated directly from v4 CSVs |
| **LaTeX Tables** | [`reports/tables/phase10_final_v4/`](reports/tables/phase10_final_v4/) | 4 LaTeX formatted tables for academic reporting |
| **Literature Database** | [`reports/papers.csv`](reports/papers.csv) & [`reports/papers.md`](reports/papers.md) | 40 verified literature references with OpenAlex/arXiv verification records |
| **Research Gap Matrix** | [`reports/gap_matrix.md`](reports/gap_matrix.md) | Systematic coverage matrix identifying specific research gaps G1–G6 |

---

## 5. Repository Layout

```text
trust-aware-vanet-frl/
  configs/         smoke.yaml, default.yaml, feddqn_phase10.yaml
  docs/            PLAN.md, ASSUMPTIONS.md, PHASE1_SPEC.md, SUMO_MICROSCOPIC_MOBILITY_SPEC.md
  environment/     vehicle.py, edge_node.py, task.py, mobility.py, network.py, vanet_env.py, sumo_adapter.py, sumo_env.py, sumo/
  rl/              dqn.py, replay_buffer.py, agent.py, reward.py
  trust/           behavioural.py, model_update.py, temporal.py
  federated/       client.py, server.py, fedavg.py, trust_aggregation.py
  attacks/         poisoning.py, free_riding.py, unreliable_node.py
  baselines/       local.py, random_offload.py, nearest_node.py, resource_aware.py
  experiments/     runner.py, metrics.py, ablation.py, sensitivity.py
  dashboard/       app.py, components.py, simulator_engine.py, three_scene.py, data_loader.py, styles.py
  results/         phase10_final_v4/ (880 checkpoints, 3 CSVs, summary JSON), sumo_smoke/
  reports/         FINAL_RESEARCH_REPORT.md, figures/, tables/, manifest
  scripts/         CLI entry points (run_smoke.py, run_sumo_smoke_experiment.py, verify_phase10_manifest.py, etc.)
  tests/           178 unit and integration tests
```

---

## 6. Project Phased Status Checklist (Phases 0–11 Complete)

- [x] **Phase 0** — audit, plan, assumptions register, README outline, config skeleton (`configs/`, `docs/PLAN.md`, `docs/ASSUMPTIONS.md`).
- [x] **Phase 1** — 40-paper verified literature DB (`reports/papers.csv`, `gap_matrix.md`), system-model math spec (`docs/PHASE1_SPEC.md`) and executable formula helpers (`specs/`) with tests.
- [x] **Phase 2** — reproducible synthetic VANET env (`environment/`, obs dim $n = 7 + 3K$ verified, `scripts/run_smoke.py`); project `.venv` on Python 3.11.9.
- [x] **Phase 3** — four baselines (local/random/nearest/resource-aware) + multi-seed CSV/JSON evaluation (`scripts/run_baselines.py`).
- [x] **Phase 4** — single-agent DQN (`rl/`, train/eval scripts, `checkpoints/dqn_default/`); held-out $0.753 \pm 0.050$, level with nearest edge.
- [x] **Phase 5** — dynamic behavioural trust (`trust/`, EQ-12/13, disconnect-neutral ageing) wired into env obs; exploratory `trust_aware` variant; trust logs + plots.
- [x] **Phase 6** — federated DQN with standard FedAvg (`federated/` client/server, `scripts/train_feddqn.py`, `scripts/evaluate_feddqn.py`).
- [x] **Phase 7** — model-update trust (EQ-14–EQ-20, `trust/model_update.py`) + trust-weighted FedAvg with threshold exclusion and safe fallback (`federated/trust_aggregation.py`, EQ-22/23).
- [x] **Phase 8** — adversarial pipeline and dynamic scenarios: sign-flip, perturbation, zero/copy/stale free-riding, and unreliable execution wrappers (`attacks/`).
- [x] **Phase 9** — interactive Streamlit research dashboard (`dashboard/app.py`, `dashboard/components.py`, `dashboard/data_loader.py`).
- [x] **Phase 10** — complete evaluation suite: 880-run matrix (630 core + 70 ablation + 180 sensitivity sweeps), statistical hypothesis testing, Holm-Bonferroni corrections, publication figures, and 894-artifact SHA-256 verified run manifest (`results/phase10_final_v4/`, `reports/PHASE10_FINAL_V4_*`).
- [x] **Phase 11** — final comprehensive research report, architecture diagram, algorithm pseudocode, reproducibility guidelines, and project closeout (`reports/FINAL_RESEARCH_REPORT.md`, `README.md`).

---

## 7. Troubleshooting & Common Setup Notes

1. **PowerShell Execution Policy:** If `Activate.ps1` is blocked, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
2. **Port Conflict on Dashboard Startup:** If port `8501` is occupied:
   ```powershell
   streamlit run dashboard/app.py --server.port 8505
   ```
3. **Reproducibility Verification Execution:** Ensure commands are invoked using the virtual environment interpreter (`.venv\Scripts\python.exe` on Windows).
