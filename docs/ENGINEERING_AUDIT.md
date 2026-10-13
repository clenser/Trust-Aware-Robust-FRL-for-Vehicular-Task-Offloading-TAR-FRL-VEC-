# ENGINEERING AUDIT & ROOT-CAUSE DIAGNOSIS
**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions (TAR-FRL-VEC)  
**Date:** October 2026  
**Auditor:** Lead Research & QA Systems Engineer  
**Workspace:** `C:\Users\srbar\Desktop\VANNET`  
**Git Branch:** `main` | **HEAD Commit:** `d87e99d`  

---

## 1. Executive Summary & Baseline Integrity

This engineering audit assesses the technical state of the TAR-FRL-VEC codebase prior to executing the Master Engineering Mission. The repository combines synthetic edge-computing simulation, Eclipse SUMO microscopic traffic modeling via TraCI, single-agent and federated Deep Q-Networks (DQN), multi-faceted trust modeling, adversarial robust aggregation, and an interactive Streamlit/Three.js dashboard.

### 1.1 Baseline Verification Results
* **Git Status:** Working tree clean on branch `main` at commit `d87e99d`. (User-managed git publishing strictly observed).
* **Historical Benchmark Manifest (Phase 10 v4):** Verified with `python scripts/verify_phase10_manifest.py --check`.
  * **Result:** `MANIFEST CHECK PASSED: 894 artifacts re-hashed, row counts stable.` All 880 experimental runs and 894 artifacts in `results/phase10_final_v4/` remain completely intact and read-only.
* **Pytest Test Suite:** Executed `pytest -q`.
  * **Result:** `178 passed in 14.12s` (100% pass rate).
* **Python Runtime:** Python 3.11.9 (64-bit AMD64) in `.venv`.
* **Eclipse SUMO & TraCI Environment:**
  * SUMO Binary: `C:\Users\srbar\AppData\Local\sumo-1.28.0\sumo-win64-1.28.0\bin\sumo.exe` (Version 1.28.0)
  * TraCI Library: Version 1.28.0 (`traci` and `sumolib` available)
  * SUMO Scenario Assets: `environment/sumo/highway.net.xml`, `highway_low.rou.xml`, `highway_medium.rou.xml`, `highway_high.rou.xml`, `highway.sumocfg`.

---

## 2. Codebase Architecture & Inventory

The repository is modularized into distinct functional domains:

```
trust-aware-vanet-frl/
├── environment/           # Vehicular edge network simulation & mobility
│   ├── config.py          # EnvConfig (network, task, compute, trust hyperparameters)
│   ├── vanet_env.py       # Core synthetic 1D-torus VehicularEdgeEnv
│   ├── sumo_adapter.py    # TraCI process management & SumoTrafficSimulator
│   ├── sumo_env.py        # SumoVehicularEdgeEnv (bridges TraCI physics with VEC tasks)
│   └── sumo/              # SUMO scenario definitions (1200m 3-lane highway network)
├── baselines/             # Decision heuristics (local_only, random_valid, nearest_edge, resource_aware)
│   ├── policies.py        # Policy dispatchers & candidate scoring functions
│   └── metrics.py         # Standardized episode metric aggregation
├── rl/                    # Reinforcement learning components
│   ├── agent.py           # Single-agent DQNAgent (PyTorch)
│   ├── replay_buffer.py   # Experience replay buffer
│   └── reward.py          # Multi-objective reward computation (Eq. 11)
├── trust/                 # Dual-layer trust management
│   ├── behavioural.py     # Task outcome tracking & Bayesian beta updating
│   ├── model_update.py    # Update deviation, norm bounds, consistency checks
│   └── temporal.py        # Temporal smoothing & exponential decay
├── federated/             # Federated learning framework
│   ├── client.py          # Simulated vehicular FL client (local gradient steps)
│   ├── server.py          # Central coordinator & aggregation engine
│   └── aggregation.py     # FedAvg, Trust-Weighted, Trust-Threshold, Median, Trimmed-Mean
├── attacks/               # Adversarial attack implementations
│   ├── poisoning.py       # Sign-flipping, Gaussian perturbation, targeted weight shifts
│   └── free_riding.py     # Zero updates, stale parameter replay, identical weight replay
├── dashboard/             # Interactive Streamlit & Three.js visualization
│   ├── app.py             # Streamlit application entry point
│   ├── simulator_engine.py# Episode trace generator & step snapshot builder
│   ├── three_scene.py     # Three.js 3D daytime highway visualizer & HUD
│   └── components.py      # Reusable UI metric cards & tables
├── scripts/               # Reproducible experiment runners & verifiers
├── tests/                 # 178 unit and integration tests across all modules
├── results/               # Results storage (Phase 10 historical frozen; smoke separate)
└── docs/                  # Specifications, assumptions, and engineering protocols
```

---

## 3. Root-Cause Defect Diagnoses

### Defect 1: Identical Policy Metrics in SUMO Smoke Experiment
* **Observed Symptom:** In `results/sumo_smoke/sumo_smoke_summary.csv`, policies `resource_aware`, `nearest_edge`, and `local_only` yielded identical metrics:
  * Success Rate: 55.0%
  * Mean Latency: 11.083 s
  * Mean Energy: 36.887 J
  * Offload Fraction: 0.000
* **Root-Cause Analysis:**
  1. In `baselines/policies.py`, heuristic policies conform to the protocol:
     ```python
     def nearest_edge(obs: list[float], mask: list[bool], ctx: dict | None = None) -> int:
         cands = (ctx or {}).get("candidates", [])
         ...
         if best_pos < 0: return 0
     ```
     `resource_aware` similarly checks:
     ```python
     def resource_aware(obs: list[float], mask: list[bool], ctx: dict | None = None) -> int:
         ctx = ctx or {}
         cands = ctx.get("candidates", [])
         env = ctx.get("env")
         if not cands or env is None: return 0
     ```
  2. In `scripts/run_baselines.py`, `ctx` was properly supplied (`ctx = {"candidates": env.candidates(), "env": env}`).
  3. However, in `dashboard/simulator_engine.py` (lines 239–244):
     ```python
     obs, mask = env.current_observation()
     try:
         action = int(policy(obs, mask))  # <-- MISSING ctx={"candidates": cands, "env": env}
     except Exception:
         action = 0
     ```
     Because `ctx` was omitted, `cands` was empty, forcing `nearest_edge` and `resource_aware` to return `action = 0` (local execution).
  4. Furthermore, `load_simulation_policy` in `simulator_engine.py` returned closures for trained neural agents with strict 2-parameter signatures `(lambda obs, mask: agent.act(obs, mask))`, which would raise a `TypeError` if a 3-argument call was made without parameter compatibility.
* **Resolution Plan:**
  - Update `dashboard/simulator_engine.py` line 240 to pass `ctx = {"candidates": cands, "env": env}`.
  - Implement adaptive dispatch: try calling `policy(obs, mask, ctx)` first; if a `TypeError` is raised, fallback to `policy(obs, mask)`.
  - Update all agent wrappers in `load_simulation_policy` to accept optional `ctx=None`.

---

### Defect 2: Handover Accounting Conflation & Spatial Corridor Horizon
* **Observed Symptom:**
  - `results/sumo_smoke/sumo_smoke_report.json` reported `handover_count = 20`.
  - Yet `sumo_smoke_summary.csv` showed `total_handovers = 0` across all policies, while `sumo_smoke_handovers.csv` contained 20 entries all labeled `type = "associate"`.
* **Root-Cause Analysis:**
  1. **Accounting Bug:** `scripts/run_sumo_smoke_experiment.py` recorded `"handover_count": len(all_handovers)` at top-level. `all_handovers` aggregated all network association events (initial associations, handovers, and disconnects), incorrectly conflating initial RSU entries with actual inter-RSU handovers.
  2. **Physical Corridor Insufficiency:**
     - Highway length = 1200 m.
     - RSU 0 located at $X = 400$ m ($R_{\text{comm}} = 300$ m, coverage $[100\text{ m}, 700\text{ m}]$).
     - RSU 1 located at $X = 800$ m ($R_{\text{comm}} = 300$ m, coverage $[500\text{ m}, 1100\text{ m}]$).
     - Handover decision boundary: Vehicles entering coverage $[500\text{ m}, 700\text{ m}]$ are closer to RSU 1 once $X > 600$ m.
     - With vehicle speed $v \approx 25$ m/s, reaching $X = 600$ m from $X = 0$ m takes $t = 24.0$ s.
     - The smoke test only simulated 20 tasks with mean inter-arrival $\Delta t \approx 0.5$ s ($t_{\text{total}} \approx 10.0$ s). In 10 seconds, vehicles only advanced $\approx 250$ m, remaining entirely within RSU 0's coverage zone.
* **Resolution Plan:**
  - Disaggregate reporting: distinctly track `total_associations`, `total_handovers`, and `total_disconnections`.
  - Expand the evaluation horizon in smoke/experiment runs (minimum 60–80 tasks or $\ge 35$ simulation seconds) to ensure vehicles traverse the inter-RSU handover boundary ($X \in [500\text{m}, 700\text{m}]$).

---

### Defect 3: TraCI Process Lifecycle & Crash Resilience
* **Observed Characteristics:**
  - `SumoTrafficSimulator` uses TraCI with port-based socket connections. Multiple concurrent Streamlit re-runs or rapid experiment executions can leave orphaned SUMO processes or encounter TCP port conflicts.
* **Resolution Plan:**
  - Ensure robust context manager exit (`__enter__`, `__exit__`), explicit `traci.close()` with process termination (`popen.terminate() / popen.kill()`).
  - Add explicit exception guards in `SumoVehicularEdgeEnv.close()` to prevent resource leaks during aborted test runs.

---

### Defect 4: Task Queue & Resource Heterogeneity Modeling
* **Observed Characteristics:**
  - CPU queue lengths drain deterministically at rate $f_{\text{cpu}} \cdot \Delta t$.
  - In high-load scenarios, queue overflow must enforce either queue cycle capping or task drop penalties.
* **Resolution Plan:**
  - Ensure candidate queue cycles and effective frequencies are strictly synchronized between TraCI step increments and offloading evaluation.
  - Verify that `candidate_score` properly weights queue occupancy ($W_{\text{queue}} = 0.5$), frequency ($W_{\text{freq}} = 0.3$), and distance ($W_{\text{dist}} = 0.2$).

---

### Defect 5: Trust & Robust Aggregation Integrity
* **Observed Characteristics:**
  - Behavioural trust updates correctly via Beta prior/posterior increments.
  - Model-update trust evaluates parameter distance $\|\Delta w_i - \bar{w}\|$, cosine similarity, and zero/stale detection.
* **Resolution Plan:**
  - Re-verify Byzantine aggregator fallbacks: ensure division-by-zero protection when all clients fall below the exclusion threshold ($T < \theta$), guaranteeing retention of the last known good global model.

---

### Defect 6: Dashboard Data Flow & Visualization Realism
* **Observed Characteristics:**
  - The Streamlit dashboard (`dashboard/app.py` & `dashboard/three_scene.py`) renders 3D vehicles and RSUs using Three.js.
  - When switching policies or running new episodes, telemetry snapshots must directly bind to real simulation steps without fallback illusions.
* **Resolution Plan:**
  - Ensure all dashboard metrics display verified metrics directly computed by `run_traced_episode`.
  - Maintain daytime highway visual aesthetics with black high-contrast telemetry text.

---

## 4. Engineering Verification Gates & Roadmap

| Phase | Description | Deliverables | Verification Gate |
|---|---|---|---|
| **Phase A** | Engineering Audit & Diagnosis | `docs/ENGINEERING_AUDIT.md` | Audit complete, baseline verified (178/178 tests, 894 manifest intact) |
| **Phase B** | Simulation & Engine Fixes | `dashboard/simulator_engine.py`, `tests/test_sumo_integration.py` | Policies produce distinct offload rates; handovers verified; tests pass |
| **Phase C** | Trust & Math Documentation | `docs/MATHEMATICAL_MODEL.md`, `docs/THREAT_MODEL.md`, `docs/RESEARCH_GAP_MATRIX.csv`, `docs/RESEARCH_QUESTIONS_AND_HYPOTHESES.md` | Formal mathematical rigor, Byzantine defense specifications verified |
| **Phase D** | Experiment Protocol & Execution | `docs/EXPERIMENT_PROTOCOL.md`, `scripts/run_sumo_experiments.py`, `results/sumo_experiments/` | Reproducible multi-seed evaluation, Phase 10 artifacts unmodified |
| **Phase E** | Mobility Validation | `docs/MOBILITY_VALIDATION.md` | Microscopic Krauss car-following validated against empirical distributions |
| **Phase F** | Dashboard Verification | `dashboard/app.py`, `dashboard/three_scene.py` | Accurate live metric rendering, zero console errors, daytime styling |
| **Phase G** | Regression & Manifest Verification | Full test suite, Manifest verifier | 100% test pass, 894/894 manifest artifacts identical |
| **Phase H** | Final Acceptance Report | `reports/ENGINEERING_ACCEPTANCE_REPORT.md` | Gates 1–8 formally evaluated and documented |

---
*Signed: Lead Research & Systems Engineer*
