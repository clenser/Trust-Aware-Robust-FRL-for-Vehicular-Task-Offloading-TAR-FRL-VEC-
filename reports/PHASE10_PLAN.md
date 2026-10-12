# Phase 10 Research Experimentation Plan

**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions  
**Phase:** Phase 10 — Full Evaluation, Ablation, and Sensitivity Analysis  
**Status:** Pre-Execution Specification & Traceability Matrix

---

## 1. Research Questions Addressed

- **RQ1 (Dynamic Trust Tracking):** How does trust track participant behaviour under varying mobility, resources, and communication channel conditions?
- **RQ2 (Component Synergy):** Does combining behavioural trust with model-update trust identify malicious and free-riding participants better than behavioural trust or FedAvg alone?
- **RQ3 (Offloading Performance):** Does trust-aware robust aggregation mitigate offloading latency, deadline misses, and reward degradation when participants are adversarial?
- **RQ4 (Vulnerability Scaling):** How do detection F1, precision, recall, and global model degradation scale as the malicious participant ratio $f$ increases from $0\%$ to $40\%$?
- **RQ5 (Robustness Boundaries):** Does the proposed method avoid penalizing honest non-IID participants across diverse mobility and task load conditions while rejecting Byzantine perturbations and free-riding?

---

## 2. Experimental Matrices & Configurations

All experiments are executed over an extended horizon of **$T = 8$ federated communication rounds** (doubling the Phase 8 horizon) with **$N = 10$ seeds ($\{0, 1, \dots, 9\}$)** for core benchmarks to provide rigorous statistical inferential power ($df = 9$).

### Matrix 1: Core Attack Sweeps & Robustness Benchmarks (10 Seeds)
- **Scenarios (6):**
  1. `clean_control` ($f = 0.0$)
  2. `poisoning_sign_flip` ($f \in \{0.1, 0.2, 0.3, 0.4\}$)
  3. `poisoning_perturbation` ($f \in \{0.1, 0.2, 0.3, 0.4\}$, $\epsilon = 2.0$)
  4. `freeriding_zero_update` ($f \in \{0.1, 0.2, 0.3, 0.4\}$)
  5. `freeriding_copy_update` ($f \in \{0.1, 0.2, 0.3, 0.4\}$)
  6. `honest_non_iid` ($f = 0.0$, 10 heterogeneous client conditions)
- **Aggregation Strategies (3):**
  - `fedavg`: Standard unweighted FedAvg baseline.
  - `trust_weighted`: Dynamic weighting by $T_{\text{final}} \in [0, 1]$.
  - `trust_threshold`: Dynamic weighting + anomaly exclusion threshold ($\tau_{\text{exc}} = 0.40$).
- **Statistical Replication:** 10 random seeds ($\{0, \dots, 9\}$).

### Matrix 2: Mandatory Ablation Study (AGENTS.md §6) (5 Seeds)
Evaluates 7 distinct architectural configurations on `poisoning_sign_flip` ($f = 0.2$) and `freeriding_zero_update` ($f = 0.2$) across 5 seeds ($\{0, 1, 2, 3, 4\}$):
1. **Ablation 1 (Complete Proposed System):** Full model with $T_{\text{beh}}$, $T_{\text{mod}}$, temporal update $\lambda = 0.9$, $\alpha = 0.6$, $\tau_{\text{exc}} = 0.40$.
2. **Ablation 2 (Without Behavioural Trust):** $\alpha = 0.0$ ($T_{\text{final}} = T_{\text{mod}}$).
3. **Ablation 3 (Without Model-Update Trust):** $\alpha = 1.0$ ($T_{\text{final}} = T_{\text{beh}}$).
4. **Ablation 4 (Without Temporal Smoothing):** Memoryless trust ($\lambda = 0.0$).
5. **Ablation 5 (Without Threshold Exclusion):** `trust_weighted` aggregation without exclusion ($\tau_{\text{exc}} = 0.0$).
6. **Ablation 6 (Federated RL Without Trust):** Standard FedAvg ($w_i \propto N_i$).
7. **Ablation 7 (RL Without Federated Learning):** Independent local DQNs evaluated without aggregation.

### Matrix 3: Sensitivity Analysis (5 Seeds)
Evaluates parametric stability on `poisoning_sign_flip` ($f = 0.2$) and `freeriding_stale_replay` ($f = 0.2$):
1. **Trust-Combination Parameter Sweep ($\alpha$):** $\alpha \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.
2. **Exclusion Threshold Sweep ($\tau_{\text{exc}}$):** $\tau_{\text{exc}} \in \{0.20, 0.30, 0.40, 0.50, 0.60\}$.
3. **Temporal Smoothing Parameter Sweep ($\lambda$):** $\lambda \in \{0.5, 0.7, 0.9, 0.95\}$.
4. **Anomaly Anomaly Score Sensitivity ($\tau_{\text{anomaly}}$):** $\tau_{\text{anomaly}} \in \{0.40, 0.50, 0.60\}$.

---

## 3. Directory Layout & Artifact Preservation

All Phase 10 artifacts are written to dedicated, isolated directories:
- Results CSV & Summaries: `results/phase10/`
- Checkpoints: `results/phase10/checkpoints/`
- Figures & LaTeX Tables: `reports/figures/phase10/`, `reports/tables/phase10/`
- Acceptance Manifest & Report: `reports/PHASE10_RUN_MANIFEST.json`, `reports/PHASE10_REPORT.md`

Historical Phase 0–9 files remain completely untouched.
