# Phase 10 Final Remediation and Execution Plan

**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions  
**Document:** `reports/PHASE10_FINAL_PLAN.md`  
**Date:** October 12, 2026  
**Status:** Pre-Execution Frozen Plan  
**Target Output Directory:** `results/phase10_final_v2/`  
**Figures Output Directory:** `reports/figures/phase10_final_v2/`  
**Tables Output Directory:** `reports/tables/phase10_final_v2/`  
**Manifest Target:** `reports/PHASE10_FINAL_RUN_MANIFEST.json`  
**Final Report Target:** `reports/PHASE10_FINAL_REPORT.md`  

---

## 1. Research Hypotheses and Primary Comparisons

### 1.1 Pre-Registered Primary Confirmatory Hypothesis (H1)
* **Hypothesis H1:** Under severe zero-update free-riding ($f = 0.40$) over an extended training horizon ($T = 8$ rounds), `trust_threshold` achieves higher clean held-out task completion than standard unweighted `fedavg`.
* **Primary Metric:** Mean held-out task completion rate on clean evaluation environments across matched random training seeds ($N = 10$).
* **Statistical Method:** Two-tailed paired Student-$t$ test (`scipy.stats.ttest_rel`), matched by training master seed. Exact $p$-value and Cohen's $d_z$ effect size will be reported. Significance threshold is set at $\alpha = 0.05$. If $p \ge 0.05$, the comparison will be formally reported as statistically inconclusive without fabricating significance.
* **Secondary Confirmatory Tests & Family-Wise Error Rate:** Holm-Bonferroni correction applied across the primary free-riding comparisons ($f \in \{0.20, 0.40\}$) and sign-flip comparisons ($f \in \{0.20, 0.40\}$).

### 1.2 Exploratory Hypotheses (H2–H5)
* **H2 (Component Synergy):** Both model-update trust ($\alpha=0.0$) and behavioural trust ($\alpha=1.0$) are necessary; disabling either increases error on multi-vector attacks.
* **H3 (Copy Deduplication):** Pairwise cosine deduplication achieves high recall ($\ge 90\%$) against bit-identical copy-update free-riding across all fractions $f \in [0.10, 0.40]$.
* **H4 (Honest Non-IID Fairness):** Honest heterogeneous participants are retained in aggregation (actual exclusion rate $< 5\%$), distinguishing benign non-IID drift from malicious attacks.
* **H5 (Unreliable Participant Safety):** Honest clients with execution delays and intermittent disconnects are penalized in behavioural timeliness/reliability without being falsely classified as model poisoners.

---

## 2. Experimental Matrices and Exact Combinations

### 2.1 Matrix A — Core Evaluation (630 Runs)
All trials execute $T = 8$ communication rounds with 10 clients and 10 master training seeds ($\{0, 1, \dots, 9\}$).

1. **A1. Clean IID Control (30 runs):**
   - 3 strategies (`fedavg`, `trust_weighted`, `trust_threshold`) $\times$ 10 master seeds.
   - Homogeneous client environment: all 10 clients use the exact same base IID environment profile (arrival rate 1.5 tasks/s, velocity 15 m/s, link loss 0.05, 1.5 GHz compute).
   - Malicious fraction $f = 0.0$.
2. **A2. Honest Non-IID Control (30 runs):**
   - 3 strategies $\times$ 10 master seeds.
   - Heterogeneous client environment: 10 distinct client profiles with varying task arrivals ($[0.8, 2.5]$), mobility speeds ($[5, 25]$), link drops ($[0.01, 0.15]$), and CPU capacities ($[1.0, 2.5]$).
   - Malicious fraction $f = 0.0$.
   - **Crucial Pairing Rule:** Attacked heterogeneous runs are compared against their matched honest non-IID control for the same strategy and seed to compute degradation.
3. **A3. Core Attack Sweeps (480 runs):**
   - 4 attacks:
     - `poisoning_sign_flip` ($\Delta = -1.0 \cdot \Delta_{\text{honest}}$)
     - `poisoning_perturbation` ($\Delta = \Delta + 2.0 \cdot u, u \sim \mathcal{N}(0, I)$)
     - `freeriding_zero_update` ($\Delta = 0$)
     - `freeriding_copy_update` ($\Delta = \Delta_{\text{peer}}$)
   - 4 attack fractions: $f \in \{0.10, 0.20, 0.30, 0.40\}$.
   - 3 strategies: `fedavg`, `trust_weighted`, `trust_threshold`.
   - 10 master seeds: $\{0, 1, \dots, 9\}$.
   - Total: $4 \times 4 \times 3 \times 10 = 480$ runs.
4. **A4. Unreliable Execution (90 runs):**
   - 3 severity profiles:
     - `low`: $P(\text{fail})=0.10, P(\text{delay})=0.10, P(\text{drop})=0.10$
     - `medium`: $P(\text{fail})=0.25, P(\text{delay})=0.25, P(\text{drop})=0.20$
     - `high`: $P(\text{fail})=0.50, P(\text{delay})=0.40, P(\text{drop})=0.30$
   - 3 strategies: `fedavg`, `trust_weighted`, `trust_threshold`.
   - 10 master seeds: $\{0, 1, \dots, 9\}$.
   - Target fraction: $20\%$ of clients (2 clients) subjected to unreliable execution.
   - **Ground-Truth Isolation:** Handled via distinct `unreliable_client_ids` attribute, keeping `malicious_client_ids` strictly empty ($f_{\text{malicious}} = 0.0$).
   - Total: $3 \times 3 \times 10 = 90$ runs.

$$\text{Subtotal Core Matrix A} = 30 + 30 + 480 + 90 = 630 \text{ runs}$$

---

### 2.2 Matrix B — Mandatory Ablations (70 Runs)
Evaluates 7 architectural configurations across 2 attack scenarios and 5 master seeds ($\{0, 1, 2, 3, 4\}$):
- **Attack Scenarios (2):**
  1. `poisoning_sign_flip` ($f = 0.20$)
  2. `freeriding_zero_update` ($f = 0.20$)
- **Configurations (7):**
  1. `full_proposed`: Full TAR-FRL ($\alpha = 0.60, \tau_{\text{exc}} = 0.40, \beta = 0.70$)
  2. `no_behavioural_trust`: $\alpha = 0.0$ ($T_{\text{final}} = T_{\text{model}}$)
  3. `no_model_trust`: $\alpha = 1.0$ ($T_{\text{final}} = T_{\text{beh}}$)
  4. `no_temporal_smoothing`: $\beta = 0.0$ (memoryless trust)
  5. `no_threshold_exclusion`: $\tau_{\text{exc}} = 0.0$ (trust-weighted only, no exclusion)
  6. `fedavg_no_trust`: Standard uniform FedAvg
  7. `local_rl_no_fl`: Independent local DQN with zero federation
- **Seeds:** 5 seeds per condition ($\{0, 1, 2, 3, 4\}$).
- **Detector Availability Rule:** For `fedavg_no_trust` and `local_rl_no_fl`, detection metrics are explicitly recorded as `N/A` (NaN) rather than misleading zero scores.

$$\text{Subtotal Ablation Matrix B} = 7 \text{ configs} \times 2 \text{ scenarios} \times 5 \text{ seeds} = 70 \text{ runs}$$

---

### 2.3 Matrix C — Full Sensitivity Analysis (180 Runs)
Evaluates parametric stability on `trust_threshold` under 2 attack scenarios ($f = 0.20$) across 5 master seeds ($\{0, 1, 2, 3, 4\}$):
- **Attack Scenarios (2):**
  1. `poisoning_sign_flip` ($f = 0.20$)
  2. `freeriding_stale_replay` ($f = 0.20$)
- **Parameters & Values (18 values):**
  1. Trust Balance ($\alpha$): $\{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ (6 values)
  2. Exclusion Threshold ($\tau_{\text{exc}}$): $\{0.20, 0.30, 0.40, 0.50, 0.60\}$ (5 values)
  3. Temporal Smoothing ($\beta$ / $\lambda$): $\{0.50, 0.70, 0.90, 0.95\}$ (4 values)
  4. Anomaly Threshold ($\tau_{\text{anomaly}}$): $\{0.40, 0.50, 0.60\}$ (3 values)

$$\text{Subtotal Sensitivity Matrix C} = (6 + 5 + 4 + 3) \times 2 \text{ scenarios} \times 5 \text{ seeds} = 18 \times 2 \times 5 = 180 \text{ runs}$$

---

### Total Planned Experimental Matrix
$$\mathbf{N_{\text{total}} = 630 \text{ (Core)} + 70 \text{ (Ablation)} + 180 \text{ (Sensitivity)} = 880 \text{ runs}}$$

---

## 3. Seed Derivation and Reproducibility Protocol

To ensure true stochastic divergence across master seeds while preserving strict paired alignment across competing strategies:
1. **Master Experiment Seed:** $S \in \{0, 1, \dots, 9\}$.
2. **Stable Component Seed Derivation:**
   - Model Weights Initialization Seed: $S_{\text{model}} = 10000 + S$
   - Client $i$ Environment Seed: $S_{\text{env}, i} = 20000 + S \times 100 + i$
   - Client $i$ Task Arrival Seed: $S_{\text{task}, i} = 30000 + S \times 100 + i$
   - Server Client Selection Seed: $S_{\text{sel}} = 40000 + S$
   - Attack Perturbation Seed: $S_{\text{atk}} = 50000 + S$
   - Held-Out Evaluation Seeds: Fixed set of 5 independent seeds $\{100, 101, 102, 103, 104\}$ applied identically to all trials.
3. **Paired Invariant:** At a given master seed $S$, `fedavg`, `trust_weighted`, and `trust_threshold` receive identical initial Q-networks and identical local client environments/tasks.
4. **Held-Out Isolation:** The evaluation seed set $\{100..104\}$ is completely disjoint from all training seeds and never enters training buffers or trust estimation.

---

## 4. Acceptance Criteria
Phase 10 remediation will be accepted if and only if:
1. Exact row count matches the frozen plan: 630 core + 70 ablation + 180 sensitivity = 880 runs. Zero duplicates, zero missing.
2. Unreliable execution is fully executed with separate ground-truth labeling (`unreliable_client_ids`), reporting injected failures and delays during training.
3. Clean IID and Honest Non-IID controls use distinct verified client configurations.
4. Primary statistical test (paired $t$-test) is executed via SciPy and exact $p$-values are reported without fabrication.
5. All 126 Phase 0–9 regression tests and new coverage tests pass.
6. Zero git commits or pushes are made; all historical artifacts remain preserved.
