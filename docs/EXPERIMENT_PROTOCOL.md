# EXPERIMENTAL EVALUATION PROTOCOL
**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions (TAR-FRL-VEC)  
**Document Version:** 2.0  
**Status:** Approved Experimental Standard  

---

## 1. Experimental Design & Objectives

This protocol governs all experimental evaluations conducted within the TAR-FRL-VEC project. The objective is to provide a reproducible, statistically rigorous, and transparent benchmarking methodology evaluating task offloading efficiency, trust convergence, Byzantine attack resistance, and microscopic mobility dynamics.

### 1.1 Strict Isolation Principle
* **Historical Benchmark Isolation:** The historical Phase 10 benchmark matrix (880 runs, 894 manifest artifacts in `results/phase10_final_v4/` and `reports/PHASE10_FINAL_V4_RUN_MANIFEST.json`) represents frozen baseline evidence. No new experiment may overwrite, alter, or rename those artifacts.
* **New Experiment Namespace:** All microscopic SUMO traffic experiments and modern evaluations must be persisted strictly into `results/sumo_experiments/` or `results/sumo_smoke/`.

---

## 2. Experimental Factors & Parameter Configurations

### 2.1 Environmental & Network Parameters
| Parameter | Symbol | Benchmark Value(s) | Description |
|---|---|---|---|
| Highway Length | $L$ | $1200\text{ m}$ (SUMO) / $2000\text{ m}$ (Macro) | Longitudinal highway corridor length |
| Highway Lanes | $N_{\text{lanes}}$ | 3 lanes | Forward unidirectional lanes |
| Roadside Units | $M$ | 2 RSUs | Edge servers spaced at $400\text{ m}$ and $800\text{ m}$ |
| RSU Compute Freq | $f_m^{\text{edge}}$ | $6.0\text{--}10.0\text{ GHz}$ | High-capacity edge compute clock |
| Vehicular Compute Freq | $f_i$ | $1.2\text{--}2.8\text{ GHz}$ | Heterogeneous vehicular CPU clock |
| Wireless Comm Range | $R_{\text{comm}}$ | $250.0\text{ m}$ (SUMO) / $150.0\text{ m}$ (Macro) | Cellular-V2X / DSRC radio coverage radius |
| Channel Bandwidth | $B$ | $10\text{ MHz}$ | Orthogonal V2I/V2V transmission bandwidth |
| Transmit Power | $P_{\text{tx}}$ | $0.5\text{ W}$ ($27\text{ dBm}$) | Onboard vehicular radio transmit power |
| Noise Power Spectral Density | $N_0$ | $1.0\times 10^{-9}\text{ W}$ | Background thermal noise |
| Path Loss Exponent | $\gamma$ | $3.0$ | Vehicular urban/highway propagation exponent |

### 2.2 Traffic Mobility Levels (SUMO Krauss Model)
* **Low Density:** Insertion rate $\sim 0.1\text{ veh/s}$, desired velocity $v_{\text{des}} \in [12, 18]\text{ m/s}$, free-flow traffic.
* **Medium Density:** Insertion rate $\sim 0.25\text{ veh/s}$, desired velocity $v_{\text{des}} \in [18, 25]\text{ m/s}$, synchronized flow.
* **High Density:** Insertion rate $\sim 0.4\text{ veh/s}$, desired velocity $v_{\text{des}} \in [22, 33]\text{ m/s}$, dense platooning with lane-changing.

### 2.3 Offloading Policies Evaluated
1. **Local Execution (`local_only`):** All computational tasks are executed onboard the requesting vehicle's local CPU ($a = 0$).
2. **Nearest RSU (`nearest_edge`):** Offloads to the nearest eligible edge server in radio range; falls back to local execution if no RSU is in range.
3. **Resource-Aware Heuristic (`resource_aware`):** Scores candidates based on free queue capacity ($50\%$), clock frequency ($30\%$), and proximity ($20\%$).
4. **Random Valid (`random_valid`):** Selects uniformly at random among all mask-valid actions from a seeded PRNG stream.
5. **Single-Agent DQN (`dqn_trained_default`):** Deep Q-Network trained on individual vehicle experience with replay buffer.
6. **Federated FedAvg (`fedavg`):** Distributed DQN trained across vehicular clients aggregated using standard sample-weighted FedAvg.
7. **Trust-Aware Robust FRL (`feddqn_global_final` / TAR-FRL):** Proposed framework combining dual-layer trust with threshold-based Byzantine exclusion ($\theta = 0.40$).

---

## 3. Evaluated Metrics & Mathematical Definitions

1. **Task Completion Rate ($CR$):**
   $$CR = \frac{\sum_{k=1}^K \mathbb{I}_{\text{success}}(k)}{K}$$
2. **Deadline Miss Rate ($MR$):**
   $$MR = \frac{\sum_{k=1}^K \mathbb{I}_{\text{miss}}(k)}{K}$$
3. **Task Failure Rate ($FR$):**
   $$FR = \frac{\sum_{k=1}^K \mathbb{I}_{\text{fail}}(k)}{K}$$
4. **Mean End-to-End Latency ($\bar{T}$):**
   $$\bar{T} = \frac{1}{K} \sum_{k=1}^K \left(T_k^{\text{tx}} + T_k^{\text{exe}}\right)$$
5. **Mean Energy Consumption ($\bar{E}$):**
   $$\bar{E} = \frac{1}{K} \sum_{k=1}^K E_k$$
6. **Offload Fraction ($OF$):**
   $$OF = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(a_k > 0)$$
7. **Network Handover Accounting:**
   - **True Handovers ($N_{\text{ho}}$):** Direct transitions from RSU $A$ to RSU $B$ without dropping into an unassociated state ($X \in [550\text{m}, 650\text{m}]$).
   - **RSU Associations ($N_{\text{assoc}}$):** Initial entries into RSU radio coverage.
   - **RSU Disconnections ($N_{\text{disc}}$):** Exits from RSU radio coverage into unassociated gaps.

---

## 4. Multi-Seed Statistical Methodology

To guarantee reproducibility and avoid reporting anomalies from single stochastic runs:
- Major comparative experiments must evaluate a minimum of 3 distinct random seeds (e.g., seeds 42, 101, 2024).
- Reports must publish sample mean $\bar{x}$ and sample standard deviation $s$:
  $$\bar{x} = \frac{1}{S} \sum_{s=1}^S x_s, \quad s = \sqrt{\frac{1}{S-1} \sum_{s=1}^S (x_s - \bar{x})^2}$$
- When presenting baseline superiority, statistical significance must be demonstrated via non-overlapping standard error bands or paired t-tests ($p < 0.05$).

---
*Compliant with ACM SIGCOMM / IEEE INFOCOM reproducibility standards.*
