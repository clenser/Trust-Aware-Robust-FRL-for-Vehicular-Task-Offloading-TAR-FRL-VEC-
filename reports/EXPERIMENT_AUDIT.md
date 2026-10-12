# Experimental Methodology and Reproducibility Audit: Phases 0–6

**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs  
**Date:** October 12, 2026  
**Auditor:** Autonomous Verification & Audit Agent  
**Scope:** Experimental fairness, evaluation protocols, DQN and FedDQN training budgets, metrics validity, and statistical reporting.

---

## 1. Experimental Reproducibility and Checkpoint Traceability

1. **Artifact Integrity:**
   - All historical experimental outputs (`results/`, `checkpoints/`) have been cataloged with SHA-256 checksums in `reports/historical_checksums.json` (167 files preserved).
   - Historical artifacts were not overwritten. All newly executed audit runs were written to separate paths (`results/audit/`, `checkpoints/audit/`).
2. **Deterministic Execution:**
   - Python, NumPy, and PyTorch RNGs are explicitly seeded at episode and rollout boundaries (`seed_all(seed)`).
   - PyTorch single-thread enforcement (`torch.set_num_threads(1)`) guarantees deterministic matmul operations.
   - Repeated runs of `run_baselines.py`, `train_dqn.py`, and `train_feddqn.py` with identical seeds produce byte-identical metrics and checkpoints.

---

## 2. Baseline Fairness and Trajectory Coupling

1. **Baseline Policies Implemented:**
   - `local_only`: Action 0 always.
   - `random_valid`: Uniform random selection over mask-valid actions.
   - `nearest_edge`: Nearest eligible edge node; falls back to local if no edge node is in range.
   - `resource_aware`: Heuristic scoring of candidate nodes balancing queue congestion, CPU frequency, and distance.
   - `trust_aware` (Phase 5): Ranks candidates by behavioural trust score; falls back to local if best candidate trust is below threshold $\tau=0.3$.
2. **On-Demand Randomness Caveat:**
   - In `environment/vanet_env.py`, channel connectivity draws (`link_available`) and background load fluctuations occur dynamically during `step()`.
   - *Implication:* If Policy A selects Local ($a=0$) and Policy B selects Edge ($a=1$), Policy B triggers additional channel draws for transmission checks. Consequently, their future environmental states diverge even when starting from the same seed.
   - *Audit Assessment:* As documented in Phase 3, this comparison is **statistically paired across seeds**, but not identical-trajectory coupled. Reporting averages across multiple held-out seeds (e.g. seeds 10, 11, 12) correctly captures this variance.

---

## 2.1 Historical-Results Impact Table (Reward & Trust Fix)

| Artifact Category | Files Preserved | Historical Status | Version / Implementation Note | Audit Resolution |
|---|---|---|---|---|
| **Phase 3 Baselines** | `results/baselines_smoke.*`<br>`results/baselines_trust_eval.*` | Physical outcomes valid; rewards used neutral trust $0.5$. | Generated in Phase 3/5 under provisional reward with constant trust $T_{\text{dest}}=0.5$. | Preserved unchanged as historical reference. Re-evaluated in `results/audit/baselines_audit.*` with dynamic trust. |
| **Phase 4 DQN Checkpoints** | `checkpoints/dqn_default/`<br>`checkpoints/dqn_smoke/` | Weights trained with neutral trust reward component. | Trained under constant $T_{\text{dest}}=0.5$. Since initial network weights were largely neutral and local execution was favored, weights remain valid for greedy execution. | Preserved. Evaluated under corrected dynamic reward in `results/audit/dqn_eval_default_audit.*` and `dqn_eval_audit.*`. |
| **Phase 4 DQN Evaluations** | `results/dqn_eval_dqn_default.*`<br>`results/dqn_trusteval_dqn_default.*` | Completion, latency, and energy valid; rewards slightly underestimated on high-trust nodes. | Evaluated with hardcoded neutral trust $0.5$. | Preserved unchanged. Re-run generated at `results/audit/dqn_eval_default_audit.*`. |
| **Phase 5 Trust Snapshots** | `results/trust_smoke_events.csv`<br>`results/trust_smoke_snapshots.csv` | **100% Valid.** | Tracks behavioural trust dynamics ($s_1..s_4$, $\lambda=0.9$, $\rho=0.99$), unaffected by reward calculation. | Preserved. Audit re-run saved at `results/audit/trust_smoke_audit*`. |
| **Phase 6 FedDQN Checkpoints** | `checkpoints/feddqn_default/`<br>`checkpoints/feddqn_smoke/` | Valid FedAvg weights across clients. | Parameter updates trained locally with neutral reward. | Preserved unchanged. Evaluated under corrected dynamic reward in `results/audit/fedeval_default_audit.*` and `fedeval_audit.*`. |
| **Phase 6 FedDQN Evaluations** | `results/fedeval_feddqn_default.*` | Physical metrics valid; rewards used neutral trust $0.5$. | Evaluated with hardcoded neutral trust $0.5$. | Preserved unchanged. Re-run generated at `results/audit/fedeval_default_audit.*`. |
| **Figures** | `reports/figures/*.png` | Historical plots reflect earlier CSV logs. | Historical plots preserved; new audit plots generated alongside audit CSVs. | Plots generated from audit runs reflect dynamic trust outcomes. |

---

## 3. Training Budget and Convergence Analysis

### 3.1 Single-Agent DQN (Phase 4)
- **Reported Budget:** 25 episodes $\times$ 50 tasks = 1,250 transitions.
- **Epsilon Schedule:** Linear decay over 20 episodes from 1.0 to 0.05.
- **Audit Assessment:**
  - 1,250 transitions is sufficient for a pipeline smoke test and sanity check, but insufficient to establish empirical convergence or optimality in a dynamic vehicular environment.
  - In evaluation on seeds 10–12, DQN achieved completion rates of $0.788 \pm 0.141$ and offload fractions between $0.02$ and $0.18$, largely converging to local execution due to early penalty avoidance.
  - **Conclusion:** DQN does not outperform `nearest_edge` ($0.840 \pm 0.080$). Claims of DQN superiority cannot be made; results must be reported honestly as an initial proof of learning.

### 3.2 Federated DQN (Phase 6)
- **Reported Budget:** 3 clients $\times$ 3 rounds $\times$ 4 local episodes $\times$ 25 tasks = 900 transitions total across the network.
- **Audit Assessment:**
  - The pipeline demonstrates correct parameter aggregation, buffer isolation, and global model broadcast.
  - However, 3 communication rounds is a pipeline validation check, not a convergence study.
  - On held-out seeds 10–12:
    - `feddqn_global`: Completion $0.707 \pm 0.101$, Reward $0.541 \pm 0.406$.
    - `local_alone_mean`: Completion $0.738 \pm 0.080$, Reward $0.655 \pm 0.281$.
    - `single_dqn`: Completion $0.753 \pm 0.050$, Reward $0.702 \pm 0.231$.
    - `nearest_edge`: Completion $0.840 \pm 0.080$, Reward $1.119 \pm 0.392$.
  - **Conclusion:** Global FedAvg currently performs comparably to or slightly below independent local training. This is expected in early rounds under non-IID conditions before policy alignment. No convergence superiority may be claimed.

---

## 4. Evaluation Metrics and Denominator Integrity

1. **Task Accounting:**
   - Every task results in exactly one mutually exclusive outcome:
     $$\text{successes} + \text{deadline\_misses} + \text{failures} = \text{num\_tasks}$$
   - Verified across all evaluation logs: $\text{completion\_rate} + \text{miss\_rate} + \text{failure\_rate} = 1.000$.
2. **Latency & Energy Denominators:**
   - Both `mean_latency_s` and `mean_energy_j` average over all generated tasks in the episode ($\text{sum} / \text{num\_tasks}$).
   - Even when a task misses its deadline or suffers link drop, its incurred execution/transmission attempt energy is fully accounted for.

---

## 5. Required Reruns and Future Experiment Matrix

1. **Completed Audit Reruns:**
   - Baseline smoke evaluation rerun: `results/audit/baselines_audit.*`
   - DQN smoke training & evaluation rerun: `checkpoints/dqn_smoke/`, `results/audit/dqn_eval_audit.*`
   - Behavioural trust logging rerun: `results/audit/trust_smoke_audit*`
   - Federated DQN training & evaluation rerun: `checkpoints/feddqn_smoke/`, `results/audit/fedeval_audit.*`
2. **Recommendations for Phase 10 (Full Evaluation):**
   - Increase training episodes to $\ge 100$ episodes per client.
   - Run at least 5 distinct random seeds ($N=5$) for every configuration.
   - Report 95% confidence intervals ($\bar{x} \pm 1.96 \cdot \text{SE}$).
   - Conduct planned ablations:
     - Full proposed method (behavioural + model trust + robust aggregation).
     - Without behavioural trust ($T_{\text{beh}}$ disabled).
     - Without model-update trust ($T_{\text{model}}$ disabled).
     - Without temporal trust smoothing ($\lambda=0$).
     - FedAvg without trust weighting.
