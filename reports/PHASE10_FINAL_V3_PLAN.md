# Phase 10 Final v3 Experimental Evaluation Plan

## 1. Objectives & Scope
This plan specifies the corrected experimental protocol for Phase 10 Final v3 in accordance with the project contract (`AGENTS.md`). The evaluation executes an 880-run matrix spanning core scenarios, component ablations, and sensitivity sweeps to rigorously answer Research Questions RQ1–RQ5.

## 2. Parameter Control & System Specification
- **Smoothing Factor ($\lambda$)**: Fixed to $\lambda = 0.70$ across all core and ablation runs, with guaranteed propagation and persistence through client/environment resets (`c.set_trust_lambda(0.70)`).
- **Client Base Heterogeneity**:
  - **Clean IID**: Fully homogeneous clients ($2.0\text{ GHz}$, $10\text{ MHz}$).
  - **Honest Non-IID**: Explicit resource heterogeneity configured per client:
    - Client 0: $1.0\text{ GHz}$, $5.0\text{ MHz}$
    - Client 1: $1.2\text{ GHz}$, $6.5\text{ MHz}$
    - Client 2: $1.5\text{ GHz}$, $8.0\text{ MHz}$
    - Client 3: $1.8\text{ GHz}$, $10.0\text{ MHz}$
    - Client 4: $2.0\text{ GHz}$, $10.0\text{ MHz}$
    - Client 5: $2.2\text{ GHz}$, $12.0\text{ MHz}$
    - Client 6: $2.4\text{ GHz}$, $14.0\text{ MHz}$
    - Client 7: $2.6\text{ GHz}$, $16.0\text{ MHz}$
    - Client 8: $2.8\text{ GHz}$, $18.0\text{ MHz}$
    - Client 9: $3.0\text{ GHz}$, $20.0\text{ MHz}$
- **Detection & Exclusion Thresholds**:
  - $m_2$ direction anomaly threshold: $\tau_{\text{dir}} = 0.45$
  - $m_3$ staleness threshold: $\tau_{\text{stale}} = 0.50$
  - $m_4$ origin threshold: $\tau_{\text{orig}} = 0.00$
  - Combined anomaly threshold: $\tau_{\text{anomaly}} = 0.50$
  - Trust exclusion threshold: $\tau_{\text{exclusion}} = 0.35$
  - Combination weight: $\alpha = 0.50$

## 3. Experimental Matrices
1. **Matrix A: Core Scenarios (630 runs)**
   - 9 Scenario Groups:
     - `clean_iid` (1 fraction: 0.0) $\times$ 3 strategies $\times$ 10 seeds = 30 runs
     - `honest_non_iid` (1 fraction: 0.0) $\times$ 3 strategies $\times$ 10 seeds = 30 runs
     - `poisoning_sign_flip` (4 fractions: 0.1, 0.2, 0.3, 0.4) $\times$ 3 strategies $\times$ 10 seeds = 120 runs
     - `poisoning_perturbation` (4 fractions: 0.1, 0.2, 0.3, 0.4) $\times$ 3 strategies $\times$ 10 seeds = 120 runs
     - `freeriding_zero_update` (4 fractions: 0.1, 0.2, 0.3, 0.4) $\times$ 3 strategies $\times$ 10 seeds = 120 runs
     - `freeriding_copy_update` (4 fractions: 0.1, 0.2, 0.3, 0.4) $\times$ 3 strategies $\times$ 10 seeds = 120 runs
     - `unreliable_low` (1 fraction: 0.2) $\times$ 3 strategies $\times$ 10 seeds = 30 runs
     - `unreliable_medium` (1 fraction: 0.2) $\times$ 3 strategies $\times$ 10 seeds = 30 runs
     - `unreliable_high` (1 fraction: 0.2) $\times$ 3 strategies $\times$ 10 seeds = 30 runs
   - **Total Core**: $30 + 30 + 120 + 120 + 120 + 120 + 30 + 30 + 30 = 630$ runs.

2. **Matrix B: Ablation Studies (70 runs)**
   - 7 Configurations evaluated under `freeriding_zero_update` ($f=0.30$) over 10 seeds ($0..9$):
     - `full_proposed`
     - `no_behavioural_trust` ($\alpha = 0.0$)
     - `no_model_trust` ($\alpha = 1.0$)
     - `no_temporal_smoothing` ($\lambda = 0.0$)
     - `no_threshold_exclusion` ($\tau_{\text{exc}} = 0.0$)
     - `fedavg_no_trust` (FedAvg)
     - `local_rl_no_fl` (independent training without federated aggregation)

3. **Matrix C: Sensitivity Analysis (180 runs)**
   - Evaluated under `freeriding_stale_replay` ($f=0.30$) over 5 seeds ($0..4$, 36 configurations $\times$ 5 = 180 runs):
     - $\alpha \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ (6 values)
     - $\tau_{\text{exc}} \in \{0.2, 0.3, 0.4, 0.5, 0.6\}$ (5 values)
     - $\lambda \in \{0.50, 0.70, 0.90, 0.95\}$ (4 values)
     - $m_2$ direction threshold $\in \{0.40, 0.50, 0.60\}$ with $\tau_{\text{anomaly}} = 0.50$ (3 values)

## 4. Execution Integrity & Fault Accounting
- Stale replay explicitly handles round 0 without injecting falsified updates; preserves origin round metadata for $r \ge 1$.
- Fault accounting tallies injected drops, fails, delays, and `tasks_with_any_fault` across all unreliable wrapper instances at the end of the simulation.
- Unreliable participants are tracked with `is_malicious = False`.
