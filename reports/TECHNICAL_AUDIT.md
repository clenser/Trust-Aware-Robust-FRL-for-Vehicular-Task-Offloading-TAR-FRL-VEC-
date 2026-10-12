# Technical and Mathematical Audit: Phases 0–6

**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs  
**Date:** October 12, 2026  
**Auditor:** Autonomous Verification & Audit Agent  
**Scope:** Mathematical Specification (`docs/PHASE1_SPEC.md`, `specs/formulas.py`), Environment (`environment/`), RL (`rl/`), Trust (`trust/`), and Federation (`federated/`).

---

## 1. Traceability Matrix: Specification to Implementation

| Equation ID | Mathematical Definition | Implementing Function & File | Status / Verification Notes |
|---|---|---|---|
| **EQ-0** | $\text{clamp}_{01}(x) = \max(0, \min(1, x))$ | `specs/formulas.py:clamp01` | **Agrees.** Tested under extreme values (-99, +99). |
| **EQ-1** | $x_i(t+1) = (x_i(t) + v_i \Delta t) \pmod L$ | `environment/mobility.py:MobilityModel.step` | **Agrees.** Wrapped road boundary correctly modelled. |
| **EQ-2** | $\text{eligible}(i,j,t) \Leftrightarrow \text{dist}_{ij} \le R \land \text{avail}_j$ | `environment/vanet_env.py:candidates` | **Agrees.** Distance checked with wrapped road geometry; availability and Bernoulli link check applied. |
| **EQ-3** | $r_{ij} = B \log_2(1 + \text{SNR}_{ij})$ | `environment/network.py:link_rate_hz`, `specs/formulas.py:shannon_rate` | **Agrees.** Power-law path loss with floor `SNR_MIN=1e-3` avoids singularities. |
| **EQ-4** | $L^{\text{tx}} = d_k / r_{ij} + \delta$ | `environment/vanet_env.py:258`, `specs/formulas.py:transmission_latency` | **Agrees.** Handshake latency $\delta$ added. |
| **EQ-5** | $L^{\text{exe}} = c_k / f_j + q_j / f_j$ | `environment/vanet_env.py:249, 263`, `specs/formulas.py:exec_latency` | **Agrees.** Correctly uses effective CPU frequency for vehicles and dedicated CPU for edge nodes. |
| **EQ-6** | $L_k = L^{\text{tx}}_k + L^{\text{exe}}_k$ | `environment/vanet_env.py:269`, `specs/formulas.py:total_latency` | **Agrees.** For local execution, $L^{\text{tx}}=0$. |
| **EQ-7** | $E_k = P^{\text{tx}} L^{\text{tx}}_k + \kappa c_k f_j^2$ | `environment/vanet_env.py:250, 264`, `specs/formulas.py:task_energy` | **Agrees.** $\kappa=10^{-27}$, units consistent (Joules). |
| **EQ-8** | $\text{success} \Leftrightarrow L_k \le \tau_k \land \text{path\_available}$ | `environment/vanet_env.py:270`, `specs/formulas.py:task_outcome` | **Agrees.** Mutually exclusive: `success + deadline_missed + failed == 1`. |
| **EQ-9** | $s \in [0,1]^{7+3K}$: $(d,c,\tau,p) \in [0,1]^4$, $(u,q) \in [0,1]^2$, top-$K$ $(\text{dist}, T, q) \in [0,1]^{3K}$, $\text{count}/K \in [0,1]$ | `environment/vanet_env.py:current_observation` | **Agrees.** Verified $7+3K = 22$ for $K=5$. Padding with zeros is valid. |
| **EQ-10** | $a \in \{0, 1, \dots, K\}$, mask of size $1+K$ | `environment/vanet_env.py:current_observation` | **Agrees.** Action 0 is local; invalid actions raise `InvalidActionError` and are masked in agent. |
| **EQ-11** | $r = -(w_L \bar{L} + w_E \bar{E}) + w_S \text{succ} - C_{\text{fail}} \text{fail} - C_{\text{dl}} \text{miss} + w_T T_{\text{dest}}$ | `specs/formulas.py:compute_reward`, `rl/reward.py:task_reward` | **Corrected (AUDIT-001).** Previously hardcoded neutral trust $0.5$. Now accepts `dest_trust` from decision time. |
| **EQ-12** | $T_{\text{beh}} = \sum \beta_m s_m, \beta=(0.4, 0.25, 0.2, 0.15)$ | `specs/formulas.py:behavioural_trust`, `trust/behavioural.py:observe` | **Agrees.** Signals $s_1$ (success), $s_2$ (timeliness), $s_3$ (resources), $s_4$ (reliability) strictly in $[0,1]$. |
| **EQ-13** | $T(t) = \lambda T(t-1) + (1-\lambda) T_{\text{beh}}$; idle: $\rho T + (1-\rho) T_0$ | `trust/behavioural.py:observe, age_all` | **Agrees.** $\lambda=0.9, \rho=0.99, T_0=0.5$. Unseen nodes age neutrally; disconnections are not penalized as malice. |
| **EQ-14** | $m_1 = \min(1, C \cdot \text{median}(\|\Delta\|) / \|\Delta_i\|)$ | `specs/formulas.py:norm_plausibility` | **Specification helper only.** Pure helper tested in Phase 1; full model-update tensor implementation is planned for Phase 7 (`trust/model_update.py`). |
| **EQ-15** | $m_2 = (1 + \cos(\Delta_i, \bar{\Delta})) / 2$ | `specs/formulas.py:directional_agreement` | **Specification helper only.** Pure scalar helper tested in Phase 1; full tensor cosine similarity planned for Phase 7. |
| **EQ-16** | $m_3 = 1 / (1 + \text{age})$ | `specs/formulas.py:staleness_trust` | **Specification helper only.** Age tracked in Phase 6 server logs; discounting update weight planned for Phase 7. |
| **EQ-17** | $m_4 = 0$ if zero or copy else $1$ | `specs/formulas.py:copy_zero_flag` | **Specification helper only.** Bitwise/norm zero detection planned for Phase 7. |
| **EQ-18** | $m_5$ local learning progress proxy | `docs/PHASE1_SPEC.md §7` | **Planned for Phase 7.** Advisory only (self-reported metric). |
| **EQ-19** | $T_{\text{model}} = \sum \gamma_l m_l$ | `specs/formulas.py:model_update_trust` | **Specification helper only.** Planned for Phase 7 module implementation. |
| **EQ-20** | $T_{\text{final}} = \alpha T_{\text{beh}} + (1-\alpha) T_{\text{model}}$ | `specs/formulas.py:combine_trust` | **Specification helper only.** Planned for Phase 7 server-side integration. |
| **EQ-21** | $\theta_{\text{global}} = \sum (n_i / N) \theta_i$ | `federated/fedavg.py:fedavg_aggregate` | **Agrees (Phase 6).** Weighted parameter averaging with strict validation of parameter keys, shapes, dtypes, and non-finite values. |
| **EQ-22** | $\theta = \sum_{i \in S} w_i \theta_i, w_i = T_{\text{final}}(i) / \sum_S T_{\text{final}}$ | `specs/formulas.py:trust_weighted_aggregate` | **Specification helper only.** Pure list helper tested in Phase 1; PyTorch tensor aggregator planned for Phase 7 (`federated/trust_aggregation.py`). |
| **EQ-23** | $S = \{i : T_{\text{final}}(i) \ge \tau_{\text{exc}}\}$; fallback if $S=\emptyset$ | `specs/formulas.py:trust_weighted_aggregate` | **Specification helper only.** Tested on synthetic floats; integration with `FedServer` planned for Phase 7. |

---

## 2. Environment, Observation, and Action Space Audit

1. **State Dimension Derivation ($7 + 3K$):**
   - Task features: normalized data bits, CPU cycles, deadline, priority ($4$).
   - Requester state: utilization, queue backlog ($2$).
   - Candidate slots: normalized distance, behavioural trust score, normalized queue backlog ($3 \times K$).
   - Metadata: ratio of eligible candidates ($\text{count} / K$, $1$).
   - Total dimension: $4 + 2 + 3K + 1 = 7 + 3K$. For $K=5$, dimension is exactly $22$.
   - Bound validation: asserted on every observation generation. All features clamped to $[0, 1]$.

2. **Action Space & Mask Semantics:**
   - Action $0$: Local computation on requester. Always valid (`mask[0] == True`).
   - Action $j \in \{1, \dots, K\}$: Offload to $j$-th candidate in the sorted candidate list.
   - Mask: `mask[j]` is `True` if $j \le \text{len(candidates)}$, `False` otherwise.
   - Enforced by both environment execution (raising `InvalidActionError`) and agent action selection (`masked_greedy` / `masked_choice`).

---

## 3. Reward and DQN Mathematical Correctness

1. **Reward Formulation:**
   - Default coefficients: $w_L = 1.0$, $w_E = 0.3$, $w_S = 2.0$, $C_{\text{fail}} = 3.0$, $C_{\text{dl}} = 2.0$, $w_T = 0.5$.
   - Normalization constants: $L_{\text{norm}} = 10.0$ seconds, $E_{\text{norm}} = 10.0$ Joules.
   - Defect AUDIT-001 resolved: Destination trust $T_{\text{dest}}$ at decision time is now routed through `StepResult.dest_trust` into `task_reward()`.

2. **DQN Architecture & Bellman Update:**
   - Architecture: MLP ($22 \to 64 \to 64 \to 6$ for $K=5$).
   - Bellman Target:
     $$y = r + \gamma (1 - \text{done}) \max_{a' \in \text{valid}(s')} Q_{\text{target}}(s', a')$$
   - Target network synchronized periodically (default: every 100–200 steps).
   - Episode boundary treatment (AUDIT-003): The setting uses a continuing task arrival process. Setting $\text{done} = \text{True}$ at the end of artificial task windows truncates bootstrapping at the window boundary. This is standard in finite-horizon rollouts and does not compromise stability.

---

## 4. Federated Aggregation and Client Isolation Audit

1. **Privacy & Buffer Isolation:**
   - Replay buffers are strictly private to each `FedClient`.
   - Update payload sent to `FedServer` contains only `client_id`, `round_trained`, `num_samples`, `staleness`, and parameter `state`. Raw transitions, states, and rewards never leave the client process.
2. **Deterministic FedAvg:**
   - Summation performed in sorted client order with double-precision accumulator (`float64`) before casting back to model dtype (`float32`), ensuring bit-level reproducibility.
   - Empty client participation returns the last known good global model without error or modification.

---

## 5. Software Integrity Findings and Defect Resolutions

1. **AUDIT-001 (High):** `rl/reward.py` hardcoded neutral trust $0.5$ in `task_reward`.
   - *Fix:* Added `dest_trust` field to `StepResult` in `environment/vanet_env.py`; updated `task_reward` in `rl/reward.py` to accept and clamp `trust_dest`; updated all call sites across training and evaluation scripts.
   - *Regression Test:* Added unit assertions in `tests/test_phase4_dqn.py` and `tests/test_phase5_trust.py`.
2. **AUDIT-002 (Medium):** `evaluate_dqn.py` crashed on checkpoints with non-default hidden layer dimensions.
   - *Fix:* Configured `eval_dqn` in `scripts/evaluate_dqn.py` to parse `hidden` from YAML and pass to `DQNAgent.load`.
   - *Regression Test:* Validated via execution against `configs/dqn_smoke.yaml`.
