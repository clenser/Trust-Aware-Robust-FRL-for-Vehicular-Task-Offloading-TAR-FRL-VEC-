# Phase 1 — Mathematical Specification (provisional, precedes implementation)

> Status: **specification only**. No simulator, DQN, federated, attack, or
> dashboard code exists. Every numbered equation maps to a pure helper in
> `specs/formulas.py` (same equation numbers in docstrings) covered by
> `tests/test_phase1_formulas.py`. Gaps/evidence: `reports/gap_matrix.md`.

## 1. Entities and notation

- Vehicles `V = {1..N}`, roadside edge nodes `E = {1..M}`, discrete steps `t`.
- Vehicle `i`: position `x_i(t)`, speed `v_i`, CPU freq `f_i`, utilisation `u_i(t)`,
  queue `q_i(t)`, bandwidth `B_i`, trust `T(i,t) ∈ [0,1]`.
- Task `k`: data size `d_k` (bits), CPU cycles `c_k`, deadline `τ_k` (s),
  priority `p_k ∈ {1,2,3}`. No payload content is modelled (AGENTS.md §2).

## 2. Mobility and connectivity (assumptions A3–A4, to be built in Phase 2)

- (EQ-1) Position update: `x_i(t+1) = x_i(t) + v_i·Δt (+ bounded turn noise)`.
  Mobility level sets the speed range (low/medium/high TBD in Phase 2).
- (EQ-2) Link eligibility: `eligible(i,j,t) ⇔ dist(i,j,t) ≤ R ∧ avail_j(t)`,
  with `avail` an intermittent Bernoulli/exposure process. A lost link with no
  outcome observed is **not** negative evidence (see §6).

## 3. Communication and computation model

- (EQ-3) Shannon rate: `r_{ij} = B·log2(1 + SNR_{ij})`, `SNR` from distance-based
  path loss + noise (exact loss law fixed in Phase 2).
- (EQ-4) Transmission latency: `L^{tx}_k = d_k / r_{ij}` (plus fixed handshake `δ`).
- (EQ-5) Execution latency: `L^{exe}_k = c_k / f_j + wait(q_j)`, queue wait linear
  in queued cycles ahead (discipline fixed in Phase 2).
- (EQ-6) Total latency: `L_k = L^{tx}_k + L^{exe}_k` (local execution: `L^{tx}=0`).
- (EQ-7) Energy: `E_k = P^{tx}·L^{tx}_k + κ·c_k·f_j²` (κ chip constant; values TBD).
- (EQ-8) Outcome: `success_k = 1 ⇔ L_k ≤ τ_k ∧ path_available`, else 0;
  deadline miss and failure recorded separately for metrics.

## 4. DQN state and action spaces (finite, implementable)

- (EQ-9) State `s ∈ ℝ^n`, all components min–max normalised to [0,1] using
  documented bounds: task `(d,c,τ,p)` (4), local `(u,q)` (2), per-candidate
  `(dist_norm, trust, load)` for top-`K` candidates by distance (3K), plus
  `eligible_count/K` (1). `n = 7 + 3K`. Shorter candidate lists are
  zero-padded with an availability mask; `K` fixed in Phase 2 (default 5).
- (EQ-10) Action space, finite: `a ∈ {0} ∪ {1..K}`, `0` = execute locally,
  `a=j` = offload to j-th candidate. Actions pointing at padded/ineligible
  candidates are **masked out** (invalid-action masking, not reward-only).

> **Phase 2 erratum history (2026-10-11, resolved in validation fix):** this
> section originally stated `n = 8 + 3K`, miscounting the enumerated components
> (4 + 2 + 3K + 1 = **7 + 3K**). Corrected in place; the component list was and
> remains authoritative. `environment/` implements 7 + 3K. No component was
> added or removed.
- DQN update (standard, F02–F04): `Q(s,a) ← Q + η·(r + γ·max Q_target(s′,·) − Q)`,
  with replay buffer, target network, ε-greedy; Double-DQN (F03) deferred as an
  optional later comparison. Full DQN is Phase 4.

## 5. Reward function (signs fixed; weights provisional)

- (EQ-11) `r = −(w_L·L̄ + w_E·Ē) + w_S·success − C_fail·fail − C_dl·miss + w_T·T_dest`,
  where `L̄, Ē` are normalised latency/energy, `fail` = execution failure,
  `miss` = deadline violation, `T_dest` = destination trust (local: own trust).
- Provisional defaults: `w_L=1.0, w_E=0.3, w_S=2.0, C_fail=3.0, C_dl=2.0, w_T=0.5`
  (sensitivity sweep in Phase 10). Hard constraints (e.g. invalid actions) use
  masking (§4), never reward alone (AGENTS.md §4.2).

## 6. Behavioural trust (Phase 5; signals from B01/B02/B06)

- Signals per interaction with node `i`, each in [0,1]:
  `s1` verified success (1/0), `s2` timeliness (`clip(1 − L/τ)`),
  `s3` resource availability (advertised-vs-observed capacity match),
  `s4` communication reliability (ACK/return rate).
- (EQ-12) `T_beh(i,t) = Σ_m β_m·s_m`, `Σβ=1`, defaults `β=(0.4,0.25,0.2,0.15)` TBD.
- (EQ-13) Temporal update: `T(i,t) = λ·T(i,t−1) + (1−λ)·obs`, `λ∈[0,1)` (default 0.9).
  **Disconnect rule**: link loss with no outcome → no observation update; trust
  ages neutrally toward prior `T₀=0.5` by factor `ρ` per idle round (default 0.99),
  never a malice label (G5).
- All trust clamped to [0,1] (EQ-0 `clamp01`).

## 7. Model-update trust (Phase 7; detectors from D01–D03/D05/E01–E03)

For client update `Δ_i` vs round aggregate/global model, each indicator ∈ [0,1]
(higher = more trustworthy):
- (EQ-14) Norm plausibility: `m1 = 1` if `‖Δ_i‖ ≤ C·median(‖Δ‖)`, else
  `C·median/‖Δ_i‖` (norm clipping bound `C`, cf. D06).
- (EQ-15) Directional agreement: `m2 = (1 + cos(Δ_i, Δ̄))/2` (cf. D05, but
  reference is the **peer aggregate**, not a server root dataset — G1).
- (EQ-16) Staleness: `m3 = 1/(1 + age_rounds)` (age 0 → 1).
- (EQ-17) Copy/zero flags: `m4 = 0` if `Δ_i == 0` or bit-identical to the
  broadcast global delta or another client's update (cf. E01), else 1.
- (EQ-18) Local-progress evidence (provisional, low weight): `m5` from
  client-reported TD-loss decrease, capped and normalised; treated as
  *advisory only* (self-reported → gameable; limitation U4).
- (EQ-19) `T_model(i,t) = Σ_l γ_l·m_l`, defaults `γ=(0.25,0.3,0.15,0.2,0.1)` TBD.
- (EQ-20) Combined trust: `T_final = α·T_beh + (1−α)·T_model` (binding form;
  default `α=0.6`, sweep in Phase 10).

## 8. Federated aggregation (Phases 6–7; baselines F05/D02/F09/D01)

- (EQ-21) FedAvg: `θ ← Σ_i (n_i/N)·θ_i` over participating clients.
- (EQ-22) Trust-weighted: `θ ← Σ_{i∈S} w_i·θ_i`, `w_i = T_final(i)/Σ_S T_final`.
- (EQ-23) With exclusion: `S = {i : T_final(i) ≥ τ_exc}` (default 0.3, sweep);
  **fallback**: if `S` empty → keep last global `θ` unchanged and log the round
  (no division by zero, no model destruction).
- Only parameters + scalar metadata (data count, staleness age, loss delta) are
  communicated — never replay buffers (binding, Phase 6 tests).

## 9. Threat model (attacks Phase 8; capabilities from D04/D06/F10/E01–E03)

- Attacker controls fraction `f ∈ {0, 0.1, 0.2, 0.3, 0.4}` of clients (G6).
- **Poisoning**: `Δ̃ = −s·Δ` (sign-flip, strength `s`) or `Δ + ε·z` (perturbation,
  strength `‖ε·z‖/‖Δ‖`); backdoor-style targeted corruption deferred.
- **Free-riding**: zero update, rebroadcast of global delta (copy), replay of own
  stale update (age ≥ 1), or random-noise update without training.
- **Unreliable execution**: fail/delay tasks with probability `p_u`, or flap
  availability — recorded as behaviour, scored by §6 (not auto-malice).
- Ground-truth attacker labels stored separately from detector outputs;
  detection metrics (precision/recall/FP/FN) computed against them (G5).

## 10. Research questions and testable hypotheses

- RQ1–RQ5 per AGENTS.md §3 (unchanged).
- H1 (RQ1): temporally smoothed multi-signal trust tracks planted reliability
  changes with lag ≤ 1/(1−λ) rounds. Falsifier: lag exceeds bound or [0,1] violated.
- H2 (RQ2): combined trust detects poisoners/free-riders at higher F1 than
  behavioural trust alone at fixed FP ≤ 5%. Falsifier: no F1 gain over seeds.
- H3 (RQ3): trust-aware offloading improves completion rate and mean latency vs
  FedAvg-DQN at equal malicious fractions. Falsifier: CIs overlap.
- H4 (RQ4): completion degrades monotonically with `f`; trust-weighted+exclusion
  degrades slower than FedAvg. Falsifier: non-monotonic or reversed ordering.
- H5 (RQ5): under high mobility + non-IID tasks, honest-client FP rate stays ≤ 10%.
  Falsifier: FP exceeds bound (detector too aggressive, cf. D01 limitation).

## 11. Metrics (definitions frozen here; implementation Phase 3/10)

Mean task latency (s), completion/deadline-satisfaction rate, failure rate,
energy per task (J), mean episode reward + convergence rounds, detection
precision/recall/FP/FN vs ground truth, global-model degradation under attack
(Δ reward vs clean), communication/compute/memory/aggregation overhead. All
results carry `{config, seed}`; multi-seed mean±std with CIs (AGENTS.md §6).

## 12. Remaining uncertainties (→ owner phases)

- U1 reward weights/coefficient scales (sweep Phase 10; defaults §5 provisional).
- U2 `α`, `λ`, `τ_exc`, `C`, `K` defaults (sweeps Phase 10; §6–§8 provisional).
- U3 exact path-loss/queue/task-distribution laws (fixed Phase 2).
- U4 gameability of self-reported loss-delta `m5` (mitigate: low weight + audit).
- U5 sim-vs-real gap: simulated clients share one process; documented, not solved.
- U6 DQN function-approximation has no convergence guarantee under Byzantine
  aggregation (G4) — all robustness claims stay empirical.
