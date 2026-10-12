# Assumptions Register (Phase 0 — provisional, to be formalised in Phase 1/2)

> Every entry is labelled **provisional** until backed by the Phase 1
> specification and Phase 2 implementation. Nothing here is a measured result.

## A. Simulation paradigm

| ID | Assumption | Status | Resolved by |
|---|---|---|---|
| A1 | A synthetic, discrete-step Python simulation is sufficient for the minimum project; packet-level fidelity (SUMO/NS-3/OMNeT++) is not required. | Provisional (per AGENTS.md constraints) | Phase 1 spec, Phase 2 env |
| A2 | Tasks are fully described by `(data_size, cpu_cycles, deadline, priority)`; no image/sensor payloads or vision models are needed. | Provisional | Phase 1–2 |
| A3 | Mobility levels low/medium/high can be represented as parameterised speed/position-update ranges plus intermittent link availability. | Provisional | Phase 2 |
| A4 | Communication eligibility is a function of distance vs. range + availability flag; detailed fading/MAC modelling is out of scope for the minimum. | Provisional | Phase 1–2 |

## B. Learning & federation

| ID | Assumption | Status | Resolved by |
|---|---|---|---|
| B1 | DQN with a discrete action space `{local} ∪ {eligible vehicle} ∪ {eligible edge/RSU}` is an adequate starting point. | Provisional | Phase 1 (finite, implementable), Phase 4 |
| B2 | Simulated federated clients (one process, partitioned synthetic experience) are an acceptable stand-in for deployed vehicles for the minimum project; the sim-vs-real gap must be documented. | Provisional | Phase 1, Phase 6 |
| B3 | Clients share model updates/weights + permitted metadata only, never raw replay buffers. | Binding constraint | Phase 6 tests |

## C. Trust & security

| ID | Assumption | Status | Resolved by |
|---|---|---|---|
| C1 | Trust values live in `[0, 1]`; combined trust `T_final = alpha*T_behaviour + (1-alpha)*T_model`. | Binding formula, parameters provisional | Phase 1, 5, 7 |
| C2 | Temporary mobility-related disconnection must **not** be auto-labelled malicious. Smoothing/decay must distinguish unavailability from adversarial behaviour. | Binding constraint | Phase 5 tests |
| C3 | Trust weighting alone is **not** a complete poisoning defence; anomaly validation + threshold exclusion + fallback (keep last good global model when nobody is eligible) are required. | Binding constraint | Phase 7 |
| C4 | Honest non-IID updates may legitimately diverge and must not be auto-rejected merely for differing. | Binding constraint | Phase 7 tests |

## D. Evaluation & engineering

| ID | Assumption | Status | Resolved by |
|---|---|---|---|
| D1 | Smoke configs run in seconds–minutes on a student laptop / Colab; full sweeps are separate and multi-seed. | Provisional | Phase 2, 10 |
| D2 | Metrics (latency, completion/deadline rate, failure rate, energy, reward/convergence, precision/recall/FP/FN, overhead) are computed from saved raw outputs with config+seed metadata; single runs are never presented as definitive. | Binding rule | Phase 3, 10 |
| D3 | Dashboard shows only real run outputs; any demo/sample data is explicitly labelled. | Binding rule | Phase 9 |

## Change log

- 2026-10-11: Phase 0 initial register created. No amendments yet.
- 2026-10-11 (Phase 1): literature corpus of 40 verified papers completed
  (`reports/papers.csv`, `reports/gap_matrix.md`). A1/A2 (synthetic sim, abstract
  tasks) confirmed viable by DROO/FedDOVe/DRL-edge precedents. B2 (simulated
  clients) positioned as horizontal FRL per Qi et al. taxonomy; sim-vs-real gap
  remains open (U5). B3/C1–C4 binding constraints unchanged. C2/C3 parameters
  still provisional (U2). D1–D3 metric/reporting rules frozen in
  `docs/PHASE1_SPEC.md` §11. New uncertainties U1–U6 registered in spec §12;
  novelty stays a provisional hypothesis (gap G-statements, not claims).
- 2026-10-11 (Phase 2): U3 resolved — path-loss law, FIFO queue discipline,
  task/mobility distributions, K=5, neutral-trust placeholder documented in
  `environment/README.md`. A3/A4 now implemented. Found and documented one
  Phase 1 erratum (state dim 7+3K, not 8+3K) and one ambiguity (EQ-11
  latency/energy norms; provisional norms in smoke script only, calibration
  deferred to Phase 4). Python still 3.10.11 (no 3.11 available); code kept
  3.10/3.11-compatible.
- 2026-10-11 (Phase 2 validation fix): SPEC §4 formula corrected in place to
  `n = 7 + 3K` (history note retained); dimension declared once as
  `EnvConfig.obs_dim` and asserted on every observation; component-order tests
  added for K in {1,2,5,8}. Python 3.11.9 installed (winget), project `.venv`
  created on 3.11 — see validation report for install/test outcomes.
- 2026-10-11 (Phase 3): four baselines implemented on the real env API with
  documented tie-breaks/fallbacks (`baselines/`); shared metrics + multi-seed
  CSV/JSON evaluation (`scripts/run_baselines.py`); smoke-matrix results saved
  to `results/baselines_smoke.*` (synthetic only, no real-world claims).
  Fairness caveat recorded: per-link draws are consumed on demand, so exact
  connectivity can diverge across policies per seed (aggregated over seeds).
- 2026-10-12 (Phase 4): conventional DQN (MLP, replay, target net, masked
  epsilon-greedy) trained on real env transitions; EQ-11 reward with explicit
  provisional norms (10s, 10J, neutral trust); held-out eval seeds 10-12 vs
  all four baselines. DQN competitive with nearest_edge within noise — no
  superiority claimed. Training-curve variance partly reflects per-episode
  seed changes, documented in `rl/README.md`.
- 2026-10-12 (Phase 5): behavioural trust per SPEC §6 (`trust/` table, EQ-12
  β=(0.4,0.25,0.2,0.15), λ=0.9, neutral ageing ρ=0.99); env obs trust slot now
  per-candidate lookup, dim/order/mask unchanged; disconnects age neutrally,
  never punished; s2 carried (not zeroed) on path failure — documented minimal
  resolution. Additive `trust_aware` policy (threshold 0.3 fallback to local);
  exploratory comparison only, no attackers present, no malice claims.
- 2026-10-12 (Phase 6): federated DQN with standard FedAvg (`federated/`
  client/server/validation); n_i = round-local transitions; global distributed
  to online+target nets; staleness accepted+logged (no weighting — Phase 7);
  buffers never leave clients (tested); synthetic per-client heterogeneity
  recorded. Held-out: feddqn_global 0.707±0.101 vs local_alone 0.738±0.080,
  single_dqn 0.753±0.050, nearest_edge 0.840±0.080 — no superiority claimed.
  Phase 0 guard test rescoped to skeleton files (documented in-test).
