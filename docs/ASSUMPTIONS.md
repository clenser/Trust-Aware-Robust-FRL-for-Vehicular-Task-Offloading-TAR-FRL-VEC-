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
