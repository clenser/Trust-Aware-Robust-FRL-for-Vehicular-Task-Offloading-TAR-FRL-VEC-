# Phase 2 — Synthetic Vehicular-Edge Environment

> Implements SPEC §§1–4 (EQ-1–EQ-10). Resolves SPEC uncertainty **U3**
> (path-loss law, queue discipline, task distributions, mobility ranges).
> Reward (EQ-11) is intentionally NOT computed here — that is Phase 4.

## Architecture

```text
VehicularEdgeEnv (vanet_env.py)   orchestrates one task-step at a time
├── EnvConfig (config.py)         YAML scenario + documented synthetic defaults
├── Task / TaskGenerator (task.py)
├── Vehicle / EdgeNode            nodes with FIFO cycle queues
├── MobilityModel (mobility.py)   EQ-1: 1-D road, wrap-around
└── network.py                    EQ-2 eligibility, EQ-3 Shannon rate
```

All latency/energy/outcome math delegates to `specs/formulas.py`
(EQ-4–EQ-8) — never duplicated. Public API: `VehicularEdgeEnv`,
`reset(seed)`, `current_observation() -> (obs, mask)`, `step(action) ->
StepResult`, `run_episode(policy)`, `InvalidActionError`.

## Erratum on SPEC §4 (resolved in validation fix)

The spec originally stated state dimension `n = 8 + 3K` but enumerated
4 (task) + 2 (local) + 3K (candidates) + 1 (eligible count) = **7 + 3K**
components. The formula is corrected to `n = 7 + 3K` in
`docs/PHASE1_SPEC.md` (history note retained there). No component was added
or removed. The dimension is declared once in code as
`EnvConfig.obs_dim` (`OBS_BASE_DIM + 3*top_k`) and asserted on every
observation build; the action count is `EnvConfig.action_dim = 1 + K`.

## Step semantics (one task per step)

1. Advance clock to task arrival; drain FIFO queues; drift utilisation;
   flap node availability (p=0.02); move vehicles (EQ-1).
2. Candidate set (EQ-2) is fixed at observation time and cached per task,
   so the mask shown to the policy matches what `step` executes.
3. Execute local (EQ-4 skipped, `L_tx=0`) or remote (EQ-4–EQ-7); result
   counts only if the return path survives (EQ-8, synthetic Bernoulli draw).
4. Enqueue task cycles on the destination; emit next task (requesters cycle
   round-robin over vehicles).

## Synthetic assumptions (placeholders, not measurements)

| Process | Law | Defaults |
|---|---|---|
| Speeds | Uniform per level | low 5–12, med 12–22, high 22–33 m/s |
| Road | 1-D torus segment | 2000 m, wrapped distance |
| Arrivals | Exponential gaps | mean 1.0 / 0.5 / 0.25 s (low/med/high load) |
| Data size | Uniform(1,10) Mbit × load mult | mult 0.5 / 1.0 / 2.0 |
| Intensity | Uniform(200,2000) cycles/bit | — |
| Deadline | slack × local-exec time | slack ∼ Uniform(1.5,3) |
| Priority | Categorical | {1:0.5, 2:0.3, 3:0.2} |
| Vehicle CPU | Uniform | 1–3 GHz, util ∼ U(0,0.7) + N(0,0.05) drift |
| Edge CPU | Uniform | 5–10 GHz, evenly spaced RSUs ± 50 m jitter |
| SNR | `P·k0/((d+1)^γ·N)` | k0=2e-3, γ=3, N=1e-9 W, floor 1e-3 |
| Link availability | Bernoulli per link per step | p=0.95; node flap p=0.02/step |
| Queues | FIFO cycles, `wait=q/f` | drain `q −= f·dt` |
| Candidates | nearest-first, top-K | K=5; trust slot = neutral 0.5 (Phase 5) |

Units: metres, seconds, bits, Hz, watts, joules, cycles. All RNG flows
through one `numpy.random.Generator(seed)`; same seed ⇒ identical episode.

## Known spec ambiguity (explicit, not silent)

Phase 1 EQ-11 normalises latency/energy without fixing bounds. The env does
not compute rewards; `scripts/run_smoke.py` illustrates EQ-11 with provisional
norms (10 s, 10 J) and neutral trust, clearly labelled. Final reward
calibration belongs to Phase 4 (sweep in Phase 10).
