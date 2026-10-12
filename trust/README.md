# Phase 5 — Dynamic Behavioural Trust

> Replaces the neutral-trust placeholder with evidence-driven scores.
> No malice labels: disconnection ages trust neutrally, never punitively.

## Equations (SPEC §6, implemented faithfully in `trust/behavioural.py`)

Per observed execution on node `i`:
```text
s1 = success ? 1 : 0                                   (verified outcome)
s2 = path_ok ? clamp(1 - L/τ) : last_s2[i] (else 0.5)  (timeliness; missing carried, not punished)
s3 = clamp(expected_exe / actual_exe)                  (advertised-vs-observed resources)
s4 = path_ok ? 1 : 0                                   (communication reliability)
T_beh = 0.4·s1 + 0.25·s2 + 0.2·s3 + 0.15·s4            (EQ-12)
T(i,t) = 0.9·T(i,t-1) + 0.1·T_beh                      (EQ-13 temporal update)
idle:  T = 0.99·T + 0.01·0.5                           (EQ-13 neutral ageing toward prior)
```

`expected_exe` uses the destination queue/freq captured at decision time
(before dynamics advance); `actual_exe` is the realised execution part.
Defaults (`β`, `λ=0.9`, `ρ=0.99`, prior `0.5`) live in `EnvConfig`
(`trust_beta/lambda/rho/prior`) and are configurable, not hard-coded.

## Minimal SPEC resolution (explicit)

`s2` on path failure is *missing* evidence (result never returned), so the
node keeps its last timeliness instead of scoring 0 — otherwise communication
failures would masquerade as slowness. `s1=0`/`s4=0` still record the failure
factually. Cause uncertainty (congestion vs node slowness on deadline misses)
is documented, not resolved by fiat.

## Observation integration (no redesign)

Dimension stays **7+3K**, order unchanged, mask unchanged. The per-candidate
trust slot now carries `env.trust.get(kind, id)` — the *candidate's own*
score, so nodes are never confused. `reset()` rebuilds the table (prior 0.5,
history cleared); same seed reproduces trust trajectories.

## Trust-aware policy (exploratory)

`trust_aware` (additive; the four Phase 3 policies are byte-untouched):
picks the eligible candidate with max trust, tie-break (trust ↓, dist ↑,
id ↑); falls back to local if none eligible **or** best trust <
`trust_threshold` (0.3). Exploratory comparison only — no robustness claims
(there are no attackers in Phase 5).

## Logging

`scripts/log_trust.py` writes per-task snapshots, per-update events with
evidence (`s1..s4`, before/after), a JSON summary, and a trust-over-time plot
from the actual run.
