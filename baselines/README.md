# Phase 3 — Baseline Offloading Policies

> Non-learned reference policies evaluated through the real Phase 2
> environment API. No learning, federation, trust, attacks, or dashboard.

## Policy protocol

`policy(obs, mask, ctx) -> int`, with `ctx = {"candidates", "env"}`.
Only decision-time observables may be used (candidate distances, advertised
queues/freqs, obs vector, mask). Fallback for every policy: **action 0**.

| Policy | Rule | Tie-break | Fallback |
|---|---|---|---|
| `local_only` | always 0 | — | — |
| `random_valid` | uniform over mask-valid actions, seeded `Generator(seed)` | — (chance) | mask always contains 0 |
| `nearest_edge` | nearest `kind=="edge"` candidate | dist ↑, edge id ↑ | 0 if no edge eligible |
| `resource_aware` | argmax score over eligible candidates | score ↓, dist ↑, id ↑ | 0 if none eligible |

## Scoring rule (`candidate_score`)

```text
score = 0.5 * (1 - queue/queue_max) + 0.3 * (freq/10GHz) + 0.2 * (1 - dist/range)
```

Weights are provisional and documented (sensitivity is Phase 10 work).
Queue/CPU are *advertised* values — observable in this synthetic env; a real
deployment would need a signalling protocol (documented limitation).

## Fairness protocol

Every policy × seed starts from a fresh `env.reset(seed)`: identical task
stream, placement, mobility law, and dynamics law. Caveat (documented, not
hidden): per-link/per-return Bernoulli draws are consumed on demand, so exact
connectivity sequences can diverge across policies after the first differing
action. Multi-seed mean ± stdev absorbs this; it is inherent randomness, not
a workload difference. No env calculation was changed to favour any policy.

## Outputs

`scripts/run_baselines.py --config <yaml> --seeds 0,1,2 --out <prefix>`
writes `<prefix>.csv` (one row per policy × seed with config/seed metadata)
and `<prefix>.json` (rows + aggregates + reward caveat). Reward column uses
SPEC EQ-11 defaults with neutral trust and provisional norms — illustrative
until Phase 4. Synthetic results are not real-world evidence.
