# H2 against the standard robust aggregators (pre-specified)

`experiments/baseline_study.py`; raw results `evidence/baseline_study.*`, verdicts
`results/baseline_study/baseline_verdicts.json`.  Do-no-harm (DNH) minus each baseline,
completion, paired by seed, Holm-corrected across the nine tests.  Same seeds, protocol and
sign-flip attack as `docs/HYPOTHESES.md`; 24 seeds at 24 / 40 % attackers, 12 with no
attackers.  Multi-Krum assumes a Byzantine fraction of 20 % (the code default), i.e. it is
*not* told the true attacker ratio.

## Mean completion (%)

| Attackers | FedAvg | **do-no-harm** | Multi-Krum | coordinate median | trimmed mean |
|---|---|---|---|---|---|
| 0 % | 78.7 | 77.8 | 75.7 | 72.3 | 78.4 |
| 24 % | 70.6 | **76.6** | 76.7 | 61.1 | 70.1 |
| 40 % | 61.6 | **77.8** | 76.4 | 61.3 | 65.4 |

## DNH minus baseline (pp, 95 % CI, Holm p)

| Baseline | 0 % | 24 % | 40 % |
|---|---|---|---|
| Multi-Krum | +2.2 [-1.5, +5.8] (0.87) | -0.1 [-1.4, +1.3] (0.92) | +1.4 [-0.9, +3.7] (0.87) |
| coordinate median | +5.6 [+1.1, +10.0] (0.093, raw only) | **+15.5 [+11.9, +19.2] (<0.001)** | **+16.5 [+13.7, +19.3] (<0.001)** |
| trimmed mean | -0.6 [-1.7, +0.6] (0.87) | **+6.5 [+3.8, +9.3] (<0.001)** | **+12.4 [+8.9, +15.8] (<0.001)** |

## Reading

* **H2 holds against conventional robust aggregation too**, for the median and the trimmed
  mean: both are no better than FedAvg once 24-40 % of the clients are attackers (a
  coordinate-wise statistic cannot separate a coordinated minority at these shares).
* **DNH does not beat Multi-Krum; it ties it** at every ratio (differences within noise).
  Multi-Krum is a strong baseline here because the attack is a plain sign flip with a large
  norm.  The point estimate favours DNH with no attackers (+2.2 pp: Multi-Krum always discards
  a fixed share of honest clients) but it is not resolved.  A paper must say "matches
  Multi-Krum", not "outperforms the state of the art".
* The comparison is for the sign-flip attack only; the stealth / partial / scaling comparison
  is in `docs/DNH_DETECTOR.md` (DNH vs the deployed rule) and, for the stealth-capable v2 rule,
  below.
