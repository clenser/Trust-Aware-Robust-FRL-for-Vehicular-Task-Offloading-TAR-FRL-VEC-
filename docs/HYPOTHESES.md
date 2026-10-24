# Roadmap hypotheses H1-H5: pre-specified confirmation study

Run 23-24 Oct 2026 (`experiments/hypothesis_study.py`; results in
`results/hypothesis_study/`, raw data in `evidence/`, Fig. 16).  The ten
comparisons and the Holm correction across them were fixed in the script before any
cell was run.  "Trust-aware" means the do-no-harm rule (`docs/DNH_DETECTOR.md`);
"non-trust-aware FRL" means FedAvg.  Paired by seed: 24 seeds for the 24 % and 40 %
ratios, 12 for the offline-clients and mobility conditions.

## Verdicts

| # | Hypothesis (roadmap wording, scope tested) | Test | Paired difference [95 % CI] | Holm p | Verdict |
|---|---|---|---|---|---|
| P1 | **H2** robustness under increasing malicious participation | completion, DNH - FedAvg, 40 % attackers | +16.2 pp [+12.0, +20.4] | <0.001 | **supported** |
| P2 | H2 | same, 24 % (default) | +6.1 pp [+2.7, +9.4] | 0.007 | **supported** |
| P3 | **H1** lower latency than non-trust-aware FRL | completed-task latency, 40 % | -3.3 ms [-6.3, -0.4] | 0.089 | raw CI only |
| P4 | H1 | same, 24 % | -0.5 ms [-1.4, +0.4] | 0.277 | not established |
| P5 | **H3** joint beats behavioural trust alone | completion, joint - behavioural-only, 40 % | +19.7 pp [+12.3, +27.1] | <0.001 | **supported** |
| P6 | H3 | same, 24 % | +5.3 pp [+2.5, +8.1] | 0.005 | **supported** |
| P7 | H3 | DNH - behavioural-only, 40 % | +22.6 pp [+15.6, +29.5] | <0.001 | **supported** |
| P8 | **H4** advantage under varied conditions | DNH - FedAvg, 40 % clients offline per round | +9.6 pp [+1.1, +18.1] | 0.089 | raw CI only |
| P9 | H4 | synthetic NGSIM-style mobility | +9.9 pp [+3.6, +16.1] | 0.020 | **supported** |
| P10 | H4 | authentic NGSIM US-101 replay | +8.5 pp [+4.1, +13.0] | 0.007 | **supported** |
| - | **H5** no impractical overhead | aggregation time, 25 clients, idle machine | 136 ms vs 12 ms FedAvg; no extra upload | - | **met** |

Seven of ten tests are supported after Holm correction, two only on the raw
interval, one is not established.

## Read these before quoting them

* **H1 is supported for task completion, not for latency.**  Latency of completed
  tasks differs by only 0.5-3 ms (about 3 % at 40 %, nothing at 24 %).  The
  hypothesis wording ("reduces average task latency") is therefore *not* established
  in the strict sense; the benefit comes through reliability, not speed.
* **H3 is supported literally but weakly informative.**  Behavioural trust alone has
  no signal about model poisoning, so it is no defence at all (55.3 % at 40 %
  attackers, *below* FedAvg's 61.6 %).  The earlier finding that model-update trust
  alone matches the joint score, and that behavioural trust adds about 1 pp even when
  attackers sabotage V2V tasks, still stands.  The result shows the model-update tier
  is necessary, not that the two tiers together are better than the model tier alone.
* **The rule was designed after the earlier benchmark.**  The deployed joint rule
  failed H2 in that benchmark; the do-no-harm rule was built in response.  Its
  weights were fitted on separate recorded runs (seeds 1000-1001); the study seeds
  (42-1515, then +1000) are disjoint from those, and no tuning followed the first
  closed-loop study, but 12 of the 24 study seeds had been seen in that study.
* **Scope.**  The sign-flip attack, 25 vehicles, simulation only.  Norm-matched and
  partial attacks are not detected (`docs/DNH_DETECTOR.md` section 4).  FedAvg is the
  comparator; the comparison with Multi-Krum, median and trimmed-mean in this
  rule's own closed loop has not been run.
* **H4 is supported under the conditions tested** (offline clients, two mobility
  sources), not under "any" mobility or connectivity.

## What this changes

H2, H3 and H4 can be claimed with these scopes; H5 can be claimed; H1 can be claimed
only as a reliability benefit.  RQ3 (trust inside the RL decision) remains
unsupported (`docs/DNH_DETECTOR.md` section 6).

## Addendum (24 Oct 2026): robustness checks

* **Against robust aggregators** (`docs/BASELINES.md`): DNH ties Multi-Krum at 0 / 24 / 40 % attackers
  (differences within noise), and beats coordinate median (+15.5 / +16.5 pp at 24 / 40 %) and trimmed mean
  (+6.5 / +12.4 pp) after Holm correction.  H2 therefore holds against conventional robust aggregation for
  the median and trimmed mean, and is a tie against Multi-Krum.
* **Across trace-calibrated edge deployments** (`experiments/deployment_study.py`; measured-utilisation load,
  per-seed capacity and load draws; 24 % attackers; 12 seeds): DNH - FedAvg = +4.4 pp [+0.5, +8.3] at 05:00
  and +4.3 pp [+0.3, +8.4] at 20:00 (Holm p = 0.059 each: raw interval only).  Same size and sign as the
  24-seed estimate (+6.1 pp); 12 seeds are too few to clear the correction.
* **Stealth attackers**: see `docs/DNH_DETECTOR.md` section 7.
* **Larger fleets** (`experiments/scale_study.py`; 24 % attackers; 16 paired seeds (first run
  used 8, extended with the same protocol); RSU capacity unchanged):

| Fleet | FedAvg | do-no-harm | Multi-Krum | DNH - FedAvg (pp, CI, Holm p) | DNH - Multi-Krum |
|---|---|---|---|---|---|
| 50 | 61.6 | 63.0 | 64.6 | +1.3 [-1.0, +3.6] (0.72) | -1.6 [-3.8, +0.6] |
| 100 | 59.0 | 60.4 | 61.1 | +1.4 [-1.2, +4.0] (0.72) | -0.7 [-2.2, +0.7] |

  The sign is preserved but the effect is small and **not resolved** at 16 seeds (the 8-seed point estimates,
  +2.8 / +1.9 pp, shrank to +1.3 / +1.4 when seeds were added), and the pattern with Multi-Krum (a tie) is unchanged.
  Read this as: no evidence that the rule's advantage over FedAvg holds beyond 25 vehicles, not evidence that it fails.  Larger fleets also load the fixed RSUs more (completion falls from about 71 % at 25 vehicles to
  59-62 %), so this mixes fleet size with edge load.  Aggregation cost (loaded machine, median per round): do-no-harm
  1.1 s at 50 and 3.5 s at 100 vehicles vs FedAvg 0.09 / 0.17 s and Multi-Krum 0.20 / 0.74 s - the rule is O(n^2 d) and
  is the slowest of the three; at about 100 clients it approaches the round period.
