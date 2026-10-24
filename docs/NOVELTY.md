# Novelty: what this work contributes, and what it does not (24 Oct 2026)

Plain statement, written against the literature check in `docs/LITERATURE_GAP.md` and
`docs/LITERATURE_REVIEW.md` (a keyword-screened local collection plus abstract-level searches,
**not** a systematic database review; every "we found no ..." below is limited by that).

## Contributions that the evidence supports

1. **A downstream-impact evaluation of defences for a federated offloading *policy*.**  Each
   defence is judged by task completion against no-attack, undefended and oracle references, on
   paired seeds, with pre-specified Holm-corrected tests (`docs/HYPOTHESES.md`,
   `docs/BASELINES.md`).  In the located literature the federated-offloading papers assume
   benign clients and the robust-FL papers measure classification accuracy; we found no study
   that does both.  (Surveys list both as open problems.)
2. **A false-positive-budgeted exclusion rule ("do-no-harm") built on absolute, not relative,
   statistics.**  The deployed relative-z-score rule excludes 10.3 % of honest clients with no
   attackers; the new rule 1.7 %, removing the defence's no-attacker cost (-6.4 pp -> -0.8 pp)
   while keeping the gain under attack (+6.1 pp at 24 %, +16.2 pp at 40 % over FedAvg, 24 paired
   seeds).  It ties Multi-Krum and beats coordinate median, trimmed mean and an adapted FLTrust
   (`docs/BASELINES.md`).  A **label-free v2** calibrates the bounds on attacker-free runs only
   and raises detection of norm-matched attackers from 0.51 to 0.85 at < 1 % false exclusion
   (`docs/DNH_DETECTOR.md` section 7).
3. **Measured threat-model and simulator sensitivity.**  Behavioural trust is worthless when the
   attackers are honest in the data plane and worth about 1 pp when they sabotage V2V tasks; the
   "learned policy vs heuristic" verdict swings by up to 12 pp with simulator semantics and by
   up to 43 pp between the allocation-view and measured edge-load models
   (`docs/EDGE_TRACE_USE.md`).  A correction found during this work (the allocation-view load
   overstated measured utilisation ten-fold) is itself a reproducibility lesson.
4. **A reproducible platform**: simulator, trace-calibrated edge side (Alibaba trace), 3D console,
   ~400 tests, raw evidence with checksums.

## What is NOT novel (do not claim it)

* Excluding clients by robust statistics (Krum, FLAME, FLTrust) and smoothing client reputation
  over rounds (RepuNet, FLARE) are published; so is the observation that binary exclusion creates
  false positives (FLARE).  The rule's novelty is the combination "absolute statistics +
  false-positive budget + downstream evaluation in an offloading task", nothing more.
* The do-no-harm rule **ties** Multi-Krum; it is not shown to be better than the strongest classic
  baseline.  Its advantages over it are a smaller no-attacker cost (point estimate only,
  unresolved) and no need to assume an attacker fraction.
* The trust formula combining behavioural and model trust is a recombination of known statistics.

## Hypothesis scoreboard

H2, H3, H4, H5 supported with the stated scope; H1 only for completion (latency effect <= 3 ms).
RQ3 (trust inside the learned policy) is not supported.

## Limits a reviewer will raise (all stated in the paper)

Simulation only; sign-flip is the main attack (partial sign-flip is not detected, norm-matched is
detected but barely hurts FedAvg, so no downstream gain is shown for it); FLTrust was adapted and
performs poorly in this setting; 25 vehicles in the main studies (at 50 and 100 vehicles the gain over FedAvg
shrinks to +2.8 / +1.9 pp and is unresolved, and the rule's aggregation cost grows to 1-3 s per round -
`docs/HYPOTHESES.md`); the learner beats the best heuristic only in the benchmark environment and
ties it under measured edge load.
