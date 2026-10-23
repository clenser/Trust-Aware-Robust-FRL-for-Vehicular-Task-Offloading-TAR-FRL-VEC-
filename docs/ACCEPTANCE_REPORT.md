# Acceptance report

Status vocabulary: **PASS** - verified with the evidence cited; **FAIL** - checked
and not met; **BLOCKED** - cannot be verified here, with the exact blocker.  A
criterion that was not verified is not marked PASS.  A PASS says the *work* is
correct and documented; it does not say that the *scientific result* is
favourable (several results are negative - see section 3).

Active run: `run_20261023_022608_env_v21_coverage_features`
(`results/manifest.json`, 16 seeds, validation passed).  Test status: 335 passed, 0 failed on the final code (criterion 1).

---

## 1. Repository integrity and reproducibility - PASS

Evidence
* **Full suite, run to completion once** (before the last UI / documentation
  edits): 323 passed, 6 failed.  All 6 failures were in
  `tests/test_plot_artifacts.py` (the figure set grew from 6 to 10 and one
  verification test hard-coded 36 episode seeds); they were fixed and that file
  re-run: 32 passed.
* **Targeted tests after the later edits:** `tests/test_environment_v2.py`,
  `tests/test_dashboard_contract.py`, `tests/test_trust_geometry.py`,
  `tests/test_paper_claims_consistency.py` (6), the headless front-end harness,
  the real-browser script (35 checks) and the four live API scripts all pass.
* **Closing full-suite gate: PASSED.**  After the last edits the complete suite
  was run once more: **335 passed, 0 failed** (10 min 36 s).  An earlier attempt at
  this run had been killed by the host for low memory at about 22 % and was
  repeated after memory was freed.
* `tests/test_paper_claims_consistency.py` fails if a headline number in
  `README.md` / `docs/PAPER.md` differs from the active run.
* One transient failure of `tests/live/verify_live_autofl.py` occurred in the
  first sequential live run and did **not** reproduce in three later runs; its
  cause was not identified (timing-dependent live script).
* `--resume` was added after the host killed one benchmark for low memory; the
  first v2.0 run was completed that way and is flagged
  `resumed_from_partial_results`.  The active run was not interrupted.
* `trust_audit` -> 0 disagreements over 630 records; `verify_statistics` ->
  "every published aggregate, confidence interval and paired statistic is
  reproduced from the raw per-seed records".
* `results/trust_calibration.json` was regenerated after an overflow-safe rewrite
  of the detector's logistic and is identical in content (apart from timestamps).

Unresolved: Bit-identical reproduction across different CPUs / PyTorch builds is
not claimed.  Files: `docs/REPRODUCIBILITY.md`, `tests/`.

## 2. Simulation fidelity and documented assumptions - PASS (documented), fidelity limited

* `docs/ENVIRONMENT_SPEC.md` was rewritten to match the code; it fixes earlier
  mismatches (coverage radii, shadowing deviation, trust weights, evaluation seed
  offset, queue model, communication volume) and marks every element
  `[simulated]`, `[trace-replayed]` or `[measured]`.
* Audit findings that were **fixed**: RSU choice was silently overridden (actions
  1-3 interchangeable); `fl_min_clients` NameError under partial participation;
  `comm_overhead_kb` was a hard-coded constant; `poison_type` config ignored;
  trust bonus made offloading attractive by construction; reported latency mixed in
  the penalty value of failed tasks; vehicle service order biased to low ids.
* Audit findings that are **documented, not fixed**: single-user rate is almost
  distance-independent (rate cap), no interference, transmit energy excludes
  circuit power, deterministic queues with synchronous arrivals, one road /
  CPU class (`docs/LIMITATIONS.md` S1-S8).  Fidelity to a real network is **not**
  claimed.

## 3. Mobility and spatial placement - PASS

* Trace replay preserves source identities; absent vehicles are inactive and
  generate nothing (`tests/test_ngsim_ingestion.py`, `tests/test_blocker_fixes.py`).
* Viewport geometry (road length, lanes, RSU position and coverage interval) is
  taken from the backend `scene`; coverage is the exact road interval from the 3-D
  distance (`tests/test_dashboard_contract.py`, `tests/browser_verification.py` §5, 7b, 8).
* RSU placement is user-configurable only for the procedural highway, validated
  (range, count, overlap) and rejected for trace replay (`tests/test_environment_v2.py`;
  browser check "backend uses the configured RSU positions", "overlapping RSUs rejected").
* Verified with the authentic NGSIM file in a real browser: trace geometry
  681 m, RSU inputs disabled, active trace vehicles present.
* Memory: the 4.1 M-row trace loads in 2.3 s / 0.3 GB after decimation to the
  simulation step (positions unchanged; ingestion tests pass).
* Not done: lane changes in the procedural mobility, SUMO/Veins coupling
  (assessed in `docs/RELATED_WORK.md`, not integrated).

## 4. Task, communication and energy model consistency - PASS (consistent, simplified)

* One success criterion for local, RSU and V2V execution; failure reasons recorded.
* Shared-uplink model (100 MHz cell, 20 MHz per UE, 2.7 bit/s/Hz cap) with unit
  tests (`TestUplinkContention`); transmit energy = radiated power x airtime.
* Energy: local execution is about 31x cheaper per task than the learned policy
  (0.0004 vs 0.0127 J), reported as a limitation, not hidden.
* Not modelled: arrival jitter inside a slot, task dependencies, result download.

## 5. RL training and FL lifecycle - PASS

* Every round records attempted / uploaded / accepted / rejected, status in
  {`model_updated`, `model_unchanged`, `no_model_produced`,
  `not_attempted_insufficient_updates`}, gradient steps actually taken, mean loss
  of clients that optimised, and measured payload bytes
  (`tests/test_environment_v2.py::TestFederatedRunnerLifecycle`).  All 480 rounds of
  the proposed method in the main comparison were `model_updated`.
* Mean training loss fell from 4.98 (round 1) to 2.71 (round 30); 352 gradient
  steps per round across 25 clients.
* Initialisation, data access and the evaluation protocol are explicit
  (`docs/METHODOLOGY.md` §8); evaluation is inert (parameter hash before/after).
* Partial participation and connectivity-gated uploads are implemented and
  exercised (experiment G).
* Unresolved: 480 gradient steps per client is a small budget (a pilot gave more at
  32 epochs on the earlier features); the discount factor was not swept.

## 6. Trust detection and attack robustness - PASS as an evaluation; the detector does **not** meet its operational criterion

* Ground truth, detector decision and task outcome are separate; the decision
  function has no access to roles (`tests/test_trust_validation.py`).
* Evaluated against full, partial, norm-matched and intermittent sign flips and free
  riding, with benign small-update and heterogeneous clients, on identical updates for
  five detectors (deployed, Krum, median-cosine, MAD-distance, none).
* Results (real federated updates, 16 seeds): TPR 0.991, **FPR 0.042** (the
  operational criterion FPR <= 0.01 is **not met**); false exclusions rise to about 8 %
  late in training; partial / norm-matched / intermittent attacks TPR 0.53 / 0.73 / 0.77.
* A MAD-distance rule matches the deployed detector on held-out scenarios; the joint
  threshold is degenerate at default weights (`tests/test_trust_geometry.py`);
  behavioural trust adds nothing measurable.  Adaptive attackers are out of scope.
* Identifiability limits are reported, not hidden (`docs/LIMITATIONS.md` T1-T9).

## 7. Statistical validity and baseline fairness - PASS, with an explicit power limit

* Matched evaluation episodes, 16 seeds (sweeps 6), paired differences with t-based
  CIs, paired t-tests, Holm correction over the 8 pre-specified claims, minimum
  detectable effect printed (about 5-6 pp).
* Verdicts are mechanical (`experiments/claims_report.py`): only **H6**
  (federation beats independent learning, +12.7 pp) is supported after correction;
  H1 (attack impact, raw p = 0.009) fails Holm (0.061).
* Reference rows (clean FedAvg, oracle exclusion) frame the downstream question;
  all learned methods share architecture, optimiser, budget and seeds.
* Limits: the design cannot resolve a 2-3 pp defence benefit; sweeps are exploratory
  (6 seeds, no multiplicity correction); Wilson intervals on pooled detection
  counts are optimistic.

## 8. UI quality and correspondence with backend state - PASS (Chromium only)

* `tests/browser_verification.py` drives the real page in Chromium: **35 checks, 0
  failed** - load without console errors, start / pause / resume / reset, advancing
  step counter, **two tabs do not speed the simulation up**, backend-derived scene,
  client- and server-side validation, new session with 12 vehicles, RSU placement,
  authentic-trace replay, Research tab with the mechanically computed claim
  verdicts, no horizontal overflow at 1600 / 1280 / 900 / 390 px, theme toggle.
  Screenshots: `browser_check/`.
* Headless harness (`tests/frontend_behaviour.js`, `tests/frontend_resilience.js`)
  and live API scripts (`tests/live/`) pass.
* The viewport draws only backend values; offloading links only for reported events;
  ground truth is an opt-in overlay; the live latency is over completed tasks; the
  Research tab never mixes live numbers with benchmark results; empty and loading
  states and error banners exist.
* Not verified: Firefox / Safari, screen-reader use, very large vehicle counts
  (backend range is capped at 60).

## 9. Experiment provenance and artefact integrity - PASS, with two recorded gaps

* `results/manifest.json` accounts for every run directory: one active run; the first
  v2.0 full run and the earlier runs are `superseded` and kept; smoke and aborted
  directories are marked and must not be cited.
* Per run: effective configuration, seeds, dataset SHA-256, training protocol,
  environment model, limitations, figure manifest with source hashes; the figures
  are generated from the raw records and cross-checked.
* Gaps: (a) the active run's recorded code fingerprint omits
  `trust/alt_detectors.py` and `experiments/experiment_runner.py` (added to the list
  afterwards); (b) `web/app.py` changed after the run (claims API, form sync) - none
  of the other 16 fingerprinted files changed (checked).

## 10. Research novelty and literature positioning - PASS (no novelty claimed)

* `docs/RELATED_WORK.md`: matrix of papers whose records were read in this project
  (abstract level), the narrowed gap (downstream effect of client exclusion on a
  vehicular offloading policy), and an explicit list of what is **not** novel.
* Not claimed: a new aggregation algorithm, a new trust formula, real-world validity.
* Caveats: Krum, FedAvg, IDM, DQN-family and statistics references are cited from
  background knowledge and flagged as not re-verified; a 2026 federated-meta-DRL
  preprint and several IoV reputation papers were seen only in search listings and
  are not used as support.

## 11. Research-paper completeness and claim accuracy - PASS (draft), venues unverified

* `docs/PAPER.md` has the abstract, gap, contributions, related work, system model,
  threat model, equations, protocol, results with uncertainty, ablations,
  negative results, validity threats, reproducibility and references; its numbers
  are guarded by `tests/test_paper_claims_consistency.py`.
* BLOCKED (venues): the submission deadlines and page limits of the candidate
  venues were **not** verified (search results gave the ComSoc listing for VNC 2026
  and journal scope statements only).  Minimum additional work: read each venue's
  current call for papers.  No acceptance is implied.

---

## Summary of changes actually implemented

Environment model 2.1 (honoured RSU choice, shared uplink, neutral reward, RSU load and
coverage-relative state, seeded service order), corrected FL lifecycle accounting and
measured communication, new attacks (partial, norm-matched, intermittent), shadow
detectors and an oracle bound, resumable parallel benchmark with a persistent
worker pool, data-driven claims report with Holm correction, detector comparison,
four new figures, a rewritten console (road viewport, validated configuration,
research view with verdicts), a real-browser verification script, and rewritten
documentation.

## Most important scientific findings

1. Federation helps (+12.7 pp); the attack hurts undefended FedAvg by about 6 pp
   (not significant after Holm); an oracle would recover about 5 pp; the trust-aware
   defence recovers about 2 pp (not established).
2. The defence clearly helps at 40 % attackers (+17.9 pp) and appears to cost about
   10 pp with none (not established at n = 6).
3. The detector's false-exclusion rate (4 %, rising to about 8 %) is higher than on
   synthetic data; a MAD-distance rule matches it; behavioural trust adds nothing.
4. The "RL beats a heuristic" conclusion depends on simulator semantics and on the
   state representation (+3 / -12 / -0.4 pp across variants).

## Before / after for changes that affect measured performance

| Change | Measure | Before | After |
|---|---|---|---|
| Honour RSU choice + uplink contention + neutral reward (v1 -> v2.0) | Learned trust-aware minus Nearest-RSU | +3.0 pp (v1) | about -12 pp (v2.0) |
| Coverage-relative state (v2.0 -> v2.1) | Trust-aware completion | 62.9 % | 74.6 % (+11.7 pp, CI [+7.0, +16.4]) |
| Same | Clean FedAvg completion | 66.3 % | 78.5 % |
| Same | Non-federated D3QN | 66.6 % | 65.7 % (unchanged) |
| Completed-task latency | Reported latency | included failure penalty | completed tasks only |

## Commands to reproduce the final results

```bash
pip install -r requirements.txt
python -m experiments.experiment_runner --label env_v21_coverage_features     # ~2.5 h, VANNET_JOBS=6
python -m experiments.trust_audit results/benchmark_results.json results/trust_audit.json
python -m experiments.verify_statistics results/benchmark_results.json results/statistics_verification.json
python -m experiments.claims_report
python -m experiments.detector_comparison
python -m experiments.research_figures
python -m pytest tests -q -p no:cacheprovider
python run_simulation.py            # then: python tests/browser_verification.py
```

## Locations

Active run: `results/run_20261023_022608_env_v21_coverage_features/` (pointer
`results/benchmark_results.json`); verdicts `results/claims_report.md`; figures
`plots/` (10); paper `docs/PAPER.md`; browser screenshots `browser_check/`.

## Remaining limitations that could affect peer review

Simulation-only evidence; a nearly distance-flat channel; deterministic queues;
one road and CPU class; short evaluation horizons; scripted non-adaptive
attackers; a post-hoc state-representation change; sweeps with 6 seeds; a design
that cannot resolve a defence benefit below about 5 pp at the default attack
ratio; novelty limited to the evaluation protocol and the characterisation.

## What the evidence genuinely supports

A correct, reproducible simulator and evaluation protocol; the finding that
federation helps and that a sign-flip attack degrades an undefended policy by about
6 pp (before correction); that excluding poisoned clients helps clearly only under
heavy contamination; that the studied trust detector finds simple poisoning but
pays for it in false exclusions and gains nothing from behavioural trust.  It does
**not** support the claim that trust-aware federated RL improves vehicular task
offloading at the default attack ratio, nor any real-world claim.

## Addendum: supplementary experiments

Supplementary run `run_20261023_100858_supplementary_edge_node_staleness` (6 seeds, exploratory, Holm across S1-S8):

| Claim | Paired difference (pp, 95 % CI) | Verdict |
|---|---|---|
| S1 trust features vs masked, fixed faulty RSU | -0.01 [-2.17, +2.16] | not established |
| S2 learned vs Nearest-RSU, faulty RSU | +8.06 [+2.94, +13.19], Holm p = 0.069 | not established after Holm |
| S3 learned vs trust-greedy heuristic | +0.40 [-3.94, +4.73] | not established |
| S4 faulty RSU vs no fault | -5.07 [-9.71, -0.43], Holm p = 0.22 | not established after Holm |
| S5 staleness discount vs none | +1.55 [-4.42, +7.52] | not established |
| S6 trust-aware vs FedAvg, intermittent connectivity | +2.84 [-9.49, +15.16] | not established |
| S7 trust features vs masked, rotating fault | -1.53 [-3.32, +0.25] | not established |
| S8 learned vs Resource-Aware, rotating fault | -10.07 [-11.53, -8.61], Holm p < 0.001 | **contradicted** |

Reading: in these experiments behavioural trust features in the state give the
learned policy **no measurable benefit**, a simple trust-aware greedy heuristic
matches it, and when the faulty RSU moves the learned policy (about 60 % completion)
is clearly worse than the Resource-Aware heuristic (about 69 %).  The framework's
only supported headline result remains H6 (federation gain over non-federated
learning).  The dual-tier (behavioural + model-update) trust hypothesis is **not**
supported by this evidence and must not be claimed as a contribution; what can be
claimed is the evaluation methodology, the corrected labels for unreliable-but-honest
vehicles, and the negative results with their limits.  Possible reasons (untested):
short training budget, a state that is too coarse to separate a transient fault from
congestion.