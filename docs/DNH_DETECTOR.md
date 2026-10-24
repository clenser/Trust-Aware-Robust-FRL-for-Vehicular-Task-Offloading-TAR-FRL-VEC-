# Do-no-harm exclusion and the learner study (23 Oct 2026)

Both studies were run with success criteria written **before** looking at any
result (`experiments/dnh_study.py`, `experiments/learner_study.py` docstrings).
This page reports what was met and what was not.

## 1. Why a new exclusion rule

The deployed detector scores every client with robust z-scores against its peers.
A z-score is relative: as honest updates become alike during training the spread
shrinks and a harmless client becomes a "3-sigma outlier" (honest clients 9, 15,
17 and 22 in one trace had only mildly negative agreement, -0.05 to -0.2, and
norm ratios of 1.3-2, yet were flagged for most of training).  In the closed loop
the deployed rule excludes 10.3 % of honest clients with no attackers, 4.6 % at
24 % attackers.

## 2. The rule (`trust/dnh_detector.py`)

* Two **absolute** statistics per client: `ratio` (update norm / median norm) and
  `cos_loo` (cosine with the leave-one-out trimmed mean of the other clients).
  On the recorded real updates at 24 % and 40 % attackers their ROC AUC for
  sign-flip is 0.995 and 0.947 respectively; the pairwise-agreement statistic used
  by the deployed rule reaches only 0.84.
* Logistic score `s = b0 + b1*log2(ratio) - b2*cos_loo`, fitted on four recorded
  runs (seeds 1000-1001) and **never on the evaluation seeds** (42-1515).
* Smoothed with an EWMA (0.5) over rounds; excluded while the evidence exceeds a
  threshold set at the 99.5th percentile of the evidence honest clients reach
  (a 0.5 % false-positive budget).
* Guards: updates that are (almost) zero are free riders (absolute norm test);
  at most 45 % of the participants can be excluded in a round.
* Survivors are aggregated by plain FedAvg.  No role label is read.

**What is not new.**  Reputation smoothed over rounds and the false-positive
problem of binary exclusion are published (RepuNet, FLARE; see
`docs/LITERATURE_GAP.md`).  The contribution claimed here is limited to: absolute
harm statistics with an explicit false-positive budget, evaluated by *downstream
task completion* in a federated offloading task.

## 3. Closed-loop results (12 seeds, paired, sign-flip attack)

| Attackers | Method | Completion % | vs FedAvg (paired, 95 % CI) | Honest clients excluded |
|---|---|---|---|---|
| 0 % | FedAvg | 78.7 | - | - |
| 0 % | deployed trust-aware | 72.3 | -6.4 pp [-13.3, +0.5] | 10.3 % |
| 0 % | **do-no-harm** | 77.8 | **-0.8 pp [-1.9, +0.3]** | 1.7 % |
| 24 % | FedAvg | 73.4 | - | - |
| 24 % | deployed trust-aware | 75.0 | +1.6 pp [-3.3, +6.4] | 4.6 % |
| 24 % | **do-no-harm** | 76.4 | +3.0 pp [-3.8, +9.8] | 0.6 % |
| 40 % | FedAvg | 61.0 | - | - |
| 40 % | deployed trust-aware | 77.0 | +16.0 pp [+8.9, +23.1] | 0.5 % |
| 40 % | **do-no-harm** | 78.6 | **+17.6 pp [+10.6, +24.6]** | 1.5 % |

do-no-harm minus trust-aware: +5.6 pp [-0.6, +11.8] (0 %), +1.4 pp [-2.8, +5.7]
(24 %), +1.6 pp [-0.3, +3.5] (40 %).  Detection TPR: 0.996 at 24 %, 0.985 at 40 %.

**Criteria.**  (S1) cost at 0 % attackers within 2 pp of FedAvg: **met**.  (S2)
not below the deployed rule and above FedAvg with a CI excluding zero at 40 %:
**met**.  At 24 % the gain over FedAvg (+3.0 pp) is **not resolved** by 12 seeds;
the oracle headroom there is only about 5 pp, so no detector can show a large
effect.  The improvement over the deployed rule is not significant at any single
ratio; its robust properties are the 6-10x lower false-exclusion rate and the
removal of the no-attack cost.

## 4. Other attack families (6 seeds, 24 %)

| Attack | FedAvg | deployed | do-no-harm | TPR deployed / dnh | Reading |
|---|---|---|---|---|---|
| norm-matched sign flip ("stealth") | 74.5 | 70.2 (-4.3) | 74.4 (-0.1) | 0.73 / 0.51 | neither detects the poisoners; the deployed rule's false exclusions hurt, the new rule does no harm |
| partial sign flip (50 % of coordinates) | 76.6 | 74.1 (-2.5) | 76.4 (-0.2) | 0.56 / 0.50 | same |
| scaling x3 | 78.9 | 76.7 (-2.2) | 77.2 (-1.7 [-3.1, -0.3]) | 1.0 / 1.0 | both detect it; excluding all attackers (and free riders) costs 1.7 pp against FedAvg, which this mild attack barely hurts |

A TPR of about 0.5 means only the free riders were caught.  **The rule does not
defend against norm-matched or partial attacks** (related to the "Normalized attack" of
Fang et al., which targets the angle of updates); it only avoids making things worse.

## 5. Overhead (idle machine, 25 clients, 36,614 parameters)

Aggregation time per round: FedAvg 12 ms, deployed trust-aware 52 ms, do-no-harm
136 ms, Multi-Krum 18 ms, coordinate median 16 ms.  A round occurs about every
10 simulated seconds and the upload size is unchanged, so the overhead is 124 ms
per round.  (The "532 ms vs 38 ms" in the roadmap table was measured on a loaded
machine; the numbers above are the idle-machine ones and are not comparable
to it.)

## 6. Learner study

Question: does giving the federated DQN the heuristics' own delay estimates
(state v2.2) and bandit targets (gamma = 0) let it beat the best simple
heuristic?  12 seeds, paired against Resource-Aware.

| Environment | learned (est + bandit) vs Resource-Aware | same, trust features hidden |
|---|---|---|
| benchmark | **+6.6 pp [+3.6, +9.6]** | +6.8 pp [+3.6, +10.0] |
| rotating faulty RSU | -0.4 pp [-2.9, +2.0] | -1.1 pp [-3.6, +1.5] |
| edge-trace peak hour | -7.6 pp [-13.4, -1.8] (-6.4 pp with the RSU load observable) | -9.5 pp [-15.7, -3.2] |
| data-plane attack + V2V | +0.0 pp [-4.5, +4.5] | -0.3 pp [-4.9, +4.2] |

Pilot (6 seeds, benchmark): the original learner +1.6 pp [-4.8, +8.1], with
delay estimates only (gamma 0.95) +5.7 pp [+1.8, +9.6], with bandit targets +8.5 pp.

*Correction (24 Oct 2026):* the "edge-trace peak hour" row uses the allocation-view load, an upper bound that overstates measured CPU utilisation about tenfold (`docs/EDGE_TRACE_USE.md`); the learner was not re-run under the measured-utilisation model, so that row is a stress test, not a statement about typical edge load.

**Criteria.**  Winning on the benchmark environment: **met** (first positive
learned-policy result in this project).  Winning under *hidden-state* conditions
(faulty RSU, co-tenant load, V2V sabotage): **not met** - parity at best, a loss at
the edge-trace peak hour even with the load observable.  **Trust features add
nothing** in any environment (the masked learner is statistically identical), so
RQ3's "trust improves the offloading decision" remains unsupported.  The benchmark
gain depends on the simulator semantics (Section 6.5 of the paper) and on the
heuristics' information being given to the learner; it is not evidence that RL
beats a well-engineered controller.

## 7. Stealth attackers: a label-free v2 rule and a negative result (24 Oct 2026)

**Why v1 missed norm-matched attackers.**  The v1 weights were fitted on the sign-flip
attack, where the norm ratio alone separates attackers; the logistic therefore leaned on
`ratio`, which a norm-matched attacker leaves at 1.  The direction statistic `cos_loo` itself
separates them almost perfectly in the undefended loop (ROC AUC 0.999 for norm-matched, 0.998
sign-flip, 0.887 partial; `experiments/impact_diag.py`).

**v2 (`CalibratedDetector`, strategy `do_no_harm_v2`).**  No attack is used for fitting:
the 99.5th percentile of the honest norm ratio (2.05) and the 0.5th percentile of the honest
`cos_loo` (-0.245) are measured on **attacker-free** runs; a client's margin is the larger of
its excess over either bound; the margin is smoothed over rounds and the client is excluded
while it is positive.  Same free-rider and 45 % cap guards.

| Closed loop, 24 % attackers | TPR / FPR deployed | TPR / FPR v1 | **TPR / FPR v2** | v2 - FedAvg (pp) |
|---|---|---|---|---|
| norm-matched sign flip (6 seeds) | 0.73 / 5.4 % | 0.51 / 1.2 % | **0.85 / 0.8 %** | +0.3 [-3.9, +4.5] |
| partial sign flip, 50 % (6 seeds) | 0.57 / 5.9 % | 0.50 / 0.5 % | 0.50 / 0.9 % | +0.7 [-0.6, +2.0] |
| scaling x3 (6 seeds) | 1.0 / 3.7 % | 1.0 / 0.0 % | 1.0 / 0.8 % | -1.1 [-3.3, +1.0] |
| sign flip (12 seeds) | 0.995 / 4.6 % | 0.996 / 0.6 % | 0.975 / 1.6 % | +4.1 [-1.2, +9.4] |
| sign flip, 40 % attackers | 0.92 / 0.5 % | 0.985 / 1.5 % | 0.943 / 2.2 % | +14.9 [+7.4, +22.5] |
| sign flip, no attackers | - / 10.3 % | - / 1.7 % | - / 1.7 % | -1.9 [-4.9, +1.2] |

* **Detection of norm-matched attackers rises from 0.51 to 0.85** at under 1 % false exclusion,
  with no attack-specific fitting.  This is a detection result, not a downstream one: the
  norm-matched attack barely lowers FedAvg's completion (74.5 % vs 78.7 % clean), so there is
  nothing large for any defence to recover, and v2's +0.3 pp is unresolved.
* The price is a slightly lower sign-flip detection and a slightly higher false-exclusion rate
  than v1 (the union of two bounds, each at a 0.5 % budget).
* **Partial sign flip is still not detected** (TPR 0.5, the free riders): those poisoners stay
  positively correlated with honest clients, and the honest-calibrated bound is conservative.
  The statistic has signal (AUC 0.887); a lower bound would catch some at a higher false-exclusion
  rate, which was not pursued.

**Negative result: impact-validated aggregation.**  Scoring each uploaded model by its change
in value error on trusted RSU-side data was tested as a norm- and direction-free signal
(`experiments/impact_diag.py`, 2 seeds x 4 attacks).  It does **not** work here: ROC AUC
0.21 (sign flip), 0.32 (norm-matched), 0.45 (partial), 0.28 (scaling) - below 0.5, i.e.
inverted, because honest models fitted to their own (non-IID, epsilon-greedy) data fit the
heuristic-generated root data *worse* than the global model.  The idea is dropped; a trusted
root set that matches the clients' behaviour policy would be needed.

## 8. Learner under the measured-utilisation edge load (24 Oct 2026)

Re-run of the learner study with `load_model: cpu_trace` (realistic co-tenant load), 12 paired seeds:

| Setting | Resource-Aware | Nearest-RSU | learned (est + bandit) | learned - Resource-Aware |
|---|---|---|---|---|
| 05:00 | 70.9 | 69.0 | 71.0 | +0.1 pp [-3.5, +3.6] |
| 20:00 | 70.8 | 67.7 | 70.2 | -0.5 pp [-3.4, +2.4] |

Under realistic load the learner **ties** the best heuristic: the -7.6 pp loss of section 6 was an
artefact of the allocation-view stress load.  There is also little to win (Resource-Aware and
Nearest-RSU are within 3 pp), so the benchmark-environment gain (+6.6 pp) does not carry over to
realistic edge load.

## 9. Norm-matched attack at 40 % attackers: Multi-Krum is harmful (24 Oct 2026, exploratory)

At 24 % attackers the norm-matched attack barely lowers FedAvg's completion, so no defence could show a gain.
At 40 % it does (72.2 % vs 78.7 % clean).  12 paired seeds, run **after** seeing the 24 % result, so exploratory
(the four comparisons below are Holm-adjusted among themselves only):

| Method | Completion % | TPR / FPR | minus FedAvg (pp, 95 % CI) |
|---|---|---|---|
| FedAvg | 72.2 | - | - |
| deployed trust-aware | 71.2 | 0.61 / 3.6 % | -1.0 [-4.6, +2.6] |
| do-no-harm v1 | 70.7 | 0.54 / 2.7 % | -1.5 [-3.5, +0.4] |
| **do-no-harm v2 (label-free)** | **73.8** | **0.81 / 1.8 %** | **+1.6 [-2.1, +5.2]** |
| Multi-Krum | 67.1 | 0.25 / 16.8 % | **-5.1 [-8.1, -2.2]** |

v2 minus Multi-Krum: **+6.7 pp [+2.4, +11.0], p = 0.006** (Holm over four comparisons: 0.024).  Multi-Krum is
*worse than no defence* here: it always discards a fixed share of clients, and against a norm-matched reversal
that share is mostly honest (FPR 16.8 %, TPR 0.25).  v2 is the only rule above FedAvg, but its gain over FedAvg
is **not resolved**.  This is the one setting in this study where the new rule clearly beats the strongest
baseline.
