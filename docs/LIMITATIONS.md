# Limitations and threats to validity

This document states what the evidence does **not** establish.  Numbers refer
to the active run (`results/manifest.json`); the full tables and the verdict of
every claim are in `results/claims_report.md`.

---

## 1. External validity (simulation)

| # | Limitation | Consequence |
|---|---|---|
| S1 | **Everything except the optional mobility replay is simulated**: tasks, channel, queues, rewards, attacks.  Task sizes, cycles and deadlines come from a catalogue derived from 3GPP/ETSI service classes and scaled by 0.1; it is a modelling assumption. | Completion, latency and energy are properties of this simulator.  They are not measurements of a vehicle, an RSU or a network. |
| S2 | **Only mobility can be authentic.**  NGSIM US-101 records kinematics only. | `authentic_trace` authenticates the vehicle motion, not the workload.  No result is a real-world measurement. |
| S3 | **The channel is nearly distance-independent inside coverage** (single-user rate is at the 54 Mbps cap for most samples), has no interference term and no transmitter circuit power. | Coverage cut-off, handover and cell contention drive the communication cost; transmit energy is a lower bound.  Conclusions that depend on a smoothly varying link rate are not supported. |
| S4 | **Queues are deterministic multi-core servers, not M/M/c.**  All tasks of a slot arrive together (in a seeded random order). | Queueing delays are not a model of a real loss system; there is no arrival jitter inside a slot. |
| S5 | **One on-board CPU class (0.8-2.0 GHz), one RSU layout, one road.**  The offloading advantage depends on the deadline catalogue and the CPU class. | A different class moves the local/offload trade-off. |
| S6 | **Evaluation horizons are short**: 5 episodes x 30 steps (15 s of simulated time each) after 30 rounds x 20 steps of training. | Nothing is claimed about long-run behaviour or non-stationary traffic. |
| S7 | **The decision problem is close to a contextual bandit** (the next task does not depend on the action except through queue load) and the discount factor was not tuned or swept. | The RL formulation may be heavier than the problem needs; a bandit baseline was not run. |
| S8 | **Sweeps use 6 seeds, the main comparison 16.**  Paired confidence intervals in the sweeps are wide. | Sweep results are exploratory; the verdicts in `claims_report.md` are computed with the stated rule and, for the eight headline claims, a Holm correction. |

## 2. What the downstream evidence supports

| # | Finding | Status |
|---|---|---|
| D1 | The sign-flip attack lowers the completion rate of undefended FedAvg by about 6 percentage points (paired, 16 seeds, raw p < 0.01). | **Raw p significant; fails Holm correction across the 8 pre-specified claims** (adjusted p ~ 0.06).  Reported with both numbers. |
| D2 | Trust-aware aggregation recovers about 2 pp relative to FedAvg at the default 24 % attack ratio. | **Not established** (CI contains 0; the design resolves about 5-6 pp). |
| D3 | An oracle that removes exactly the malicious clients recovers about 5 pp. | Raw p < 0.05, fails Holm.  Bounds what *any* detector can gain at this attack strength: the headroom is small. |
| D4 | At 40 % attackers the defence recovers about 18 pp (6 seeds, CI excludes 0); with **no** attackers it loses about 10 pp (6 seeds, CI contains 0). | The benefit at high contamination is supported; the cost of false exclusions when there is nothing to defend against is large in magnitude but not statistically established. |
| D5 | The learned trust-aware policy is statistically indistinguishable from Nearest-RSU and from a greedy resource-aware heuristic. | **Not established** either way.  Under the first state representation (v2.0) it was 12 pp *worse* than Nearest-RSU; the improvement came from a post-hoc state-feature change (section 4). |
| D6 | Federated learning beats independent per-vehicle learning by about 13 pp (no attackers). | Supported. |

## 3. The trust detector

| # | Limitation | Evidence |
|---|---|---|
| T1 | **False exclusions on real federated updates are about 4 % overall and rise as training converges** (about 1 per round early, about 8 % of honest clients late; about 10 % with no attackers).  The synthetic harness reports 0.8 %. | `results/claims_report.md` §3, §4b; benchmark per-round confusion counts. |
| T2 | **The joint trust threshold is nearly degenerate.**  With behavioural trust at its 0.70 prior (every vehicle that is not used as a V2V helper), exclusion only happens through the model-trust override (`T_m <= 0.20`); `tau` between 0.35 and 0.45 changes nothing, and behavioural trust cannot exclude a client with an honest update at `alpha = 0.5`. | `tests/test_trust_geometry.py` |
| T3 | **Behavioural trust adds nothing measurable.**  Attackers are behaviourally honest, so `T_b` carries no signal against poisoning; in the mixed-threat experiment the behaviour-only configuration reaches TPR 0.15 at FPR 0.11, and the joint configuration is indistinguishable from model-only (TPR 0.62 vs 0.59; completion difference -2.5 pp, CI [-7.8, +2.8]). | experiments D, E, K |
| T4 | **A simple robust-distance rule matches the deployed score.**  On held-out synthetic scenarios, `distance_mad` has the same pooled TPR/FPR as the deployed detector (0.864 / 0.0083 vs 0.865 / 0.0082); on real FL updates it has *lower* FPR (0.016 vs 0.042) at slightly lower TPR (0.972 vs 0.991). | `results/detector_comparison.md`, claims report §3 |
| T5 | **Partial, norm-matched and intermittent attacks are missed.**  TPR 0.53 (25 % of coordinates flipped), 0.73 (norm-matched sign flip), 0.77 (attacks in half of the rounds). | claims report §4 |
| T6 | **Scale normalisation is fragile on low-dispersion populations; a small honest update is indistinguishable from a free rider in one round.**  When two cases give the detector the same evidence, no rule can separate them (identifiability limit). | held-out `honest_benign_outlier`; `tests/test_participation_protocol.py` |
| T7 | **Adaptive attackers are out of scope.**  Only the norm-matched sign flip is a (weak) adaptive attack; an attacker that knows the rule can evade it. | threat model |
| T8 | **The ground truth counts every non-honest role as positive.**  In the mixed-threat experiment task-dropping helpers (honest models) are counted as malicious, which caps the model-update TPR at about 0.6 by construction. | experiment K note |
| T9 | **Confidence intervals on pooled detection counts are optimistic** (the same client is judged every round; decisions are not independent). | stated in the report |

## 4. Process threats (what could mislead a reader)

* **The state representation was changed after seeing results.**  The first full
  run (v2.0 features) showed that 27 % of the learned policy's failures were
  `out_of_coverage`; the state was changed to coverage-relative distances plus
  in-coverage flags (v2.1) and the benchmark re-run.  Both runs are retained
  (`results/run_20261022_215233_env_v2_protocol`, superseded) and compared in
  `claims_report.md` §14.  This is a design iteration on the *learner's input*,
  not on the detector or the attack, but it is a researcher degree of freedom.
* **Environment model v2 was introduced after an audit** (RSU choice honoured,
  contention, neutral reward).  The earlier model (v1) is reproducible and was
  re-run in experiment J: under v1 the learned policies beat Nearest-RSU by about
  +3 pp (paired, 6 seeds, CI [+2.2, +3.7] for trust-aware and [+1.1, +3.9] for FedAvg) while
  trust-aware and FedAvg are indistinguishable (+0.5 pp, CI [-0.8, +1.9]); under v2.1 the learned
  policy is statistically indistinguishable from Nearest-RSU.  The conclusion "RL beats a
  heuristic" depends on which simulator semantics are used.
* **Training budget.**  480 local gradient steps per client.  A 32-epoch pilot
  on the v2.0 features gave 78 % for clean FedAvg vs 68 % at 16 epochs, so the
  learned policies were still budget-limited under v2.0 and may be under v2.1.
* **The run was interrupted once** (host memory reaping) and resumed; two cells
  of the first full run were reloaded from checkpoint.  The final (v2.1) run
  was not interrupted.
* **Seeds.**  A single seed draws the road, task mix, roles and network
  initialisation together; the between-seed spread (about +/-4-6 pp at 95 %)
  dominates the effects being measured.
* **No multiplicity correction in the sweeps.**  Sweep verdicts are exploratory.

## 5. Energy

Local execution consumes about 30 times less energy per task than the learned
offloading policies (transmission dominates).  Any framing of the contribution
as *energy efficiency* is unsupported; the policies trade energy for
completion.

## 6. Replication limits

* The full v2.1 benchmark took about 2.5 hours on 6 worker processes (about
  0.8 GB each).  A smoke configuration (`--quick`) exercises every code path in
  a few minutes and never updates the active pointer.
* The authentic NGSIM export is 1.5 GB and its download endpoint is
  intermittently unreliable; the retrieval script installs only a hash-verified
  file.  The licence recorded in `data/PROVENANCE.md` is CC BY-SA 3.0
  (USDOT Socrata catalogue, recorded 2026-10-21); re-check it before redistributing derived files.
* Determinism: every trial re-seeds Python, NumPy and PyTorch at its start, so
  per-seed records do not depend on the number of workers.  Bit-identical
  results across different CPUs or PyTorch builds are not guaranteed.

## 7. What must never be claimed

* That trust-aware aggregation improves task completion at the default attack
  ratio (not established), or that it improves it "by X %" without the interval.
* That the detector has a near-zero false-positive rate in general.
* That behavioural trust improves detection or protects the policy.
* That the learned policy beats simple heuristics (not established under v2.1,
  contradicted under v2.0, true under the earlier v1 simulator).
* That trust-aware FL is more energy-efficient than local execution.
* That any result is a measured real-world result.
* That a superseded, smoke or aborted run is current evidence.

## Supplementary results and what they do not show

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