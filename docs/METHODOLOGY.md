# Methodology

This document describes, in the order a reader needs them: how a client
update is turned into a trust decision, how the exclusion threshold is
selected, how the held-out evaluation is kept independent, which attack
models are used, and which mobility sources the results come from.

Everything described here is implemented in the repository and covered by
tests; the file/function references are the primary anchor.

---

## 1. The trust computation

### 1.1 What a "client update" is

After every federated round each participating client uploads its locally
trained weights.  The server forms the flat delta vector

$$\Delta\theta_i = \mathrm{flatten}(\theta_i^{\text{local}} - \theta^{\text{global}})$$

(`trust/model_trust.py::ModelTrustEvaluator.extract_flat_update`).  A client
that performed no gradient step uploads the global weights unchanged, so its
delta is exactly zero.

### 1.2 The participation rule (what is judged at all)

A client holds a replay buffer of transitions.  The DQN update is a no-op
below `DQNAgent.MIN_TRAIN_BATCH = 4` transitions, so a client that holds
fewer than that **could not have produced an update**: its upload carries no
information.

Such a client is a **non-participant** of the round:

* it is excluded from the aggregate (a zero delta would only dilute it);
* it is **not judged** by the trust engine;
* it is **not counted** in the confusion matrix - in either direction.

The rule uses the client's *available experience*
(`FederatedClient.available_samples`), which is a role-independent
participation statistic.  The ground-truth role is never consulted.  The
same rule is applied by the live console (`web/app.py`), so the benchmark
and the deployed system implement one protocol.

A **free rider** holds plenty of experience and withholds it.  It therefore
stays inside the detection population, where its zero upload is exactly the
signature the norm-ratio rule looks for.

*Implementation:* `experiments/experiment_runner.py` (benchmark),
`web/app.py` (console), `federated/client.py` (the statistic).

### 1.3 The model-update tier (`T_m`)

`trust/model_trust.py` scores every client **relative to its population**:

1. **Agreement with the crowd.**  For each client the *median* cosine
   similarity to all the other clients.  A sign-flipping attacker is
   anti-correlated with the honest majority, so its median agreement is
   strongly negative; an honest client in a heterogeneous population has a
   non-negative median agreement.  The statistic needs no global reference
   direction, so a group of attackers cannot define one.
2. **Robust z-scores.**  Each client statistic is converted into a z-score
   against the population median, using the median absolute deviation (MAD)
   with an inter-quartile-range fallback and a standard-deviation fallback.
   A client is suspicious when it is an *extreme outlier among its peers*,
   not when it merely differs from them.
3. **Scale checks.**  A norm far below the population median is free
   riding; a norm far above it is a scaling attack.
4. **Degenerate-reference repair.**  If the population median norm is zero -
   which happens when more than half of the participants uploaded nothing -
   the reference is recomputed over the clients that did upload something.
   Without the repair every informative client would obtain an unbounded norm
   ratio and be flagged as a scale outlier.

The score is a weighted combination
$T_m = \lambda_{\cos}\,c_{\text{agree}} + \lambda_{\text{dist}}\,c_{\text{dist}} +
\lambda_{\text{norm}}\,c_{\text{norm}}$ and is clipped to $[0.05, 1]$.

### 1.4 The behavioural tier (`T_b`)

`trust/trust_model.py` tracks four observable factors per node: historical
behaviour $B$, response reliability $R$, availability $A$ and communication
reliability $C$.  A node that has never been contacted has **no observed
history**: the prior parameters (`prior_successes`/`prior_total`) are kept
strictly separate from the measured ones (`observed_successes`/
`observed_total`), and the historical factor uses the prior as a Beta
pseudo-count, so the prior's influence decays as evidence accumulates.

### 1.5 The joint trust and the exclusion decision

As implemented in `TrustEngine.compute_joint_trust`:

$$T_{\text{joint},i}=\begin{cases}
T_{b,i} & \alpha \ge 1 \text{ (model update ignored)}\\
T_{m,i} & \alpha \le 0 \text{ (behaviour ignored)}\\
\min\big(\alpha T_{b,i}+(1-\alpha)T_{m,i},\,0.30\big) & 0<\alpha<1 \text{ and } T_{m,i}\le 0.20\ \text{(model breach)}\\
\alpha T_{b,i}+(1-\alpha)T_{m,i} & \text{otherwise}
\end{cases}$$

with `alpha = 0.50`, clipped to [0.05, 1].  A client is **excluded** when its
joint trust is below `tau = 0.40`.  (An earlier version of this document wrote
a `1.5 T_m` term that the code does not contain.)

**What this arithmetic implies** (`tests/test_trust_geometry.py`): a client whose
behavioural trust is at the 0.70 prior is excluded only when `T_m <= 0.20`
(through the override) or `T_m < 0.10`; changing `tau` inside 0.35-0.45 has no
effect; and at `alpha = 0.5` behavioural trust alone cannot exclude a client with
an honest update.  The "joint" score therefore behaves as a hard threshold on
the model-update score in the experiments reported here.

The exclusion decision is taken **before** the ground truth is consulted.
`TrustEngine.filter_trusted_participants` receives the roles only to
*classify* decisions that have already been made
(`trust/trust_engine.py`).  The equivalent function in the validation
harness, `experiments/trust_validation.py::compute_decisions`, has no
parameter through which a role could reach it at all.

---

## 2. Threshold calibration (validation only)

`experiments/calibrate_trust.py::select_exclusion_threshold` sweeps a grid of
candidate thresholds over the **validation** suite (seed base 1) and applies
a pre-registered rule:

1. **Detection plateau** - keep the thresholds whose mean validation detection
   rate is maximal.
2. **False-positive plateau** - inside the detection plateau, keep the
   thresholds that exclude the fewest honest clients.
3. **Centrality** - take the **median** of the surviving interval: the lowest
   threshold of the plateau is the most aggressive exclusion rule, the
   highest the most fragile, and the median is the most central operating
   point.

The function accepts **validation results only**; there is deliberately no
parameter through which held-out results could be supplied.  A test replaces
the held-out suite with a function that raises and confirms that the
selected threshold does not change.

The selected threshold is then evaluated **once** on the held-out suite
(seed base 9001), whose scenario *shapes* - unseen attack strengths,
intermittent participation, small legitimate updates, stronger honest
heterogeneity - were never used for tuning.

The realistic (federated-protocol) version of the same analysis, including
the threshold grid and the participation-rule comparison, is in
`experiments/detector_study.py`; its output is `results/detector_study.md`.

---

## 3. Held-out evaluation

Two independent held-out layers are used.

| Layer | Artefact | Seed base | Used for |
|---|---|---|---|
| Synthetic validation harness | `experiments/trust_validation.py` | 9001 | Scenario-level detection behaviour, threshold assessment |
| Real federated protocol | `experiments/experiment_runner.py` | separate seeds per experiment | Offloading metrics, mobility sweep |

The synthetic harness builds flat update vectors directly, so it can control
the heterogeneity, the attack strength and the participation structure
exactly.  The real protocol produces the updates from actual local training,
so it is the external-validity check on the synthetic conclusions.

---

## 4. Threat model and attack implementations

`attacks/poisoning.py`, `attacks/free_riding.py`, `federated/client.py`.

**Adversary.**  A fraction of the vehicles (`attack_ratio`, default 0.24, i.e.
6 of 25) are malicious *clients*: they train honestly on their own data and
then upload a manipulated model update.  Half of them poison, half free-ride.
They do not collude, do not see other clients' updates, and do not know the
detector (**non-adaptive**), except the norm-matched variant below.  The server
is trusted.  Malicious RSUs are not modelled.  A separate configuration adds
**unreliable helpers** (`unreliable_helper_ratio`): vehicles whose update is
honest but which drop or delay V2V tasks, to test behavioural trust.

| Attack | Implementation | Setting in the benchmark |
|---|---|---|
| `sign_flip` | `delta' = -1.5 s * delta` (strength `s`) | main attack, `s = 1.5` (factor 2.25) |
| `partial_sign_flip` | flip a fixed random fraction of the coordinates | fraction 0.25 (experiment F) |
| `stealth_sign_flip` | `delta' = -delta` (norm equals the honest update) | experiment F; defeats norm tests |
| `scaling` | `delta' = 3 s * delta` | available, not in the benchmark grid |
| `gaussian_noise` | add N(0, (0.4 s)^2) to every weight | available, not in the benchmark grid |
| intermittent | the poisoning client attacks with probability `p` per round | `p = 0.5` (experiment F) |
| `free_riding` | re-upload the global model (zero update) or a stale model | the other half of the adversaries |

The detector was developed against sign flip, scaling, noise and free riding;
the partial, norm-matched and intermittent attacks are the *unseen* cases used
to report where it fails.

---

## 5. Mobility sources

| Mode | Source | Authentic? |
|---|---|---|
| `highway` | procedural Intelligent Driver Model kinematics | no |
| `trace` | calibrated synthetic NGSIM-style trace | no |
| `authentic_trace` | replay of the processed **FHWA NGSIM US-101** trajectory file | **mobility only** |

> **What the authentic trace is and is not.**  The US-101 file is a genuine
> record of vehicle kinematics (position, speed, acceleration, lane).  It
> contains **no** computation task requests, **no** edge queues, **no**
> cellular SINR dynamics, **no** reinforcement-learning rewards and **no**
> attack labels.  Those are modelled.  The authentic trace is therefore
> authentic *mobility*, not an authentic workload, and no result in this
> repository may be described as a measured real-world measurement.

The dataset provenance, licences and SHA-256 checksums are in
[`data/PROVENANCE.md`](../data/PROVENANCE.md) and in each run's
`metadata.json` (`dataset_provenance`).

---

## 6. Queue model

The edge and vehicle queues are **deterministic single-server / multi-core
queues with explicit service times**.  They are **not** M/M/c queues and no
result here is claimed to reproduce a real loss system.  The on-board
processing capacity is calibrated to the automotive SoC class (0.8-2.0 GHz of
effective single-thread throughput), not to a desktop core.

---

## 7. The operational criterion

No *universal* acceptable false-positive or detection rate is claimed.  The
criterion below is defined for this system, justified from the cost
asymmetry of the two errors, and reported as **met** or **not met** for every
scenario.  A scenario that fails it is reported as a limitation, not hidden.

### 7.1 The two error costs

| Error | Consequence | Reversible? |
|---|---|---|
| **Missed attacker** (false negative) | The poisoned update enters the aggregated model.  With 25 participants an attacker carries ~4 % of the weight, and a sign-flipped or scaled update of a DQN policy can flip the greedy action selection. | **No** - the global model is already corrupted and is redistributed to every client. |
| **False exclusion** (false positive) | One honest client does not contribute to one round's aggregate. | **Yes** - the client participates again in the next round, and the trust engine's history carries the decision forward. |

The asymmetry is roughly two orders of magnitude in the number of rounds
needed to recover.  The criterion is therefore deliberately asymmetric: a
**much** tighter bound on the false-positive rate than on the false-negative
rate.

### 7.2 The criterion

For a scenario with at least 100 honest decisions:

| Metric | Criterion | Rationale |
|---|---|---|
| False-positive rate | **<= 0.01** (one honest client in a hundred) | The error is recoverable, but ten false exclusions per thousand honest decisions would systematically erode participation; below 1 %, the aggregate loses less than the attacker's own weight. |
| Detection rate, full-strength attack | **>= 0.95** | A detection rate below 0.95 lets a poisoned update into the model in a meaningful fraction of rounds, which is not tolerable given that the error is not reversible. |
| Detection rate, partial attack | **reported, no threshold** | The detector's behaviour on a partially aligned attack is a property of the attack, not a pass/fail property of the system; it is reported per strength. |
| Precision | **reported** | Precision is not part of the criterion because the attack ratio is a scenario parameter, not a system property. |

### 7.3 Where the criterion is met or not

The criterion is evaluated mechanically for the active run in
`results/claims_report.md` (sections 3, 4 and 4b).  On real federated updates
the deployed detector's false-positive rate (about 4 %) **does not meet** the
<= 0.01 criterion, and neither do the held-out `honest_benign_outlier`
scenario or the partial/norm-matched/intermittent attacks (detection rate below
0.95 for what are partial or adaptive attacks).  These are documented
limitations (`docs/LIMITATIONS.md`, T1, T5, T6), not exceptions to be explained
away.

---

## 8. Experimental protocol

**Separation.**  Training and evaluation use separate environments, separate
seeds (evaluation seeds = training seed + 500,000 + episode index) and separate
task streams.  Evaluation is *inert*: greedy policy, no replay writes, no
gradient steps; the parameters of every client are hashed before and after and
must be identical.  Threshold selection used the synthetic validation suite
(seed base 1) only; the held-out suite (seed base 9001 and 9001-9020 for the
detector comparison) was never used for selection.

**Initialisation and data.**  Federated methods start every client from the
server's single initial global model; the non-federated baseline gives every
client an independent initialisation.  Each client trains only on its own
replay buffer (its own vehicle's transitions).  A client with fewer than four
transitions cannot take a gradient step and is a *non-participant* (section
1.2).  Training: 30 rounds x 20 simulation steps, 16 local gradient steps per
client per round (480 in total), epsilon-greedy for the first 40 % of rounds.
Evaluation: 5 episodes x 30 steps per seed with the client's own (post-sync)
network.  Matched evaluation: every method is evaluated on the same
environments, workloads and episode seeds, so differences are paired by seed.

**FL lifecycle accounting.**  Each round records: attempted (an aggregation was
run), uploaded, accepted, rejected, `status` in `{model_updated, model_unchanged,
no_model_produced, not_attempted_insufficient_updates}`, the number of gradient
steps taken, the mean training loss of clients that actually optimised, and the
measured payload in bytes.  A round is never reported as a success unless the
global model changed.

**Fairness of budgets.**  All learned methods use the same architecture,
optimiser, replay, training rounds and gradient steps; the aggregation rule is
the only difference.  Communication volume is identical across federated
methods by construction (reported anyway).  The heuristics (Local-Only,
Random, Nearest-RSU, a greedy minimum-estimated-delay "Resource-Aware") see only
information a vehicle could observe and are coverage-aware.

**Reference rows that frame the downstream question.**
`FedAvg_Clean_Reference` (same method, no attackers) measures how much the
attack costs; `Oracle_Exclusion_Upper_Bound` (removes the truly malicious
clients using the ground-truth role - **analysis only**, not a deployable
method) bounds what any detector can gain.

**Statistics.**  Per-method mean, sample SD and Student-t 95 % CI over seeds;
paired differences with a paired t-test; Cohen's d_z.  The eight headline
claims (`claims_report.py`) are tested once each and Holm-adjusted.  A verdict
is SUPPORTED only if the CI excludes zero in the claimed direction **and** the
Holm-adjusted p-value is below 0.05; the minimum detectable effect at 80 %
power is printed next to every claim.  Detection rates are pooled from confusion
counts with Wilson intervals (optimistic, see LIMITATIONS T9).

**Shadow detectors.**  The server also passes the same round of updates through
alternative exclusion rules (`trust/alt_detectors.py`: Krum selection with an
assumed Byzantine fraction of 0.25, median-cosine, MAD-distance); their
decisions are scored but never used, so every detector sees identical updates.

**Cost control.**  The suite checkpoints every cell, can be resumed
(`--resume`), reuses cells that are identical to a main-comparison cell, and
runs seeds in parallel from one persistent worker pool.


---

## 9. Supplementary experiments (edge-node fault, intermittent connectivity)

`python -m experiments.supplementary_suite` writes its **own** versioned run
(`run_kind: supplementary`, a manifest entry with `supplements: <active run>`), so
the active run is never modified.  It uses the parent's sweep seeds and the same
training and evaluation protocol, so its cells can be paired with the parent's by
seed.

| Experiment | Question | Cells |
|---|---|---|
| L faulty edge node | Does *behavioural* trust help the offloading decision when one RSU is unreliable? | learned policy with trust features; same policy with trust features masked; masked policy without a fault; Local-Only, Nearest-RSU, Resource-Aware and the trust-aware greedy heuristic under the fault |
| M intermittent connectivity | Do stale updates hurt, and does a staleness discount help? | FedAvg; trust-aware without staleness handling; trust-aware with the discount (each client offline in 40 % of rounds, missing the broadcast) |
| K2 mixed threat | How does the detector treat honest task-dropping vehicles when labels are correct? | joint trust; model-trust only |

The claims of the supplementary run (S1-S6) are tested with the same rule as the
headline claims (paired CI, Holm correction across the six).  They are exploratory
(6 seeds).
