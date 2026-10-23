# Evaluating trust-aware federated RL for VANET offloading: a downstream-impact protocol

*Sreemanta Barman.  Subtitle: does excluding poisoned clients protect a federated task-offloading policy?  A paired evaluation in a simulated vehicular edge network.*

*Draft manuscript generated from the active benchmark run.  Every number is
taken from `results/claims_report.md` / `results/benchmark_results.json` (run
`run_20261023_022608_env_v21_coverage_features`); nothing is rounded in a
favourable direction.  The work is simulation-only.*

---

## Abstract

Trust-aware offloading, federated reinforcement learning (FRL) for vehicular
edge computing, and Byzantine-robust aggregation have each been studied, but we
found no evaluation that measures whether *excluding poisoned clients actually
protects the task performance of a federated offloading policy*.  We build a
reproducible simulator (IDM or replayed NGSIM US-101 mobility, a shared-uplink
channel model, deterministic multi-core edge queues) and train a dueling double
DQN offloading policy with federated averaging across 25 vehicles, 24 % of which
upload poisoned or free-riding updates.  A population-relative trust detector
that combines model-update and behavioural trust is compared with FedAvg,
coordinate median, trimmed mean and Multi-Krum, and with two reference rows -
a no-attack control and an oracle that removes exactly the malicious clients -
on matched evaluation episodes over 16 seeds.  The detector finds sign-flip
poisoning (TPR 0.991) but excludes 4.2 % of honest clients (FPR 0.042), a rate
that grows as training converges.  Downstream, the attack lowers the completion
rate of undefended FedAvg by 5.9 percentage points (95 % CI [-10.1, -1.7]; raw
p = 0.009, Holm-adjusted p = 0.061); trust-aware aggregation recovers 2.0 pp
(CI [-2.0, +6.0]), which the design cannot distinguish from zero, while an oracle
that removes the attackers would recover 5.2 pp.  At 40 % attackers the defence
helps clearly (+17.9 pp, CI [+6.1, +29.7]); with no attackers it costs 9.9 pp
(CI [-22.6, +2.8]).  Behavioural trust adds no measurable value, a simple
MAD-distance rule matches the deployed detector on held-out data, and the
learned policy is statistically indistinguishable from a nearest-RSU heuristic.
Two modelling changes - how the simulator treats the chosen RSU and how
distances enter the state - move the policy-versus-heuristic comparison by
-12 to +3 pp, which we report as a sensitivity of the conclusion.  We release
all code, seeds, raw per-seed records and a claim-by-claim verdict table.

**Keywords:** vehicular edge computing, task offloading, federated reinforcement
learning, model poisoning, trust management, reproducibility.

---

## 1. Introduction

Vehicles that offload perception and planning tasks to roadside edge servers face
high mobility, intermittent connectivity and heterogeneous resources.
Reinforcement learning is a natural way to learn the offloading decision;
federated learning lets vehicles train a shared policy without sharing raw
experience.  Federation exposes the shared policy to *model poisoning* by
compromised or selfish vehicles, and trust mechanisms have been proposed to
exclude them.

The roadmap for this project asked whether a trust-aware FRL framework improves
secure and efficient offloading.  A literature check (Section 2) shows that the
combination of the three ingredients is not by itself novel.  We therefore ask a
narrower question that a simulator can answer:

> **RQ.** In a mobility-driven offloading simulation with intermittent, non-IID
> clients, does excluding poisoned clients measurably protect task completion,
> and what does it cost in false exclusions?

**Contributions.**  (1) A downstream-impact protocol: each defence is compared
with a no-attack reference, an undefended policy under attack and an oracle
exclusion bound on matched episodes, with mechanically computed verdicts
(confidence interval, Holm correction, minimum detectable effect).  (2) An
audited, switchable simulator whose corrections are themselves evaluated: they
change the conclusion.  (3) A characterisation of a population-relative trust
detector on real federated updates, including its false-exclusion cost and the
degeneracy of its joint-trust threshold.  (4) A complete, reproducible
artefact.  We do **not** claim a new aggregation algorithm: the detector's
statistics are recombinations of known ones and a MAD-distance rule matches it.

## 2. Related work

*Trust-aware vehicular offloading.*  Zhang et al. (GLOBECOM 2023) combine a
recommendation-based trust model with DDPG for offloading across V2I/V2V modes
but do not federate.  Consul et al. (ACM SIGCOMM'24 ZTA workshop) score edge
nodes and manage offloading with asynchronous federated RL, targeting untrusted
RSUs rather than poisoned clients.  *FRL for offloading.*  Chen et al. (IEEE IoT
J. 2026, FedVTO) combine FL and DRL with entropy-adaptive weighting for privacy;
poisoning is not part of the threat model.  *Robust FRL.*  Fan et al. (NeurIPS
2021) and Ganesh et al. (arXiv 2024) give guarantees for federated policy
gradient with faulty or adversarial agents on generic RL benchmarks.  *Robust
FL.*  Krum (Blanchard et al. 2017), coordinate median and trimmed mean (Yin et
al. ICML 2018), FoolsGold (Fung et al. 2018) and FLTrust (Cao et al. NDSS 2021,
which needs a clean root dataset) are the standard references; Fang et al.
(USENIX Security 2020) show that adaptive local poisoning can defeat several of
them.  The matrix in `docs/RELATED_WORK.md` records what each paper's abstract
states; we found no study of the downstream effect of client exclusion on a
vehicular offloading policy.

## 3. System model

**Mobility** is either an Intelligent Driver Model on a 3.2 km loop (simulated)
or a replay of the FHWA NGSIM US-101 trajectories (681 m segment, 8 lane ids,
2,847 vehicles; the only measured data), with source-vehicle identities
preserved and absent vehicles generating no tasks.  **Edge**: three RSUs
(coverage 480-650 m, 8-16 cores, deterministic multi-core service).  **Channel**:
log-distance path loss with 3 dB shadowing and Rayleigh fading at 5.9 GHz, a
20 MHz UE channel, a 100 MHz cell shared OFDMA-style among simultaneous uplinks
(rate capped at 2.7 bit/s/Hz), no interference, 25 ms handover.  **Tasks**: four
service classes (simulated catalogue), one task per vehicle per 0.5 s, non-IID
task mixture per vehicle (Dirichlet 0.4); a task succeeds only if its sojourn
time meets its deadline, identically for local, RSU and V2V execution.
**Decision**: local, RSU 1-3 or V2V; the action names an RSU and is honoured (a
vehicle outside that RSU's coverage fails).  **State** (20 features): task,
speed, coverage-relative RSU distances and in-coverage flags, queue and CPU,
RSU load, behavioural trust of RSUs and peer.  **Reward**: success bonus minus
normalised delay and energy, failure penalty, no trust bonus (a bonus of 0.7 on
every offload but not on local execution would make offloading optimal by
construction; it is kept as a sensitivity).  Full specification and every
parameter: `docs/ENVIRONMENT_SPEC.md`.

## 4. Method

**Federated RL.**  Each vehicle runs a dueling double DQN on its own replay
buffer; every 20 steps all clients with at least four transitions upload their
weights after 16 local gradient steps; the server aggregates and broadcasts.
All federated methods start from one global model; the non-federated baseline
uses independent initialisations.

**Threat model.**  Of 25 vehicles, 6 are malicious clients (3 sign-flip with
factor 2.25, 3 free riders).  They do not collude, do not know the detector and
the server is trusted (non-adaptive).  Variants: 25 % partial sign flip,
norm-matched sign flip, an intermittent attacker (p = 0.5).

**Trust detector.**  For client i with update delta_i, the model-update score
T_m combines (a) the robust z of the *median pairwise cosine similarity* with
the other clients, (b) the robust z of the distance to the coordinate-wise
median, and (c) population-relative norm checks; behavioural trust T_b is an
exponential moving average of observed task outcomes, delay margin, SINR and
node load.  The joint trust is alpha T_b + (1 - alpha) T_m (alpha = 0.5) with an
override that caps it at 0.30 when T_m <= 0.20; clients below tau = 0.40 are
excluded and the rest are averaged with weights T_i n_i.  A client with fewer
than four transitions cannot train and is a non-participant, not a suspect.
**Observation (tested):** at the neutral behavioural prior, exclusion occurs
only through the T_m <= 0.20 override; tau in [0.35, 0.45] is inert and T_b alone
cannot exclude a model-honest client at alpha = 0.5.

## 5. Experimental protocol

Training: 30 rounds x 20 steps, 16 local gradient steps per round.  Evaluation:
5 episodes x 30 steps per seed with a greedy, inert policy on separate seeds;
the parameter hash is verified unchanged.  Every method sees the same
environments and task streams, so differences are paired by seed.  Main
comparison: 16 seeds; sweeps: 6 seeds.  Reference rows: *clean FedAvg* (no
attackers) and *oracle exclusion* (analysis only).  Baselines: local-only,
random, nearest-RSU, a greedy minimum-estimated-delay heuristic (both
coverage-aware).  Eight claims were fixed before the intervals were computed and
are tested once each with a paired t-test; a claim is *supported* only if the
95 % CI excludes zero in the claimed direction and the Holm-adjusted p < 0.05.
Detectors are scored in *shadow mode* on identical updates.  The threshold was
selected on validation data only; held-out scenarios (seed bases 9001-9020) were
never used for selection.

## 6. Results

### 6.1 Policies (Table 1; Fig. 1, 7)

| Method | Completion, % (mean ± 95 % CI) | Energy, J/task |
|---|---|---|
| Clean FedAvg (no attackers) | 78.5 ± 1.2 | 0.0160 |
| Oracle exclusion | 77.8 ± 1.1 | 0.0163 |
| Multi-Krum | 77.1 ± 1.7 | 0.0141 |
| Nearest-RSU (heuristic) | 75.0 ± 0.3 | 0.0182 |
| **Trust-aware (proposed)** | **74.6 ± 2.8** | 0.0127 |
| Trimmed mean | 72.7 ± 2.5 | 0.0123 |
| FedAvg under attack | 72.5 ± 4.0 | 0.0128 |
| Greedy resource-aware | 72.3 ± 0.9 | 0.0021 |
| Non-federated D3QN | 65.7 ± 1.7 | 0.0086 |
| Coordinate median | 61.4 ± 4.3 | 0.0048 |
| Local only | 59.1 ± 1.2 | 0.0004 |
| Random | 32.5 ± 0.4 | 0.0063 |

*Table 1.*  16 seeds; latency of completed tasks is 97-103 ms for all methods and
is not discriminative.  Local execution uses 31 x less energy than the proposed
policy.

### 6.2 Downstream claims (Fig. 7)

| # | Claim | Paired difference, pp [95 % CI] | raw p | Holm p | Verdict |
|---|---|---|---|---|---|
| H1 | Attack degrades FedAvg | -5.92 [-10.11, -1.73] | 0.009 | 0.061 | not established after Holm |
| H2 | Trust-aware recovers vs FedAvg | +2.04 [-1.95, +6.04] | 0.292 | 0.585 | not established |
| H3 | Oracle recovers vs FedAvg | +5.22 [+1.12, +9.32] | 0.016 | 0.096 | not established after Holm |
| H4 | Residual loss vs clean remains | -3.88 [-7.33, -0.43] | 0.030 | 0.151 | not established after Holm |
| H5 | Trust-aware beats Multi-Krum | -2.50 [-6.08, +1.08] | 0.158 | 0.476 | not established |
| H6 | Federation beats non-federated | +12.73 [+10.35, +15.11] | <0.001 | <0.001 | **supported** |
| H7 | Learned beats Nearest-RSU | -0.41 [-3.20, +2.37] | 0.757 | 0.757 | not established |
| H8 | Learned beats greedy heuristic | +2.33 [-0.67, +5.34] | 0.119 | 0.476 | not established |

The design resolves differences of about 5-6 pp (80 % power), so "not
established" means that the data neither show nor rule out the effect.

### 6.3 Attack ratio (Fig. 8)

| Attack ratio | Trust-aware minus FedAvg, pp [95 % CI] | n | Deployed FPR |
|---|---|---|---|
| 0.00 | -9.88 [-22.55, +2.79] | 6 | 0.103 |
| 0.24 | +2.04 [-1.95, +6.04] | 16 | 0.042 |
| 0.40 | **+17.86 [+6.07, +29.66]** | 6 | 0.001 |

The defence matters when contamination is high and costs completion when there
is nothing to defend against; the zero-attack cost is large but its interval
contains 0 at n = 6.

### 6.4 The detector (Fig. 9, 10)

On the real federated updates of the main comparison the deployed detector has
TPR 0.991 and FPR 0.042 (2,854 TP, 26 FN, 387 FP, 8,733 TN; Wilson intervals are
optimistic).  Shadow rules on identical updates: MAD-distance TPR 0.972 / FPR
0.016, Krum 0.500 / 0.158, median-cosine 0.329 / 0.175.  On 20 held-out
synthetic seed bases the MAD-distance rule matches the deployed detector
(0.864 / 0.0083 vs 0.865 / 0.0082) and Krum has FPR 0.174.  False exclusions
rise from about 1 % of honest clients in early rounds to about 8 % late
(Fig. 9 right), consistent with small benign updates being indistinguishable
from free riding.  Detection by attack: partial sign flip (25 %) TPR 0.53,
norm-matched sign flip 0.73, intermittent attacker 0.77 (FPR 0.03-0.05).
Ablation: without model trust the detector finds 8 % of attackers (FPR 0.10);
without behavioural trust 92 % (FPR 0.039); alpha = 0.8 gives 0.996 / 0.032.
In the mixed-threat experiment (poisoners plus task-dropping helpers) the
model-only configuration reaches 76.6 % completion, the joint 74.1 % and
behaviour-only 72.6 % (FedAvg 71.5 %); behaviour-only detection has TPR 0.15.

### 6.5 Sensitivity of the conclusion to simulator choices

| Simulator variant | Learned trust-aware policy minus Nearest-RSU |
|---|---|
| v1: silent RSU re-association, uncontended uplink, trust bonus | +2.97 pp [+2.20, +3.74] (n = 6) |
| v2.0: honoured RSU choice, contention, neutral reward, road-length-normalised distances | about -12 pp (27 % of failures were out-of-coverage) |
| **v2.1 (final): coverage-relative distances + in-coverage flags** | **-0.41 pp [-3.20, +2.37] (n = 16)** |

Moving from v2.0 to v2.1 features improved the proposed policy by +11.7 pp
(CI [+7.0, +16.4]).  The v2.1 change was made **after** seeing the v2.0 results;
both runs are retained.  With the legacy trust bonus (lambda = 2.5) the trust-aware policy is +2.2 pp
[-0.5, +4.9] above Nearest-RSU and +2.7 pp [-1.5, +6.9] above FedAvg (6 seeds), i.e.
still not distinguishable from zero.

### 6.6 Robustness (6 seeds each; exploratory)

Partial participation (50 %), connectivity-gated uploads and extreme task
heterogeneity (Dirichlet 0.1) leave trust-aware and FedAvg within each other's
intervals; the mobility source changes the absolute level (authentic NGSIM
replay 65.8 ± 9.5 % vs 74.6 % on the procedural highway) but not the detector's
behaviour (TPR 0.981, FPR 0.055).

### 6.7 System cost

Federated traffic is 209.5 MiB per run (25 clients, 30 rounds, 146 kB per model
transfer).  Server-side aggregation takes 38 ms for FedAvg and 532 ms for the
trust-aware pipeline, which in the benchmark also includes the shadow detectors.

## 7. Discussion

*What is established.*  Federation helps (+12.7 pp over independent learning).
The attack hurts the undefended policy by about 6 pp, though that estimate does
not survive correction across eight claims.  The detector finds sign-flip
attackers almost perfectly and the oracle shows that perfect detection could
recover only about 5 pp at this attack strength.  *What is not.*  Trust-aware
aggregation does not demonstrably protect the policy at the default ratio; its
benefit appears at 40 % attackers.  Its cost is false exclusion, which a simpler
rule reduces.  Behavioural trust, the paper's second tier, contributes nothing
measurable here because attackers behave honestly in V2V tasks.  *Negative and
fragile results.*  The learned policy does not beat a nearest-RSU heuristic
under the final model; its advantage in the earlier simulator was an artefact of
how the simulator re-associated vehicles.  Partial, norm-matched and
intermittent attacks are largely missed.

## 8. Threats to validity

Simulation only (no real tasks, channel or attacks); a nearly distance-flat
single-user link and no interference; deterministic queues with synchronous
arrivals; one road and one CPU class; short evaluation horizons; scripted,
non-adaptive attackers; between-seed spread (about 4-6 pp) comparable to the
effects; sweeps with 6 seeds and no multiplicity correction; a post-hoc change
of the state representation; ground-truth labels that count task-dropping
helpers as malicious in the mixed-threat experiment; Wilson intervals on
non-independent decisions.  Details: `docs/LIMITATIONS.md`.

## 9. Reproducibility

Code, configuration, seeds and per-seed records are in the repository.  The
active run, its metadata (code hash, data hashes, environment versions,
training protocol), the figure manifest and the verdict table are under
`results/`.  Commands: `docs/REPRODUCIBILITY.md`.  The full benchmark takes
about 2.5 hours on 6 worker processes; `--quick` runs a smoke configuration;
`--resume` continues an interrupted run.

## 10. Conclusion

In this simulator, excluding poisoned clients has a clear benefit only under
heavy contamination, a modest and statistically unestablished benefit at 24 %
attackers, and a measurable cost when there are none.  The evidence supports a
careful evaluation methodology and a candid characterisation of a trust-based
defence, not a claim that trust-aware federated RL improves vehicular
offloading.

---

## Candidate venues (to be checked against current calls before submission)

Selection reasoning, not a guarantee of acceptance.  Evidence collected on
2026-10-22 from public pages; deadlines and page limits were **not** verified
and must be read from each venue's current call.

| Venue | Why it may fit | What to check |
|---|---|---|
| IEEE Vehicular Networking Conference (VNC) | Vehicular networking; ComSoc listing shows the 2026 edition (Montréal, 8-10 June) and a poster/demo track | The 2027 technical-paper deadline and page limit; the scope favours networking results, and this work is evaluation-centred |
| IEEE Transactions on Vehicular Technology | Has published federated-learning papers for vehicular edge scenarios (a 2025 paper on vehicular edge FL in mobile scenarios was found) | Whether the contribution level (evaluation methodology, mostly negative results) meets a transactions standard; author instructions were not retrieved |
| IEEE Transactions on Intelligent Transportation Systems | Has published federated multi-agent DRL task-offloading papers (2025) | Scope statement was not retrieved; expect a requirement for stronger methodological novelty |
| Workshop on reproducibility / negative results in networking or ML systems | The main contribution is a rigorous protocol and honest null/negative findings | Venue scope and format |

The results are best matched to a venue that values evaluation methodology and
reproducibility; a method-novelty venue would need the stronger-attack and
lower-false-exclusion follow-ups listed in `docs/SCIENTIFIC_CONTRIBUTION.md`.

---

## References

Verified against publisher, arXiv or institutional records in this project
(abstract level):

1. L. Zhang, H. Guo, X. Zhou, J. Liu, "Trusted Task Offloading in Vehicular Edge Computing Networks: A Reinforcement Learning Based Solution," IEEE GLOBECOM 2023, pp. 6711-6716.
2. P. Consul et al., "A Reliable Zero-Trust Network for Task Offloading in Vehicular Systems Using an Asynchronous Federated Learning Approach in 6G," ACM SIGCOMM'24 Workshop on Zero Trust Architecture for Next Generation Communications, 2024.
3. Y. Chen et al., "An Entropy-Based Privacy-Preserving Federated Deep Reinforcement Learning Framework for Task Offloading in Vehicular Edge Computing Networks," IEEE Internet of Things Journal 13(5):8011-8024, 2026, doi:10.1109/JIOT.2025.3602798.
4. F. X. Fan et al., "Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee," NeurIPS 2021, arXiv:2110.14074.
5. S. Ganesh et al., "Global Convergence Guarantees for Federated Policy Gradient Methods with Adversaries," arXiv:2403.09940.
6. X. Cao, M. Fang, J. Liu, N. Z. Gong, "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping," NDSS 2021, arXiv:2012.13995.
7. M. Fang, X. Cao, J. Jia, N. Z. Gong, "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning," USENIX Security 2020.
8. D. Yin, Y. Chen, R. Kannan, P. Bartlett, "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates," ICML 2018 (PMLR 80).
9. C. Fung, C. J. M. Yoon, I. Beschastnikh, "Mitigating Sybils in Federated Learning Poisoning," arXiv:1808.04866.
10. FHWA, "NGSIM US-101 data" (fact sheet www.fhwa.dot.gov/publications/research/operations/07030; data catalogued at data.transportation.gov).

Standard references cited from background knowledge and **not re-verified online
in this project**: P. Blanchard et al., "Machine Learning with Adversaries:
Byzantine Tolerant Gradient Descent," NeurIPS 2017; H. B. McMahan et al.,
"Communication-Efficient Learning of Deep Networks from Decentralized Data,"
AISTATS 2017; M. Treiber, A. Hennecke, D. Helbing, "Congested traffic states in
empirical observations and microscopic simulations," Phys. Rev. E 62, 2000;
V. Mnih et al., Nature 518, 2015; H. van Hasselt et al., AAAI 2016; Z. Wang et
al., ICML 2016; 3GPP TR 36.885 (V2X evaluation methodology); S. Holm, Scand. J.
Statistics 6, 1979; E. B. Wilson, JASA 22, 1927.

## Supplementary experiments: edge-node faults and intermittent connectivity

Experiments L (fixed faulty RSU), N (rotating faulty RSU), M (40 % offline clients, missed broadcasts, staleness discount) and K2 (mixed threat with corrected labels) are described in METHODOLOGY.md section 9.

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

Target-venue note: with one supported headline claim and null or negative supplementary results, the work fits a workshop or an evaluation/negative-results track better than IEEE TVT/TMC or INFOCOM main tracks until a method improvement is demonstrated.

---

## Extended experiments: data-plane attacks, V2V trust and empirical edge calibration

*Run `run_20261023_165008_extended_data_plane_edge_calibration` (6 seeds, same protocol as the supplementary run; exploratory, not multiplicity-corrected; Fig. 13).  The active benchmark is unchanged.*

**Motivation.**  A system audit (`docs/SYSTEM_AUDIT.md`) found that the benchmark's model-level attackers behaved honestly as V2V helpers, so the behavioural tier of the trust framework had nothing to detect.  The "behavioural trust adds nothing" finding is therefore conditional on that threat model.  We add a data-plane attacker (poisoning vehicles drop 70 % of V2V tasks and slow the rest 3x; free riders refuse 80 %), compute-rich helper vehicles (30 % at 4-8 GHz; without them a peer is never faster than the requester and V2V cannot pay off), and a trust-aware peer-selection rule.

**P - V2V heuristics under the data-plane attack.**  The attack costs a trust-blind V2V heuristic 2.9 pp (CI [-3.6, -2.2]).  A heuristic that skips peers with behavioural trust below 0.5 recovers 0.9 pp (CI [+0.7, +1.1]), and one that also selects the peer by trust, proximity and spare queue recovers 2.2 pp (CI [+1.5, +2.9]), i.e. it is back to the unattacked level.  Two cautions: (i) the peer-selection rule also gains 1.6 pp (CI [+1.1, +2.1]) *without* any attack, so part of the gain is spare-queue and CPU awareness, not trust; the trust-gating effect alone is +0.9 pp under attack versus +0.1 pp without (a small but attack-specific effect); (ii) under attack a trust-blind V2V heuristic is 1.8 pp worse than the same heuristic ignoring V2V altogether (CI [-3.3, -0.3]).  This is the first measurement in this project in which behavioural trust has a measurable, attack-specific benefit, and it is small (about 1 pp).

**Q - learned policy under the data-plane attack.**  With trust features 77.6 % versus 76.7 % masked (+0.9 pp, CI [-1.4, +3.3]): not established.  The learned policy does not exploit behavioural trust even where a heuristic does.

**R - empirical edge calibration** (Alibaba edge-platform trace: co-tenant load following the measured diurnal profile, measured hand-over delay and loss).  At the 20:00 peak the co-tenant share reaches about 60 % of RSU CPU and Nearest-RSU collapses to 16.8 % completion; Resource-Aware holds 59.9 %.  The learned policy (58.3 %) is identical to Local-Only: it has collapsed to local execution.  At the 05:00 trough it is 12.9 pp below Resource-Aware (CI [-24.6, -1.2]).  **The learned policy still does not beat a simple load-aware heuristic.**

**Reading.**  These results strengthen the honest-evaluation framing: the threat model determines whether behavioural trust matters, load-awareness matters more than trust or learning, and the DQN has not learned to use the information the heuristics use.  Effects in P are small and come from 6 seeds.  Limits: the attack is scripted, the trace calibrates only the edge side (it contains no vehicular data), and its capacity heterogeneity is damped and clipped.

---

## Method attempt: a do-no-harm exclusion rule and a stronger learner

*Studies `results/dnh_study`, `results/learner_study`; full tables in `docs/DNH_DETECTOR.md`; success criteria were fixed before running.  12 paired seeds (6 for the other attack families); Fig. 14-15.*

**Exclusion rule.**  The deployed detector's false exclusions (10.3 % of honest clients with no attackers, 4.6 % at 24 %) come from relative z-scores that tighten as training converges.  We replace them with two absolute harm statistics (norm ratio; cosine with a leave-one-out trimmed mean), a logistic score fitted on separate recorded runs, an exponentially-weighted evidence over rounds, a 0.5 % false-positive budget for the threshold, and a cap on the excluded fraction.  Reputation over rounds and the false-positive problem of binary exclusion are not new (RepuNet, FLARE); the contribution is the absolute-statistic, budgeted rule and its downstream evaluation.

**Results.**  With no attackers the rule costs -0.8 pp against FedAvg (CI [-1.9, +0.3]) versus -6.4 pp for the deployed rule (false exclusions 1.7 % vs 10.3 %).  At 40 % attackers it gains +17.6 pp over FedAvg (CI [+10.6, +24.6]; +1.6 pp over the deployed rule, CI [-0.3, +3.5]).  At 24 % the gain over FedAvg (+3.0 pp, CI [-3.8, +9.8]) is not resolved.  Norm-matched and partial sign-flip attackers are **not** detected (TPR about 0.5, the free riders); the rule merely avoids the deployed rule's harm there.  Cost: 136 ms per aggregation versus 12 ms (FedAvg) and 52 ms (deployed), idle machine, 25 clients.

**Learner.**  Giving the DQN the heuristics' own delay estimates and bandit targets (gamma = 0) beats Resource-Aware on the benchmark environment (+6.6 pp, CI [+3.6, +9.6]) but only ties it under a rotating faulty RSU (-0.4 pp) and under the data-plane attack (+0.0 pp), and loses at the edge-trace peak hour (-7.6 pp).  Hiding the trust features changes nothing in any environment.

**Reading.**  Two of the roadmap's open items move: the cost of the defence at zero attackers is largely removed, and a learned policy beats the best simple heuristic in the benchmark environment.  Three do not: no significant benefit at the default 24 % attack ratio, no benefit from trust features in the learner, and no advantage under hidden-state conditions.
