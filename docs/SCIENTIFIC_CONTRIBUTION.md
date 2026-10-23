# Scientific contribution: what the evidence supports

This document replaces an earlier version that claimed novelty and defence
results the data do not support (for example "complete policy collapse under
30 % attacks", "0 % attack detection" for FedAvg as a *performance* result, and
a deployment path).  It states the contribution at the level the evidence
reaches, and no higher.  Literature positioning is in
[`RELATED_WORK.md`](RELATED_WORK.md); the numbers are in
[`results/claims_report.md`](../results/claims_report.md).

---

## 1. Research question, as narrowed by the evidence

The roadmap asks whether a trust-aware federated reinforcement-learning
framework improves secure and efficient task offloading in dynamic vehicular
edge conditions.  The literature check shows that trust-aware offloading,
federated-RL offloading and Byzantine-robust FRL each exist separately; the
combination "VANET + trust + FL + RL" is **not** novel in itself.  The question
this project can answer with its implementation is the narrower one:

> In a mobility-driven vehicular edge-offloading simulation with intermittent,
> non-IID clients, **does excluding poisoned clients from the aggregation of a
> federated offloading policy measurably protect task completion, and at what
> cost in false exclusions?**

This is a gap in *evaluation evidence* (downstream effect on the task metric,
measured against a clean and an oracle reference), not a new algorithm.

## 2. Contributions that the implementation does support

1. **A downstream-impact evaluation protocol.**  Every defence is compared
   against a no-attack reference, an undefended policy under attack, and an
   oracle that removes the true attackers, on matched evaluation episodes, with
   verdicts produced mechanically (confidence interval, Holm correction,
   minimum detectable effect).  Detection rates (TPR/FPR) are reported but are
   never presented as evidence that the policy is protected.
2. **A reproducible simulator with explicit provenance**, whose model was audited
   and corrected (honoured RSU choice, shared uplink, neutral reward, measured
   communication, completed-task latency) with each change testable through a
   switch that reproduces the previous behaviour.  The effect of those
   corrections on the research conclusion is itself a result (section 4).
3. **A participation rule** that separates "had no experience" from "malicious"
   and removed a documented implementation defect (an elevated false-positive
   rate caused by clients that had not yet trained being judged as free riders).
4. **An honest characterisation of the trust detector**: where it works
   (sign-flip, scaling, noise, free riding), where it fails (partial,
   norm-matched, intermittent), what it costs (false exclusions that grow with
   training), and that its "joint" score is, at the default weights, a threshold
   on the model-update score (`tests/test_trust_geometry.py`).

## 3. What the downstream evidence says (active run, 16 paired seeds)

* The poisoning attack lowers task completion of undefended FedAvg by about
  6 pp (raw p < 0.01; **not** significant after Holm correction across the 8
  pre-specified claims).
* Removing exactly the attackers would recover about 5 pp; trust-aware
  aggregation recovers about 2 pp, which is **not distinguishable from zero**
  (the design resolves about 5-6 pp).
* At 40 % attackers the defence recovers about 18 pp (CI excludes 0, 6 seeds);
  with no attackers it costs about 10 pp (CI contains 0, 6 seeds), because it
  excludes honest clients.
* Multi-Krum is not distinguishable from trust-aware aggregation.
* Federated learning beats independent per-vehicle learning by about 13 pp.
* The learned policy is statistically indistinguishable from the best simple
  heuristic (Nearest-RSU); under the earlier simulator semantics it was better,
  under the first corrected semantics and features it was worse.

## 4. Sensitivity of the conclusion to modelling choices

| Simulator variant | Learned policy vs Nearest-RSU (paired) |
|---|---|
| v1: silent re-association, uncontended uplink, trust bonus | about +3 pp |
| v2.0: honoured RSU choice, contention, neutral reward, road-length-normalised distances | about -12 pp |
| v2.1: as v2.0 with coverage-relative distances and in-coverage flags (final) | about -0.4 pp (n.s.) |

The headline "RL beats a heuristic" is therefore a property of simulator
semantics and of the state representation, not of the learning framework.

## 5. Contributions that are NOT supported

* **Novelty of the trust formula or aggregation.**  The cosine / robust-z
  statistics, `alpha T_b + (1-alpha) T_m` and trust-weighted averaging are
  recombinations of known elements; the MAD-distance rule matches the deployed
  detector on held-out data.
* **Protection of the global policy at the default attack ratio.**  Not
  established.
* **Value of behavioural trust.**  Not shown; at default weights it cannot
  exclude a client with an honest update, and in the mixed-threat experiment it
  does not help.
* **Efficiency.**  The policies use about 30 times the energy of local
  execution; the benefit, where there is one, is completion rate.
* **Real-world validity.**  Simulation only.

## 6. Evidence that would be needed to strengthen the claims

* More seeds (or variance reduction) for the 24 % attack ratio: the SD of paired
  seed differences is about 7.5 pp, so resolving a 2.5 pp defence benefit at 80 %
  power needs roughly 70 paired seeds.
* A stronger attack regime, justified by a threat model, in which the undefended
  policy is degraded by more than the oracle headroom of about 5 pp.
* A detector with a lower false-exclusion rate (the MAD-distance rule is a
  candidate, but it was only evaluated in shadow mode, not as the aggregator).
* A bandit-formulated baseline and a discount-factor sweep.
* A packet-level network simulator (ns-3 / Veins) to replace the abstracted
  channel; this was assessed and not integrated (cost/benefit in `RELATED_WORK.md`).

## Honest status of the dual-tier trust claim

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

---

## Update (24 Oct 2026, overnight): see `docs/NOVELTY.md`

Baselines, a label-free stealth-capable variant, trace-calibrated deployments and a scale check were added; the do-no-harm rule ties Multi-Krum and beats median, trimmed mean and an adapted FLTrust.  `docs/NOVELTY.md` states what is and is not novel.

## Update (24 Oct 2026): hypotheses confirmed with scope

A pre-specified, Holm-corrected study (`docs/HYPOTHESES.md`) supports H2, H3, H4 and H5 for the sign-flip attack in simulation, and H1 only as a reliability benefit (latency effect at most 3 ms).  This revises the earlier statement that the project is not a "trust-aware FRL beats FedAvg" paper: with the do-no-harm rule it is, for this attack and these conditions; the stealth-attack limitation and the unsupported RQ3 remain.

## Update (23 Oct 2026): what the method attempt changed

Two roadmap items were attempted with pre-registered criteria (`docs/DNH_DETECTOR.md`):

* **Do-no-harm exclusion** - criterion met: the cost of the defence with no
  attackers falls from -6.4 pp to -0.8 pp and false exclusions from 10.3 % to
  1.7 %; at 40 % attackers it keeps the gain (+17.6 pp).  The gain at the default
  24 % ratio is still unresolved and norm-matched attackers are still missed.
  Persistence-based reputation is prior art; the claim is limited to the
  absolute-statistic, false-positive-budgeted rule and its downstream evaluation.
* **Learner** - met on the benchmark environment only (+6.6 pp over the best
  heuristic); not met under hidden-state conditions; trust features add nothing.

The paper's contribution therefore becomes: an evaluation protocol, a measured
threat-model dependence of behavioural trust, and one modest method result (the
detector) with its limits stated.  It is still not a "trust-aware FRL beats
FedAvg" paper.
