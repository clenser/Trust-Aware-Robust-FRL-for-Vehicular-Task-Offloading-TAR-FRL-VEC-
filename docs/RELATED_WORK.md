# Related work and research positioning

Every entry below was checked against its publisher, arXiv or institutional
record during this work (2026-10-22).  Where only an abstract could be read,
the entry says so; nothing here was taken from memory of a citation alone, and
no claim is made about a paper's content beyond what its abstract states.

## 1. Literature matrix

| # | Work | Problem | Threat model | Learning / aggregation | Evaluation (as stated) | Limitation relevant to VANNET |
|---|---|---|---|---|---|---|
| 1 | Zhang, Guo, Zhou, Liu. *Trusted Task Offloading in Vehicular Edge Computing Networks: A Reinforcement Learning Based Solution.* IEEE GLOBECOM 2023, pp. 6711-6716 (NWPU record; abstract read) | V2I/V2V trusted offloading | Malicious vehicles "inevitably" present; adversary not formally defined in the abstract | Recommendation-based trust model + DDPG; **not federated** | Numerical results; latency and "credibility" | No federated aggregation, so no model-update trust and no poisoning of a shared policy |
| 2 | Consul et al. *A Reliable Zero-Trust Network for Task Offloading in Vehicular Systems Using an Asynchronous Federated Learning Approach in 6G.* ACM SIGCOMM'24 Workshop on Zero Trust Architecture for Next-Gen Communications (City Research Online record; abstract read) | Offloading to untrusted edge access points | Untrusted RSUs / edge nodes | Edge Node Score + federated asynchronous RL for resource management | Simulation, no metrics in the abstract | Trust is on the *edge node*, not on federated *clients*; model poisoning by clients is outside the stated threat model |
| 3 | Chen et al. *An Entropy-Based Privacy-Preserving Federated Deep RL Framework for Task Offloading in VEC Networks* (FedVTO). IEEE IoT Journal 13(5):8011-8024, 2026, DOI 10.1109/JIOT.2025.3602798 (HBKU record; abstract read) | Offloading with privacy leakage | Privacy leakage, not poisoning | Three-tier FL + DRL, entropy-adapted weighting | Simulation vs. "traditional methods" | No Byzantine or poisoning threat model |
| 4 | Fan et al. *Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee* (FedPG-BR). NeurIPS 2021, arXiv:2110.14074 (abstract read) | FRL with faulty or adversarial agents | Fewer than half of agents faulty or adversarial | Variance-reduced federated policy gradient with a gradient-based Byzantine filter | Standard RL benchmarks | Not a vehicular offloading setting; policy-gradient agents, no mobility, no behavioural trust |
| 5 | Ganesh et al. *Global Convergence Guarantees for Federated Policy Gradient Methods with Adversaries*, arXiv:2403.09940 (abstract read) | FPG with arbitrary-value adversaries | Arbitrary adversarial messages to the server | Robust federated policy gradient | Theory plus experiments | Same: not a vehicular task-offloading study |
| 6 | Cao, Fang, Liu, Gong. *FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping.* NDSS 2021, arXiv:2012.13995 (abstract read) | Byzantine-robust FL | Local model poisoning, including adaptive attacks | Server-held *root dataset* defines a trust anchor; cosine trust score and norm normalisation | Six datasets | Requires a clean root dataset at the server; none exists for an RL policy trained on private vehicle experience |
| 7 | Fang, Cao, Jia, Gong. *Local Model Poisoning Attacks to Byzantine-Robust Federated Learning.* USENIX Security 2020 (abstract read) | Attacks on robust aggregation | Optimised local model poisoning | Targets Krum, trimmed mean, median | Four datasets | Shows that distance/statistics based defences can be defeated by *adaptive* attackers; our threat model is non-adaptive (see LIMITATIONS S5) |
| 8 | Yin, Chen, Ramchandran, Bartlett. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.* ICML 2018 (PMLR v80) | Robust aggregation | Byzantine workers | Coordinate-wise median and trimmed mean | Theory plus experiments | Our median / trimmed-mean baselines |
| 9 | Blanchard, El Mhamdi, Guerraoui, Stainer. *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (Krum). NeurIPS 2017 | Robust aggregation | Byzantine workers | Distance-based selection | See paper | Our Multi-Krum baseline; the secondary description of Krum was read, the original was not retrieved in this session |
| 10 | Fung, Yoon, Beschastnikh. *Mitigating Sybils in Federated Learning Poisoning* (FoolsGold), arXiv:1808.04866 | Sybil poisoning | Sybils whose updates are unusually similar | Similarity-based learning-rate adaptation | Label-flip and backdoor attacks | Exploits *similarity* of attackers; our attackers are independent |
| 11 | Wan, Wang. Trust-evaluation-based federated learning for the Internet of Vehicles, AHSWN 58(3-4), 2024 (search-result abstract only) | Model poisoning and non-IID slowdown in IoV FL | Malicious nodes | Trust-evaluation-based FL | Not verified | Supervised-learning FL, no offloading policy |

Not verified in this session, and therefore **not** cited as support: the
2026 preprint on federated meta-DRL offloading (arXiv:2605.18437) was seen in a
search listing only, and the blockchain/reputation IoV papers were seen as
search snippets only.

## 2. What the matrix shows

* **Trust-aware offloading** (row 1, 2) and **federated RL offloading** (rows 2,
  3) are both published.  Neither addresses poisoning of the *shared policy* by
  federated clients.
* **Byzantine-robust FRL** (rows 4, 5) exists with theory, but on generic RL
  benchmarks, not on mobility-driven edge offloading with intermittent
  participation.
* **Byzantine-robust FL aggregation** (rows 6-10) is mature.  Its published
  weaknesses are adaptive attackers (row 7) and the root-dataset requirement of
  FLTrust (row 6).

## 3. The gap, stated narrowly

> Is there a measurable *downstream* benefit - on task completion in a
> vehicular edge-offloading simulation - of excluding poisoned clients from the
> aggregation of a federated *offloading policy* when the clients are
> intermittently present, hold non-IID workloads, and may be benign-but-small?

The closest prior work does not test this: row 1 has no federation, rows 2-3 do
not model client poisoning, rows 4-5 are not vehicular.  This is a **gap in
evaluation evidence**, not a claim of an algorithmic invention.

## 4. What VANNET does and does not claim

* **Not claimed as novel:** the combination "VANET + trust + FL + RL"; the
  cosine / robust-z outlier statistics; the joint trust formula
  `alpha*T_beh + (1-alpha)*T_model`; trust-weighted averaging.  All are
  recombinations of known elements (rows 6, 8, 9, 10).
* **What the implementation contributes**, if the evidence supports it:
  1. a *participation rule* that separates "had no experience" from
     "malicious", with a documented defect it fixed (CLAIMS C10);
  2. an evaluation protocol that keeps ground truth, detector decision and
     task outcome separate, and that measures the **downstream** effect of
     the defence, not only its TPR/FPR;
  3. a reproducible simulator with provenance and an explicit statement of what
     is measured, trace-replayed or simulated.
* **What the evidence required to claim a downstream benefit is:** a
  statistically established difference between *defended* and *undefended*
  policies under attack, relative to a *clean* (no-attack) reference, on
  matched evaluation episodes, with the attack strong enough that the
  undefended policy is measurably degraded.  `results/claims_report.md` states
  whether the active run provides it.

## 5. Datasets and tools considered

| Resource | What it actually contains | Decision |
|---|---|---|
| FHWA NGSIM US-101 (data.transportation.gov / data.gov catalog; FHWA fact sheet, www.fhwa.dot.gov/publications/research/operations/07030) | Vehicle trajectories on southbound US-101, Los Angeles, 15 June 2005, sampled every 0.1 s | **Used** for mobility replay only.  It contains no tasks, channels, rewards, FL updates or attack labels.  The data.gov catalogue entry only says "see this page"; the project's own provenance record (`data/PROVENANCE.md`, recorded 2026-10-21 from the USDOT Socrata catalogue, dataset `8ect-6jqj`) lists **CC BY-SA 3.0 Unported**.  That record was not re-fetched in this session; re-check the dataset page before redistributing derived files |
| SUMO / Veins / OMNeT++ / ns-3 | Microscopic traffic and detailed V2X network simulation | **Not integrated.**  They would replace the abstracted channel with packet-level simulation at a large integration and runtime cost; the research question here concerns the learning and aggregation layer.  Listed as future work |
| 3GPP TR 36.885 | V2X evaluation methodology (path loss, mobility) | Used as the *style* of the channel and workload model; the implementation is a documented approximation, not a conformance claim |
