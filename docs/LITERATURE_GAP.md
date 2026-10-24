# Literature positioning and gap matrix (23 Oct 2026)

**Status: partial (about 20 works located; abstract level, 24 Oct 2026 update).**  The roadmap asks for 30-40 papers with at least 10 highly
relevant.  This table contains only works that were actually located and read at
the abstract level in web searches on 23 Oct 2026 (listed with the URL used).  It
is *not* a systematic review; the claims below are therefore worded as "we found
no ..." and must be re-checked against a full review (Scopus / IEEE Xplore /
Google Scholar, with the search strings in section 3) before submission.

## 1. Matrix

| Work | VANET / VEC | Offloading | RL | FL | Trust / reputation | Poisoning attack | What it does not do |
|---|---|---|---|---|---|---|---|
| Zero-trust asynchronous federated RL for offloading (SIGCOMM NextGen workshop 2024) [City Research Online](https://openaccess.city.ac.uk/id/eprint/33578/) | yes | yes | yes | yes | node security scoring | no model-update poisoning evaluated (per abstract) | no downstream measurement of exclusion |
| FedVTO, entropy-based privacy-preserving federated DRL offloading (2025) [HBKU](https://elmi.hbku.edu.qa/en/publications/an-entropy-based-privacy-preserving-federated-deep-reinforcement-/) | yes | yes | yes | yes | no | no | privacy, not security of updates |
| Trusted task offloading in VEC with DDPG, recommendation-based trust [NWPU](https://pure.nwpu.edu.cn/en/publications/trusted-task-offloading-in-vehicular-edge-computing-networks-a-re) | yes | yes | yes (DDPG) | no | yes | no (behavioural trust only) | no federation |
| Trusted and efficient offloading in VEC (2024) [NWPU](https://pure.nwpu.edu.cn/en/publications/trusted-and-efficient-task-offloading-in-vehicular-edge-computing) | yes | yes | yes | no | yes | no | no federation |
| Federated DRL offloading with power control, Moon & Lim, Sensors 2022 [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9782689/) | yes | yes | yes | yes | no | no | benign clients only |
| FedMAGS, federated meta-DRL for heterogeneous DAG tasks (2026 preprint) [arXiv](https://arxiv.org/pdf/2605.18437) | yes | yes | yes | yes | no | no | benign clients only |
| Uddin et al., survey of DRL offloading in VEC [arXiv 2502.06963](https://arxiv.org/abs/2502.06963) | yes | yes | yes | not central | - | - | survey |
| Provably robust federated RL; the Normalized attack, Fang, Wang, Gong 2025 [arXiv 2502.08123](https://arxiv.org/abs/2502.08123) | no | no | yes | yes | no | yes (angle-based attack on FedPG-BR) | no vehicular setting, no task metric |
| RoHFL, robust hierarchical FL for the Internet of Vehicles [PolyU](https://research.polyu.edu.hk/en/publications/toward-robust-hierarchical-federated-learning-in-internet-of-vehi/) | yes | no | no | yes | reputation in aggregation | yes | supervised FL, not offloading |
| Reputation + blockchain for IoV FL, Mun et al. 2025 [KU](https://khazna.ku.ac.ae/en/publications/emerging-blockchain-and-reputation-management-in-federated-learni/) | yes | no | no | yes | yes | low-quality updates | not offloading |
| Trust-based client selection for FL in IoV [UAEU](https://research.uaeu.ac.ae/en/publications/a-trust-based-client-selection-framework-for-federated-learning-i/) | yes | no | no | yes | yes | faulty updates | not offloading |
| FLARE, adaptive multi-dimensional reputation (2025) [arXiv 2511.14715](https://arxiv.org/abs/2511.14715) | no | no | no | yes | yes (continuous) | yes | argues that binary exclusion creates false positives; no vehicular / offloading task |
| RepuNet, reputation for malicious clients in decentralised FL [arXiv 2506.19892](https://arxiv.org/abs/2506.19892) | no | no | no | yes (DFL) | yes (EMA across rounds) | yes | no vehicular / offloading task |
| FLAME [arXiv 2101.02281](https://arxiv.org/abs/2101.02281); FoolsGold [RAID 2020](https://www.cs.ubc.ca/~bestchai/papers/foolsgold-raid2020.pdf); FLTrust, Krum (classic robust aggregators, cited secondhand) | no | no | no | yes | partly | yes | supervised FL; FoolsGold-type rules penalise legitimately similar clients |
| SoK on benchmarking poisoning defences in FL [arXiv 2502.03801](https://arxiv.org/abs/2502.03801) | no | no | no | yes | - | yes | supervised FL |
| Wu, Wang, Fan, Fan, DRL vehicle selection for asynchronous FL in VEC (CNIS 2023) [arXiv 2304.02832](https://arxiv.org/abs/2304.02832) | yes | no (FL training, not offloading) | yes (selector) | yes | "bad nodes" removed | not stated in abstract | the learned task is the FL model, not an offloading policy |
| Palit, Adaptive FL defences via trust-aware DQN (2025) [arXiv 2510.01261](https://arxiv.org/abs/2510.01261) | no | no | yes (RL defence controller) | yes | yes (long-horizon trust) | yes (poisoning, backdoor; CIFAR-10) | no vehicular or offloading task; baselines are other controllers (per abstract) |
| Decentralized federated policy gradient with Byzantine fault tolerance (DecByzPG) [arXiv 2401.03489](https://arxiv.org/abs/2401.03489) | no | no | yes | yes (decentralised) | no | yes (random actions, noise, near-zero-mean) | policy-gradient benchmarks, no vehicular task |
| Blanchard et al., Krum, NIPS 2017 [paper](https://papers.NeurIPS.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf) | no | no | no | yes | no | yes | supervised learning; Multi-Krum is the averaging variant |
| Baruch, Baruch, Goldberg, A Little Is Enough, NeurIPS 2019 [arXiv 1902.06156](https://arxiv.org/abs/1902.06156) | no | no | no | yes | no | yes (attack inside the honest variance) | shows distance- and variance-based defences can be circumvented; the closest published relative of the norm-matched attack here |

## 2. What the matrix supports (and does not)

* **Supported:** among the works above, none combines (a) a vehicular edge
  offloading policy trained by federated RL, (b) model-update poisoning, and (c) a
  *downstream task-metric* evaluation of whether exclusion protects the policy.
  This is an observation about a partial search, not a proof of absence.
* **Not novel:** per-round exclusion by robust statistics (Krum, FLAME, FLTrust),
  reputation smoothed over rounds (RepuNet, FLARE), and the observation that
  binary exclusion creates false positives (FLARE).  The do-no-harm rule in this
  repository must therefore **not** be presented as the first persistence-based or
  reputation-based defence.  Its defensible contribution is narrower: a rule built
  on *absolute* harm statistics with an explicit false-positive budget, evaluated
  downstream in an offloading task (see `docs/DNH_DETECTOR.md`).
* **Blame the defence, not only the attacker:** Baruch et al. and Fang et al. both show
  that distance- or angle-based robust aggregators can be defeated by attackers that stay
  inside the honest spread.  Multi-Krum's weak detection of norm-matched attackers
  measured here (TPR about 0.25 at 17 % false exclusion; no benefit over FedAvg on the 12 confirmation
  seeds, -5 pp on the first 12) is consistent with them; it is a task-level measurement of a known failure
  mode, not a new failure mode.
* **Normalized attack (Fang et al.)** shows that angle-based attacks defeat
  distance-based robust FRL aggregators; the stealth / norm-matched attacks in
  this repository are the same family and are *not* defended by any rule here.

## 3. To complete before submission

Search strings: `"federated reinforcement learning" AND ("vehicular" OR "VANET")
AND ("poison*" OR "Byzantine" OR "malicious")`; `"trust" AND "task offloading" AND
"federated"`; `"false positive" AND "honest clients" AND ("exclusion" OR "robust
aggregation")`.  Build the full 30-40 paper matrix and re-verify every row in the
full text.
