# Comprehensive Research, Novelty, and Literature Audit: Phases 0–6

**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions  
**Date of Audit:** October 12, 2026  
**Auditor:** Autonomous Verification & Audit Agent  
**Status:** Audit Completed; Deficiencies Documented and Corrected.

---

## 1. Executive Summary & Objective

This research audit re-evaluates the foundational claims, literature corpus (`reports/papers.csv` and `reports/papers.md`), and proposed research gaps (G1–G6 in `reports/gap_matrix.md`) against the actual published state of research up to late 2025/2026.

The goal of this project is to simulate vehicular offloading, track dynamic participant trust, train local reinforcement-learning (DQN) agents, aggregate updates federatively, and test resilience against adversarial participants. Prior to advancing to Phase 7 (model-update trust and robust aggregation), this audit rigorously addresses whether:
1. The 40-paper literature baseline is verified and authoritative.
2. The stated research gaps G1–G6 remain defensible or must be narrowed.
3. The proposed candidate contribution is novel in its specific mechanism, threat model, and evaluation, rather than an ungrounded combination of buzzwords.

---

## 2. Re-Verification of Existing 40-Paper Literature Corpus

The corpus of 40 papers recorded in `reports/papers.csv` was subjected to independent verification.

### 2.1 Verification Overview
- **Authoritative Identifiers:** All 40 papers have valid DOIs, arXiv identifiers, or authoritative proceedings URLs (ACM, IEEE, PMLR, USENIX, Nature).
- **Core Foundations (F01–F12):** Foundational citations for DQN (Mnih 2013/2015), Double-DQN (van Hasselt 2015), Sutton & Barto (2018), FedAvg (McMahan 2017), FedProx (Li 2020), SCAFFOLD (Karimireddy 2020), Krum (Blanchard 2017), Backdoor FL (Bagdasaryan 2020), and MEC surveys (Mach 2017, Mao 2017) are authentic, canonical, and accurately cited.
- **Vehicular Edge & DRL Offloading (A01–A05):** Chen & Wang (2020), Huang et al. (DROO, 2020), Xue et al. (2025), Wang et al. (2023), and Wu et al. (2023) accurately reflect the state of decentralized DRL offloading.
- **Trust Management in VANETs (B01–B07):** Surveys by Hbaieb et al. (2022) and Amari et al. (2023), blockchain-based systems (Yang 2019, Lu 2018, Zhang 2021), multi-signal models (El-Sayed 2022), and DRL-trust control (Zhang 2022) are accurately indexed.
- **Federated Reinforcement Learning (C01–C06):** Theory papers (Woo et al. 2023, Wang et al. 2024), early frameworks (Zhuo 2019), surveys (Qi 2021), multi-timescale edge FL (Yu 2021), and FedDOVe (Sethi & Pal 2023) are verified.
- **Robust Aggregation, Attacks, and Free-Riding (D01–D06, E01–E04):** FoolsGold (Fung 2020), coordinate-wise median/trimmed-mean (Yin 2018), Zeno (Xie 2019), poisoning attacks (Fang 2020), FLTrust (Cao 2021), backdoor clipping (Sun 2019), free-riding analysis (Fraboni 2021, Lin 2019, Wang PASS 2024), and non-IID data (Zhao 2018) are verified.

---

## 3. Literature Advances (2024–2026) & Frontier Analysis

A targeted search into the 2024–2026 literature reveals substantial activity in Byzantine-robust FRL and trust-aware edge computing:

1. **Provably Robust Federated Reinforcement Learning (WWW 2025):**  
   *Authors:* Minghong Fang, Xilong Wang, Neil Zhenqiang Gong (arXiv:2502.08123).  
   *Contribution:* Demonstrates that existing Byzantine-robust aggregation rules (including coordinate-wise median, trimmed mean, and distance-based filtering) are vulnerable to a novel **Normalized attack** that maximizes update angular deviation. Proposes an ensemble-based FRL approach with majority voting / geometric median action selection with theoretical security bounds.  
   *Impact on Project:* Directly impacts G2 and G4. Shows that statistical aggregation defences alone fail against angular poisoning in FRL, reinforcing the need for multi-factor update trust combined with behavioural verification.

2. **Byzantine-Robust FRL via Critical Parameter Analysis (2025):**  
   *Authors:* Wenzheng Jiang et al. (*Int. J. Machine Learning and Cybernetics*).  
   *Contribution:* Evaluates the sensitivity of neural network parameters in FRL to isolate and prune adversarial gradient perturbations.

3. **Cognitive Trust-Aware & Zero-Trust FRL in VEC (2025–2026):**  
   Emerging works in vehicular edge computing integrate zero-trust architecture (ZTA) and multi-agent FRL for computation offloading. However, most require complex blockchain consensus or assume server root datasets.

---

## 4. Re-Examination and Critical Assessment of Gaps G1–G6

| Gap ID | Claim & Closest Papers with Verified Links | What Prior Work Solved (Verified Facts) | Residual Unsolved Gap (Precise Research Problem) | Audit Status |
|---|---|---|---|---|
| **G1** | **Combined Behavioural and Model-Update Trust in FRL**<br>• FLTrust ([Cao et al., NDSS 2021, arXiv:2012.13995](https://arxiv.org/abs/2012.13995))<br>• Multifaceted Trust ([El-Sayed et al., IEEE TITS 2022, 10.1109/TITS.2022.3187788](https://doi.org/10.1109/TITS.2022.3187788))<br>• Provably Robust FRL ([Fang et al., WWW 2025, arXiv:2502.08123](https://arxiv.org/abs/2502.08123)) | FLTrust solved Byzantine-robust aggregation using cosine similarity to a *server-trained model update on a clean root dataset*. El-Sayed solved direct/indirect multi-signal trust for vehicle routing. Fang et al. solved robust action selection via policy ensembles. | In decentralized vehicular edge environments where edge servers lack clean root datasets for DQN offloading policies, how to estimate model-update trust from peer consensus and combine it with physical task execution trust ($T_{\text{beh}}$) without server-side root bootstrapping remains unmeasured. | **Supported (Narrowed)** |
| **G2** | **Adversarial & Trust Evaluation for Federated DQN Offloading**<br>• FedDOVe ([Sethi & Pal, FGCS 2023, 10.1016/j.future.2022.11.012](https://doi.org/10.1016/j.future.2022.11.012))<br>• Two-Timescale FRL Offloading ([Yu et al., IEEE JIOT 2021, arXiv:2009.10601](https://arxiv.org/abs/2009.10601)) | FedDOVe demonstrated federated deep Q-learning for vehicular fog task offloading under *completely honest and reliable participants*. Yu et al. solved multi-timescale offloading without adversaries. | Neither FedDOVe nor prior VEC federated DQN works test model poisoning (sign-flip, perturbation), free-riding (stale/zero updates), or execution unreliability during collaborative offloading. | **Supported** |
| **G3** | **Transferability of Supervised FL Defences to Non-Stationary Q-Networks**<br>• FoolsGold ([Fung et al., RAID 2020, arXiv:1808.04866](https://arxiv.org/abs/1808.04866))<br>• Free-rider Attacks ([Fraboni et al., AISTATS 2021, arXiv:2006.11901](https://arxiv.org/abs/2006.11901))<br>• Non-IID FL ([Zhao et al., arXiv 2018, arXiv:1806.00582](https://arxiv.org/abs/1806.00582)) | Prior art characterized poisoning and free-riding defences for *supervised image classification* where loss gradients point toward a stationary empirical distribution. | In federated DQN, non-stationary TD targets and exploratory divergence create legitimate update variance across honest non-IID clients. Whether norm-clipping and cosine filters induce false-positive exclusions against honest edge participants has not been empirically quantified. | **Supported** |
| **G4** | **Theoretical Robustness Limitations in Neural FedRL**<br>• Heterogeneity in Fed-Q ([Woo et al., ICML 2023, arXiv:2305.10697](https://arxiv.org/abs/2305.10697))<br>• Convergence Rates in Fed-Q ([Wang et al., arXiv 2024, arXiv:2409.03897](https://arxiv.org/abs/2409.03897))<br>• Byzantine FRL Bounds ([Fang et al., WWW 2025, arXiv:2502.08123](https://arxiv.org/abs/2502.08123)) | Woo et al. and Wang et al. established finite-sample bounds for *tabular Q-learning* across heterogeneous environments. Fang et al. proved bounds for *ensemble policy voting*. | No convergence guarantee exists for single-model FedAvg parameter aggregation of deep Q-networks under arbitrary Byzantine gradient manipulation. Consequently, claims of robustness in neural federated DQN must be empirical and benchmark-driven. | **Supported** |
| **G5** | **Disambiguating Disconnection from Adversarial Malice**<br>• VANET Trust Survey ([Amari et al., IEEE Access 2023, 10.1109/ACCESS.2023.3268991](https://doi.org/10.1109/ACCESS.2023.3268991))<br>• IoV Trust Survey ([Hbaieb et al., Comput. Netw. 2022, 10.1016/j.comnet.2021.108558](https://doi.org/10.1016/j.comnet.2021.108558)) | Established taxonomies of trust management in vehicular networks acknowledging link volatility and mobility handovers. | Concrete trust scoring mechanisms that isolate intermittent channel drops (unobserved tasks) from active misbehaviour (reported failures, delays) without false penalization are rarely evaluated with precision/recall against ground-truth labels. | **Supported** |
| **G6** | **Malicious Client Ratio Sweeps (0% to 40%) in Offloading Systems**<br>• Local Model Poisoning ([Fang et al., USENIX Security 2020](https://www.usenix.org/conference/usenixsecurity20/presentation/fang))<br>• Backdoor FL ([Sun et al., NeurIPS 2019, arXiv:1911.07963](https://arxiv.org/abs/1911.07963)) | Evaluated degradation under varying Byzantine ratios in supervised distributed optimization. | Vehicular edge offloading papers rarely sweep malicious participant ratios from 0% to 40% to quantify the exact operational point where federated offloading policies collapse relative to local execution. | **Supported** |

---

## 5. Definition of the Actual Research Contribution

The contribution of this project is **not**:
- The invention of federated learning or deep Q-networks.
- A formally proven Byzantine consensus guarantee for deep neural networks.
- A generic combination of "blockchain, trust, AI, and 6G."

The candidate contribution is **strictly defined as**:
1. **A Dual-Trust Architecture for Decentralized VEC:** A lightweight mechanism that decouples and tracks:
   - *Behavioural Trust ($T_{\text{beh}}$):* Evaluated by requesters from observable physical outcomes (task success, timeliness relative to deadline, queue fidelity, path ACK), with neutral temporal ageing that explicitly avoids penalizing mobility-induced disconnects as malice.
   - *Model-Update Trust ($T_{\text{model}}$):* Evaluated by the coordinator without requiring a server root dataset, using peer-relative norm plausibility, directional cosine agreement, staleness discounting, and zero/duplicate parameter detection.
2. **Robust Aggregation Protocol:** A trust-weighted parameter aggregation rule with an empirically evaluated exclusion threshold ($\tau_{\text{exc}}$) and a deterministic fallback that preserves the last known good global model when zero clients exceed the trust threshold.
3. **Empirical Benchmarking under Heterogeneity and Adversaries:** Comprehensive evaluation across 0% to 40% malicious client populations, benchmarking against Local Execution, Random Offloading, Nearest Node, Resource-Aware Offloading, Single-Agent DQN, and Standard FedAvg DQN under mobility and non-IID workloads.
