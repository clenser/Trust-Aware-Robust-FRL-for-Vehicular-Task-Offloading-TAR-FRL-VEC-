# Research-Gap Matrix — Phase 1 (provisional)

> Every gap below is phrased as a **question pending evidence**, not a novelty
> claim. "Not found" means *not found in the 40-paper Phase 1 corpus and the
> searches run* — not a proven global absence. Corpus: [`papers.csv`](papers.csv).

Legend: Y = addressed · P = partial/tangential · – = not addressed.
Columns: **V**ANET/VEC setting · **O**ffloading decisions · **R**L/DRL ·
**F**ederated learning · behaviour **B** trust · model-update **M** trust ·
**A**ttacks evaluated · robust **G** aggregation.

## Coverage matrix

| ID | Short title | V | O | R | F | B | M | A | G | Limitation w.r.t. our RQs |
|---|---|---|---|---|---|---|---|---|---|---|
| F01–F04 | DQN / Double-DQN / RL book | – | – | Y | – | – | – | – | – | Single-agent, no federation/trust/vehicles |
| F05–F08 | FedAvg / FedProx / SCAFFOLD / FL survey | – | – | – | Y | – | – | – | P | Supervised FL; no RL, no trust, no vehicles |
| F09 | Krum | – | – | – | Y | – | – | Y | Y | Needs honest majority; no trust, no RL |
| F10 | Backdoor FL | – | – | – | Y | – | – | Y | P | Attack only; evades anomaly detectors |
| F11–F12 | MEC surveys | P | Y | – | – | – | – | – | – | Background models only |
| A01 | Decentralized DDPG offloading | P | Y | Y | – | – | – | – | – | Honest users; continuous control, not DQN |
| A02 | DROO | P | Y | Y | – | – | – | – | – | Honest users; no federation/trust |
| A03 | Multi-agent TD3 vehicular | Y | Y | Y | – | – | – | – | – | Multi-agent (out of minimum scope); no trust/adversary |
| A04 | Collaborative RSU offloading | Y | Y | Y | – | – | – | – | – | Cooperation incentives, not security |
| A05 | DRL vehicle selection for AFL | Y | P | Y | Y | P | – | P | – | Filters "bad" nodes but no trust model or attack study |
| B01–B02 | IoV/VANET trust surveys | Y | – | – | – | Y | – | P | – | Taxonomies; no RL/FL/model-update trust |
| B03–B04 | Blockchain trust / BARS | Y | – | – | – | Y | – | P | – | Needs blockchain infra (out of scope); no learning |
| B05 | AI-enabled trust (AIT) | Y | – | – | – | Y | – | Y | – | Learned trust, but no FL/RL integration |
| B06 | Multifaceted trust | Y | – | – | – | Y | – | – | – | Fusion weights; no adversarial FL evaluation |
| B07 | Trust + DRL (SDVN) | Y | P | Y | – | Y | – | – | – | Trust informs RL, but single-controller, no federation |
| C01–C02 | Federated Q-learning theory | – | – | Y | Y | – | – | – | – | Tabular Q-learning, no DQN/function approximation, no adversary |
| C03 | FedRL framework (Zhuo) | – | – | Y | Y | – | – | – | – | Privacy-noise focus; no trust/vehicles |
| C04 | FRL taxonomy (Qi) | P | P | Y | Y | – | – | – | – | Survey; defines HFRL slot our work fills |
| C05 | DRL-meets-FL edge (Yu) | P | Y | Y | Y | – | – | – | – | Federated DRL offloading but no trust/adversary model |
| C06 | FedDOVe | Y | Y | Y | Y | – | – | – | – | Federated DQN offloading **without** trust or attacks — key comparison target |
| D01 | FoolsGold | – | – | – | Y | – | P | Y | Y | Similarity-only; honest non-IID clients can look alike-hostile |
| D02 | Median / trimmed mean | – | – | – | Y | – | – | Y | Y | Coordinate-wise statistics; no trust, break under adaptive poisoning (D04) |
| D03 | Zeno | – | – | – | Y | P | – | Y | Y | Suspicion scores need a clean reference signal; assumes ≥1 honest worker |
| D04 | Local-model poisoning (Fang) | – | – | – | Y | – | – | Y | – | Attack paper: defeats D02/F09-style statistics — motivates trust beyond statistics |
| D05 | FLTrust | – | – | – | Y | – | Y | Y | Y | **Closest analogue**: trust = similarity to *server reference update* — requires a server root dataset, unavailable in pure VANET setting |
| D06 | Backdoor + clipping (Sun) | – | – | – | Y | – | P | Y | P | Mitigation only; no detection/attribution |
| E01 | Fraboni free-riders | – | – | – | Y | – | P | Y | – | Analysis + recommendations; detection left open |
| E02 | Lin free-riders + STD-DAGMM | – | – | – | Y | – | P | Y | P | Preprint; supervised-FL setting, not federated DRL |
| E03 | PASS audit | – | P | – | Y | – | P | Y | P | Parameter auditing incl. >50% adversaries; IoT supervised FL, not DRL |
| E04 | Zhao non-IID measurement | – | – | – | Y | – | – | – | – | Motivates non-IID study; no defence proposed |

## Evidence-supported gaps (provisional, verified 2026 audit)

- **G1. Combined behavioural + model-update trust for federated DQN offloading without server root datasets.**
  B-group covers behavioural trust without model updates; D05 (FLTrust, NDSS 2021) covers update trust but requires a clean server root dataset; WWW 2025 (Fang et al.) uses ensemble voting. Combining peer-observed task outcomes ($T_{\text{beh}}$) with serverless model update metrics ($T_{\text{model}}$) in vehicular edge networks remains unmeasured. → RQ1, RQ2.
- **G2. Adversarial & trust evaluation for federated DQN offloading.**
  C06 (FedDOVe) and C05 evaluate federated DQN offloading under strictly honest, cooperative participants. Prior art lacks evaluation under model poisoning (sign-flip, perturbation), free-riding, and execution unreliability. → RQ3, RQ4.
- **G3. Transferability of supervised-FL defences to non-stationary Q-networks.**
  E01–E03, D01, and D05 address supervised learning with stationary distributions; in DQN, evolving policies and moving TD targets produce legitimate inter-client variance that risk false-positive exclusion of honest non-IID edge participants. → RQ2, RQ5.
- **G4. Absence of Byzantine convergence guarantees for deep Q-networks.**
  C01–C02 establish bounds only for tabular Q-learning; WWW 2025 bounds hold for policy ensembles. No guarantees exist for single-model FedAvg parameter aggregation of deep Q-networks under Byzantine poisoning; all robustness claims must remain strictly empirical. → RQ5.
- **G5. Disconnect-vs-malice disambiguation under intermittent connectivity.**
  B-group taxonomies model vehicular mobility but rarely evaluate detectors with precision/recall against ground-truth attacker labels while subject to channel dropouts and link volatility. → RQ1, RQ5.
- **G6. Malicious-client-ratio sensitivity sweeps in vehicular offloading.**
  D06 and D04 demonstrate that adversary fraction drives backdoor and poisoning impact in supervised FL, but vehicular task offloading literature lacks systematic sweeps from 0% to 40% malicious client populations. → RQ4.

## Provisional contribution hypothesis (NOT a claim)

> A dynamic trust model combining behavioural trust (task success, timeliness,
> resources, reliability) with model-update trust (norm/deviation, consistency,
> learning-progress evidence, stale/copied/zero detection) and trust-weighted
> aggregation with threshold exclusion *may* improve resilient federated DQN
> offloading — **iff** Phases 5–10 experiments confirm RQ1–RQ5 hypotheses.
> The nearest prior art to beat or compare against: C06 (no-trust federated DQN),
> D05 (server-rooted trust), D01 (similarity-only defence), F05 (plain FedAvg).
