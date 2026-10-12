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

## Evidence-supported gaps (provisional)

- **G1. Behavioural + model-update trust for federated DQN is untested in-corpus.**
  B-group covers behavioural trust without model updates; D05 covers update trust
  without behaviour and needs a server root dataset; no paper combines both for FRL.
  → RQ1, RQ2.
- **G2. No trust/adversary study exists for federated DQN offloading.**
  C06 (FedDOVe) and C05 do federated (D)RL offloading with honest participants only;
  A-group likewise assumes honesty. → RQ3, RQ4.
- **G3. Free-rider/poisoning defences are unevaluated on federated DRL updates.**
  E01–E03, D01–D03, D05 target supervised FL with stationary targets; Q-network
  updates are non-stationary (moving TD targets), so detector transfer is unproven
  and false positives on honest non-IID clients are unquantified. → RQ2, RQ5.
- **G4. FedRL theory stops at tabular Q-learning.**
  C01–C02 give no guarantees for DQN function approximation, let alone
  trust-weighted robust aggregation under Byzantine clients. Our claims must stay
  empirical. → RQ5.
- **G5. Disconnect-vs-malice disambiguation is unmeasured.**
  B-group models mobility but rarely scores detectors with precision/recall against
  ground-truth attacker labels under intermittent connectivity. → RQ1, RQ5.
- **G6. Malicious-ratio sensitivity is missing for offloading.**
  D06 shows adversary fraction drives backdoor success in supervised FL; no
  offloading paper sweeps 0–40% malicious clients on latency/completion. → RQ4.

## Provisional contribution hypothesis (NOT a claim)

> A dynamic trust model combining behavioural trust (task success, timeliness,
> resources, reliability) with model-update trust (norm/deviation, consistency,
> learning-progress evidence, stale/copied/zero detection) and trust-weighted
> aggregation with threshold exclusion *may* improve resilient federated DQN
> offloading — **iff** Phases 5–10 experiments confirm RQ1–RQ5 hypotheses.
> The nearest prior art to beat or compare against: C06 (no-trust federated DQN),
> D05 (server-rooted trust), D01 (similarity-only defence), F05 (plain FedAvg).
