# Literature Database — Phase 1 (verified only)

> **40 papers, every one independently verified by the agent** via the arXiv API,
> OpenAlex DOI records, PMLR proceedings pages, the USENIX publisher page, or
> bibliographic records (APA/Google Books/author page) for the textbook.
> Machine-readable table: [`papers.csv`](papers.csv).
> Coverage matrix and gaps: [`gap_matrix.md`](gap_matrix.md).
> Mathematical specification: [`../docs/PHASE1_SPEC.md`](../docs/PHASE1_SPEC.md).

## Method

1. Five parallel topic searches (vehicular offloading+DRL; VANET trust;
   federated RL; poisoning+robust aggregation; free-riding+non-IID), each with
   2–6 recorded queries, returning only papers with a DOI and/or arXiv ID plus
   the verification URL actually seen. Seminal foundations (DQN, FedAvg,
   FedProx, SCAFFOLD, Krum, backdoor FL, MEC surveys) were added by the agent.
2. The agent re-verified **every** candidate: bulk arXiv API `id_list` queries
   (29 IDs, all resolved with matching titles/authors), bulk OpenAlex DOI
   lookup (15 DOIs, all resolved), PMLR pages (Fraboni AISTATS 2021; Xie Zeno
   ICML 2019), the USENIX Security'20 page (Fang et al.), and an OpenAlex +
   web-search check for the Krum proceedings title and the Sutton & Barto book.
3. Corrections applied during verification (recorded in `papers.csv` notes):
   FedDOVe first author is *Vivek* (not Vishal) Sethi; BARS 4th author is
   Zhenglin Liu; B5/B7 author spellings corrected per OpenAlex; BARS canonical
   DOI uses slashes; online-vs-issue years noted for B1/B3/B5/C6/C7.
4. Search coverage is limited (5 topic bundles, English-language, 2013–2025);
   absence from this corpus is **not** proof of a global research gap
   (see `gap_matrix.md` G-statements, all provisional).

## Composition (40 papers)

| Group | IDs | Content |
|---|---|---|
| F — Foundations | F01–F12 | DQN/Double-DQN/RL textbook; FedAvg/FedProx/SCAFFOLD/FL survey; Krum; backdoor FL; 2× MEC surveys |
| A — Vehicular offloading + DRL | A01–A05 | Decentralized DDPG; DROO; multi-agent TD3; collaborative RSU offloading; DRL vehicle selection for async FL |
| B — VANET/IoV trust | B01–B07 | 2 surveys; blockchain trust; BARS reputation; AI-enabled trust; multifaceted trust; trust+DRL |
| C — Federated (D)RL | C01–C06 | Fed Q-learning theory ×2; FedRL framework; FRL taxonomy; DRL-meets-FL edge; FedDOVe |
| D — Attacks + robust aggregation | D01–D06 | FoolsGold; median/trimmed-mean theory; Zeno; local-model poisoning; FLTrust; backdoor+clipping study |
| E — Free-riding + non-IID | E01–E04 | Fraboni free-rider analysis; Lin STD-DAGMM; PASS audit; Zhao non-IID measurement |

## Considered but excluded (with reason — not cited as evidence)

- Qi & Ma, *Vehicular Edge Computing via Deep Reinforcement Learning* (arXiv:1901.04290): relevant but explicitly a "preliminary report" preprint.
- X. Wang et al., *MADDPG task offloading in vehicle-edge computing* (DOI 10.1145/3640912.3640992, CNML'23): verified, but workshop proceedings — weakest review signal in group A.
- Liu et al., *Lifelong Federated RL* (DOI 10.1109/LRA.2019.2931179): verified; cloud-robotics navigation, tangential to vehicular offloading.
- Lim et al., *Federated RL for IoT control* (DOI 10.3390/s20051359): verified; same-type devices, weaker venue, covered by C04 taxonomy.
- Acar et al., *FedDyn* (arXiv:2111.04263): verified ICLR 2021 method; extra non-IID comparator beyond FedProx/FedNova scope — revisit in Phase 10 if time permits.
- Recasens et al., *FRIDA* (arXiv:2410.05020): verified preprint but unreviewed, and privacy-attack machinery is out of minimum scope.
- Lin et al. free-riders appears once (E02); group D cross-references it instead of duplicating.

## Verification log (agent-run, 2026-10-11)

- `export.arxiv.org/api/query?id_list=` ×3 batches: 8/8, 10/10, 11/11 arXiv records resolved
  (one ID, 2305.10697, needed a single-record re-fetch; confirmed).
- `api.openalex.org/works?filter=doi:` ×3 lookups: 3/3, 12/12, 1/1 DOIs resolved
  (BARS required the slash-form DOI; hyphen form returns nothing — recorded).
- PMLR v130 (Fraboni) and v97 (Zeno) pages fetched; USENIX Security'20 Fang page
  fetched (confirms no-DOI policy, pp. 1605–1622); Krum proceedings title confirmed
  via OpenAlex (papers.nips.cc landing page); Sutton & Barto 2nd ed. confirmed via
  author page + APA record + Google Books (ISBN 9780262039246).
