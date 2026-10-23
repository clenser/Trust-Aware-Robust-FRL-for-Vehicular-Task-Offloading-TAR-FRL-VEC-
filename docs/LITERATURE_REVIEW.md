# Literature review from the local paper collection (23 Oct 2026)

**Status: keyword-screened, partly read.**  This replaces the "partial" status of
`LITERATURE_GAP.md` for the 76 PDFs in `D:\Papers`; it does not replace the
database search that file still requires.

## 1. Method

`data/source/screen_papers.py` converts the first 40 pages of every PDF and counts
seven term groups (vehicular, offloading, RL, FL, trust, poisoning/Byzantine,
edge).  A group is "present" at >= 5 matches.  Output: `docs/literature_screen.csv`.
Two files yield no extractable text (BVFLEMR and Xiang et al. 2022; likely
image-only) and are not counted, which is why 74 of 76 are analysed.  I then read the abstracts of the candidates
(offloading papers and vehicular-security papers) and searched the offloading
papers for `malicious / poison / attack / Byzantine / trust`.  Counts show what a
paper talks about, not how well it does it.

## 2. What the collection is

74 readable papers.  It is dominated by general FL surveys and blockchain-FL
work; very little is on federated *reinforcement learning for offloading*.

| Axis combination (all present at >= 5 matches) | Papers |
|---|---|
| FL + vehicular | 35 |
| FL + poisoning / Byzantine / malicious | 49 |
| FL + trust / reputation | 44 |
| FL + vehicular + poisoning | 26 |
| FL + offloading | 13 |
| FL + RL | 18 |
| FL + RL + offloading | 7 |
| FL + offloading + poisoning | 7 |
| FL + RL + offloading + poisoning | 4 |

The four papers in the last row are all **surveys** (Lim et al., *Federated
learning in mobile edge networks*, IEEE COMST 2020 [`draft1.pdf`]; Khan et al.,
2009.13012; Nguyen et al., IEEE COMST 2021; AbdulRahman et al., IEEE IoT-J 2020).  None is a primary study that trains an offloading policy by
federated RL and attacks the federation.

## 3. The closest primary studies (read at abstract level + attack-term search)

| Paper | Task | FL | RL | Malicious clients? |
|---|---|---|---|---|
| Han et al., *FL-based computation offloading optimisation in edge-supported IoT*, ACM TURC 2019 | offloading | yes | yes | no mention of malicious / poison / attack / trust |
| Li et al., *Task offloading mechanism based on federated RL in MEC*, Digital Commun. Networks 2023 | offloading, DDQN | yes | yes | none; handles slow clients with an outlier test on **communication delay** |
| Chen et al. (Warwick), *Computation offloading in edge-assisted FL* (EAFL) | offloading to cut straggler delay | yes | no | none |
| Kim, *Cooperative FL-based task offloading for tactical edge networks*, IEEE Access 2021 | offloading, bargaining | yes | no | none |
| *FL for energy-efficient task computing in wireless networks*, ICC 2020 (SVM-based user association) | user association | yes | no | none |
| Cao et al., *T-FedHA*, Expert Syst. Appl. 2024 | trusted hierarchical async FL for IoT | yes | no | yes: trust management selects devices; **not** offloading, not RL |

Reading: the federated-offloading literature in this folder assumes benign
participants (the attack-term search returned zero hits in the first five rows).
The trust / poisoning-defence literature (T-FedHA, PrivateFL [`S0957417423027124`],
BlockDFL, the blockchain-FL surveys) is supervised FL, mostly evaluated on
classification accuracy.  The vehicular-security FL papers (VAN-FED-IDS
[`S0167404824001822`], VeReMi misbehaviour detection [`S2542660524000696`], FL
attack detection for the IoV [`IEEE_TRANS_1`], FL with local DP for VANETs
[`S0020025523013026`]) use FL *for* detection or privacy, not as the object being
protected.  So the combination studied here (a federated offloading **policy**,
update poisoning, and a **task-metric** evaluation of the defence) is not
covered by anything in the collection.  This supports the positioning in
`LITERATURE_GAP.md` and is still limited by the collection: it is a reader's
folder, not a systematic search.

## 4. What the collection adds to the argument

1. **The gap is real inside this corpus, and it is narrow.**  Phrase it as "the
   federated-offloading papers we located assume benign clients; the robust-FL
   papers do not measure offloading quality".  Do not say "first".
2. **A protocol contribution stays defensible.**  The offloading papers above
   have no adversary at all, so none can have evaluated a defence by its effect on
   offloading quality.  (I did not audit their statistics, so no claim is made
   about their seed or CI practice.)
3. **Surveys flag the open problem.**  The FL-for-IoT and FL-in-ITS surveys
   (Khan 2021, Nguyen 2021, Zhang 2024 [`2403.07444`]) list poisoning / trust and
   vehicular edge computing as separate open problems; use them for motivation,
   not as evidence of absence.
4. **Related-work material for the paper:** blockchain-based trust and reputation
   (Kasyap and Tripathy's PrivateFL, T-FedHA, BlockDFL), hierarchical / asynchronous FL (T-FedHA, ChainsFL),
   topology-aware FL in edge computing (3659205), and incentive design (zhan2021)
   are the natural "orthogonal approaches" paragraph.  They cost extra
   infrastructure (consensus, ledgers) that the simulator does not model; say so.

## 5. Not done

* No full-text reading beyond abstracts and targeted term searches.
* Several files are off-topic or general-audience (`answers.pdf` is a MANET/VANET
  study-notes sheet; face-recognition FL, healthcare FL, a credit-modelling
  paper, the differential-privacy monograph) and are left out of the argument.
* The database search (IEEE Xplore, Scopus, Scholar) with the strings in
  `LITERATURE_GAP.md` section 3 is still required before submission.
