# Publication plan (October 2026)

## Where the project stands

The roadmap hoped for a *new trust-aware aggregation method that beats FedAvg*.
The evidence does not support that (see `results/claims_report.md`): H1-H5 are not
established after Holm correction, the learned policy is indistinguishable from a
Nearest-RSU heuristic, and the trust formula recombines known statistics.  Chasing
the original claim risks an open-ended loop.  The plan below publishes what the
evidence *does* support, and keeps one optional method contribution that is
reported only if it wins.

## Chosen path: hybrid (A as the base, B as a bonus)

**Path A - honest evaluation paper (the paper to write now).**
Working title: *Evaluating Trust-Aware Federated RL for VANET Offloading: A
Downstream-Impact Protocol.*

| Element | From (roadmap) | To (what the evidence supports) |
|---|---|---|
| Contribution | novel defence mechanism | evaluation protocol + honest characterisation |
| H1-H5 | "supported" | tested; not established at 24 % attackers; established only at 40 % |
| Dual-tier trust | proposed as effective | model-only matches joint; behavioural adds nothing **when attackers are honest in the data plane** |
| Learned policy | beats heuristics | indistinguishable from Nearest-RSU; simulator-sensitive |
| New in this revision | - | data-plane attack and V2V study (exp. P, Q), empirical edge calibration from the Alibaba trace (exp. R), faulty-edge and connectivity figures (Fig. 11-13) |

**Path B - bonus.** The extended suite asks whether the behavioural tier earns its
place once the attack also appears in the data plane (malicious or selfish V2V
helpers) and whether trust-aware peer selection helps.  If experiment P/Q shows a
paired, CI-excluding-zero gain, it becomes a second, clearly labelled
contribution; if not, it is reported as another negative result.  Nothing in the
paper claims it in advance.

## Venues (checked 23 Oct 2026; re-verify dates before submitting)

* **IEEE VNC (Vehicular Networking Conference).**  VNC 2026 (Montreal, 8-10 June
  2026) has passed, including its poster/demo deadline (extended to 1 April 2026
  per a secondary source).  Watch the VNC 2027 call; the poster/demo track suits
  this simulator + protocol well, and the main track is realistic only with a
  positive method result.
* **ACM/IEEE SEC workshops** (EdgeCAV - edge computing for connected and
  autonomous vehicles; FederatedEdge).  The 2026 editions' deadlines have passed;
  the 2027 editions are the natural next target for a short evaluation paper.
* **IEEE ISSRE RENE (Replications and Negative Results).**  The only venue found
  that explicitly accepts negative results, but it is a software-reliability
  venue; fit is weak.
* **Journals.**  IEEE TVT / IoT-J expect a positive method contribution; the
  honest-evaluation framing is a poor fit unless exp. P/Q produce one.
  Reproducibility-oriented journal special issues and the IEEE Open Journal of
  Vehicular Technology are worth checking for a full-length version.

No dedicated IEEE "negative results in vehicular edge learning" venue was found;
the most realistic route is a short or workshop paper first, then a journal
extension that adds the extended experiments.

## Checklist before submission

1. Rebuild every number in `docs/PAPER.md` from `results/claims_report.md`
   (`tests/test_paper_claims_consistency.py` enforces this).
2. Re-run `python -m experiments.extended_suite` only if the environment code
   changes; the run is versioned and recorded in `results/manifest.json`.
3. Verify `evidence/SHA256SUMS` against the decompressed raw results.
4. Keep the "what must not be claimed" list in the README in the abstract's
   limitations sentence.
5. Check the target venue's current call (page limits, anonymisation, deadline).
