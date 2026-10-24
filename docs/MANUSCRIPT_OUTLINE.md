# Manuscript outline: one story, every claim tied to evidence (24 Oct 2026)

**Working title.**  *A False-Positive-Budgeted Exclusion Rule for Federated Offloading Policies: Downstream
Evaluation under Poisoning in Vehicular Edge Networks*

**One-sentence contribution.**  Excluding poisoned clients from a federated offloading policy is only worth doing
if it does not exclude honest ones; a rule built on absolute (non-relative) update statistics with a measured
false-positive budget removes the defence's no-attacker cost and matches the strongest classic baseline, and a
label-free variant keeps working where Multi-Krum gives no benefit.

## Abstract (draft; every number is in the cited file)

Federated reinforcement learning lets vehicles train a shared task-offloading policy, but compromised clients can
poison it and exclusion defences can discard honest ones.  We evaluate defences by their downstream effect on task
completion in a trace-calibrated vehicular edge simulator, on paired seeds with pre-specified, Holm-corrected tests.
The population-relative trust rule we started from excludes 10.3 % of honest clients with no attackers and costs
6.4 pp of completion.  A rule based on two absolute statistics, smoothed over rounds and thresholded at a measured
0.5 % false-positive budget, cuts false exclusions to 1.7 % and the no-attacker cost to 0.8 pp, while improving
completion over FedAvg by 6.1 pp at 24 % and 16.2 pp at 40 % sign-flip attackers (24 seeds).  It ties Multi-Krum and
beats coordinate median, trimmed mean and an adapted FLTrust; a label-free variant calibrated on attacker-free runs
detects norm-matched attackers (TPR 0.85, FPR < 1 %) and, at 40 % attackers, beats Multi-Krum by 6.6 pp (confirmed
on fresh seeds), where Multi-Krum gives no benefit over FedAvg.  We also report negative results: partial sign-flip attackers are not detected,
trust features do not help the learned policy, the gain shrinks and becomes unresolved at 50-100 vehicles, and
conclusions about heuristics versus learning swing with the edge-load model.  Code, seeds and raw results are released.

## Sections and the evidence behind each claim

| Section | Claim | Evidence |
|---|---|---|
| 3 Simulator | IDM/NGSIM mobility, V2X channel, edge queues, trace-calibrated RSUs, data-plane attacks | `docs/ENVIRONMENT_SPEC.md`, `docs/EDGE_TRACE_USE.md`, `docs/SYSTEM_AUDIT.md` |
| 4 Method | absolute statistics, false-positive budget, label-free v2 | `trust/dnh_detector.py`, `docs/DNH_DETECTOR.md` sec. 2, 7 |
| 5 Protocol | paired seeds, oracle/clean references, Holm | `docs/METHODOLOGY.md`, `experiments/hypothesis_study.py` |
| 6.1 Cost of defence | -6.4 -> -0.8 pp; FPR 10.3 -> 1.7 % | `docs/DNH_DETECTOR.md` sec. 3 (Fig. 14) |
| 6.2 Hypotheses H1-H5 | H2-H5 supported with scope, H1 completion only | `docs/HYPOTHESES.md` (Fig. 16) |
| 6.3 Baselines | ties Multi-Krum, beats median / trimmed mean / FLTrust-style | `docs/BASELINES.md` (Fig. 17) |
| 6.4 Stealth | v2 TPR 0.85; beats Multi-Krum at 40 % (exploratory) | `docs/DNH_DETECTOR.md` sec. 7, 9 |
| 6.5 Robustness | deployments, scale, learner under realistic load | `docs/HYPOTHESES.md` addendum, `docs/DNH_DETECTOR.md` sec. 8 |
| 7 Limits | partial attacks, simulation only, scale cost O(n^2 d), FLTrust adaptation | `docs/LIMITATIONS.md` |
| 2 Related work | gap is narrow; persistence/reputation is prior art | `docs/LITERATURE_GAP.md`, `docs/LITERATURE_REVIEW.md`, `docs/NOVELTY.md` |

## Move to an appendix or a second paper

Data-plane/V2V study, learner study, edge-load sensitivity, console.  They are real but dilute the single story.

## Before submission

1. Database literature search with the strings in `docs/LITERATURE_GAP.md` section 3; verify every row in full text.
2. Re-verify the target call (2027 editions), page limit and anonymisation.
3. Report the 40 % norm-matched result in the main text: the v2 vs Multi-Krum comparison is confirmed (C1); the gain over FedAvg is raw-interval only (C2) and the Multi-Krum harm did not replicate (C3).
4. Optional strengtheners: more seeds at 24 % for the stealth rows, partial-attack bound, scale (done to 16 seeds: still unresolved).
