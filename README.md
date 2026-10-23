# VANNET

**Trust-aware federated reinforcement learning for secure and efficient task offloading in vehicular edge networks - a reproducible simulator, evaluation protocol and research console.**

VANNET trains a dueling double-DQN task-offloading policy across simulated
vehicles with federated averaging, lets a fraction of the vehicles upload
poisoned or free-riding updates, and measures - on matched episodes, paired by
seed - whether a population-relative trust detector protects the policy.  It is
a **simulation study**: only the optional vehicle motion can be replayed from
real data (FHWA NGSIM US-101).

> **Read this first.**  The evidence is mixed and some of it is negative; the
> verdict of every claim is computed mechanically in
> [`results/claims_report.md`](results/claims_report.md).  The limitations are in
> [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md); a paper-style write-up is in
> [`docs/PAPER.md`](docs/PAPER.md).

## Status at a glance (24 Oct 2026)

**Roadmap hypotheses** (pre-specified, Holm-corrected, paired seeds; details in [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md)):

| | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Trust-aware FRL lowers latency vs non-trust-aware FRL | **Partly** | completion improves; latency of completed tasks differs by <= 3 ms (24 %: none) |
| H2 | Higher reliability / robustness than FedAvg as malicious share grows | **Supported** | +6.1 pp at 24 %, +16.2 pp at 40 % attackers (24 seeds) |
| H3 | Joint (behavioural + model) trust beats behavioural alone | **Supported** | +5.3 pp at 24 %, +19.7 pp at 40 %; behavioural-only is no defence |
| H4 | Advantage holds under other mobility / connectivity | **Supported** | synthetic and authentic NGSIM mobility; offline clients on the raw interval |
| H5 | No impractical overhead | **Met** | 136 ms vs 12 ms per round for FedAvg, no extra upload |

**Built:** simulator (IDM / NGSIM mobility, V2X channel, edge queues, V2V helpers), federated dueling-DQN
offloading, behavioural + model-update trust, a do-no-harm exclusion rule, robust-aggregation baselines,
data-plane attacks, Alibaba edge-trace calibration, a 3D research console, ~400 tests, raw evidence with checksums.

**Not achieved / limits:** no benefit from trust inside the learned policy; the learner beats the best heuristic
only in the benchmark environment; norm-matched and partial attackers are not detected; simulation only;
H1 latency not supported; the rule ties Multi-Krum at 24 / 40 % attackers (see `docs/BASELINES.md`).

Roadmap: [`docs/PUBLICATION_PLAN.md`](docs/PUBLICATION_PLAN.md) | audit: [`docs/SYSTEM_AUDIT.md`](docs/SYSTEM_AUDIT.md) |
literature: [`docs/LITERATURE_GAP.md`](docs/LITERATURE_GAP.md), [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) |
edge trace: [`docs/EDGE_TRACE_USE.md`](docs/EDGE_TRACE_USE.md).

## Detailed findings (earlier benchmark: active run, 16 paired seeds; later studies marked)

| Finding | Result |
|---|---|
| Federation vs independent per-vehicle learning (no attackers) | **+12.7 pp** completion, CI [+10.4, +15.1] - supported |
| Sign-flip attack vs undefended FedAvg | -5.9 pp, CI [-10.1, -1.7]; raw p = 0.009, **Holm p = 0.061** (not established after correction) |
| Trust-aware aggregation vs FedAvg at 24 % attackers | +2.0 pp, CI [-2.0, +6.0] - **not established** (design resolves ~5-6 pp) |
| Oracle that removes exactly the attackers vs FedAvg | +5.2 pp, CI [+1.1, +9.3] - the headroom of *any* detector is small |
| Trust-aware vs FedAvg at 40 % attackers (6 seeds) | **+17.9 pp**, CI [+6.1, +29.7] |
| Trust-aware vs FedAvg with no attackers (6 seeds) | -9.9 pp, CI [-22.6, +2.8] - cost of false exclusions |
| Detector on real federated updates | TPR 0.991, **FPR 0.042**, growing from ~1 % to ~8 % of honest clients as training converges |
| Simple MAD-distance rule vs deployed detector (held-out) | equal (TPR 0.864 / FPR 0.008 vs 0.865 / 0.008) |
| Partial / norm-matched / intermittent attacks | TPR 0.53 / 0.73 / 0.77 - largely missed |
| Behavioural trust | adds nothing measurable; at default weights it cannot exclude a model-honest client |
| Learned policy vs Nearest-RSU heuristic | -0.4 pp, CI [-3.2, +2.4] - **not established**; depends on simulator semantics |
| Energy | local execution uses ~31x less energy per task |
| *Extended run (6 seeds):* data-plane attack on V2V helpers | costs a trust-blind V2V heuristic 2.9 pp; trust-aware peer selection recovers 2.2 pp (part of it not trust-related) - small |
| *Extended run:* learned policy under that attack | trust features +0.9 pp, CI [-1.4, +3.3] - not established |
| *Hypothesis study (24 seeds, Holm-corrected):* H2 / H3 / H4 / H5 | **supported with stated scope**: +6.1 pp at 24 % and +16.2 pp at 40 % attackers over FedAvg; joint trust beats behavioural-only (+5.3 / +19.7 pp); holds on synthetic and authentic NGSIM mobility; H1 only for completion, latency effect <= 3 ms ([`docs/HYPOTHESES.md`](docs/HYPOTHESES.md)) |
| *Method attempt (12 seeds):* do-no-harm exclusion | no-attacker cost -0.8 pp (deployed rule -6.4 pp); false exclusions 1.7 % vs 10.3 %; +17.6 pp at 40 % attackers; 24 % gain unresolved; norm-matched / partial attacks not detected |
| *Method attempt:* learner with delay-estimate inputs | +6.6 pp over Resource-Aware in the benchmark environment; parity or loss under hidden-state conditions; trust features add nothing |
| *Extended run:* Alibaba edge-trace calibration, peak hour | Nearest-RSU 16.8 %, Resource-Aware 59.9 %, learned policy = Local-Only (58.3 %) |

What must **not** be claimed: that the *original* joint trust rule improves completion at
the default attack ratio (the do-no-harm rule does, +6.1 pp vs FedAvg), that any detector has a
near-zero false-positive rate, that behavioural trust protects the policy, that the learned policy
beats heuristics outside the benchmark environment, that stealth (norm-matched) attackers are handled,
or that any result is a real-world measurement.

## Quick start

```bash
pip install -r requirements.txt
python run_simulation.py            # http://127.0.0.1:8000
```

The console has two tabs:

* **Simulation** - run / pause / resume / reset; a to-scale road viewport (RSU
  coverage intervals, vehicles coloured by detector trust, offloading links for
  events the backend reported, optional ground-truth overlay); live federated
  round status, client table with exclusion reasons, trust distribution, latency
  of completed tasks, event timeline.  *New-session* parameters (mobility source,
  vehicles, attack ratio, RSU positions) are separated from *live* parameters
  (aggregation strategy, alpha, tau, automatic rounds) and validated.
* **Research results** - the active run's baselines, paired differences,
  detector counts with Wilson intervals, every sweep, figures, data provenance
  and limitations, read from `/api/research/summary` (nothing hard-coded, no live
  numbers mixed in).

The simulation is advanced by one central loop; browser tabs only subscribe, so
extra tabs do not speed it up (verified by `tests/browser_verification.py`).

## Attacks, V2V and the edge trace

* Attackers poison / withhold model uploads **and, in the console and the
  extended suite, sabotage the V2V tasks they serve** (`adversarial.data_plane`;
  off in the published benchmark).  The 3D scene draws sabotaged V2V links.
* `network.v2v_peer_selection: trust_aware` and `mobility.vehicle_compute`
  (compute-rich helper vehicles, an assumption) make V2V meaningful.
* `edge_calibration` uses the Alibaba edge-platform trace (`Full_trace`, 41 GB, not
  committed) through an 8 KB summary built by `data/source/prepare_edge_trace.py`.
* The console has collapsible Controls / Monitor panels, a Focus mode and a freely
  navigable 3D camera (right-drag pan, WASD, Q/E, R to reset).
* Audit of all of this: [`docs/SYSTEM_AUDIT.md`](docs/SYSTEM_AUDIT.md); publication
  strategy: [`docs/PUBLICATION_PLAN.md`](docs/PUBLICATION_PLAN.md).

## Mobility sources

| Mode | Source | Status |
|---|---|---|
| `highway` | Intelligent Driver Model, 3.2 km loop | simulated |
| `trace` | calibrated synthetic NGSIM-style trace | synthetic, **not** authentic |
| `authentic_trace` | FHWA NGSIM US-101 replay (681 m, 2,847 vehicles) | measured **mobility only** |

NGSIM contains no tasks, channel measurements, rewards, model updates or attack
labels.  Provenance, licence notes and checksums: [`data/PROVENANCE.md`](data/PROVENANCE.md).

## Reproducing the results

```bash
python -m experiments.experiment_runner --label my_run       # full benchmark, ~2.5 h, 6 workers
VANNET_JOBS=4 python -m experiments.experiment_runner ...    # less RAM (each worker peaks ~0.8 GB)
python -m experiments.experiment_runner --resume results/run_<...>   # continue an interrupted run
python -m experiments.experiment_runner --quick              # smoke test, never published

python -m experiments.trust_audit results/benchmark_results.json results/trust_audit.json
python -m experiments.verify_statistics results/benchmark_results.json results/statistics_verification.json
python -m experiments.claims_report                          # verdict table
python -m experiments.detector_comparison                    # held-out detector comparison
python -m experiments.research_figures                       # figures 7-10
python -m experiments.extended_suite                         # data-plane / V2V / edge calibration (~40 min)
python -m experiments.supplementary_figures                  # figures 11-13
python -m experiments.export_evidence                        # gzipped raw results + SHA256SUMS
python -m pytest tests -q -p no:cacheprovider                # full test suite (run once)
python tests/browser_verification.py                         # real-browser check (server running)
```

Full guide and the list of artefacts: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository map

| Path | Content |
|---|---|
| `environment/` | mobility (IDM, trace replay), channel, RSU and vehicle queues, the offloading environment |
| `rl/` | dueling double DQN, replay buffer, reward |
| `federated/` | client (local training, attacks), server (aggregation strategies, shadow detectors) |
| `trust/` | behavioural trust, model-update trust, joint trust, alternative detectors |
| `attacks/` | sign flip, partial / norm-matched / intermittent, scaling, noise, free riding |
| `baselines/` | heuristics and robust aggregation baselines |
| `experiments/` | benchmark suite, statistics, audit, claims report, detector comparison, figures |
| `web/` | FastAPI backend, console (`index.html`, `sim_view.js`, `dashboard.js`, `research.js`) |
| `results/` | `manifest.json`, the active run, superseded runs (kept), claims report |
| `docs/` | environment spec, methodology, limitations, related work, scientific contribution, paper draft, reproducibility, acceptance report |
| `tests/` | unit, protocol, contract and browser tests |

## Run history (all retained, only one active)

`results/manifest.json` lists every run.  The active run is the one with
`published_as_active: true`.  Earlier runs (including the first full run of the
corrected environment, whose learned policies had not learned RSU coverage) are
marked `superseded` and compared in `results/claims_report.md` section 14.
Smoke and aborted runs are marked as such and must not be cited.

## Documentation

* [`docs/ENVIRONMENT_SPEC.md`](docs/ENVIRONMENT_SPEC.md) - what the simulator implements
* [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) - trust computation, threat model, protocol, statistics
* [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) - what the evidence does not establish
* [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) - verified literature matrix and positioning
* [`docs/SCIENTIFIC_CONTRIBUTION.md`](docs/SCIENTIFIC_CONTRIBUTION.md) - the contribution at the level the data support
* [`docs/PAPER.md`](docs/PAPER.md) - manuscript draft with results, validity threats and venue notes
* [`docs/ACCEPTANCE_REPORT.md`](docs/ACCEPTANCE_REPORT.md) - PASS / FAIL / BLOCKED per criterion with evidence

## Supplementary results (summary)

Edge-node fault, rotating fault and intermittent-connectivity experiments are in the Research tab and docs/LIMITATIONS.md. They do **not** show a benefit of behavioural trust features; under a rotating fault the learned policy is worse than the Resource-Aware heuristic.
