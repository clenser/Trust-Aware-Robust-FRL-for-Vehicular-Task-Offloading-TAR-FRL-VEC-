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

## What the evidence supports (active run, 16 paired seeds)

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

What must **not** be claimed: that trust-aware FL improves completion at the
default attack ratio, that the detector has a near-zero false-positive rate,
that behavioural trust protects the policy, that the learned policy beats simple
heuristics, or that any result is a real-world measurement.

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
