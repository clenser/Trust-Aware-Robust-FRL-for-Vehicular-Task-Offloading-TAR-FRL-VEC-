# Reproducibility guide

Everything a new researcher needs to reproduce the reported artefacts, and
nothing that cannot be executed from a clean checkout.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.14.3 (the run metadata records the exact interpreter string) |
| PyTorch | 2.11.0+cpu |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| Platform | Windows-11-10.0.26300-SP0 (the code is platform independent) |
| Node.js | v24.14.1 (only needed for the front-end bundle; no build step) |
| Playwright | Chromium, for the browser verification suite |

```bash
pip install -r requirements.txt
python -m pip install playwright && python -m playwright install chromium
```

The exact versions used for the published run are recorded in
`results/<run_id>/metadata.json` under `environment_versions`, and the
content fingerprint of every source file that determines the results is
recorded under `code_revision`.

---

## 2. Datasets, licences and checksums

| Artefact | Path | Licence | Recorded in |
|---|---|---|---|
| NGSIM US-101 raw export | `data/source/us101_raw.csv` | CC BY-SA 3.0 Unported, as listed in the USDOT Socrata catalogue (dataset `8ect-6jqj`; recorded 2026-10-21 in `data/PROVENANCE.md`) - attribution to USDOT/FHWA required, derived files share the licence | `data/source/us101_raw.sha256` |
| Processed metric trace | `data/traces/ngsim_us101_authentic.csv` | derived from the above (CC BY-SA 3.0 share-alike applies) | `*.provenance.json` |
| Synthetic NGSIM-style trace | `data/synthetic_traces/synthetic_ngsim_highway.csv` | CC BY-SA 3.0 (project) | run metadata |
| Task profile catalogue | `data/synthetic_traces/synthetic_v2x_task_profiles.csv` | CC BY-SA 3.0 (project) | run metadata |
| Synthetic Manhattan grid | `data/synthetic_traces/synthetic_manhattan_grid.csv` | CC BY-SA 3.0 (project) | - |

Every run records the **SHA-256 of the files it actually loaded**, in
`metadata.json` under `dataset_provenance`.  The verification script
`data/source/verification_record.json` records an independent byte-for-byte
comparison against a second download of the source file; it is a
*reproducibility* fingerprint, **not** an authenticity proof.

```bash
python data/source/verification_record.json   # (read-only; see data/PROVENANCE.md)
python data/source/prepare_ngsim.py            # rebuild the processed trace
```

---

## 3. Effective configuration and seeds

| Setting | Value | Source |
|---|---|---|
| Environment model | version 2.1: RSU choice honoured, OFDMA uplink contention (100 MHz cell, 20 MHz per UE), `lambda_trust = 0`, coverage-relative state (20 features) | `environment_model`, `reward` in `configs/default_config.yaml` |
| Trust weight alpha / threshold tau | 0.50 / 0.40 (selected on validation data) | `trust_management` |
| Participation rule | `exclude_non_participants: true` | `trust_management` |
| Training | 30 rounds x 20 steps, 16 local gradient steps per round (480 per client) | `federated_learning` |
| Evaluation | 5 episodes x 30 steps per seed, seed offset 500000, inert | `EvaluationConfig` |
| Attack | ratio 0.24, `sign_flip`, strength 1.5, probability 1.0 | `adversarial` |
| Main-comparison seeds (16) | 42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515 | `DEFAULT_SEEDS` |
| Sweep seeds (6) | the first six of the above | `run_experiment_suite` |
| Detector comparison | 20 held-out seed bases 9001-9020 | `experiments/detector_comparison.py` |
| Trust-validation seeds | validation 1, held-out 9001 | `experiments/calibrate_trust.py` |

The **resolved** configuration of a run is written to
`results/<run_id>/effective_config.json`; the same values are embedded in every
per-seed record (`effective_config`), and the run metadata records the training
protocol, the environment model and the limitations that apply to the run.

---

## 4. Which artefacts are active and which are historical

`results/manifest.json` is the index.  Each entry carries `run_id`,
`directory`, `num_seeds`, `seeds`, `validation_status`,
`published_as_active` and (for retired runs) `superseded` /
`superseded_reason`.

| Rule | Meaning |
|---|---|
| `published_as_active: true` and not `superseded` | the **only** published result set |
| `superseded: true` | preserved for the record; **must not be cited** |
| `validation_status: "aborted"` | an aborted invocation; no artefacts |

`results/benchmark_results.json` (the active pointer) is byte-identical to
the active run's own `benchmark_results.json`; this is asserted by
`tests/test_plot_artifacts.py`.

---

## 5. Reproducing each reported artefact

### 5.1 Full benchmark - required only if the *method or the protocol* changed

```bash
python -m experiments.experiment_runner --label <label>          # 6 worker processes by default
VANNET_JOBS=4 python -m experiments.experiment_runner --label <label>   # fewer workers = less RAM
python -m experiments.experiment_runner --resume results/run_<timestamp>_<label>   # continue an interrupted run
python -m experiments.experiment_runner --quick                  # smoke configuration, never published
```

* Runtime: about 2.5 hours with 6 workers (each worker peaks at about 0.8 GB of
  RAM); a smoke run takes a few minutes.
* Every finished experiment cell is checkpointed to
  `results/run_<...>/partial_results.json`; `--resume` reloads finished cells and
  never recomputes them.  Cells identical to a main-comparison cell (same method,
  configuration and seeds) are reused rather than re-run.
* Writes `results/run_<timestamp>_<label>/` and, **only if the run passes
  validation and is a full run**, replaces the active pointer and updates the
  manifest (the previous active run is marked superseded, never deleted).
* Regenerates the six publication figures and their manifest.
* **Not** needed to regenerate the tables or the figures from an existing run.

### 5.2 Figures and summary tables from the retained raw records - minutes

```bash
python -m experiments.plotter \
    results/benchmark_results.json plots \
    --meta results/benchmark_results_metadata.json --clean
```

Recomputes every statistic from the per-seed records, cross-checks them
against the published summary, removes stale figures, and writes
`plots/figures_manifest.json`.

### 5.3 Detection-metric audit - seconds

```bash
python -m experiments.trust_audit results/benchmark_results.json results/trust_audit.json
```

Recomputes every reported detection rate from the raw TP/FP/FN/TN counts and
prints the per-scenario / per-seed table.

### 5.4 Statistical verification - seconds

```bash
python -m experiments.verify_statistics \
    results/benchmark_results.json results/statistics_verification.json
```

Recomputes the aggregates, confidence intervals, paired differences, paired
*t* tests and Cohen's *d* for every metric from the per-seed records, derives
the deadline-violation counts from the raw evaluation episodes, and checks
that every method saw the same episodes, tasks, resources and budgets with an
inert evaluation.  Exit code 0 means no disagreement.

### 5.5 Claims report and detector evaluation table - seconds

```bash
python -m experiments.claims_report
python -m experiments.detector_comparison        # held-out detector comparison, ~1 minute
```

`claims_report` writes `results/claims_report.md`: the eight pre-specified
downstream claims with paired confidence intervals, Holm-adjusted p-values,
minimum detectable effects and mechanical verdicts; per-method tables; the
detector tables; every sweep; the effect of the state-representation change;
and the list of unsupported claims.

### 5.6 Trust-detector calibration and held-out assessment - ~1 minute

```bash
python -m experiments.calibrate_trust results/trust_calibration.json
```

Selects the threshold on the validation suite and evaluates it once on the
held-out suite.

### 5.7 Detector study (participation comparison, threshold sensitivity,
heterogeneity and participation sweeps) - about 15 minutes

```bash
python -m experiments.detector_study --out results/detector_study.json \
    --md results/detector_study.md
```

### 5.8 Tests

```bash
python -m pytest tests -q -p no:cacheprovider    # full suite
node tests/frontend_behaviour.js && node tests/frontend_resilience.js   # headless front-end harness
```

Run the full suite once, not in a loop; it re-runs simulations.

### 5.9 Browser and live verification

```bash
python run_simulation.py                  # leave running (http://127.0.0.1:8000)
python tests/browser_verification.py      # real Chromium via Playwright; screenshots in ./browser_check
python tests/live/verify_live_workflow.py
python tests/live/verify_live_lifecycle.py
python tests/live/verify_live_autofl.py
python tests/live/verify_live_authentic_trace.py
```

`browser_verification.py` drives the real page: load without console errors,
start / pause / resume / reset, the step counter, **two tabs not doubling the
simulation speed**, backend-derived viewport geometry, parameter validation
(client and backend), a new session with a different vehicle count, the
authentic-trace replay, the Research tab, layout at four widths, and the theme
toggle.

---

## 6. What does *not* need the full benchmark

| Question | Artefact | Command |
|---|---|---|
| Are the published metrics reproducible? | `results/statistics_verification.json` | 5.4 |
| Is every detection rate consistent with its raw counts? | `results/trust_audit.json` | 5.3 |
| Can the figures be regenerated? | `plots/figures_manifest.json` | 5.2 |
| What are the claims, verdicts and their limits? | `results/claims_report.md` | 5.5 |
| How does the detector compare with alternatives on held-out data? | `results/detector_comparison.md` | 5.5 |
| Which threshold is selected, and does it hold out? | `results/trust_calibration.json` | 5.6 |

Only a change to the **method** (trust computation, reward, workload,
participation protocol, environment model, state representation) invalidates the
stored benchmark; a change to the *reporting* does not.


---

## 7. Supplementary run and console performance

```bash
python -m experiments.supplementary_suite                 # ~1-2 h; VANNET_JOBS=3 for low memory
python -m experiments.supplementary_suite --quick         # smoke test, never published
python -m experiments.supplementary_suite --resume results/run_<...>
```

The supplementary run is listed in `results/manifest.json` with
`kind: supplementary` and `supplements: <active run id>`; the Research tab shows
it under the active run.  The console serves versioned static assets with a
one-hour cache lifetime, the research summary with an ETag (304 when unchanged)
and gzip, figures with a revalidating cache header, and never caches live state.
The telemetry stream sends the federated block (rounds, client table) only when
it changes.
