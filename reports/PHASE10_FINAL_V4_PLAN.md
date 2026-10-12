# Phase 10 Final v4 Experimental Evaluation Plan

**Status:** corrected protocol. Supersedes `PHASE10_FINAL_V3_PLAN.md` (kept as
historical). All v4 artifacts live under `results/phase10_final_v4/`,
`reports/figures/phase10_final_v4/`, `reports/tables/phase10_final_v4/`.

## 1. Fixed core proposed-method configuration

| Parameter | Core value | Source |
|---|---|---|
| Trust smoothing lambda | `0.70` | YAML `scenario.trust_lambda` |
| Combination weight alpha | `0.50` | YAML `federated.alpha` |
| Exclusion threshold tau_exc | `0.35` | YAML `federated.tau_exc` |
| Direction threshold m2 | `0.45` | YAML `detector.m2_dir_threshold` |
| Staleness threshold m3 | `0.50` | YAML `detector.m3_stale_threshold` |
| Originality threshold m4 | `0.00` | YAML `detector.m4_orig_threshold` |
| Composite anomaly tau | `0.50` | YAML `detector.tau_anomaly` |
| Federated rounds | `8` | YAML `federated.rounds` |
| Federated clients | `10` | YAML `federated.num_clients` |

**Precedence rule** (implemented in `resolve_detector_params`, tested): an
explicit per-run override (ablation/sensitivity `PlanItem` field) wins; else
the YAML section value; else the `CORE_DEFAULTS` table in
`scripts/run_phase10_final_evaluation.py`. Resolved *effective* values drive
the server, detection accounting, clients, evaluation resets, and CSV rows.
A planned value is never logged as effective unless applied; `runtime_...` /
`effective_...` columns must agree with the trust-table lambda in every run.

## 2. Matrix design (880 runs total; pre-execution assertion enforced)

**Matrix A — core (630 runs)**, seeds 0–9, strategies fedavg / trust_weighted /
trust_threshold: clean IID 30; honest Non-IID 30; sign-flip × {0.1,0.2,0.3,0.4}
120; Gaussian perturbation × {0.1,…,0.4} 120; zero-update × {0.1,…,0.4} 120;
copy-update × {0.1,…,0.4} 120; unreliable low/medium/high (fraction 0.20) 90.

**Matrix B — ablations (70 = 7 × 2 × 5)**, seeds 0–4, scenarios sign-flip f=0.20
and zero-update f=0.20:
1. `full_proposed` (alpha 0.50, tau 0.35, lambda 0.70, trust_threshold);
2. `no_behavioural_trust` (alpha 0.00, rest core);
3. `no_model_trust` (alpha 1.00, rest core);
4. `no_temporal_smoothing` (lambda 0.00, rest core);
5. `no_threshold_exclusion` (trust_threshold strategy, tau_exc 0.00, rest core —
   the strategy is NOT switched, to avoid confounding exclusion with weighting);
6. `fedavg_no_trust` (FedAvg);
7. `local_rl_no_fl` (independent local RL, no aggregation).

**Matrix C — sensitivity (180 = (6+5+4+3) × 2 × 5)**, seeds 0–4, scenarios
sign-flip f=0.20 and stale-replay f=0.20, one parameter swept at a time:
alpha {0.0,…,1.0}, tau_exc {0.20,…,0.60}, lambda {0.50,0.70,0.90,0.95},
m2_dir_threshold {0.40,0.50,0.60}.

## 3. Client resources

Clean IID: all clients CPU 2.0 GHz, bandwidth 10 MHz, mobility/task_load medium,
link_reliability 0.90. Honest Non-IID: CPU {1.0,1.2,1.5,1.8,2.0,2.2,2.4,2.6,2.8,3.0} GHz
and bandwidth {5.0,6.5,8.0,10.0,10.0,12.0,14.0,16.0,18.0,20.0} MHz per client ID,
all other conditions fixed to the IID baseline, so resource heterogeneity is the
isolated non-IID factor. A config-level test instantiates every client and
verifies the exact per-vehicle CPU/bandwidth mapping after resets.

## 4. Unreliable-execution semantics (local-experience corruption)

Faults are injected into the tasks executed through one unreliable *client's*
environment (its observed outcomes, rewards, and trust updates), regardless of
destination. This models corruption of that participant's local training
experience — it does NOT simulate destination-specific edge failures, and no
report claims that. Unreliable clients are labeled non-malicious
(`is_malicious=False`, disjoint from malicious IDs). Per task at most one
trust observation exists (retract-and-re-observe). Outcome precedence for
overlapping events: drop > fail > delay; counters record each sampled event
while `tasks_with_any_fault` counts the union.

## 5. Stale-replay provenance

Round 0 never replays (no older update exists; `attack_applied=False`). For
r ≥ 1 the source update's original `round_trained` is preserved and staleness
is `current_round − origin_round`. If a requested age has no matching update,
the earliest available prior update is used and the *actual* age is reported.
Origin metadata is never rewritten. The server stamps staleness from the same
genuine metadata the detector scores.

## 6. Statistics and reporting

Paired seed-matched comparisons (trust_threshold − fedavg) with mean
difference, Student-t 95% CI, n, t statistic, raw p, Cohen's dz; Holm–Bonferroni
*adjusted p-values* with step-down decisions; zero-variance cases report
null statistics with status `undefined_zero_variance` (never NaN-as-evidence,
never equivalence claims). Figures use 95% CI error bars computed from the
plotted cells; sensitivity is shown per scenario and parameter. The 8-round
horizon supports stability observations only — no convergence claims. Held-out
completion and training-time fault counts are labeled separately.

## 7. Reproduction

```powershell
.venv/Scripts/python.exe scripts/run_phase10_final_evaluation.py --outdir results/phase10_final_v4 --seeds-core 10 --seeds-sec 5
.venv/Scripts/python.exe scripts/analyze_phase10_results.py --results-dir results/phase10_final_v4 --out-json results/phase10_final_v4/phase10_summary_metrics.json
.venv/Scripts/python.exe scripts/plot_phase10_final_figures.py --results-dir results/phase10_final_v4 --out-figs reports/figures/phase10_final_v4 --out-tables reports/tables/phase10_final_v4
.venv/Scripts/python.exe scripts/verify_phase10_manifest.py --check
.venv/Scripts/python.exe -m pytest tests -q
```
