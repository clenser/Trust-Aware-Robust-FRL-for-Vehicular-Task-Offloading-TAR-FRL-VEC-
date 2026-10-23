# Using the Alibaba edge trace (EdgeWorkloadsTraces) - 23 Oct 2026

The trace describes a commercial edge platform (149 sites, ~7,400 tenant VMs).  It
has no vehicles, tasks, V2X radio or attacks, so it only calibrates the RSU /
backhaul side (see `environment/edge_calibration.py`).  `data/source/prepare_edge_trace.py`
reduces it to `data/traces/edge_platform_summary.json`; the raw files (`D:\Full_trace`,
~63 GB) are never committed.

## What was already used
`PM.csv` and `VM.csv` (capacity, VM core allocation), `SITE_RTT.csv` (backhaul RTT/loss),
`VM_BW_THREE_MONTHS.csv` and `VM_BANDWIDTH.csv` (diurnal bandwidth profile).

## New: `VM_CPU.csv` (18 GB, block-sampled, 40 x 8 MiB, 17,809 VM series, 5.1 M samples)
| Quantity | Measured |
|---|---|
| mean / median VM CPU utilisation | 8.0 % / 2 % |
| 95th percentile | 38 % |
| within-VM lag-1 autocorrelation (2 min) | 0.82 (0.78 at 10 min, 0.71 at 30 min, 0.63 at 1 h) |
| within-VM sd | 0.043 |
| steps with a jump > 0.2 | 0.26 % |
| peak-to-mean hourly (UTC+8) | 1.35 (bandwidth profile used before: 1.83) |

**Finding about the earlier calibration.**  It scaled the load to the VM core
*allocation* ratio (p75 = 0.64), which is reserved capacity, not CPU use.  Tenants use
~8 % of what they hold, so a measured-utilisation share is ~0.5 x 0.08 = 4 % of an RSU,
about an order of magnitude below the allocation view.  Both are now available:

```yaml
edge_calibration:
  enabled: true
  load_model: "bandwidth_profile"   # default, unchanged: allocation view (upper bound)
  # load_model: "cpu_trace"         # allocation ratio x measured utilisation (diurnal + AR(1))
```

Default behaviour is unchanged (tested).  Limits: the VM series are per-VM, not
per-site aggregates; the model uses the within-VM spread for one aggregate stream and
the all-site mean utilisation, so it is a plausible stand-in for site load, not a
reconstruction.  The file is not time-ordered and sampling is by byte block.

## Study `experiments/edge_load_study.py` (heuristics only, 12 paired seeds)
Pre-registered: EL1 - the Resource-Aware minus Nearest-RSU gap is larger under the
allocation-view load than under cpu_trace; EL2 - under cpu_trace completion equals the
no-calibration run.

| Setting | Nearest-RSU | Resource-Aware | RA - NR (pp, 95 % CI) |
|---|---|---|---|
| no calibration | 74.8 % | 71.7 % | -3.0 [-3.7, -2.3] |
| allocation view, 05:00 | 62.5 % | 70.2 % | +7.7 [+4.1, +11.3] |
| allocation view, 20:00 | 16.5 % | 60.0 % | +43.4 [+40.0, +46.9] |
| cpu_trace, 05:00 | 69.0 % | 70.9 % | +1.9 [-1.0, +4.8] |
| cpu_trace, 20:00 | 67.7 % | 70.8 % | +3.1 [+0.04, +6.1] |

* **EL1 supported.**  Who wins among load-blind and load-aware heuristics depends
  almost entirely on the load model: 43 pp apart at the evening peak under the
  allocation view, 3 pp under measured utilisation.  Any "learned policy beats
  Resource-Aware under load" claim must say which load model it used.
* **EL2 not supported as worded** (attribution resolved by the ablation below).  Resource-Aware under cpu_trace is within noise of
  no calibration (-0.9 / -1.0 pp, CIs include 0), but Nearest-RSU still loses 5.8 / 7.1
  pp.  The calibration switch also applies capacity heterogeneity and measured
  hand-over delay, so this study cannot attribute that loss to co-tenant load alone;
  an ablation of those two components is not yet run.
* Nothing here involves the learned policy or the attacks; it only fixes how hard
  the edge side is.  Raw results: `results/edge_load_study/` (git-ignored like other
  results; regenerate with `python -m experiments.edge_load_study`).

## Ablation (20:00, 12 paired seeds; `--ablation`)
Effect on Nearest-RSU completion versus no calibration:

| Component switched on alone | Nearest-RSU change (pp, 95 % CI) | Resource-Aware change |
|---|---|---|
| hand-over RTT only | 0.00 [-0.01, +0.01] | 0.00 |
| capacity spread only | -4.1 [-7.5, -0.7] | -0.7 [-1.8, +0.4] |
| measured CPU load only | -2.8 [-3.0, -2.6] | -0.2 [-0.26, -0.17] |
| allocation-view load only | -62.4 [-62.7, -62.0] | -12.8 [-13.6, -12.1] |

The 6-7 pp Nearest-RSU loss under `cpu_trace` is capacity spread (about 4 pp) plus
measured load (about 3 pp).  The hand-over RTT has no effect on completion in this
protocol.  Load-aware routing is robust to every realistic component (< 1 pp); the
allocation-view load is the only setting that breaks a heuristic badly.  With
`cpu_trace` load alone the Resource-Aware vs Nearest-RSU gap is -0.45 pp [-1.1, +0.2]:
indistinguishable.
