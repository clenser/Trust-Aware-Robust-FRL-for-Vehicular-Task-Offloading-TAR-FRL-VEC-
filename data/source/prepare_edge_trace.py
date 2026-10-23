"""Reduce the Alibaba edge-platform trace (EdgeWorkloadsTraces, ~41 GB of CSV)
to a small, committed calibration summary.

Source: Xu et al., "From Cloud to Edge: A First Look at Public Edge Platforms",
ACM IMC 2021 (NEP platform).  The raw files are NOT redistributed here.

Files read (all optional except PM.csv):
  PM.csv                  physical machines per edge site (cores, memory)
  VM.csv                  customer VMs per site (allocated cores)
  SITE_RTT.csv            inter-site ICMP RTT / loss        (36 GB, block-sampled)
  VM_BW_THREE_MONTHS.csv  per-VM bandwidth, 3 months        (0.45 GB, full pass)
  VM_BANDWIDTH.csv        per-VM bandwidth, 5-min snapshots (6.5 GB, block-sampled)

Large files are sampled by seeking to evenly spread byte offsets and parsing
whole lines inside fixed-size blocks, so the pass takes minutes, not hours,
and the sampling is deterministic.  The summary records exactly how each file
was read, so the numbers are not presented as a full-population census.

Usage: python data/source/prepare_edge_trace.py [trace_dir] [out_json]
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

BLOCK_BYTES = 16 * 1024 * 1024
N_BLOCKS = 40
Q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def _quantiles(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {f"p{int(q * 100):02d}": float(np.quantile(x, q)) for q in Q} | {
        "mean": float(x.mean()), "n": int(x.size)}


def _block_sample(path, parse, n_blocks=N_BLOCKS, block=BLOCK_BYTES):
    """Yield DataFrames parsed from evenly spread, line-aligned byte blocks."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        header = fh.readline().decode("utf-8", "replace").strip()
        start = fh.tell()
        offsets = np.linspace(start, max(start, size - block), n_blocks).astype(np.int64)
        for off in offsets:
            fh.seek(int(off))
            if off > start:
                fh.readline()  # discard the partial line
            raw = fh.read(block)
            raw = raw[: raw.rfind(b"\n") + 1]
            yield parse(header, raw), int(off)


def _read_lines(header, raw):
    import io
    return pd.read_csv(io.BytesIO(header.encode() + b"\n" + raw), low_memory=False)


def summarise_sites(trace_dir):
    pm = pd.read_csv(os.path.join(trace_dir, "PM.csv"))
    per_site = pm.groupby("site_id").agg(
        pms=("pm_name", "count"), cores=("cores", "sum"),
        memory_mb=("memory", "sum")).reset_index()
    out = {
        "num_sites": int(per_site.shape[0]),
        "num_pms": int(pm.shape[0]),
        "cores_per_pm": _quantiles(pm["cores"]),
        "cores_per_site": _quantiles(per_site["cores"]),
        "pms_per_site": _quantiles(per_site["pms"]),
        "memory_gb_per_pm": _quantiles(pm["memory"] / 1024.0),
    }
    vm_path = os.path.join(trace_dir, "VM.csv")
    if os.path.exists(vm_path):
        vm = pd.read_csv(vm_path, usecols=["site_id", "cores"])
        alloc = vm.groupby("site_id")["cores"].sum()
        per_site = per_site.set_index("site_id").join(alloc.rename("vm_cores")).fillna(0)
        ratio = per_site["vm_cores"] / per_site["cores"].clip(lower=1)
        out["vm_core_allocation_ratio"] = _quantiles(ratio)
        out["num_vms"] = int(vm.shape[0])
        out["vm_cores"] = _quantiles(vm["cores"])
    return out


def summarise_rtt(path):
    rtts, losses, pair_med = [], [], {}
    offsets = []
    for df, off in _block_sample(path, _read_lines):
        offsets.append(off)
        df = df[df["type"] == "icmp"]
        rtts.append(df["rtt"].to_numpy(float))
        losses.append(df["loss"].to_numpy(float))
        for (a, b), g in df.groupby(["from_site_id", "to_site_id"]):
            pair_med.setdefault((a, b), []).append(float(g["rtt"].median()))
    rtt = np.concatenate(rtts)
    loss = np.concatenate(losses)
    ok = (rtt > 0)
    pair_vals = np.array([np.median(v) for v in pair_med.values()])
    loss_pct = loss[ok]
    return {
        "method": f"{len(offsets)} line-aligned blocks of {BLOCK_BYTES >> 20} MiB at evenly spread offsets",
        "rtt_ms": _quantiles(rtt[ok]),
        "loss_percent": _quantiles(loss_pct),
        "fraction_samples_with_loss": float((loss_pct > 0).mean()),
        "site_pairs_observed": int(len(pair_med)),
        "pair_median_rtt_ms": _quantiles(pair_vals),
        # Empirical RTT sample (deciles of a fine grid) so the simulator can
        # draw from the measured distribution rather than a fitted one.
        "rtt_ms_grid_p01_p99": [float(np.quantile(rtt[ok], q)) for q in np.linspace(0.01, 0.99, 99)],
        "loss_percent_grid_p01_p99": [float(np.quantile(loss_pct, q)) for q in np.linspace(0.01, 0.99, 99)],
    }


def summarise_bw_three_months(path):
    hour_sum = np.zeros(24)
    hour_cnt = np.zeros(24)
    dow_sum = np.zeros(7)
    dow_cnt = np.zeros(7)
    per_vm_mean = {}
    for chunk in pd.read_csv(path, chunksize=2_000_000):
        ts = pd.to_datetime(chunk["report_time"], errors="coerce")
        bw = (chunk["pub_down_bw"] + chunk["pub_up_bw"]) * 8e-6  # bytes/s -> Mbit/s
        m = ts.notna() & bw.notna() & np.isfinite(bw)
        h = ts.dt.hour[m].to_numpy()
        d = ts.dt.dayofweek[m].to_numpy()
        b = bw[m].to_numpy()
        hour_sum += np.bincount(h, weights=b, minlength=24)
        hour_cnt += np.bincount(h, minlength=24)
        dow_sum += np.bincount(d, weights=b, minlength=7)
        dow_cnt += np.bincount(d, minlength=7)
        g = chunk.assign(bw=bw)[m].groupby("vm_id")["bw"].agg(["sum", "count"])
        for vm, (s, c) in zip(g.index, g.to_numpy()):
            a = per_vm_mean.get(vm)
            per_vm_mean[vm] = (s, c) if a is None else (a[0] + s, a[1] + c)
    hourly = hour_sum / np.maximum(hour_cnt, 1)
    daily = dow_sum / np.maximum(dow_cnt, 1)
    vm_means = np.array([s / c for s, c in per_vm_mean.values()])
    return {
        "method": "full single pass",
        "rows_per_hour_bin": hour_cnt.astype(int).tolist(),
        "mean_bw_mbit_by_hour": hourly.tolist(),
        "normalised_load_by_hour": (hourly / hourly.max()).tolist(),
        "normalised_load_by_weekday": (daily / daily.max()).tolist(),
        "vm_mean_bw_mbit": _quantiles(vm_means),
        "num_vms_observed": int(len(vm_means)),
        "peak_to_mean_hourly": float(hourly.max() / hourly.mean()),
    }


def summarise_vm_bandwidth(path):
    ratios, per_site = [], {}
    for df, _ in _block_sample(path, _read_lines, n_blocks=24, block=8 * 1024 * 1024):
        total = (df["pub_down_bw"] + df["pub_up_bw"]) * 8e-6
        ratios.append(total.to_numpy())
        for s, g in total.groupby(df["site_id"]):
            per_site.setdefault(s, []).append(float(g.mean()))
    allbw = np.concatenate(ratios)
    site_means = np.array([np.mean(v) for v in per_site.values()])
    return {
        "method": "24 line-aligned blocks of 8 MiB at evenly spread offsets",
        "vm_bw_mbit": _quantiles(allbw),
        "site_mean_bw_mbit": _quantiles(site_means),
        "sites_observed": int(len(per_site)),
        # Spread of mean load across sites: drives per-RSU heterogeneity.
        "site_load_cv": float(site_means.std() / max(site_means.mean(), 1e-9)),
    }


def file_fingerprint(path, head_bytes=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(head_bytes))
    return {"bytes": os.path.getsize(path), "sha256_first_8MiB": h.hexdigest()}


def main():
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "Full_trace"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/traces/edge_platform_summary.json"
    files = {n: os.path.join(trace_dir, n) for n in
             ("PM.csv", "VM.csv", "SITE_RTT.csv", "VM_BW_THREE_MONTHS.csv", "VM_BANDWIDTH.csv")}
    summary = {
        "source": "Alibaba NEP edge platform (EdgeWorkloadsTraces); Xu et al., IMC 2021",
        "scope": ("Edge-platform compute capacity, inter-site RTT/loss and VM bandwidth. "
                  "Contains NO vehicular tasks, mobility, V2X channel or attack labels; "
                  "used only to calibrate the RSU/edge side of the simulator."),
        "files": {n: file_fingerprint(p) for n, p in files.items() if os.path.exists(p)},
    }
    summary["sites"] = summarise_sites(trace_dir)
    for key, name, fn in (("rtt", "SITE_RTT.csv", summarise_rtt),
                          ("bandwidth_three_months", "VM_BW_THREE_MONTHS.csv", summarise_bw_three_months),
                          ("bandwidth_snapshots", "VM_BANDWIDTH.csv", summarise_vm_bandwidth)):
        if os.path.exists(files[name]):
            print("reading", name, flush=True)
            summary[key] = fn(files[name])
    summary["prepared_by"] = "VANNET / Sreemanta Barman"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
