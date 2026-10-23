"""Edge-side calibration from the Alibaba edge-platform trace (EdgeWorkloadsTraces).

What the data is, and is not
----------------------------
The trace (Xu et al., IMC 2021) describes a commercial edge platform: the
compute capacity of 149 sites, the VMs that tenants rent on them, the
inter-site RTT / loss, and per-VM bandwidth over three months.  It contains no
vehicles, no tasks, no V2X radio and no attacks.  It is therefore used ONLY to
make the RSU / backhaul side of the simulator empirical:

* co-tenant background load: an RSU is a shared edge site, so a fraction of its
  CPU is occupied by other tenants, following the measured diurnal bandwidth
  profile (``normalised_load_by_hour``) scaled to the measured VM core
  allocation ratio;
* capacity heterogeneity between sites (cores per site);
* the delay of an RSU-to-RSU hand-over, drawn from the measured RTT
  distribution instead of a constant.

The summary json is produced by ``data/source/prepare_edge_trace.py``; the 41 GB
of raw CSV are not redistributed.  The module is inert unless
``edge_calibration.enabled`` is true, so every earlier result is unchanged.
"""

import json
import os

import numpy as np


class EdgeCalibration:
    def __init__(self, summary, peak_background_share=None, start_hour=12.0,
                 day_compression=288.0, capacity_heterogeneity=True,
                 backhaul_rtt=True, seed=0, load_model="bandwidth_profile",
                 tenant_alloc_ratio=None):
        self.summary = summary
        if load_model not in ("bandwidth_profile", "cpu_trace"):
            raise ValueError("load_model must be 'bandwidth_profile' or 'cpu_trace'")
        self.load_model = load_model
        bw = summary["bandwidth_three_months"]
        self.hourly = np.asarray(bw["normalised_load_by_hour"], dtype=float)
        if peak_background_share is None:
            peak_background_share = summary["sites"]["vm_core_allocation_ratio"]["p75"]
        self.peak_share = float(np.clip(peak_background_share, 0.0, 0.9))
        self.start_hour = float(start_hour)
        self.day_compression = float(day_compression)
        self.capacity_heterogeneity = bool(capacity_heterogeneity)
        self.backhaul_rtt = bool(backhaul_rtt)
        self._rtt = np.asarray(summary["rtt"]["rtt_ms_grid_p01_p99"], dtype=float)
        self._loss = np.asarray(summary["rtt"]["loss_percent_grid_p01_p99"], dtype=float)
        self._cores = summary["sites"]["cores_per_site"]
        self.rng = np.random.RandomState(int(seed) + 31)
        if load_model == "cpu_trace":
            cpu = summary["vm_cpu"]          # needs prepare_edge_trace.py --only-vm-cpu
            self._cpu_mean = float(cpu["cpu_rate"]["mean"])
            hourly = np.asarray(cpu["mean_cpu_by_hour_utc8"], dtype=float)
            self._cpu_hourly = hourly / hourly.mean()   # mean-one diurnal shape
            self._phi2 = float(cpu["ar1_phi_per_2min"])
            self._sd = float(cpu["within_vm_sd"])
            if tenant_alloc_ratio is None:
                tenant_alloc_ratio = summary["sites"]["vm_core_allocation_ratio"]["p50"]
            self._alloc = float(np.clip(tenant_alloc_ratio, 0.0, 1.0))
            self._noise_rng = np.random.RandomState(int(seed) + 47)   # keeps self.rng's stream unchanged
            self._z, self._z_t = {}, {}

    @classmethod
    def from_config(cls, config, seed=0):
        cfg = (config or {}).get("edge_calibration") or {}
        if not cfg.get("enabled", False):
            return None
        path = cfg.get("summary_path", "data/traces/edge_platform_summary.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "edge_calibration.enabled is true but %s is missing; "
                "run data/source/prepare_edge_trace.py" % path)
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)
        return cls(summary,
                   peak_background_share=cfg.get("peak_background_share"),
                   start_hour=float(cfg.get("start_hour", 12.0)),
                   day_compression=float(cfg.get("day_compression", 288.0)),
                   capacity_heterogeneity=bool(cfg.get("capacity_heterogeneity", True)),
                   backhaul_rtt=bool(cfg.get("backhaul_rtt", True)),
                   seed=seed,
                   load_model=cfg.get("load_model", "bandwidth_profile"),
                   tenant_alloc_ratio=cfg.get("tenant_alloc_ratio"))

    # ------------------------------------------------------------------
    def hour_of_day(self, sim_time_sec):
        return (self.start_hour + sim_time_sec * self.day_compression / 3600.0) % 24.0

    def load_profile(self, hour):
        """Normalised (0..1) co-tenant load at an hour of the day (linear interpolation)."""
        h0 = int(np.floor(hour)) % 24
        h1 = (h0 + 1) % 24
        f = hour - np.floor(hour)
        return float((1.0 - f) * self.hourly[h0] + f * self.hourly[h1])

    def _cpu_trace_share(self, sim_time_sec, site_factor, idx):
        """Measured-utilisation co-tenant share: allocated-core share x VM CPU utilisation.

        Utilisation follows the trace's mean diurnal shape plus a per-RSU AR(1)
        deviation with the persistence (phi per 2 min) and within-VM spread
        measured in VM_CPU.csv; phi is rescaled to the simulation step.
        """
        h = self.hour_of_day(sim_time_sec)
        h0, f = int(np.floor(h)) % 24, h - np.floor(h)
        shape = (1.0 - f) * self._cpu_hourly[h0] + f * self._cpu_hourly[(h0 + 1) % 24]
        last = self._z_t.get(idx)
        if last is None:
            z = self._sd * self._noise_rng.randn()
        elif sim_time_sec > last:
            real_sec = (sim_time_sec - last) * self.day_compression
            phi = self._phi2 ** (real_sec / 120.0)
            z = phi * self._z[idx] + np.sqrt(max(0.0, 1.0 - phi * phi)) * self._sd * self._noise_rng.randn()
        else:
            z = self._z[idx]
        self._z[idx], self._z_t[idx] = z, sim_time_sec
        util = np.clip(self._cpu_mean * shape * site_factor + z, 0.0, 1.0)
        return float(np.clip(self._alloc * util, 0.0, 0.9))

    def background_share(self, sim_time_sec, site_factor=1.0, idx=0):
        """Fraction of an RSU's CPU taken by co-tenants at ``sim_time_sec``."""
        if self.load_model == "cpu_trace":
            return self._cpu_trace_share(sim_time_sec, site_factor, idx)
        share = self.peak_share * self.load_profile(self.hour_of_day(sim_time_sec)) * site_factor
        return float(np.clip(share, 0.0, 0.9))

    def site_factors(self, n):
        """Per-RSU multiplicative factors for background load (clipped, seeded)."""
        if not self.capacity_heterogeneity:
            return [1.0] * n
        return [float(np.clip(self.rng.lognormal(0.0, 0.25), 0.6, 1.5)) for _ in range(n)]

    def capacity_scale(self, n):
        """Per-RSU CPU scale drawn from the measured cores-per-site distribution.

        A site's capacity relative to the median site is sampled from the
        measured quantiles (p05..p95) and damped (power 0.35, clipped to
        0.6-1.4): the measured spread is 10x, far wider than the three
        similar roadside units of the highway deployment.
        """
        if not self.capacity_heterogeneity:
            return [1.0] * n
        qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        vals = np.array([self._cores[k] for k in ("p05", "p25", "p50", "p75", "p95")], dtype=float)
        vals = vals / self._cores["p50"]
        out = []
        for _ in range(n):
            q = self.rng.uniform(0.05, 0.95)
            out.append(float(np.clip(np.interp(q, qs, vals) ** 0.35, 0.6, 1.4)))
        return out

    def handover_delay_sec(self):
        """One-way RSU-to-RSU hand-over delay: half of a measured inter-site RTT."""
        if not self.backhaul_rtt:
            return 0.025
        return float(self.rng.choice(self._rtt)) / 2000.0

    def handover_lost(self):
        """True when the hand-over packet is lost, with the measured loss distribution."""
        if not self.backhaul_rtt:
            return False
        return bool(self.rng.rand() < float(self.rng.choice(self._loss)) / 100.0)
