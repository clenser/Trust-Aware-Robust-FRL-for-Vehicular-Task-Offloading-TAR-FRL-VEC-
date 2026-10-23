"""
3GPP-style V2X wireless communication and channel model.

Stochastic model
----------------
Each *link interaction* is described by exactly one
:class:`ChannelObservation`.  The observation is drawn once and then used for

* the achievable data rate and the resulting transmission delay, and
* the SINR that is recorded in the behavioural trust update.

Drawing the channel twice (once for the transmission, once for the trust
update) would describe two different radio realizations for a single
transmission, so the observation is shared explicitly.

Path loss
---------
The path-loss expressions are the widely used *log-distance* forms that the
3GPP V2X / TR 36.885 studies report for highway deployments:

    PL_LOS  [dB] = 32.4 + 20*log10(f_c/1GHz) + 20*log10(d/1m)
    PL_NLOS [dB] = 38.3 + 17.5*log10(d/1m) + 24.9*log10(f_c/1GHz)

They are *approximations* of the 3GPP evaluation methodology, not a
conformance claim: the table is a log-distance fit, the shadowing is a
zero-mean Gaussian with a 3 dB standard deviation, and the small-scale
fading is an independent unit-mean exponential (Rayleigh power) gain.  No
correlated fading, no antenna pattern and no interference are modelled, and
the achievable rate is additionally clipped to the 0.5-54 Mbps range that
5G NR V2X / 802.11bd practical MCS limits impose.

Computation energy
------------------
The dynamic energy model is the standard "scaled capacitance" form

    E = kappa * f^2 * N          [J]

with ``N`` the number of executed cycles and ``f`` the operating frequency
in Hz.  ``kappa`` has units of J*s^2/cycle.  The value used here is
``1e-30``, which corresponds to **1 pJ per cycle at 1 GHz** - the standard
figure of a ~1 mW processor running at 1 GHz (1 mW / 1e9 cycles/s).  The
previous value of ``1e-27`` produced 1000 pJ per cycle at 1 GHz, i.e. three
orders of magnitude above the value stated in the code's own comment and
physically implausible (0.22 Gcycles on a 4 GHz core would have cost 3.5 J).
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class ChannelObservation:
    """One radio realization of a single link interaction.

    All quantities describe the *same* sample, so the transmission decision
    and the trust observation can never disagree about the link.
    """
    distance_m: float
    is_los: bool
    path_loss_db: float
    shadowing_db: float
    small_scale_fading: float   # linear power gain, unit mean
    channel_gain_linear: float  # 10^(-PL/10) including shadowing
    rx_power_w: float
    noise_power_w: float
    interference_w: float
    sinr_linear: float
    sinr_db: float
    data_rate_mbps: float

    def to_dict(self):
        return asdict(self)


class V2XChannel:
    # Energy per cycle at 1 GHz for the reference kappa (1 pJ/cycle).
    KAPPA_J_S2_PER_CYCLE = 1e-30

    def __init__(self, carrier_freq_ghz=5.9, bandwidth_mhz=20.0, tx_power_dbm=23.0,
                 noise_power_dbm=-95.0, kappa=None):
        self.fc = carrier_freq_ghz  # 5.9 GHz ITS band
        self.bandwidth = bandwidth_mhz * 1e6  # Hz
        self.tx_power = 10 ** ((tx_power_dbm - 30) / 10)  # Watts (~0.2 W)
        self.noise_power = 10 ** ((noise_power_dbm - 30) / 10)  # Watts
        # Effective switched-capacitance factor, 1 pJ/cycle at 1 GHz.
        self.kappa = float(kappa) if kappa is not None else self.KAPPA_J_S2_PER_CYCLE
        self.tx_power_dbm = tx_power_dbm
        self.noise_power_dbm = noise_power_dbm
        self.bandwidth_mhz = bandwidth_mhz
        # Practical MCS clip for 5G NR V2X / 802.11bd on a 20 MHz channel.
        self.min_rate_mbps = 0.5
        self.max_rate_mbps = 54.0
        # Log-distance path-loss coefficients (see the module docstring).
        self.pl_los_const = 32.4
        self.pl_los_freq_coeff = 20.0
        self.pl_los_dist_coeff = 20.0
        self.pl_nlos_const = 38.3
        self.pl_nlos_dist_coeff = 17.5
        self.pl_nlos_freq_coeff = 24.9
        self.shadowing_std_db = 3.0

    # ------------------------------------------------------------------
    # Path loss
    # ------------------------------------------------------------------
    def path_loss_db(self, distance_m, is_los=True, shadowing_db=None):
        """Log-distance path loss, optionally with a supplied shadowing term.

        When ``shadowing_db`` is given it is used verbatim, which is how a
        :class:`ChannelObservation` keeps its path loss and its shadowing
        consistent.
        """
        d = max(float(distance_m), 1.0)
        if is_los:
            pl_db = (self.pl_los_const
                     + self.pl_los_freq_coeff * math.log10(self.fc)
                     + self.pl_los_dist_coeff * math.log10(d))
        else:
            pl_db = (self.pl_nlos_const
                     + self.pl_nlos_dist_coeff * math.log10(d)
                     + self.pl_nlos_freq_coeff * math.log10(self.fc))
        if shadowing_db is None:
            shadowing_db = float(np.random.normal(0.0, self.shadowing_std_db))
        return pl_db + shadowing_db, shadowing_db

    def calculate_path_loss_db(self, distance_m, is_los=True):
        """Backward-compatible path-loss helper (draws its own shadowing)."""
        pl_db, _ = self.path_loss_db(distance_m, is_los)
        return pl_db

    # ------------------------------------------------------------------
    # The shared channel sample
    # ------------------------------------------------------------------
    def sample_channel(self, distance_m, is_los=None, interference_w=0.0,
                       max_range=None) -> ChannelObservation:
        """Draw ONE channel realization for a single link interaction.

        The returned observation is the only channel sample that may be used
        for this interaction: it carries the transmission performance *and*
        the SINR that the trust model records.
        """
        d = float(distance_m)
        if is_los is None:
            # Line of sight is assumed for the near field, non-line-of-sight
            # beyond 150 m, matching the 3GPP highway evaluation practice.
            is_los = d < 150.0
        is_los = bool(is_los)

        pl_db, shadowing_db = self.path_loss_db(d, is_los)
        channel_gain = 10 ** (-pl_db / 10.0)
        # Small-scale fading: exponential power gain with unit mean (Rayleigh).
        small_scale = float(np.random.exponential(1.0))
        rx_power = self.tx_power * channel_gain * small_scale
        sinr = rx_power / (self.noise_power + interference_w)
        sinr = max(sinr, 1e-6)
        sinr_db = float(10.0 * math.log10(sinr))

        rate = 0.0
        if max_range is None or d <= max_range:
            bps = self.bandwidth * math.log2(1.0 + sinr)
            rate = float(np.clip(bps / 1e6, self.min_rate_mbps, self.max_rate_mbps))

        return ChannelObservation(
            distance_m=d,
            is_los=is_los,
            path_loss_db=float(pl_db),
            shadowing_db=float(shadowing_db),
            small_scale_fading=small_scale,
            channel_gain_linear=float(channel_gain),
            rx_power_w=float(rx_power),
            noise_power_w=float(self.noise_power),
            interference_w=float(interference_w),
            sinr_linear=float(sinr),
            sinr_db=sinr_db,
            data_rate_mbps=rate,
        )

    # ------------------------------------------------------------------
    # Backward-compatible helpers (each draws its own sample)
    # ------------------------------------------------------------------
    def calculate_snr(self, distance_m, is_los=True, interference_w=0.0):
        obs = self.sample_channel(distance_m, is_los=is_los, interference_w=interference_w)
        return obs.sinr_linear

    def calculate_sinr_db(self, distance_m, is_los=True, interference_w=0.0):
        """Returns physical SINR in dB from a *new* channel sample."""
        obs = self.sample_channel(distance_m, is_los=is_los, interference_w=interference_w)
        return obs.sinr_db

    def calculate_data_rate_mbps(self, distance_m, max_range=300.0, interference_w=0.0):
        """Returns the achievable data rate from a *new* channel sample."""
        obs = self.sample_channel(distance_m, interference_w=interference_w,
                                  max_range=max_range)
        return obs.data_rate_mbps

    # ------------------------------------------------------------------
    # Transmission
    # ------------------------------------------------------------------
    def transmission_delay_from_observation(self, observation: ChannelObservation,
                                            data_size_mb: float):
        """Transmission delay and energy derived from a shared observation."""
        rate_mbps = observation.data_rate_mbps
        if rate_mbps <= 0.0:
            return float('inf'), 0.0
        data_bits = float(data_size_mb) * 8.0
        rate_bps = rate_mbps * 1e6
        delay_sec = (data_bits * 1e6) / rate_bps
        energy_joules = self.tx_power * delay_sec
        return delay_sec, energy_joules

    def transmission_delay_shared(self, observation: ChannelObservation,
                                  data_size_mb: float, cell_bandwidth_mhz: float,
                                  concurrent_users: int):
        """Transmission delay and energy when the cell bandwidth is shared.

        OFDMA-style equal sharing: each of ``concurrent_users`` vehicles gets
        ``min(UE channel bandwidth, cell bandwidth / users)``.  While at most
        ``cell / UE`` users transmit, every user keeps the full UE channel and
        the result equals :meth:`transmission_delay_from_observation`.  Beyond
        that the per-user bandwidth, and hence the rate, shrinks in proportion.
        The per-user rate is still bounded by the practical MCS clip.
        """
        if observation.data_rate_mbps <= 0.0:
            return float('inf'), 0.0
        users = max(1, int(concurrent_users))
        bw_hz = min(self.bandwidth, cell_bandwidth_mhz * 1e6 / users)
        # The practical MCS clip is a *spectral-efficiency* cap: 54 Mbps on
        # the 20 MHz UE channel is 2.7 bit/s/Hz.  A user squeezed into a narrower
        # allocation therefore cannot exceed 2.7 bit/s/Hz x its bandwidth.
        se_cap = self.max_rate_mbps * 1e6 / self.bandwidth
        se = min(math.log2(1.0 + observation.sinr_linear), se_cap)
        rate_mbps = float(np.clip(bw_hz * se / 1e6, self.min_rate_mbps, self.max_rate_mbps))
        delay_sec = (float(data_size_mb) * 8.0) / rate_mbps
        return delay_sec, self.tx_power * delay_sec

    def compute_transmission_delay(self, data_size_mb, distance_m, max_range=300.0):
        """Draws a fresh observation and returns (delay, energy).

        Prefer :meth:`sample_channel` +
        :meth:`transmission_delay_from_observation` when the same sample must
        also be used for a trust update.
        """
        obs = self.sample_channel(distance_m, max_range=max_range)
        return self.transmission_delay_from_observation(obs, data_size_mb)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def compute_computation_delay(self, cpu_cycles_giga, cpu_freq_ghz):
        """Execution time (s) and energy (J) for a computation.

        Time:  t = N / f
        Energy: E = kappa * f^2 * N   (kappa in J*s^2/cycle)

        With the default kappa = 1e-30 this yields 1 pJ per cycle at 1 GHz.
        """
        if cpu_freq_ghz <= 0:
            return float('inf'), 0.0
        delay_sec = cpu_cycles_giga / cpu_freq_ghz
        f_hz = cpu_freq_ghz * 1e9
        cycles = cpu_cycles_giga * 1e9
        energy_joules = self.kappa * (f_hz ** 2) * cycles
        return delay_sec, energy_joules

    def energy_per_cycle_j(self, cpu_freq_ghz):
        """Energy per executed cycle at the given frequency (J/cycle)."""
        f_hz = float(cpu_freq_ghz) * 1e9
        return self.kappa * f_hz ** 2

    def describe(self):
        """Machine-readable description of the channel and energy model."""
        return {
            "carrier_frequency_ghz": self.fc,
            "bandwidth_mhz": self.bandwidth_mhz,
            "tx_power_dbm": self.tx_power_dbm,
            "noise_power_dbm": self.noise_power_dbm,
            "path_loss_model": "3GPP-style log-distance (TR 36.885 style fit, not a conformance claim)",
            "shadowing_std_db": self.shadowing_std_db,
            "small_scale_fading": "independent unit-mean exponential power gain (Rayleigh)",
            "interference": "not modelled (zero)",
            "rate_clip_mbps": [self.min_rate_mbps, self.max_rate_mbps],
            "energy_model": "E = kappa * f^2 * N",
            "kappa_j_s2_per_cycle": self.kappa,
            "energy_per_cycle_at_1ghz_pj": self.energy_per_cycle_j(1.0) * 1e12,
            "energy_per_cycle_at_4ghz_pj": self.energy_per_cycle_j(4.0) * 1e12,
        }
