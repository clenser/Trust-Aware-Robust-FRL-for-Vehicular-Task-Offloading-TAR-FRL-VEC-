"""Phase 2 environment configuration: YAML loading, defaults, validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

MOBILITY_LEVELS = ("low", "medium", "high")
TASK_LOADS = ("low", "medium", "high")

#: EQ-9 component counts: 4 task + 2 local + 3 per candidate + 1 count.
#: State dimension is therefore n = 7 + 3K (SPEC §4, corrected erratum).
OBS_TASK_DIM = 4
OBS_LOCAL_DIM = 2
OBS_PER_CANDIDATE = 3
OBS_TAIL_DIM = 1
OBS_BASE_DIM = OBS_TASK_DIM + OBS_LOCAL_DIM + OBS_TAIL_DIM  # = 7

#: Vehicle speed ranges in m/s per mobility level (synthetic; see environment/README.md).
SPEED_RANGES: dict[str, tuple[float, float]] = {
    "low": (5.0, 12.0),
    "medium": (12.0, 22.0),
    "high": (22.0, 33.0),
}

#: Mean task inter-arrival gap in seconds per load level (synthetic).
MEAN_GAP_S: dict[str, float] = {"low": 1.0, "medium": 0.5, "high": 0.25}

#: Task data-size multiplier per load level (synthetic).
LOAD_SIZE_MULT: dict[str, float] = {"low": 0.5, "medium": 1.0, "high": 2.0}


@dataclass
class EnvConfig:
    """All environment parameters with units. Defaults are synthetic placeholders."""

    seed: int = 0
    num_vehicles: int = 5
    num_edge_nodes: int = 2
    mobility: str = "low"
    task_load: str = "low"
    comm_range_m: float = 150.0
    num_tasks: int = 50
    road_length_m: float = 2000.0
    top_k: int = 5  # EQ-10: max offload candidates (SPEC U2 default K=5)
    link_reliability: float = 0.95  # per-link per-step availability probability
    bandwidth_hz: float = 10e6
    handshake_s: float = 0.005
    tx_power_w: float = 0.5
    # Path-loss law (SPEC U3 resolution): snr = tx_power * pl_k0 / ((d+1)^gamma * noise_w)
    pl_k0: float = 2e-3
    pl_gamma: float = 3.0
    noise_w: float = 1e-9
    kappa: float = 1e-27  # EQ-7 chip constant (synthetic)
    # Observation normalisation bounds (EQ-9; SPEC U3 resolution, documented).
    data_bits_max: float = 20e6
    cycles_max: float = 40e9
    deadline_max_s: float = 30.0
    queue_cycles_max: float = 50e9
    # Neutral trust placeholder until Phase 5 implements trust scoring.
    neutral_trust: float = 0.5
    # Behavioural-trust dynamics (SPEC §6 defaults; Phase 5 implements).
    trust_beta: tuple = (0.4, 0.25, 0.2, 0.15)
    trust_lambda: float = 0.9
    trust_rho: float = 0.99
    trust_prior: float = 0.5
    trust_threshold: float = 0.3  # trust_aware policy falls back to local below this
    vehicle_cpu_ghz: float | None = None  # optional explicit vehicle compute override

    @property
    def obs_dim(self) -> int:
        """Declared EQ-9 state dimension: n = 7 + 3K (SPEC §4)."""
        return OBS_BASE_DIM + OBS_PER_CANDIDATE * self.top_k

    @property
    def action_dim(self) -> int:
        """Declared EQ-10 action count: 1 (local) + K (candidates)."""
        return 1 + self.top_k

    def validate(self) -> None:
        """Raise ValueError on any out-of-range parameter."""
        if self.num_vehicles < 1:
            raise ValueError("num_vehicles must be >= 1")
        if self.num_edge_nodes < 0:
            raise ValueError("num_edge_nodes must be >= 0")
        if self.mobility not in MOBILITY_LEVELS:
            raise ValueError(f"mobility must be one of {MOBILITY_LEVELS}")
        if self.task_load not in TASK_LOADS:
            raise ValueError(f"task_load must be one of {TASK_LOADS}")
        if self.comm_range_m <= 0:
            raise ValueError("comm_range_m must be > 0")
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 < self.link_reliability <= 1.0:
            raise ValueError("link_reliability must be in (0, 1]")
        if not 0.0 <= self.trust_lambda < 1.0:
            raise ValueError("trust_lambda must be in [0, 1)")
        if not 0.0 < self.trust_rho <= 1.0:
            raise ValueError("trust_rho must be in (0, 1]")
        if abs(sum(self.trust_beta) - 1.0) > 1e-9:
            raise ValueError("trust_beta must sum to 1")
        if self.vehicle_cpu_ghz is not None and self.vehicle_cpu_ghz <= 0:
            raise ValueError("vehicle_cpu_ghz must be positive")

    @classmethod
    def from_raw(cls, raw: dict) -> "EnvConfig":
        """Build from a parsed project YAML dict (same mapping as from_yaml)."""
        scenario = (raw or {}).get("scenario", {}) or {}
        federated = (raw or {}).get("federated", {}) or {}
        t_lam = scenario.get("trust_lambda", federated.get("trust_lambda", (raw or {}).get("trust_lambda", None)))
        kwargs: dict = {
            "seed": int((raw or {}).get("seed", 0)),
            "num_vehicles": int(scenario.get("num_vehicles", 5)),
            "num_edge_nodes": int(scenario.get("num_edge_nodes", 2)),
            "mobility": str(scenario.get("mobility", "low")),
            "task_load": str(scenario.get("task_load", "low")),
            "comm_range_m": float(scenario.get("connectivity_range_m", 150.0)),
            "num_tasks": int(scenario.get("num_steps", 50)),
        }
        if t_lam is not None:
            kwargs["trust_lambda"] = float(t_lam)
        if "bandwidth_hz" in scenario:
            kwargs["bandwidth_hz"] = float(scenario["bandwidth_hz"])
        if "vehicle_cpu_ghz" in scenario:
            kwargs["vehicle_cpu_ghz"] = float(scenario["vehicle_cpu_ghz"])
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg


def from_yaml(path: str | Path) -> EnvConfig:
    """Build an EnvConfig from a project YAML file (reads the `scenario` section).

    Unknown keys (e.g. malicious_client_ratio, placeholders_only) are ignored:
    attacks arrive in Phase 8. Missing keys fall back to EnvConfig defaults.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EnvConfig.from_raw(data)


@dataclass
class EpisodeSummary:
    """Aggregate outcome of one episode (counts only; no learning)."""

    num_tasks: int = 0
    successes: int = 0
    deadline_misses: int = 0
    failures: int = 0
    mean_latency_s: float = 0.0
    mean_energy_j: float = 0.0
    offload_fraction: float = 0.0
