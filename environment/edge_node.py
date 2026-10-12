"""Roadside edge (RSU) node model (SPEC §1): fixed position, CPU, queue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EdgeNode:
    """A stationary roadside unit serving offloaded tasks (no background load)."""

    eid: int
    x_m: float
    cpu_freq_hz: float
    queue_cycles: float = 0.0
    available: bool = True

    def validate(self) -> None:
        if self.cpu_freq_hz <= 0:
            raise ValueError("cpu freq must be > 0")
        if self.queue_cycles < 0:
            raise ValueError("queue_cycles must be >= 0")

    def drain_queue(self, dt_s: float) -> None:
        self.queue_cycles = max(0.0, self.queue_cycles - self.cpu_freq_hz * dt_s)

    def enqueue(self, cycles: float) -> None:
        self.queue_cycles += cycles
