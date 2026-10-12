"""Vehicle model (SPEC §1): position, speed, CPU, queue, radio."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vehicle:
    """A task-generating vehicle that can also execute offloaded tasks."""

    vid: int
    x_m: float
    speed_mps: float
    cpu_freq_hz: float
    utilisation: float = 0.0  # background load fraction in [0, 1]
    queue_cycles: float = 0.0  # queued work ahead (FIFO, in cycles)
    bandwidth_hz: float = 10e6
    tx_power_w: float = 0.5
    available: bool = True  # node-level up/down flag (intermittent participation)

    def validate(self) -> None:
        if self.cpu_freq_hz <= 0 or self.bandwidth_hz <= 0 or self.tx_power_w <= 0:
            raise ValueError("cpu freq, bandwidth and tx power must be > 0")
        if not 0.0 <= self.utilisation <= 1.0:
            raise ValueError("utilisation must be in [0, 1]")
        if self.queue_cycles < 0:
            raise ValueError("queue_cycles must be >= 0")

    @property
    def effective_freq_hz(self) -> float:
        """CPU share left for new tasks after background utilisation."""
        return self.cpu_freq_hz * max(0.05, 1.0 - self.utilisation)

    def drain_queue(self, dt_s: float) -> None:
        """Serve queued cycles over elapsed time dt (FIFO service model)."""
        self.queue_cycles = max(0.0, self.queue_cycles - self.effective_freq_hz * dt_s)

    def enqueue(self, cycles: float) -> None:
        self.queue_cycles += cycles
