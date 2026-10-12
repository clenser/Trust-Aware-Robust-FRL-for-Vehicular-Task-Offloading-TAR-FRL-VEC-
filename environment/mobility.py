"""Mobility (SPEC EQ-1): 1-D road positions with wrap-around.

Resolution of SPEC U3: vehicles move on a straight segment [0, road_length)
with constant per-vehicle speed drawn from the mobility-level range plus small
bounded noise each step; positions wrap (torus) to keep density stationary.
This is synthetic and makes no claim about real traffic traces.
"""

from __future__ import annotations

from environment.config import SPEED_RANGES
from environment.vehicle import Vehicle


class MobilityModel:
    """Advances vehicle positions; owns no RNG state beyond the shared generator."""

    def __init__(self, rng, road_length_m: float, mobility: str) -> None:
        self._rng = rng
        self._road = road_length_m
        self._lo, self._hi = SPEED_RANGES[mobility]

    def sample_speed(self) -> float:
        return float(self._rng.uniform(self._lo, self._hi))

    def step(self, vehicles: list[Vehicle], dt_s: float) -> None:
        """EQ-1: x(t+1) = x(t) + v*dt + bounded noise, wrapped to the segment."""
        for v in vehicles:
            noise = float(self._rng.uniform(-1.0, 1.0)) * dt_s
            v.x_m = (v.x_m + v.speed_mps * dt_s + noise) % self._road
