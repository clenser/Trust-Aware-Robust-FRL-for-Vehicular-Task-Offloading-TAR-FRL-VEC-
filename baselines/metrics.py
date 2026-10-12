"""Shared metric definitions for baseline evaluation (SPEC §11, subset).

Rate denominators use num_tasks throughout. Completion (success) is reported
separately from deadline misses and path failures:
success + missed + failed == num_tasks always holds.
"""

from __future__ import annotations

import math
import statistics


def episode_metrics(num_tasks: int, successes: int, misses: int, failures: int,
                    latencies: list[float], energies: list[float],
                    offloads: int, rewards: list[float]) -> dict:
    """Compute one episode's metrics from raw per-task outcomes."""
    if num_tasks <= 0:
        raise ValueError("num_tasks must be > 0")
    if successes + misses + failures != num_tasks:
        raise ValueError("outcome counts must sum to num_tasks")
    return {
        "num_tasks": num_tasks,
        "successes": successes,
        "deadline_misses": misses,
        "failures": failures,
        "completion_rate": successes / num_tasks,
        "miss_rate": misses / num_tasks,
        "failure_rate": failures / num_tasks,
        "mean_latency_s": float(sum(latencies) / num_tasks),
        "mean_energy_j": float(sum(energies) / num_tasks),
        "offload_fraction": offloads / num_tasks,
        "mean_reward": float(sum(rewards) / num_tasks),
    }


def aggregate(rows: list[dict], keys: list[str]) -> dict:
    """Mean ± sample-stdev per metric across seeds (variability, not just averages)."""
    out: dict = {}
    for k in keys:
        vals = [r[k] for r in rows]
        out[k] = {
            "mean": float(statistics.fmean(vals)),
            "stdev": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": len(vals),
        }
        for v in vals:
            if not math.isfinite(v):
                raise ValueError(f"non-finite metric {k}: {vals}")
    return out
