"""Federated coordinator: selection, FedAvg aggregation, distribution, logging.

Staleness policy (documented, Phase 6): every accepted update is aggregated
with its age `current_round - round_trained` recorded. Stale updates are NOT
down-weighted or rejected here — that is Phase 7's trust-aware aggregation.
A round with zero usable updates keeps the previous global model unchanged
and is logged as skipped (no division by zero, no model destruction).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from federated.fedavg import fedavg_aggregate, payload_bytes


class FedServer:
    """Holds the global model; sees parameters + scalars, never raw experience."""

    def __init__(self, template_state: dict[str, torch.Tensor]) -> None:
        self.template = {k: v.cpu().clone() for k, v in template_state.items()}
        self.global_state = {k: v.cpu().clone() for k, v in template_state.items()}
        self.round_log: list[dict] = []

    def select_clients(self, client_ids: list[int], fraction: float,
                       rng: np.random.Generator) -> list[int]:
        """Seeded uniform subset without replacement; always at least one."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        k = max(1, int(round(len(client_ids) * fraction)))
        return sorted(int(i) for i in rng.choice(client_ids, size=k, replace=False))

    def aggregate_round(self, updates: list[dict], current_round: int) -> dict:
        """Validate, stamp staleness, FedAvg-aggregate, replace global. Returns the log."""
        t0 = time.perf_counter()
        for u in updates:
            u["staleness"] = current_round - u["round_trained"]
        states = [u["state"] for u in updates]
        weights = [float(u["num_samples"]) for u in updates]
        ids = [str(u["client_id"]) for u in updates]
        try:
            new_global = fedavg_aggregate(states, weights, self.template, ids)
            status = "aggregated"
        except ValueError as exc:
            new_global = self.global_state  # keep last good model
            status = f"skipped: {exc}"
        if status == "aggregated":
            self.global_state = new_global
        log = {
            "round": current_round,
            "status": status,
            "participants": [u["client_id"] for u in updates],
            "weights": weights,
            "staleness": [u["staleness"] for u in updates],
            "upload_bytes": sum(payload_bytes(u["state"]) for u in updates),
            "download_bytes_per_client": payload_bytes(self.global_state),
            "agg_time_s": round(time.perf_counter() - t0, 4),
        }
        self.round_log.append(log)
        return log

    def get_global(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.global_state.items()}
