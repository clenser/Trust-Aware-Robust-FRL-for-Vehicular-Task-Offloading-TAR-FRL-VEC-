"""Federated coordinator: selection, FedAvg / trust-aware aggregation, distribution, logging.

Supported aggregation strategies:
- "fedavg": Standard sample-count-weighted FedAvg (SPEC EQ-21, Phase 6).
- "trust_weighted": Trust-weighted FedAvg using T_final (SPEC EQ-22, Phase 7).
- "trust_threshold": Trust-weighted aggregation with exclusion threshold tau_exc
  and fallback model retention (SPEC EQ-23, Phase 7).

Staleness policy:
- Age `current_round - round_trained` is stamped on every update.
- Under "fedavg", staleness is logged only.
- Under trust-aware strategies, staleness reduces indicator m3 (EQ-16) and thus T_model (EQ-19).

Fallback policy:
- If zero updates are received or all clients are excluded by threshold, the previous
  global model is preserved unchanged, and the round is logged as fallback/skipped.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch

from federated.fedavg import fedavg_aggregate, payload_bytes
from federated.trust_aggregation import trust_weighted_aggregate
from trust.model_update import (
    DEFAULT_ALPHA,
    DEFAULT_CLIP_C,
    DEFAULT_GAMMA,
    DEFAULT_TAU_EXC,
    UpdateTrustReport,
    evaluate_model_updates,
)


class FedServer:
    """Holds the global model; sees parameters + scalars, never raw experience."""

    def __init__(
        self,
        template_state: dict[str, torch.Tensor],
        strategy: Literal["fedavg", "trust_weighted", "trust_threshold"] = "fedavg",
        alpha: float = DEFAULT_ALPHA,
        tau_exc: float = DEFAULT_TAU_EXC,
        clip_c: float = DEFAULT_CLIP_C,
        indicator_weights: dict[str, float] | None = None,
    ) -> None:
        self.template = {k: v.cpu().clone() for k, v in template_state.items()}
        self.global_state = {k: v.cpu().clone() for k, v in template_state.items()}
        self.strategy = strategy
        self.alpha = float(alpha)
        self.tau_exc = float(tau_exc)
        self.clip_c = float(clip_c)
        self.indicator_weights = indicator_weights or DEFAULT_GAMMA
        self.round_log: list[dict] = []

    def select_clients(
        self,
        client_ids: list[int],
        fraction: float,
        rng: np.random.Generator,
    ) -> list[int]:
        """Seeded uniform subset without replacement; always at least one."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        k = max(1, int(round(len(client_ids) * fraction)))
        return sorted(int(i) for i in rng.choice(client_ids, size=k, replace=False))

    def aggregate_round(
        self,
        updates: list[dict],
        current_round: int,
        behavioural_trusts: dict[int | str, float] | None = None,
    ) -> dict:
        """Validate, stamp staleness, aggregate according to strategy, replace global. Returns the log."""
        t0 = time.perf_counter()
        for u in updates:
            u["staleness"] = current_round - u["round_trained"]

        states = [u["state"] for u in updates]
        sample_weights = [float(u["num_samples"]) for u in updates]
        ids = [u["client_id"] for u in updates]
        str_ids = [str(u["client_id"]) for u in updates]

        trust_reports: list[UpdateTrustReport] = []
        accepted_ids: list[int | str] = []
        aggregation_weights: list[float] = []

        if not updates:
            new_global = self.global_state
            status = "skipped: no updates provided"
        elif self.strategy == "fedavg":
            # Phase 6 baseline: sample-count weighted FedAvg
            try:
                new_global = fedavg_aggregate(states, sample_weights, self.template, str_ids)
                status = "aggregated"
                accepted_ids = ids
                total_samples = sum(sample_weights)
                aggregation_weights = [w / total_samples for w in sample_weights] if total_samples > 0 else []
            except ValueError as exc:
                new_global = self.global_state  # keep last good model
                status = f"skipped: {exc}"
        elif self.strategy in ("trust_weighted", "trust_threshold"):
            # Phase 7: Evaluate model update trust indicators
            trust_reports = evaluate_model_updates(
                updates=updates,
                reference_state=self.global_state,
                behavioural_trusts=behavioural_trusts,
                weights=self.indicator_weights,
                alpha=self.alpha,
                clip_c=self.clip_c,
            )
            t_finals = [rep.t_final for rep in trust_reports]
            threshold = self.tau_exc if self.strategy == "trust_threshold" else 0.0

            try:
                new_global, accepted_ids, aggregation_weights = trust_weighted_aggregate(
                    states=states,
                    trust_scores=t_finals,
                    template=self.template,
                    client_ids=ids,
                    threshold=threshold,
                    fallback_state=self.global_state,
                )
                if not accepted_ids:
                    status = "skipped: all clients excluded below threshold"
                else:
                    status = "aggregated"
            except ValueError as exc:
                new_global = self.global_state
                status = f"skipped: {exc}"
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

        if status == "aggregated":
            self.global_state = new_global

        excluded_ids = [cid for cid in ids if cid not in accepted_ids]

        log = {
            "round": current_round,
            "strategy": self.strategy,
            "status": status,
            "participants": ids,
            "accepted": accepted_ids,
            "excluded": excluded_ids,
            "weights": aggregation_weights if aggregation_weights else sample_weights,
            "staleness": [u["staleness"] for u in updates],
            "upload_bytes": sum(payload_bytes(u["state"]) for u in updates),
            "download_bytes_per_client": payload_bytes(self.global_state),
            "agg_time_s": round(time.perf_counter() - t0, 4),
            "trust_reports": [r.to_dict() for r in trust_reports],
        }
        self.round_log.append(log)
        return log

    def get_global(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.global_state.items()}
