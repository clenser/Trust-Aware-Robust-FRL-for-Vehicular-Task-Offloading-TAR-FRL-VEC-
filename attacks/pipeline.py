"""Pipeline integration for applying configured attacks between client training and server aggregation."""

from __future__ import annotations

import copy
import numpy as np
import torch

from attacks.base import AttackType, ClientScenarioLabel, ScenarioAssignment
from attacks.free_riding import apply_copy_update, apply_zero_update, create_stale_update
from attacks.model_poisoning import apply_perturbation_attack, apply_sign_flip_attack


class AttackPipeline:
    """Manages attack application across rounds while maintaining evaluation isolation."""

    def __init__(self, scenario: ScenarioAssignment, seed: int = 42) -> None:
        self.scenario = scenario
        self.seed = seed
        self._rng = np.random.default_rng(seed + 777)
        self.history_updates: dict[int, list[dict]] = {}

    def transform_update(
        self,
        update: dict,
        reference_state: dict[str, torch.Tensor],
        current_round: int,
        peer_updates: list[dict] | None = None,
    ) -> tuple[dict, dict]:
        """Apply attack according to client's assigned scenario role.

        Returns:
        - attacked_update: update dictionary to be submitted to the FedServer.
        - audit_info: diagnostic metadata for the experiment runner (never given to server/detector).
        """
        cid = update["client_id"]
        label = self.scenario.client_labels.get(
            cid,
            ClientScenarioLabel(client_id=cid, is_malicious=False, attack_type=AttackType.NONE),
        )

        # Store honest copy in history
        if cid not in self.history_updates:
            self.history_updates[cid] = []
        self.history_updates[cid].append(copy.deepcopy(update))

        audit_info = {
            "client_id": cid,
            "round": current_round,
            "is_malicious": label.is_malicious,
            "attack_type": label.attack_type.value,
            "attack_strength": label.attack_strength,
            "attack_applied": False,
        }

        if not label.is_malicious or label.attack_type == AttackType.NONE:
            return update, audit_info

        # Create copy for mutation
        attacked = copy.deepcopy(update)
        audit_info["attack_applied"] = True

        if label.attack_type == AttackType.SIGN_FLIP:
            attacked["state"] = apply_sign_flip_attack(
                honest_state=update["state"],
                reference_state=reference_state,
                strength=label.attack_strength,
            )
        elif label.attack_type == AttackType.PERTURBATION:
            attack_rng = np.random.default_rng(self.seed + current_round * 1000 + cid)
            attacked["state"] = apply_perturbation_attack(
                honest_state=update["state"],
                reference_state=reference_state,
                strength=label.attack_strength,
                rng=attack_rng,
            )
        elif label.attack_type == AttackType.ZERO_UPDATE:
            attacked["state"] = apply_zero_update(reference_state=reference_state)
        elif label.attack_type == AttackType.COPY_UPDATE:
            # Copy from first available peer update, or honest state if no peer
            target_state = None
            if peer_updates:
                for p in peer_updates:
                    if p["client_id"] != cid:
                        target_state = p["state"]
                        break
            if target_state is None:
                target_state = update["state"]
            attacked["state"] = apply_copy_update(target_state=target_state)
        elif label.attack_type == AttackType.STALE_REPLAY:
            # Replay historical update from previous rounds if available
            hist = self.history_updates.get(cid, [])
            prior_updates = hist[:-1] if len(hist) > 1 else []
            if not prior_updates or current_round == 0:
                # In round 0 or with no historical update yet, cannot replay an earlier update
                attacked = update
                audit_info["attack_applied"] = False
                audit_info["origin_round"] = current_round
                audit_info["staleness_age"] = 0
            else:
                # Target staleness age from attack strength (default 2 or 1)
                target_age = max(1, int(round(label.attack_strength))) if label.attack_strength >= 1 else 1
                target_round = max(0, current_round - target_age)
                # Find matching historical update by round_trained, or fallback to earliest prior
                chosen = None
                for u_prev in prior_updates:
                    if u_prev.get("round_trained") == target_round:
                        chosen = u_prev
                        break
                if chosen is None:
                    chosen = prior_updates[0]

                origin_round = int(chosen.get("round_trained", 0))
                attacked = copy.deepcopy(chosen)
                # Preserve genuine originating round metadata!
                attacked["round_trained"] = origin_round
                attacked["staleness"] = max(0, current_round - origin_round)
                audit_info["attack_applied"] = True
                audit_info["origin_round"] = origin_round
                audit_info["staleness_age"] = max(0, current_round - origin_round)
        elif label.attack_type == AttackType.UNRELIABLE:
            # For unreliable node, parameters may be trained normally, but behaviour is faulty
            pass
        else:
            raise ValueError(f"Unknown attack type: {label.attack_type}")

        return attacked, audit_info
