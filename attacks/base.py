"""Base definitions and interfaces for Phase 8 adversarial and dynamic scenarios.

Maintains strict separation between:
1. Ground-truth scenario labels (held by experiment runner, NEVER passed to trust detector).
2. Server-visible update payloads (parameters + metadata ONLY).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np


class AttackType(str, Enum):
    NONE = "none"
    SIGN_FLIP = "sign_flip"
    PERTURBATION = "perturbation"
    ZERO_UPDATE = "zero_update"
    COPY_UPDATE = "copy_update"
    STALE_REPLAY = "stale_replay"
    UNRELIABLE = "unreliable"


@dataclass
class ClientScenarioLabel:
    """Ground-truth scenario assignment for a client in an experimental run.

    CRITICAL SECURITY & METHODOLOGICAL INVARIANT:
    These labels exist ONLY in the experiment runner / evaluation harness.
    They must NEVER be injected into client updates or passed into the trust detector.
    """
    client_id: int
    is_malicious: bool
    attack_type: AttackType
    attack_strength: float = 1.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "is_malicious": self.is_malicious,
            "attack_type": self.attack_type.value,
            "attack_strength": self.attack_strength,
            "notes": self.notes,
        }


@dataclass
class ScenarioAssignment:
    """Deterministic assignment of client roles for a given attack fraction and seed."""
    num_clients: int
    malicious_fraction: float
    attack_type: AttackType
    attack_strength: float = 1.0
    malicious_client_ids: list[int] = field(default_factory=list)
    client_labels: dict[int, ClientScenarioLabel] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        num_clients: int,
        malicious_fraction: float,
        attack_type: AttackType,
        seed: int,
        attack_strength: float = 1.0,
    ) -> ScenarioAssignment:
        if not 0.0 <= malicious_fraction <= 1.0:
            raise ValueError(f"malicious_fraction must be in [0, 1], got {malicious_fraction}")
        
        num_malicious = int(round(num_clients * malicious_fraction))
        rng = np.random.default_rng(seed + 999)
        
        # Deterministically select which client IDs are malicious
        all_ids = list(range(num_clients))
        if num_malicious > 0 and attack_type != AttackType.NONE:
            malicious_ids = sorted(int(i) for i in rng.choice(all_ids, size=num_malicious, replace=False))
        else:
            malicious_ids = []

        client_labels: dict[int, ClientScenarioLabel] = {}
        for cid in all_ids:
            if cid in malicious_ids:
                client_labels[cid] = ClientScenarioLabel(
                    client_id=cid,
                    is_malicious=True,
                    attack_type=attack_type,
                    attack_strength=attack_strength,
                )
            else:
                client_labels[cid] = ClientScenarioLabel(
                    client_id=cid,
                    is_malicious=False,
                    attack_type=AttackType.NONE,
                    attack_strength=0.0,
                )

        return cls(
            num_clients=num_clients,
            malicious_fraction=malicious_fraction,
            attack_type=attack_type,
            attack_strength=attack_strength,
            malicious_client_ids=malicious_ids,
            client_labels=client_labels,
        )
