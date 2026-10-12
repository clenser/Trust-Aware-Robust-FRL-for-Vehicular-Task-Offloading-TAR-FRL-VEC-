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
    unreliable_fraction: float = 0.0
    unreliable_client_ids: list[int] = field(default_factory=list)
    unreliable_severity: str = "none"

    @classmethod
    def create(
        cls,
        num_clients: int,
        malicious_fraction: float,
        attack_type: AttackType,
        seed: int,
        attack_strength: float = 1.0,
        unreliable_fraction: float = 0.0,
        unreliable_severity: str = "none",
    ) -> ScenarioAssignment:
        if not 0.0 <= malicious_fraction <= 1.0:
            raise ValueError(f"malicious_fraction must be in [0, 1], got {malicious_fraction}")
        if not 0.0 <= unreliable_fraction <= 1.0:
            raise ValueError(f"unreliable_fraction must be in [0, 1], got {unreliable_fraction}")
        
        num_malicious = int(round(num_clients * malicious_fraction))
        rng = np.random.default_rng(seed + 999)
        
        # Deterministically select which client IDs are malicious
        all_ids = list(range(num_clients))
        if num_malicious > 0 and attack_type != AttackType.NONE:
            malicious_ids = sorted(int(i) for i in rng.choice(all_ids, size=num_malicious, replace=False))
        else:
            malicious_ids = []

        # Deterministically select which client IDs are unreliable (from non-malicious clients)
        num_unreliable = int(round(num_clients * unreliable_fraction))
        eligible_unreliable = [cid for cid in all_ids if cid not in malicious_ids]
        if num_unreliable > 0 and len(eligible_unreliable) > 0:
            rng_u = np.random.default_rng(seed + 8888)
            actual_unreliable_count = min(num_unreliable, len(eligible_unreliable))
            unreliable_ids = sorted(int(i) for i in rng_u.choice(eligible_unreliable, size=actual_unreliable_count, replace=False))
        else:
            unreliable_ids = []

        client_labels: dict[int, ClientScenarioLabel] = {}
        for cid in all_ids:
            if cid in malicious_ids:
                client_labels[cid] = ClientScenarioLabel(
                    client_id=cid,
                    is_malicious=True,
                    attack_type=attack_type,
                    attack_strength=attack_strength,
                )
            elif cid in unreliable_ids:
                # Crucial invariant: Unreliable clients are honest, not malicious!
                client_labels[cid] = ClientScenarioLabel(
                    client_id=cid,
                    is_malicious=False,
                    attack_type=AttackType.NONE,
                    attack_strength=0.0,
                    notes=f"unreliable_{unreliable_severity}",
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
            unreliable_fraction=unreliable_fraction,
            unreliable_client_ids=unreliable_ids,
            unreliable_severity=unreliable_severity,
        )
