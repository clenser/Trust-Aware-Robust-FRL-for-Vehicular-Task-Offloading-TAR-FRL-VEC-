"""Public API of federated learning components (Phase 6 FedAvg and Phase 7 Trust-aware Aggregation)."""

from federated.client import ALLOWED_UPDATE_KEYS, FedClient
from federated.fedavg import fedavg_aggregate, payload_bytes, validate_update
from federated.server import FedServer
from federated.trust_aggregation import DEFAULT_TAU_EXC, trust_weighted_aggregate

__all__ = [
    "ALLOWED_UPDATE_KEYS",
    "DEFAULT_TAU_EXC",
    "FedClient",
    "FedServer",
    "fedavg_aggregate",
    "payload_bytes",
    "trust_weighted_aggregate",
    "validate_update",
]
