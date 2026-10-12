"""Phase 8: Adversarial and Dynamic Scenarios package."""

from attacks.base import AttackType, ClientScenarioLabel, ScenarioAssignment
from attacks.free_riding import apply_copy_update, apply_zero_update, create_stale_update
from attacks.metrics import (
    compute_confidence_interval_95,
    compute_confusion_matrix,
    compute_degradation,
    compute_detection_metrics,
    validate_task_accounting,
)
from attacks.model_poisoning import apply_perturbation_attack, apply_sign_flip_attack
from attacks.pipeline import AttackPipeline
from attacks.unreliable_execution import UnreliableEnvWrapper, UnreliableProfile

__all__ = [
    "AttackType",
    "ClientScenarioLabel",
    "ScenarioAssignment",
    "AttackPipeline",
    "apply_sign_flip_attack",
    "apply_perturbation_attack",
    "apply_zero_update",
    "apply_copy_update",
    "create_stale_update",
    "UnreliableProfile",
    "UnreliableEnvWrapper",
    "compute_confusion_matrix",
    "compute_detection_metrics",
    "compute_confidence_interval_95",
    "compute_degradation",
    "validate_task_accounting",
]
