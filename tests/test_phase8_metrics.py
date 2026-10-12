"""Unit tests for Phase 8 metrics, confusion matrix, accounting validation, and statistical functions."""

import pytest

from attacks.metrics import (
    compute_confidence_interval_95,
    compute_confusion_matrix,
    compute_degradation,
    compute_detection_metrics,
    validate_task_accounting,
)


def test_confusion_matrix_counts():
    y_true = [True, True, False, False, True]
    y_pred = [True, False, False, True, True]

    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["tp"] == 2  # idx 0, 4
    assert cm["fn"] == 1  # idx 1
    assert cm["fp"] == 1  # idx 3
    assert cm["tn"] == 1  # idx 2


def test_detection_metrics_math_and_zero_denominators():
    # Regular case: TP=2, FP=1, FN=1, TN=1
    y_true = [True, True, False, False, True]
    y_pred = [True, False, False, True, True]

    met = compute_detection_metrics(y_true, y_pred)
    # Precision = 2 / (2 + 1) = 2/3
    assert met["precision"] == pytest.approx(2.0 / 3.0)
    # Recall = 2 / (2 + 1) = 2/3
    assert met["recall"] == pytest.approx(2.0 / 3.0)
    # F1 = 2/3
    assert met["f1"] == pytest.approx(2.0 / 3.0)
    # FPR = 1 / (1 + 1) = 0.5
    assert met["fpr"] == pytest.approx(0.5)

    # Zero denominator: no positive predictions
    met_zero = compute_detection_metrics([True, False], [False, False])
    assert met_zero["precision"] == 0.0
    assert met_zero["recall"] == 0.0
    assert met_zero["f1"] == 0.0

    # Clean condition: no ground truth malicious clients
    met_clean = compute_detection_metrics([False, False, False], [False, False, False])
    assert met_clean["fp"] == 0.0
    assert met_clean["fpr"] == 0.0


def test_task_accounting_identity():
    assert validate_task_accounting(successes=7, deadline_misses=2, failures=1, num_tasks=10) is True

    with pytest.raises(ValueError):
        validate_task_accounting(successes=7, deadline_misses=2, failures=0, num_tasks=10)


def test_confidence_interval():
    # Empty
    ci_empty = compute_confidence_interval_95([])
    assert ci_empty["n"] == 0

    # Single
    ci_one = compute_confidence_interval_95([0.75])
    assert ci_one["mean"] == 0.75
    assert ci_one["stdev"] == 0.0
    assert ci_one["ci95_half"] == 0.0

    # Multiple
    vals = [0.8, 0.82, 0.78, 0.81, 0.79]
    ci_mult = compute_confidence_interval_95(vals)
    assert ci_mult["n"] == 5
    assert ci_mult["mean"] == pytest.approx(0.80)
    assert ci_mult["ci95_low"] < ci_mult["mean"] < ci_mult["ci95_high"]


def test_degradation():
    # Higher is better (completion rate)
    deg_comp = compute_degradation(clean_metric=0.85, attack_metric=0.65, higher_is_better=True)
    assert deg_comp == pytest.approx(0.20)

    # Lower is better (latency)
    deg_lat = compute_degradation(clean_metric=2.0, attack_metric=3.5, higher_is_better=False)
    assert deg_lat == pytest.approx(1.5)
