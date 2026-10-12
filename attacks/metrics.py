"""Research metrics for Phase 8: Detection, offloading performance, degradation, and statistical reporting."""

from __future__ import annotations

import math
import statistics
from typing import Any


def compute_confusion_matrix(
    ground_truth: list[bool],
    predictions: list[bool],
) -> dict[str, int]:
    """Compute TP, FP, TN, FN for binary detection against ground truth."""
    if len(ground_truth) != len(predictions):
        raise ValueError(f"Length mismatch: {len(ground_truth)} != {len(predictions)}")

    tp = fp = tn = fn = 0
    for y_true, y_pred in zip(ground_truth, predictions):
        if y_true and y_pred:
            tp += 1
        elif not y_true and y_pred:
            fp += 1
        elif not y_true and not y_pred:
            tn += 1
        else:
            fn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_detection_metrics(
    ground_truth: list[bool],
    predictions: list[bool],
) -> dict[str, float]:
    """Compute Precision, Recall, F1, FPR, TPR with safe zero-denominator handling."""
    cm = compute_confusion_matrix(ground_truth, predictions)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = recall
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "accuracy": float(accuracy),
    }


def validate_task_accounting(
    successes: int,
    deadline_misses: int,
    failures: int,
    num_tasks: int,
) -> bool:
    """Verify core accounting identity: successes + deadline_misses + failures == num_tasks."""
    total = successes + deadline_misses + failures
    if total != num_tasks:
        raise ValueError(
            f"Accounting identity violation: {successes} + {deadline_misses} + {failures} = {total} != {num_tasks}"
        )
    return True


def compute_confidence_interval_95(values: list[float]) -> dict[str, Any]:
    """Compute sample mean, standard deviation, and Student-t 95% confidence interval half-width.

    Reports:
    - mean: sample mean
    - stdev: sample standard deviation (Bessel-corrected, ddof=1)
    - ci95_half: half-width of the 95% confidence interval
    - ci95_low: mean - ci95_half
    - ci95_high: mean + ci95_half
    - n: sample count
    - method: 'student_t' for small samples (n <= 10), 'asymptotic_normal' for n > 10
    - confidence_level: 0.95
    """
    n = len(values)
    if n == 0:
        return {
            "mean": 0.0,
            "stdev": 0.0,
            "ci95_half": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n": 0,
            "method": "none",
            "confidence_level": 0.95,
        }
    if n == 1:
        m = float(values[0])
        return {
            "mean": m,
            "stdev": 0.0,
            "ci95_half": 0.0,
            "ci95_low": m,
            "ci95_high": m,
            "n": 1,
            "method": "single_sample",
            "confidence_level": 0.95,
        }

    mean_val = float(statistics.fmean(values))
    std_val = float(statistics.stdev(values))
    # Two-sided Student-t critical values for alpha = 0.05:
    # df = n - 1:
    t_crits = {
        2: 12.706,  # df = 1
        3: 4.303,   # df = 2
        4: 3.182,   # df = 3
        5: 2.776,   # df = 4
        6: 2.571,   # df = 5
        7: 2.447,   # df = 6
        8: 2.365,   # df = 7
        9: 2.306,   # df = 8
        10: 2.262,  # df = 9
    }
    method = "student_t" if n in t_crits else "asymptotic_normal"
    t_val = t_crits.get(n, 1.96)
    half_width = float(t_val * (std_val / math.sqrt(n)))

    return {
        "mean": mean_val,
        "stdev": std_val,
        "ci95_half": half_width,
        "ci95_low": mean_val - half_width,
        "ci95_high": mean_val + half_width,
        "n": n,
        "method": method,
        "confidence_level": 0.95,
    }


def compute_degradation(
    clean_metric: float,
    attack_metric: float,
    higher_is_better: bool = True,
) -> float:
    """Compute performance degradation from clean baseline:

    If higher is better (e.g. completion rate, reward): clean - attack (positive = degraded).
    If lower is better (e.g. latency, failure rate): attack - clean (positive = degraded).
    """
    return (clean_metric - attack_metric) if higher_is_better else (attack_metric - clean_metric)
