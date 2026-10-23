"""
Alternative client-exclusion rules, evaluated in SHADOW mode.

The deployed detector (``trust.model_trust.ModelTrustEvaluator`` plus the joint
trust threshold) decides which updates are aggregated.  To compare it with
alternatives *on identical clients, data partitions, seeds and attacks*, the
server also passes the very same round of updates through the rules below.
Their decisions are recorded and scored against the ground truth, but they
never influence the aggregate, so every detector sees exactly the same
sequence of updates.

All rules are stateless functions of one round of flattened updates
``{client_id: delta_vector}`` and return the set of client ids they would
exclude.  None of them reads the ground-truth role.

* ``krum``           - Multi-Krum selection (Blanchard et al., NeurIPS 2017)
                       with an assumed Byzantine fraction (default 0.25, a
                       fixed design parameter that does not use the realised
                       attacker count).
* ``cosine_median``  - exclude a client whose median pairwise cosine
                       similarity to the others is negative.
* ``distance_mad``   - exclude a client whose distance to the coordinate-wise
                       median is a robust-z outlier (> 3 MAD-scaled units) or
                       whose norm is more than 3x / less than 0.15x the
                       population median norm.
* ``norm_clip_only`` - never excludes anyone (norm clipping is an aggregation
                       change, not a detector); kept as the "no detection"
                       reference row so that every table has the same shape.
"""

from typing import Dict, Set

import numpy as np


def _matrix(updates: Dict) -> (list, np.ndarray):
    ids = list(updates.keys())
    mat = np.array([np.asarray(updates[c], dtype=float) for c in ids])
    return ids, mat


def krum_exclusion(updates: Dict, byzantine_fraction: float = 0.25) -> Set:
    ids, mat = _matrix(updates)
    n = len(ids)
    if n <= 3:
        return set()
    f = max(1, int(round(n * byzantine_fraction)))
    f = min(f, n - 3)
    d = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=2)
    k = max(1, n - f - 2)
    scores = np.array([np.sort(d[i])[1:k + 1].sum() for i in range(n)])
    keep = set(np.argsort(scores)[:n - f].tolist())
    return {ids[i] for i in range(n) if i not in keep}


def cosine_median_exclusion(updates: Dict) -> Set:
    ids, mat = _matrix(updates)
    n = len(ids)
    if n <= 2:
        return set()
    norms = np.linalg.norm(mat, axis=1)
    cos = (mat @ mat.T) / (np.outer(norms, norms) + 1e-12)
    np.fill_diagonal(cos, np.nan)
    med = np.nanmedian(cos, axis=1)
    return {ids[i] for i in range(n) if med[i] < 0.0}


def distance_mad_exclusion(updates: Dict, z: float = 3.0,
                           low: float = 0.15, high: float = 3.0) -> Set:
    ids, mat = _matrix(updates)
    n = len(ids)
    if n <= 2:
        return set()
    norms = np.linalg.norm(mat, axis=1)
    med_norm = float(np.median(norms)) + 1e-12
    centre = np.median(mat, axis=0)
    dist = np.linalg.norm(mat - centre[None, :], axis=1)
    med_d = float(np.median(dist))
    mad = 1.4826 * float(np.median(np.abs(dist - med_d))) + 1e-12
    out = set()
    for i in range(n):
        ratio = norms[i] / med_norm
        if (dist[i] - med_d) / mad > z or ratio > high or ratio < low:
            out.add(ids[i])
    return out


def no_detection(updates: Dict) -> Set:
    return set()


ALTERNATIVES = {
    "krum": krum_exclusion,
    "cosine_median": cosine_median_exclusion,
    "distance_mad": distance_mad_exclusion,
    "no_detection": no_detection,
}


def shadow_decisions(updates: Dict) -> Dict[str, Set]:
    """Run every alternative on one round of updates."""
    return {name: fn(updates) for name, fn in ALTERNATIVES.items()}


def score(excluded: Set, client_ids, malicious_ids) -> Dict[str, int]:
    """Confusion counts of one decision against the ground truth."""
    tp = fp = fn = tn = 0
    for c in client_ids:
        mal = c in malicious_ids
        ex = c in excluded
        if ex and mal:
            tp += 1
        elif ex and not mal:
            fp += 1
        elif mal:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
