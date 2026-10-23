"""Do-no-harm exclusion rule for federated model updates.

Motivation (measured, see ``docs/DNH_DETECTOR.md``).  The deployed detector
scores every client by robust z-scores against its peers.  A z-score is
RELATIVE: as honest updates become more alike during training the population
spread shrinks and a perfectly benign client becomes a "3-sigma outlier".  That
is why its false-exclusion rate grows as training converges.

This rule never judges a client by how unusual it is.  It scores harm with two
ABSOLUTE statistics and excludes only on persistent evidence:

* ``ratio``   - the client's update norm relative to the median norm.
* ``cos_loo`` - cosine of the client's update with the coordinate-wise trimmed
                mean of ALL OTHER clients' updates (leave-one-out, so a client
                cannot pull the reference towards itself).

A logistic score ``s = b0 + b1*log2(ratio) - b2*cos_loo`` (weights fitted on
held-out training traces, never on the evaluation seeds) is smoothed over rounds
with an exponentially-weighted average, and a client is excluded only while the
smoothed evidence exceeds a threshold.  Two guards bound the damage a false
alarm can do: updates that are exactly (or almost) zero are free riders and are
excluded by an absolute norm test, and at most ``max_exclude_fraction`` of the
participants can be excluded in a round (the honest-majority assumption every
robust aggregator makes), highest evidence first.

The detector never reads a role label.
"""

import numpy as np

#: Fitted logistic weights (see experiments/fit_dnh.py); log-odds of "poisoned".
DEFAULT_WEIGHTS = {"b0": -7.078555, "b_ratio": 7.278349, "b_cos": 7.672823}
#: 99.5th percentile of the smoothed evidence honest clients reach on the training
#: traces (a 0.5 % false-positive budget); see experiments/fit_dnh.py.
DEFAULT_THRESHOLD = -0.357836
#: Label-free calibration of CalibratedDetector: 99.5th percentile of the honest norm ratio and
#: 0.5th percentile of the honest leave-one-out cosine on attacker-free runs (seeds 1000-1001).
CALIBRATION = {"ratio_hi": 2.047, "cos_lo": -0.245}


def _loo_trimmed_means(live_mat, trim):
    """Row i -> coordinate-wise trimmed mean of all OTHER rows, for every row at once.

    One sort of the (m, d) matrix plus prefix sums replaces m separate sorts:
    removing the element of rank r from a sorted column and summing the
    trimmed window is two prefix-sum differences.
    """
    m, d = live_mat.shape
    k = max(1, int(trim * (m - 1)))
    if m - 1 <= 2 * k:
        k = 0
    S = np.sort(live_mat, axis=0)
    # rank of each element within its column = number of strictly smaller entries
    rank = np.empty((m, d), dtype=np.int64)
    for i in range(m):
        rank[i] = (live_mat < live_mat[i]).sum(axis=0)
    P = np.vstack([np.zeros((1, d), dtype=S.dtype), np.cumsum(S, axis=0, dtype=np.float64)])          # P[t] = sum S[:t]
    lo, hi = k, m - 2 - k                                            # window in the others-list, inclusive
    cnt = hi - lo + 1
    r = rank                                                         # (m, d)
    # others[j] = S[j] for j < r, S[j+1] for j >= r
    a_hi = np.minimum(r - 1, hi)                                     # part 1: j in [lo, a_hi]
    part1 = np.where(a_hi >= lo, np.take_along_axis(P, (a_hi + 1).clip(0, m), 0) - P[lo][None, :], 0.0)
    b_lo = np.maximum(r, lo)                                         # part 2: j in [b_lo, hi] -> S[j+1]
    part2 = np.where(b_lo <= hi, np.take_along_axis(P, (hi + 2) * np.ones_like(r), 0)
                     - np.take_along_axis(P, (b_lo + 1).clip(0, m), 0), 0.0)
    return (part1 + part2) / cnt


def loo_features(matrix, trim=0.25):
    """Per-row ``(ratio, cos_loo, norms, live)`` for an ``(n, d)`` matrix of updates."""
    n = matrix.shape[0]
    norms = np.linalg.norm(matrix, axis=1)
    live = norms > 1e-9
    med = float(np.median(norms[live])) if live.any() else 0.0
    ratio = norms / max(med, 1e-12)
    cos = np.zeros(n)
    idx = np.where(live)[0]
    if len(idx) >= 4:
        sub = matrix[idx]
        ref = _loo_trimmed_means(sub, trim)
        num = (sub * ref).sum(axis=1)
        cos[idx] = num / (norms[idx] * np.linalg.norm(ref, axis=1) + 1e-12)
    return ratio, cos, norms, live


class CalibratedDetector:
    """Label-free variant ("do-no-harm v2"): both statistics are judged against the range
    honest clients reach, measured on ATTACKER-FREE runs only, so no attack is needed to
    fit it and a direction-only (norm-matched) attacker is covered.

    A client's margin is ``max(log2(ratio) - log2(R), (C - cos_loo) / w_cos)`` where ``R`` is a
    high quantile of the honest norm ratio and ``C`` a low quantile of the honest cosine; a
    positive margin means "outside what honest clients do".  The margin is smoothed over
    rounds (EWMA); a client is excluded while the smoothed margin is positive.  Free-rider
    and cap guards are as in ``DoNoHarmDetector``.
    """

    def __init__(self, ratio_hi, cos_lo, w_cos=0.25, ewma=0.5, max_exclude_fraction=0.45,
                 free_rider_ratio=0.15, min_participants=4):
        self.ratio_hi, self.cos_lo, self.w_cos = float(ratio_hi), float(cos_lo), float(w_cos)
        self.ewma = float(ewma)
        self.max_frac = float(max_exclude_fraction)
        self.free_rider_ratio = float(free_rider_ratio)
        self.min_participants = int(min_participants)
        self.evidence = {}
        self.last = {}

    def margin(self, ratio, cos):
        return np.maximum(np.log2(np.maximum(ratio, 1e-6)) - np.log2(self.ratio_hi), (self.cos_lo - cos) / self.w_cos)

    def decide(self, updates):
        ids = list(updates)
        if len(ids) < self.min_participants:
            return set(), {}
        mat = np.array([np.asarray(updates[c], dtype=float) for c in ids])
        ratio, cos, norms, live = loo_features(mat)
        ev, blatant = {}, set()
        for i, c in enumerate(ids):
            if (not live[i]) or ratio[i] < self.free_rider_ratio:
                blatant.add(c)
                ev[c] = float("inf")
                continue
            m = float(self.margin(ratio[i], cos[i]))
            prev = self.evidence.get(c)
            e = m if prev is None else self.ewma * prev + (1.0 - self.ewma) * m
            self.evidence[c] = e
            ev[c] = e
        flagged = sorted([c for c in ids if ev[c] > 0.0], key=lambda c: -ev[c])
        cap = int(self.max_frac * len(ids))
        excluded = set(flagged[:cap]) | (blatant if len(blatant) <= cap else set(sorted(blatant)[:cap]))
        self.last = {"ratio": dict(zip(ids, ratio)), "cos": dict(zip(ids, cos)), "evidence": ev}
        return excluded, ev


class DoNoHarmDetector:
    def __init__(self, weights=None, ewma=0.5, threshold=DEFAULT_THRESHOLD, max_exclude_fraction=0.45,
                 free_rider_ratio=0.15, min_participants=4):
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.ewma = float(ewma)
        self.threshold = float(threshold)
        self.max_frac = float(max_exclude_fraction)
        self.free_rider_ratio = float(free_rider_ratio)
        self.min_participants = int(min_participants)
        self.evidence = {}
        self.last = {}

    def score(self, ratio, cos):
        return self.w["b0"] + self.w["b_ratio"] * np.log2(np.maximum(ratio, 1e-6)) - self.w["b_cos"] * cos

    def decide(self, updates):
        """``{cid: flat_update}`` -> ``(excluded_set, {cid: evidence})``."""
        ids = list(updates)
        if len(ids) < self.min_participants:
            return set(), {}
        mat = np.array([np.asarray(updates[c], dtype=float) for c in ids])
        ratio, cos, norms, live = loo_features(mat)
        ev = {}
        blatant = set()
        for i, c in enumerate(ids):
            if (not live[i]) or ratio[i] < self.free_rider_ratio:
                blatant.add(c)
                ev[c] = float("inf")
                continue
            s = float(self.score(ratio[i], cos[i]))
            prev = self.evidence.get(c)
            e = s if prev is None else self.ewma * prev + (1.0 - self.ewma) * s
            self.evidence[c] = e
            ev[c] = e
        flagged = sorted([c for c in ids if ev[c] > self.threshold], key=lambda c: -ev[c])
        cap = int(self.max_frac * len(ids))
        excluded = set(flagged[:cap]) | (blatant if len(blatant) <= cap else set(sorted(blatant)[:cap]))
        self.last = {"ratio": dict(zip(ids, ratio)), "cos": dict(zip(ids, cos)), "evidence": ev}
        return excluded, ev
