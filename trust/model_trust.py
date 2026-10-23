"""
Federated model-update trust evaluation.

Purpose
-------
Detect Byzantine model updates (sign flipping, gradient scaling, noise
injection, free riding) **without** punishing the genuine gradient
heterogeneity that non-IID task partitions produce.

Why a population-relative score
-------------------------------
The naive approach - "compare every update with the coordinate-wise median
and punish low cosine similarity" - fails on non-IID data.  When the clients
hold genuinely different task distributions their updates point in different
directions *by construction*, and a fixed directional threshold then
excludes honest clients en masse.  The validation suite in
``experiments/trust_validation.py`` measured exactly this failure: an
absolute directional rule excluded 19 of 20 honest non-IID clients.

The evaluator below therefore scores a client against the *population*:

* **Agreement with the crowd.**  For every client the *median* cosine
  similarity to all the other clients is computed.  A sign-flipping
  attacker is anti-correlated with the honest majority, so its median
  agreement is strongly negative, while an honest client - even in a
  heterogeneous population - has a non-negative median agreement.  The
  statistic needs no global reference direction, so a group of attackers
  cannot define one.
* **Robust z-scores.**  Every client statistic is converted into a z-score
  against the population median, using the median absolute deviation (MAD).
  A client is suspicious when it is an *extreme outlier among its peers*,
  not when it merely differs from them.
* **Scale checks.**  A norm far below the population median is free riding;
  a norm far above it is a scaling attack.

Identifiability limit
---------------------
When the honest population's update directions are *mutually orthogonal*,
"the negative of a random direction" is itself a random direction, so a
sign-flipping attacker is statistically indistinguishable from another
honest client.  No detector can separate them in that regime, and the
validation harness records this case explicitly instead of hiding it.
"""

import numpy as np
import torch
from scipy.special import expit as _expit   # overflow-free logistic 1 / (1 + exp(-x))


class ModelTrustEvaluator:
    """Scores model updates in [0, 1]; 1 = fully trusted, 0.05 = hostile.

    Parameters
    ----------
    trim_fraction:
        Kept for interface compatibility; the population-relative statistics
        no longer need a trimmed reference direction.
    z_threshold:
        A client is flagged as an extreme outlier when its robust z-score
        exceeds this value in magnitude.
    norm_low_ratio / norm_high_ratio:
        Free-riding and scaling thresholds, relative to the population median
        update norm.
    """

    def __init__(self, lambda_cos=0.55, lambda_dist=0.35, lambda_norm=0.10,
                 trim_fraction=0.25, z_threshold=3.0,
                 norm_low_ratio=0.15, norm_high_ratio=3.0,
                 history_weight=0.3):
        self.lambda_cos = lambda_cos
        self.lambda_dist = lambda_dist
        self.lambda_norm = lambda_norm
        self.trim_fraction = float(trim_fraction)
        self.z_threshold = float(z_threshold)
        self.norm_low_ratio = float(norm_low_ratio)
        self.norm_high_ratio = float(norm_high_ratio)
        self.history_weight = float(history_weight)
        self.history_scores = {}
        self.last_details = {}
        # True when the population reference had to be repaired because more
        # than half of the participants uploaded no usable update.
        self.last_population_degenerate = False

    # ------------------------------------------------------------------
    @staticmethod
    def extract_flat_update(local_state_dict, global_state_dict):
        """Flattened delta vector: Delta theta = theta_local - theta_global."""
        deltas = []
        for key in global_state_dict.keys():
            diff = (local_state_dict[key].float() - global_state_dict[key].float()).cpu().numpy().flatten()
            deltas.append(diff)
        return np.concatenate(deltas)

    @staticmethod
    def _robust_scale(values):
        """A robust scale estimate that does not degenerate on a contaminated
        population.

        The median absolute deviation (MAD) is the classic choice, but it
        collapses to zero whenever more than half of the population shares the
        same value - which is exactly the situation when a *minority* of the
        clients is scaled up, and it would hide the attack.  The normalised
        inter-quartile range stays informative for up to 25 % contamination.
        The smaller of the two available estimates is used, so a contaminated
        population cannot inflate the scale and hide the outliers.
        """
        v = np.asarray(values, dtype=float)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        q75, q25 = np.percentile(v, [75, 25])
        iqr = float(q75 - q25)

        if mad > 1e-12:
            return float(1.4826 * mad)
        if iqr > 1e-12:
            return float(iqr / 1.349)
        s = float(np.std(v))
        return s if s > 1e-12 else 0.0

    @classmethod
    def _robust_z(cls, values):
        """Median / robust-scale z-score of a 1-D sample."""
        values = np.asarray(values, dtype=float)
        med = float(np.median(values))
        scale = cls._robust_scale(values)
        if scale <= 0.0:
            return np.zeros_like(values), med, 0.0
        return (values - med) / scale, med, scale

    # ------------------------------------------------------------------
    def evaluate_model_updates(self, client_updates_dict):
        """
        Takes ``{client_id: flat_delta_vector}`` and returns
        ``{client_id: model_trust_score in [0.05, 1]}``.

        The scores are *population relative*: they express how extreme a
        client is among its peers, not how close it is to an absolute
        reference direction.
        """
        self.last_population_degenerate = False
        if not client_updates_dict:
            return {}

        client_ids = list(client_updates_dict.keys())
        matrix = np.array([np.asarray(client_updates_dict[cid], dtype=float)
                            for cid in client_ids])
        norms = np.linalg.norm(matrix, axis=1)
        n = len(client_ids)

        # Population statistics.
        #
        # The reference norm is the population median.  If that median is
        # degenerate (zero) the reference is meaningless: every client that
        # *did* upload something would obtain an unbounded norm ratio and
        # would be flagged as a scale outlier.  A zero median means more than
        # half of the participants uploaded no usable update, so the
        # reference is recomputed over the clients that did upload something.
        # In normal operation this branch is never taken and the behaviour is
        # identical to the plain population median.
        norm_median = float(np.median(norms)) + 1e-12
        if norm_median <= 1e-11:
            informative = norms[norms > 1e-11]
            if informative.size > 0:
                norm_median = float(np.median(informative)) + 1e-12
                self.last_population_degenerate = True
        norm_ratios = norms / norm_median

        # ---- Per-client agreement with the crowd ---------------------
        # For every client, the *median* cosine similarity to all the other
        # clients.  A sign-flipping attacker is anti-correlated with the
        # honest majority, so its median agreement is strongly negative,
        # while an honest client - even in a heterogeneous population - has a
        # non-negative median agreement.  The statistic needs no global
        # reference direction, so attackers cannot define one.
        denom = np.outer(norms, norms) + 1e-12
        pairwise = (matrix @ matrix.T) / denom
        np.fill_diagonal(pairwise, np.nan)
        with np.errstate(invalid="ignore"):
            agreement = np.nanmedian(pairwise, axis=1)
        agreement = np.nan_to_num(agreement, nan=0.0)

        # Robust, population-relative z-scores.
        agreement_z, _, _ = self._robust_z(agreement)

        # Distance from the population centre, in units of the population's
        # own spread (again a robust, population-relative measure).
        centre = np.median(matrix, axis=0)
        dist_all = np.linalg.norm(matrix - centre[None, :], axis=1)
        dist_z, _, _ = self._robust_z(dist_all)
        ratio_z, _, _ = self._robust_z(np.log10(norm_ratios + 1e-12))

        scores = {}
        details = {}
        for idx, cid in enumerate(client_ids):
            d_norm = float(norms[idx])
            ratio = float(norm_ratios[idx])
            a_z = float(agreement_z[idx])

            # 1. Free riding: the update is negligible next to the population.
            if d_norm < 1e-4 or ratio < self.norm_low_ratio:
                score = 0.05
                details[cid] = {"reason": "free_riding", "norm_ratio": ratio,
                                 "agreement": float(agreement[idx]),
                                 "agreement_z": a_z}
                self.history_scores[cid] = score
                scores[cid] = score
                continue

            # 2. Agreement with the crowd.  A strongly negative z-score means
            #    the client is anti-correlated with its peers.
            if a_z < -self.z_threshold:
                cos_score = 0.05
            elif a_z < -self.z_threshold * 0.5:
                cos_score = 0.25
            else:
                # Map the z-score onto [0.55, 1.0].
                cos_score = float(np.clip(0.55 + 0.45 * _expit(-a_z), 0.55, 1.0))

            # 3. Distance from the population centre, relative to the spread.
            if dist_z[idx] > self.z_threshold:
                dist_score = 0.15
            else:
                dist_score = float(np.exp(-0.5 * max(0.0, dist_z[idx]) ** 2))

            # 4. Norm sanity: scaling / gradient explosion.
            if ratio >= self.norm_high_ratio:
                norm_score = float(max(0.05, 1.0 / (abs(ratio - 1.0) + 1.0)))
            elif ratio < self.norm_low_ratio:
                norm_score = 0.25
            else:
                norm_score = 1.0

            m_trust = (self.lambda_cos * cos_score +
                       self.lambda_dist * dist_score +
                       self.lambda_norm * norm_score)

            # Hostile indicators.  An extreme outlier on two independent
            # axes at once is a high-confidence attack.  An extreme *scale*
            # outlier is hostile on its own: a scaled-up or collapsed update
            # is a gradient attack regardless of the direction it points in.
            extreme_agreement = a_z < -self.z_threshold
            extreme_scale = ratio_z[idx] > self.z_threshold
            extreme_distance = dist_z[idx] > self.z_threshold

            if extreme_agreement and (extreme_scale or extreme_distance):
                m_trust = min(m_trust, 0.15)
            elif extreme_scale:
                m_trust = min(m_trust, 0.15)
            elif extreme_agreement:
                m_trust = min(m_trust, 0.30)

            final_m_score = float(np.clip(m_trust, 0.05, 1.0))

            # Smooth with the historical model score, if one exists.
            if cid in self.history_scores:
                final_m_score = ((1.0 - self.history_weight) * final_m_score
                                 + self.history_weight * self.history_scores[cid])
            self.history_scores[cid] = final_m_score
            scores[cid] = final_m_score
            details[cid] = {"reason": "scored", "norm_ratio": ratio,
                             "agreement": float(agreement[idx]),
                             "agreement_z": a_z,
                             "dist_z": float(dist_z[idx]),
                             "ratio_z": float(ratio_z[idx])}

        self.last_details = details
        return scores
