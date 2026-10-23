"""
Dynamic behavioral trust model for vehicular edge computing.

Evaluates:
- Historical behaviour B_i(t)
- Task response reliability R_i(t)
- Resource availability A_i(t)
- Communication reliability C_i(t)

Temporal update:
    T_b(t+1) = eta * T_b(t) + (1 - eta) * O_i(t)

Prior versus observation
------------------------
A node that has never been contacted has **no observed history**.  The
tracker therefore keeps two clearly separated quantities per node:

``prior_successes`` / ``prior_total``
    *Prior parameters*, not measurements.  They encode the neutral starting
    belief that a previously unseen node has behaved acceptably (7/10 by
    default, i.e. 0.7).  They are identical for every node and carry no
    information about the node's role: the tracker never sees a ground-truth
    label.

``observed_successes`` / ``observed_total``
    *Actual measurements* recorded from real interactions.

The historical factor uses the prior as a Beta pseudo-count:

    B_i = (prior_successes + observed_successes)
          / (prior_total + observed_total)

so that B_i starts at the prior value and converges on the measured success
rate as evidence accumulates.  The prior's influence therefore decays with
evidence, and its weight can be varied explicitly (see
``prior_sensitivity``).

``last_sinr`` and ``last_utilization`` start as ``None`` (not yet measured);
until a first interaction is observed the documented prior values are used
for the communication and availability factors.
"""

import numpy as np


def _canonical_id(node_id):
    """Normalizes vehicle/node identifiers to canonical format (e.g. 0 -> 'V0', '0' -> 'V0', 'V0' -> 'V0')."""
    s = str(node_id).strip()
    if s.startswith("V") and s[1:].isdigit():
        return f"V{int(s[1:])}"
    elif s.isdigit():
        return f"V{int(s)}"
    return s


class BehavioralTrustTracker:
    def __init__(self, eta=0.85, w_b=0.30, w_r=0.35, w_a=0.15, w_c=0.20,
                 default_trust=0.70, prior_successes=7, prior_total=10,
                 prior_sinr_db=16.0, prior_utilization=0.25):
        self.eta = float(eta)  # Memory decay factor
        self.w_b = float(w_b)
        self.w_r = float(w_r)
        self.w_a = float(w_a)
        self.w_c = float(w_c)
        self.default_trust = float(default_trust)

        # --- Prior parameters (documented assumptions, NOT observations) ---
        self.prior_successes = float(prior_successes)
        self.prior_total = float(prior_total)
        self.prior_sinr_db = float(prior_sinr_db)
        self.prior_utilization = float(prior_utilization)

        # Node ID -> Trust Metrics Dictionary
        self.node_records = {}

    def _init_node(self, node_id):
        canon_id = _canonical_id(node_id)
        if canon_id not in self.node_records:
            self.node_records[canon_id] = {
                "trust_score": self.default_trust,   # neutral, role-agnostic prior
                # Prior parameters (identical for every node, no role leakage)
                "prior_successes": self.prior_successes,
                "prior_total": self.prior_total,
                # Real measurements
                "observed_successes": 0,
                "observed_total": 0,
                "recent_observations": [],
                "last_sinr": None,                   # not yet measured
                "last_utilization": None,            # not yet measured
            }
        return canon_id

    # ------------------------------------------------------------------
    # Factors
    # ------------------------------------------------------------------
    def _historical_factor(self, rec):
        """B_i from the prior pseudo-count plus the real observations."""
        successes = rec["prior_successes"] + rec["observed_successes"]
        total = rec["prior_total"] + rec["observed_total"]
        if total <= 0:
            return self.default_trust
        return float(successes / total)

    def _communication_factor(self, rec):
        """C_i from the last observed SINR (or the prior until one exists)."""
        sinr = rec.get("last_sinr")
        if sinr is None:
            sinr = self.prior_sinr_db
        # Typical good SINR >= 15 dB -> 1.0, 0 dB -> 0.2
        return float(np.clip((sinr + 5.0) / 25.0, 0.0, 1.0))

    def _availability_factor(self, rec):
        """A_i from the last observed utilization (or the prior)."""
        utilization = rec.get("last_utilization")
        if utilization is None:
            utilization = self.prior_utilization
        return float(np.clip(1.0 - utilization, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def record_interaction(self, node_id, success, delay_sec, deadline_sec,
                           channel_sinr, utilization=0.5):
        """
        Records a task offloading interaction with an edge node or peer vehicle.

        Uses solely observable metrics: outcome success, measured delay vs
        deadline, measurable channel SINR, and advertised node utilization.
        """
        canon_id = self._init_node(node_id)
        rec = self.node_records[canon_id]

        # 1. Record the real observation.
        rec["observed_total"] += 1
        if success and delay_sec <= deadline_sec:
            rec["observed_successes"] += 1
        b_score = self._historical_factor(rec)

        # 2. Response reliability R_i
        if not success:
            r_score = 0.0
        else:
            time_margin = max(0.0, (deadline_sec - delay_sec) / max(1e-4, deadline_sec))
            r_score = float(np.clip(0.5 + 0.5 * time_margin, 0.2, 1.0))

        # 3. Resource availability A_i
        a_score = self._availability_factor(rec)

        # 4. Communication reliability C_i (mapped from SINR)
        c_score = self._communication_factor(rec)

        # Current observation O_i(t)
        observation = (self.w_b * b_score +
                       self.w_r * r_score +
                       self.w_a * a_score +
                       self.w_c * c_score)

        # Temporal update with exponential moving average
        rec["trust_score"] = float(np.clip(
            self.eta * rec["trust_score"] + (1.0 - self.eta) * observation,
            0.05, 1.0
        ))
        rec["last_sinr"] = float(channel_sinr)
        rec["last_utilization"] = float(utilization)
        rec["recent_observations"].append({
            "success": bool(success),
            "delay_sec": float(delay_sec),
            "deadline_sec": float(deadline_sec),
            "sinr_db": float(channel_sinr),
            "utilization": float(utilization),
        })
        if len(rec["recent_observations"]) > 32:
            del rec["recent_observations"][:-32]
        return rec["trust_score"]

    def penalize_unresponsive(self, node_id, penalty=0.25):
        """Penalizes a node for timeout or unreachability."""
        canon_id = self._init_node(node_id)
        rec = self.node_records[canon_id]
        # An unreachable node counts as a *failed* observation.
        rec["observed_total"] += 1
        rec["trust_score"] = float(max(0.05, rec["trust_score"] - penalty))
        return rec["trust_score"]

    def get_trust(self, node_id):
        canon_id = self._init_node(node_id)
        return self.node_records[canon_id]["trust_score"]

    def get_record(self, node_id):
        """The full record (prior and observed parts separated)."""
        canon_id = self._init_node(node_id)
        return self.node_records[canon_id]

    def prior_sensitivity(self, node_id, n_observations=0, n_successes=None):
        """Trust after ``n_observations`` observations under the current prior.

        Used to document and test how much the prior influences the result
        before any real evidence exists.
        """
        rec = self.get_record(node_id)
        if n_successes is None:
            n_successes = n_observations
        successes = rec["prior_successes"] + n_successes
        total = rec["prior_total"] + n_observations
        return float(successes / total) if total > 0 else self.default_trust
