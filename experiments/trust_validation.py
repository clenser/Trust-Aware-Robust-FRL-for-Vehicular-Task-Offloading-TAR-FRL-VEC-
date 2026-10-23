"""
Trust-detector validation harness.

The trust engine is evaluated against *defined* scenarios in which the
ground truth is known:

* **honest IID clients** - all clients draw the same task distribution and
  train normally.  A detector must not exclude any of them.
* **honest non-IID clients** - each client draws a different task
  distribution.  The resulting gradient heterogeneity is the classic source
  of false positives in robust-FL literature, so this is the scenario that
  matters most for the false-positive rate.
* **benign outliers** - clients that legitimately produce unusual (but
  honest) updates, e.g. a client that trained on very few samples.
* **small legitimate updates** - an honest client whose update is far
  smaller than the population's.  The detector must not confuse "small"
  with "malicious".
* **intermittent participants** - clients that participate rarely and
  therefore contribute proportionally small updates.
* **adversarial clients** - sign-flip, scaling, noise and free-riding
  attackers, at *unseen* strengths as well as at the tuned strength.

Ground-truth isolation
----------------------
The decision function (:func:`compute_decisions`) does not accept, import or
otherwise have access to the client roles.  The ground truth is only used
*afterwards*, by :func:`score_decisions`, to classify decisions that have
already been taken.  This is a structural guarantee, not a convention.

Validation / held-out split
----------------------------
:data:`VALIDATION_SCENARIOS` is used for *threshold selection*.
:data:`HELDOUT_SCENARIOS` contains additional scenario *shapes* (unseen
attack strengths, stronger honest heterogeneity) and is evaluated with
independent seeds.  Held-out results are never fed back into the selection
(see ``experiments/calibrate_trust.py``).

Honest-client exclusion and malicious-client detection are measured
separately, and the pooled confusion counts are derived from the underlying
counts rather than from averages of per-round rates.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from trust.trust_engine import TrustEngine
from trust.model_trust import ModelTrustEvaluator


# ----------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------
def make_honest_updates(n_clients: int, dim: int = 64, seed: int = 0,
                        heterogeneity: str = "iid", scale: float = 1.0,
                        n_samples: int = 0,
                        shared_component: float = 0.6) -> Dict[int, np.ndarray]:
    """Honest client updates.

    ``heterogeneity``:
      * ``iid``     - every client draws from the same direction.
      * ``non_iid`` - every client draws its own direction but they all share
                       a common component of magnitude ``shared_component``.
                       This models the realistic case of *partially overlapping*
                       task partitions (a Dirichlet mixture over a small
                       number of task types always leaves a shared component).
      * ``disjoint``- the clients' directions are mutually orthogonal.  This is
                       the documented *worst case*: a sign-flipping attacker is
                       then statistically indistinguishable from another
                       honest client, because "the negative of a random
                       direction" is itself a random direction.  No detector
                       can separate them, and the harness records this rather
                       than hiding it.
      * ``benign``  - one client produces a small but legitimate update.
    """
    rng = np.random.RandomState(seed)
    common = rng.normal(size=dim)
    common /= np.linalg.norm(common) + 1e-12

    updates = {}
    for i in range(n_clients):
        if heterogeneity == "iid":
            direction = common
        elif heterogeneity == "non_iid":
            own = rng.normal(size=dim)
            own /= np.linalg.norm(own) + 1e-12
            direction = (shared_component * common
                         + np.sqrt(max(0.0, 1.0 - shared_component ** 2)) * own)
        elif heterogeneity == "disjoint":
            own = rng.normal(size=dim)
            own /= np.linalg.norm(own) + 1e-12
            direction = own
        elif heterogeneity == "benign":
            if i == 0:
                direction = common * 0.05     # a legitimate but small update
            else:
                direction = common
        else:
            raise ValueError(heterogeneity)

        magnitude = scale * (1.0 + 0.05 * rng.normal())
        if n_samples:
            # Fewer samples -> a noisier, larger update (honest but atypical).
            magnitude *= (1.0 / max(1.0, n_samples / 32.0)) ** 0.5
        updates[i] = direction * magnitude + rng.normal(scale=0.02, size=dim)
    return updates


def apply_attack(updates: Dict[int, np.ndarray], malicious: List[int],
                 attack: str, seed: int = 0, strength: float = 1.0) -> Dict[int, np.ndarray]:
    """Apply an attack to the selected clients.

    ``strength`` scales the attack:

    * ``sign_flip``  - ``strength`` is the *fraction of coordinates* that are
      inverted.  ``1.0`` (default) flips the whole update; ``0.25`` is a much
      weaker, partially-aligned attack.
    * ``scaling``    - the update is multiplied by ``4 * strength``.
    * ``noise``      - Gaussian noise with ``sigma = 2 * strength``.
    * ``free_riding``- the update is replaced by (almost) zero.
    """
    rng = np.random.RandomState(seed + 7919)
    out = {k: v.copy() for k, v in updates.items()}
    for cid in malicious:
        delta = out[cid]
        if attack == "sign_flip":
            if strength >= 1.0:
                out[cid] = -delta
            else:
                # Partial sign flip: only a fraction of the coordinates are
                # inverted, so the attack is only partially aligned with the
                # honest population.
                mask = rng.rand(*delta.shape) < strength
                out[cid] = np.where(mask, -delta, delta)
        elif attack == "scaling":
            out[cid] = delta * (4.0 * strength)
        elif attack == "noise":
            out[cid] = delta + rng.normal(scale=2.0 * strength, size=delta.shape)
        elif attack == "free_riding":
            out[cid] = np.zeros_like(delta) * 1e-6
        else:
            raise ValueError(attack)
    return out


# ----------------------------------------------------------------------
# Declarative scenario specification
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioSpec:
    """A fully declarative description of one validation scenario."""

    name: str
    heterogeneity: str = "non_iid"
    attack: Optional[str] = None
    attack_strength: float = 1.0
    attack_fraction: float = 0.25
    n_clients: int = 20
    shared_component: float = 0.6
    dim: int = 64
    # Intermittent participation: every ``scale_every``-th client
    # contributes a proportionally smaller update.
    scale_participants: float = 0.0
    scale_every: int = 4
    # Small legitimate updates: a few honest clients train on very few
    # samples, so their update is both smaller and noisier.
    small_legit_clients: int = 0
    small_legit_every: int = 5
    # Participation.  ``idle_clients`` honest clients hold no experience at
    # all and therefore upload the global model itself; they are
    # NON-PARTICIPANTS of the round.  ``free_rider_clients`` malicious
    # clients hold plenty of experience but upload nothing, which is the
    # free-riding attack and must stay inside the detection population.
    idle_clients: int = 0
    free_rider_clients: int = 0
    # Seed offset inside the seed base (keeps the historical seed mapping).
    seed_index: int = 1
    description: str = ""


def build_scenario(spec: ScenarioSpec, seed: int) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """Materialise a scenario: the honest population and the malicious set."""
    rng = np.random.RandomState(seed + 104729)
    n = spec.n_clients

    base = make_honest_updates(n, spec.dim, seed=seed,
                               heterogeneity=spec.heterogeneity,
                               shared_component=spec.shared_component)

    if spec.small_legit_clients:
        for k in range(spec.small_legit_clients):
            cid = (k * spec.small_legit_every) % n
            base[cid] = base[cid] * 0.30 + rng.normal(scale=0.05, size=spec.dim)

    if spec.scale_participants:
        for cid in range(0, n, spec.scale_every):
            base[cid] = base[cid] * spec.scale_participants

    n_malicious = max(1, int(round(n * spec.attack_fraction))) if spec.attack else 0
    malicious = list(range(n_malicious))

    if spec.attack:
        base = apply_attack(base, malicious, spec.attack, seed=seed,
                            strength=spec.attack_strength)

    # Participation structure.  The idle clients and the free riders both
    # upload a zero update, but they are different things: an idle client has
    # no experience to contribute, a free rider has experience and withholds
    # it.  The distinction is the client's *available experience*, which is a
    # role-independent participation statistic.
    idle = []
    free_riders = []
    if spec.idle_clients or spec.free_rider_clients:
        # The idle clients come first, then the free riders, so the two
        # populations never overlap.
        for k in range(spec.idle_clients):
            cid = n - 1 - k
            idle.append(cid)
            base[cid] = np.zeros_like(base[cid])
        for k in range(spec.free_rider_clients):
            cid = n - 1 - spec.idle_clients - k
            free_riders.append(cid)
            base[cid] = np.zeros_like(base[cid])
        # The free riders are the malicious clients of this scenario; the
        # idle clients are honest and never malicious.
        malicious = (list(free_riders)
                     + [m for m in malicious
                        if m not in free_riders and m not in idle])

    return base, malicious, idle


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def compute_decisions(updates: Dict[int, np.ndarray],
                      exclusion_threshold: float = 0.40,
                      alpha: float = 0.50,
                      exclude_non_participants: bool = True,
                      non_participants: Optional[List[int]] = None) -> Dict[int, Dict[str, object]]:
    """Take the exclusion decisions for a population of updates.

    This function is deliberately **role-agnostic**: it has no parameter,
    import or attribute through which the ground truth could reach it.  The
    decisions it returns are exactly the decisions the live system would
    take.
    """
    engine = TrustEngine(alpha=alpha, exclusion_threshold=exclusion_threshold)

    # Participation rule (mirrors the benchmark and the live console): a
    # client that holds fewer transitions than the minimum training batch
    # could not perform a gradient step, so it is a non-participant of the
    # round and is not judged.  The rule uses the client's available
    # experience, never its role.  A free rider holds plenty of experience
    # and therefore stays inside the detection population, where the
    # zero-upload signature is exactly what the norm-ratio rule looks for.
    if exclude_non_participants and non_participants:
        idle = set(non_participants)
        updates = {cid: v for cid, v in updates.items() if cid not in idle}

    # The model trust evaluator works on flat delta vectors.
    scores = engine.model_evaluator.evaluate_model_updates(updates)

    decisions = {}
    for cid in updates:
        joint = engine.compute_joint_trust(f"V{cid}", scores.get(cid, 0.5))
        decisions[cid] = {
            "joint_trust": joint,
            "model_trust": scores.get(cid, 0.5),
            "excluded": joint < exclusion_threshold,
        }
    return decisions


def score_decisions(decisions: Dict[int, Dict[str, object]],
                    malicious: List[int]) -> Dict[str, int]:
    """Classify already-taken decisions against the ground truth."""
    malicious_set = set(malicious)
    tp = fp = fn = tn = 0
    for cid, d in decisions.items():
        excluded = bool(d["excluded"])
        is_malicious = cid in malicious_set
        if excluded and is_malicious:
            tp += 1
        elif excluded:
            fp += 1
        elif is_malicious:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_scenario(updates: Dict[int, np.ndarray],
                      malicious: List[int],
                      exclusion_threshold: float = 0.40,
                      alpha: float = 0.50,
                      seed: int = 0,
                      idle: Optional[List[int]] = None,
                      exclude_non_participants: bool = True) -> Dict[str, object]:
    """Run the trust filter on a scenario and derive the confusion counts.

    Returns the counts, the derived rates (None when undefined) and the
    per-client decisions.
    """
    decisions = compute_decisions(updates, exclusion_threshold=exclusion_threshold,
                                  alpha=alpha,
                                  exclude_non_participants=exclude_non_participants,
                                  non_participants=idle)
    confusion = score_decisions(decisions, malicious)
    return {
        "confusion": confusion,
        "rates": pooled_rates(**confusion),
        "decisions": decisions,
        "excluded": sorted(cid for cid, d in decisions.items() if d["excluded"]),
        "malicious": sorted(malicious),
        "non_participants": sorted(idle or []),
        "exclusion_threshold": exclusion_threshold,
        "alpha": alpha,
        "seed": seed,
        "exclude_non_participants": exclude_non_participants,
    }


def pooled_rates(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Optional[float]]:
    """Detection rates derived from the underlying confusion counts.

    A rate is ``None`` (N/A) when its denominator is zero, i.e. when the
    metric is mathematically undefined for this scenario.  A rate whose
    denominator is non-zero but whose numerator is zero is a *measured
    zero*, not a missing value.
    """
    def r(num, den):
        return (num / den) if den > 0 else None

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_decisions": tp + fp + fn + tn,
        "n_malicious": tp + fn,
        "n_honest": fp + tn,
        "tpr": r(tp, tp + fn),
        "fpr": r(fp, fp + tn),
        "fnr": r(fn, tp + fn),
        "precision": r(tp, tp + fp),
        "recall": r(tp, tp + fn),
        "accuracy": r(tp + tn, tp + fp + fn + tn),
    }


# ----------------------------------------------------------------------
# Scenario sets
# ----------------------------------------------------------------------
# The validation set is the ONLY set that may be used to select a
# threshold.  The seed indices preserve the historical seed mapping so the
# regression tests keep their meaning.
BASE_VALIDATION_SCENARIOS = (
    ScenarioSpec("honest_iid", heterogeneity="iid", seed_index=1,
                 description="All honest clients share one task direction."),
    ScenarioSpec("honest_non_iid", heterogeneity="non_iid", shared_component=0.6,
                 seed_index=2,
                 description="Partially overlapping task partitions."),
    ScenarioSpec("honest_benign_outlier", heterogeneity="benign", seed_index=3,
                 description="One honest client produces a legitimately small update."),
    ScenarioSpec("honest_varying_participation", heterogeneity="non_iid",
                 shared_component=0.6, scale_participants=0.4, scale_every=4,
                 seed_index=4,
                 description="Intermittent participants contribute smaller updates."),
    ScenarioSpec("honest_fully_disjoint", heterogeneity="disjoint", seed_index=6,
                 description="Documented worst case: mutually orthogonal honest updates."),
    ScenarioSpec("honest_small_legitimate_update", heterogeneity="non_iid",
                 shared_component=0.6, small_legit_clients=4, small_legit_every=5,
                 seed_index=7,
                 description="Honest clients trained on very few samples: small, noisy updates."),
    ScenarioSpec("honest_idle_clients", heterogeneity="non_iid",
                 shared_component=0.6, idle_clients=6, seed_index=8,
                 description="6 honest clients hold no experience: they upload the "
                             "global model and are non-participants, not attackers."),
    ScenarioSpec("honest_idle_majority", heterogeneity="non_iid",
                 shared_component=0.6, idle_clients=12, seed_index=9,
                 description="12 of 20 honest clients are non-participants: the "
                             "population reference degenerates if they are judged."),
    ScenarioSpec("adversarial_free_rider_with_idle", heterogeneity="non_iid",
                 shared_component=0.6, idle_clients=6, free_rider_clients=2,
                 seed_index=10,
                 description="2 free riders (experience withheld) next to 6 idle "
                             "honest clients: the free riders must stay detectable."),
)

# The held-out set contains scenario *shapes* that were never used for
# tuning, in particular attack strengths that differ from the tuned ones.
HELDOUT_SCENARIOS = (
    ScenarioSpec("honest_iid", heterogeneity="iid", seed_index=1,
                 description="Held-out seeds, all honest clients share one direction."),
    ScenarioSpec("honest_non_iid", heterogeneity="non_iid", shared_component=0.6,
                 seed_index=2, description="Held-out seeds, partially overlapping."),
    ScenarioSpec("honest_benign_outlier", heterogeneity="benign", seed_index=3,
                 description="Held-out seeds, legitimate small update."),
    ScenarioSpec("honest_varying_participation", heterogeneity="non_iid",
                 shared_component=0.6, scale_participants=0.4, scale_every=4,
                 seed_index=4, description="Held-out seeds, intermittent participation."),
    ScenarioSpec("honest_small_legitimate_update", heterogeneity="non_iid",
                 shared_component=0.6, small_legit_clients=4, small_legit_every=5,
                 seed_index=5, description="Held-out seeds, small legitimate updates."),
    ScenarioSpec("honest_fully_disjoint", heterogeneity="disjoint", seed_index=6,
                 description="Held-out seeds, mutually orthogonal honest updates."),
    ScenarioSpec("honest_non_iid_strong", heterogeneity="non_iid",
                 shared_component=0.3, seed_index=7,
                 description="Stronger honest heterogeneity than in validation."),
    ScenarioSpec("adversarial_sign_flip", heterogeneity="non_iid", shared_component=0.6,
                 attack="sign_flip", seed_index=8,
                 description="Full sign flip at the tuned strength, unseen seeds."),
    ScenarioSpec("adversarial_sign_flip_partial_50", heterogeneity="non_iid",
                 shared_component=0.6, attack="sign_flip", attack_strength=0.5,
                 seed_index=9,
                 description="UNSEEN strength: 50% of the coordinates are inverted."),
    ScenarioSpec("adversarial_sign_flip_partial_25", heterogeneity="non_iid",
                 shared_component=0.6, attack="sign_flip", attack_strength=0.25,
                 seed_index=10,
                 description="UNSEEN strength: 25% of the coordinates are inverted."),
    ScenarioSpec("adversarial_scaling_unseen_2x", heterogeneity="non_iid",
                 shared_component=0.6, attack="scaling", attack_strength=0.5,
                 seed_index=11,
                 description="UNSEEN strength: 2x scaling, below the norm-ratio alarm."),
    ScenarioSpec("adversarial_scaling_unseen_8x", heterogeneity="non_iid",
                 shared_component=0.6, attack="scaling", attack_strength=2.0,
                 seed_index=12, description="UNSEEN strength: 8x scaling."),
    ScenarioSpec("adversarial_noise_unseen_weak", heterogeneity="non_iid",
                 shared_component=0.6, attack="noise", attack_strength=0.25,
                 seed_index=13, description="UNSEEN strength: sigma = 0.5."),
    ScenarioSpec("adversarial_noise_unseen_strong", heterogeneity="non_iid",
                 shared_component=0.6, attack="noise", attack_strength=3.0,
                 seed_index=14, description="UNSEEN strength: sigma = 6.0."),
    ScenarioSpec("adversarial_free_riding", heterogeneity="non_iid",
                 shared_component=0.6, attack="free_riding", seed_index=15,
                 description="Free riding at the tuned strength, unseen seeds."),
    ScenarioSpec("honest_idle_clients", heterogeneity="non_iid",
                 shared_component=0.6, idle_clients=6, seed_index=16,
                 description="Held-out seeds, 6 honest non-participants."),
    ScenarioSpec("honest_idle_majority", heterogeneity="non_iid",
                 shared_component=0.6, idle_clients=12, seed_index=17,
                 description="Held-out seeds, a majority of honest non-participants."),
    ScenarioSpec("adversarial_free_rider_with_idle", heterogeneity="non_iid",
                 shared_component=0.6, idle_clients=6, free_rider_clients=2,
                 seed_index=18,
                 description="Held-out seeds, free riders next to idle honest clients."),
)


def validation_specs(attack: str = "sign_flip",
                     attack_strength: float = 1.0) -> List[ScenarioSpec]:
    """The validation scenario list for a given attack type."""
    specs = list(BASE_VALIDATION_SCENARIOS)
    specs.append(ScenarioSpec("adversarial_%s" % attack, heterogeneity="non_iid",
                               shared_component=0.6, attack=attack,
                               attack_strength=attack_strength, seed_index=5))
    return specs


def run_suite(specs, seed_base: int = 1, exclusion_threshold: float = 0.40,
              alpha: float = 0.50,
              exclude_non_participants: bool = True) -> Dict[str, Dict]:
    """Run a list of scenario specifications and return the per-scenario results."""
    results = {}
    for spec in specs:
        seed = seed_base * 100 + spec.seed_index
        updates, malicious, idle = build_scenario(spec, seed)
        res = evaluate_scenario(updates, malicious,
                                exclusion_threshold=exclusion_threshold,
                                alpha=alpha, seed=seed, idle=idle,
                                exclude_non_participants=exclude_non_participants)
        res["spec"] = spec
        results[spec.name] = res
    return results


def run_validation_suite(n_clients: int = 20, dim: int = 64,
                         exclusion_threshold: float = 0.40,
                         attack: str = "sign_flip",
                         attack_fraction: float = 0.25,
                         seed_base: int = 1,
                         shared_component: float = 0.6,
                         alpha: float = 0.50,
                         scenario_set: str = "validation",
                         exclude_non_participants: bool = True) -> Dict[str, Dict]:
    """Run the validation (or held-out) suite and return the per-scenario results.

    The seeds are derived from ``seed_base`` so that a *different* seed base
    produces a genuinely independent set of scenarios.  The detector is tuned
    on one seed base and evaluated on another.

    ``scenario_set="heldout"`` additionally includes attack strengths and
    honest-heterogeneity levels that were never used for tuning.
    """
    if scenario_set == "heldout":
        specs = HELDOUT_SCENARIOS
    elif scenario_set == "validation":
        specs = validation_specs(attack=attack)
    else:
        raise ValueError("scenario_set must be 'validation' or 'heldout'")

    # Apply the caller's population size / heterogeneity to every spec.
    specs = [ScenarioSpec(**{**spec.__dict__,
                             "n_clients": n_clients,
                             "dim": dim,
                             "attack_fraction": attack_fraction}) for spec in specs]
    return run_suite(specs, seed_base=seed_base,
                     exclusion_threshold=exclusion_threshold, alpha=alpha,
                     exclude_non_participants=exclude_non_participants)


def run_heldout_suite(n_clients: int = 20, dim: int = 64,
                      exclusion_threshold: float = 0.40,
                      attack_fraction: float = 0.25,
                      seed_base: int = 9001,
                      alpha: float = 0.50,
                      exclude_non_participants: bool = True) -> Dict[str, Dict]:
    """Run the held-out suite (never used for threshold selection)."""
    return run_validation_suite(n_clients=n_clients, dim=dim,
                                exclusion_threshold=exclusion_threshold,
                                attack_fraction=attack_fraction,
                                seed_base=seed_base, alpha=alpha,
                                scenario_set="heldout",
                                exclude_non_participants=exclude_non_participants)


def summarize(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Condense the suite into a comparable summary."""
    summary = {}
    for name, res in results.items():
        c = res["confusion"]
        summary[name] = {
            "excluded_honest": c["fp"],
            "detected_malicious": c["tp"],
            "missed_malicious": c["fn"],
            "honest_retained": c["tn"],
            "n_honest": c["fp"] + c["tn"],
            "n_malicious": c["tp"] + c["fn"],
            "tpr": res["rates"]["tpr"],
            "fpr": res["rates"]["fpr"],
            "fnr": res["rates"]["fnr"],
            "precision": res["rates"]["precision"],
            "description": res.get("spec").description if res.get("spec") else "",
        }
    return summary
