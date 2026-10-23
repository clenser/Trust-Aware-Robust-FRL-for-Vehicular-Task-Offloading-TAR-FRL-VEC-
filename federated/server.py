"""
Federated Server for VANET Task Offloading Coordination.
Handles:
1. Model distribution.
2. Collecting client model updates.
3. Trust evaluation and adversarial filtering via TrustEngine.
4. Robust trust-weighted aggregation.
5. Baseline aggregation (FedAvg, Krum, Trimmed-Mean).
"""

import time
import copy
import numpy as np
import torch
from rl.dqn import DuelingQNetwork
from federated.trust_aggregation import FederatedAggregator
from trust.trust_engine import TrustEngine
from baselines.heuristics import RobustFLBaselines
from trust import alt_detectors
from trust.dnh_detector import DoNoHarmDetector
from trust.trust_model import _canonical_id

class FederatedServer:
    def __init__(self, state_dim=20, action_dim=5, aggregation_strategy="trust_aware", alpha=None, exclusion_threshold=None, trust_engine=None, config=None, krum_byzantine_count=None, krum_byzantine_fraction=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.aggregation_strategy = aggregation_strategy
        # The assumed number of Byzantine clients for Multi-Krum.  ``None``
        # means "derive it from the realised malicious fraction"; an integer
        # overrides it explicitly (used for sensitivity analysis).
        self.krum_byzantine_count = krum_byzantine_count
        self.krum_byzantine_fraction = float(krum_byzantine_fraction)
        
        self.global_network = DuelingQNetwork(state_dim, action_dim)
        self.global_weights = copy.deepcopy(self.global_network.state_dict())
        
        if trust_engine is not None:
            self.trust_engine = trust_engine
            if alpha is not None:
                self.trust_engine.alpha = float(alpha)
            if exclusion_threshold is not None:
                self.trust_engine.exclusion_threshold = float(exclusion_threshold)
        else:
            self.trust_engine = TrustEngine(alpha=alpha, exclusion_threshold=exclusion_threshold, config=config)
            
        self.alpha = self.trust_engine.alpha
        self.exclusion_threshold = self.trust_engine.exclusion_threshold
        
        self.current_round = 0
        self.round_history = []
        # Shadow evaluation of alternative detectors (never affects the model).
        self.shadow_detectors = False
        # "none" or "discount" (weight / (1 + staleness)); see run_aggregation_round.
        self.staleness_mode = "none"
        # Do-no-harm exclusion rule (strategy "do_no_harm"); carries per-client evidence across rounds.
        self.dnh = DoNoHarmDetector()

    def get_global_weights(self):
        return copy.deepcopy(self.global_weights)

    def run_aggregation_round(self, client_models_dict, client_samples_dict, ground_truth_roles=None,
                              staleness=None):
        """
        Executes one federated round.
        client_models_dict: {cid: weights_state_dict}
        staleness: optional {cid: rounds since the client last received the global
            model}.  With ``staleness_mode == "discount"`` the aggregation weight
            (the sample count) of a client is divided by ``1 + staleness``.
        """
        start_time = time.time()
        self.current_round += 1
        mean_staleness = None
        if staleness:
            mean_staleness = float(np.mean([staleness.get(c, 0) for c in client_models_dict]))
            if self.staleness_mode == "discount":
                client_samples_dict = {c: n / (1.0 + max(0, staleness.get(c, 0)))
                                       for c, n in client_samples_dict.items()}
        
        # Extract flat delta vectors for model trust evaluation
        flat_deltas = {}
        for cid, weights in client_models_dict.items():
            flat_deltas[cid] = self.trust_engine.model_evaluator.extract_flat_update(weights, self.global_weights)

        metrics = {}
        if self.shadow_detectors and ground_truth_roles is not None and flat_deltas:
            malicious = {c for c in flat_deltas
                         if ground_truth_roles.get(c, "honest") not in ("honest", "unreliable")}
            shadow = {}
            for name, excluded_set in alt_detectors.shadow_decisions(flat_deltas).items():
                shadow[name] = alt_detectors.score(excluded_set, flat_deltas.keys(), malicious)
            metrics["shadow_detectors"] = shadow

        if self.aggregation_strategy == "oracle_exclusion":
            # ANALYSIS ONLY: an upper bound that removes the truly malicious
            # clients using the ground-truth role.  It is not a deployable
            # method; it bounds what ANY detector could achieve downstream.
            keep = {c: w for c, w in client_models_dict.items()
                    if (ground_truth_roles or {}).get(c, "honest") in ("honest", "unreliable")}
            keep_samples = {c: client_samples_dict[c] for c in keep}
            new_global = FederatedAggregator.standard_fedavg(keep, keep_samples) if keep else None
            metrics.update({
                "defense_type": "oracle_exclusion_upper_bound",
                "performs_client_detection": False,
                "attack_detection_rate": None, "false_positive_rate": None,
                "false_negative_rate": None,
                "excluded_count": len(client_models_dict) - len(keep),
                "excluded_clients": [int(c) for c in client_models_dict if c not in keep],
                "accepted_clients": [int(c) for c in keep],
                "total_clients": len(client_models_dict)})
        elif self.aggregation_strategy == "do_no_harm":
            # Persistent-evidence exclusion on absolute harm statistics (see
            # trust/dnh_detector.py), followed by plain FedAvg of the survivors.
            excluded_set, evidence = self.dnh.decide(flat_deltas)
            keep = {c: w for c, w in client_models_dict.items() if c not in excluded_set}
            keep_samples = {c: client_samples_dict[c] for c in keep}
            new_global = FederatedAggregator.standard_fedavg(keep, keep_samples) if keep else None
            for c, e in evidence.items():
                # Display score only (1 = trusted): logistic of the negative evidence.
                self.trust_engine.final_trust_scores[_canonical_id(c)] = float(
                    0.05 if not np.isfinite(e) else np.clip(1.0 / (1.0 + np.exp(np.clip(e, -30, 30))), 0.05, 1.0))
            tp = fp = tn = fn = 0
            if ground_truth_roles:
                for cid in client_models_dict:
                    mal = ground_truth_roles.get(cid, "honest") not in ("honest", "unreliable")
                    ex = cid in excluded_set
                    tp += int(ex and mal); fp += int(ex and not mal)
                    tn += int((not ex) and (not mal)); fn += int((not ex) and mal)
            tpr = tp / (tp + fn) if (tp + fn) else None
            fpr = fp / (fp + tn) if (fp + tn) else None
            metrics.update({
                "defense_type": "do_no_harm_exclusion", "performs_client_detection": True,
                "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
                "attack_detection_rate": round(float(tpr), 4) if tpr is not None else None,
                "false_positive_rate": round(float(fpr), 4) if fpr is not None else None,
                "false_negative_rate": round(float(1 - tpr), 4) if tpr is not None else None,
                "excluded_count": len(excluded_set),
                "excluded_clients": [int(c) for c in excluded_set],
                "accepted_clients": [int(c) for c in keep],
                "total_clients": len(client_models_dict)})
        elif self.aggregation_strategy == "trust_aware":
            # 1. Proposed Trust-Aware Robust Aggregation (Joint Trust + Threshold Exclusion)
            trusted_weights, excluded, det_metrics = self.trust_engine.filter_trusted_participants(
                flat_deltas, ground_truth_roles=ground_truth_roles
            )
            new_global = FederatedAggregator.trust_weighted_aggregation(
                client_models_dict, client_samples_dict, trusted_weights
            )
            metrics.update(det_metrics)
            # Report the authoritative list of excluded clients so that the
            # caller can attribute the decision to individual clients.
            metrics["excluded_clients"] = [int(c) for c in excluded]
            metrics["accepted_clients"] = [int(c) for c in trusted_weights.keys()]
            metrics["performs_client_detection"] = True
            metrics["defense_type"] = "trust_aware_exclusion"

        elif self.aggregation_strategy == "exclusion_no_weighting":
            # Ablation: Hard exclusion of untrusted nodes, followed by uniform FedAvg
            trusted_weights, excluded, det_metrics = self.trust_engine.filter_trusted_participants(
                flat_deltas, ground_truth_roles=ground_truth_roles
            )
            surviving_models = {cid: client_models_dict[cid] for cid in trusted_weights.keys()}
            surviving_samples = {cid: client_samples_dict[cid] for cid in trusted_weights.keys()}
            new_global = FederatedAggregator.standard_fedavg(surviving_models, surviving_samples)
            metrics.update(det_metrics)
            metrics["excluded_clients"] = [int(c) for c in excluded]
            metrics["accepted_clients"] = [int(c) for c in trusted_weights.keys()]
            metrics["performs_client_detection"] = True
            metrics["defense_type"] = "exclusion_without_weighting"
            
        elif self.aggregation_strategy == "trust_no_exclusion":
            # Ablation: Trust weights applied without hard exclusion.
            # This strategy performs NO client-level detection, so every
            # detection metric is mathematically undefined and must be
            # reported as None (N/A) rather than as a misleading zero.
            all_weights = {}
            for cid in client_models_dict.keys():
                m_score = self.trust_engine.model_evaluator.history_scores.get(cid, 0.5)
                all_weights[cid] = self.trust_engine.compute_joint_trust(cid, m_score)
            new_global = FederatedAggregator.trust_weighted_aggregation(
                client_models_dict, client_samples_dict, all_weights
            )
            metrics.update({
                "defense_type": "trust_weighting_without_exclusion",
                "performs_client_detection": False,
                "attack_detection_rate": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
                "true_positives": None,
                "false_positives": None,
                "true_negatives": None,
                "false_negatives": None,
                "excluded_count": 0,
                "excluded_clients": [],
                "accepted_clients": [int(c) for c in client_models_dict.keys()],
                "total_clients": len(client_models_dict),
            })
            
        elif self.aggregation_strategy == "coordinate_median":
            # Robust baseline: Coordinate-wise Median (implicit robust aggregation, no client labeling)
            new_global = RobustFLBaselines.coordinate_median(client_models_dict)
            metrics.update({
                "defense_type": "implicit_median_aggregation",
                "performs_client_detection": False,
                "attack_detection_rate": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
                "excluded_count": 0,
                "total_clients": len(client_models_dict)
            })
            
        elif self.aggregation_strategy == "trimmed_mean":
            # Robust baseline: Coordinate-wise Trimmed Mean (implicit robust aggregation, no client labeling)
            new_global = RobustFLBaselines.trimmed_mean(client_models_dict, beta=0.2)
            metrics.update({
                "defense_type": "implicit_trimmed_mean",
                "performs_client_detection": False,
                "attack_detection_rate": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
                "excluded_count": 0,
                "total_clients": len(client_models_dict)
            })
            
        elif self.aggregation_strategy == "multi_krum":
            # Robust baseline: Multi-Krum (explicit Byzantine exclusion).
            # The assumed number of Byzantine clients is configurable: the
            # default derives it from the *realised* malicious fraction of the
            # participating clients, so the assumption is not silently
            # mismatched with the actual attack ratio.
            if self.krum_byzantine_count is not None:
                num_byz = int(self.krum_byzantine_count)
            else:
                num_byz = max(1, int(round(len(client_models_dict) * self.krum_byzantine_fraction)))
            num_byz = min(num_byz, max(1, len(client_models_dict) - 3))
            new_global, excluded_cids = RobustFLBaselines.multi_krum(client_models_dict, client_samples_dict, num_byzantine=num_byz)
            
            tp = fp = tn = fn = 0
            if ground_truth_roles:
                for cid in client_models_dict.keys():
                    actual_role = ground_truth_roles.get(cid, "honest")
                    actual_malicious = (actual_role not in ("honest", "unreliable"))
                    is_excluded = cid in excluded_cids
                    if is_excluded and actual_malicious:
                        tp += 1
                    elif is_excluded and not actual_malicious:
                        fp += 1
                    elif not is_excluded and not actual_malicious:
                        tn += 1
                    elif not is_excluded and actual_malicious:
                        fn += 1
                total_mal = tp + fn
                total_hon = fp + tn
                tpr = (tp / total_mal) if total_mal > 0 else None
                fpr = (fp / total_hon) if total_hon > 0 else None
                fnr = (fn / total_mal) if total_mal > 0 else None
            else:
                tpr = fpr = fnr = None
                
            metrics.update({
                "defense_type": "multi_krum_exclusion",
                "performs_client_detection": True,
                "assumed_byzantine_clients": num_byz,
                "assumed_byzantine_fraction": round(num_byz / max(1, len(client_models_dict)), 4),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "attack_detection_rate": round(float(tpr), 4) if tpr is not None else None,
                "false_positive_rate": round(float(fpr), 4) if fpr is not None else None,
                "false_negative_rate": round(float(fnr), 4) if fnr is not None else None,
                "excluded_count": len(excluded_cids),
                "excluded_clients": [int(c) for c in excluded_cids],
                "total_clients": len(client_models_dict)
            })
            
        elif self.aggregation_strategy == "fedavg":
            # Standard FedAvg baseline (no trust weighting, no exclusion)
            new_global = FederatedAggregator.standard_fedavg(client_models_dict, client_samples_dict)
            metrics.update({
                "defense_type": "none",
                "performs_client_detection": False,
                "attack_detection_rate": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
                "excluded_count": 0,
                "total_clients": len(client_models_dict)
            })
            
        else:
            new_global = FederatedAggregator.standard_fedavg(client_models_dict, client_samples_dict)

        # Remember the previous global model so that the change produced by
        # this aggregation can be verified afterwards.
        previous_global = copy.deepcopy(self.global_weights)

        if new_global is not None:
            self.global_weights = new_global
            self.global_network.load_state_dict(new_global)

        elapsed = time.time() - start_time
        if mean_staleness is not None:
            metrics["mean_staleness"] = round(mean_staleness, 4)
        metrics["round"] = self.current_round
        metrics["aggregation_time_ms"] = round(elapsed * 1000.0, 2)
        metrics["avg_trust"] = float(np.mean(list(self.trust_engine.final_trust_scores.values()))) if self.trust_engine.final_trust_scores else 1.0
        
        # Report explicitly whether the aggregation actually produced a new
        # global model.  When every client was excluded, the aggregation yields
        # no model and the previous global weights remain in place; the caller
        # must be able to detect that and refuse to call the round a success.
        if new_global is None:
            metrics["aggregation_produced_model"] = False
            metrics["aggregation_note"] = (
                "The aggregation produced no new global model (no client update "
                "was accepted); the previous global model is unchanged."
            )
        else:
            metrics["aggregation_produced_model"] = True
            metrics["aggregation_note"] = ""
            metrics["model_changed"] = self._weights_differ(previous_global, new_global)
        
        self.round_history.append(metrics)
        return self.global_weights, metrics

    @staticmethod
    def _weights_differ(old_weights, new_weights) -> bool:
        """True when two state dicts differ numerically."""
        if old_weights is None or new_weights is None:
            return False
        if set(old_weights.keys()) != set(new_weights.keys()):
            return True
        for key in old_weights:
            a = old_weights[key]
            b = new_weights.get(key)
            if b is None:
                return True
            if hasattr(a, "shape") and hasattr(b, "shape"):
                if not torch.allclose(a.float(), b.float(), atol=1e-7):
                    return True
            elif a != b:
                return True
        return False
