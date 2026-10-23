"""
Unified Trust Engine for VANET Federated Reinforcement Learning.
Combines:
1. Behavioral Trust Tracker T_behaviour
2. Model-Update Trust Evaluator T_model
Formula:
T_final_i(t) = alpha * T_behaviour_i(t) + (1 - alpha) * T_model_i(t)
Implements adaptive exclusion threshold tau and tracks security metrics.
"""

import numpy as np
from trust.trust_model import BehavioralTrustTracker, _canonical_id
from trust.model_trust import ModelTrustEvaluator

class TrustEngine:
    def __init__(self, alpha=None, exclusion_threshold=None, config=None):
        if config is None:
            try:
                from configs.config_loader import load_config
                config = load_config()
            except Exception:
                config = None

        cfg_tm = config.get("trust_management", {}) if config else {}
        
        self.alpha = float(alpha) if alpha is not None else float(cfg_tm.get("alpha", 0.50))
        self.exclusion_threshold = float(exclusion_threshold) if exclusion_threshold is not None else float(cfg_tm.get("exclusion_threshold_tau", 0.40))
        
        eta = float(cfg_tm.get("temporal_decay_eta", 0.85))
        w = cfg_tm.get("behavioral_weights", {})
        w_b = float(w.get("historical_b", 0.30))
        w_r = float(w.get("reliability_r", 0.35))
        w_a = float(w.get("availability_a", 0.15))
        w_c = float(w.get("communication_c", 0.20))
        init_t = float(cfg_tm.get("initial_trust_prior", 0.70))
        
        self.behavior_tracker = BehavioralTrustTracker(eta=eta, w_b=w_b, w_r=w_r, w_a=w_a, w_c=w_c, default_trust=init_t)
        self.model_evaluator = ModelTrustEvaluator()
        
        self.final_trust_scores = {}
        self.attack_eval_records = []

    def compute_joint_trust(self, node_id, model_trust_score=None):
        """
        Calculates combined trust for a node using canonical ID.
        T_final = alpha * T_beh + (1 - alpha) * T_model.
        Respects ablations:
        - alpha = 1.0 -> purely behavioral trust
        - alpha = 0.0 -> purely model-update trust
        """
        canon_id = _canonical_id(node_id)
        t_beh = self.behavior_tracker.get_trust(canon_id)
        
        if model_trust_score is None or self.alpha >= 1.0:
            # Behavioral trust only (or before any model round occurs)
            t_final = t_beh
        elif self.alpha <= 0.0:
            # Model-update trust only (behavioral trust disabled)
            t_final = model_trust_score
        else:
            # Joint trust combination
            t_final = self.alpha * t_beh + (1.0 - self.alpha) * model_trust_score
            if model_trust_score <= 0.20:
                t_final = min(t_final, 0.30)
                self.behavior_tracker.penalize_unresponsive(canon_id, penalty=0.35)
            
        t_final = float(np.clip(t_final, 0.05, 1.0))
        self.final_trust_scores[canon_id] = t_final
        return t_final

    def filter_trusted_participants(self, client_updates_dict, ground_truth_roles=None):
        """
        Evaluates model updates, computes joint trust, and filters out malicious/untrusted nodes.
        Returns:
        - trusted_weights_dict: {cid: final_trust_score} for accepted nodes
        - excluded_clients: list of rejected client IDs
        - metrics: dict containing TPR, FPR, FNR detection metrics
        """
        model_scores = self.model_evaluator.evaluate_model_updates(client_updates_dict)
        
        trusted_weights = {}
        excluded = []
        
        tp = 0  # correctly identified attacker
        fp = 0  # honest node mistakenly excluded
        tn = 0  # honest node correctly kept
        fn = 0  # attacker went undetected
        unreliable_total = 0
        unreliable_excluded = 0
        
        for cid in client_updates_dict.keys():
            canon_id = _canonical_id(cid)
            m_score = model_scores.get(cid, 0.5)
            j_trust = self.compute_joint_trust(canon_id, m_score)
            
            is_excluded = j_trust < self.exclusion_threshold
            
            if ground_truth_roles:
                actual_role = ground_truth_roles.get(cid, ground_truth_roles.get(canon_id, None))
                if actual_role is None and isinstance(cid, int):
                    actual_role = ground_truth_roles.get(f"V{cid}", "honest")
                # An "unreliable" vehicle is an HONEST participant of the federated
                # round (its model update is genuine); it is a behavioural, not a
                # model-poisoning, threat and must not count as a missed attacker.
                actual_malicious = (actual_role is not None and actual_role not in ("honest", "unreliable"))
                if actual_role == "unreliable":
                    unreliable_total += 1
                    if is_excluded:
                        unreliable_excluded += 1
                
                if is_excluded and actual_malicious:
                    tp += 1
                elif is_excluded and not actual_malicious:
                    fp += 1
                elif not is_excluded and not actual_malicious:
                    tn += 1
                elif not is_excluded and actual_malicious:
                    fn += 1
                    
            if is_excluded:
                excluded.append(cid)
            else:
                trusted_weights[cid] = j_trust
                
        total_mal = (tp + fn)
        total_hon = (fp + tn)
        
        # Return None/N/A when metric is mathematically undefined (no positive/negative cases)
        tpr = (tp / total_mal) if total_mal > 0 else None
        fpr = (fp / total_hon) if total_hon > 0 else None
        fnr = (fn / total_mal) if total_mal > 0 else None
        
        metrics = {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "attack_detection_rate": round(float(tpr), 4) if tpr is not None else None,
            "false_positive_rate": round(float(fpr), 4) if fpr is not None else None,
            "false_negative_rate": round(float(fnr), 4) if fnr is not None else None,
            "total_clients": len(client_updates_dict),
            "excluded_count": len(excluded),
            "unreliable_total": unreliable_total,
            "unreliable_excluded": unreliable_excluded,
        }
        self.attack_eval_records.append(metrics)
        return trusted_weights, excluded, metrics

    def get_all_trust_scores(self):
        return {k: round(v, 3) for k, v in self.final_trust_scores.items()}
