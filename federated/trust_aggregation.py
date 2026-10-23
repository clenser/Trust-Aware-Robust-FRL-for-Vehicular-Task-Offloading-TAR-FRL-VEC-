"""
Federated Aggregation Implementations:
1. Standard FedAvg (Baseline):
   theta_global = sum(n_i / N * theta_i)
2. Trust-Weighted Robust Aggregation (Proposed):
   theta_global = sum_{i in Trusted} (T_i * n_i * theta_i) / sum_{i in Trusted} (T_i * n_i)
"""

import copy
import torch

class FederatedAggregator:
    @staticmethod
    def standard_fedavg(client_models_dict, sample_sizes_dict):
        """Standard FedAvg without trust filtering."""
        if not client_models_dict:
            return None
            
        total_samples = sum(sample_sizes_dict.values())
        if total_samples <= 0:
            total_samples = len(client_models_dict)
            sample_sizes_dict = {k: 1 for k in client_models_dict.keys()}
            
        first_cid = list(client_models_dict.keys())[0]
        aggregated_dict = copy.deepcopy(client_models_dict[first_cid])
        
        # Zero out aggregated dict
        for k in aggregated_dict.keys():
            aggregated_dict[k] = torch.zeros_like(aggregated_dict[k], dtype=torch.float32)
            
        for cid, model_state in client_models_dict.items():
            weight = sample_sizes_dict[cid] / total_samples
            for k in aggregated_dict.keys():
                aggregated_dict[k] += weight * model_state[k].float()
                
        # Cast back to original dtype
        for k in aggregated_dict.keys():
            aggregated_dict[k] = aggregated_dict[k].to(client_models_dict[first_cid][k].dtype)
            
        return aggregated_dict

    @staticmethod
    def trust_weighted_aggregation(client_models_dict, sample_sizes_dict, trust_weights_dict):
        """
        Trust-Aware Robust Aggregation (Equation 152 / 418 in Roadmap):
        theta_global = sum(T_i * n_i * theta_i) / sum(T_i * n_i)
        Only clients in trust_weights_dict are aggregated.
        """
        if not trust_weights_dict or not client_models_dict:
            return None
            
        # Calculate denominator: sum(T_i * n_i)
        total_trust_mass = sum(trust_weights_dict[cid] * max(1, sample_sizes_dict.get(cid, 1)) 
                               for cid in trust_weights_dict.keys() if cid in client_models_dict)
        
        if total_trust_mass <= 0:
            # Fallback if all excluded: keep previous global model or take highest trust
            best_cid = max(trust_weights_dict.keys(), key=lambda c: trust_weights_dict[c])
            return copy.deepcopy(client_models_dict[best_cid])
            
        first_cid = list(trust_weights_dict.keys())[0]
        aggregated_dict = copy.deepcopy(client_models_dict[first_cid])
        
        for k in aggregated_dict.keys():
            aggregated_dict[k] = torch.zeros_like(aggregated_dict[k], dtype=torch.float32)
            
        for cid, t_score in trust_weights_dict.items():
            if cid not in client_models_dict:
                continue
            effective_weight = (t_score * max(1, sample_sizes_dict.get(cid, 1))) / total_trust_mass
            for k in aggregated_dict.keys():
                aggregated_dict[k] += effective_weight * client_models_dict[cid][k].float()
                
        for k in aggregated_dict.keys():
            aggregated_dict[k] = aggregated_dict[k].to(client_models_dict[first_cid][k].dtype)
            
        return aggregated_dict
