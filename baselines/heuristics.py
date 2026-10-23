"""
Baseline Task Offloading Policies for Empirical Comparison:
1. Local Only Execution
2. Random Offloading
3. Nearest RSU Offloading
4. Greedy Resource-Aware Offloading
5. Multi-Krum Byzantine Robust Federated Aggregation
6. Coordinate-wise Trimmed Mean Aggregation
"""

import numpy as np
import copy
import torch

class LocalExecutionBaseline:
    @staticmethod
    def select_action(state, valid_actions=None):
        return 0  # Action 0 = Local execution

class RandomOffloadingBaseline:
    @staticmethod
    def select_action(state, num_actions=5):
        return np.random.randint(0, num_actions)

class NearestRSUBaseline:
    @staticmethod
    def select_action(state, rsu_distances):
        """
        rsu_distances: list of distances to RSU 1, 2, 3
        Actions: 1: RSU1, 2: RSU2, 3: RSU3
        """
        best_rsu = np.argmin(rsu_distances)
        return int(best_rsu + 1)

class ResourceAwareBaseline:
    @staticmethod
    def select_action(state, node_capacities, node_queues):
        """
        Picks the node with the lowest expected wait time (highest available compute).
        """
        avail_compute = [c / max(1, q + 1) for c, q in zip(node_capacities, node_queues)]
        best_idx = np.argmax(avail_compute)
        return int(best_idx)

class RobustFLBaselines:
    @staticmethod
    def multi_krum(client_models_dict, sample_sizes_dict, num_byzantine=1):
        """
        Multi-Krum Byzantine-robust FL baseline.
        Selects model updates closest to their neighbors in Euclidean distance.
        """
        cids = list(client_models_dict.keys())
        n = len(cids)
        if n <= 2:
            return client_models_dict[cids[0]], []
            
        flat_vectors = []
        first_cid = cids[0]
        ref_keys = list(client_models_dict[first_cid].keys())
        
        for cid in cids:
            flat = torch.cat([client_models_dict[cid][k].float().flatten() for k in ref_keys])
            flat_vectors.append(flat)
            
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flat_vectors[i] - flat_vectors[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
                
        # Krum score: sum of n - f - 2 smallest distances
        k_val = max(1, min(n - 2, n - num_byzantine - 2))
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            scores.append(np.sum(sorted_dists[1:k_val + 1]))
            
        # Multi-Krum: select m = n - num_byzantine clients with lowest scores and average them
        m_survivors = max(1, n - num_byzantine)
        ranked_indices = np.argsort(scores)
        accepted_indices = ranked_indices[:m_survivors]
        excluded_indices = ranked_indices[m_survivors:]
        
        accepted_cids = [cids[idx] for idx in accepted_indices]
        excluded_cids = [cids[idx] for idx in excluded_indices]
        
        first_cid = accepted_cids[0]
        aggregated = copy.deepcopy(client_models_dict[first_cid])
        for key in aggregated.keys():
            stacked = torch.stack([client_models_dict[cid][key].float() for cid in accepted_cids], dim=0)
            aggregated[key] = torch.mean(stacked, dim=0).to(client_models_dict[first_cid][key].dtype)
            
        return aggregated, excluded_cids

    @staticmethod
    def trimmed_mean(client_models_dict, beta=0.2):
        """
        Coordinate-wise Trimmed Mean baseline.
        Trims largest and smallest beta fraction per parameter coordinate.
        """
        cids = list(client_models_dict.keys())
        m = len(cids)
        first_cid = cids[0]
        aggregated = copy.deepcopy(client_models_dict[first_cid])
        k = int(m * beta)
        
        for key in aggregated.keys():
            stacked = torch.stack([client_models_dict[cid][key].float() for cid in cids], dim=0)
            sorted_stacked, _ = torch.sort(stacked, dim=0)
            # Trim extremes
            if k > 0 and 2 * k < m:
                trimmed = sorted_stacked[k:m - k]
            else:
                trimmed = sorted_stacked
            aggregated[key] = torch.mean(trimmed, dim=0).to(client_models_dict[first_cid][key].dtype)
            
        return aggregated

    @staticmethod
    def coordinate_median(client_models_dict):
        """Coordinate-wise Median robust aggregation baseline."""
        cids = list(client_models_dict.keys())
        first_cid = cids[0]
        aggregated = copy.deepcopy(client_models_dict[first_cid])
        for key in aggregated.keys():
            stacked = torch.stack([client_models_dict[cid][key].float() for cid in cids], dim=0)
            median_val = torch.median(stacked, dim=0).values
            aggregated[key] = median_val.to(client_models_dict[first_cid][key].dtype)
        return aggregated
