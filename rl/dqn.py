"""
Dueling Double Deep Q-Network (DQN) for VANET Task Offloading.
State vector captures:
[task_data_size, task_cpu_cycles, task_deadline, vehicle_speed,
 dist_to_rsu1, dist_to_rsu2, dist_to_rsu3,
 local_queue_len, local_cpu, rsu1_trust, rsu2_trust, rsu3_trust,
 best_v2v_dist, best_v2v_trust, rsu1_load, rsu2_load, rsu3_load]  (17 features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim=20, action_dim=5):
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature representation layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.LayerNorm(128)
        
        # Value stream V(s)
        self.val_fc = nn.Linear(128, 64)
        self.val_out = nn.Linear(64, 1)
        
        # Advantage stream A(s, a)
        self.adv_fc = nn.Linear(128, 64)
        self.adv_out = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        val = F.relu(self.val_fc(x))
        val = self.val_out(val)
        
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_vals = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_vals
