"""
Deep Q-Learning Agent with Double DQN updates for Task Offloading.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl.dqn import DuelingQNetwork
from rl.replay_buffer import ReplayBuffer

class DQNAgent:
    # Minimum number of replay transitions required for a gradient step.
    # update() is a no-op below this threshold, so callers must treat a
    # smaller buffer as "no training happened".
    MIN_TRAIN_BATCH = 4

    def __init__(self, state_dim=20, action_dim=5, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.memory = ReplayBuffer(capacity=10000)
        self.training_steps = 0
        self.loss_history = []

    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(s_tensor)
            return int(q_vals.argmax(dim=-1).item())

    def update(self):
        batch_sz = min(len(self.memory), self.batch_size)
        if batch_sz < self.MIN_TRAIN_BATCH:
            # No gradient step is possible. The caller must be able to detect
            # this and report it honestly rather than treating it as training.
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(batch_sz)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Double DQN: action selected by policy_net, evaluated by target_net
        current_q = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions)
            expected_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.training_steps += 1
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        
        # Soft update target network periodically
        if self.training_steps % 20 == 0:
            self._soft_update(tau=0.1)
            
        return loss_val

    def _soft_update(self, tau=0.1):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def get_weights(self):
        return copy.deepcopy(self.policy_net.state_dict())

    def set_weights(self, state_dict):
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
