"""
Test basic simulation flow and integration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from environment.vanet_env import VANETOffloadingEnv
from federated.client import FederatedClient
from federated.server import FederatedServer

def test_integration():
    print("Initializing VANET environment...")
    env = VANETOffloadingEnv(num_vehicles=10, attack_ratio=0.3, mobility_mode="trace")
    server = FederatedServer(state_dim=20, action_dim=5, aggregation_strategy="trust_aware")
    
    clients = {}
    for vid, v in env.vehicles.items():
        clients[vid] = FederatedClient(vehicle_id=vid, state_dim=20, action_dim=5, role=v.role)
        
    print(f"Total vehicles: {len(clients)}, Attack ratio: 0.3")
    
    # Run 2 steps of offloading
    states_dict = env._get_global_state_dict()
    for step in range(5):
        env.update_positions()
        for vid, client in clients.items():
            state, task = states_dict[vid]
            action = client.agent.select_action(state, explore=True)
            next_state, reward, done, info, next_task = env.step_vehicle(vid, action, task)
            client.agent.memory.push(state, action, reward, next_state, done)
            states_dict[vid] = (next_state, next_task)
            
    # Run local training & 1 round of federated aggregation
    client_models = {}
    client_samples = {}
    ground_truth_roles = {vid: c.role for vid, c in clients.items()}
    global_weights = server.get_global_weights()
    
    for vid, client in clients.items():
        loss = client.local_train(epochs=1)
        upload_w = client.get_upload_weights(global_weights)
        client_models[vid] = upload_w
        client_samples[vid] = max(1, client.sample_count)
        
    new_global, metrics = server.run_aggregation_round(
        client_models, client_samples, ground_truth_roles=ground_truth_roles
    )
    print("Federated Round Metrics:", metrics)
    print("Trust Scores:", server.trust_engine.get_all_trust_scores())
    assert new_global is not None, "Global weights should not be None"
    print("Basic integration test passed successfully!")

if __name__ == "__main__":
    test_integration()
