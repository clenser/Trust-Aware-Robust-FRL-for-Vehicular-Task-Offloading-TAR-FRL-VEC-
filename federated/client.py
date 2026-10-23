"""
Federated Client representing a connected vehicular agent.

Supports two local-training modes:

1. ``local_train`` - in-place training of the live agent network. Used by the
   offline experiment runner where no concurrent action selection happens.

2. ``local_train_detached`` - trains a *copy* of the local network and returns
   the resulting weights without mutating the live agent. The web console uses
   this mode so that action selection in the simulation thread can never race
   with a gradient update on the same tensors.

Both modes report the *real* training loss returned by the DQN update, and
report how many gradient steps were actually performed so that a no-op update
(e.g. replay batch smaller than the minimum) can be detected and reported
honestly.
"""

import copy
import numpy as np
from rl.agent import DQNAgent
from attacks.poisoning import ModelPoisoner
from attacks.free_riding import FreeRider


class FederatedClient:
    def __init__(self, vehicle_id, state_dim=20, action_dim=5, role="honest", poison_strength=1.0,
                 poison_type="sign_flip", attack_probability=1.0, agent_kwargs=None):
        self.vehicle_id = vehicle_id
        # Probability that a malicious client attacks in a given round (1.0 =
        # always; lower values model an intermittent attacker that behaves
        # honestly in the other rounds).
        self.attack_probability = float(attack_probability)
        # Index of the global model this client last received (0 = the initial
        # model); used to measure how stale its next upload is.
        self.model_version = 0
        self.role = role
        self.poison_strength = float(poison_strength)
        self.poison_type = poison_type
        self.agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, **(agent_kwargs or {}))
        self.previous_weights = None
        self.sample_count = 0
        # The number of transitions the client currently holds.  This is the
        # *available experience*, i.e. what it could have learned from.  It is
        # recorded before any role-specific early return so that it stays a
        # role-independent participation statistic: a client that holds fewer
        # transitions than the minimum training batch had nothing to
        # contribute, which is different from a client that holds plenty of
        # experience and chooses not to use it (free riding).
        self.available_samples = 0

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------
    def local_train(self, epochs=3):
        """Runs local DQN updates in-place from the replay buffer.

        Returns the mean loss across epochs that actually performed a gradient
        step. Returns 0.0 when no gradient step could be performed (e.g. the
        replay buffer holds fewer than the minimum batch of 4 transitions) or
        when the client is a free rider that intentionally does not train.
        """
        losses = []
        # Record the available experience for every client, including a free
        # rider that abstains from training.  The number of held transitions
        # is what distinguishes "had nothing to contribute" from "chose not
        # to contribute"; it carries no information about the client's role.
        self.available_samples = len(self.agent.memory)
        if self.role == "free_rider":
            # Free rider does NOT train locally! (saves compute/battery)
            return 0.0

        for _ in range(epochs):
            loss = self.agent.update()
            if loss > 0:
                losses.append(loss)

        self.sample_count = len(self.agent.memory)
        return float(np.mean(losses)) if losses else 0.0

    def local_train_detached(self, epochs=2):
        """Train a copy of the local network; the live agent is not mutated.

        Returns a dict:
            {
              "loss": float | None,          # real mean loss, None if no update happened
              "weights": state_dict | None,  # trained weights, None if nothing trained
              "grad_steps": int,             # number of gradient steps actually applied
              "trained": bool,               # whether a gradient step occurred
              "skipped_reason": str | None,  # why training was skipped
            }
        """
        result = {
            "loss": None,
            "weights": None,
            "grad_steps": 0,
            "trained": False,
            "skipped_reason": None,
        }

        if self.role == "free_rider":
            # Free riders deliberately abstain from local training.
            result["skipped_reason"] = "free_rider_abstains"
            return result

        buffer_len = len(self.agent.memory)
        result["samples"] = buffer_len

        if buffer_len < 4:
            # DQNAgent.update() is a no-op below 4 transitions; report it honestly
            # instead of pretending a training round happened.
            result["skipped_reason"] = "insufficient_replay_samples"
            return result

        # Build a detached copy of the agent and train that.
        shadow = DQNAgent(
            state_dim=self.agent.state_dim,
            action_dim=self.agent.action_dim,
            lr=self.agent.optimizer.param_groups[0]["lr"],
            gamma=self.agent.gamma,
            epsilon=self.agent.epsilon,
            epsilon_min=self.agent.epsilon_min,
            epsilon_decay=self.agent.epsilon_decay,
            batch_size=self.agent.batch_size,
        )
        shadow.memory = self.agent.memory  # share replay buffer (read-only usage)
        shadow.set_weights(self.agent.get_weights())
        shadow.training_steps = self.agent.training_steps

        steps_before = shadow.training_steps
        losses = []
        for _ in range(max(1, int(epochs))):
            loss = shadow.update()
            if loss > 0:
                losses.append(loss)

        grad_steps = shadow.training_steps - steps_before
        result["grad_steps"] = int(grad_steps)
        result["trained"] = grad_steps > 0
        result["weights"] = shadow.get_weights()
        result["loss"] = float(np.mean(losses)) if losses else None
        if not result["trained"]:
            result["skipped_reason"] = "no_gradient_step"
        return result

    # ------------------------------------------------------------------
    # Upload / download
    # ------------------------------------------------------------------
    def get_upload_weights(self, global_weights, local_weights=None):
        """
        Prepares model submission based on vehicle role.

        ``local_weights`` allows uploading weights produced by
        ``local_train_detached`` (i.e. trained on a copy) instead of the live
        agent's current weights.
        """
        true_local_weights = local_weights if local_weights is not None else self.agent.get_weights()

        if self.role == "honest" or self.role == "unreliable":
            self.previous_weights = copy.deepcopy(true_local_weights)
            return true_local_weights

        elif self.role == "poisoning":
            if self.attack_probability < 1.0 and np.random.rand() >= self.attack_probability:
                # Intermittent attacker: honest behaviour this round.
                self.previous_weights = copy.deepcopy(true_local_weights)
                return true_local_weights
            # Apply adversarial model poisoning with configured dynamic strength
            poisoned = ModelPoisoner.apply_attack(
                local_weights=true_local_weights,
                global_weights=global_weights,
                attack_type=self.poison_type,
                strength=self.poison_strength
            )
            return poisoned

        elif self.role == "free_rider":
            # Free rider submits either zero gradient or stale weights
            if self.previous_weights is not None and np.random.rand() < 0.5:
                return FreeRider.stale_update_attack(self.previous_weights)
            else:
                return FreeRider.zero_update_attack(global_weights)

        return true_local_weights

    def update_from_global(self, global_weights, version=None):
        """Synchronizes local model with global parameters."""
        self.agent.set_weights(global_weights)
        if version is not None:
            self.model_version = int(version)
