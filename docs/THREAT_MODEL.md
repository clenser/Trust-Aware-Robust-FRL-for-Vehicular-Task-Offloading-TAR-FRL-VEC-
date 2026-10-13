# THREAT MODEL & BYZANTINE ADVERSARIAL RESILIENCE
**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions (TAR-FRL-VEC)  
**Document Version:** 2.0  
**Status:** Approved Security Specification  

---

## 1. System Threat Landscape & Attacker Capabilities

In Vehicular Ad-Hoc Networks (VANETs) and Vehicular Edge Computing (VEC), participants are untrusted edge nodes operating in dynamic, open wireless environments. While edge servers (RSUs) are assumed to be managed infrastructure nodes with hardware security modules, client vehicles are consumer mobile devices vulnerable to software tampering, firmware compromise, selfish resource conservation, and rogue adversarial actors.

### 1.1 Attacker Types & Objectives
1. **Byzantine / Malicious Attackers:** Aim to degrade global federated model convergence, induce high offloading failure rates, increase task latencies, or cause dangerous vehicular task deadline violations.
2. **Selfish Free-Riders:** Aim to benefit from the aggregated global reinforcement learning policy while contributing zero local computational work (avoiding battery and GPU/CPU training expenditures).
3. **Unreliable / Defective Participants:** Exhibit intermittent hardware failures, high network packet drop rates, or untimely task timeouts without deliberate malice.

### 1.2 Attacker Capabilities & Limitations
* **Capabilities:**
  - Full local control over compromised client devices, local replay buffers, and generated weight updates.
  - Ability to manipulate reported model weights $w_i^r$, gradients, and execution feedback.
  - Ability to drop offloaded tasks, delay computation past deadlines, or falsely acknowledge task completion.
  - Collusion up to a fraction $f < 0.50$ of participating vehicles.
* **Limitations (Defender Assumptions):**
  - The central coordinator (RSU cluster / orchestrator) is trusted and maintains execution log integrity.
  - Raw experience replay buffers are never transmitted over the air (preserving differential privacy and bandwidth).
  - The proportion of malicious participants is bounded by the theoretical Byzantine tolerance limit ($f_{\text{malicious}} \le 40\%$).

---

## 2. Attack Vectors & Mathematical Formulations

TAR-FRL-VEC formally evaluates four attack categories implemented in `attacks/`:

### 2.1 Model Poisoning Attacks (`attacks/poisoning.py`)
Compromised vehicular clients transmit perturbed parameter updates $\tilde{w}_i^r$ to corrupt the global DQN policy:

* **Sign-Flipping Poisoning:**
  $$\tilde{w}_i^r = -\gamma_{\text{flip}} \cdot \Delta w_i^r + w^r$$
  where $\Delta w_i^r = w_i^r - w^r$ is the legitimate training increment, and $\gamma_{\text{flip}} \ge 1.0$ is the attack amplification factor. This inverts the policy gradients, causing the agent to maximize task latency and deadline failures.

* **Gaussian Noise Perturbation:**
  $$\tilde{w}_i^r = w_i^r + \mathcal{N}\left(0, \sigma_{\text{poison}}^2 \mathbf{I}\right)$$
  Injects isotropic high-variance Gaussian noise to destabilize stochastic gradient descent and disrupt DQN temporal difference convergence.

* **Targeted Deviation / Scaling:**
  $$\tilde{w}_i^r = w^r + \beta_{\text{scale}} \cdot \frac{\Delta w_i^r}{\|\Delta w_i^r\|_2}$$
  Pushes the global model into suboptimal local extrema while masquerading as large-magnitude gradient updates.

### 2.2 Free-Riding Attacks (`attacks/free_riding.py`)
Selfish nodes conserve local compute energy while attempting to receive updated global weights:

* **Zero-Update Free-Riding:**
  $$\tilde{w}_i^r = w^r \implies \Delta \tilde{w}_i^r = \mathbf{0}$$
* **Stale Parameter Replay:**
  $$\tilde{w}_i^r = w_i^{r - \tau_{\text{stale}}}$$
  Replaying cached weights from preceding rounds without executing local training steps.
* **Copied Update Replay:**
  $$\tilde{w}_i^r = w_j^r + \delta_{\epsilon}$$
  Eavesdropping or copying another client's submitted weights with minute epsilon offsets.

### 2.3 Execution Unreliability & Dropping (`attacks/unreliable_node.py`)
Nodes advertise compute availability but drop or stall computational tasks:
* **Selective Packet Dropping:** Incurring task drop probability $p_{\text{drop}} \in [0.2, 0.8]$.
* **Execution Throttling:** Artificially delaying execution until $T^{\text{exe}} > T^{\text{max}}$, triggering hard deadline misses.

### 2.4 Intermittent Disconnections vs Malicious Behavior
A critical challenge in VANETs is distinguishing benign mobility dropouts from adversarial sabotage:
* When a vehicle leaves RSU radio coverage ($d_i > R_{\text{comm}}$), link loss is a geometric property of high-speed mobility.
* TAR-FRL-VEC isolates mobility events: pure topological range exits are labeled `disconnect` without applying severe Bayesian penalty updates to the node's behavioral trust record.

---

## 3. Defense Mechanisms in TAR-FRL-VEC

The defense pipeline operates across both the offloading destination selection and the federated model aggregation phases:

```
[Incoming Local Model Updates {w_i^r}]
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Model-Update Anomaly Detection      │
│  - Parameter Distance: ||w_i - w_med||  │
│  - Cosine Alignment: sim(Δw_i, Δw_bar)  │
│  - Stale/Zero Check: ||Δw_i|| > ε       │
└──────────────────┬──────────────────────┘
                   │
                   ▼ Model Trust T_model(i)
┌─────────────────────────────────────────┐
│     Behavioural Trust Integration       │
│  - Historical Beta Prior (α_i, β_i)     │
│  - Exponential Decay Smoothing (λ=0.95) │
└──────────────────┬──────────────────────┘
                   │
                   ▼ Combined Trust: T_final = α*T_beh + (1-α)*T_model
┌─────────────────────────────────────────┐
│       Robust Aggregation Decision       │
│  - Threshold Gate: T_final >= θ (0.40)  │
│  - If S_r empty: Fallback to w^r        │
│  - Else: Normalize Trust Weights        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
       [Updated Global Model w^{r+1}]
```

### 3.1 Mathematical Properties of Robust Aggregation
1. **Division-by-Zero Resilience:** If all reporting clients are detected as malicious ($T_i < \theta \, \forall i$), the set $\mathcal{S}_r = \emptyset$. The algorithm executes safe fallback:
   $$w^{r+1} = w^r$$
   preserving the last known benign model parameters without numerical failure.
2. **Byzantine Breakdown Point:** For model updates alone, coordinate-wise median / trimmed mean achieves a breakdown point of up to 50% Byzantine clients. By coupling model updates with verified task execution histories ($T_{\text{beh}}$), TAR-FRL-VEC maintains defensible policy convergence up to 40% malicious nodes.
3. **Non-IID Fairness Guarantee:** Legitimate clients with diverse driving environments exhibit parameter updates deviating from the mean. TAR-FRL-VEC protects non-IID honest clients by examining learning progress (non-zero loss reduction) and verifying behavioral compliance, preventing false positive exclusion.

---
*Verified against NIST SP 800-30 risk assessment standards and IEEE Trans. Dependable & Secure Computing protocols.*
