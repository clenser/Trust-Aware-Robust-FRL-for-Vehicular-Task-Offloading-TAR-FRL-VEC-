# MATHEMATICAL SYSTEM SPECIFICATION
**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions (TAR-FRL-VEC)  
**Document Version:** 2.0  
**Status:** Approved Formal Specification  

---

## 1. Vehicular Edge Network Topology & Physical Environment

Consider a linear highway corridor of length $L \in \mathbb{R}^+$ meters (e.g., $L = 1200\text{ m}$ in microscopic simulation; $L = 2000\text{ m}$ in macro synthetic benchmarks).

* **Roadside Units (RSUs / Edge Servers):**
  There are $M$ edge servers indexed by $m \in \{0, 1, \dots, M-1\}$. Each RSU $m$ is positioned at a fixed spatial coordinate $x_m^{\text{rsu}} \in [0, L]$:
  $$x_m^{\text{rsu}} = (m + 1) \cdot \frac{L}{M + 1}$$
  Each RSU is equipped with a high-capacity multi-core compute cluster operating at nominal clock frequency $f_m^{\text{edge}} \in [6.0, 10.0]\text{ GHz}$ and has wireless communication coverage radius $R_{\text{comm}} \in [150.0, 300.0]\text{ m}$.

* **Vehicular Nodes:**
  A dynamic set of $N(t)$ vehicles navigate the highway corridor. Each vehicle $i \in \{0, \dots, N(t)-1\}$ at simulation time $t$ possesses:
  - 1D longitudinal position $x_i(t) \in [0, L]$ and lateral lane index $l_i(t) \in \{0, 1, 2\}$.
  - Instantaneous forward velocity $v_i(t) \in [v_{\min}, v_{\max}]$ regulated by SUMO's microscopic Krauss car-following model.
  - Onboard CPU compute capacity $f_i \in [1.2, 2.8]\text{ GHz}$.
  - Current CPU utilization $u_i(t) \in [0.0, 0.8]$, yielding effective local computation frequency:
    $$f_i^{\text{eff}}(t) = f_i \cdot (1 - u_i(t))$$
  - Local task backlog queue $Q_i(t) \ge 0$ cycles.

---

## 2. Wireless Communication & Channel Model

Offloading communication occurs over vehicle-to-infrastructure (V2I) or vehicle-to-vehicle (V2V) links using orthogonal frequency channels.

### 2.1 Spatial Distance & Connectivity
The spatial distance between transmitter $i$ and receiver $j$ along the road corridor is:
$$d_{i,j}(t) = |x_i(t) - x_j(t)|$$
An offloading link is topologically admissible if and only if:
$$d_{i,j}(t) \le R_{\text{comm}}$$

### 2.2 Path Loss and Signal-to-Noise Ratio (SNR)
Following the standardized log-distance path loss formulation:
$$\text{PL}(d_{i,j}) = K_0 \cdot (d_{i,j} + 1)^{-\gamma}$$
where $K_0 = 2.0 \times 10^{-3}$ represents the reference channel gain at 1 meter, and $\gamma = 3.0$ is the path-loss exponent for vehicular highway propagation.

The received signal-to-noise ratio $\Gamma_{i,j}(t)$ at receiver $j$ is:
$$\Gamma_{i,j}(t) = \frac{P_{\text{tx}} \cdot K_0 \cdot (d_{i,j}(t) + 1)^{-\gamma}}{N_0}$$
where $P_{\text{tx}} = 0.5\text{ W}$ (27 dBm) is the vehicular transmission power, and $N_0 = 1.0 \times 10^{-9}\text{ W}$ is the background additive white Gaussian noise (AWGN) power over bandwidth $B = 10\text{ MHz}$.

### 2.3 Achievable Data Rate & Transmission Latency
By Shannon's capacity theorem, the achievable transmission rate $R_{i,j}(t)$ (in bits/s) is:
$$R_{i,j}(t) = B \cdot \log_2\left(1 + \Gamma_{i,j}(t)\right)$$

For a computational task $k$ with input data size $D_k$ bits, the uplink transmission delay $T_{i,j}^{\text{tx}}$ is:
$$T_{i,j}^{\text{tx}} = \frac{D_k}{R_{i,j}(t)} + t_{\text{handshake}}$$
where $t_{\text{handshake}} = 5\text{ ms}$ accounts for MAC protocol association and connection establishment.

The communication energy consumption incurred by the requester vehicle is:
$$E_{i,j}^{\text{tx}} = P_{\text{tx}} \cdot T_{i,j}^{\text{tx}}$$

---

## 3. Computational Workload & Execution Latency

Tasks are dynamically generated as tuples:
$$\mathcal{T}_k = \langle k, \text{vid}_k, t_k^{\text{arr}}, D_k, C_k, T_k^{\text{max}}, p_k \rangle$$
where $D_k \in [2, 10]\text{ Mb}$ is the input data size, $C_k \in [5, 20]\text{ Gcycles}$ is the required CPU cycles, $T_k^{\text{max}} \in [5.0, 15.0]\text{ s}$ is the hard deadline, and $p_k \in \{1, 2, 3\}$ is the priority class.

### 3.1 Local Vehicle Execution
If task $k$ is executed locally on vehicle $i$ ($\text{action} = 0$):
$$T_k^{\text{exe}} = \frac{C_k + Q_i(t)}{f_i^{\text{eff}}(t)}$$
$$T_k^{\text{total}} = T_k^{\text{exe}}$$
The energy consumed by the local CPU obeys the CMOS dynamic power dissipation law:
$$E_k^{\text{local}} = \kappa \cdot \left(f_i^{\text{eff}}(t)\right)^2 \cdot C_k$$
where $\kappa = 1.0 \times 10^{-27}\text{ J}\cdot\text{s}^2$ is the chip physical capacitance coefficient.

### 3.2 Remote Edge Server (RSU) Execution
If task $k$ is offloaded to RSU $m$ ($\text{action} > 0$):
$$T_k^{\text{exe}} = \frac{C_k + Q_m^{\text{edge}}(t)}{f_m^{\text{edge}}}$$
$$T_k^{\text{total}} = T_{i,m}^{\text{tx}} + T_k^{\text{exe}}$$
From the mobile vehicle's perspective, energy consumption is limited to the wireless transmission phase:
$$E_k^{\text{total}} = E_{i,m}^{\text{tx}}$$

### 3.3 Task Completion Criteria
A task succeeds if and only if it completes within its allowable deadline and maintains physical link integrity:
$$\text{Success}(k) = \mathbb{I}\left(T_k^{\text{total}} \le T_k^{\text{max}} \land \text{LinkActive}(i, j)\right)$$
If $T_k^{\text{total}} > T_k^{\text{max}}$, the task incurs a deadline miss ($\text{Miss}(k) = 1$). If the wireless link drops before task delivery or the candidate fails, the task incurs a failure ($\text{Fail}(k) = 1$).

---

## 4. Reinforcement Learning Offloading Formulation

The task offloading problem is framed as a discrete-time Markov Decision Process (MDP):

### 4.1 State Space $\mathcal{S}$
The observation vector $s_t \in \mathbb{R}^{7 + 3K}$ for top-$K$ candidates ($K = 5$, total dimension 22) consists of:
1. Normalized task characteristics: $[\tilde{D}_k, \tilde{C}_k, \tilde{T}_k^{\text{max}}, \tilde{p}_k]$
2. Normalized requester state: $[\tilde{u}_i, \tilde{Q}_i]$
3. For each candidate $j \in \{1, \dots, K\}$: $[\tilde{d}_{i,j}, \tau_j, \tilde{Q}_j]$ (normalized distance, trust score, normalized queue)
4. Candidate density: $\frac{|\mathcal{C}_k|}{K}$

### 4.2 Action Space $\mathcal{A}$
The action space is discrete:
$$\mathcal{A} = \{0, 1, \dots, K\}$$
- $a = 0$: Execute locally.
- $a \in \{1, \dots, K\}$: Offload to the $a$-th nearest eligible candidate node in candidate set $\mathcal{C}_k$.

### 4.3 Multi-Objective Reward Function
The scalar reward $r_t$ balances latency, energy, reliability, and security:
$$r_t = - w_{\text{lat}} \cdot \min\left(1, \frac{T_k^{\text{total}}}{T_{\text{norm}}}\right) - w_{\text{ene}} \cdot \min\left(1, \frac{E_k^{\text{total}}}{E_{\text{norm}}}\right) + w_{\text{suc}} \cdot \mathbb{I}_{\text{suc}} - w_{\text{fail}} \cdot \mathbb{I}_{\text{fail}} - w_{\text{miss}} \cdot \mathbb{I}_{\text{miss}} + w_{\text{trust}} \cdot \tau_{\text{target}}$$
Coefficients:
$w_{\text{lat}} = 1.0, w_{\text{ene}} = 0.5, w_{\text{suc}} = 2.0, w_{\text{fail}} = 3.0, w_{\text{miss}} = 2.0, w_{\text{trust}} = 0.5$.  
Normalizations: $T_{\text{norm}} = 10.0\text{ s}, E_{\text{norm}} = 10.0\text{ J}$.

---

## 5. Dual-Layer Dynamic Trust Model

Trust combines historical execution behavior with federated model-update reliability:
$$T_{\text{final}}(i, r) = \alpha \cdot T_{\text{beh}}(i, r) + (1 - \alpha) \cdot T_{\text{model}}(i, r), \quad \alpha \in [0, 1]$$

### 5.1 Behavioural Trust Layer
Behavioural trust models interaction outcomes using a Bayesian Beta distribution $\text{Beta}(\alpha_i, \beta_i)$:
$$T_{\text{beh}}(i, t) = \frac{\alpha_i(t)}{\alpha_i(t) + \beta_i(t)}$$
When interaction evidence $e_t = \langle \text{success}, \text{latency}, \text{availability} \rangle$ is received:
$$\alpha_i(t+1) = \lambda \cdot \alpha_i(t) + \Delta \alpha(e_t)$$
$$\beta_i(t+1) = \lambda \cdot \beta_i(t) + \Delta \beta(e_t)$$
where $\lambda = 0.95$ is the temporal memory decay factor. Disconnections due purely to vehicle mobility (out-of-range) are not penalized as adversarial malicious failures.

### 5.2 Model-Update Trust Layer
In federated learning round $r$, each participating client $i$ transmits local parameter weights $w_i^r$. The coordinator computes:
1. **$L_2$ Parameter Distance to Median Consensus:**
   $$d_i^r = \| w_i^r - w_{\text{median}}^r \|_2$$
2. **Cosine Similarity to Reference Aggregate:**
   $$\text{sim}_i^r = \frac{\langle \Delta w_i^r, \Delta \bar{w}^r \rangle}{\|\Delta w_i^r\|_2 \cdot \|\Delta \bar{w}^r\|_2}$$
3. **Free-Riding / Stale Update Detection:**
   $$\text{Stale}(i) = \mathbb{I}\left(\|w_i^r - w_i^{r-1}\|_2 < \epsilon_{\text{zero}} \lor \|w_i^r - w_{\text{global}}^{r-1}\|_2 < \epsilon_{\text{copy}}\right)$$

The model trust score is evaluated as:
$$T_{\text{model}}(i, r) = \max\left(0, \exp\left(-\frac{d_i^r}{\sigma_d}\right) \cdot \max(0, \text{sim}_i^r) \cdot (1 - \mathbb{I}_{\text{stale}})\right)$$

---

## 6. Trust-Aware Robust Aggregation (TAR-FRL)

Let $\mathcal{K}_r$ denote the set of reporting clients in round $r$.

### 6.1 Baseline FedAvg
$$w^{r+1} = \sum_{i \in \mathcal{K}_r} \frac{n_i}{\sum_{j} n_j} w_i^r$$

### 6.2 Trust-Weighted FedAvg
$$w^{r+1} = \sum_{i \in \mathcal{K}_r} \frac{T_i^r}{\sum_{j \in \mathcal{K}_r} T_j^r} w_i^r$$

### 6.3 Trust-Threshold Robust Aggregation (Proposed TAR-FRL)
Eligible clients are filtered against an empirically validated trust threshold $\theta = 0.40$:
$$\mathcal{S}_r = \{i \in \mathcal{K}_r \mid T_{\text{final}}(i, r) \ge \theta\}$$
The robust global model update is computed as:
$$w^{r+1} = \begin{cases}
\sum_{i \in \mathcal{S}_r} \frac{T_i^r}{\sum_{j \in \mathcal{S}_r} T_j^r} w_i^r, & \text{if } \mathcal{S}_r \neq \emptyset \\
w^r, & \text{if } \mathcal{S}_r = \emptyset \text{ (Safe Fallback)}
\end{cases}$$
The safe fallback guarantees mathematical stability and prevents catastrophic model degradation or division-by-zero when all reporting clients are adversarial.

---
*Verified against IEEE/ACM Transactions benchmarks and repository implementation.*
