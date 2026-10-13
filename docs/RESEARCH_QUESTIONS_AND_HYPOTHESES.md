# RESEARCH QUESTIONS & TESTABLE HYPOTHESES
**Project:** Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs Under Dynamic Edge Conditions (TAR-FRL-VEC)  
**Document Version:** 2.0  
**Status:** Approved Research Protocol  

---

## 1. Overview

This document formally specifies the primary research questions (RQs) and testable hypotheses (Hypotheses H1–H5) guiding the TAR-FRL-VEC investigation. Each hypothesis defines precise dependent variables, mathematical criteria, and verification procedures to ensure scientific rigor and empirical defensibility.

---

## 2. Research Questions and Formal Hypotheses

### Research Question 1 (RQ1): Dynamic Trust Estimation Under Dynamic VANET Conditions
> *How can trust be dynamically estimated when participant behaviour, connectivity, and available resources change?*

* **Hypothesis 1 (H1):**  
  A dual-layer Bayesian Beta model with exponential temporal decay ($\lambda = 0.95$) enables the coordinator to adaptively track participant reliability across time-varying vehicular velocities ($12\text{--}33\text{ m/s}$) and compute queue backlogs, distinguishing benign mobility-induced disconnections from adversarial execution failures.
* **Dependent Metrics:**
  - True Negative Rate (TNR) on mobility disconnections: $\ge 90\%$.
  - Trust convergence time following node state transition: $\le 10$ interaction events.
  - Trust value boundedness: strictly $T \in [0.0, 1.0]$.
* **Validation Procedure:** Evaluate trust trajectories across low, medium, and high mobility highway runs in `tests/test_phase5_trust.py` and SUMO microscopic traces.

---

### Research Question 2 (RQ2): Synergistic Dual-Layer Malicious Detection
> *Does combining behavioural trust with model-update trust identify malicious participants better than behavioural trust alone?*

* **Hypothesis 2 (H2):**  
  Composite trust $T_{\text{final}} = \alpha \cdot T_{\text{beh}} + (1 - \alpha) \cdot T_{\text{model}}$ ($\alpha = 0.6$) achieves statistically significant improvement in detecting stealthy adversarial nodes (such as free-riders and subtle parameter-poisoning attackers who maintain seemingly compliant task execution while corrupting the global neural network) compared to behavioral trust alone.
* **Dependent Metrics:**
  - Detection Precision & Recall for poisoned model updates.
  - False Positive Rate (FPR) on honest non-IID clients: $\le 5\%$.
  - Detection Area Under the Curve (AUC-ROC): $AUC_{\text{composite}} > AUC_{\text{behavioural\_only}} + 0.15$.
* **Validation Procedure:** Benchmark against ablation configuration `without_model_update_trust` under $20\%$ and $40\%$ poisoning ratios.

---

### Research Question 3 (RQ3): Task Latency, Completion, and Reliability Under Normal Operation
> *Does trust-aware offloading improve task latency, completion rate, and reliability compared to heuristic baselines?*

* **Hypothesis 3 (H3):**  
  Under standard edge network operation, the proposed trust-aware federated DQN agent will achieve a higher task completion rate and lower average end-to-end task latency compared to local execution and random offloading, and achieve comparable latency to resource-aware heuristic scheduling while maintaining dynamic security.
* **Dependent Metrics:**
  - Task Completion Rate: $CR_{\text{TAR-FRL}} \ge 70\%$ under moderate workload.
  - Mean Latency: $T_{\text{TAR-FRL}} \le 5.0\text{ s}$ (representing a $> 50\%$ reduction compared to local execution $T_{\text{local}} \approx 13.5\text{ s}$).
  - Offload Fraction: Non-zero offloading actively utilizing eligible RSUs ($OF > 0.40$).
* **Validation Procedure:** Multi-seed comparative evaluations across `local_only`, `random_valid`, `nearest_edge`, and `resource_aware`.

---

### Research Question 4 (RQ4): Resilience Under Escalating Adversarial Ratios
> *How does system offloading performance change as the proportion of malicious participants increases from 0% to 40%?*

* **Hypothesis 4 (H4):**  
  Under escalating adversarial ratios ($f_{\text{malicious}} \in \{0\%, 10\%, 20\%, 30\%, 40\%\}$), standard FedAvg suffers catastrophic performance collapse (completion rate degradation $> 30\%$, mean reward degradation $> 50\%$), whereas TAR-FRL-VEC with trust-threshold robust aggregation ($\theta = 0.40$) bounds completion rate degradation to $\le 10\%$.
* **Dependent Metrics:**
  - Global Model Degradation Slope: $\left|\frac{\partial CR}{\partial f_{\text{malicious}}}\right|_{\text{TAR-FRL}} \ll \left|\frac{\partial CR}{\partial f_{\text{malicious}}}\right|_{\text{FedAvg}}$.
  - Safe Fallback Invocation Rate: Exact $100\%$ model parameter retention when zero clients pass the threshold ($S_r = \emptyset$).
* **Validation Procedure:** Adversarial ratio sweep ($0\%\text{--}40\%$) with 10 random seeds comparing standard FedAvg against TAR-FRL-VEC.

---

### Research Question 5 (RQ5): Robustness Under Heterogeneity, Mobility, and Non-IID Traces
> *Does the method remain useful under microscopic mobility, intermittent connectivity, heterogeneous resources, and non-IID local experience?*

* **Hypothesis 5 (H5):**  
  When deployed on realistic microscopic traffic generated by Eclipse SUMO/TraCI (with dynamic Krauss car-following, RSU handovers, heterogeneous vehicular CPU frequencies $1.2\text{--}2.8\text{ GHz}$, and non-IID task generation), TAR-FRL-VEC maintains superior task completion and lower energy consumption compared to non-collaborative and static heuristic baselines.
* **Dependent Metrics:**
  - Microscopic Handover Continuity: Successful task completion during or across inter-RSU handovers $\ge 65\%$.
  - Energy Efficiency: Mean energy per task $\le 400\text{ J}$ (including transmission and computation).
  - Cross-Environment Stability: Consistent policy ranking between synthetic 1D torus benchmarks and SUMO TraCI corridor simulations.
* **Validation Procedure:** Evaluation on `SumoVehicularEdgeEnv` across Low, Medium, and High traffic densities.

---
*Formulated in accordance with IEEE Transactions on Mobile Computing experimental guidelines.*
