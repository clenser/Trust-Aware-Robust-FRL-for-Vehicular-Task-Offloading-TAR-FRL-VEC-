# Microscopic Vehicular Traffic & TraCI Integration Specification

**Project:** Trust-Aware Robust Federated Reinforcement Learning for Vehicular Task Offloading (TAR-FRL-VEC)  
**Document:** `docs/SUMO_MICROSCOPIC_MOBILITY_SPEC.md`  
**Status:** Implemented & Verified against Eclipse SUMO v1.28.0 and TraCI v1.28.0

---

## 1. Executive Summary & Research Scope

This document specifies the integration of **Eclipse SUMO (Simulation of Urban MObility)** and **TraCI (Traffic Control Interface)** into the TAR-FRL-VEC platform. This upgrade transitions the simulation from simplified synthetic 1D constant-velocity kinematics with toroidal wraparound to genuine microscopic vehicular traffic physics.

Vehicular movement, longitudinal spacing, car-following deceleration, and lane-changing dynamics are actively computed by the SUMO microscopic engine and observed by Python reinforcement-learning agents in real time.

---

## 2. Microscopic Highway Network Architecture

The microscopic simulation models a 3-lane physical highway corridor calibrated to international highway engineering standards:

```
Lane 2 [Left / Fast Lane]:     Y = +3.6m | Speed limit = 33.33 m/s (120 km/h)
Lane 1 [Center / Cruising]:    Y =  0.0m | Speed limit = 33.33 m/s (120 km/h)
Lane 0 [Right / Slow / Van]:   Y = -3.6m | Speed limit = 33.33 m/s (120 km/h)
=============================================================================
Highway Length: 1200.0 meters | Longitudinal coordinates: X in [0.0, 1200.0]
```

### 2.1 Vehicle Chassis Classes & Kinematics
SUMO generates microscopic traffic using three heterogeneous vehicle categories parameterized in `environment/sumo/highway.rou.xml`:

| Attribute | Executive Sedan (`passenger_sedan`) | SUV / Crossover (`passenger_suv`) | Commercial Van (`commercial_van`) |
| :--- | :--- | :--- | :--- |
| **Physical Dimensions** | $4.60\text{m} \times 1.85\text{m}$ | $4.85\text{m} \times 1.95\text{m}$ | $5.20\text{m} \times 2.00\text{m}$ |
| **Max Desired Speed** | $33.33\text{ m/s}$ ($120\text{ km/h}$) | $30.00\text{ m/s}$ ($108\text{ km/h}$) | $28.00\text{ m/s}$ ($100\text{ km/h}$) |
| **Acceleration** | $2.6\text{ m/s}^2$ | $2.4\text{ m/s}^2$ | $2.0\text{ m/s}^2$ |
| **Deceleration** | $4.5\text{ m/s}^2$ | $4.2\text{ m/s}^2$ | $4.0\text{ m/s}^2$ |
| **Driver Imperfection ($\sigma$)** | $0.5$ (Krauss car-following) | $0.5$ (Krauss car-following) | $0.5$ (Krauss car-following) |
| **Lane Change Model** | LC2013 ($\text{strategic}=1.0$) | LC2013 ($\text{strategic}=1.0$) | LC2013 ($\text{cooperative}=0.8$) |
| **Insertion Lane** | Random (Lanes 0, 1, 2) | Random (Lanes 0, 1, 2) | Lane 0 (Rightmost) |

### 2.2 Microscopic Traffic Density Profiles
Vehicle insertion flows are modeled continuously at the upstream entrance ($X = 0\text{m}$) without artificial wraparound:
- **Low Density:** Flow rate $\sim 750\text{ veh/h}$ (free-flowing traffic, minimal braking).
- **Medium Density:** Flow rate $\sim 1550\text{ veh/h}$ (moderate vehicle interactions and overtaking).
- **High Density:** Flow rate $\sim 2500\text{ veh/h}$ (platooning, queue formation, frequent braking).

---

## 3. Roadside Infrastructure (RSUs), Coverage & Handover Model

Roadside Units (RSUs) are deployed at fixed physical coordinates along the highway corridor:
$$\text{RSU } j \text{ position: } X_j = (j + 1) \cdot \frac{L_{\text{road}}}{N_{\text{edge}} + 1}$$
For $N_{\text{edge}} = 2$ on a $1200\text{m}$ highway:
- $\text{RSU } 0$ is situated at $X_0 = 400.0\text{m}$.
- $\text{RSU } 1$ is situated at $X_1 = 800.0\text{m}$.

### 3.1 Wireless Radio Coverage & Link Availability
The wireless coverage radius is $R_{\text{comm}} = 300\text{m}$. A vehicle at longitudinal position $x_v$ is eligible to communicate with RSU $j$ if:
$$d(x_v, X_j) = |x_v - X_j| \le R_{\text{comm}}$$

Transmission link rate $R_{\text{tx}}$ follows the Shannon channel model under log-distance path loss:
$$\text{SNR}(d) = \frac{P_{\text{tx}} \cdot k_0}{(d + 1)^\gamma \cdot N_0}$$
$$R_{\text{tx}} = B \cdot \log_2(1 + \text{SNR}(d))$$
where bandwidth $B = 10\text{ MHz}$, transmit power $P_{\text{tx}} = 0.5\text{ W}$, path-loss constant $k_0 = 2 \times 10^{-3}$, exponent $\gamma = 3.0$, and noise floor $N_0 = 10^{-9}\text{ W}$.

### 3.2 Association, Loss of Coverage, and Handovers
1. **Initial Association:** When a vehicle enters the coverage envelope ($d \le R_{\text{comm}}$), an `associate` event is logged.
2. **RSU Handover:** When a vehicle moves into an overlapping coverage zone ($|x_v - X_0| \le 300\text{m}$ and $|x_v - X_1| \le 300\text{m}$ around $X \in [500, 700]\text{m}$) and transitions its primary association from RSU $0$ to RSU $1$, a `handover` event is logged, adding a signaling delay penalty $\Delta t_{\text{ho}} = 50\text{ms}$.
3. **Disconnection / Out of Range:** If a vehicle drives past the coverage boundary before task execution or return transmission finishes, a transmission failure is recorded.

---

## 4. Scientific Demarcation: Measured vs. Modeled Variables

To maintain scientific integrity, the project strictly distinguishes measured variables from modeled parameters:

| Variable / Domain | Mechanism | Nature | Scientific Justification |
| :--- | :--- | :--- | :--- |
| **Vehicle Kinematics** | TraCI / SUMO | **Microscopically Simulated** | Generated by Krauss car-following and LC2013 lane-changing equations. |
| **Vehicle Heading & Lane** | TraCI / SUMO | **Microscopically Simulated** | Directly extracted from the running SUMO road network. |
| **Wireless Propagation** | Python / NumPy | **Modeled (Shannon & Path Loss)** | Shannon channel with log-distance path loss ($\gamma=3.0$). Not packet-level RF ray tracing. |
| **Task Workloads** | Python / NumPy | **Modeled (Workload Model)** | Data size ($1\text{--}10\text{ Mbit}$), compute intensity ($200\text{--}2000\text{ cycles/bit}$), deadlines. |
| **Edge Queues & Processing** | Python / NumPy | **Modeled (M/M/1 FIFO Model)** | CPU clock cycles drained per second by effective frequency $f_{\text{cpu}}$. |
| **Behavioural Trust Scoring** | Python / NumPy | **Algorithmic (SPEC §6)** | Multi-criteria smoothing table ($\beta=[0.4, 0.25, 0.2, 0.15]$, $\lambda=0.9$, $\rho=0.99$). |

---

## 5. Mobility Dataset Evaluation & Trace Replay

As established in `reports/MOBILITY_DATA_EVALUATION.md`:
1. **SUMO Microscopic Engine (Current Implementation):** Produces authentic car-following and lane-changing physics with zero credential barriers, deterministic seed repeatability, and multi-lane vehicular dynamics.
2. **External Trace-Driven Datasets (e.g., highD, LuST):**
   - *highD Dataset:* Restricted by non-redistributable academic license agreements requiring manual registration and identity verification. Cannot be bundled into public repositories.
   - *LuST Scenario:* A full urban scenario requiring high memory and complex external road graphs without computational workload annotations.
3. **Trace Adapter Architecture:**
   The `SumoVehicularEdgeEnv` adapter interface is decoupled from the simulator backend. To replay static external CSV trajectory traces, an adapter can feed external $(x, y, v, \text{lane})$ records directly to `veh_snapshots` while preserving the exact same edge queueing, offloading decision, and trust update pipeline.

---

## 6. Windows Installation & CLI Verification

### 6.1 Prerequisites
Eclipse SUMO v1.28.0 and Python TraCI are installed locally:
- Python packages: `pip install traci sumolib`
- Portable SUMO Binaries: `C:\Users\srbar\AppData\Local\sumo-1.28.0\sumo-win64-1.28.0\bin\sumo.exe`
- Environment Variable: `SUMO_HOME = C:\Users\srbar\AppData\Local\sumo-1.28.0\sumo-win64-1.28.0`

### 6.2 Standalone Verification Commands

1. **Verify SUMO executable version:**
   ```powershell
   & "C:\Users\srbar\AppData\Local\sumo-1.28.0\sumo-win64-1.28.0\bin\sumo.exe" --version
   ```

2. **Run headless simulation of highway scenario:**
   ```powershell
   & "C:\Users\srbar\AppData\Local\sumo-1.28.0\sumo-win64-1.28.0\bin\sumo.exe" -c environment\sumo\highway.sumocfg --end 100
   ```

3. **Run graphical SUMO-GUI (Optional debug visualization):**
   ```powershell
   & "C:\Users\srbar\AppData\Local\sumo-1.28.0\sumo-win64-1.28.0\bin\sumo-gui.exe" -c environment\sumo\highway.sumocfg
   ```

4. **Execute dedicated SUMO smoke experiment:**
   ```powershell
   .venv\Scripts\python.exe scripts\run_sumo_smoke_experiment.py --tasks 20 --seed 42 --density medium
   ```

5. **Run automated test suite:**
   ```powershell
   .venv\Scripts\pytest tests\test_sumo_integration.py -v
   ```
