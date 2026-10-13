# Mobility Data Evaluation & Integration Assessment

## Project: Trust-Aware Robust FRL for Vehicular Task Offloading (TAR-FRL-VEC)

This document provides a formal evaluation of external real-world and trace-driven vehicular mobility datasets against the operational requirements, simulation constraints, and scientific scope of the TAR-FRL-VEC research platform.

---

## 1. Candidate Mobility Sources

### 1.1 LuST SUMO Traffic Scenario
- **Source:** Luxembourg SUMO Traffic (LuST) Scenario (`https://github.com/lcodeca/LuSTScenario`)
- **Nature:** Macroscopic, mesoscopic, and microscopic synthetic-calibrated urban traffic scenario for Luxembourg City.
- **Underlying Engine:** SUMO (Simulation of Urban MObility) via TraCI (`sumolib` / `traci`).
- **Data Attributes:** Detailed road network topology (OSM-derived XML), inductive loop sensors, traffic lights, bus lines, vehicular trajectories.
- **Evaluation:**
  - **Prerequisites & Toolchain:** Requires a full SUMO binary installation (`sumo` / `sumo-gui`), XML network compilers (`netconvert`), and Python TraCI bindings. SUMO is not pre-installed in standard student laptop environments, Google Colab runtime defaults, or lightweight CI runners without custom system packages.
  - **Offloading & Edge Compatibility:** LuST specifies physical kinematics (position, speed, acceleration) but contains **no computational workloads, edge server allocations, CPU cycles, task deadlines, or wireless channel telemetry**. Integrating LuST requires wrapping TraCI in an external loop and manually generating task offloading attributes.
  - **Platform Constraint:** Binding scope rule §2 explicitly states: *"Do not require SUMO, NS-3, or OMNeT++ for the minimum project."*

### 1.2 highD Highway Drone Trajectory Dataset
- **Source:** highD Dataset (`https://levelxdata.com/highd-dataset/`)
- **Nature:** Naturalistic vehicle trajectories recorded by drones over German highways (six locations, ~110,500 vehicles, 420 km/h total recording time).
- **Format:** High-resolution CSV files and video recordings (bounding boxes, lateral/longitudinal coordinates, lane assignments, velocities, accelerations).
- **Evaluation:**
  - **Licensing & Access Barrier:** highD is protected by an academic non-commercial license agreement requiring **individual manual registration, identity verification, and non-redistributable click-through terms**. The dataset cannot be redistributed within a public GitHub repository or downloaded autonomously without credentials.
  - **File Size & Footprint:** The raw dataset spans several gigabytes, exceeding standard repository limits and typical student laptop bandwidth quotas.
  - **Edge Semantics:** highD provides real-world trajectory kinematics but has no communication links, roadside units (RSUs), task queues, or security labels.

### 1.3 Official SUMO Scenario Catalogue
- **Source:** SUMO Scenario Catalogue (`https://sumo.dlr.de/docs/Data/Scenarios.html`)
- **Scenarios Evaluated:** Bologna Urban, Cologne (TAPAS), Monaco, InTAS (Ingolstadt).
- **Evaluation:**
  - Identical toolchain limitations to LuST: all scenarios depend on external SUMO binaries, TraCI inter-process communication, and non-trivial setup scripts.
  - None provide task offloading, Byzantine adversaries, or federated learning hooks out-of-the-box.

---

## 2. Comparative Matrix

| Criterion | LuST Scenario | highD Dataset | SUMO Catalogue | Current TAR-FRL-VEC Engine |
| :--- | :--- | :--- | :--- | :--- |
| **Data Type** | Urban Micro-traffic | Highway Drone Traces | Urban / Highway | Synthetic Multi-lane VANET |
| **Toolchain Dependency** | SUMO + TraCI | Standalone CSV parser | SUMO + TraCI | Standalone Python 3.11+ / PyTorch |
| **Licensing** | EPL 2.0 (Open) | Restricted Academic Form | Varied (Open) | Project Open Source |
| **Download Barrier** | Clone repo (~100 MB) | Credential approval required | External download (~50-500 MB) | Zero download / Deterministic Seed |
| **Windows Compatibility** | Requires Win SUMO bin | Full (CSV) | Requires Win SUMO bin | Native Cross-platform |
| **Task / Edge Attributes** | None (kinematics only) | None (kinematics only) | None (kinematics only) | Native (CPU, tasks, queues, trust) |
| **Reproduction Reliability** | Dependent on SUMO build | Dependent on local storage | Dependent on SUMO build | 100% Deterministic (PRNG seed) |

---

## 3. Engineering & Methodological Decision

In strict adherence to `AGENTS.md` §2 (*"Use a Python-based synthetic or trace-driven vehicular-edge environment. Do not require SUMO, NS-3, or OMNeT++ for the minimum project... If an external dataset is unavailable or unsuitable, keep the existing dataset and build the visual simulation using the current environment"*):

1. **Retain the Deterministic Synthetic Vehicular-Edge Environment:** The core environment (`VehicularEdgeEnv`) accurately models wireless transmission loss (path-loss exponent $\gamma=3.0$), link availability, multi-lane spatial distribution, queue delays, and heterogeneous CPU processing without external binary dependencies.
2. **Build an Expressive 2D Multi-Lane Visual Road Simulation:** Render vehicles across distinct highway lanes with realistic longitudinal offsets, roadside edge units (RSUs) with visible radio coverage radii ($R_{\text{comm}} = 300\text{m}$), and dynamic offload vectors.
3. **Scientific Demarcation:** The simulation dashboard explicitly demarcates modelled synthetic vehicular dynamics from real-world telemetry, ensuring zero confusion between empirical traces and algorithmic offloading models.
