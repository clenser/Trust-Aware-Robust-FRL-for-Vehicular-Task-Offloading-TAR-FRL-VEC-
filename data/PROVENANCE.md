# Data Provenance, Authenticity & Workload Specification

This document provides complete traceability, verification checksums, standards provenance, and explicit scope declarations for all datasets and simulation profiles utilized in this repository.

---

## 1. Explicit Research & Simulation Scope Disclosure

> [!IMPORTANT]
> **Simulation Boundary Declaration:**  
> In vehicular edge computing research, real-world highway sensor logs (such as FHWA NGSIM) record solely physical kinematics: vehicle coordinates, velocities, accelerations, and lane positions.
> 
> **Real-world vehicle trajectory logs DO NOT record:**
> 1. Task-offloading requests or decisions,
> 2. Onboard computation queue status or edge server availability,
> 3. Real-time cellular / 5G NR-V2X channel SINR or packet transmissions,
> 4. Reinforcement learning rewards or cost functions,
> 5. Cyberattack labels, model poisoning, or Byzantine participants.
> 
> Consequently, **this project does NOT claim an end-to-end "real-data-only" training pipeline.** 
> While vehicle kinematic behavior and traffic headways are calibrated to empirical traffic studies, **all task generation, 3GPP wireless propagation channels, edge server queues, RL state-action-reward loops, and adversarial attack scenarios are simulated** according to established IEEE/3GPP/ETSI scientific models.

---

## 2. Authentic NGSIM US-101 Trajectory Dataset

### 2.1 Metadata & Provenance
* **Dataset Name:** Next Generation Simulation (NGSIM) US Highway 101 Dataset
* **Issuing Agency:** Federal Highway Administration (FHWA), United States Department of Transportation (USDOT)
* **Location:** US-101 freeway corridor in Los Angeles (Hollywood), California, USA
* **Segment Length:** ~640-680 meters, measured directly from the processed trace
  (`Local_Y` spans 1.4 m - 675.9 m in the US-101 subset).
* **Lane identifiers:** the US-101 subset contains **8 distinct `Lane_ID` values**.
  These are *lane identifiers*, and this repository has **not** validated their
  mapping to eight continuous physical lanes: NGSIM numbers lanes per direction
  of travel, so the identifier set mixes the two carriageways.  The deployment
  therefore treats the trace's lane field as a categorical key and does not
  claim a physical lane count.
* **Capture Instrument:** High-resolution synchronized video cameras mounted on the 36-story 10880 Wilshire Boulevard building
* **Official USDOT Open Data Portal:**  
  [https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
* **FHWA Technical Archive:**  
  [https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)
* **License:** Creative Commons Attribution | Share Alike 3.0 Unported (CC BY-SA 3.0) — the licence field published in the official USDOT Socrata catalogue for dataset `8ect-6jqj`, verified 2026-10-21.  Attribution: USDOT / FHWA, *Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data*.

### 2.2 Authentic Schema & Reference Preprocessing Pipeline
The raw FHWA NGSIM export provides trajectories sampled at **0.1 s (10 Hz)**
across four road sites.  The pipeline implemented in
[`data/source/prepare_ngsim.py`](file:///c:/Users/LENOVO/Desktop/VANNET/data/source/prepare_ngsim.py)
performs exactly the following steps - and nothing more:

1. **Site filtering:** only the records whose `Location` is `us-101` are kept
   (the I-80, Lankershim and Peachtree records describe different roads and
   are discarded).
2. **Duplicate removal:** exact duplicate rows are dropped, keeping the first
   occurrence.
3. **Invalid-record removal:** rows with a non-numeric `Global_Time`,
   `Local_X`, `Local_Y`, `v_Vel` or `Lane_ID` are dropped.
4. **Unit conversion:** coordinates and speeds are converted from Imperial
   (feet, ft/s) to Metric (meters, m/s) with the factor 0.3048.
5. **Time re-basing:** `time_sec` is measured from the earliest retained
   `Global_Time`, and the source `Vehicle_ID` is preserved verbatim.
6. **Provenance sidecar:** the source checksum, the filter rules, the
   row/duplicate/anomaly counts, the derived geometry and the output checksum
   are written to `data/traces/ngsim_us101_authentic.provenance.json`.

**Temporal sampling.** The simulator's discrete timestep is 0.5 s while the
source is sampled every 0.1 s.  The replayer uses **nearest-frame
interpolation** on the recorded timestamps; it does not downsample the source
data and it applies **no** smoothing, ramp removal or acceleration filtering.
No trajectory is synthesised beyond the recorded samples.

**Timestamp-scale anomaly.** The Peachtree records use a different epoch
scale from the other three sites (about 1.16e9 versus 1.12e12).  The pipeline
*detects and reports* this anomaly in the provenance sidecar, and the site
filter removes the affected site before any time arithmetic is done.

**Authentic source directory.** Raw source files and download scripts are
isolated in [`data/source/`](file:///c:/Users/LENOVO/Desktop/VANNET/data/source/)
and are never modified by the processing pipeline.
The raw FHWA NGSIM recording provides trajectories sampled at 10 Hz ($dt = 0.1\text{ s}$). When using raw NGSIM files, the standard preprocessing steps are:
1. **Unit Conversion:** Convert coordinates from Imperial (feet, ft/s) to Metric (meters, m/s):
   $$\text{pos\_x} = \text{Local\_X} \times 0.3048, \quad \text{pos\_y} = \text{Local\_Y} \times 0.3048, \quad \text{speed\_mps} = \text{v\_Vel} \times 0.3048$$
2. **Temporal Downsampling:** Downsample from 10 Hz ($0.1\text{ s}$) to the simulation discrete timestep $\Delta t = 0.5\text{ s}$.
3. **Trajectory Filtering:** Remove non-mainline ramps and tracking noise via moving-average Gaussian filtering on acceleration.
4. **Authentic Source Directory:** Raw source files and download scripts are isolated in [`data/source/`](file:///c:/Users/LENOVO/Desktop/VANNET/data/source/) to prevent accidental overwriting.

---

## 3. Synthetic vs. Authentic File Separation

To guarantee data integrity and eliminate any ambiguity:

| Directory / File | Type | Description |
| :--- | :--- | :--- |
| [`data/source/`](file:///c:/Users/LENOVO/Desktop/VANNET/data/source/) | **Authentic Source Reference** | Download instructions, schema mapping, and checksum verification for genuine FHWA NGSIM data. |
| [`data/synthetic_traces/synthetic_ngsim_highway.csv`](file:///c:/Users/LENOVO/Desktop/VANNET/data/synthetic_traces/synthetic_ngsim_highway.csv) | **Calibrated Synthetic Model** | Generated by [`data/generate_traces.py`](file:///c:/Users/LENOVO/Desktop/VANNET/data/generate_traces.py) using an Intelligent Driver Model (IDM) calibrated to NGSIM headway and speed distributions ($16-30\text{ m/s}$). |
| [`data/synthetic_traces/synthetic_manhattan_grid.csv`](file:///c:/Users/LENOVO/Desktop/VANNET/data/synthetic_traces/synthetic_manhattan_grid.csv) | **Synthetic Benchmark** | 4x4 Manhattan urban grid simulation for city mobility benchmarks. |
| [`data/synthetic_traces/synthetic_v2x_task_profiles.csv`](file:///c:/Users/LENOVO/Desktop/VANNET/data/synthetic_traces/synthetic_v2x_task_profiles.csv) | **Standardized Workload Model** | Parametric autonomous vehicle task catalog defined under 3GPP/ETSI specifications. |

---

## 4. 3GPP & ETSI V2X Task Profile Provenance

Task workload attributes ($D_k$, $C_k$, $\tau_k$) are mapped directly from published automotive computing and telecommunication standards:
* **3GPP TR 36.885:** "Study on LTE-based V2X Services", Release 14/15.
* **3GPP TS 22.186:** "Service requirements for enhanced V2X scenarios (Release 16/17)".
* **ETSI TR 103 562:** "Analysis of the Pre-standardization Study on Cooperative Perception Services (CPS)".
* **ETSI GS MEC 002:** "Multi-access Edge Computing (MEC); Use Cases and Requirements".

### Task Catalog Parameter Mapping & Simulation Scaling

> [!IMPORTANT]
> **Parameter classification.** The data sizes, CPU-cycle counts, deadlines and
> energy weights used by the simulator are **modelling assumptions**, not values
> read from a standard.  The standards named below define the *workload
> classes* and the *communication context*; they do **not** publish the exact
> parameters used here, and this repository has not verified any specific
> "full specification" figure against the cited passage.  The previous version
> of this table presented unverified reference numbers in a way that implied
> they were sourced; that has been corrected.
>
> The parameterisation is a **0.1x frame-level scaling**: an offloaded task
> represents an instantaneous frame/sub-burst of a larger workload rather than a
> full multi-second aggregate.  The scaling factor and the resulting
> parameters are assumptions of this study, chosen so that the resulting
> service times are commensurate with the 0.5 s simulation epoch and with the
> 65-280 ms deadlines.  They are documented here so that a reader can
> substitute different values.

| Workload Class | Context (not a parameter source) | Simulation parameter - **modelling assumption** | Deadline - assumption | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cooperative Perception** | ETSI TR 103 562, cooperative perception service | Mean $0.45\text{ MB}$, Std $0.08\text{ MB}$, $0.22\text{ Gcycles}$ | $160\text{ ms}$ | 3 (High) |
| **Sensor Fusion** | 3GPP TS 22.186, sensor-fusion use case | Mean $0.18\text{ MB}$, Std $0.03\text{ MB}$, $0.12\text{ Gcycles}$ | $90\text{ ms}$ | 4 (Critical) |
| **Video Analytics** | ETSI GS MEC 002, video-analytics use case | Mean $0.95\text{ MB}$, Std $0.15\text{ MB}$, $0.40\text{ Gcycles}$ | $280\text{ ms}$ | 1 (Medium) |
| **Trajectory Planning** | 3GPP TR 36.885, V2X service context | Mean $0.08\text{ MB}$, Std $0.02\text{ MB}$, $0.06\text{ Gcycles}$ | $65\text{ ms}$ | 5 (Critical) |

**Realised moments.** The sampler draws from a *truncated* normal bounded to
$[0.01, 4\times\text{mean}]$, so the realised mean and standard deviation are
close to, but not exactly, the tabulated values.
`tests/test_model_validation.py` asserts that the realised mean stays within
2 % (data size) and 5 % (CPU cycles) of the declared value and that the
samples stay inside the declared bounds, so the realised moments are
*reported by the tests* rather than assumed to be exact.

### Non-IID Dirichlet Partitioning
In real deployments, vehicles experience heterogeneous application demands depending on their sensor suite (e.g., LiDAR-equipped autonomous EV vs. basic connected car). Task generation samples from a **Dirichlet distribution**:
$$\mathbf{p}_i \sim \text{Dir}(\mathbf{\alpha}), \quad \alpha = [0.4, 0.4, 0.4, 0.4]$$
This guarantees realistic Non-IID task distributions across the vehicular fleet without relying on arbitrary synthetic assignments.
