# Specification of the VANNET simulation environment (model version 2.1)

This document states what the simulator **actually implements**, with the
file that implements it.  It supersedes earlier versions of this file, which
contained several values that did not match the code (coverage radius, shadowing
deviation, trust weights, evaluation seed offset); those have been corrected
here and are covered by tests.  Everything in this document is a **modelling
choice**, not a measurement; the status of each element is marked
`[simulated]`, `[trace-replayed]` or `[measured]`.

| Element | Status |
|---|---|
| Vehicle kinematics, `highway` mode | `[simulated]` Intelligent Driver Model |
| Vehicle kinematics, `authentic_trace` mode | `[trace-replayed]` FHWA NGSIM US-101 trajectories (the only authentic data) |
| Wireless channel, interference, rates | `[simulated]` |
| Task arrivals, sizes, cycles, deadlines | `[simulated]` catalogue derived from 3GPP/ETSI service classes, scaled by 0.1 |
| Edge and vehicle queues | `[simulated]` deterministic multi-core servers |
| Rewards, attacks, trust values | `[simulated]` |

Model switches live in `configs/default_config.yaml` under
`environment_model` and `reward`; setting `honor_rsu_choice: false`,
`uplink_contention: false`, `lambda_trust: 2.5`, `state_features: v2.0`
reproduces the earlier simulator versions for before/after comparisons.

---

## 1. Mobility (`environment/mobility.py`)

### 1.1 Procedural highway (`highway`) `[simulated]`

Intelligent Driver Model (Treiber et al., 2000) on a closed loop of
`road_length_m = 3200 m`, five lanes of 3.75 m, `dt = 0.5 s`:

$$\dot v_i = a_{\max}\Big[1-\Big(\tfrac{v_i}{v_0}\Big)^{4}-\Big(\tfrac{s^*}{s_i}\Big)^{2}\Big],\quad
s^*=s_0+\max\!\Big(0,\;v_iT+\tfrac{v_i\Delta v_i}{2\sqrt{a_{\max}b}}\Big)$$

| Parameter | Value |
|---|---|
| desired speed `v_0` | 20-27 m/s, drawn per vehicle |
| minimum gap `s_0`, headway `T` | 16 m, 1.4 s |
| `a_max`, `b` | 2.4, 3.2 m/s² |
| speed clip | 14-32 m/s |

Lanes are fixed (no lane changes); the road is a loop, so vehicles re-enter at
x = 0.  Vehicles are updated in place in sequence, which is a first-order
approximation of a simultaneous update.

### 1.2 Trace replay (`trace`, `authentic_trace`) `[trace-replayed]`

`TraceDrivenMobility` replays a recorded trajectory file.  Source vehicle
identities are preserved: a slot is bound to a source vehicle for as long as it
stays in the observation window and is released (and re-bound to a new
arrival) when it leaves; a vehicle that is not in the window is marked
`active = False`, generates no tasks, consumes no resources and contributes no
training transition.  No position is extrapolated.  When the recording is
exhausted the replay ends (policy `stop`); wrapping is opt-in.

For the authentic file the road segment is 681 m long with 8 lane identifiers
and 2,847 distinct vehicles over 2,772 s (measured on the processed file).
RSUs are placed at 20 %, 50 % and 80 % of the observed segment so that no RSU
lies outside the replayed trajectories.

**Memory.** The simulation steps at 0.5 s while the file is sampled at 0.1 s,
so the frames in between are never read.  The loader measures the geometry on
the full file and then keeps only frames at multiples of the step (replayed
positions are unchanged); this reduced the resident size of the 4.1 M-row file
to about 0.3 GB.

`authentic_trace` authenticates **mobility only**.  NGSIM contains no task
requests, channel measurements, queues, rewards, model updates or attack labels.

---

## 2. Edge deployment (`environment/rsu.py`, `configs/default_config.yaml`)

| RSU | x (m) | lateral y (m) | height (m) | coverage radius (m) | cores | total CPU (GHz) |
|---|---|---|---|---|---|---|
| RSU_1 | 420 | 15 | 14 | 480 | 8 | 32 |
| RSU_2 | 1480 | -42 | 38 | 650 | 16 | 48 |
| RSU_3 | 2460 | 38 | 28 | 550 | 12 | 40 |

Coverage is a 3-D distance test `sqrt(dx² + dy² + (h_rsu - 1.5)²) <= R`.
The covered road interval is therefore `|dx| <= sqrt(R² - dy² - dz²)`, which is
exactly what the viewport draws.  For the procedural highway the three x
positions can be configured (0-3200 m); for trace replay they are fixed.

Each RSU is a **deterministic multi-core server** (not an M/M/c queue): a task
goes to the core that frees up first, waits until it is free and is then served
for `N / f_core`.  The queue is bounded (50); overflow is a drop.  Tasks that
arrive in the same simulation slot all arrive at the slot time, in a seeded
random order (`VANETOffloadingEnv.vehicle_order`), so contention within a slot
is real but unbiased towards low vehicle identifiers.

Vehicles are single-server queues (capacity 15) with an on-board CPU drawn from
U(0.8, 2.0) GHz (automotive SoC class).

---

## 3. Communication (`environment/network.py`)

### 3.1 Path loss and fading

For distance `d` (m, 3-D) and carrier `f_c = 5.9 GHz` (log-distance fits in the
style of 3GPP TR 36.885; **not a conformance claim**):

$$PL_\text{LOS}=32.4+20\log_{10}f_c+20\log_{10}d,\qquad
PL_\text{NLOS}=38.3+17.5\log_{10}d+24.9\log_{10}f_c$$

LOS below 150 m, NLOS above.  Log-normal shadowing is zero-mean Gaussian with
**3 dB** standard deviation for both cases; small-scale fading is an independent
unit-mean exponential power gain (Rayleigh).  One channel realisation is drawn
per interaction and reused for the transmission and the trust update.

### 3.2 Rate

`SINR = P_tx · 10^(-PL/10) · h / (N_0 + I)` with `P_tx = 23 dBm`,
`N_0 = -95 dBm` over the 20 MHz UE channel, and **no interference term**
(`I = 0`).  The rate is Shannon `B log2(1+SINR)` clipped to 0.5-54 Mbps, i.e. a
spectral-efficiency cap of 2.7 bit/s/Hz on 20 MHz.

> **Audit finding.**  Within the coverage radius of these RSUs the SINR is
> high enough that the single-user rate sits at the 54 Mbps cap for most
> samples, so the *single-user* link is almost distance-independent.  Distance
> matters through coverage (a hard cut-off), handover, and the contention
> below, not through a smoothly varying rate.  Adding interference or a
> distance-dependent MCS table is future work.

### 3.3 Uplink contention (version 2)

The cell bandwidth is `cell_bandwidth_mhz = 100` (the 3GPP NR FR1 maximum
channel bandwidth), shared OFDMA-style among the vehicles that transmit to the
same RSU **in the current slot**, this vehicle included:

$$B_\text{user}=\min\!\big(20\text{ MHz},\;100\text{ MHz}/n\big),\quad
R=\min\!\big(54,\;B_\text{user}\cdot\min(\log_2(1+\gamma),\,2.7)\big)\ \text{Mbps}$$

Up to five simultaneous users keep the full 20 MHz channel; beyond that each
user's rate shrinks in proportion.  `T_tx = 8 D / R`, `E_tx = P_tx · T_tx`
(radiated power only; transmitter circuit power is not modelled, so transmit
energy is a **lower bound**).  The model-update uplink of federated learning is
accounted separately (section 8).

### 3.4 Mobility during transmission and handover

The end position is `x + v·T_tx` (modulo the road length on the closed highway
only; a trace replay does not wrap).  If the vehicle leaves the serving RSU's
coverage during the transmission and another RSU covers the end position, a
handover adds 25 ms; otherwise the task is lost.

---

## 4. Computation and energy (`environment/network.py::compute_computation_delay`)

`T_comp = C / f`, `E_comp = κ f² C` with `κ = 1e-30 J·s²/cycle`, i.e. 1 pJ per
cycle at 1 GHz.  RSU energy is `E_comp` on the serving core (4 GHz); local
energy uses the vehicle's own frequency.  Transmission energy dominates the
offloading energy (tens of mJ against a few mJ of computation), which is why
local execution is roughly 30 times cheaper per task in the measured results.

---

## 5. Tasks (`environment/task.py`, `data/synthetic_traces/synthetic_v2x_task_profiles.csv`)

Four service classes with truncated-normal sizes and cycle counts (**simulated
catalogue, scaled by 0.1; not quoted from a standard**): cooperative
perception (0.45 MB, 0.22 Gcycles, 160 ms), sensor fusion (0.18 MB, 0.12 G,
90 ms), video analytics (0.95 MB, 0.40 G, 280 ms), trajectory planning
(0.08 MB, 0.06 G, 65 ms).  Each vehicle draws its task type from a fixed
Dirichlet mixture (concentration 0.4; configurable, `task_profiles.dirichlet_alpha`;
`non_iid: false` gives a uniform mixture), so clients are non-IID.  One task per
vehicle per 0.5 s slot.

A task **succeeds** only if it was accepted **and** its sojourn time (queue wait
+ service, plus transmission and handover for offloading) is within its
deadline - the same criterion for local, RSU and V2V execution.  A failed task
books a penalty delay (1.5 x deadline, 2.0 x for a separated V2V pair) used only
inside the reward; **reported latency is over completed tasks only**.  Failure
reasons are recorded (`out_of_coverage`, `deadline_or_rejected`).

Not modelled: task arrival jitter inside a slot, task dependencies, result
download, priorities in the queue.

---

## 6. Offloading decision problem (`environment/vanet_env.py`)

Actions: 0 local, 1-3 RSU_1..3, 4 best V2V peer.  **The action names an RSU and
is honoured** (version 2): if the vehicle is outside that RSU's coverage no link
exists, nothing is transmitted (zero energy) and the task fails with
`out_of_coverage`; the RSU is not penalised.  The previous version silently
re-associated the vehicle with a better cell, which made actions 1-3
interchangeable (the learned policies were 99 % "offload to any RSU").

**State (version 2.1, 20 features):** task size/10, cycles/5, deadline/300,
speed/35, `min(d_i/R_i, 2)/2` for the three RSUs (0.5 = coverage boundary),
local queue/15, local CPU/8, three RSU behavioural-trust values, best-peer
distance/200, best-peer trust, three normalised RSU loads (queued tasks /
2 x cores), three in-coverage flags.  Version 2.0 (17 features) used
`d_i/(L/2)` and no flags; it is retained as `state_features: v2.0`.

**Reward:**

$$r=\begin{cases}\delta_s-\alpha_\ell\,\min(T/\tau,2)-\beta_e\,E/E_\text{norm}+\lambda_t\,T_\text{target} & \text{success}\\
-\gamma_f-2\alpha_\ell-\mathbb 1[T_\text{target}<0.4] & \text{failure}\end{cases}$$

`δ_s = 3, α_ℓ = 2, β_e = 1, γ_f = 5, E_norm = 10 mJ`, and **`λ_t = 0`** by
default.  With `λ_t = 2.5` and a trust prior of 0.7 for every node, every
offloading action receives a constant +1.75 that local execution does not, which
makes offloading attractive by construction; it is kept as a sensitivity
experiment (experiment I).  The energy term is large relative to the delay term
(a typical offload costs about -3.6 in energy units against about -1 for delay),
so the reward prefers local execution whenever it would meet the deadline.  The
next state does not depend on the action except through queue load, so the
decision problem is close to a contextual bandit; `γ = 0.95` therefore mostly
adds target noise (a sensitivity on `γ` was not run - see LIMITATIONS).

---

## 7. Behavioural trust (`trust/trust_model.py`)

Per node, `T_b(t+1) = η T_b(t) + (1-η) O(t)` with `η = 0.85`, and

$$O=0.30\,B+0.35\,R+0.15\,A+0.20\,C$$

`B` is the Beta-pseudo-count delivery ratio `(7+s)/(10+n)` (prior 7/10 is an
assumption, kept separate from observations), `R` the response reliability
`clip(0.5+0.5·margin, 0.2, 1)` (0 on failure), `A = 1 - utilisation`,
`C = clip((SINR_dB+5)/25, 0, 1)`.  Behavioural trust of a *vehicle* only moves
when that vehicle is used as a V2V helper, which is rare; until then it stays at
the 0.70 prior.

## 8. Model-update trust and federated aggregation

See `docs/METHODOLOGY.md`.  Joint trust is `α T_b + (1-α) T_m` with `α = 0.5`,
exclusion threshold `τ = 0.40`, and a model-trust override (`T_m <= 0.20` caps
the joint trust at 0.30).  **Consequence**
(`tests/test_trust_geometry.py`): for a client at the neutral behavioural prior,
exclusion happens only when `T_m <= 0.20`; moving `τ` between 0.35 and 0.45
changes nothing, and behavioural trust alone cannot exclude a client with an
honest update at `α = 0.5`.

Federated traffic is measured, not assumed: each round every participating
client uploads one fp32 copy of the model (36,614 parameters = 146,456 bytes in the 20-feature model)
and the server broadcasts the aggregate to all clients.


---

## 9. Extensions for the problem statement's edge-node and connectivity requirements

The problem statement asks for a framework that operates under intermittent
connectivity and heterogeneous hardware and that handles *edge nodes and
vehicles* that may be unreliable.  Three model features address this; all are
off by default, so the main benchmark is unchanged.

**Unreliable edge node** (`adversarial.faulty_rsus`, `environment/rsu.py`).  An
RSU can be given `fault_drop_prob` (an admitted task is silently dropped with this
probability) and `fault_delay_factor` (service times are multiplied).  The
vehicle pays the transmission energy, the task fails, and the behavioural trust
tracker records the failure, so the RSU's trust value in the state falls.  The
fault is **injected and scripted**, not measured.  `environment_model.
mask_trust_features` zeroes the four trust entries of the state (three RSU
trust values and the best-peer trust), an ablation that removes behavioural trust
from the policy's information.

**Intermittent connectivity and stale updates** (`federated_learning.
connectivity_dropout`, `missed_broadcast`, `staleness_mode`).  Each client is offline
in a round with the given probability (seeded Bernoulli, independent of coverage).
With `missed_broadcast` an offline client also misses the model broadcast and keeps
training from its old model, so its next upload is *stale* by the number of
rounds it missed; its delta is computed against the current global model and
therefore contains the drift.  With `staleness_mode: discount` the aggregation
weight (sample count) of a client is divided by `1 + staleness` for every
aggregation rule.  The outage process is not derived from mobility; it is a
documented simplification.

**Ground-truth labels.**  Vehicles with role `unreliable` (task-dropping helpers)
are *honest participants of the federated round*: their model updates are genuine.
Only `poisoning` and `free_rider` clients are positives for the model-update
detector; the number of unreliable vehicles that are excluded is reported
separately (`unreliable_total`, `unreliable_excluded`).  An earlier version
counted them as attackers, which capped the measured detection rate.

**Heuristic using behavioural trust.**  `Trust_Greedy` is the greedy
minimum-estimated-delay heuristic that skips an RSU whose behavioural trust is
below 0.5, a fair non-learning user of behavioural trust for the comparison with
the learned policy.

**Console performance.**  The interactive console draws task sizes with a fast
rejection sampler (same distribution; Kolmogorov-Smirnov test in
`tests/test_task_sampling.py`; different random stream) because scipy's truncated
normal costs about 4.5 ms per task.  Benchmarks never use it
(`task_profiles.fast_sampling: false`), so their workloads are unchanged and
paired across runs.  The per-vehicle task mix is cached (a pure function of the
vehicle id, identical output).
