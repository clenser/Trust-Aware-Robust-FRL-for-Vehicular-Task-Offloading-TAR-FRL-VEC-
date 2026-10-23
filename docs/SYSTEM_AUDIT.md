# System audit: attacks, V2V and the console (23 Oct 2026)

Requested: "Is there actually an attack by a malicious vehicle in the simulation?
I also don't see many V2V interactions.  Check whether these behaviours are
implemented correctly and realistically."  Each finding below was verified by
running the code, not by reading the documentation.

## 1. Is there an attack by a malicious vehicle?

**Before this audit: only in the federated-learning plane, and invisible in the
scene.**

* `poisoning` and `free_rider` vehicles corrupt or withhold their *model upload*
  (`federated/client.py`).  As V2V helpers they were identical to honest vehicles
  (`Vehicle.execute_as_v2v_helper` only misbehaved for the separate `unreliable`
  role, whose default ratio is 0).  No offloaded task was ever affected by an
  attacker, so nothing malicious was visible in the 3D scene, and the behavioural
  trust tier had nothing to detect - which is part of why the benchmark found it
  "adds nothing".
* The console's event feed labelled a task "role poisoning" using the role of the
  *requesting* vehicle, not of the node that served it (wrong semantics).

**Now (all default-off in the offline benchmark so published results are
unchanged; default-on in the console):**

* `adversarial.data_plane` - poisoning vehicles drop 70 % of the V2V tasks they
  are given and slow the rest 3x (sabotage); free riders refuse 80 % (selfish).
  Verified: V2V success against a poisoning helper falls to ~0 % while honest
  helpers are unaffected (`tests/test_data_plane_and_edge.py`).
* The console draws a magenta "V2V HELPER SABOTAGE" link when a task offloaded to a
  malicious helper fails, counts sabotaged tasks, and labels the helper's role
  correctly (`ground_truth_malicious` now refers to the *target*).
* The scripted attack is an assumption (fixed probabilities), stated as such.

## 2. V2V interactions

Findings, with the measured evidence:

| Finding | Evidence | Action |
|---|---|---|
| Peers are almost always in range (0.7 % of vehicles have none), so V2V is *available*. | 25 vehicles, 3.2 km, 180 m range | none needed |
| V2V is physically dominated: vehicle SoCs are 0.8-2 GHz, the same as the requester, so a peer is never meaningfully faster; V2V tasks succeed ~11-15 % of the time. | random-V2V run | realistic for equal vehicles; added optional compute-rich helpers (`mobility.vehicle_compute`, an assumption) |
| No heuristic baseline ever chose V2V, and the learned policy rarely does. | `baselines/heuristics.py`, `_greedy_delay_action` | added `Greedy_V2V` and `Trust_V2V` baselines (50 Mbps nominal link, the measured ~54 Mbps) |
| Peer choice was "nearest", ignoring trust, queue and CPU, although the state exposed "best-peer trust". | `vanet_env.py` | added `network.v2v_peer_selection: trust_aware`; the state feature and the action now use the same rule |
| **Bug:** in trace replay a vehicle absent from the observation window could be chosen as a peer. | `step_vehicle` peer loop had no `active` check | fixed (`_v2v_candidates`); affects trace modes only |
| Global NumPy RNG used for roles / CPUs / drop draws. | `reset()` | unchanged (seeded by the runner); noted in LIMITATIONS |

## 3. Console / UI

* Side panels were always open and the viewport was ~42 vh.  Now: Controls and
  Monitor panels collapse (state remembered), the monitor is tabbed (Learning /
  Trust / History), sections fold, the viewport fills the window, and there are
  Focus and Full-screen modes.
* 3D navigation was clamped (pitch 0.08-1.45 rad, zoom 40-3200, look-at locked to
  the road centre-line and auto re-centred).  Now: right-drag / shift-drag pans,
  WASD / arrow keys fly, Q / E move down / up, pitch +-, zoom 4-9000, R or
  double-click resets; auto-follow stops while the camera is free.
* The legend is hidden behind a button; the page layout was checked in Chrome.

## 4. Other items

* **NGSIM hash check**: a trace with a `.provenance.json` is verified against its
  recorded SHA-256 when loaded (`VANNET_SKIP_TRACE_HASH=1` skips it).
* **Results in the repository**: `.gitignore` no longer excludes the evidence;
  the 270 MB raw dump is shipped gzipped under `evidence/` with checksums.
