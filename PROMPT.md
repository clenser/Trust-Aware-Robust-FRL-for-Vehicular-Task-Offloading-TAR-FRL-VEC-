# MASTER ENGINEERING MISSION — TAR-FRL-VEC
## Turn the current repository into a working, research-defensible and presentation-ready vehicular edge-computing system

You are the lead research engineer, Python simulation engineer, reinforcement-learning engineer, federated-learning/security researcher, QA engineer, and product/UI engineer for this repository.

Repository:
https://github.com/clenser/Trust-Aware-Robust-FRL-for-Vehicular-Task-Offloading-TAR-FRL-VEC-

Local Windows workspace:
C:\Users\srbar\Desktop\VANNET

Research title:
**Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in Vehicular Edge Networks Under Dynamic Edge Conditions**

Your mission is to improve the EXISTING project into a technically correct, end-to-end working research prototype with a polished interactive dashboard and defensible experimental evidence.

This is not a request to create another plan, merely write documentation, add decorative graphs, or redesign the dashboard without fixing the simulation. You must inspect, implement, test, and verify the actual application.

---

# 0. NON-NEGOTIABLE RULES

1. Work in the existing repository. Do not replace it with a new scaffold or unrelated demo.
2. Start by inspecting the working tree, active branch, current HEAD, recent commits and uncommitted changes. Do not overwrite local work.
3. Do not commit, push, reset, rebase, force-push, delete branches, or discard user changes. Leave all Git operations that publish or destroy work to the user.
4. **Preserve the historical Phase 10 benchmark completely.** The matrix contains 880 runs and its manifest covers 894 artifacts. Do not regenerate, retrain, rewrite, rename, delete or overwrite those artifacts. Run the existing manifest checker before and after your changes. Any integrity failure is a release blocker.
5. Do not claim that tests, SUMO, the browser, screenshots or experiments passed unless you actually executed and verified them. Separate tests that genuinely ran from tests that skipped, used mocks, or only checked availability.
6. Never fabricate data, metrics, confidence intervals, literature citations, simulation results, screenshots or experimental success.
7. Do not silently substitute the synthetic mobility environment when SUMO was requested. A fallback is acceptable only when it is explicitly selected or clearly reported as a fallback.
8. Do not confuse a 3D animation with a functioning simulation. Every displayed vehicle position, speed, lane, connection, offloading event, handover and KPI must derive from the actual episode trace or an explicitly labelled illustrative mode.
9. Avoid unnecessary dependencies, complex frameworks and costly cloud APIs. Prioritize a reliable local Windows setup and the existing Python/Streamlit/Three.js architecture.
10. Use incremental, reviewable changes. Do not start the large experiment matrix before correctness tests and small end-to-end experiments pass.
11. Never claim publication readiness simply because the application starts or the test suite is green.
12. Continue through implementation and verification wherever possible. Do not stop after producing an audit or asking for approval for ordinary, reversible code changes.

If an implementation requirement cannot be completed in the current environment, implement what can be verified, document the precise blocker, and provide a reproducible next command. Never fake the missing result.

---

# 1. ESTABLISH THE REAL CURRENT STATE

Inspect the repository at its actual local HEAD, not only an old report or GitHub README.

Inspect at least:

- README.md, AGENTS.md, requirements.txt and all project configuration files.
- `environment/vanet_env.py`, `environment/config.py`, `environment/mobility.py`, `environment/network.py`, `environment/task.py`, `environment/vehicle.py`, `environment/edge_node.py`.
- `environment/sumo_adapter.py`, `environment/sumo_env.py`, and all files under `environment/sumo/`.
- All relevant `rl/`, `trust/`, `federated/`, `attacks/`, `baselines/`, `experiments/`, and `specs/` modules.
- `dashboard/app.py`, `dashboard/components.py`, `dashboard/simulator_engine.py`, `dashboard/three_scene.py`, `dashboard/styles.py`, `dashboard/data_loader.py`, and bundled static assets.
- SUMO integration tests, existing unit/regression tests, experiment scripts, configurations, checkpoints, smoke results and research reports.
- The latest commit history and the actual diffs between the last known working version and current HEAD.

First record:

- Active branch and HEAD SHA.
- Clean/dirty Git status and all existing user modifications.
- Python version, virtual environment, dependency availability, SUMO binary, SUMO_HOME, TraCI and sumolib status.
- Current test results, skipped tests and failures.
- Existing manifest-verification result.
- Dashboard startup result.
- Whether SUMO genuinely launches and advances through TraCI.
- Whether the latest SUMO smoke output can be reproduced.
- Whether a browser-rendered screenshot can be captured and inspected.

Do not modify files during this initial audit.

Create `docs/ENGINEERING_AUDIT.md` containing verified findings, defects, evidence, severity and planned fixes. Continue into implementation after the audit.

---

# 2. DEFINE THE SCIENTIFIC SYSTEM CORRECTLY

The project is a hybrid vehicular-edge research simulator.

## 2.1 Microscopic mobility

Use Eclipse SUMO and Python TraCI for:

- Individual vehicle coordinates and trajectories.
- Multi-lane traffic.
- Car-following and lane-changing.
- Acceleration, deceleration, speed and headway.
- Vehicle insertion, departure and temporary absence.
- Traffic density and traffic-dependent connectivity.
- RSU association and handovers.

The existing highway is approximately 1,200 metres long with three lanes and two RSUs. Verify its actual geometry and configuration before relying on these values.

## 2.2 Analytical models

The following may remain mathematical abstractions, provided they are correct, documented, internally consistent and validated:

- Task arrivals and workload generation.
- Task payload, required CPU cycles, priority and deadline.
- Wireless path loss, Shannon-style link rate and fading.
- Transmission and propagation delay.
- Vehicle and edge computation.
- Server queues.
- Energy consumption.
- Participant trust.
- DQN training and federated aggregation.

Do not describe the model as packet-level networking or physical RF ray-tracing.

## 2.3 Research objective

Investigate whether combining dynamic behavioural trust and federated model-update trust can protect federated reinforcement learning against malicious, free-riding and unreliable participants while maintaining competitive task latency, completion rate and energy consumption.

The minimum complete solution must contain:

1. A working vehicle-edge environment.
2. Validated local and offloading decisions.
3. DQN and federated DQN functionality.
4. Behavioural trust.
5. Model-update trust.
6. Trust-aware robust aggregation.
7. Explicit attack implementations.
8. Strong baseline comparisons.
9. Controlled, reproducible experiments.
10. Honest statistical evaluation.
11. An inspectable 3D simulation.
12. A polished research dashboard.

Do not add blockchain, digital twins, 6G or additional RL algorithms unless the literature review identifies a specific need.

---

# 3. PRIORITY ONE — REPAIR THE ACTUAL SUMO/OFFLOADING PIPELINE

Trace the complete causal path:

SUMO state → active vehicles → connectivity and candidate selection → task assignment → action mask → policy action → task execution → latency/energy/success → behavioural evidence → trust update → episode telemetry → dashboard and exported metrics.

Every transition must be traceable and tested.

## 3.1 Correct environment time and state

Inspect `SumoTrafficSimulator` and `SumoVehicularEdgeEnv`.

Verify:

- TraCI launches with the expected configuration and deterministic seed.
- The simulation step is advanced exactly as intended.
- Vehicle positions, speed, lane, acceleration, heading and active status are retrieved correctly.
- Vehicle IDs remain stable during a run and are not incorrectly reused after departure.
- New vehicles and departing vehicles are handled correctly.
- Reset restarts or resets SUMO cleanly and does not leak subprocesses or TraCI connections.
- Simulation termination and error handling are reliable.
- The environment cannot accidentally keep stale vehicles after they leave the SUMO network.
- The SUMO environment and synthetic environment share a documented observation/action contract.
- A failed SUMO startup cannot silently produce a synthetic result labelled as SUMO.

Use TraCI subscriptions where useful to reduce repetitive retrieval overhead. If profiling shows socket overhead is material, evaluate libsumo as an optional performance improvement rather than introducing it without evidence.

## 3.2 Correct the offloading decision

The policy must make a real decision among feasible destinations.

For every task:

1. Select the requester from the current eligible population.
2. Generate a task with recorded data size, compute cycles, priority and deadline.
3. Determine the currently feasible local, V2V and V2I/RSU destinations.
4. Compute distance and link feasibility using current state.
5. Construct a valid action mask.
6. Select an action using the requested policy.
7. Execute the task through the same well-defined task model for every policy.
8. Record the selected destination and all intermediate calculations.
9. Update the trust evidence for the participant that actually executed the task.
10. Advance the episode and record the next state.

Do not silently convert invalid offloading actions into local execution. Explicitly distinguish invalid action, local execution, infeasible destination, link failure, execution failure and deadline miss.

Make all policies use the same workload, environment parameters and evaluation seed when conducting paired comparisons. Random policies must be seeded and must not share hidden state with other policies.

## 3.3 Repair the smoke experiment's identical-policy issue

The committed smoke CSV reports identical results for resource-aware, nearest-edge and local-only policies:

- Success rate: 55%.
- Mean latency: approximately 11.08 seconds.
- Mean energy: approximately 36.89 J.
- Offloading fraction: 0.

Do not hard-code policy-specific output to make the results look different.

Trace the action path and determine why these policies produce identical metrics. Investigate:

- Whether the action mask excludes all offloading candidates.
- Whether SUMO positions or RSU distances are stale.
- Whether local-only is implemented correctly.
- Whether heuristic policies receive different candidate/resource information.
- Whether task generation and evaluation are accidentally identical or reused.
- Whether destination IDs are mapped incorrectly.
- Whether action values are converted or clipped incorrectly.
- Whether all offloading actions incur an unintended failure penalty.
- Whether the observation dimension and action mapping match the trained model.

Add tests that deliberately create environments in which local execution, RSU offloading and V2V offloading should lead to different expected behaviour. The tests must check the selected action, destination and resulting metrics—not merely that the environment runs.

## 3.4 Repair handover and association accounting

The existing smoke report has `handover_count = 20`, while the per-policy CSV reports zero `total_handovers`.

Define these as separate events:

- Initial association.
- RSU-to-RSU handover.
- Disconnection from coverage.
- Reconnection after a gap.
- Vehicle departure.

A handover is an actual transition from one valid RSU association to a different RSU association. Initial association and disconnection are not handovers.

Use a consistent event schema with at least:

`episode_id`, `time_s`, `simulation_step`, `vehicle_id`, `sumo_vehicle_id`, `event_type`, `from_rsu`, `to_rsu`, `x_m`, and relevant task ID when applicable.

Do not count the same event multiple times. Aggregate the event log and the summary from the same source of truth.

Add deterministic tests for:
- Outside coverage → first association.
- RSU 0 → RSU 1.
- RSU association → disconnection.
- Disconnection → reconnection.
- Vehicle departure.
- No transition → no handover.

Check the actual coverage geometry and overlapping RSU regions. The configured corridor, RSU positions and communication radius must agree across environment logic, telemetry, renderer and documentation.

## 3.5 Model task execution and queues consistently

Inspect the mathematical equations and code paths for:

- Upload and download payload.
- V2I/V2V transmission rate.
- Transmission latency.
- Handover signalling latency.
- Edge or vehicle CPU capacity.
- Queue waiting time and execution time.
- Energy consumption.
- Deadline success.

Use SI units internally and convert only for display. State explicitly whether queue time, propagation delay and retransmission are included.

If M/M/1 or FIFO approximations are used, implement and document the assumptions correctly. Do not use a formula outside its valid operating conditions. Handle unstable or overloaded queues explicitly.

The handover penalty must be applied only to tasks that actually experience the relevant handover condition; do not add a blanket penalty to unrelated tasks.

Create unit tests with hand-calculated cases for local execution, edge execution, weak connectivity, a deadline miss, an out-of-range destination and a handover.

---

# 4. PRIORITY TWO — MAKE THE RESEARCH METHOD END TO END

Audit the actual algorithms against `specs/`, the research roadmap and the report.

## 4.1 Trust model

Trace behavioural trust through task outcomes, response timeliness, resource availability and communication reliability.

Trace model-update trust through parameter/update norm, direction, staleness, copied or zero updates, progress evidence and any other implemented indicators.

Verify that:

- Every trust indicator is bounded and mathematically well-defined.
- Neutral non-participation is not automatically treated as malicious behaviour.
- Mobility-induced disconnection can be distinguished from intentional free-riding where the evidence supports that distinction.
- New clients receive a documented prior.
- Trust evolves at the correct time scale.
- Invalid or missing update metadata is handled explicitly.
- Honest non-IID clients are not automatically rejected merely because their updates differ from the majority.
- The final trust score is calculated from the documented formula.
- Thresholds are configurable and tested, not arbitrarily tuned on the test outcomes.

Log each participant's trust components, combined score, evidence, aggregation eligibility and final decision.

## 4.2 Federated reinforcement learning

Verify that each client actually trains a local model and that the server aggregates compatible model updates.

Trace one full round:

1. Distribute the global model.
2. Select available clients.
3. Train each participating client using its local experience/data.
4. Generate a local update.
5. Apply configured malicious or unreliable behaviour when enabled.
6. Calculate behavioural and model-update trust from valid evidence.
7. Apply the selected aggregation strategy.
8. Update the global model or use the documented safe fallback.
9. Distribute the new global model.
10. Evaluate the resulting model on a held-out evaluation environment.
11. Record convergence, performance and overhead.

Check that the global model really changes when valid updates are aggregated. Verify state-dict compatibility, finite values, deterministic client ordering, sample counts, trust weights and exclusion rules.

Do not report successful learning merely because a checkpoint loads. Record training reward and held-out policy performance.

## 4.3 Baselines and attacks

Preserve the existing policies and compare at least:

- Local-only execution.
- Random-valid offloading.
- Nearest feasible edge.
- Resource-aware heuristic.
- Single-agent DQN.
- Standard FedAvg-based federated DQN.
- Trust-weighted aggregation.
- Trust-threshold aggregation.
- At least one suitable published robust aggregation baseline after a literature audit.

Investigate median, trimmed mean, Krum or another relevant robust aggregator. Implement only methods whose assumptions are satisfied by the number of clients and threat model. If an algorithm's assumptions do not hold, mark it ineligible rather than presenting an invalid comparison.

Support reproducible attack scenarios for:

- Zero-update and stale-update free-riding.
- Model poisoning, including a justified sign-flip or perturbation attack.
- Unreliable task execution.
- Intermittent participation.
- Honest non-IID clients.

Test attacks at 0%, 10%, 20%, 30% and 40% where the client population supports those proportions. Report the actual number of malicious participants and attack strategy for every run.

## 4.4 Novelty and research claims

The original roadmap explicitly requires a literature-backed research gap. Do not declare the method novel because it combines several established techniques.

Create or update:

- `docs/LITERATURE_REVIEW.md`
- `docs/RESEARCH_GAP_MATRIX.csv`
- `docs/RESEARCH_QUESTIONS_AND_HYPOTHESES.md`
- `docs/MATHEMATICAL_MODEL.md`
- `docs/THREAT_MODEL.md`

Use actual papers with verifiable title, authors, year, venue and DOI or official URL. Review recent federated RL/VEC offloading and trust-aware or Byzantine-robust aggregation. Compare the exact mechanism, threat assumptions, baselines and limitations of each paper.

Identify which research gap remains after the literature audit. If the existing method does not support a defensible novelty claim, state this honestly and recommend a narrower, testable contribution. Do not invent a stronger contribution to make the paper sound impressive.

---

# 5. PRIORITY THREE — USE REAL MOBILITY DATA RESPONSIBLY

Use public trajectory data to validate mobility; do not treat it as a substitute for the entire research environment.

Investigate:

- NGSIM from the U.S. Department of Transportation.
- highD, subject to its access and licensing requirements.
- Additional public traffic trajectories only when relevant and legally usable.

Record official source, citation, licence/access requirements, download instructions, file checksums and preprocessing decisions.

Do not commit large raw datasets to Git. Provide a script or documented process that downloads or imports data into an ignored data directory when permitted. If access requires a request or acceptance of terms, explain that requirement and do not bypass it.

Build a trajectory-analysis and calibration tool that can calculate, as supported by the dataset:

- Speed and acceleration distributions.
- Headway distributions.
- Lane-change rates.
- Vehicle density and flow.
- Time-space trajectories.
- Vehicle count over time.

Compare SUMO outputs with the selected real dataset under comparable road geometry and traffic conditions. Report distribution differences and calibration parameters, not just screenshots.

If feasible, create a trace-replay mode for recorded trajectories. Clearly distinguish replayed observations from live SUMO vehicle dynamics. Do not claim that replayed trajectories provide measured wireless channels, task arrivals or edge computation.

Maintain a clear provenance label for every experiment: `SUMO_SYNTHETIC_DEMAND`, `REAL_TRAJECTORY_REPLAY`, or another precise mode.

---

# 6. PRIORITY FOUR — BUILD A PRESENTABLE, ACTUALLY INTERACTIVE DASHBOARD

Retain the existing Streamlit application and Three.js renderer unless the audit proves a structural replacement is necessary. Focus on coherent information architecture, correct data flow, readability and real functionality.

The dashboard must explain the research to someone who has not seen the code.

## 6.1 Overall layout

Use a professional research-lab visual system: clean typography, consistent spacing, clear section hierarchy, restrained colours, accessible contrast and responsive panels.

Create a clear primary navigation structure such as:

1. **Overview** — research objective, current run status and key findings.
2. **Live Simulation** — 3D highway and controls.
3. **Task Offloading** — decisions, destinations, latency and energy.
4. **Trust & Security** — trust evolution, attack activity and detection errors.
5. **Federated Learning** — clients, rounds, aggregation and convergence.
6. **Experiments & Results** — baselines, density, attack ratios, confidence intervals.
7. **Mobility Validation** — SUMO versus public trajectory statistics.
8. **Reproducibility** — configuration, seeds, provenance, manifest and exports.

If the existing app has a simpler navigation model, achieve this structure without breaking existing functionality or needlessly duplicating the app.

## 6.2 Overview

Show understandable, correctly calculated KPIs:

- Active vehicles.
- Current traffic density.
- Mean vehicle speed.
- Tasks processed.
- Task completion rate.
- Mean task latency.
- Mean energy per task.
- Current RSU associations and true handover count.
- Current experiment/policy.
- Current trust/attack state when applicable.

Each KPI must have a clear unit and explanatory tooltip or caption. Do not present historical Phase 10 metrics as if they came from the live SUMO run.

## 6.3 Live 3D simulation

Keep the daytime highway visualisation, but verify the actual rendering in a browser.

Requirements:

- Real vehicle positions from the selected simulation trace.
- Correct lane placement and vehicle orientation.
- Smooth interpolation between recorded frames without inventing research telemetry.
- Play, pause, reset, speed control, step controls and a working timeline scrubber.
- Camera presets, orbit/zoom and follow-vehicle mode.
- Clear RSU locations and coverage boundaries.
- Visible task-requester and destination identity.
- Offloading communication animation only when an offload action actually occurs.
- Separate visual styles for V2I, V2V and local execution.
- Correct success, deadline-miss and failure indicators.
- Vehicle inspector with ID, speed, lane, position, connected RSU and task status.
- Useful empty, loading and error states.
- Responsive layout without clipped controls or panels.

The animated path must interpolate between trace frames. The task outcome, trust values, selected action and KPIs must remain tied to recorded simulation events. Interpolation must not change the recorded experiment metrics.

Do not rely on a CDN for essential Three.js assets when local assets are available. If a dependency fails, show an explicit diagnostic rather than an empty canvas.

## 6.4 Live analytics

Build charts from real trace and experiment data, not static arrays or hand-authored illustrative values.

Required plots include:

- Active vehicles over simulation time.
- Speed distribution and mean speed.
- Lane occupancy and density.
- RSU association over time.
- Genuine handover events over time.
- Task latency distribution.
- Completion and deadline-miss rates.
- Local/V2V/V2I action distribution.
- Trust score history and exclusion events.
- Per-round client participation.
- Global reward or held-out performance over FL rounds.
- Attack detection precision, recall, F1, FPR and FNR.
- Baseline performance with uncertainty intervals.

Each chart must explain its axes, units, sample count and data source. When data is unavailable, show a useful empty state instead of fake values.

## 6.5 Experiment controls

Allow the user to select:

- Mobility engine.
- Traffic density.
- Random seed.
- Number of tasks.
- Number of vehicles or traffic-flow settings where supported.
- Offloading policy.
- RSU configuration where supported.
- Attack type and malicious-client fraction.
- Federated rounds and client participation.
- Trust parameters and aggregation strategy.

Validate combinations before execution. Make the selected configuration visible throughout the run and in all exported outputs.

Use Streamlit session state and caching carefully. Avoid rerunning an expensive experiment on every UI interaction. Add clear run, stop/cancel where safely implementable, progress, completion and error states. Do not block the UI indefinitely when SUMO hangs or terminates unexpectedly.

## 6.6 Exports

Allow users to export the current run's:

- Configuration JSON/YAML.
- Task-level results CSV.
- Vehicle trace CSV.
- RSU event CSV.
- Trust history CSV.
- FL round/client metrics.
- Summary JSON.
- Experiment metadata and seed.
- Figures generated from the selected experiment.

Use separate, clearly named directories for new experiments. Never write live results into the historical Phase 10 artifact directory.

---

# 7. PRIORITY FIVE — EXPERIMENTS THAT CAN SUPPORT RESEARCH CLAIMS

The existing SUMO smoke test is only an integration check. Do not treat it as a scientific comparison.

First create small deterministic tests and short experiments. Only after they pass, build a reproducible matrix of experiments.

## 7.1 Mobility validation

Test at least low, medium and high traffic densities. Ensure the scenarios actually produce different vehicle counts, speeds, headways or other relevant statistics. Do not trust the scenario label alone.

For every run record:
- Seed and configuration.
- SUMO version and simulation step.
- Actual vehicle count and density.
- Speed, acceleration and headway distributions.
- Vehicle arrivals and departures.
- Association, handover and disconnection events.
- Warnings, collisions or teleports where relevant.

## 7.2 Policy evaluation

Use paired seeds and comparable workloads for the baseline policies. Ensure that the environment and tasks are not accidentally identical because of a state-reset or seeding bug.

Report:
- Task completion and deadline-miss rate.
- Mean, median and percentile latency.
- Energy per task.
- Local/V2V/V2I selection rates.
- Communication failures.
- Queue delays.
- Handover frequency.
- Throughput.
- Per-policy runtime.

Do not assume offloading must always outperform local execution. Use scenarios where offloading is expected to help and scenarios where it is expected to be disadvantageous, and verify that the outcomes respond appropriately.

## 7.3 Security and FL evaluation

For each relevant malicious-client ratio, evaluate FedAvg, trust-weighted aggregation, trust-threshold aggregation and the selected robust baseline under matched seeds.

Measure:
- Held-out reward/performance.
- Task latency, completion and energy.
- Model degradation under attacks.
- Attack detection precision, recall, F1, FPR and FNR.
- Honest-client false exclusions.
- Client participation and exclusion rates.
- Global-model convergence.
- Training and aggregation time.
- Communication and model-update overhead.
- Memory usage when practical to measure.

Perform the mandatory ablations:
1. Complete proposed method.
2. Without behavioural trust.
3. Without model-update trust.
4. Without temporal trust.
5. Without trust-aware aggregation.
6. Federated RL without trust.
7. Non-federated RL baseline.

Run sensitivity analysis on the trust weighting, temporal update, exclusion threshold, client count and malicious fraction.

## 7.4 Statistics

Use multiple random seeds for major comparisons. Report means, standard deviations and appropriate confidence intervals. Use paired tests for paired-seed comparisons when assumptions are reasonable.

Correct for multiple comparisons where appropriate. State the test, sample count, assumptions and exact result. Report effect sizes and uncertainty, not only p-values.

Do not tune parameters on the same evaluation runs used to claim improvement. Keep a clear separation between training, validation and held-out evaluation.

If the trust-aware method does not significantly improve task performance, report that. It may still show security benefits, but those benefits must be measured and distinguished from task-performance claims.

---

# 8. FIX THE EXPERIMENT AND DATA INTEGRITY ARCHITECTURE

Create a consistent experiment record with a unique run ID and metadata.

Every result should include:
- Experiment configuration.
- Seed.
- Mobility engine and traffic scenario.
- Policy and checkpoint identity.
- Attack configuration.
- Task and vehicle counts.
- Simulation time horizon.
- Dependency versions.
- Result metrics.
- Artifact paths and checksums.

Build a single, tested aggregation path so that CSV, JSON, dashboard KPIs and plots are derived from the same raw event/result data.

Add invariant checks for:
- Total tasks = successes + deadline misses + actual failures, using explicit mutually exclusive definitions.
- Handover count = number of `handover` events, not total event rows.
- Offload fraction = offloaded tasks / eligible evaluated tasks, with a documented denominator.
- Success rate and deadline-miss rate use explicit denominators.
- Aggregated statistics agree with task-level records.
- Reported sample count equals actual valid observations.
- No NaN/Infinity values silently enter results.
- Repeated seeds/configurations are distinguishable.
- No live run overwrites a previous run.

Create a small reproducible SUMO experiment script with a documented CLI and a machine-readable report. Keep it separate from the Phase 10 benchmark.

---

# 9. AUTOMATED TESTING AND REAL BROWSER VERIFICATION

Add tests for the real defects discovered during the audit. Use mocks for isolated unit tests, but do not confuse mocked tests with actual SUMO integration tests.

Required test layers:

### Unit tests
- Task and latency equations.
- Action masks and invalid actions.
- Trust calculations and temporal updates.
- Aggregation and malicious updates.
- Event classification and handover accounting.
- Result aggregation and invariant checks.
- Data validation and export schemas.

### SUMO integration tests
- Detect the installed binary and TraCI.
- Launch a headless SUMO process.
- Advance multiple simulation steps.
- Observe actual vehicles moving and changing state.
- Verify lane and speed telemetry.
- Verify association, handover and disconnection logic with a scenario that actually triggers them.
- Run a small traced episode.
- Verify correct teardown after success and exceptions.
- Test clear failure behaviour when SUMO is unavailable.

If SUMO is unavailable, mark integration tests as skipped with an explicit reason. Do not report them as executed integration coverage.

### Regression tests
- Existing test suite.
- Existing synthetic environment.
- Policy/checkpoint loading.
- Dashboard AppTest.
- Phase 10 manifest integrity.

### Browser/UI tests
Launch Streamlit and open the actual dashboard using an available browser automation tool. If a browser automation dependency is missing, report that fact and use the strongest available alternative.

Inspect at least:
1. Overview page.
2. Live simulation initial state.
3. Playing and pausing.
4. Scrubbing the timeline.
5. Vehicle selection/inspection.
6. RSU coverage and offloading overlay.
7. Charts after an actual SUMO trace.
8. Experiment completion and error states.
9. Exported data consistency.
10. A narrow viewport or responsive layout.

Capture screenshots only after the real application has loaded. Inspect the screenshots for layout defects, illegible text, broken rendering, missing controls and misleading visuals. A successful Streamlit AppTest alone does not prove that Three.js rendered correctly in a browser.

---

# 10. PERFORMANCE, STABILITY AND PORTABILITY

Target the existing Windows developer environment first.

- Keep the existing `.venv` unless there is a demonstrated reason to recreate it.
- Check Python and dependency compatibility before adding packages.
- Keep SUMO discovery robust and diagnostic.
- Ensure SUMO subprocesses and TraCI connections are closed.
- Avoid leaking processes across repeated Streamlit reruns.
- Use sensible timeouts and actionable error messages.
- Avoid launching multiple simulations unintentionally.
- Cache immutable datasets and static assets, not mutable live simulation state.
- Avoid loading all historical experimental artifacts into the live simulation page.
- Profile before optimizing.
- Keep the 3D trace payload small enough for normal laptop use.
- Keep the synthetic fallback usable for development, but visibly label it.

Document a clean setup and run procedure for Windows. Add a quick smoke test that can be completed before any long experiment.

---

# 11. REQUIRED IMPLEMENTATION PHASES

Execute the following phases in order.

## Phase A — Audit and baseline
Inspect repository and local state. Record test and manifest baseline. Reproduce the SUMO smoke discrepancies. Create the audit report.

## Phase B — Simulation correctness
Repair the SUMO lifecycle, vehicle-state synchronization, candidate generation, action masks, offloading execution, task accounting, and handover logic. Add deterministic regression tests.

## Phase C — Trust and federated correctness
Validate the full trust and aggregation path. Fix algorithmic defects, add attack and ablation tests, and verify global model updates.

## Phase D — Reproducible evaluation
Repair the smoke experiment and output schema. Build short low/medium/high-density experiments, paired-policy evaluation and robust metric aggregation. Do not run the full research matrix until the smaller checks pass.

## Phase E — Dashboard and UX
Refactor the existing dashboard into a coherent research application. Connect every KPI and chart to real results. Improve the 3D simulation controls, vehicle inspector, traffic analytics and experiment status.

## Phase F — Real trajectory validation
Add data provenance and a reproducible trajectory import/analysis/calibration workflow using legally accessible public data. Do not make this a blocker for the basic local simulation if dataset access requires approval.

## Phase G — Verification
Run the complete applicable test suite, the SUMO integration tests on a host where SUMO is actually installed, the manifest check, AppTest, a real short SUMO episode and a browser inspection. Fix regressions before reporting completion.

## Phase H — Documentation and delivery
Update README and technical documentation so they reflect the actual implementation, supported commands, scientific assumptions, verified findings and known limitations. Create a final engineering report and an acceptance checklist.

Do not skip phases just to make the work appear finished. If time or tool limits prevent completing a phase, report its exact status and continue with the next highest-value correctness work.

---

# 12. HISTORICAL BENCHMARK PROTECTION

The existing Phase 10 v4 matrix is frozen.

Before changes, run:

` .venv\Scripts\python.exe scripts\verify_phase10_manifest.py --check `

Use the exact repository command if the current environment differs.

After changes, run it again.

The required result is that all 894 historical artifacts still match their original SHA-256 manifest. If the checker fails, stop research experiments, identify the exact changed file, and restore integrity without overwriting unrelated user changes.

All new SUMO outputs must go under a separate, clearly versioned directory, such as `results/sumo_experiments/`. Never reuse or mutate the historical Phase 10 result directory.

---

# 13. ACCEPTANCE GATES — NO FALSE PASS

The project is not complete until the following gates are evaluated with evidence.

### Gate 1 — Environment
- SUMO genuinely launches through TraCI.
- Multiple simulation steps advance correctly.
- Vehicle state changes are observed.
- Clean teardown works.

### Gate 2 — Offloading
- A traced task has an identifiable requester, task parameters, valid action mask, chosen destination, actual execution outcome and complete latency/energy breakdown.
- Local, V2V and V2I actions work where feasible.
- Different policies produce different decisions in deliberately discriminating test scenarios.
- Identical outcomes are explained when they are legitimate.

### Gate 3 — Handover
- Initial association, handover, disconnection and reconnection are separate event types.
- Deterministic tests trigger each event.
- All handover summaries equal the number of true handover events.

### Gate 4 — Trust and FL
- Local model training and aggregation are demonstrated.
- The global model update is validated.
- Behavioural trust and model-update trust affect the documented decision path.
- Attacks and honest non-IID clients are tested.
- Exclusions and fallback decisions are auditable.

### Gate 5 — Evidence
- Task-level results agree with summaries and plots.
- Multi-seed policy evaluation works.
- Statistical results are generated from actual independent runs.
- No research claim exceeds the evidence.

### Gate 6 — Dashboard
- The actual application loads.
- The live simulation is visible.
- Playback and inspection work.
- The analytics change when a different experiment runs.
- Charts and metrics use real data.
- No critical UI exceptions or broken controls remain.

### Gate 7 — Reproducibility
- Commands and configuration are documented.
- The smoke experiment can be repeated.
- Outputs include seed, provenance and metadata.
- The historical manifest is unchanged.

### Gate 8 — Delivery
- Relevant tests pass.
- Skipped tests are disclosed.
- Browser verification is recorded.
- Remaining limitations are explicit.
- No commits or pushes have been made without permission.

A gate that has not been tested is **NOT VERIFIED**, not PASS.

---

# 14. FINAL DELIVERABLES

Produce:

1. `docs/ENGINEERING_AUDIT.md`
2. `docs/MATHEMATICAL_MODEL.md`
3. `docs/THREAT_MODEL.md`
4. `docs/RESEARCH_GAP_MATRIX.csv`
5. `docs/RESEARCH_QUESTIONS_AND_HYPOTHESES.md`
6. `docs/MOBILITY_VALIDATION.md`
7. `docs/EXPERIMENT_PROTOCOL.md`
8. Updated project README and Windows quick-start instructions.
9. Corrected SUMO environment and task-offloading logic.
10. Regression and SUMO integration tests.
11. Reproducible experiment scripts and result schemas.
12. A polished, fully connected research dashboard.
13. Browser screenshots of the verified working interface, when the tooling allows it.
14. `reports/ENGINEERING_ACCEPTANCE_REPORT.md` with each gate marked PASS, FAIL or NOT VERIFIED and linked to actual evidence.

The acceptance report must state the precise Git SHA inspected, actual test counts, skipped tests, manifest result, SUMO availability, smoke-run configuration, policy metrics, browser-verification status and remaining defects.

---

# 15. FINAL WORKING PRINCIPLE

Prioritize in this order:

**Correct simulation → correct task decisions → correct trust and FL → valid experiments → honest evidence → clear visualization → polished UI.**

The objective is not to make the dashboard look sophisticated while the underlying policies are inert. The objective is to make the application demonstrate, in a way a researcher can inspect, that vehicle mobility changes connectivity, connectivity changes offloading feasibility, offloading changes task outcomes, participant behaviour changes trust, trust affects aggregation, and the resulting model is evaluated against meaningful baselines.

Begin by inspecting the current local repository and preserving existing changes. Then execute the phases, fixing the highest-severity issues first. Do not claim completion until the acceptance gates have been evaluated.