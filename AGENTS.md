# AGENTS.md

## Trust-Aware Federated Reinforcement Learning for Secure and Efficient Task Offloading in VANETs

This file is the project-wide contract for any coding agent (OpenCode,
Codex, Claude Code, or similar). Read it before changing code. Implement
only the current phase requested by the user. Do not skip gates,
fabricate test results, or silently change the research scope.

## 1. Project goal

Build a final-year research project that simulates vehicles offloading
computational tasks to nearby vehicles or roadside edge servers. The
system must learn offloading decisions, track participant trust, train
local reinforcement-learning agents, aggregate their updates
federatively, and test resilience against malicious or unreliable
participants.

The research contribution is not simply combining VANET, trust,
federated learning, and RL. It must be a clearly specified and
experimentally tested mechanism. The central candidate contribution is a
dynamic trust model combining: - Behavioural trust: verified task
success/failure, response timeliness, resource availability, and
communication reliability. - Model-update trust: update
magnitude/deviation, similarity or consistency across rounds, evidence
of local learning progress, and stale/copied/zero-update detection. -
Trust-aware robust aggregation: dynamically weight eligible client
updates and optionally exclude clients below an empirically selected
threshold.

Treat this as a research hypothesis, not a proven result. Confirm the
research gap against actual papers before claiming novelty.

## 2. Binding scope and constraints from the project roadmap

-   Use a Python-based synthetic or trace-driven vehicular-edge
    environment.
-   Do not require SUMO, NS-3, or OMNeT++ for the minimum project.
-   Do not collect a real-world dataset manually. Generate reproducible
    synthetic tasks, mobility, connectivity, and resource conditions.
    Trace-driven data is optional, not a blocker.
-   Do not train computer-vision models or download pretrained
    perception models. Workloads such as object detection are
    represented by task size, CPU cycles, deadline, and priority.
-   Start with DQN, as specified in the roadmap. Do not begin with SAC,
    blockchain, digital twins, 6G, SDN, complex consensus, or
    multi-agent architectures unless the literature review demonstrates
    a need. Keep DQN as the required minimum; any additional RL
    algorithm is an optional later comparison.
-   Use Python 3.11+, NumPy, Pandas, PyTorch, Matplotlib, and pytest.
    Use custom, understandable federated-learning code initially; Flower
    is optional only if it materially helps.
-   Use Streamlit for a polished, accessible, interactive research
    dashboard so the project remains Python-centred. Keep the simulation
    and algorithm logic independent from the UI.
-   Work incrementally. Do not build the entire project in one change.
-   Never claim a test, experiment, plot, benchmark, security guarantee,
    or result succeeded unless it was actually run and its output
    inspected.
-   Keep the code runnable on a typical student laptop or Google Colab.
    Provide small smoke-test configurations and separate larger
    experiment configurations.
-   No secrets, API keys, or paid services should be required.
-   Do not fabricate citations, papers, numerical results, or novelty
    claims.

## 3. Required research questions

RQ1. How can trust be dynamically estimated when participant behaviour,
connectivity, and available resources change? RQ2. Does combining
behavioural trust with model-update trust identify malicious
participants better than behavioural trust alone? RQ3. Does trust-aware
offloading improve task latency, completion, and reliability? RQ4. How
does performance change as the proportion of malicious participants
increases? RQ5. Does the method remain useful under mobility,
intermittent connectivity, heterogeneous resources, and non-IID local
experience?

## 4. Required system behaviour

### 4.1 Synthetic VANET environment

Represent vehicles, edge/RSU nodes, tasks, positions, speeds,
communication range, connectivity, CPU capacity, CPU utilisation, queue
length, bandwidth, battery/energy cost, and trust. Generate tasks with
data size, required CPU cycles, deadline, and priority. Include
low/medium/high mobility and intermittent connections. Document all
mathematical assumptions.

### 4.2 DQN offloading agent

Use a discrete action space: local execution, offload to an eligible
neighbouring vehicle, or offload to an eligible edge/RSU node. Keep the
state vector documented and normalised. Use a reward based on latency,
energy, task failure/deadline violations, reliability, and selected-node
trust. Explain reward coefficients and avoid relying on reward alone for
hard safety requirements.

### 4.3 Trust management

Keep behavioural trust and model-update trust separate, then combine
them: T_final(i,t) = alpha \* T_behaviour(i,t) + (1-alpha) \*
T_model(i,t) with all trust values constrained to \[0,1\]. Update trust
temporally, document smoothing and decay, and avoid treating every
mobility-related disconnect as malicious. A node may be temporarily
unavailable without being adversarial.

### 4.4 Federated reinforcement learning

Each client trains a local DQN from its own generated experience.
Clients send model updates/weights and permitted metadata, not raw
experience buffers. The coordinator aggregates updates and distributes a
global model. Document the difference between simulated clients and real
deployed vehicles.

### 4.5 Attacks and robust aggregation

At minimum implement: - Model poisoning (e.g. controlled parameter
perturbation and/or sign-flip). - Free-riding (zero, copied, or stale
updates without meaningful local training). - Unreliable execution
(failures, delayed responses, inconsistent availability). - Intermittent
participation and stale updates.

Compare standard FedAvg, trust-weighted aggregation, and trust-weighted
aggregation with threshold-based exclusion. Validate update anomalies;
trust weights alone are not a complete poisoning defence. Handle the
case where no client is eligible without division by zero or destroying
the last known good global model.

### 4.6 Professional visual interface

Create a visually polished Streamlit research dashboard, not just raw
terminal output. Use a consistent layout, typography, spacing, colour
system, meaningful charts, clear labels, and responsive behaviour. Avoid
excessive gradients, decorative clutter, fake live data, or unsupported
claims. Show: - Project overview and current experiment configuration. -
Simulation controls: vehicles/clients, edge nodes, mobility, task load,
malicious-client ratio, connectivity, and random seed. - Network/vehicle
status and task-offloading outcomes. - Trust scores and their changes
over time. - Federated rounds, accepted/rejected clients, and global
reward/convergence. - Comparison plots for latency, completion rate,
energy, attack detection, false positives, and overhead. - Baseline
selector and experiment history/export where feasible. - A clear
distinction between illustrative demo runs and completed benchmark
experiments. Use deterministic sample/demo data only when explicitly
labelled as demo data. All charts must be driven by actual experiment
outputs.

## 5. Required baseline methods

Implement and clearly identify: 1. Local execution. 2. Random
offloading. 3. Nearest eligible node. 4. Resource-aware offloading. 5.
Single-agent DQN. 6. Federated DQN with standard FedAvg. 7.
Trust-weighted federated DQN. 8. Proposed complete trust-aware federated
DQN with update trust and robust aggregation. 9. At least one relevant
published baseline, after selecting a paper and documenting what was
reproduced.

Do not claim a published baseline was reproduced if only a simplified
approximation was implemented. Label approximations honestly.

## 6. Evaluation plan

Vary and record: - Malicious-client fraction: 0%, 10%, 20%, 30%, 40%. -
Mobility: low, medium, high. - Vehicle/client population: start small
(e.g. 20), scale later if compute permits. - Resource heterogeneity:
CPU, bandwidth, queue length, task arrival rates. - Non-IID client task
distributions. - Intermittent participation and update staleness. -
Poisoning strength. - Trust smoothing coefficient, trust-combination
alpha, and exclusion threshold.

Report: - Mean task latency. - Task completion/deadline satisfaction
rate. - Task failure rate and offloading reliability. - Energy
consumption. - Mean episode reward and convergence. - Attack detection
precision/recall, false-positive and false-negative rates where ground
truth is available. - Global-model degradation under attack. -
Communication, computation, memory, and aggregation overhead.

Use multiple random seeds for major experiments; report mean and
standard deviation and confidence intervals or suitable statistical
tests when appropriate. Never present one run as definitive evidence.

Mandatory ablations: - Complete model. - Without behavioural trust. -
Without model-update trust. - Without temporal trust update. - Without
trust-aware aggregation. - Federated RL without trust. - RL without
federated learning.

## 7. Repository structure

Keep modules separated and straightforward. Suggested structure:

``` text
trust-aware-vanet-frl/
  AGENTS.md
  README.md
  requirements.txt
  pyproject.toml
  configs/
    smoke.yaml
    default.yaml
    experiments.yaml
  environment/
    vanet_env.py
    vehicle.py
    edge_node.py
    task.py
    mobility.py
    network.py
    tests/
  rl/
    dqn.py
    replay_buffer.py
    agent.py
    reward.py
    tests/
  trust/
    behavioural.py
    model_update.py
    temporal.py
    tests/
  federated/
    client.py
    server.py
    fedavg.py
    trust_aggregation.py
    tests/
  attacks/
    poisoning.py
    free_riding.py
    unreliable_node.py
    tests/
  baselines/
    local.py
    random_offload.py
    nearest_node.py
    resource_aware.py
  experiments/
    runner.py
    metrics.py
    ablation.py
    sensitivity.py
  dashboard/
    app.py
    components.py
    styles.py
  results/
    .gitkeep
  reports/
    figures/
    tables/
  scripts/
  tests/
```

The agent may adapt this structure if the existing repository has sound
conventions. Inspect the repository before creating or moving files.
Never overwrite existing work blindly.

## 8. Engineering rules

-   Start every phase by inspecting the repository, current branch,
    files, dependencies, and tests.
-   Before editing, state a short implementation plan and list the files
    expected to change.
-   Implement only the requested phase. Do not start later phases
    without permission.
-   Write unit tests for core logic and integration/smoke tests for
    workflows.
-   Run relevant tests, formatting/lint checks if configured, and a
    minimal end-to-end run.
-   Fix failures introduced by the change. If a check cannot be run, say
    exactly why.
-   Avoid giant monolithic files, hidden global state, hard-coded
    experiment results, and silent exception handling.
-   Set random seeds and persist experiment configuration with results.
-   Use type hints and docstrings for public classes/functions; use
    readable names.
-   Keep mathematical assumptions and equations documented in README or
    reports.
-   Avoid unnecessary dependencies and keep installation instructions
    current.
-   The dashboard must call the same tested core functions used by
    scripts, not duplicate simulation logic.
-   Never fake a security result or label a heuristic as a formally
    guaranteed Byzantine defence.
-   Preserve user changes. Do not delete or rewrite files unrelated to
    the active phase.

## 9. Phase workflow and acceptance gates

A phase is complete only when its expected deliverables exist, tests
run, manual checks pass, and the agent reports evidence. A successful
build alone is not enough.

### Phase 0: Repository audit and research plan

Deliver: repository audit, project plan, requirements, research-question
list, assumptions register, and initial README outline. No major
algorithm implementation. Gate: scope matches this file; roadmap
constraints are acknowledged; no unsupported novelty claim.

### Phase 1: Research evidence and mathematical specification

Deliver: literature search plan, paper-tracking CSV/Markdown table,
research-gap matrix template populated only with verified sources,
system model, state/action/reward definitions,
task/communication/computation assumptions, threat model, and
hypotheses. Gate: every paper is real and traceable; research gap is
phrased as a question pending evidence; DQN action space is finite and
implementable.

### Phase 2: Synthetic VANET environment

Deliver: vehicle/edge/task models, mobility, connectivity, resource and
queue model, task execution outcomes, seeded generators, unit tests, and
a basic CLI simulation. Gate: same seed produces repeatable outcomes;
moving vehicles can enter/leave range; disconnections and resource
limits affect task results; no RL required yet.

### Phase 3: Baselines and metrics

Deliver: local, random, nearest-node and resource-aware policies; shared
metric definitions; CSV/JSON result export; tests. Gate: baselines make
valid decisions, handle no eligible server, and produce sensible,
reproducible metrics.

### Phase 4: Single-agent DQN

Deliver: DQN, replay buffer, target network, epsilon-greedy exploration,
reward implementation, training/evaluation scripts, checkpoints,
learning curves. Gate: shape/range tests pass; actions are valid; a
smoke run completes; learning curves and rewards come from real runs. Do
not claim convergence without evidence.

### Phase 5: Dynamic behavioural trust

Deliver: trust signals, temporal update, trust-aware destination
eligibility/ranking, logging and tests. Gate: successful timely
behaviour improves trust; repeated failures reduce it; temporary
disconnection is not automatically labelled malicious; trust remains in
\[0,1\].

### Phase 6: Federated DQN

Deliver: simulated clients, local training, update exchange,
coordinator, standard FedAvg, global evaluation, participation/dropout
handling. Gate: raw replay buffers are not transmitted; local clients
train; global model updates; interrupted clients and stale updates are
handled.

### Phase 7: Model-update trust and robust aggregation

Deliver: update quality indicators, model trust, combined trust,
trust-weighted aggregation, threshold exclusion, safe fallback for no
eligible clients, comparisons against FedAvg. Gate: synthetic
poison/free-rider cases are detected or down-weighted according to
defined rules; honest non-IID updates are not automatically rejected
merely because they differ; all decisions are logged.

### Phase 8: Adversarial and dynamic scenarios

Deliver: poisoning, free-riding, unreliable execution, intermittent
participation, non-IID task distributions, configurable attack strengths
and malicious ratios. Gate: attacks are reproducible and labelled;
ground-truth malicious status is recorded; metrics distinguish detection
from task-performance impact.

### Phase 9: Polished research dashboard

Deliver: Streamlit interface, controls, KPI cards, network/decision
visualisation, trust and federation charts, baseline comparisons,
experiment status, exportable results. Gate: controls change actual
simulation parameters; dashboard charts use actual run outputs; errors
are understandable; demo data is explicitly labelled; layout is readable
on laptop-size screens.

### Phase 10: Full evaluation, ablation and sensitivity

Deliver: scripted experiment matrix, multiple seeds, baseline
comparison, attack-rate sweeps, mobility/resource/non-IID tests,
ablation and sensitivity analysis, confidence intervals or justified
statistics. Gate: every table/figure can be regenerated from saved
configuration and raw results; no fabricated values; proposed method is
not assumed to win.

### Phase 11: Final report and reproducibility

Deliver: complete README, architecture figure, algorithm pseudocode,
methodology, threat model, experiment setup, results, limitations,
conclusion, references, reproducibility instructions, and final
presentation/demo checklist. Gate: every claim maps to code, a verified
source, or measured evidence; assumptions and limitations are explicit;
a clean environment can run the smoke test.

## 10. Standard coding-agent prompt template

Use this template for every phase:

> Read `AGENTS.md` completely. Work only on Phase \[NUMBER: NAME\].
> First inspect the repository and report its current state, then
> provide a concise plan and the exact files you intend to change.
> Implement the phase to its acceptance gate without starting later
> phases. Add or update tests. Run the relevant tests and a minimal
> end-to-end check, inspect the actual outputs, and fix regressions. Do
> not fabricate test results, benchmark results, citations, or claims.
> Preserve unrelated user changes. Update documentation and
> configuration. At the end, report: (1) files changed, (2)
> implementation summary, (3) commands run and actual outcomes, (4)
> manual checks I must perform, (5) known limitations, and (6) the exact
> acceptance-gate status. If a requirement is ambiguous or a check
> cannot run, state it instead of guessing.

## 11. Required final report from the agent after each phase

The agent must answer in this structure: 1. **Phase and scope** 2.
**What was implemented** 3. **Files changed** 4. **Tests executed and
actual results** 5. **Manual test checklist** 6. **Known limitations /
unresolved failures** 7. **Acceptance gate: PASS / PARTIAL / FAIL**,
with evidence 8. **Recommended next phase**, without starting it

Do not accept "done" as proof. Accept a phase only when code, tests,
actual outputs, and manual checks support the claim.
