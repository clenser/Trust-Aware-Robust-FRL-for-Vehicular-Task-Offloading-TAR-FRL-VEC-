# Phase-by-Phase Coding Agent Prompts and Manual Acceptance Tests

Use one prompt at a time in OpenCode. Do not move to the next phase
until the current phase passes its gate. Keep `AGENTS.md` in the
repository root.

## Phase 0: Audit and setup

### Prompt

Read `AGENTS.md` completely. Perform Phase 0 only: audit this repository
and create a practical implementation plan for the trust-aware federated
DQN VANET project. Inspect existing files, Git status, Python version,
dependencies, and tests before editing. Do not implement the simulation
or algorithms yet. Create/update only the project plan, assumptions
register, initial README, dependency strategy, and configuration
skeleton if appropriate. Preserve all existing user work. Run any
relevant checks and report actual outcomes. Finish with files changed,
test results, manual checks, limitations, and PASS/PARTIAL/FAIL against
Phase 0.

### Expect

-   An audit of what already exists.
-   A phased implementation plan and explicit assumptions.
-   A clear statement that this is a synthetic Python simulation, not a
    packet-level network simulator.
-   No major RL implementation yet.

### Manually test

-   Open `AGENTS.md`, README and the plan.
-   Confirm DQN is the required starting algorithm.
-   Confirm the plan does not require collecting a real dataset, SUMO,
    NS-3, OMNeT++, or pretrained vision models.
-   Check `git diff` to ensure unrelated files were not changed.

## Phase 1: Literature review and mathematical specification

### Prompt

Read `AGENTS.md`. Implement Phase 1 only. Build a literature-review
workflow and mathematical specification before major coding. Create a
paper-tracking table and research-gap matrix template. Use real,
traceable publications only; include DOI or publisher links where
available and never invent citations. If web/research access is
unavailable, create the template and list search queries instead of
fabricating papers. Define the vehicle/edge/task model, finite DQN state
and action space, reward terms, trust equations, communication and
computation assumptions, threat model, research questions, hypotheses,
and measurable metrics. Clearly label unverified research gaps as
provisional. Add tests for any executable mathematical helpers. Report
changed files, evidence, manual checks, limitations and gate status.

### Expect

-   A research plan, real-paper tracking table or clearly marked empty
    template.
-   A finite, implementable DQN action space.
-   Explicit assumptions and equations for task latency, trust and
    reward.
-   No claim of novelty without literature evidence.

### Manually test

-   Click/check every citation and DOI.
-   Verify the state variables are measurable in the simulator.
-   Confirm actions include local execution and eligible edge/vehicle
    destinations.
-   Check reward signs: latency, energy and failures should be
    penalized; timely success/reliability should be rewarded.
-   Confirm trust is bounded between 0 and 1 and temporary disconnection
    is not automatically treated as an attack.

## Phase 2: Synthetic VANET environment

### Prompt

Read `AGENTS.md`. Implement Phase 2 only: a reproducible Python
synthetic vehicular-edge environment. Create vehicle, edge node, task,
mobility, communication/connectivity, resource/queue, and
execution-outcome models. Support configurable low/medium/high mobility,
finite communication range, intermittent connections, heterogeneous
CPU/bandwidth/queues, and task size/CPU cycles/deadline/priority. Use
seeded random generators and document equations/assumptions. Do not
implement DQN or federated learning yet. Add unit tests and a small CLI
smoke simulation. Run tests and the smoke simulation, inspect actual
output, and report exact commands/results and the acceptance gate.

### Expect

-   A simulation that generates vehicles and tasks and determines which
    nodes can communicate.
-   Task execution results affected by resource limits,
    distance/connectivity and deadlines.
-   Reproducible runs when the same seed/configuration is used.

### Manually test

-   Run the documented smoke command.
-   Run twice with the same seed and compare outputs.
-   Change the seed and verify some generated conditions change.
-   Set a node unavailable or overloaded and confirm task outcomes
    change.
-   Check that vehicles outside communication range are not eligible
    destinations.
-   Confirm no RL code was added prematurely.

## Phase 3: Baselines and metrics

### Prompt

Read `AGENTS.md`. Implement Phase 3 only. Add local-execution,
random-offloading, nearest-eligible-node, and resource-aware baseline
policies using the same environment and shared metric definitions.
Handle the case where no remote node is eligible. Export raw
per-task/per-episode results to CSV or JSON with configuration and seed
metadata. Add tests ensuring all policies return valid actions and
metrics are computed consistently. Run a small comparison and report
actual results without claiming any policy is superior unless the run
supports it.

### Expect

-   Four non-learning baselines.
-   Comparable latency, task completion/failure, energy and reward
    metrics.
-   Exported results with seed/configuration information.

### Manually test

-   Run every baseline on the same configuration and seed.
-   Check CSV/JSON files open and contain headers, task counts, seed and
    policy names.
-   Test an empty-neighbour scenario.
-   Verify metric denominators and units make sense.
-   Confirm no made-up results are hard-coded into charts.

## Phase 4: Single-agent DQN

### Prompt

Read `AGENTS.md`. Implement Phase 4 only: a conventional DQN agent with
replay buffer, target network, epsilon-greedy exploration, configurable
learning rate, discount factor, batch size and target-update frequency.
Integrate it with the environment and existing metrics. Save checkpoints
and training logs; create learning/reward curves from real runs. Keep
action/state dimensions validated. Include a small fast smoke
configuration and tests for tensor shapes, action validity, replay
sampling and checkpoint loading. Do not claim convergence from a short
smoke run. Run tests and one smoke training/evaluation and report actual
outputs.

### Expect

-   A trainable DQN baseline that can choose local or remote execution.
-   Training logs and plots generated from real episodes.
-   Reproducible configuration and checkpoint files.

### Manually test

-   Run smoke training from README instructions.
-   Confirm episodes finish and logs/checkpoints are created.
-   Confirm all selected actions are valid for the current state.
-   Inspect reward and loss plots for valid axes and real data.
-   Run evaluation with exploration disabled.
-   Treat unstable reward as a result to investigate, not something to
    hide.

## Phase 5: Dynamic behavioural trust

### Prompt

Read `AGENTS.md`. Implement Phase 5 only. Add a behavioural trust model
using verified task success, response timeliness, resource availability
and communication reliability, with a documented temporal update and
values constrained to \[0,1\]. Integrate trust into destination
ranking/eligibility or the documented reward without changing unrelated
DQN internals. Log trust observations and changes. Add tests for
positive evidence, repeated failures, resource changes, temporal
smoothing, bounds, and temporary disconnection. Do not label a node
malicious based solely on one mobility-related dropout. Run tests and a
small scenario showing trust evolution.

### Expect

-   Trust scores that change with evidence.
-   Logs/plots showing trust over time.
-   Trust influencing offloading decisions according to documented
    rules.

### Manually test

-   Create one consistently reliable and one repeatedly failing node.
-   Verify their trust scores diverge over repeated interactions.
-   Simulate temporary network loss and confirm it is not automatically
    treated as malicious intent.
-   Check scores always remain within \[0,1\].
-   Verify the dashboard or logs explain why trust changed.

## Phase 6: Federated DQN

### Prompt

Read `AGENTS.md`. Implement Phase 6 only. Create simulated federated
clients that train local DQN models on their own local experience and a
coordinator that aggregates model parameters using standard FedAvg.
Support client dropout, intermittent participation, update age/staleness
metadata and global evaluation. Do not transmit replay buffers or raw
experiences. Add tests for aggregation arithmetic, client participation,
model compatibility, stale-update handling and global model
distribution. Compare local DQN with federated DQN in a small
reproducible run. Report actual results and limitations.

### Expect

-   Multiple local clients and one federated coordinator.
-   A global model updated across rounds.
-   Standard FedAvg baseline and participation logs.

### Manually test

-   Run at least two clients for multiple rounds.
-   Confirm global parameters change after eligible updates.
-   Confirm raw replay buffers/trajectories are not sent to the
    coordinator.
-   Disable a client and confirm the round completes safely.
-   Submit a stale update and verify the documented policy is applied.
-   Check model dimensions match across clients.

## Phase 7: Model-update trust and robust aggregation

### Prompt

Read `AGENTS.md`. Implement Phase 7 only. Add model-update trust signals
such as update norm/deviation, consistency across rounds,
stale/copied/zero-update evidence and a carefully defined local-progress
signal where meaningful. Combine model-update trust with behavioural
trust using the documented formula. Implement and compare standard
FedAvg, trust-weighted aggregation, and trust-weighted aggregation with
threshold exclusion. Add poisoning/free-rider test fixtures, audit logs
showing accepted/down-weighted/rejected updates and a safe fallback when
no client qualifies. Avoid assuming every divergent update is malicious
because non-IID clients can legitimately differ. Run controlled tests
and report false positives as well as detection successes.

### Expect

-   Distinct behavioural and model-update trust.
-   Three aggregation strategies.
-   Transparent per-client aggregation decisions.
-   Safe handling when all clients are rejected.

### Manually test

-   Send an honest update and confirm it is considered.
-   Send a zero/copy/stale update and check the defined detection
    behaviour.
-   Send a poisoned update with several strengths and inspect its
    weight.
-   Use honest clients with different data distributions and check they
    are not all rejected for being different.
-   Set all trust scores below threshold and confirm no crash or invalid
    weights.
-   Check aggregation weights are finite and normalized where
    aggregation occurs.

## Phase 8: Attack and dynamic-condition evaluation

### Prompt

Read `AGENTS.md`. Implement Phase 8 only. Add reproducible scenarios for
parameter poisoning, free-riding, unreliable task execution,
intermittent participation, non-IID local experiences and configurable
attack strength. Record ground-truth attacker labels separately from
detector predictions. Support malicious-client ratios of 0%, 10%, 20%,
30% and 40%, and low/medium/high mobility. Add experiment scripts and
metrics for task performance, attack precision/recall,
false-positive/negative rate and model degradation. Keep attack
injection separate from defence logic. Run a small matrix first,
validate outputs, and report actual results without claiming success in
every scenario.

### Expect

-   Switchable attack scenarios with known ground truth.
-   Reproducible results by seed.
-   Security and task-performance metrics.

### Manually test

-   Run with 0% malicious clients as a control.
-   Run with a small malicious fraction and confirm attack labels are
    recorded.
-   Verify poisoning actually changes the intended submitted update.
-   Verify free-riders do not perform the specified meaningful local
    training.
-   Check detection precision/recall calculations against a tiny
    hand-checkable example.
-   Confirm mobility/connectivity changes affect the simulation.

## Phase 9: Visually polished Streamlit dashboard

### Prompt

Read `AGENTS.md`. Implement Phase 9 only. Build a professional Streamlit
dashboard using the tested project core, not duplicated simulation
logic. Include an overview, configurable experiment controls,
simulation/network view, task outcomes, trust evolution, federated-round
status, accepted/rejected clients, convergence, baseline comparisons,
attack metrics and export of actual results. Use consistent typography,
spacing, accessible contrast, restrained colour, clear units and
meaningful chart titles. Make it responsive for typical laptop screens.
Clearly label demo/sample data; never display fabricated benchmark
results as real. Add a dashboard smoke test where practical and update
run instructions. Run the app or its available automated checks and
report actual outcomes.

### Expect

-   A clean project dashboard that looks suitable for a final-year
    project demonstration.
-   Interactive controls that affect real simulation parameters.
-   Charts and tables generated from experiment results.

### Manually test

-   Start the dashboard using the documented command.
-   Change vehicle count, mobility, malicious ratio and seed, then run
    again.
-   Verify results change only when the experiment is rerun and
    correspond to the chosen configuration.
-   Check every chart has title, units/labels and legends where needed.
-   Check empty results, invalid parameters and failed runs show
    understandable messages.
-   Export results and open the file.
-   Confirm no chart uses hard-coded fake benchmark values.

## Phase 10: Full experiments, ablation and sensitivity

### Prompt

Read `AGENTS.md`. Implement Phase 10 only. Build a configuration-driven
experiment runner for baseline comparisons, malicious ratios, poisoning
strengths, mobility, resource heterogeneity, non-IID conditions and
intermittent participation. Implement the mandatory ablations: complete
model; without behavioural trust; without model-update trust; without
temporal trust; without trust-aware aggregation; federated RL without
trust; and RL without federated learning. Run multiple seeds for major
comparisons, save raw data and configurations, and generate plots/tables
with means and standard deviations plus confidence intervals or
justified statistical tests where appropriate. Start with a smoke matrix
and scale only if runtime permits. Never hard-code expected outcomes.
Report incomplete experiments explicitly.

### Expect

-   A repeatable experiment matrix.
-   Raw data, summary tables and publication-quality figures.
-   Comparisons and ablations that reveal which components matter.

### Manually test

-   Run the small experiment matrix end-to-end.
-   Confirm each result row records configuration and seed.
-   Regenerate one plot from saved raw results.
-   Verify averages and standard deviations manually for a tiny sample.
-   Confirm ablation switches really disable the intended component.
-   Check all baselines use the same environment/configuration for fair
    comparison.
-   Verify runtime is practical and interrupted experiments can be
    resumed or clearly marked incomplete.

## Phase 11: Final report and reproducibility

### Prompt

Read `AGENTS.md`. Implement Phase 11 only. Prepare the final technical
documentation and report using only verified sources and measured
results. Document the research gap and limits of the novelty claim,
system assumptions, state/action/reward formulation, trust equations,
federated procedure, threat model, experimental setup, baselines,
results, ablation, sensitivity, statistical validation, limitations and
future work. Include a clear architecture figure and algorithm
pseudocode. Ensure every plotted value can be traced to saved raw
results. Update README with setup, smoke-test, dashboard and experiment
commands. Do not invent citations or improve results by altering data.
Run a clean-environment smoke test where feasible and produce a final
reproducibility checklist.

### Expect

-   A complete report/paper draft and project documentation.
-   Architecture figure and reproducible result plots.
-   Clear limitations and no unsupported claims.

### Manually test

-   Follow README from a clean virtual environment.
-   Run the smoke test and launch the dashboard.
-   Regenerate at least one final figure from raw data.
-   Click every reference and verify bibliographic details.
-   Match each claim in the conclusion to a table/figure or source.
-   Confirm the report distinguishes simulation results from real-world
    deployment evidence.

## Final rule

After every phase, review the agent's report and run the manual
checklist yourself. Do not proceed if the gate is FAIL. If PARTIAL, fix
or explicitly document the missing requirement before continuing. Keep
the repository under Git version control and commit only after reviewing
the diff.
