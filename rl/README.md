# Phase 4 — Single-Agent DQN for Task Offloading

> Learned baseline on the real Phase 2 env. No federation, trust, attacks,
> or dashboard. Reward norms provisional (see below) — do not over-interpret
> small margins over heuristics.

## Architecture

MLP `obs_dim → hidden → hidden → action_dim` (ReLU), Adam + MSE against the
masked Bellman target, separate target network, uniform replay, epsilon-greedy
over mask-valid actions only. See `rl/agent.py`, `rl/dqn.py`.

## Commands

```powershell
.venv\Scripts\python.exe scripts\train_dqn.py --config configs\dqn_smoke.yaml     # pipeline check
.venv\Scripts\python.exe scripts\train_dqn.py --config configs\dqn_default.yaml   # 25 eps x 50 tasks
.venv\Scripts\python.exe scripts\evaluate_dqn.py --config configs\dqn_default.yaml
```

## Reward (exact, provisional)

`task_reward` = SPEC EQ-11 defaults with `L̄=clamp(L/10s)`, `Ē=clamp(E/10J)`,
destination trust 0.5. Non-finite latency/energy raise. Norms are NOT tuned;
final calibration is Phase 10 work.

## Limitations

- One training run (seed 0, episodes seeded 1000+); per-episode workloads
  differ by seed, so the training curve mixes learning with task-stream
  variance — held-out eval (seeds 10–12) is the real comparison.
- 25 episodes × 50 tasks is small; loss fell (≈4.5→≈3.8) but nothing has
  converged; treat all DQN numbers as preliminary.
- Fairness caveat (Phase 3): shared config+seeds, but trajectories can diverge
  across policies (on-demand RNG draws) — not perfectly paired.
