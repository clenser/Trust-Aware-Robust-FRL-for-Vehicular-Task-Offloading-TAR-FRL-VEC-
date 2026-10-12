# Phase 6 & Phase 7 — Federated DQN and Robust Aggregation

> Single-process **synthetic simulation** of federated learning.
> Supports FedAvg (Phase 6), Trust-Weighted FedAvg (Phase 7, EQ-22), and
> Trust-Aware Aggregation with Threshold Exclusion & Fallback (Phase 7, EQ-23).
> No attacks or malicious clients implemented in Phase 7 (scheduled for Phase 8).

---

## 1. Supported Aggregation Strategies

Configured via `strategy` in `FedServer` or config YAML (`federated.strategy`):

| Strategy | Equation | Selection & Weighting | Fallback Policy |
| :--- | :--- | :--- | :--- |
| `fedavg` | EQ-21 | Sample-count weighted: \(w_i = n_i / \sum n_j\) | Revert to previous global model on error/empty |
| `trust_weighted` | EQ-22 | Trust-weighted: \(w_i = T_{\text{final}}(i) / \sum T_{\text{final}}\) | Revert to previous global model on zero total trust |
| `trust_threshold` | EQ-23 | Eligible set \(S = \{i : T_{\text{final}}(i) \ge \tau_{\text{exc}}\}\); \(w_i = T_{\text{final}}(i) / \sum_{S} T_{\text{final}}\) | Retain last known good global model unchanged if \(S = \emptyset\) |

Default parameters:
- Combination weight: \(\alpha = 0.60\)
- Exclusion threshold: \(\tau_{\text{exc}} = 0.30\)
- Norm clipping multiplier: \(C = 3.0\)

---

## 2. Server Architecture (`federated/server.py`)

- `FedServer(template_state, strategy="fedavg", alpha=0.6, tau_exc=0.3, clip_c=3.0)`:
  - Maintains `template`, `global_state`, and `round_log`.
  - Stamped staleness: `current_round - round_trained`.
  - When evaluating trust: invokes `trust.evaluate_model_updates()` with global reference model and client behavioural summaries.
  - Aggregates parameters in `float64` precision; casts back to template dtype.
  - Returns detailed round log: `status`, `participants`, `accepted`, `excluded`, `weights`, `staleness`, `upload_bytes`, `download_bytes_per_client`, `agg_time_s`, and `trust_reports`.

---

## 3. Communication & Privacy Invariants

- **Zero Replay Transmission:** Clients package only `client_id`, `round_trained`, `num_samples`, `staleness`, and parameter `state`. Raw transitions never leave the client's local buffer.
- **Analytical Payloads:** Payloads are estimated as 4 bytes per floating-point parameter (`payload_bytes`), recorded in logs.

---

## 4. Evaluation and CLI Usage

```powershell
# Run smoke test comparison
.venv\Scripts\python.exe scripts\evaluate_trust_aggregation.py --config configs\feddqn_smoke.yaml --seeds 0 1

# Train with specific strategy
.venv\Scripts\python.exe scripts\train_feddqn.py --config configs\feddqn_default.yaml
```
