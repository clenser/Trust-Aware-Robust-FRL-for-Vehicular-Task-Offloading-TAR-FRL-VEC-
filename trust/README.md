# Phase 5 & Phase 7 — Dynamic Behavioural and Model-Update Trust

> Standalone trust modules covering:
> - Phase 5: Dynamic Behavioural Trust (EQ-12, EQ-13) in `trust/behavioural.py`
> - Phase 7: Model-Update Trust & Multi-Signal Combined Trust (EQ-14 through EQ-20) in `trust/model_update.py`
>
> All trust scores live in \([0, 1]\). Neutral ageing and non-IID variations are never conflated with malice.

---

## 1. Behavioural Trust (SPEC §6, `trust/behavioural.py`)

Per observed execution on node \(i\):
```text
s1 = success ? 1 : 0                                   (verified outcome)
s2 = path_ok ? clamp(1 - L/τ) : last_s2[i] (else 0.5)  (timeliness; missing carried, not punished)
s3 = clamp(expected_exe / actual_exe)                  (advertised-vs-observed resources)
s4 = path_ok ? 1 : 0                                   (communication reliability)
T_beh = 0.4·s1 + 0.25·s2 + 0.2·s3 + 0.15·s4            (EQ-12)
T(i,t) = 0.9·T(i,t-1) + 0.1·T_beh                      (EQ-13 temporal update)
idle:  T = 0.99·T + 0.01·0.5                           (EQ-13 neutral ageing toward prior)
```

`expected_exe` uses the destination queue/freq captured at decision time (before dynamics advance); `actual_exe` is the realised execution part.
Defaults (\(\beta\), \(\lambda=0.9\), \(\rho=0.99\), prior 0.5) live in `EnvConfig` (`trust_beta/lambda/rho/prior`) and are configurable.

---

## 2. Model-Update Trust Indicators (SPEC §7, `trust/model_update.py`)

For client update \(\Delta_i = \theta_i - \theta_{\text{prev\_global}}\) vs round aggregate / global model:

### EQ-14: Norm Plausibility (\(m_1\))
```text
m_1 = 1.0  if ||Δ_i|| <= C · median({||Δ_j||})
      clamp01(C · median / ||Δ_i||)  otherwise
```
Default clipping multiplier: \(C = 3.0\).
If median is 0: returns 1.0 if \(||\Delta_i|| == 0\), else 0.0.

### EQ-15: Directional Agreement (\(m_2\))
```text
m_2 = (1 + cos(Δ_i, Δ_bar)) / 2
```
where \(\bar{\Delta} = \frac{1}{|P|} \sum_{j \in P} \Delta_j\) is the peer aggregate direction (no root dataset required).
Evaluated with `float64` precision. Boundary cases (zero deltas) return 0.5 neutral agreement (or 1.0 if both zero).

### EQ-16: Staleness Trust (\(m_3\))
```text
m_3 = 1 / (1 + age_rounds)
```
where `age_rounds` = `current_round - round_trained`. Age 0 yields 1.0.

### EQ-17: Copy / Zero Flags (\(m_4\))
```text
m_4 = 0.0  if ||Δ_i|| < tol OR ||Δ_i - Δ_j|| < tol for any peer j != i
      1.0  otherwise
```
Default tolerance: `tol = 1e-7`. Detects zero parameter deltas and duplicated/copied updates without local training.

### EQ-18: Local Progress Proxy (\(m_5\))
```text
m_5 = clamp01(progress_proxy)  (or 0.5 if unavailable / neutral prior)
```
Advisory low weight (\(\gamma_5 = 0.1\)) acknowledging gameability of self-reported indicators (limitation U4).

### EQ-19: Composite Model-Update Trust (\(T_{\text{model}}\))
```text
T_model = Σ_l γ_l · m_l
```
Default indicator weights: \(\gamma = (0.25, 0.30, 0.15, 0.20, 0.10)\) for (norm, direction, staleness, originality, progress).

---

## 3. Combined Trust (SPEC §7, EQ-20)

```text
T_final = α · T_beh + (1 - α) · T_model
```
Default combination factor: \(\alpha = 0.60\).
Both components are in \([0, 1]\), ensuring \(T_{\text{final}} \in [0, 1]\).

---

## 4. Key Properties & Invariants

1. **Deterministic Execution:** Fixed seeds and sorted parameter keys guarantee byte-reproducible calculations.
2. **Honest Non-IID Safety:** Legitimate gradient diversity across honest vehicles does not lead to unfair exclusion.
3. **No Malice Assumptions:** Ageing and missing timeliness remain neutral, never automatically punitive.
4. **Isolated Tensors:** Model updates only contain state dictionaries and metadata; no raw transitions or replay buffers are transmitted.
