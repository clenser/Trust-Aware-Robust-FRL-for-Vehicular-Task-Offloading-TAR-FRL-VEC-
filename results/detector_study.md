### A. Participation-rule comparison (synthetic harness)

Identical scenarios and seeds; the rule is selected on the validation suite.

| Suite | Rule | Honest FP | Honest denom. | Honest FPR | Free-rider TPR | Pooled TPR | Pooled FPR |
|---|---|---|---|---|---|---|---|
| validation | all judged | 19 | 160 | 0.1187 | 1.0000 | 1.0000 | 0.1404 |
| validation | excluded | 1 | 142 | 0.0070 | 1.0000 | 1.0000 | 0.0065 |
| heldout | all judged | 19 | 180 | 0.1056 | 1.0000 | 0.9048 | 0.0786 |
| heldout | excluded | 1 | 162 | 0.0062 | 1.0000 | 0.9048 | 0.0034 |

Selected rule (validation only): **exclude non-participants**

### B. Threshold sensitivity (validation selection, held-out assessment)

| tau | validation honest FPR | validation mean attack TPR | held-out honest FPR | held-out mean attack TPR |
|---|---|---|---|---|
| 0.25 | 0.0000 | 0.000 | 0.0000 | 0.000 |
| 0.30 | 0.0000 | 0.000 | 0.0000 | 0.000 |
| 0.35 | 0.0070 | 1.000 | 0.0062 | 0.911 |
| 0.40 | 0.0070 | 1.000 | 0.0062 | 0.911 |
| 0.45 | 0.0070 | 1.000 | 0.0062 | 0.911 |
| 0.50 | 0.0211 | 1.000 | 0.0185 | 0.978 |
| 0.55 | 0.0634 | 1.000 | 0.0494 | 0.978 |
| 0.60 | 0.0775 | 1.000 | 0.0679 | 0.978 |

Selected tau on validation: **0.40** (interval [0.35, 0.4, 0.45]); held-out at that tau: honest FPR 0.0062, mean attack TPR 0.911

### C. Honest false-positive rate vs. honest heterogeneity

| Shared component | tp,fp,fn,tn | Honest FPR [95 % Wilson CI] |
|---|---|---|
| 1.0 | 0,1,0,19 | 0.0500 [0.0089, 0.2361] |
| 0.9 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.8 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.7 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.6 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.5 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.4 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.3 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.2 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.1 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |
| 0.0 | 0,0,0,20 | 0.0000 [0.0000, 0.1611] |

### D. Participation sweep

| Idle honest clients | Rule | Confusion (tp,fp,fn,tn) | TPR | FPR |
|---|---|---|---|---|
| 0 | all judged | 2,0,0,18 | 1.000 | 0.000 |
| 0 | exclude non-participants | 2,0,0,18 | 1.000 | 0.000 |
| 2 | all judged | 2,2,0,16 | 1.000 | 0.111 |
| 2 | exclude non-participants | 2,0,0,16 | 1.000 | 0.000 |
| 4 | all judged | 2,4,0,14 | 1.000 | 0.222 |
| 4 | exclude non-participants | 2,0,0,14 | 1.000 | 0.000 |
| 6 | all judged | 2,6,0,12 | 1.000 | 0.333 |
| 6 | exclude non-participants | 2,0,0,12 | 1.000 | 0.000 |
| 8 | all judged | 2,8,0,10 | 1.000 | 0.444 |
| 8 | exclude non-participants | 2,0,0,10 | 1.000 | 0.000 |
| 10 | all judged | 2,10,0,8 | 1.000 | 0.556 |
| 10 | exclude non-participants | 2,0,0,8 | 1.000 | 0.000 |
| 12 | all judged | 2,12,0,6 | 1.000 | 0.667 |
| 12 | exclude non-participants | 2,0,0,6 | 1.000 | 0.000 |
| 14 | all judged | 2,14,0,4 | 1.000 | 0.778 |
| 14 | exclude non-participants | 2,0,0,4 | 1.000 | 0.000 |

### E. Realistic protocol (real federated rounds, identical seeds)

| Mobility mode | Rule | Confusion (tp,fp,fn,tn) | Honest denom. | TPR [95 % CI] | FPR [95 % CI] | FP by mechanism |
|---|---|---|---|---|---|---|
| highway | all judged | 180,0,0,570 | 570 | 1.0000 [0.9791, 1.0000] | 0.0000 [0.0000, 0.0067] | - |
| highway | exclude non-participants | 180,0,0,570 | 570 | 1.0000 [0.9791, 1.0000] | 0.0000 [0.0000, 0.0067] | - |
| trace | all judged | 179,0,1,570 | 570 | 0.9944 [0.9692, 0.9990] | 0.0000 [0.0000, 0.0067] | - |
| trace | exclude non-participants | 179,0,1,570 | 570 | 0.9944 [0.9692, 0.9990] | 0.0000 [0.0000, 0.0067] | - |
| authentic_trace | all judged | 174,69,6,501 | 570 | 0.9667 [0.9292, 0.9846] | 0.1211 [0.0968, 0.1504] | free_riding_by_norm_ratio=51, joint_trust_below_threshold=18 |
| authentic_trace | exclude non-participants | 165,0,0,519 | 519 | 1.0000 [0.9772, 1.0000] | 0.0000 [0.0000, 0.0073] | - |
