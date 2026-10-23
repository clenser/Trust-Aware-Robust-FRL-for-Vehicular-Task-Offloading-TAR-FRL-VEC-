# Detector comparison on held-out scenarios

Identical clients, partitions, seeds and attacks for every detector; 20 independent held-out seed bases (9001-9020), 20 clients per scenario. Pooled over seeds; Wilson 95 % intervals; N/A = undefined denominator.

Krum assumes a fixed Byzantine fraction of 0.25 (a design constant, not the realised attacker count).  `no_detection` is the reference row.

| Scenario | deployed_trust TPR / FPR | krum TPR / FPR | cosine_median TPR / FPR | distance_mad TPR / FPR |
|---|---|---|---|---|
| honest_iid | N/A / 0.007 | N/A / 0.250 | N/A / 0.000 | N/A / 0.015 |
| honest_non_iid | N/A / 0.003 | N/A / 0.250 | N/A / 0.000 | N/A / 0.005 |
| honest_benign_outlier | N/A / 0.060 | N/A / 0.250 | N/A / 0.000 | N/A / 0.060 |
| honest_varying_participation | N/A / 0.000 | N/A / 0.250 | N/A / 0.000 | N/A / 0.000 |
| honest_small_legitimate_update | N/A / 0.000 | N/A / 0.250 | N/A / 0.000 | N/A / 0.000 |
| honest_fully_disjoint | N/A / 0.007 | N/A / 0.250 | N/A / 0.487 | N/A / 0.010 |
| honest_non_iid_strong | N/A / 0.003 | N/A / 0.250 | N/A / 0.015 | N/A / 0.015 |
| adversarial_sign_flip | 0.98 / 0.010 | 1.00 / 0.000 | 1.00 / 0.000 | 0.87 / 0.000 |
| adversarial_sign_flip_partial_50 | 0.64 / 0.007 | 0.99 / 0.003 | 0.46 / 0.000 | 0.67 / 0.000 |
| adversarial_sign_flip_partial_25 | 0.28 / 0.013 | 0.80 / 0.067 | 0.07 / 0.000 | 0.32 / 0.000 |
| adversarial_scaling_unseen_2x | 0.97 / 0.000 | 1.00 / 0.000 | 0.00 / 0.000 | 1.00 / 0.000 |
| adversarial_scaling_unseen_8x | 1.00 / 0.000 | 1.00 / 0.000 | 0.00 / 0.000 | 1.00 / 0.000 |
| adversarial_noise_unseen_weak | 1.00 / 0.000 | 1.00 / 0.000 | 0.16 / 0.000 | 1.00 / 0.000 |
| adversarial_noise_unseen_strong | 1.00 / 0.000 | 1.00 / 0.000 | 0.53 / 0.000 | 1.00 / 0.000 |
| adversarial_free_riding | 1.00 / 0.000 | 0.00 / 0.333 | 0.00 / 0.000 | 1.00 / 0.000 |
| honest_idle_clients | N/A / 0.007 | N/A / 0.286 | N/A / 0.000 | N/A / 0.007 |
| honest_idle_majority | N/A / 0.013 | N/A / 0.250 | N/A / 0.000 | N/A / 0.019 |
| adversarial_free_rider_with_idle | 1.00 / 0.013 | 0.00 / 0.333 | 0.00 / 0.000 | 1.00 / 0.008 |

## Pooled over all adversarial scenarios (TPR) and all honest scenarios (FPR)

| Detector | TP | FN | TPR [95 % CI] | FP | TN | FPR [95 % CI] |
|---|---|---|---|---|---|---|
| deployed_trust | 727 | 113 | 0.865 [0.8407, 0.8869] | 48 | 5832 | 0.0082 [0.0062, 0.0108] |
| krum | 679 | 161 | 0.808 [0.7803, 0.8335] | 1021 | 4859 | 0.1736 [0.1642, 0.1835] |
| cosine_median | 222 | 618 | 0.264 [0.2356, 0.2951] | 201 | 5679 | 0.0342 [0.0298, 0.0391] |
| distance_mad | 726 | 114 | 0.864 [0.8395, 0.8858] | 49 | 5831 | 0.0083 [0.0063, 0.011] |
| no_detection | 0 | 840 | 0.000 [0.0, 0.0046] | 0 | 5880 | 0.0000 [0.0, 0.0007] |
