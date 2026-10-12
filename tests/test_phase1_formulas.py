"""Phase 1 tests: executable checks for the spec helpers in specs/formulas.py.

These lock the equations of docs/PHASE1_SPEC.md (bounds, signs, spot values,
fallback safety). They are NOT simulator/RL/federated/attack implementations.
"""

import math

import pytest

from specs.formulas import (
    behavioural_trust,
    clamp01,
    combine_trust,
    compute_reward,
    copy_zero_flag,
    directional_agreement,
    exec_latency,
    fedavg,
    idle_ageing,
    minmax_norm,
    model_update_trust,
    norm_plausibility,
    shannon_rate,
    staleness_trust,
    task_energy,
    task_outcome,
    temporal_update,
    total_latency,
    transmission_latency,
    trust_weighted_aggregate,
)


def test_trust_always_bounded():
    assert combine_trust(2.0, -1.0, 0.6) == pytest.approx(0.6 * 1.0 + 0.4 * 0.0)
    assert clamp01(99.0) == 1.0 and clamp01(-3.0) == 0.0
    assert temporal_update(1.0, 1.0) == pytest.approx(1.0)
    assert temporal_update(0.0, 0.0) == pytest.approx(0.0)


def test_temporal_update_spot_value():
    # T = 0.9 * 0.5 + 0.1 * 1.0 = 0.55
    assert temporal_update(0.5, 1.0, lam=0.9) == pytest.approx(0.55)


def test_disconnect_is_not_malice():
    t = 0.8
    aged = idle_ageing(t)
    assert 0.5 < aged < t  # drifts toward prior 0.5, never punished to 0
    assert idle_ageing(0.0) >= 0.0


def test_reward_signs():
    good = compute_reward(0.1, 0.1, True, False, False, 0.9)
    bad = compute_reward(0.9, 0.9, False, True, True, 0.1)
    assert good > 0 > bad
    # latency/energy penalised, success/trust rewarded
    assert compute_reward(0.5, 0.1, True, False, False, 0.5) < compute_reward(
        0.1, 0.1, True, False, False, 0.5
    )


def test_latency_energy_spot_values():
    assert shannon_rate(1e6, 3.0) == pytest.approx(2e6)
    assert transmission_latency(2e6, 1e6) == pytest.approx(2.0)
    assert exec_latency(3e9, 1e9, 0.5) == pytest.approx(3.5)
    assert total_latency(2.0, 3.5) == pytest.approx(5.5)
    assert task_energy(0.5, 2.0, 1e-27, 1e9, 1e9) == pytest.approx(2.0)  # 0.5*2 + 1e-27*1e9*1e18


def test_task_outcome_separates_miss_from_fail():
    assert task_outcome(0.5, 1.0, True)["success"] is True
    assert task_outcome(2.0, 1.0, True)["deadline_missed"] is True
    assert task_outcome(0.5, 1.0, False)["failed"] is True


def test_behavioural_trust_weights_and_bounds():
    s = {"success": 1.0, "timeliness": 0.5, "resources": 0.5, "reliability": 0.5}
    assert behavioural_trust(s) == pytest.approx(0.4 + 0.25 * 0.5 + 0.2 * 0.5 + 0.15 * 0.5)
    with pytest.raises(ValueError):
        behavioural_trust({"success": 1.0})


def test_model_update_trust_flags():
    assert copy_zero_flag(True, False) == 0.0
    assert copy_zero_flag(False, True) == 0.0
    assert copy_zero_flag(False, False) == 1.0
    assert norm_plausibility(10.0, 1.0, 3.0) == pytest.approx(0.3)
    assert norm_plausibility(2.0, 1.0, 3.0) == 1.0
    assert directional_agreement(1.0) == 1.0
    assert directional_agreement(-1.0) == 0.0
    assert staleness_trust(0) == 1.0 and staleness_trust(3) == pytest.approx(0.25)
    m = model_update_trust({"norm": 1, "direction": 1, "staleness": 1, "originality": 1, "progress": 1})
    assert m == pytest.approx(1.0)


def test_fedavg_spot_value():
    assert fedavg([[1.0, 2.0], [3.0, 4.0]], [1.0, 3.0]) == pytest.approx([2.5, 3.5])
    with pytest.raises(ValueError):
        fedavg([[1.0]], [0.0])


def test_trust_aggregation_threshold_and_fallback():
    thetas = [[1.0, 1.0], [100.0, 100.0], [2.0, 2.0]]
    agg, eligible = trust_weighted_aggregate(thetas, [0.9, 0.1, 0.8], threshold=0.3)
    assert eligible == [0, 2]
    assert agg == pytest.approx([(0.9 * 1 + 0.8 * 2) / 1.7] * 2)
    # nobody eligible -> fallback returned unchanged, no crash
    agg2, eligible2 = trust_weighted_aggregate(thetas, [0.1, 0.0, 0.2], threshold=0.3, fallback=[7.0, 7.0])
    assert eligible2 == [] and agg2 == [7.0, 7.0]
    with pytest.raises(ValueError):
        trust_weighted_aggregate(thetas, [0.0, 0.0, 0.0], threshold=0.3, fallback=None)


def test_minmax_norm_guards():
    assert minmax_norm(5.0, 0.0, 10.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        minmax_norm(1.0, 2.0, 2.0)
