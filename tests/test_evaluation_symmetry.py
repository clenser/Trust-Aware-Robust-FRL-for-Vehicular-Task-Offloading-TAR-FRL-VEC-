"""
Evaluation-symmetry tests.

The task-completion metric must be defined in the *same* way for every
execution path.  Before the fix, the local path returned "accepted" as its
success flag without ever comparing the sojourn time with the task deadline,
while every offloading path compared the delay with the deadline.  A purely
local baseline therefore could not miss a deadline by construction, which made
the completion comparison between the local and the offloading paths
asymmetric.

These tests pin the corrected behaviour:

* a local task that finishes after its deadline counts as a failure;
* a local task that finishes within its deadline counts as a success;
* the local and the offloaded path use the same criterion.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from environment.vanet_env import VANETOffloadingEnv


def _make_env():
    env = VANETOffloadingEnv()
    return env


class TestLocalExecutionRespectsTheDeadline(unittest.TestCase):

    def test_local_task_missing_its_deadline_is_a_failure(self):
        env = _make_env()
        vid = sorted(env.vehicles.keys())[0]
        v = env.vehicles[vid]
        task = env.task_generator.generate_task(vid, env.current_time_sec)
        # Make the task impossible to meet locally: a huge CPU demand with a
        # short deadline.
        task.cpu_cycles_giga = 400.0
        task.deadline_ms = 1.0
        task.deadline_sec = 0.001
        _next_state, _r, _done, info, _t = env.step_vehicle(vid, 0, task)
        self.assertTrue(info["present"])
        self.assertFalse(info["success"],
                         "a local task that misses its deadline must fail")
        self.assertGreater(info["delay_ms"], task.deadline_ms)

    def test_local_task_within_its_deadline_succeeds(self):
        env = _make_env()
        vid = sorted(env.vehicles.keys())[0]
        v = env.vehicles[vid]
        task = env.task_generator.generate_task(vid, env.current_time_sec)
        task.cpu_cycles_giga = 0.001
        task.deadline_ms = 10000.0
        task.deadline_sec = 10.0
        _next_state, _r, _done, info, _t = env.step_vehicle(vid, 0, task)
        self.assertTrue(info["present"])
        self.assertTrue(info["success"],
                        "a local task that meets its deadline must succeed")

    def test_the_criterion_is_the_same_as_the_offloaded_path(self):
        """The success flag is ``delay <= deadline`` in both branches."""
        import inspect
        from environment import vanet_env
        source = inspect.getsource(vanet_env.VANETOffloadingEnv.step_vehicle)
        self.assertIn("delay_sec <= task.deadline_sec", source,
                      "the local branch must compare the delay with the deadline")
        # The offloading branches use exactly the same expression.
        self.assertGreaterEqual(source.count("delay_sec <= task.deadline_sec"), 2,
                                "the offloading branches must use the same criterion")

    def test_execute_local_still_reports_the_sojourn_time(self):
        """``execute_local`` returns the sojourn time; the caller compares it
        with the deadline, as its own docstring states."""
        from environment.vehicle import Vehicle
        doc = Vehicle.execute_local.__doc__
        self.assertIn("deadline", doc)


class TestCompletionMetricIsSymmetricAcrossPaths(unittest.TestCase):

    def test_a_purely_local_policy_can_miss_deadlines(self):
        """Regression: a purely local baseline reported 100 % completion
        regardless of the latency it produced."""
        from experiments.experiment_runner import (
            run_multiseed_experiment, EvaluationConfig)
        from configs.config_loader import load_config
        cfg = load_config()
        eval_config = EvaluationConfig(episodes=1, steps_per_episode=3,
                                       seed_offset=500_000)
        res = run_multiseed_experiment([42], config=cfg, eval_config=eval_config,
                                       baseline="Local_Only", attack_ratio=0.24,
                                       mobility_mode="highway")
        rec = res["per_seed_results"][0]
        self.assertIsNotNone(rec["completion_rate"])
        self.assertLess(rec["completion_rate"], 100.0,
                        "a purely local policy must be able to miss deadlines")


if __name__ == "__main__":
    unittest.main()
