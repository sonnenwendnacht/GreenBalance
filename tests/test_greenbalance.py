"""Model invariants and reproducibility checks, using the standard library runner."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import greenbalance as gb


class EnvironmentTests(unittest.TestCase):
    def test_exact_parallel_queue_dynamics_and_reward_accounting(self):
        env = gb.CloudEnv([5, 2])
        self.assertEqual(env.state, (0.0, 0.0, 0.0, 5))
        first = env.step(0)
        self.assertEqual(first, gb.Transition((2.0, 0.0, 0.0, 2), -6.0, False, 2.0, 2.0))
        last = env.step(2)
        self.assertEqual(last, gb.Transition((0.0, 0.0, 1.0, 0), -3.0, True, 1.0, 1.0))
        self.assertEqual(first.reward + last.reward, -(2.0 + 1.0 + 2 * (2.0 + 1.0)))
        with self.assertRaises(RuntimeError):
            env.step(0)
        self.assertEqual(env.reset(), (0.0, 0.0, 0.0, 5))

    def test_work_conservation_including_idle_servers(self):
        env = gb.CloudEnv([3, 8, 1, 5])
        for action in (2, 1, 0, 2):
            before = list(env.queues)
            before[action] += env.state[3]
            served = sum(min(q, speed) for q, speed in zip(before, env.config.speeds))
            transition = env.step(action)
            self.assertAlmostEqual(transition.queue_work, sum(before) - served)
            self.assertTrue(all(q >= 0 for q in env.queues))

    def test_invalid_workloads_and_actions(self):
        for workload in ([], [0], [-1], [1.5], [True]):
            with self.subTest(workload=workload), self.assertRaises(ValueError):
                gb.CloudEnv(workload)
        env = gb.CloudEnv([1])
        for action in (-1, 3, True, 0.5):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.current_step, 0)

    def test_metrics_keep_queue_energy_and_total_separate(self):
        metrics = gb.evaluate_policy([5, 2], gb.EnvironmentConfig(energy_weight=3.0), "round_robin")
        self.assertEqual(metrics, gb.Metrics(2, 2.0, 3.5, 12.5, 0.0))

    def test_shortest_completion_uses_work_and_server_speed(self):
        # First choose the fastest server, then the medium server because its
        # predicted completion time is 10/2, below (7+10)/3 on the fastest.
        metrics = gb.evaluate_policy([10, 10], gb.EnvironmentConfig(), "shortest_completion")
        self.assertEqual(metrics, gb.Metrics(2, 19.0, 3.5, 26.0, 12.0))


class LearningTests(unittest.TestCase):
    def test_terminal_update_never_bootstraps(self):
        agent = gb.QLearningAgent(gb.AgentConfig(alpha=0.5, gamma=0.9))
        state = (0.0, 0.0, 0.0, 5)
        terminal = (10.0, 0.0, 0.0, 0)
        agent.q_table[agent.state_key(state)] = np.array([4.0, 0.0, 0.0])
        agent.q_table[agent.state_key(terminal)] = np.array([1000.0, 2000.0, 3000.0])
        agent.update(state, 0, -2.0, terminal, True)
        self.assertEqual(agent.q_table[agent.state_key(state)][0], 1.0)

    def test_nonterminal_update_uses_discounted_future(self):
        agent = gb.QLearningAgent(gb.AgentConfig(alpha=0.5, gamma=0.9))
        state = (0.0, 0.0, 0.0, 5)
        next_state = (1.0, 0.0, 0.0, 2)
        agent.q_table[agent.state_key(next_state)] = np.array([1.0, 3.0, 2.0])
        agent.update(state, 2, -2.0, next_state, False)
        self.assertAlmostEqual(agent.q_table[agent.state_key(state)][2], 0.35)

    def test_discretization_matches_original_digitize_edges(self):
        agent = gb.QLearningAgent()
        self.assertEqual(agent.state_key((9.0, 10.0, 40.0, 5)), (0, 1, 3, 5))

    def test_evaluation_is_read_only_including_unseen_states_and_rng(self):
        agent, _ = gb.train(gb.ExperimentConfig(n_jobs=12, training_episodes=3))
        before = {key: value.copy() for key, value in agent.q_table.items()}
        rng_before = json.dumps(agent.rng.bit_generator.state, sort_keys=True)
        epsilon_before = agent.epsilon
        gb.evaluate_policy([100, 200, 300], gb.EnvironmentConfig(), "q_learning", agent)
        self.assertEqual(set(before), set(agent.q_table))
        for key in before:
            np.testing.assert_array_equal(before[key], agent.q_table[key])
        self.assertEqual(json.dumps(agent.rng.bit_generator.state, sort_keys=True), rng_before)
        self.assertEqual(agent.epsilon, epsilon_before)

    def test_agent_random_streams_are_independent(self):
        first = gb.QLearningAgent(seed=123)
        second = gb.QLearningAgent(seed=123)
        unrelated = gb.QLearningAgent(seed=456)
        state = (0.0, 0.0, 0.0, 5)
        expected = [first.choose_action(state) for _ in range(20)]
        for _ in range(57):
            unrelated.choose_action(state)
        self.assertEqual(expected, [second.choose_action(state) for _ in range(20)])


class ReproducibilityTests(unittest.TestCase):
    def test_workload_length_phases_and_seed(self):
        for jobs in (1, 7, 1000):
            trace = gb.generate_workload(jobs, 15)
            self.assertEqual(len(trace), jobs)
            self.assertEqual(trace, gb.generate_workload(jobs, 15))
            quiet = 3 * jobs // 10
            busy = 4 * jobs // 10
            self.assertTrue(all(1 <= job <= 3 for job in trace[:quiet]))
            self.assertTrue(all(5 <= job <= 10 for job in trace[quiet:quiet + busy]))
            self.assertTrue(all(1 <= job <= 8 for job in trace[quiet + busy:]))

    def test_global_numpy_random_state_does_not_affect_experiment(self):
        config = gb.ExperimentConfig(n_jobs=20, training_episodes=4)
        np.random.seed(9)
        expected_random = np.random.random()
        np.random.seed(9)
        first = gb.run_experiment(config)
        self.assertEqual(np.random.random(), expected_random)
        np.random.seed(981)
        np.random.random(100)
        second = gb.run_experiment(config)
        self.assertEqual(first, second)

    def test_same_held_out_arrivals_reach_every_policy(self):
        config = gb.ExperimentConfig(n_jobs=20, training_episodes=3, evaluation_seeds=(500, 501))
        with patch("greenbalance.evaluate_policy", wraps=gb.evaluate_policy) as evaluator:
            result = gb.run_experiment(config)
        self.assertEqual(evaluator.call_count, 8)
        for offset, seed in ((0, 500), (4, 501)):
            workload = evaluator.call_args_list[offset].args[0]
            self.assertEqual(workload, gb.generate_workload(20, seed))
            for call in evaluator.call_args_list[offset:offset + 4]:
                self.assertIs(call.args[0], workload)
            digests = {row["workload_sha256"] for row in result["evaluation"] if row["workload_seed"] == seed}
            self.assertEqual(digests, {gb.workload_hash(workload)})

    def test_evaluation_seed_order_does_not_change_any_policy_result(self):
        config = gb.ExperimentConfig(n_jobs=20, training_episodes=3, evaluation_seeds=(500, 501))
        forward = gb.run_experiment(config)
        reverse = gb.run_experiment(replace(config, evaluation_seeds=(501, 500)))
        by_key = lambda rows: {(row["workload_seed"], row["policy"]): row for row in rows}
        self.assertEqual(by_key(forward["evaluation"]), by_key(reverse["evaluation"]))

    def test_metadata_identifies_configuration_and_source(self):
        config = gb.ExperimentConfig(n_jobs=5, training_episodes=1)
        result = gb.run_experiment(config)
        canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(result["configuration_sha256"], hashlib.sha256(canonical).hexdigest())
        source = Path(gb.__file__).read_bytes()
        self.assertEqual(result["environment"]["source_sha256"]["greenbalance.py"], hashlib.sha256(source).hexdigest())
        self.assertEqual(result["environment"]["numpy_version"], np.__version__)

    def test_invalid_configuration(self):
        invalid_experiments = (
            {"n_jobs": 0}, {"n_jobs": True}, {"training_episodes": 0},
            {"n_jobs": 10_000, "training_episodes": 10_000},
            {"evaluation_seeds": ()}, {"evaluation_seeds": (500, 500)},
            {"evaluation_seeds": (0,)}, {"evaluation_seeds": (-1,)},
            {"agent_seed": -1}, {"baseline_seed": True},
            {"training_seed_start": 2**63 - 1, "training_episodes": 2},
        )
        for kwargs in invalid_experiments:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gb.ExperimentConfig(**kwargs)
        for kwargs in ({"speeds": (0, 2, 1)}, {"speeds": (3, 2)}, {"energy_weight": float("nan")}, {"assignment_energy": (-1, 1, 1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gb.EnvironmentConfig(**kwargs)
        for kwargs in ({"alpha": 0}, {"gamma": 2}, {"epsilon": 0, "epsilon_min": 0.1}, {"queue_bins": (10, 10)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gb.AgentConfig(**kwargs)


class CliTests(unittest.TestCase):
    def test_cli_runs_records_valid_json_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            command = [sys.executable, gb.__file__, "--jobs", "12", "--episodes", "2", "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("No physical energy or latency units", completed.stdout)
            result = json.loads(output.read_text())
            self.assertEqual(len(result["evaluation"]), 12)
            self.assertTrue(all(row["jobs"] == 12 for row in result["evaluation"]))
            before = output.read_bytes()
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("already exists", repeated.stderr)
            self.assertEqual(output.read_bytes(), before)

    def test_cli_rejects_overlapping_workload_seeds(self):
        command = [sys.executable, gb.__file__, "--evaluation-seeds", "1"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("must be disjoint", completed.stderr)


if __name__ == "__main__":
    unittest.main()
