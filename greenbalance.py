"""Reproducible tabular Q-learning in a synthetic scheduling model.

The costs in this module are queue-work and assignment-energy proxies. They
are not measurements of job latency, electricity use, or production systems.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import mean, stdev
from typing import Sequence

import numpy as np


POLICIES = ("q_learning", "round_robin", "random", "shortest_completion")
MAX_TRAINING_STEPS = 5_000_000


def _integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _finite(value: float, name: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


@dataclass(frozen=True)
class EnvironmentConfig:
    speeds: tuple[float, ...] = (3.0, 2.0, 1.0)
    assignment_energy: tuple[float, ...] = (2.0, 1.5, 1.0)
    energy_weight: float = 2.0

    def __post_init__(self) -> None:
        if len(self.speeds) != 3 or len(self.assignment_energy) != 3:
            raise ValueError("the original model has exactly three servers")
        for speed in self.speeds:
            _finite(speed, "server speed", 1e-12, 1e12)
        for cost in self.assignment_energy:
            _finite(cost, "assignment-energy cost", 0.0, 1e12)
        _finite(self.energy_weight, "energy weight", 0.0, 1e12)


@dataclass(frozen=True)
class AgentConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    queue_bins: tuple[float, ...] = (10.0, 20.0, 40.0)

    def __post_init__(self) -> None:
        _finite(self.alpha, "alpha", 1e-12, 1.0)
        for name in ("gamma", "epsilon", "epsilon_min", "epsilon_decay"):
            _finite(getattr(self, name), name, 0.0, 1.0)
        if self.epsilon_min > self.epsilon:
            raise ValueError("epsilon_min cannot exceed initial epsilon")
        if not self.queue_bins:
            raise ValueError("queue_bins must not be empty")
        for boundary in self.queue_bins:
            _finite(boundary, "queue-bin boundary", 1e-12, 1e12)
        if any(a >= b for a, b in zip(self.queue_bins, self.queue_bins[1:])):
            raise ValueError("queue_bins must be strictly increasing")


@dataclass(frozen=True)
class ExperimentConfig:
    n_jobs: int = 200
    training_episodes: int = 100
    training_seed_start: int = 0
    evaluation_seeds: tuple[int, ...] = (1_000_000, 1_000_001, 1_000_002)
    agent_seed: int = 42
    baseline_seed: int = 43
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def __post_init__(self) -> None:
        _integer(self.n_jobs, "n_jobs", 1, 10_000)
        _integer(self.training_episodes, "training_episodes", 1, 10_000)
        if self.n_jobs * self.training_episodes > MAX_TRAINING_STEPS:
            raise ValueError(f"training is capped at {MAX_TRAINING_STEPS:,} transitions")
        for name in ("training_seed_start", "agent_seed", "baseline_seed"):
            _integer(getattr(self, name), name, 0, 2**63 - 1)
        if self.training_seed_start + self.training_episodes > 2**63:
            raise ValueError("training seed range is too large")
        if not 1 <= len(self.evaluation_seeds) <= 50:
            raise ValueError("provide between 1 and 50 evaluation seeds")
        for seed in self.evaluation_seeds:
            _integer(seed, "evaluation seed", 0, 2**63 - 1)
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation seeds must be unique")
        training_seeds = range(
            self.training_seed_start,
            self.training_seed_start + self.training_episodes,
        )
        if any(seed in training_seeds for seed in self.evaluation_seeds):
            raise ValueError("training and evaluation workload seeds must be disjoint")
        if not isinstance(self.environment, EnvironmentConfig):
            raise ValueError("environment must be an EnvironmentConfig")
        if not isinstance(self.agent, AgentConfig):
            raise ValueError("agent must be an AgentConfig")


def generate_workload(n_jobs: int, seed: int) -> tuple[int, ...]:
    """Create exactly n_jobs, with quiet/busy/mixed phases of about 30/40/30%."""
    _integer(n_jobs, "n_jobs", 1, 10_000)
    _integer(seed, "workload seed", 0, 2**63 - 1)
    rng = np.random.default_rng(seed)
    quiet = 3 * n_jobs // 10
    busy = 4 * n_jobs // 10
    sizes = np.concatenate(
        (
            rng.integers(1, 4, quiet),
            rng.integers(5, 11, busy),
            rng.integers(1, 9, n_jobs - quiet - busy),
        )
    )
    return tuple(int(size) for size in sizes)


State = tuple[float, float, float, int]


@dataclass(frozen=True)
class Transition:
    state: State
    reward: float
    done: bool
    queue_work: float
    assignment_energy: float


class CloudEnv:
    """One arrival and one unit of parallel server service per step.

    Each job joins one server. All servers then process up to their configured
    speed. The episode ends after the final arrival; remaining work is reported
    without draining the queues. This preserves the original notebook model.
    """

    def __init__(
        self, workload: Sequence[int], config: EnvironmentConfig | None = None
    ) -> None:
        self.config = config if config is not None else EnvironmentConfig()
        if not isinstance(self.config, EnvironmentConfig):
            raise ValueError("config must be an EnvironmentConfig")
        self.workload = tuple(workload)
        if not self.workload:
            raise ValueError("workload must not be empty")
        for size in self.workload:
            _integer(size, "job size", 1, 1_000_000)
        self.reset()

    def reset(self) -> State:
        self.current_step = 0
        self.queues = [0.0, 0.0, 0.0]
        return self.state

    @property
    def state(self) -> State:
        job = self.workload[self.current_step] if not self.done else 0
        return (self.queues[0], self.queues[1], self.queues[2], int(job))

    @property
    def done(self) -> bool:
        return self.current_step >= len(self.workload)

    def step(self, action: int) -> Transition:
        if self.done:
            raise RuntimeError("episode is finished; call reset before stepping")
        _integer(action, "action", 0, 2)
        self.queues[action] += self.workload[self.current_step]
        for index, speed in enumerate(self.config.speeds):
            self.queues[index] = max(0.0, self.queues[index] - speed)
        queue_work = sum(self.queues)
        energy = self.config.assignment_energy[action]
        reward = -(queue_work + self.config.energy_weight * energy)
        self.current_step += 1
        return Transition(self.state, reward, self.done, queue_work, energy)


class QLearningAgent:
    def __init__(self, config: AgentConfig | None = None, seed: int = 42) -> None:
        self.config = config if config is not None else AgentConfig()
        if not isinstance(self.config, AgentConfig):
            raise ValueError("config must be an AgentConfig")
        _integer(seed, "agent seed", 0, 2**63 - 1)
        self.rng = np.random.default_rng(seed)
        self.epsilon = self.config.epsilon
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    def state_key(self, state: State) -> tuple[int, ...]:
        return tuple(bisect_right(self.config.queue_bins, q) for q in state[:3]) + (
            int(state[3]),
        )

    def greedy_action(self, state: State) -> int:
        """Evaluation neither adds table entries nor consumes random numbers.

        Exact ties, including unseen states, choose the first server.
        """
        values = self.q_table.get(self.state_key(state))
        return 0 if values is None else int(np.argmax(values))

    def choose_action(self, state: State) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))
        values = self.q_table.get(self.state_key(state), np.zeros(3))
        ties = np.flatnonzero(values == np.max(values))
        return int(self.rng.choice(ties))

    def update(
        self, state: State, action: int, reward: float, next_state: State, done: bool
    ) -> None:
        _integer(action, "action", 0, 2)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        values = self.q_table.setdefault(self.state_key(state), np.zeros(3))
        # Terminal transitions have no future return, even if their key exists.
        future = 0.0
        if not done:
            next_values = self.q_table.get(self.state_key(next_state))
            if next_values is not None:
                future = float(np.max(next_values))
        target = reward + self.config.gamma * future
        values[action] += self.config.alpha * (target - values[action])

    def decay_exploration(self) -> None:
        self.epsilon = max(
            self.config.epsilon_min, self.epsilon * self.config.epsilon_decay
        )


@dataclass(frozen=True)
class Metrics:
    jobs: int
    queue_work_sum: float
    assignment_energy_proxy: float
    composite_cost: float
    final_queue_work: float


def train(config: ExperimentConfig) -> tuple[QLearningAgent, list[float]]:
    agent = QLearningAgent(config.agent, config.agent_seed)
    costs = []
    for seed in range(
        config.training_seed_start,
        config.training_seed_start + config.training_episodes,
    ):
        env = CloudEnv(generate_workload(config.n_jobs, seed), config.environment)
        state = env.state
        cost = 0.0
        while not env.done:
            action = agent.choose_action(state)
            transition = env.step(action)
            agent.update(state, action, transition.reward, transition.state, transition.done)
            state = transition.state
            cost -= transition.reward
        agent.decay_exploration()
        costs.append(cost)
    return agent, costs


def evaluate_policy(
    workload: Sequence[int],
    config: EnvironmentConfig,
    policy: str,
    agent: QLearningAgent | None = None,
    random_seed: int = 43,
    workload_seed: int = 0,
) -> Metrics:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if policy == "q_learning" and agent is None:
        raise ValueError("q_learning requires a trained agent")
    _integer(random_seed, "random baseline seed", 0, 2**63 - 1)
    _integer(workload_seed, "workload seed", 0, 2**63 - 1)
    # A seed-specific stream makes random actions independent of policy order.
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, workload_seed]))
    env = CloudEnv(workload, config)
    queue_total = 0.0
    energy_total = 0.0
    while not env.done:
        if policy == "q_learning":
            action = agent.greedy_action(env.state)  # type: ignore[union-attr]
        elif policy == "round_robin":
            action = env.current_step % 3
        elif policy == "random":
            action = int(rng.integers(3))
        else:
            # FIFO work ahead plus this job, divided by server service rate.
            action = min(
                range(3),
                key=lambda i: (env.queues[i] + env.state[3]) / config.speeds[i],
            )
        transition = env.step(action)
        queue_total += transition.queue_work
        energy_total += transition.assignment_energy
    return Metrics(
        jobs=len(env.workload),
        queue_work_sum=queue_total,
        assignment_energy_proxy=energy_total,
        composite_cost=queue_total + config.energy_weight * energy_total,
        final_queue_work=sum(env.queues),
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def workload_hash(workload: Sequence[int]) -> str:
    """SHA-256 of compact JSON integer job sizes; no native byte-order ambiguity."""
    return hashlib.sha256(_json_bytes([int(size) for size in workload])).hexdigest()


def source_metadata() -> dict:
    root = Path(__file__).resolve().parent
    paths = (
        "greenbalance.py",
        "render_results.py",
        "tests/test_greenbalance.py",
        "requirements.txt",
        ".github/workflows/ci.yml",
        "proj.ipynb",
        "greenbalance.pdf",
    )
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "random_generator": "numpy.random.Generator(PCG64)",
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths
            if (root / name).is_file()
        },
    }


def run_experiment(config: ExperimentConfig) -> dict:
    agent, training_costs = train(config)
    rows = []
    for seed in config.evaluation_seeds:
        # Every policy receives the same immutable arrivals, generated once.
        workload = generate_workload(config.n_jobs, seed)
        digest = workload_hash(workload)
        for policy in POLICIES:
            metrics = evaluate_policy(
                workload, config.environment, policy, agent,
                config.baseline_seed, seed,
            )
            rows.append({
                "workload_seed": seed,
                "workload_sha256": digest,
                "policy": policy,
                **asdict(metrics),
            })
    configuration = asdict(config)
    return {
        "schema_version": 1,
        "description": "Synthetic queue-work and assignment-energy proxy experiment",
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(_json_bytes(configuration)).hexdigest(),
        "environment": source_metadata(),
        "training": {
            "episode_composite_costs": training_costs,
            "q_table_states": len(agent.q_table),
            "final_epsilon": agent.epsilon,
        },
        "evaluation": rows,
        "summary": summarize(rows),
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        result[policy] = {}
        for metric in (
            "queue_work_sum", "assignment_energy_proxy", "composite_cost", "final_queue_work"
        ):
            values = [row[metric] for row in selected]
            result[policy][metric] = {
                "mean": mean(values),
                "sample_std": stdev(values) if len(values) > 1 else None,
            }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--training-seed-start", type=int, default=0)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[1_000_000, 1_000_001, 1_000_002])
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--baseline-seed", type=int, default=43)
    parser.add_argument("--energy-weight", type=float, default=2.0)
    parser.add_argument("--output", type=Path, help="write a new JSON result file (never overwrite)")
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig(
            n_jobs=args.jobs,
            training_episodes=args.episodes,
            training_seed_start=args.training_seed_start,
            evaluation_seeds=tuple(args.evaluation_seeds),
            agent_seed=args.agent_seed,
            baseline_seed=args.baseline_seed,
            environment=EnvironmentConfig(energy_weight=args.energy_weight),
        )
        if args.output is not None and args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        result = run_experiment(config)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print("Synthetic scheduling costs; lower is better. No physical energy or latency units.")
    print(f"{config.training_episodes} training traces; {len(config.evaluation_seeds)} held-out traces; {config.n_jobs} jobs each")
    print(f"{'Policy':22} {'Mean queue work':>17} {'Mean energy proxy':>19} {'Mean total cost':>17}")
    for policy, metrics in result["summary"].items():
        print(
            f"{policy:22} {metrics['queue_work_sum']['mean']:17.2f} "
            f"{metrics['assignment_energy_proxy']['mean']:19.2f} "
            f"{metrics['composite_cost']['mean']:17.2f}"
        )
    if args.output is not None:
        print(f"Recorded configuration, source hashes, and per-seed metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
