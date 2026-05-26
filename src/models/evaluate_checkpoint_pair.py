from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.agent import Predator, Prey
from config.config import SEED
from environment.multi_agent_gym_env import MultiAgentEcoSimEnv
from models.Q_learning import QLearningAgent
from models.coexistence_metrics import coexistence_score


@dataclass
class EventTracker:
    prey_births: int = 0
    predator_births: int = 0
    predation_events: int = 0
    cumulative_prey_deaths: int = 0
    cumulative_predator_deaths: int = 0


@contextlib.contextmanager
def track_ecosim_events(tracker: EventTracker):
    original_prey_reproduce = Prey.reproduce
    original_predator_reproduce = Predator.reproduce
    original_predator_eat = Predator.eat

    def wrapped_prey_reproduce(self, *args, **kwargs):
        offspring = original_prey_reproduce(self, *args, **kwargs)
        if offspring is not None:
            tracker.prey_births += 1
        return offspring

    def wrapped_predator_reproduce(self, *args, **kwargs):
        offspring = original_predator_reproduce(self, *args, **kwargs)
        if offspring is not None:
            tracker.predator_births += 1
        return offspring

    def wrapped_predator_eat(self, *args, **kwargs):
        reward = original_predator_eat(self, *args, **kwargs)
        if reward > 0:
            tracker.predation_events += 1
        return reward

    Prey.reproduce = wrapped_prey_reproduce
    Predator.reproduce = wrapped_predator_reproduce
    Predator.eat = wrapped_predator_eat
    try:
        yield
    finally:
        Prey.reproduce = original_prey_reproduce
        Predator.reproduce = original_predator_reproduce
        Predator.eat = original_predator_eat


def clone_frozen_agent(model_path: Path, agent_id: int = 0) -> QLearningAgent:
    agent = QLearningAgent.load_model_from_file(str(model_path))
    agent.agent_id = agent_id
    agent.epsilon = 0.0
    return agent


def mean_or_zero(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def build_step_snapshot(env, step_index: int, reward: float, cumulative_reward: float, tracker: EventTracker, previous_alive_ids: set[int], known_agent_types: Dict[int, str]):
    all_agents = list(env.all_agents)
    alive_agents = [agent for agent in all_agents if agent.is_alive()]
    prey_agents = [agent for agent in alive_agents if agent.agent_type == "PREY"]
    predator_agents = [agent for agent in alive_agents if agent.agent_type == "PREDATOR"]

    current_alive_ids = {agent.agent_id for agent in alive_agents}
    for agent in all_agents:
        known_agent_types.setdefault(agent.agent_id, agent.agent_type)

    deaths = previous_alive_ids - current_alive_ids
    prey_deaths = sum(1 for agent_id in deaths if known_agent_types.get(agent_id) == "PREY")
    predator_deaths = sum(1 for agent_id in deaths if known_agent_types.get(agent_id) == "PREDATOR")

    # Accumulate into tracker (resets per run, not per module load)
    tracker.cumulative_prey_deaths += prey_deaths
    tracker.cumulative_predator_deaths += predator_deaths

    reference_agent = alive_agents[0] if alive_agents else None
    reference_observation = reference_agent.get_observation(env.env) if reference_agent is not None else np.zeros(6, dtype=np.float32)
    system_coexistence = coexistence_score(len(prey_agents), len(predator_agents))
    system_extinct = int(len(prey_agents) == 0 or len(predator_agents) == 0)

    snapshot = {
        "step": step_index,
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "reference_agent_type": reference_agent.agent_type if reference_agent is not None else "NONE",
        "reference_agent_energy": float(reference_agent.energy) if reference_agent is not None else 0.0,
        "reference_agent_sensor_flag": int(reference_observation[6]),
        "reference_agent_target_detected": int(reference_observation[4]),
        "prey_population": len(prey_agents),
        "predator_population": len(predator_agents),
        "total_alive": len(alive_agents),
        "system_coexistence_score": system_coexistence,
        "system_extinct": system_extinct,
        "avg_prey_energy": mean_or_zero(agent.energy for agent in prey_agents),
        "avg_predator_energy": mean_or_zero(agent.energy for agent in predator_agents),
        "min_prey_energy": float(min((agent.energy for agent in prey_agents), default=0.0)),
        "max_prey_energy": float(max((agent.energy for agent in prey_agents), default=0.0)),
        "min_predator_energy": float(min((agent.energy for agent in predator_agents), default=0.0)),
        "max_predator_energy": float(max((agent.energy for agent in predator_agents), default=0.0)),
        "cumulative_prey_births": tracker.prey_births,
        "cumulative_predator_births": tracker.predator_births,
        "cumulative_predation_events": tracker.predation_events,
        "cumulative_prey_deaths": tracker.cumulative_prey_deaths,
        "cumulative_predator_deaths": tracker.cumulative_predator_deaths,
    }
    return snapshot, current_alive_ids


def run_long_episode_experiment(
    prey_model_path: Path,
    predator_model_path: Path,
    run_name: str,
    seed_offset: int,
    long_episode_steps: int,
    eval_seed: int,
    num_prey: int,
    num_predators: int,
    map_path,
    memory: bool,
    results_dir: Path,
):
    prey_policy = clone_frozen_agent(prey_model_path, agent_id=0)
    predator_policy = clone_frozen_agent(predator_model_path, agent_id=1)

    tracker = EventTracker()
    records: List[Dict] = []

    with track_ecosim_events(tracker):
        env = MultiAgentEcoSimEnv(
            num_prey=num_prey,
            num_predators=num_predators,
            map_path=map_path,
        )

        env.reset(seed=eval_seed + seed_offset)
        previous_alive_ids = {agent.agent_id for agent in env.all_agents if agent.is_alive()}
        known_agent_types = {agent.agent_id: agent.agent_type for agent in env.all_agents}
        cumulative_reward = 0.0

        for step_index in range(1, long_episode_steps + 1):
            actions: Dict[int, int] = {}
            for agent in env.all_agents:
                if not agent.is_alive():
                    continue
                policy = prey_policy if agent.agent_type == "PREY" else predator_policy
                observation = agent.get_observation(env.env)
                state = policy.discretize_state(observation)
                actions[agent.agent_id] = policy.select_action(state, training=False)

            _, reward, done, info = env.step(actions)
            cumulative_reward += reward

            snapshot, previous_alive_ids = build_step_snapshot(
                env,
                step_index,
                reward,
                cumulative_reward,
                tracker,
                previous_alive_ids,
                known_agent_types,
            )
            records.append(snapshot)

            if done:
                break

    results_df = pd.DataFrame(records)
    results_df["prey_model_path"] = str(prey_model_path)
    results_df["predator_model_path"] = str(predator_model_path)
    results_df["run_name"] = run_name
    results_df["seed"] = eval_seed + seed_offset
    results_df["long_episode_steps"] = long_episode_steps

    csv_path = results_dir / f"{run_name}__{prey_model_path.stem}__{predator_model_path.stem}_{len(results_df)}steps.csv"
    results_df.to_csv(csv_path, index=False)

    summary = {
        "run_name": run_name,
        "prey_model_path": str(prey_model_path),
        "predator_model_path": str(predator_model_path),
        "seed": eval_seed + seed_offset,
        "steps_ran": int(len(results_df)),
        "final_reward": float(results_df["cumulative_reward"].iloc[-1]) if not results_df.empty else 0.0,
        "final_prey_population": int(results_df["prey_population"].iloc[-1]) if not results_df.empty else 0,
        "final_predator_population": int(results_df["predator_population"].iloc[-1]) if not results_df.empty else 0,
        "final_system_coexistence_score": float(results_df["system_coexistence_score"].iloc[-1]) if not results_df.empty else 0.0,
        "mean_system_coexistence_score": float(results_df["system_coexistence_score"].mean()) if not results_df.empty else 0.0,
        "min_system_coexistence_score": float(results_df["system_coexistence_score"].min()) if not results_df.empty else 0.0,
        "system_survival_fraction": float((results_df["system_extinct"] == 0).mean()) if not results_df.empty else 0.0,
        "system_extinction_step": int(results_df.loc[results_df["system_extinct"] == 1, "step"].iloc[0]) if not results_df.empty and (results_df["system_extinct"] == 1).any() else 0,
        "cumulative_predation_events": int(results_df["cumulative_predation_events"].iloc[-1]) if not results_df.empty else 0,
        "cumulative_prey_births": int(results_df["cumulative_prey_births"].iloc[-1]) if not results_df.empty else 0,
        "cumulative_predator_births": int(results_df["cumulative_predator_births"].iloc[-1]) if not results_df.empty else 0,
        "csv_path": str(csv_path),
    }
    return results_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic long-horizon evaluation for a checkpoint pair.")
    parser.add_argument("--prey-model", required=True, type=Path)
    parser.add_argument("--predator-model", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "src" / "models" / "results" / "long_horizon_evaluation")
    parser.add_argument("--cycle-ids", nargs="+", type=int, default=[1])
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--main-agent-types", nargs="+", default=["SYSTEM"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--num-prey", type=int, default=30)
    parser.add_argument("--num-predators", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import config.config as config_module
    import environment.gym_env as gym_env_module
    import environment.multi_agent_gym_env as multi_env_module
    import models.Q_learning as q_learning_module

    config_module.STEPS_PER_EPISODE = args.steps
    gym_env_module.STEPS_PER_EPISODE = args.steps
    multi_env_module.STEPS_PER_EPISODE = args.steps
    q_learning_module.STEPS_PER_EPISODE = args.steps

    results_dir = args.output_dir / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for cycle_id in args.cycle_ids:
        for seed_offset in args.seed_offsets:
            for eval_label in args.main_agent_types:
                run_label = "system" if eval_label.upper() == "SYSTEM" else eval_label.lower()
                run_name = f"cycle{cycle_id}_{run_label}_seed{seed_offset}"
                print("Running", run_name)
                _, summary = run_long_episode_experiment(
                    args.prey_model,
                    args.predator_model,
                    run_name,
                    seed_offset,
                    args.steps,
                    args.seed,
                    args.num_prey,
                    args.num_predators,
                    None,
                    False,
                    results_dir,
                )
                all_summaries.append(summary)
                print(
                    f"Saved {summary['csv_path']} | steps={summary['steps_ran']} | "
                    f"mean system score={summary['mean_system_coexistence_score']:.4f} | "
                    f"final prey={summary['final_prey_population']} | final predator={summary['final_predator_population']}"
                )

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = results_dir / f"{args.run_name}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("All done. Summary saved to", summary_csv)