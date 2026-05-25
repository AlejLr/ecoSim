from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import contextlib
import copy
import random
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.config import SEED
from agents.agent import Prey, Predator
from environment.gym_env import EcoSimEnv
from models.Q_learning import QLearningAgent

# Evaluation settings
cycle_ids = [1, 2]
seed_offsets = [0, 1, 2]
long_episode_steps = 2000

# Override global caps
import config.config as config_module
import environment.gym_env as gym_env_module
import environment.multi_agent_gym_env as multi_env_module
import models.Q_learning as q_learning_module
config_module.STEPS_PER_EPISODE = long_episode_steps
gym_env_module.STEPS_PER_EPISODE = long_episode_steps
multi_env_module.STEPS_PER_EPISODE = long_episode_steps
q_learning_module.STEPS_PER_EPISODE = long_episode_steps

# Deterministic eval
eval_seed = SEED
frozen_policy_epsilon = 0.0

# Environment params
num_prey = 30
num_predators = 10
map_path = None
memory = False

results_dir = ROOT / 'src' / 'models' / 'results' / 'long_horizon_evaluation' / 'run40_all_cycles_2000'
results_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class EventTracker:
    prey_births: int = 0
    predator_births: int = 0
    predation_events: int = 0


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
    all_agents = [env.agent] + list(env.other_agents)
    alive_agents = [agent for agent in all_agents if agent.is_alive()]
    prey_agents = [agent for agent in alive_agents if agent.agent_type == 'PREY']
    predator_agents = [agent for agent in alive_agents if agent.agent_type == 'PREDATOR']

    current_alive_ids = {agent.agent_id for agent in alive_agents}
    for agent in all_agents:
        known_agent_types.setdefault(agent.agent_id, agent.agent_type)

    deaths = previous_alive_ids - current_alive_ids
    prey_deaths = sum(1 for agent_id in deaths if known_agent_types.get(agent_id) == 'PREY')
    predator_deaths = sum(1 for agent_id in deaths if known_agent_types.get(agent_id) == 'PREDATOR')

    main_agent = env.agent
    main_observation = main_agent.get_observation(env.env)

    snapshot = {
        'step': step_index,
        'reward': reward,
        'cumulative_reward': cumulative_reward,
        'main_agent_type': main_agent.agent_type,
        'main_agent_energy': float(main_agent.energy),
        'main_agent_sensor_flag': int(main_observation[6]),
        'main_agent_target_detected': int(main_observation[4]),
        'prey_population': len(prey_agents),
        'predator_population': len(predator_agents),
        'total_alive': len(alive_agents),
        'avg_prey_energy': mean_or_zero(agent.energy for agent in prey_agents),
        'avg_predator_energy': mean_or_zero(agent.energy for agent in predator_agents),
        'min_prey_energy': float(min((agent.energy for agent in prey_agents), default=0.0)),
        'max_prey_energy': float(max((agent.energy for agent in prey_agents), default=0.0)),
        'min_predator_energy': float(min((agent.energy for agent in predator_agents), default=0.0)),
        'max_predator_energy': float(max((agent.energy for agent in predator_agents), default=0.0)),
        'cumulative_prey_births': tracker.prey_births,
        'cumulative_predator_births': tracker.predator_births,
        'cumulative_predation_events': tracker.predation_events,
        'cumulative_prey_deaths': build_step_snapshot.cumulative_prey_deaths + prey_deaths,
        'cumulative_predator_deaths': build_step_snapshot.cumulative_predator_deaths + predator_deaths,
    }
    return snapshot, current_alive_ids

build_step_snapshot.cumulative_prey_deaths = 0
build_step_snapshot.cumulative_predator_deaths = 0


def resolve_model_path(model_name: str, species: str) -> Path:
    model_name = str(model_name).strip()
    models_dir = ROOT / 'src' / 'models'

    direct_path = Path(model_name)
    if direct_path.exists():
        return direct_path

    if not model_name.endswith('.pkl'):
        model_name = f'{model_name}.pkl'

    exact_path = models_dir / model_name
    if exact_path.exists():
        return exact_path

    candidates = sorted(models_dir.glob(f'trained_{species.lower()}_*.pkl'), key=lambda p: p.stat().st_mtime)
    partial = [path for path in candidates if Path(model_name).stem.lower() in path.stem.lower()]
    if partial:
        return partial[-1]

    raise FileNotFoundError(f'Could not resolve {species.lower()} model from {model_name!r}')


def run_long_episode_experiment(prey_model_path, predator_model_path, main_agent_type, run_name, seed_offset):
    prey_policy = clone_frozen_agent(prey_model_path, agent_id=0)
    predator_policy = clone_frozen_agent(predator_model_path, agent_id=1)
    prey_policy.epsilon = 0.0
    predator_policy.epsilon = 0.0

    if main_agent_type == 'PREY':
        main_policy = prey_policy
        opponent_policy = predator_policy
        same_species_policy = clone_frozen_agent(prey_model_path, agent_id=2)
        opponent_label = 'PREDATOR'
        same_species_label = 'PREY'
    else:
        main_policy = predator_policy
        opponent_policy = prey_policy
        same_species_policy = clone_frozen_agent(predator_model_path, agent_id=2)
        opponent_label = 'PREY'
        same_species_label = 'PREDATOR'

    same_species_policy.epsilon = 0.0

    tracker = EventTracker()
    records: List[Dict] = []

    with track_ecosim_events(tracker):
        env = EcoSimEnv(
            agent_id=0,
            num_prey=num_prey,
            num_predators=num_predators,
            agent_type=main_agent_type,
            map_path=map_path,
            memory=memory,
            opponent_agent=opponent_policy,
            opponent_type=opponent_label,
            same_species_agent=same_species_policy,
            same_species_type=same_species_label,
            frozen_policy_epsilon=frozen_policy_epsilon,
        )

        observation = env.reset(seed=eval_seed + seed_offset)
        previous_alive_ids = {agent.agent_id for agent in [env.agent] + list(env.other_agents) if agent.is_alive()}
        known_agent_types = {agent.agent_id: agent.agent_type for agent in [env.agent] + list(env.other_agents)}
        cumulative_reward = 0.0

        for step_index in range(1, long_episode_steps + 1):
            main_state = main_policy.discretize_state(observation)
            action = main_policy.select_action(main_state, training=False)
            observation, reward, done, info = env.step(action)
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
    results_df['prey_model_path'] = str(prey_model_path)
    results_df['predator_model_path'] = str(predator_model_path)
    results_df['run_name'] = run_name
    results_df['seed'] = eval_seed + seed_offset
    results_df['long_episode_steps'] = long_episode_steps

    csv_path = results_dir / f"{run_name}__{prey_model_path.stem}__{predator_model_path.stem}_{len(results_df)}steps.csv"
    results_df.to_csv(csv_path, index=False)

    summary = {
        'run_name': run_name,
        'prey_model_path': str(prey_model_path),
        'predator_model_path': str(predator_model_path),
        'seed': eval_seed + seed_offset,
        'steps_ran': int(len(results_df)),
        'final_reward': float(results_df['cumulative_reward'].iloc[-1]) if not results_df.empty else 0.0,
        'final_prey_population': int(results_df['prey_population'].iloc[-1]) if not results_df.empty else 0,
        'final_predator_population': int(results_df['predator_population'].iloc[-1]) if not results_df.empty else 0,
        'cumulative_predation_events': int(results_df['cumulative_predation_events'].iloc[-1]) if not results_df.empty else 0,
        'cumulative_prey_births': int(results_df['cumulative_prey_births'].iloc[-1]) if not results_df.empty else 0,
        'cumulative_predator_births': int(results_df['cumulative_predator_births'].iloc[-1]) if not results_df.empty else 0,
        'csv_path': str(csv_path),
    }
    return results_df, summary


if __name__ == '__main__':
    all_summaries = []
    for cycle_id in cycle_ids:
        prey_model = ROOT / 'src' / 'models' / f'trained_prey_40_protocol2_cycle{cycle_id}.pkl'
        predator_model = ROOT / 'src' / 'models' / f'trained_predator_40_protocol2_cycle{cycle_id}.pkl'
        for seed_offset in seed_offsets:
            for agent_type in ['PREY', 'PREDATOR']:
                run_name = f'cycle{cycle_id}_{agent_type.lower()}_seed{seed_offset}'
                print('Running', run_name)
                df, summary = run_long_episode_experiment(prey_model, predator_model, agent_type, run_name, seed_offset)
                all_summaries.append(summary)
                print(f"Saved {summary['csv_path']} | steps={summary['steps_ran']} | final prey={summary['final_prey_population']} | final predator={summary['final_predator_population']}")

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = results_dir / 'run40_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print('All done. Summary saved to', summary_csv)
