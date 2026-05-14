"""Run a population simulation using the saved prey and predator policies.

This script loads `trained_prey.pkl` and `trained_predator.pkl`, applies them
to all agents of the matching species, and reports how the system behaves
without further learning.

Run: python -m src.models.run_saved_agents_population
"""
import copy
import csv
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config.config import *
from environment.environment import grid_env
from agents.agent import Prey, Predator
from models.Q_learning import QLearningAgent


def _load_policy(path):
    policy = QLearningAgent.load_model_from_file(str(path))
    policy.epsilon = 0.0
    return policy


def _spawn_agents(env, num_prey, num_predators, prey_policy, predator_policy):
    agents = []
    agent_id = 0

    for _ in range(num_prey):
        x = random.randint(0, env.width - 1)
        y = random.randint(0, env.height - 1)
        prey = Prey(agent_id, (x, y))
        prey.q_learning = copy.deepcopy(prey_policy)
        prey.q_learning.agent_id = agent_id
        agents.append(prey)
        env.agents.append(prey)
        env.agent_grid[x, y] += 1
        env.agents_by_position[(x, y)].append(prey)
        agent_id += 1

    for _ in range(num_predators):
        x = random.randint(0, env.width - 1)
        y = random.randint(0, env.height - 1)
        pred = Predator(agent_id, (x, y))
        pred.q_learning = copy.deepcopy(predator_policy)
        pred.q_learning.agent_id = agent_id
        agents.append(pred)
        env.agents.append(pred)
        env.agent_grid[x, y] += 1
        env.agents_by_position[(x, y)].append(pred)
        agent_id += 1

    return agents


def run_population(episodes=5, max_steps=100, num_prey=4, num_predators=2):
    models_dir = Path(__file__).resolve().parent
    prey_model_path = models_dir / "trained_prey.pkl"
    predator_model_path = models_dir / "trained_predator.pkl"

    if not prey_model_path.exists():
        raise FileNotFoundError(f"Missing prey model: {prey_model_path}")
    if not predator_model_path.exists():
        raise FileNotFoundError(f"Missing predator model: {predator_model_path}")

    prey_policy = _load_policy(prey_model_path)
    predator_policy = _load_policy(predator_model_path)

    results_dir = models_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loaded policies:")
    print(f"  prey: {prey_model_path}")
    print(f"  predator: {predator_model_path}")
    print(f"  carrying capacity: {PREY_CARRYING_CAPACITY}")

    summary_rows = []

    for episode in range(episodes):
        env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
        env.generate()
        agents = _spawn_agents(env, num_prey, num_predators, prey_policy, predator_policy)

        per_step_rows = []
        total_kills = 0

        for step in range(max_steps):
            alive_agents = [a for a in agents if a.is_alive()]
            if not alive_agents:
                break

            random.shuffle(alive_agents)
            prey_before = len([a for a in alive_agents if a.agent_type == "PREY"])
            predator_before = len([a for a in alive_agents if a.agent_type == "PREDATOR"])

            for agent in alive_agents:
                if not agent.is_alive():
                    continue

                obs = agent.get_observation(env)
                state = agent.q_learning.discretize_state(obs)
                action = agent.q_learning.select_action(state, training=False)
                reward = agent.action(action, env)

                if agent.agent_type == "PREDATOR" and action == 8 and reward > 0:
                    total_kills += 1

            # Allow depleted grass tiles to advance toward regrowth
            env.update_resources()

            # Reproduction for any species that implements it
            alive_prey_count = len([a for a in agents if a.is_alive() and a.agent_type == "PREY"])
            offspring_list = []
            next_agent_id = max(a.agent_id for a in agents) + 1 if agents else 0
            for agent in list(agents):
                if agent.is_alive() and hasattr(agent, "reproduce"):
                    offspring = agent.reproduce(
                        env,
                        next_agent_id,
                        current_prey_count=alive_prey_count,
                        carrying_capacity=PREY_CARRYING_CAPACITY,
                                            all_agents=agents,
                    )
                    if offspring is not None:
                        if offspring.agent_type == "PREY":
                            offspring.q_learning = copy.deepcopy(prey_policy)
                        else:
                            offspring.q_learning = copy.deepcopy(predator_policy)
                        offspring.q_learning.agent_id = next_agent_id
                        offspring_list.append(offspring)
                        next_agent_id += 1
                        if offspring.agent_type == "PREY":
                            alive_prey_count += 1

            for offspring in offspring_list:
                agents.append(offspring)
                env.agents.append(offspring)
                x, y = offspring.position
                env.agent_grid[x, y] += 1
                env.agents_by_position[(x, y)].append(offspring)

            dead_agents = [a for a in agents if not a.is_alive()]
            for dead_agent in dead_agents:
                dead_agent.die(env)

            prey_after = len([a for a in agents if a.is_alive() and a.agent_type == "PREY"])
            predator_after = len([a for a in agents if a.is_alive() and a.agent_type == "PREDATOR"])

            per_step_rows.append((step, prey_after, predator_after, total_kills))

        avg_prey = np.mean([row[1] for row in per_step_rows]) if per_step_rows else 0
        avg_predator = np.mean([row[2] for row in per_step_rows]) if per_step_rows else 0
        summary_rows.append((episode, len(per_step_rows), avg_prey, avg_predator, total_kills))

        csv_path = results_dir / f"saved_agents_episode_{episode}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "prey_count", "predator_count", "predator_kills"])
            writer.writerows(per_step_rows)

        print(
            f"Episode {episode}: steps={len(per_step_rows)} avg_prey={avg_prey:.2f} "
            f"avg_predator={avg_predator:.2f} predator_kills={total_kills} -> saved {csv_path}"
        )

    print("\nSummary across episodes:")
    for episode, steps, avg_prey, avg_predator, kills in summary_rows:
        print(
            f"  Ep {episode}: steps={steps} avg_prey={avg_prey:.2f} "
            f"avg_predator={avg_predator:.2f} predator_kills={kills}"
        )


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    run_population(episodes=5, max_steps=100, num_prey=4, num_predators=2)
