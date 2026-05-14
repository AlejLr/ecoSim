"""Probe whether the saved predator policy can hunt when placed near prey.

This is a focused sanity check using the saved prey and predator policies.
It spawns one prey and one predator in adjacent cells and runs a short rollout.

Run: python -m src.models.run_saved_agents_hunt_probe
"""
import copy
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config.config import *
from environment.environment import grid_env
from agents.agent import Prey, Predator
from models.Q_learning import QLearningAgent


def run_probe(max_steps=20):
    models_dir = Path(__file__).resolve().parent
    prey_model_path = models_dir / "trained_prey.pkl"
    predator_model_path = models_dir / "trained_predator.pkl"

    prey_policy = QLearningAgent.load_model_from_file(str(prey_model_path))
    predator_policy = QLearningAgent.load_model_from_file(str(predator_model_path))
    prey_policy.epsilon = 0.0
    predator_policy.epsilon = 0.0

    env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
    env.generate()

    center = (env.width // 2, env.height // 2)
    prey = Prey(0, center)
    predator = Predator(1, (center[0], min(env.height - 1, center[1] + 1)))

    prey.q_learning = copy.deepcopy(prey_policy)
    prey.q_learning.agent_id = prey.agent_id
    predator.q_learning = copy.deepcopy(predator_policy)
    predator.q_learning.agent_id = predator.agent_id

    env.agents.extend([prey, predator])
    env.agent_grid[prey.position[0], prey.position[1]] += 1
    env.agents_by_position[prey.position].append(prey)
    env.agent_grid[predator.position[0], predator.position[1]] += 1
    env.agents_by_position[predator.position].append(predator)

    print(f"Starting probe: prey at {prey.position}, predator at {predator.position}")

    kill_step = None
    for step in range(max_steps):
        if not prey.is_alive() or not predator.is_alive():
            break

        # Predator first, then prey, to expose hunt behavior immediately.
        for agent in (predator, prey):
            if not agent.is_alive():
                continue
            obs = agent.get_observation(env)
            state = agent.q_learning.discretize_state(obs)
            action = agent.q_learning.select_action(state, training=False)
            reward = agent.action(action, env)
            if agent is predator and action == 8 and reward > 0 and prey.energy <= 0:
                kill_step = step
                break

        dead_agents = [a for a in env.agents if not a.is_alive()]
        for dead_agent in dead_agents:
            dead_agent.die(env)

        if kill_step is not None:
            break

    print(f"Prey alive: {prey.is_alive()}")
    print(f"Predator alive: {predator.is_alive()}")
    print(f"Kill step: {kill_step}")
    if kill_step is None:
        print("No hunt occurred in the probe.")
    else:
        print("Predator successfully hunted the prey in the probe.")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    run_probe(max_steps=20)
