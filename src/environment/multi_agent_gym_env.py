import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict

from config.config import *
from environment.environment import grid_env
from agents.agent import Prey, Predator
from models.Q_learning import QLearningAgent


class MultiAgentEcoSimEnv(gym.Env):
    """Multi-agent EcoSim where all agents learn via Q-learning.

    All agents of the same species share observations through the same 7-dim
    state space. Each agent has its own Q-table (or a shared one under Protocol 3).

    Observation: 6 normalized floats
      PREY:     [energy, primary_dist, primary_dir_x, primary_dir_y, food_detected, predator_detected]
      PREDATOR: [energy, prey_dist, prey_dir_x, prey_dir_y, prey_detected, prey_density]

    Action space: 10 discrete actions (0-7: move directions, 8: eat, 9: idle)
    """

    def __init__(self, num_prey=12, num_predators=2, map_path=None):
        super(MultiAgentEcoSimEnv, self).__init__()

        self.num_prey = num_prey
        self.num_predators = num_predators
        self.map_path = map_path

        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        self.env = None
        self.all_agents = []
        self.steps = 0
        self.next_agent_id = 1000
        self.prey_carrying_capacity = PREY_CARRYING_CAPACITY

        self.prey_rewards = defaultdict(float)
        self.predator_rewards = defaultdict(float)

    def reset(self, seed=None):
        """Reset environment with multi-agent setup."""
        from config.config import SEED

        if seed is None:
            seed = SEED

        if seed is not None:
            self.action_space.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])

        if self.map_path:
            self.env.use_test(self.map_path)
        else:
            self.env.generate()

        self.all_agents = []
        agent_counter = 0

        for i in range(self.num_prey):
            x = random.randint(0, self.env.width - 1)
            y = random.randint(0, self.env.height - 1)
            prey = Prey(agent_counter, (x, y))
            prey.q_learning = QLearningAgent(agent_id=agent_counter, num_actions=10, num_states=540)
            self.all_agents.append(prey)
            self.env.agents.append(prey)
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(prey)
            agent_counter += 1

        for i in range(self.num_predators):
            x = random.randint(0, self.env.width - 1)
            y = random.randint(0, self.env.height - 1)
            pred = Predator(agent_counter, (x, y))
            pred.q_learning = QLearningAgent(agent_id=agent_counter, num_actions=10, num_states=540)
            self.all_agents.append(pred)
            self.env.agents.append(pred)
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(pred)
            agent_counter += 1

        self.steps = 0
        self.prey_rewards.clear()
        self.predator_rewards.clear()
        self.prey_carrying_capacity = PREY_CARRYING_CAPACITY

        if self.all_agents:
            return self.all_agents[0].get_observation(self.env)
        return np.zeros(6, dtype=np.float32)

    def step(self, actions=None):
        """Execute one step with ALL agents.

        Actions dict maps agent_id -> action_idx for externally controlled agents.
        Agents not in the dict use their own Q-table (epsilon-greedy).

        Step ordering:
          1. Collect observations and execute actions; accumulate step rewards.
          2. Advance resources.
          3. Remove dead agents.
          4. Resolve reproduction; accumulate reproduction rewards.
          5. Q-update each agent with full reward (step + reproduction).
          6. Add offspring to environment.
          7. Decay epsilon once if episode is done.
        """
        if actions is None:
            actions = {}

        alive_agents = [a for a in self.all_agents if a.is_alive()]

        if not alive_agents:
            return np.zeros(6, dtype=np.float32), 0, True, {}

        random.shuffle(alive_agents)

        # --- Phase 1: Execute actions, collect transitions (defer Q-update) ---
        # Maps agent_id -> (state, action, step_reward, next_state, died)
        transitions = {}

        for agent in alive_agents:
            if not agent.is_alive():
                continue

            obs = agent.get_observation(self.env)
            state = agent.q_learning.discretize_state(obs)

            action = actions.get(agent.agent_id,
                                 agent.q_learning.select_action(state, training=True))

            step_reward = agent.action(action, self.env)
            died_this_step = not agent.is_alive()
            if died_this_step:
                step_reward += DEATH_PENALTY
            agent.episode_reward += step_reward

            next_obs = agent.get_observation(self.env)
            next_state = agent.q_learning.discretize_state(next_obs)

            transitions[agent.agent_id] = (state, action, step_reward, next_state, died_this_step)

            if agent.agent_type == "PREY":
                self.prey_rewards[agent.agent_id] += step_reward
            else:
                self.predator_rewards[agent.agent_id] += step_reward

        # --- Phase 2: Advance ecological resources ---
        self.env.update_resources()
        self.steps += 1

        # --- Phase 3: Remove dead agents ---
        for dead in [a for a in self.all_agents if not a.is_alive()]:
            dead.die(self.env)

        # --- Phase 4: Reproduction ---
        offspring_list = []
        repro_rewards = {}  # agent_id -> REPRODUCTION_REWARD

        current_prey_count = sum(1 for a in self.all_agents if a.is_alive() and a.agent_type == "PREY")
        current_predator_count = sum(1 for a in self.all_agents if a.is_alive() and a.agent_type == "PREDATOR")

        for agent in self.all_agents:
            if not agent.is_alive() or not hasattr(agent, 'reproduce'):
                continue

            if agent.agent_type == "PREY":
                offspring = agent.reproduce(
                    self.env,
                    self.next_agent_id,
                    current_prey_count=current_prey_count,
                    carrying_capacity=self.prey_carrying_capacity,
                    all_agents=self.all_agents,
                )
            else:
                offspring = agent.reproduce(
                    self.env,
                    self.next_agent_id,
                    current_prey_count=current_prey_count,
                    current_predator_count=current_predator_count,
                    all_agents=self.all_agents,
                )

            if offspring is not None:
                repro_rewards[agent.agent_id] = REPRODUCTION_REWARD
                agent.episode_reward += REPRODUCTION_REWARD
                if agent.agent_type == "PREY":
                    self.prey_rewards[agent.agent_id] += REPRODUCTION_REWARD
                    current_prey_count += 1
                else:
                    self.predator_rewards[agent.agent_id] += REPRODUCTION_REWARD
                    current_predator_count += 1
                offspring_list.append((agent, offspring))
                self.next_agent_id += 1

        # --- Phase 5: Q-updates with complete reward (step + reproduction) ---
        for agent in alive_agents:
            aid = agent.agent_id
            if aid not in transitions:
                continue
            state, action, step_reward, next_state, died = transitions[aid]
            total_reward = step_reward + repro_rewards.get(aid, 0.0)
            agent.q_learning.update(state, action, total_reward, next_state, died)

        # --- Phase 6: Add offspring to environment, inherit parent Q-table ---
        for parent, offspring in offspring_list:
            offspring.q_learning = QLearningAgent(
                agent_id=offspring.agent_id, num_actions=10, num_states=540
            )
            for state_key, q_vals in parent.q_learning.q_table.items():
                offspring.q_learning.q_table[state_key] = np.array(q_vals, copy=True)
            offspring.q_learning.epsilon = min(0.5, parent.q_learning.epsilon * 3)
            offspring.q_learning.epsilon_decay = parent.q_learning.epsilon_decay
            offspring.q_learning.epsilon_min = parent.q_learning.epsilon_min
            self.all_agents.append(offspring)
            self.env.agents.append(offspring)
            x, y = offspring.position
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(offspring)

        # --- Phase 7: Check done and decay epsilon once per episode ---
        alive_agents = [a for a in self.all_agents if a.is_alive()]
        done = self.steps >= STEPS_PER_EPISODE or len(alive_agents) == 0

        if done:
            # Epsilon decays once per completed episode, not per step.
            for agent in self.all_agents:
                if agent.q_learning is not None:
                    agent.q_learning.decay_epsilon()

        if alive_agents:
            main_agent = alive_agents[0]
            return main_agent.get_observation(self.env), main_agent.episode_reward, done, {
                "num_alive": len(alive_agents),
                "prey_count": sum(1 for a in alive_agents if a.agent_type == "PREY"),
                "predator_count": sum(1 for a in alive_agents if a.agent_type == "PREDATOR"),
                "prey_rewards": dict(self.prey_rewards),
                "predator_rewards": dict(self.predator_rewards),
            }

        return np.zeros(6, dtype=np.float32), 0, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_agent_learning_stats(self):
        """Get learning statistics for all alive agents."""
        stats = {"prey": {}, "predator": {}}

        for agent in self.all_agents:
            if not agent.is_alive():
                continue
            q_table_size = len(agent.q_learning.q_table)
            avg_q_value = (
                float(np.mean([np.mean(v) for v in agent.q_learning.q_table.values()]))
                if agent.q_learning.q_table else 0.0
            )
            agent_stats = {
                "id": agent.agent_id,
                "reward": agent.episode_reward,
                "q_states_visited": q_table_size,
                "avg_q_value": avg_q_value,
                "epsilon": agent.q_learning.epsilon,
            }
            bucket = "prey" if agent.agent_type == "PREY" else "predator"
            stats[bucket][agent.agent_id] = agent_stats

        return stats
