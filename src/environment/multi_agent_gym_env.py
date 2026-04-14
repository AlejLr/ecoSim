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
    """Multi-agent EcoSim where all agents learn via Q-learning
    
    Each agent (Prey and Predator) has its own QLearningAgent that learns
    a policy for its species. This allows studying emergent behavior.
    
    Observation: 5 normalized floats [energy, thirst, food_count, water_count, other_agents]
    Action space: 11 discrete actions (0-7: move, 8: eat, 9: drink, 10: idle)
    Reward: energy_gained - step_penalty, or death_penalty on death
    """
    
    def __init__(self, num_prey=4, num_predators=2, map_path=None):
        super(MultiAgentEcoSimEnv, self).__init__()
        
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.map_path = map_path
        
        # Action space: 8 moves + eat + drink + idle
        self.action_space = spaces.Discrete(11)
        
        # Observation space: [energy, thirst, food_nearby, water_nearby, other_agents]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        # Initialize environment
        self.env = None
        self.all_agents = []  # All agents that learn
        self.steps = 0
        
        # Track metrics per species
        self.prey_rewards = defaultdict(float)  # agent_id -> total reward
        self.predator_rewards = defaultdict(float)
        
    def reset(self):
        """Reset environment with multi-agent setup"""
        # Create environment
        self.env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
        
        if self.map_path:
            self.env.use_test(self.map_path)
        else:
            self.env.generate()
        
        # Create all learning agents
        self.all_agents = []
        agent_counter = 0
        
        # Create prey
        for i in range(self.num_prey):
            x = random.randint(0, self.env.width - 1)
            y = random.randint(0, self.env.height - 1)
            prey = Prey(agent_counter, (x, y))
            prey.q_learning = QLearningAgent(agent_id=agent_counter, num_actions=11, num_states=675)
            self.all_agents.append(prey)
            self.env.agents.append(prey)
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(prey)
            agent_counter += 1
        
        # Create predators
        for i in range(self.num_predators):
            x = random.randint(0, self.env.width - 1)
            y = random.randint(0, self.env.height - 1)
            pred = Predator(agent_counter, (x, y))
            pred.q_learning = QLearningAgent(agent_id=agent_counter, num_actions=11, num_states=675)
            self.all_agents.append(pred)
            self.env.agents.append(pred)
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(pred)
            agent_counter += 1
        
        self.steps = 0
        self.prey_rewards.clear()
        self.predator_rewards.clear()
        
        # Return observation of all agents (for compatibility, return first agent's obs)
        if self.all_agents:
            return self.all_agents[0].get_observation(self.env)
        return np.zeros(5, dtype=np.float32)
    
    def step(self, actions=None):
        """Execute one step with ALL agents taking Q-learning actions
        
        Args:
            actions: If provided, dict mapping agent_id -> action_idx
                    If None, all agents use their Q-learning policy
        
        Returns:
            obs, rewards, dones, infos (for compatibility, returns main agent data)
        """
        if actions is None:
            actions = {}
        
        alive_agents = [a for a in self.all_agents if a.is_alive()]
        
        if not alive_agents:
            return np.zeros(5, dtype=np.float32), 0, True, {}
        
        # All agents take actions and learn immediately
        for agent in alive_agents:
            if agent.is_alive():
                # Get current state and observation
                obs = agent.get_observation(self.env)
                state = agent.q_learning.discretize_state(obs)
                
                # Select action via epsilon-greedy
                if agent.agent_id in actions:
                    action = actions[agent.agent_id]
                else:
                    action = agent.q_learning.select_action(state, training=True)
                
                # Execute action and get immediate reward
                reward = agent.action(action, self.env)
                agent.episode_reward += reward
                
                # Get next state after action
                next_obs = agent.get_observation(self.env)
                next_state = agent.q_learning.discretize_state(next_obs)
                
                # Update Q-table IMMEDIATELY (not deferred)
                agent.q_learning.update(state, action, reward, next_state, not agent.is_alive())
                
                # Track by species
                if agent.agent_type == "PREY":
                    self.prey_rewards[agent.agent_id] += reward
                else:
                    self.predator_rewards[agent.agent_id] += reward
        
        # Decay epsilon for all agents
        for agent in alive_agents:
            agent.q_learning.decay_epsilon()
        
        # Advance episode
        self.steps += 1
        
        # Clean up dead agents (Q-table already updated during step)
        dead_agents = [a for a in self.all_agents if not a.is_alive()]
        for dead_agent in dead_agents:
            dead_agent.die(self.env)

        # Recompute alive agents after cleanup for consistent done/info
        alive_agents = [a for a in self.all_agents if a.is_alive()]
        done = self.steps >= STEPS_PER_EPISODE or len(alive_agents) == 0
        
        # Return main agent's observation (for compatibility)
        if alive_agents:
            main_agent = alive_agents[0]
            return main_agent.get_observation(self.env), main_agent.episode_reward, done, {
                "num_alive": len(alive_agents),
                "prey_count": len([a for a in alive_agents if a.agent_type == "PREY"]),
                "predator_count": len([a for a in alive_agents if a.agent_type == "PREDATOR"]),
                "prey_rewards": dict(self.prey_rewards),
                "predator_rewards": dict(self.predator_rewards),
            }
        
        return np.zeros(5, dtype=np.float32), 0, done, {}
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass
    
    def close(self):
        """Clean up"""
        pass
    
    def get_agent_learning_stats(self):
        """Get learning statistics for all agents"""
        stats = {
            "prey": {},
            "predator": {}
        }
        
        for agent in self.all_agents:
            if agent.is_alive():
                q_table_size = len(agent.q_learning.q_table)
                avg_q_value = np.mean([np.mean(v) for v in agent.q_learning.q_table.values()]) if agent.q_learning.q_table else 0
                
                agent_stats = {
                    "id": agent.agent_id,
                    "reward": agent.episode_reward,
                    "q_states_visited": q_table_size,
                    "avg_q_value": avg_q_value,
                    "epsilon": agent.q_learning.epsilon
                }
                
                if agent.agent_type == "PREY":
                    stats["prey"][agent.agent_id] = agent_stats
                else:
                    stats["predator"][agent.agent_id] = agent_stats
        
        return stats
