import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from config.config import *
from environment.environment import grid_env
from agents.agent import Agent, Prey, Predator


class EcoSimEnv(gym.Env):
    """EcoSim as a Gym environment for single-agent Q-learning training
    
    Main learning agent is typically PREY. Other agents are random actors.
    
    Observation: 5 normalized floats [energy, thirst, food_count, water_count, other_agents]
    Action space: 11 discrete actions (0-7: move, 8: eat, 9: drink, 10: idle)
    Reward: energy_gained - step_penalty, or death_penalty on death
    """
    
    def __init__(self, agent_id=0, num_prey=4, num_predators=2, agent_type="PREY", map_path=None):
        super(EcoSimEnv, self).__init__()
        
        self.agent_id = agent_id
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.agent_type = agent_type.upper()
        self.map_path = map_path
        
        # Action space: 8 moves + eat + drink + idle
        self.action_space = spaces.Discrete(11)
        
        # Observation space: [energy, thirst, food_nearby, water_nearby, other_agents]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        # Initialize environment
        self.env = None
        self.agent = None
        self.other_agents = []
        self.steps = 0
        
    def reset(self):
        """Reset environment and return initial observation"""
        # Create environment
        self.env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
        
        if self.map_path:
            self.env.use_test(self.map_path)
        else:
            self.env.generate()
        
        # Create main learning agent
        x = random.randint(0, self.env.width - 1)
        y = random.randint(0, self.env.height - 1)
        
        if self.agent_type == "PREY":
            self.agent = Prey(self.agent_id, (x, y))
        else:
            self.agent = Predator(self.agent_id, (x, y))
        
        self.env.agents.append(self.agent)
        self.env.agent_grid[x, y] += 1
        self.env.agents_by_position[(x, y)].append(self.agent)
        
        # Create other agents (random actors, not learning)
        self.other_agents = []
        
        # Add prey
        for i in range(self.num_prey):
            x = random.randint(0, self.env.width - 1)
            y = random.randint(0, self.env.height - 1)
            prey = Prey(self.agent_id + i + 1, (x, y))
            self.other_agents.append(prey)
            self.env.agents.append(prey)
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(prey)
        
        # Add predators
        for i in range(self.num_predators):
            x = random.randint(0, self.env.width - 1)
            y = random.randint(0, self.env.height - 1)
            pred = Predator(self.agent_id + self.num_prey + i + 1, (x, y))
            self.other_agents.append(pred)
            self.env.agents.append(pred)
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(pred)
        
        self.steps = 0
        self.agent.episode_reward = 0
        
        return self._get_obs()
    
    def step(self, action):
        """Execute one step and return (obs, reward, done, info)"""
        if not self.agent.is_alive():
            return self._get_obs(), DEATH_PENALTY, True, {}
        
        # Main agent takes action
        reward = self.agent.action(action, self.env)
        self.agent.episode_reward += reward
        
        # Other agents move randomly
        for other_agent in self.other_agents:
            if other_agent.is_alive():
                other_agent.test(self.env)
        
        # Check if agent died
        done = not self.agent.is_alive()
        if done:
            reward += DEATH_PENALTY
        
        # Check step limit
        self.steps += 1
        if self.steps >= STEPS_PER_EPISODE:
            done = True
        
        # Clean up dead agents
        dead_agents = [a for a in self.env.agents if not a.is_alive()]
        for dead_agent in dead_agents:
            dead_agent.die(self.env)
        
        obs = self._get_obs()
        info = {"episode_reward": self.agent.episode_reward, "steps": self.steps}
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get observation from agent - returns numpy array"""
        obs_array = self.agent.get_observation(self.env)
        return obs_array
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass
    
    def close(self):
        """Clean up"""
        pass
