import gym
from gym import spaces
import numpy as np

from config.config import *
from environment.environment import grid_env
from agents.agent import Agent


class EcoSimEnv(gym.Env):
    """Simplified EcoSim as a Gym environment for single-agent Q-learning training
    
    Observation: 4 normalized floats [energy, thirst, food_count, water_count]
    Action space: 11 discrete actions (0-7: move, 8: eat, 9: drink, 10: idle)
    Reward: energy_gained - step_penalty, or death_penalty on death
    """
    
    def __init__(self, agent_id=0, num_other_agents=5, map_path=None):
        super(EcoSimEnv, self).__init__()
        
        self.agent_id = agent_id
        self.num_other_agents = num_other_agents
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
        
        # Create main agent
        import random
        x = random.randint(0, self.env.width - 1)
        y = random.randint(0, self.env.height - 1)
        self.agent = Agent(self.agent_id, (x, y))
        self.env.agents.append(self.agent)
        self.env.agent_grid[x, y] += 1
        self.env.agents_by_position[(x, y)].append(self.agent)
        
        # Create other agents (dummy agents that don't learn)
        self.other_agents = []
        for i in range(self.num_other_agents):
            x = random.randint(0, self.env.width - 1)
            y = random.randint(0, self.env.height - 1)
            other_agent = Agent(self.agent_id + i + 1, (x, y))
            self.other_agents.append(other_agent)
            self.env.agents.append(other_agent)
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(other_agent)
        
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
            else:
                # Dead agents might respawn or just be removed
                pass
        
        # Grow tiles (if needed)
        for x in range(self.env.width):
            for y in range(self.env.height):
                tile = self.env.tiles[x][y]
                if hasattr(tile, 'grow'):
                    tile.grow()
        
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
        """Get observation from agent"""
        obs_dict = self.agent.get_observation(self.env)
        
        # Convert dict to numpy array in consistent order
        obs_array = np.array([
            obs_dict["energy"],
            obs_dict["thirst"],
            obs_dict["food_nearby"],
            obs_dict["water_nearby"],
            obs_dict["other_agents_nearby"]
        ], dtype=np.float32)
        
        return obs_array
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass
    
    def close(self):
        """Clean up"""
        pass
