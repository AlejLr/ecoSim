import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from pathlib import Path

from config.config import *
from environment.environment import grid_env
from agents.agent import Prey, Predator
from models.Q_learning import QLearningAgent


class EcoSimEnv(gym.Env):
    """EcoSim as a Gym environment for single-agent Q-learning training
    
    Main learning agent is typically PREY. Other agents are random actors.
    
    Observation: 5 normalized floats [energy, thirst, food_count, water_count, other_agents]
    Action space: 11 discrete actions (0-7: move, 8: eat, 9: drink, 10: idle)
    Reward: energy_gained - step_penalty, or death_penalty on death
    """
    
    def __init__(self, agent_id=0, num_prey=4, num_predators=2, agent_type="PREDATOR", map_path=None, memory=False):
        super(EcoSimEnv, self).__init__()
        
        self.agent_id = agent_id
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.agent_type = agent_type.upper()
        self.map_path = map_path
        self.memory = memory  # Use pre-trained models for other agents
        
        # Action space: 8 moves + eat + drink + idle
        self.action_space = spaces.Discrete(11)
        
        # Observation space: [energy, thirst, target_distance, target_dir_x, target_dir_y, target_detected]
        # Same for both prey and predator (6 dims)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        
        # Initialize environment
        self.env = None
        self.agent = None
        self.other_agents = []
        self.steps = 0
        self.next_agent_id = agent_id + 1000  # Start offspring IDs high to avoid conflicts
        
        # Load pre-trained models if memory is enabled
        self.trained_prey_model = None
        self.trained_predator_model = None
        if self.memory:
            self._load_trained_models()
    
    def _load_trained_models(self):
        """Load pre-trained models for opponent agents (not for learning agent)"""
        print("Loading opponent models...")
        try:
            models_dir = Path(__file__).resolve().parent.parent / "models"
            
            # Only load the OPPOSITE agent type to avoid self-reference
            # If we're training PREY, load PREDATOR model for predator agents
            # If we're training PREDATOR, load PREY model for prey agents
            
            if self.agent_type == "PREY":
                # Training prey - load predator model for other predators
                pred_path = models_dir / "trained_predator.pkl"
                if pred_path.exists():
                    try:
                        print(f"  Loading PREDATOR opponent from {pred_path}...")
                        self.trained_predator_model = QLearningAgent.load_model_from_file(str(pred_path))
                        print(f"✓ Loaded trained PREDATOR opponent")
                    except Exception as e:
                        print(f"⚠ Error loading PREDATOR model: {e}")
                        self.trained_predator_model = None
                else:
                    print(f"⚠ Predator model not found at {pred_path}")
                    
            else:  # Training predator
                # Training predator - load prey model for other prey
                prey_path = models_dir / "trained_prey.pkl"
                if prey_path.exists():
                    try:
                        print(f"  Loading PREY opponent from {prey_path}...")
                        self.trained_prey_model = QLearningAgent.load_model_from_file(str(prey_path))
                        print(f"✓ Loaded trained PREY opponent")
                    except Exception as e:
                        print(f"⚠ Error loading PREY model: {e}")
                        self.trained_prey_model = None
                else:
                    print(f"⚠ Prey model not found at {prey_path}")
                    
        except Exception as e:
            print(f"⚠ Error in model loading: {e}. Using random actions.")
        
    def _get_other_agent_action(self, agent):
        """Get action for other agent - use trained model if available, else random"""
        if agent.agent_type == "PREY" and self.trained_prey_model:
            obs = agent.get_observation(self.env)
            state = self.trained_prey_model.discretize_state(obs)
            action = self.trained_prey_model.select_action(state, training=False)
            return action
        elif agent.agent_type == "PREDATOR" and self.trained_predator_model:
            obs = agent.get_observation(self.env)
            state = self.trained_predator_model.discretize_state(obs)
            action = self.trained_predator_model.select_action(state, training=False)
            return action
        else:
            # Fallback to random action
            return random.randint(0, 10)
        
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
        
        # Other agents move - use trained models if memory enabled, else random
        for other_agent in self.other_agents:
            if other_agent.is_alive():
                if self.memory and (self.trained_prey_model or self.trained_predator_model):
                    # Use trained model for action selection
                    action = self._get_other_agent_action(other_agent)
                    other_agent.action(action, self.env)
                else:
                    # Use random movement
                    other_agent.test(self.env)
        
        # Handle reproduction for other prey agents
        offspring_list = []
        for agent in self.other_agents:
            if agent.is_alive() and hasattr(agent, 'reproduce') and agent.agent_type == "PREY":
                offspring = agent.reproduce(self.env, self.next_agent_id)
                if offspring is not None:
                    offspring_list.append(offspring)
                    self.next_agent_id += 1
        
        # Add offspring to environment
        for offspring in offspring_list:
            self.other_agents.append(offspring)
            self.env.agents.append(offspring)
            x, y = offspring.position
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(offspring)
        
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
