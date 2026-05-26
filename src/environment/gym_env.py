import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from pathlib import Path

from config.config import *
from config.utils import get_latest_model_path
from environment.environment import grid_env
from agents.agent import Prey, Predator
from models.Q_learning import QLearningAgent


class EcoSimEnv(gym.Env):
    """EcoSim as a Gym environment for single-agent Q-learning training
    
    Main learning agent is typically PREY. Other agents are random actors.
    
    Observation: 6 normalized floats [energy, primary_dist, primary_dir_x, primary_dir_y, target_detected, context]
    Action space: 10 discrete actions (0-7: move, 8: eat, 9: idle)
    Reward: energy_gained - step_penalty, or death_penalty on death
    """
    
    def __init__(self, agent_id=0, num_prey=4, num_predators=2, agent_type="PREDATOR", map_path=None, memory=False, opponent_agent=None, opponent_type=None, same_species_agent=None, same_species_type=None, frozen_policy_epsilon=0.01, grid_size=None):
        super(EcoSimEnv, self).__init__()
        
        self.agent_id = agent_id
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.agent_type = agent_type.upper()
        self.map_path = map_path
        self.memory = memory  # Use pre-trained models for other agents
        self.opponent_agent = opponent_agent  # Direct agent for alternating training
        self.opponent_type = opponent_type.upper() if opponent_type else None
        self.same_species_agent = same_species_agent  # Frozen policy for learner species background agents
        self.same_species_type = same_species_type.upper() if same_species_type else None
        self.frozen_policy_epsilon = float(frozen_policy_epsilon)
        self.grid_size = tuple(grid_size) if grid_size is not None else GRID_SUBENV
        
        # Action space: 8 moves + eat + idle
        self.action_space = spaces.Discrete(10)
        
        # Observation space: 6-dim, same shape for both species
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
        
        if self.opponent_agent is not None:
            opponent_label = self.opponent_type or "unknown"
            print(f"✓ Using custom opponent agent for {opponent_label}")
        if self.same_species_agent is not None:
            same_species_label = self.same_species_type or "unknown"
            print(f"✓ Using frozen same-species policy for {same_species_label} (eps={self.frozen_policy_epsilon:.3f})")
        elif self.memory:
            self._load_trained_models()
    
    def _load_trained_models(self):
        """Load pre-trained models for opponent agents (not for learning agent)"""
        print("Loading opponent models (latest version)...")
        try:
            # Only load the OPPOSITE agent type to avoid self-reference
            # If we're training PREY, load PREDATOR model for other predators
            # If we're training PREDATOR, load PREY model for other prey agents
            
            if self.agent_type == "PREY":
                # Training prey - load latest predator model for other predators
                pred_path = get_latest_model_path("PREDATOR")
                if pred_path and pred_path.exists():
                    try:
                        print(f"  Loading PREDATOR opponent from {pred_path}...")
                        self.trained_predator_model = QLearningAgent.load_model_from_file(str(pred_path))
                        print(f"✓ Loaded trained PREDATOR opponent")
                    except Exception as e:
                        print(f"⚠ Error loading PREDATOR model: {e}")
                        self.trained_predator_model = None
                else:
                    print(f"⚠ Predator model not found (will use random actions)")
                    
            else:  # Training predator
                # Training predator - load latest prey model for other prey
                prey_path = get_latest_model_path("PREY")
                if prey_path and prey_path.exists():
                    try:
                        print(f"  Loading PREY opponent from {prey_path}...")
                        self.trained_prey_model = QLearningAgent.load_model_from_file(str(prey_path))
                        print(f"✓ Loaded trained PREY opponent")
                    except Exception as e:
                        print(f"⚠ Error loading PREY model: {e}")
                        self.trained_prey_model = None
                else:
                    print(f"⚠ Prey model not found (will use random actions)")
                    
        except Exception as e:
            print(f"⚠ Error in model loading: {e}. Using random actions.")
        
    def _get_other_agent_action(self, agent):
        """Get action for other agent - use custom opponent, trained model, or random"""

        def _nearest_saved_state_action(model, state_tuple):
            """Return greedy action from the closest saved state, or None if unavailable."""
            saved_keys = getattr(model, 'saved_state_keys', None)
            if not saved_keys:
                return None

            if state_tuple in saved_keys:
                return model.select_action(state_tuple, training=False)

            # Euclidean distance over the discrete tuple is enough for a cheap fallback.
            nearest_state = min(
                saved_keys,
                key=lambda saved_state: sum((int(a) - int(b)) ** 2 for a, b in zip(saved_state, state_tuple)),
            )
            return model.select_action(nearest_state, training=False)

        def _target_within_action_radius(current_agent):
            """Check whether a relevant target is within ACTION_RADIUS for a loaded policy."""
            target_type = "PREY" if current_agent.agent_type == "PREDATOR" else "PREDATOR"
            for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                    check_pos = (current_agent.position[0] + dx, current_agent.position[1] + dy)
                    if check_pos not in self.env.agents_by_position:
                        continue
                    for nearby_agent in self.env.agents_by_position[check_pos]:
                        if nearby_agent.agent_type == target_type and nearby_agent.is_alive():
                            return True
            return False

        # Tiny exploration for frozen policies to avoid fully deterministic swarms.
        if self.frozen_policy_epsilon > 0 and random.random() < self.frozen_policy_epsilon:
            return random.randint(0, 9)

        obs = None
        state = None

        def _ensure_state(model):
            nonlocal obs, state
            if obs is None:
                obs = agent.get_observation(self.env)
            if state is None:
                state = model.discretize_state(obs)
            return state

        def _eat_if_target_is_close():
            # Only meaningful for predator policies because action 8 means "eat prey".
            if agent.agent_type != "PREDATOR":
                return None
            if not _target_within_action_radius(agent):
                return None
            return 8
        
        # Priority 1: Use custom opponent agent if provided and matches type
        if self.opponent_agent is not None and self.opponent_type == agent.agent_type:
            try:
                state = _ensure_state(self.opponent_agent)
                if state not in getattr(self.opponent_agent, 'saved_state_keys', set()):
                    fallback_action = _eat_if_target_is_close()
                    if fallback_action is not None:
                        return fallback_action
                    action = _nearest_saved_state_action(self.opponent_agent, state)
                    if action is not None:
                        return action
                action = self.opponent_agent.select_action(state, training=False)
                return action
            except Exception:
                pass

        # Priority 2: Use frozen same-species policy for background agents of training species
        if self.same_species_agent is not None and self.same_species_type == agent.agent_type:
            try:
                state = _ensure_state(self.same_species_agent)
                if state not in getattr(self.same_species_agent, 'saved_state_keys', set()):
                    fallback_action = _eat_if_target_is_close()
                    if fallback_action is not None:
                        return fallback_action
                    action = _nearest_saved_state_action(self.same_species_agent, state)
                    if action is not None:
                        return action
                action = self.same_species_agent.select_action(state, training=False)
                return action
            except Exception:
                pass
        
        # Priority 3: Use trained model if available
        if agent.agent_type == "PREY" and self.trained_prey_model:
            state = _ensure_state(self.trained_prey_model)
            if state not in getattr(self.trained_prey_model, 'saved_state_keys', set()):
                fallback_action = _eat_if_target_is_close()
                if fallback_action is not None:
                    return fallback_action
                action = _nearest_saved_state_action(self.trained_prey_model, state)
                if action is not None:
                    return action
            action = self.trained_prey_model.select_action(state, training=False)
            return action
        elif agent.agent_type == "PREDATOR" and self.trained_predator_model:
            state = _ensure_state(self.trained_predator_model)
            if state not in getattr(self.trained_predator_model, 'saved_state_keys', set()):
                fallback_action = _eat_if_target_is_close()
                if fallback_action is not None:
                    return fallback_action
                action = _nearest_saved_state_action(self.trained_predator_model, state)
                if action is not None:
                    return action
            action = self.trained_predator_model.select_action(state, training=False)
            return action
        else:
            # Fallback to random action
            return random.randint(0, 9)
        
    def reset(self, seed=None):
        """Reset environment and return initial observation
        
        Args:
            seed: Optional seed for reproducibility (from gymnasium API)
        """
        from config.config import SEED
        
        if seed is None:
            seed = SEED
        
        # Seed the action space if seed is provided
        if seed is not None:
            self.action_space.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        # Create environment
        self.env = grid_env(self.grid_size[0], self.grid_size[1])
        
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
                if self.opponent_agent is not None or self.same_species_agent is not None or (self.memory and (self.trained_prey_model or self.trained_predator_model)):
                    other_action = self._get_other_agent_action(other_agent)
                    other_reward = other_agent.action(other_action, self.env)
                else:
                    other_agent.test(self.env)
                    other_reward = 0

                other_agent.episode_reward += other_reward

        # Advance ecological resources after all agents act
        self.env.update_resources()
        
        # Check if main agent died
        done = not self.agent.is_alive()
        if done:
            reward += DEATH_PENALTY
        
        # Clean up dead agents BEFORE reproduction (prevent dead from reproducing)
        dead_agents = [a for a in self.env.agents if not a.is_alive()]
        for dead_agent in dead_agents:
            dead_agent.die(self.env)
        
        # Handle reproduction for all other agents that support it
        offspring_list = []
        current_prey_count = len([a for a in self.other_agents if a.is_alive() and a.agent_type == "PREY"])
        current_predator_count = len([a for a in self.other_agents if a.is_alive() and a.agent_type == "PREDATOR"])
        
        for agent in self.other_agents:
            if agent.is_alive() and hasattr(agent, 'reproduce'):
                # Count reproduction attempts for telemetry
                try:
                    self.env.reproduction_attempts += 1
                except Exception:
                    pass

                if agent.agent_type == "PREY":
                    offspring = agent.reproduce(
                        self.env,
                        self.next_agent_id,
                        current_prey_count=current_prey_count,
                        carrying_capacity=PREY_CARRYING_CAPACITY,
                        all_agents=self.other_agents + [self.agent],
                    )
                else:  # PREDATOR
                    offspring = agent.reproduce(
                        self.env,
                        self.next_agent_id,
                        current_prey_count=current_prey_count,
                        current_predator_count=current_predator_count,
                        all_agents=self.other_agents + [self.agent],
                    )
                
                if offspring is not None:
                    from agents.agent import reward_for_agent
                    # Award reproduction reward to background agent (BUG FIX: was not being accumulated)
                    agent.episode_reward += reward_for_agent(
                        agent.agent_type,
                        event="reproduce",
                    )
                    offspring_list.append(offspring)
                    self.next_agent_id += 1
                    try:
                        self.env.reproduction_successes += 1
                    except Exception:
                        pass
                    if offspring.agent_type == "PREY":
                        current_prey_count += 1
                    else:
                        current_predator_count += 1
        
        # Add offspring to environment
        for offspring in offspring_list:
            self.other_agents.append(offspring)
            self.env.agents.append(offspring)
            x, y = offspring.position
            self.env.agent_grid[x, y] += 1
            self.env.agents_by_position[(x, y)].append(offspring)
        
        # Handle main agent reproduction
        if self.agent.is_alive() and hasattr(self.agent, 'reproduce'):
            if self.agent.agent_type == "PREY":
                offspring = self.agent.reproduce(
                    self.env,
                    self.next_agent_id,
                    current_prey_count=current_prey_count,
                    carrying_capacity=PREY_CARRYING_CAPACITY,
                    all_agents=self.other_agents + [self.agent],
                )
            else:  # PREDATOR
                offspring = self.agent.reproduce(
                    self.env,
                    self.next_agent_id,
                    current_prey_count=current_prey_count,
                    current_predator_count=current_predator_count,
                    all_agents=self.other_agents + [self.agent],
                )
            
            if offspring is not None:
                from agents.agent import reward_for_agent
                repro_reward = reward_for_agent(self.agent.agent_type, event="reproduce")
                reward += repro_reward
                self.agent.episode_reward += repro_reward

                self.other_agents.append(offspring)
                self.env.agents.append(offspring)
                x, y = offspring.position
                self.env.agent_grid[x, y] += 1
                self.env.agents_by_position[(x, y)].append(offspring)
        
        # Check step limit
        self.steps += 1
        if self.steps >= STEPS_PER_EPISODE:
            done = True
        
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
