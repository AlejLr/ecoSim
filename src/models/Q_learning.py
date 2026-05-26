import numpy as np
from collections import defaultdict
import random

from config.config import *


class QLearningAgent:
    """Tabular Q-Learning agent with discretized state space

    PREY (6-dim, conditional encoding):
      [energy, primary_dist, primary_dir_x, primary_dir_y, food_detected, predator_detected]
      When predator_detected=1, obs[1-3] point toward predator; otherwise toward food.

    PREDATOR (6-dim):
      [energy, prey_dist, prey_dir_x, prey_dir_y, prey_detected, prey_density]
      prey_density=1 if 2+ prey are within vision radius.

    Discretization:
    - energy:          5 levels  [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    - primary_dist:    3 levels  [close, medium, far]
    - primary_dir_x:   3 levels  [left, center, right]
    - primary_dir_y:   3 levels  [up, center, down]
    - target_detected: 2 levels  [not detected, detected]   (obs[4])
    - context:         2 levels  [0, 1]                     (obs[5])

    Total state space: 5 × 3 × 3 × 3 × 2 × 2 = 540 states
    Action space: 10 actions (8 moves + eat + idle)
    """

    def __init__(self, agent_id=0, num_actions=10, num_states=540, epsilon_start=None, epsilon_decay=None, epsilon_min=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.num_states = num_states
        
        # Q-table: maps (state) -> {action: q_value}
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        
        # Exploration parameters
        self.epsilon = EPSILON_START if epsilon_start is None else float(epsilon_start)
        self.epsilon_decay = EPSILON_DECAY if epsilon_decay is None else float(epsilon_decay)
        self.epsilon_min = EPSILON_MIN if epsilon_min is None else float(epsilon_min)
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        
        # For tracking
        self.episode_rewards = []
        self.episode_steps = []
    
    def discretize_state(self, obs):
        """Convert 6-dim continuous observation to discrete state tuple.

        obs: [energy, primary_dist, primary_dir_x, primary_dir_y, target_detected, context]
        State space: 5 × 3 × 3 × 3 × 2 × 2 = 1,080 states
        """
        energy = int(np.clip(obs[0] * 5, 0, 4))
        primary_distance = int(np.clip(obs[1] * 3, 0, 2))
        direction_x = int(np.clip(obs[2] * 3, 0, 2))
        direction_y = int(np.clip(obs[3] * 3, 0, 2))
        target_detected = int(np.clip(obs[4], 0, 1))
        context = int(np.clip(obs[5], 0, 1))

        return (energy, primary_distance, direction_x, direction_y, target_detected, context)

    def _state_distance(self, state_a, state_b):
        """Compute a cheap distance between two discrete state tuples."""
        return sum((int(a) - int(b)) ** 2 for a, b in zip(state_a, state_b))

    def nearest_known_state(self, state):
        """Return the closest known state in the Q-table, or None if empty."""
        known_states = getattr(self, 'saved_state_keys', None)
        if known_states is None:
            known_states = self.q_table.keys()

        known_states = list(known_states)
        if not known_states:
            return None

        return min(known_states, key=lambda known_state: self._state_distance(known_state, state))
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection
        
        state: discretized state tuple
        training: if False, use greedy only (no exploration)
        returns: action index (0-10)
        """
        def valid_actions_for_state(state_tuple):
            """Mask out actions that are clearly invalid in the current state.

            For now we only prevent `eat` when the target is not detected. This
            keeps exploration from wasting steps on blind hunting attempts while
            leaving the rest of the action space unchanged.
            """
            valid_actions = list(range(self.num_actions))
            target_detected = int(state_tuple[4]) if len(state_tuple) > 4 else 0
            if target_detected <= 0 and 8 in valid_actions:
                valid_actions.remove(8)
            return valid_actions

        valid_actions = valid_actions_for_state(state)

        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.choice(valid_actions)
        else:
            # Exploit: best Q-value, with nearest-state smoothing for unseen states
            if state in self.q_table:
                q_values = self.q_table[state]
            else:
                nearest_state = self.nearest_known_state(state)
                q_values = self.q_table[nearest_state] if nearest_state is not None else self.q_table[state]
            masked_q_values = np.array(q_values, copy=True)
            invalid_actions = set(range(self.num_actions)) - set(valid_actions)
            for action in invalid_actions:
                masked_q_values[action] = -np.inf
            return int(np.argmax(masked_q_values + np.random.random(self.num_actions) * 1e-6))
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_exploration(self, epsilon=None, epsilon_decay=None, epsilon_min=None):
        """Override exploration schedule parameters for a specific training protocol."""
        if epsilon is not None:
            self.epsilon = float(epsilon)
        if epsilon_decay is not None:
            self.epsilon_decay = float(epsilon_decay)
        if epsilon_min is not None:
            self.epsilon_min = float(epsilon_min)
    
    def save_model(self, filepath):
        """Save Q-table and metadata to file (convert defaultdict to dict for pickling)"""
        import pickle
        from pathlib import Path
        # Capture a small snapshot of important config values so experiments can be reproduced
        try:
            from config.config import (
                PREY_REPRODUCTION_THRESHOLD,
                PREY_REPRODUCTION_PROB_SCALE,
                PREY_CARRYING_CAPACITY,
                HUNTING_SUCCESS_BONUS,
                GRASS_ENERGY,
                TILE_DISTRIBUTION,
            )
            config_snapshot = {
                'PREY_REPRODUCTION_THRESHOLD': PREY_REPRODUCTION_THRESHOLD,
                'PREY_REPRODUCTION_PROB_SCALE': PREY_REPRODUCTION_PROB_SCALE,
                'PREY_CARRYING_CAPACITY': PREY_CARRYING_CAPACITY,
                'HUNTING_SUCCESS_BONUS': HUNTING_SUCCESS_BONUS,
                'GRASS_ENERGY': GRASS_ENERGY,
                'TILE_DISTRIBUTION': TILE_DISTRIBUTION,
            }
        except Exception:
            config_snapshot = None
        # Create directory if needed
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict for pickling
            'num_actions': self.num_actions,
            'num_states': self.num_states
        }
        # Attach config snapshot when available
        if config_snapshot is not None:
            model_data['config_snapshot'] = config_snapshot
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model_from_file(filepath):
        """Load Q-table and metadata from file and return agent"""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new agent with saved metadata
        agent = QLearningAgent(
            agent_id=0, 
            num_actions=model_data['num_actions'],
            num_states=model_data['num_states']
        )
        # Populate q_table with saved values
        agent.q_table = defaultdict(lambda: np.zeros(agent.num_actions), model_data['q_table'])
        # Keep a fast membership set for fallback logic in frozen/background policies.
        agent.saved_state_keys = set(model_data['q_table'].keys())
        # Optional: attach config snapshot for downstream inspection
        agent.config_snapshot = model_data.get('config_snapshot', None)
        return agent


def train_agent(env, agent, num_episodes=None):
    """Train a Q-learning agent on the environment
    
    env: Gym environment
    agent: QLearningAgent
    num_episodes: number of training episodes (uses config if None)
    
    returns: (episode_rewards, episode_steps)
    """
    if num_episodes is None:
        num_episodes = NUM_EPISODES
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"  Starting episode {episode + 1}...", end=' ', flush=True)
         
        obs = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < STEPS_PER_EPISODE:
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_obs, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            step += 1
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
        elif (episode + 1) % 10 == 0:
            print("done")
    
    return episode_rewards, episode_steps


def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate trained agent (no exploration)
    
    env: Gym environment
    agent: QLearningAgent
    num_episodes: number of evaluation episodes
    
    returns: (average_reward, average_steps)
    """
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < STEPS_PER_EPISODE:
            # Greedy action (no exploration)
            action = agent.select_action(state, training=False)
            next_obs, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            episode_reward += reward
            step += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
    
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    print(f"Evaluation over {num_episodes} episodes:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Steps: {avg_steps:.2f}")
    
    return avg_reward, avg_steps
