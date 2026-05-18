import numpy as np
from collections import defaultdict
import random

from config.config import *


class QLearningAgent:
    """Tabular Q-Learning agent with discretized state space
    
    Both PREY and PREDATOR agents use identical 8-dimensional observations:
    [energy, thirst, target_distance, target_dir_x, target_dir_y, target_detected, can_reproduce, water_nearby]
    
    PREY: target = nearest predator (focus on survival)
    PREDATOR: target = nearest prey (focus on hunting)
    can_reproduce: binary flag indicating if reproduction is currently possible
    water_nearby: binary flag indicating if water is accessible within ACTION_RADIUS
    
    Discretization:
    - energy: 5 levels [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    - thirst: 5 levels [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    - target_distance: 3 levels [close, medium, far]
    - target_dir_x: 3 levels [left, center, right]
    - target_dir_y: 3 levels [up, center, down]
    - target_detected: 2 levels [not detected, detected]
    - can_reproduce: 2 levels [no, yes]
    - water_nearby: 2 levels [no, yes]
    
    Total state space: 5 × 5 × 3 × 3 × 3 × 2 × 2 × 2 = 5,400 states
    Action space: 11 actions (8 moves + eat + drink + idle)
    """
    
    def __init__(self, agent_id=0, num_actions=11, num_states=5400):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.num_states = num_states
        
        # Q-table: maps (state) -> {action: q_value}
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        
        # Exploration parameters
        self.epsilon = EPSILON_START
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        
        # For tracking
        self.episode_rewards = []
        self.episode_steps = []
    
    def discretize_state(self, obs):
        """Convert continuous observation to discrete state
        
        obs: numpy array [energy, thirst, target_distance, target_dir_x, target_dir_y, target_detected, can_reproduce, water_nearby]
        returns: state tuple for Q-table indexing
        
        State space: 5 × 5 × 3 × 3 × 3 × 2 × 2 × 2 = 5,400 states
        """
        # Ensure obs is normalized [0, 1]
        energy = int(np.clip(obs[0] * 5, 0, 4))
        thirst = int(np.clip(obs[1] * 5, 0, 4))
        target_distance = int(np.clip(obs[2] * 3, 0, 2))
        direction_x = int(np.clip(obs[3] * 3, 0, 2))
        direction_y = int(np.clip(obs[4] * 3, 0, 2))
        target_detected = int(np.clip(obs[5], 0, 1))
        can_reproduce = int(np.clip(obs[6], 0, 1))
        water_nearby = int(np.clip(obs[7], 0, 1))
        
        return (energy, thirst, target_distance, direction_x, direction_y, target_detected, can_reproduce, water_nearby)
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection
        
        state: discretized state tuple
        training: if False, use greedy only (no exploration)
        returns: action index (0-10)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best Q-value
            q_values = self.q_table[state]
            return np.argmax(q_values + np.random.random(self.num_actions) * 1e-6)
    
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
    
    def apply_death_penalty(self, state, penalty_reward):
        """Apply death penalty to all Q-values for a given state
        
        When an agent dies, penalize the state it was in so the agent learns to avoid
        similar situations in future episodes.
        
        Args:
            state: The state tuple where death occurred
            penalty_reward: Negative reward to apply (typically DEATH_PENALTY)
        """
        # Apply penalty to all actions in this state
        # This discourages the agent from taking actions that lead to death
        for action in range(self.num_actions):
            current_q = self.q_table[state][action]
            # Update toward the penalty with full learning rate for stronger effect
            self.q_table[state][action] = current_q + self.learning_rate * (penalty_reward - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def save_model(self, filepath):
        """Save Q-table and metadata to file (convert defaultdict to dict for pickling)"""
        import pickle
        from pathlib import Path
        
        # Create directory if needed
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict for pickling
            'num_actions': self.num_actions,
            'num_states': self.num_states
        }
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
