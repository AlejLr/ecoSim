import numpy as np
from collections import defaultdict
import random

from config.config import *


class QLearningAgent:
    """Tabular Q-Learning agent with discretized state space
    
    State discretization:
    - energy bucket: 5 levels [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    - thirst bucket: 5 levels [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    - food_nearby bucket: 3 levels [low, medium, high]
    - water_nearby bucket: 3 levels [low, medium, high]
    - other_agents bucket: 3 levels [none, some, many]
    
    Total state space: 5 × 5 × 3 × 3 × 3 = 675 states
    Action space: 11 actions
    """
    
    def __init__(self, agent_id=0, num_actions=11, num_states=675):
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
        
        obs: numpy array [energy, thirst, food_nearby, water_nearby, other_agents_nearby]
        returns: state tuple for Q-table indexing
        """
        # Ensure obs is normalized [0, 1]
        energy = int(np.clip(obs[0] * 5, 0, 4))
        thirst = int(np.clip(obs[1] * 5, 0, 4))
        food = int(np.clip(obs[2] * 3, 0, 2))
        water = int(np.clip(obs[3] * 3, 0, 2))
        agents = int(np.clip(obs[4] * 3, 0, 2))
        
        return (energy, thirst, food, water, agents)
    
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
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def save_model(self, filepath):
        """Save Q-table to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load Q-table from file"""
        import pickle
        with open(filepath, 'rb') as f:
            q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.num_actions), q_dict)
        print(f"Model loaded from {filepath}")


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
