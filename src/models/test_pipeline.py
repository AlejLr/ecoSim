"""Quick test of full training pipeline"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import *
from environment.gym_env import EcoSimEnv
from models.Q_learning import QLearningAgent


print("="*60)
print("QUICK PIPELINE TEST - 5 Training Episodes")
print("="*60 + "\n")

# Create environment
env = EcoSimEnv(agent_id=0, num_prey=2, num_predators=1, agent_type="PREY")

# Create Q-learning agent
q_agent = QLearningAgent(agent_id=0, num_actions=11, num_states=675)

# Train for 5 episodes
for episode in range(5):
    obs = env.reset()
    state = q_agent.discretize_state(obs)
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:  # Limit to 100 steps for quick test
        # Select action
        action = q_agent.select_action(state, training=True)
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action)
        next_state = q_agent.discretize_state(next_obs)
        
        # Update Q-table
        q_agent.update(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        steps += 1
    
    q_agent.decay_epsilon()
    print(f"Episode {episode+1}/5 | Reward: {episode_reward:7.2f} | Steps: {steps:3d}")

print("\n" + "="*60)
print("✓ Pipeline test successful!")
print("="*60 + "\n")

# Test saving
model_path = Path(__file__).parent / "test_prey_agent.pkl"
q_agent.save_model(str(model_path))
print(f"✓ Model saved to {model_path}")

# Test evaluation
print("\nEvaluating with greedy policy...")
q_agent.epsilon = 0.0

eval_rewards = []
for episode in range(3):
    obs = env.reset()
    state = q_agent.discretize_state(obs)
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = q_agent.select_action(state, training=False)
        next_obs, reward, done, info = env.step(action)
        next_state = q_agent.discretize_state(next_obs)
        state = next_state
        episode_reward += reward
        steps += 1
    
    eval_rewards.append(episode_reward)
    print(f"Eval Episode {episode+1}/3 | Reward: {episode_reward:7.2f} | Steps: {steps:3d}")

print(f"\nAvg Eval Reward: {np.mean(eval_rewards):.2f}")
print("\n" + "="*60)
print("✓ All tests passed! System ready for full training.")
print("="*60 + "\n")
