"""Simple multi-agent training test without plotting"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import *
from environment.multi_agent_gym_env import MultiAgentEcoSimEnv

print("\n" + "="*70)
print("MULTI-AGENT Q-LEARNING TEST - 5 Episodes")
print("="*70 + "\n")

env = MultiAgentEcoSimEnv(num_prey=3, num_predators=2)

all_rewards = []
all_survivors = []

for episode in range(5):
    obs = env.reset()
    done = False
    
    while not done:
        obs, reward, done, info = env.step()
    
    # Get stats
    prey_alive = info["prey_count"]
    pred_alive = info["predator_count"]
    all_survivors.append((prey_alive, pred_alive))
    
    # Get average reward
    all_rewards_this_ep = list(info["prey_rewards"].values()) + list(info["predator_rewards"].values())
    avg_reward = np.mean(all_rewards_this_ep) if all_rewards_this_ep else 0
    all_rewards.append(avg_reward)
    
    print(f"Episode {episode+1}/5 | Prey: {prey_alive} | Predators: {pred_alive} | Avg Reward: {avg_reward:7.2f}")

print("\n" + "="*70)
print("Results Summary:")
print(f"  Avg Prey Survival: {np.mean([s[0] for s in all_survivors]):.1f}")
print(f"  Avg Predator Survival: {np.mean([s[1] for s in all_survivors]):.1f}")
print(f"  Avg Episode Reward: {np.mean(all_rewards):.2f}")
print("="*70 + "\n")

# Print learned Q-tables
print("Learned Q-tables:")
for agent in env.all_agents:
    q_states = len(agent.q_learning.q_table)
    print(f"  Agent {agent.agent_id} ({agent.agent_type}): {q_states} states visited")

print("\n✓ Multi-agent training test complete!\n")
