"""Quick test of multi-agent learning"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.multi_agent_gym_env import MultiAgentEcoSimEnv

print("Creating multi-agent environment...")
env = MultiAgentEcoSimEnv(num_prey=2, num_predators=1)

print("Resetting environment...")
obs = env.reset()
print(f"✓ Created {len(env.all_agents)} agents")

print("Running 3 steps...")
for step in range(3):
    obs, r, d, i = env.step()
    print(f"Step {step+1}: prey_alive={i['prey_count']}, predators_alive={i['predator_count']}, done={d}")

print("\n✓ Multi-agent environment works!")

# Print agent stats
print("\nAgent learning stats:")
for agent in env.all_agents:
    if agent.is_alive():
        q_size = len(agent.q_learning.q_table)
        print(f"  Agent {agent.agent_id} ({agent.agent_type}): {q_size} states visited, reward={agent.episode_reward:.1f}")
