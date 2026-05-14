"""Simple smoke test to verify Q-table updates occur immediately after actions.

Run: python -m src.models.test_q_update
"""
import random
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from environment.multi_agent_gym_env import MultiAgentEcoSimEnv


def run_smoke_test():
    env = MultiAgentEcoSimEnv(num_prey=2, num_predators=1)
    obs = env.reset()

    # Before stepping, q_tables should be empty
    q_tables_before = [len(a.q_learning.q_table) for a in env.all_agents]
    print("Q-table sizes before step:", q_tables_before)

    # Step the environment a few times
    for _ in range(3):
        obs, reward, done, info = env.step()
        if done:
            break

    q_tables_after = [len(a.q_learning.q_table) for a in env.all_agents]
    print("Q-table sizes after steps:", q_tables_after)

    # Assert that at least one agent learned (visited states recorded)
    if not any(size > 0 for size in q_tables_after):
        raise AssertionError("No Q-table entries found after stepping; updates may be broken.")

    print("Smoke test passed: Q-tables updated immediately after actions.")


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    run_smoke_test()
