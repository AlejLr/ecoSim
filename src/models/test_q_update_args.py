"""Test that QLearningAgent.update is called with correct arguments during a step.

This test monkeypatches agents' update methods to record parameters and asserts
that updates receive (state, action, reward, next_state, done) tuples and at
least one update occurs.

Run: python -m src.models.test_q_update_args
"""
import random
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from environment.multi_agent_gym_env import MultiAgentEcoSimEnv


def run_args_test():
    env = MultiAgentEcoSimEnv(num_prey=2, num_predators=1)
    obs = env.reset()

    records = []

    # Monkeypatch each agent's update to record calls
    for agent in env.all_agents:
        orig_update = agent.q_learning.update

        def make_recorder(a):
            def recorder(state, action, reward, next_state, done):
                records.append((a.agent_id, state, action, reward, next_state, done))
                # Call original to preserve behavior
                return orig_update(state, action, reward, next_state, done)
            return recorder

        agent.q_learning.update = make_recorder(agent)

    # Step environment a few times to trigger updates
    for _ in range(5):
        obs, reward, done, info = env.step()
        if done:
            break

    if not records:
        raise AssertionError("No update calls recorded; Q.update may not be invoked.")

    # Validate recorded tuple structure
    for rec in records:
        agent_id, state, action, reward, next_state, done = rec
        if not isinstance(state, tuple):
            raise AssertionError(f"State must be discretized tuple, got {type(state)}")
        if not isinstance(action, int):
            raise AssertionError(f"Action must be int, got {type(action)}")
        if not isinstance(reward, (int, float)):
            raise AssertionError(f"Reward must be numeric, got {type(reward)}")
        if not isinstance(next_state, tuple):
            raise AssertionError(f"Next state must be tuple, got {type(next_state)}")
        if not isinstance(done, bool):
            raise AssertionError(f"Done flag must be bool, got {type(done)}")

    print(f"Recorded {len(records)} Q.update calls across agents. Argument shapes OK.")
    print("Q-update argument ordering test passed.")


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    run_args_test()
