import os
import sys
import numpy as np

# Ensure project `src` is on sys.path so local imports work when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Q_learning import QLearningAgent


def main():
    models_dir = os.path.join(os.path.dirname(__file__))
    tmp_path = os.path.join(models_dir, 'temp_verify_model.pkl')

    # Create agent and populate q_table with deterministic values
    agent = QLearningAgent(agent_id=42)

    # Define a deterministic test state and q-values
    test_state = (0, 0, 0, 1, 1, 1)
    q_values = np.arange(agent.num_actions, dtype=float)
    # make action 0 the highest by reversing
    q_values = q_values[::-1]
    agent.q_table[test_state] = q_values.copy()

    # Save model
    agent.save_model(tmp_path)

    # Load model back
    loaded = QLearningAgent.load_model_from_file(tmp_path)

    # Verify q_table keys and values
    original = dict(agent.q_table)
    loaded_dict = dict(loaded.q_table)

    ok = True
    if set(original.keys()) != set(loaded_dict.keys()):
        print('FAIL: q_table keys differ')
        ok = False
    else:
        for k in original.keys():
            if not np.array_equal(original[k], loaded_dict[k]):
                print(f'FAIL: q-values differ for state {k}')
                ok = False
                break

    # Confirm evaluation (greedy) action selection: training=False should be deterministic
    # Even if epsilon is large on the loaded agent
    loaded.epsilon = 1.0
    actions = [loaded.select_action(test_state, training=False) for _ in range(20)]
    if len(set(actions)) != 1:
        print('FAIL: select_action(training=False) is not deterministic')
        ok = False
    else:
        greedy_action = actions[0]
        # verify greedy_action equals argmax of q-values
        expected = int(np.argmax(q_values))
        if greedy_action != expected:
            print(f'FAIL: greedy action {greedy_action} != expected {expected}')
            ok = False

    if ok:
        print('OK: save/load preserved Q-values and evaluation uses greedy actions')
    else:
        print('One or more checks failed')

    # Cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass


if __name__ == '__main__':
    main()
