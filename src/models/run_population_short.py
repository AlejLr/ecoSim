"""Run a short multi-agent population simulation and report prey/predator counts.

This script runs the `MultiAgentEcoSimEnv` for a few episodes and prints
per-episode average population sizes and a simple CSV of per-step counts.

Run: python -m src.models.run_population_short
"""
import csv
import random
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config.utils import set_global_seed
from environment.multi_agent_gym_env import MultiAgentEcoSimEnv


def run_short(episodes=5, max_steps=100, num_prey=4, num_predators=2):
    env = MultiAgentEcoSimEnv(num_prey=num_prey, num_predators=num_predators)

    all_episode_stats = []

    for ep in range(episodes):
        obs = env.reset()
        step = 0
        episode_counts = []

        while step < max_steps:
            obs, reward, done, info = env.step()
            # Info includes counts when alive agents exist
            prey_count = info.get('prey_count', None)
            predator_count = info.get('predator_count', None)
            if prey_count is None or predator_count is None:
                # Fallback: count from env
                alive = [a for a in env.all_agents if a.is_alive()]
                prey_count = len([a for a in alive if a.agent_type == 'PREY'])
                predator_count = len([a for a in alive if a.agent_type == 'PREDATOR'])

            episode_counts.append((step, prey_count, predator_count))

            step += 1
            if done:
                break

        # Aggregate
        avg_prey = np.mean([c[1] for c in episode_counts]) if episode_counts else 0
        avg_pred = np.mean([c[2] for c in episode_counts]) if episode_counts else 0
        all_episode_stats.append({'episode': ep, 'avg_prey': avg_prey, 'avg_predator': avg_pred, 'steps': len(episode_counts)})

        # Save per-episode CSV
        out_dir = Path(__file__).parent / 'results'
        out_dir.mkdir(exist_ok=True)
        csv_path = out_dir / f'pop_episode_{ep}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step','prey_count','predator_count'])
            writer.writerows(episode_counts)

        print(f"Episode {ep}: steps={len(episode_counts)} avg_prey={avg_prey:.2f} avg_predator={avg_pred:.2f} -> saved {csv_path}")

    # Summary
    print("\nSummary across episodes:")
    for s in all_episode_stats:
        print(f"  Ep {s['episode']}: steps={s['steps']} avg_prey={s['avg_prey']:.2f} avg_predator={s['avg_predator']:.2f}")


if __name__ == '__main__':
    # Set global seed for reproducibility
    set_global_seed()
    run_short(episodes=5, max_steps=100, num_prey=4, num_predators=2)
