"""Multi-seed experiment runner for robust MARL training results.

Because Multi-Agent Reinforcement Learning (MARL) is non-stationary, we run
multiple training sessions with different random seeds and average results.
This provides more robust and statistically significant findings.

Usage:
    python -m src.models.run_multi_seed_experiment PREY 5 100
    # Runs PREY training 5 times with different seeds, 100 episodes each
    
    python -m src.models.run_multi_seed_experiment PREDATOR 10 500
    # Runs PREDATOR training 10 times with different seeds, 500 episodes each
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import *
from config.utils import set_global_seed, get_next_run_number
from environment.gym_env import EcoSimEnv
from environment.logger import save_environment_log
from models.Q_learning import QLearningAgent


def train_with_seed(agent_type: str, num_episodes: int, seed: int, run_label: str) -> Tuple[List[float], List[int], float]:
    """Train agent with a specific seed.
    
    Args:
        agent_type: "PREY" or "PREDATOR"
        num_episodes: Number of training episodes
        seed: Random seed for this run
        run_label: Label for this run (e.g., "seed_1_of_5")
        
    Returns:
        (episode_rewards, episode_steps, final_eval_reward)
    """
    # Set seed for this run
    set_global_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training {agent_type} with seed={seed}")
    print(f"({run_label})")
    print(f"{'='*60}\n")
    
    # Create environment with memory disabled for baseline
    env = EcoSimEnv(
        agent_id=0,
        num_prey=6,
        num_predators=2,
        agent_type=agent_type,
        map_path=None,
        memory=False  # Use random agents for baseline
    )
    
    # Create Q-learning agent
    q_agent = QLearningAgent(agent_id=0, num_actions=10, num_states=540)
    
    episode_rewards = []
    episode_steps = []
    
    # Training phase
    for episode in range(num_episodes):
        obs = env.reset(seed=seed)
        state = q_agent.discretize_state(obs)
        episode_reward = 0
        episode_step_count = 0
        done = False
        
        while not done:
            # Select action
            action = q_agent.select_action(state, training=True)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            next_state = q_agent.discretize_state(next_obs)
            
            # Update Q-table
            q_agent.update(state, action, reward, next_state, done)
            
            state = next_state
            obs = next_obs
            episode_reward += reward
            episode_step_count += 1
        
        # Decay epsilon
        q_agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step_count)
        
        # Progress update
        if (episode + 1) % max(10, num_episodes // 10) == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {episode+1:4d}/{num_episodes} | Avg Reward (last 10): {avg_reward:8.2f}")
    
    # Evaluation phase (greedy policy)
    print(f"\n  Evaluation phase (greedy policy)...")
    original_epsilon = q_agent.epsilon
    q_agent.epsilon = 0.0
    
    eval_rewards = []
    for _ in range(10):
        obs = env.reset(seed=seed)
        state = q_agent.discretize_state(obs)
        episode_reward = 0
        done = False
        
        while not done:
            action = q_agent.select_action(state, training=False)
            next_obs, reward, done, info = env.step(action)
            next_state = q_agent.discretize_state(next_obs)
            state = next_state
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    q_agent.epsilon = original_epsilon
    final_eval_reward = np.mean(eval_rewards)
    
    print(f"  ✓ Final evaluation reward (greedy): {final_eval_reward:.2f}")
    
    return episode_rewards, episode_steps, final_eval_reward


def run_multi_seed_experiment(agent_type: str = "PREY", 
                              num_seeds: int = 5, 
                              num_episodes: int = 100,
                              seeds: List[int] = None):
    """Run training with multiple seeds and aggregate results.
    
    Args:
        agent_type: "PREY" or "PREDATOR"
        num_seeds: Number of different seeds to run
        num_episodes: Episodes per run
        seeds: Specific list of seeds, or None for auto-generated
    """
    
    if seeds is None:
        # Generate diverse seeds
        seeds = [42, 123, 456, 789, 1000][:num_seeds]
        if num_seeds > 5:
            # Add more if needed
            seeds.extend([i * 1337 for i in range(1, num_seeds - 4)])
    
    print(f"\n{'='*70}")
    print(f"MULTI-SEED EXPERIMENT: {agent_type} Agent")
    print(f"{'='*70}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Episodes per run: {num_episodes}")
    print(f"Seeds: {seeds}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Get experiment identifier
    experiment_run = get_next_run_number()
    
    # Save environment documentation
    save_environment_log(experiment_run)
    
    # Store all results
    all_rewards = []  # List of (seed, episode_rewards) tuples
    all_steps = []
    final_evals = []
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Run training for each seed
    for i, seed in enumerate(seeds):
        run_label = f"seed {seed} ({i+1}/{num_seeds})"
        
        try:
            episode_rewards, episode_steps, final_eval = train_with_seed(
                agent_type=agent_type,
                num_episodes=num_episodes,
                seed=seed,
                run_label=run_label
            )
            
            all_rewards.append((seed, episode_rewards))
            all_steps.append((seed, episode_steps))
            final_evals.append((seed, final_eval))
            
            # Save individual run results
            csv_path = results_dir / f"multiseed_individual_{agent_type.lower()}_{experiment_run}_seed_{seed}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'reward', 'steps', 'seed'])
                for ep, (reward, steps) in enumerate(zip(episode_rewards, episode_steps)):
                    writer.writerow([ep + 1, reward, steps, seed])
            
            print(f"  ✓ Saved: {csv_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error training with seed {seed}: {e}")
            continue
    
    # Compute aggregate statistics
    print(f"\n{'='*70}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*70}\n")
    
    # Ensure all runs have same length for averaging
    if not all_rewards:
        print("✗ No successful runs!")
        return
    
    min_len = min(len(rewards) for _, rewards in all_rewards)
    
    # Trim all to same length
    all_rewards_trimmed = [(seed, rewards[:min_len]) for seed, rewards in all_rewards]
    
    # Compute statistics per episode
    episode_means = []
    episode_stds = []
    episode_mins = []
    episode_maxs = []
    
    for ep in range(min_len):
        rewards_at_ep = [rewards[ep] for _, rewards in all_rewards_trimmed]
        episode_means.append(np.mean(rewards_at_ep))
        episode_stds.append(np.std(rewards_at_ep))
        episode_mins.append(np.min(rewards_at_ep))
        episode_maxs.append(np.max(rewards_at_ep))
    
    # Compute final metrics
    final_eval_means = [eval_reward for _, eval_reward in final_evals]
    
    print(f"Completed {len(final_evals)} runs out of {num_seeds}")
    print(f"\nFinal Evaluation Rewards (Greedy Policy):")
    print(f"  Mean:   {np.mean(final_eval_means):8.2f}")
    print(f"  Std:    {np.std(final_eval_means):8.2f}")
    print(f"  Min:    {np.min(final_eval_means):8.2f}")
    print(f"  Max:    {np.max(final_eval_means):8.2f}")
    
    print(f"\nTraining Rewards (Last 10 Episodes Average):")
    last_10_means = [np.mean(rewards[-10:]) for _, rewards in all_rewards_trimmed]
    print(f"  Mean:   {np.mean(last_10_means):8.2f}")
    print(f"  Std:    {np.std(last_10_means):8.2f}")
    print(f"  Min:    {np.min(last_10_means):8.2f}")
    print(f"  Max:    {np.max(last_10_means):8.2f}")
    
    # Save aggregate results
    summary_path = results_dir / f"multiseed_summary_{agent_type.lower()}_{experiment_run}.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'mean_reward', 'std_reward', 'min_reward', 'max_reward'])
        for ep, (mean, std, min_r, max_r) in enumerate(zip(episode_means, episode_stds, episode_mins, episode_maxs)):
            writer.writerow([ep + 1, mean, std, min_r, max_r])
    
    print(f"\n✓ Aggregate results saved: {summary_path.name}")
    
    # Save individual seed results
    seeds_path = results_dir / f"multiseed_seeds_{agent_type.lower()}_{experiment_run}.csv"
    with open(seeds_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'final_evaluation_reward'])
        for seed, eval_reward in final_evals:
            writer.writerow([seed, eval_reward])
    
    print(f"✓ Seed results saved: {seeds_path.name}")
    
    # Generate plots
    print(f"\nGenerating comparison plots...")
    generate_comparison_plots(
        all_rewards_trimmed,
        episode_means,
        episode_stds,
        agent_type,
        experiment_run,
        results_dir
    )
    
    print(f"\n{'='*70}")
    print(f"Multi-seed experiment complete!")
    print(f"Experiment ID: Run #{experiment_run}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


def generate_comparison_plots(all_rewards_trimmed: List[Tuple[int, List[float]]],
                             episode_means: List[float],
                             episode_stds: List[float],
                             agent_type: str,
                             experiment_run: int,
                             results_dir: Path):
    """Generate comparison plots across seeds.
    
    Args:
        all_rewards_trimmed: List of (seed, rewards) tuples
        episode_means: Mean reward per episode
        episode_stds: Std dev per episode
        agent_type: "PREY" or "PREDATOR"
        experiment_run: Run identifier
        results_dir: Directory to save plots
    """
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Individual runs
        ax = axes[0, 0]
        for seed, rewards in all_rewards_trimmed:
            ax.plot(rewards, alpha=0.5, linewidth=0.8, label=f"Seed {seed}")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Individual Training Runs (by Seed)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Plot 2: Mean with std envelope
        ax = axes[0, 1]
        episodes = range(1, len(episode_means) + 1)
        ax.plot(episodes, episode_means, 'b-', linewidth=2, label='Mean')
        ax.fill_between(episodes, 
                        np.array(episode_means) - np.array(episode_stds),
                        np.array(episode_means) + np.array(episode_stds),
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Mean ± Std Dev Across Seeds')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: Moving average comparison
        ax = axes[1, 0]
        window = min(20, max(5, len(episode_means) // 5))
        for seed, rewards in all_rewards_trimmed:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(moving_avg, alpha=0.6, linewidth=1, label=f"Seed {seed}")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (Moving Avg)')
        ax.set_title(f'{window}-Episode Moving Average Comparison')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Plot 4: Convergence box plot
        ax = axes[1, 1]
        # Get last 25% of episodes for each run
        final_quarter_rewards = []
        for seed, rewards in all_rewards_trimmed:
            start_idx = int(len(rewards) * 0.75)
            final_quarter_rewards.append(rewards[start_idx:])
        
        labels = [f"Seed {seed}" for seed, _ in all_rewards_trimmed]
        ax.boxplot(final_quarter_rewards, labels=labels)
        ax.set_ylabel('Reward')
        ax.set_title('Final Quarter Distribution (Convergence)')
        ax.grid(alpha=0.3, axis='y')
        
        plt.suptitle(f'{agent_type} Agent - Multi-Seed Experiment Results (Run #{experiment_run})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = results_dir / f"multiseed_comparison_{agent_type.lower()}_{experiment_run}.png"
        plt.savefig(str(plot_path), dpi=150)
        print(f"  ✓ Saved: {plot_path.name}")
        plt.close()
        
    except Exception as e:
        print(f"  ⚠ Could not generate plots: {e}")


def main():
    """Main multi-seed experiment runner."""
    
    agent_type = "PREY"
    num_seeds = 5
    num_episodes = 100
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].upper()
        if agent_type not in ["PREY", "PREDATOR"]:
            print("Invalid agent type. Use 'PREY' or 'PREDATOR'")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            num_seeds = int(sys.argv[2])
        except ValueError:
            print("Invalid number of seeds")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        try:
            num_episodes = int(sys.argv[3])
        except ValueError:
            print("Invalid number of episodes")
            sys.exit(1)
    
    # Run experiment
    run_multi_seed_experiment(
        agent_type=agent_type,
        num_seeds=num_seeds,
        num_episodes=num_episodes
    )


if __name__ == "__main__":
    main()
