"""Training script for Q-learning agent on simplified EcoSim environment"""

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import *
from config.utils import set_global_seed, get_next_run_number
from environment.gym_env import EcoSimEnv
from environment.logger import save_environment_log
from models.Q_learning import QLearningAgent


def _mean_or_zero(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _collect_episode_metrics(env, q_agent):
    all_agents = [env.agent] + list(env.other_agents)
    alive_agents = [agent for agent in all_agents if agent.is_alive()]
    alive_prey = [agent for agent in alive_agents if agent.agent_type == "PREY"]
    alive_predators = [agent for agent in alive_agents if agent.agent_type == "PREDATOR"]
    prey_agents = [agent for agent in all_agents if agent.agent_type == "PREY"]
    predator_agents = [agent for agent in all_agents if agent.agent_type == "PREDATOR"]

    return {
        "prey_population_end": len(alive_prey),
        "predator_population_end": len(alive_predators),
        "average_prey_energy": _mean_or_zero(agent.energy for agent in alive_prey),
        "average_predator_energy": _mean_or_zero(agent.energy for agent in alive_predators),
        "mean_prey_reward": _mean_or_zero(agent.episode_reward for agent in prey_agents),
        "mean_predator_reward": _mean_or_zero(agent.episode_reward for agent in predator_agents),
        "epsilon": q_agent.epsilon,
    }


def train_agent(num_episodes=100, agent_type="PREY", num_prey=4, num_predators=2, run_number=None):
    """
    Train Q-learning agent on EcoSim environment
    
    Args:
        num_episodes: Number of training episodes
        agent_type: "PREY" or "PREDATOR"
        num_prey: Number of other prey agents
        num_predators: Number of predator agents
        run_number: Run identifier for logging
    
    Returns:
        q_agent: Trained QLearningAgent
        episode_rewards: List of rewards per episode
    """
    
    print(f"\n{'='*60}")
    print(f"Training {agent_type} Q-Learning Agent (Run #{run_number})")
    print(f"Episodes: {num_episodes}")
    print(f"Hyperparameters: LR={LEARNING_RATE}, DF={DISCOUNT_FACTOR}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = EcoSimEnv(
        agent_id=0,
        num_prey=num_prey,
        num_predators=num_predators,
        agent_type=agent_type,
        map_path=None
    )
    
    # Create Q-learning agent
    q_agent = QLearningAgent(agent_id=0, num_actions=11, num_states=5400)
    
    episode_rewards = []
    episode_steps = []

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"training_log_{agent_type.lower()}_{run_number}.csv"
    if log_path.exists():
        log_path.unlink()
    with open(log_path, "w", newline="") as log_file:
        logger = csv.DictWriter(log_file, fieldnames=[
            "episode",
            "mean_prey_reward",
            "mean_predator_reward",
            "prey_population_end",
            "predator_population_end",
            "average_prey_energy",
            "average_predator_energy",
            "epsilon",
            "episode_reward",
            "episode_steps",
        ])
        logger.writeheader()

        for episode in range(num_episodes):
            obs = env.reset()
            state = q_agent.discretize_state(obs)
            episode_reward = 0
            episode_step_count = 0
            done = False

            while not done:
                # Select action
                action = q_agent.select_action(state, training=True)

                # Take step in environment
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

            metrics = _collect_episode_metrics(env, q_agent)
            logger.writerow({
                "episode": episode + 1,
                "mean_prey_reward": metrics["mean_prey_reward"],
                "mean_predator_reward": metrics["mean_predator_reward"],
                "prey_population_end": metrics["prey_population_end"],
                "predator_population_end": metrics["predator_population_end"],
                "average_prey_energy": metrics["average_prey_energy"],
                "average_predator_energy": metrics["average_predator_energy"],
                "epsilon": metrics["epsilon"],
                "episode_reward": episode_reward,
                "episode_steps": episode_step_count,
            })
            log_file.flush()

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_steps = np.mean(episode_steps[-10:])
                print(f"Episode {episode+1:3d}/{num_episodes} | "
                      f"Avg Reward (last 10): {avg_reward:8.2f} | "
                      f"Avg Steps: {avg_steps:6.1f} | "
                      f"Epsilon: {q_agent.epsilon:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final avg reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Training log saved to: {log_path}")
    print(f"{'='*60}\n")
    
    # Generate training curves (optional - suppress errors if plotting fails)
    try:
        from models.plotting import plot_training_curves
        plot_training_curves(
            str(log_path),
            agent_type=agent_type,
            output_filename=f"training_curves_{agent_type.lower()}.png"
        )
    except Exception as e:
        print(f"Note: Could not generate training curves ({e})")
    
    return q_agent, episode_rewards, episode_steps


def evaluate_agent(q_agent, num_episodes=10, agent_type="PREY", num_prey=4, num_predators=2):
    """
    Evaluate trained Q-learning agent using greedy policy (no exploration)
    
    Args:
        q_agent: Trained QLearningAgent
        num_episodes: Number of evaluation episodes
        agent_type: "PREY" or "PREDATOR"
        num_prey: Number of other prey agents
        num_predators: Number of predator agents
    
    Returns:
        eval_rewards: List of rewards per episode
    """
    
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_type} Q-Learning Agent (Greedy Policy)")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = EcoSimEnv(
        agent_id=0,
        num_prey=num_prey,
        num_predators=num_predators,
        agent_type=agent_type,
        map_path=None
    )
    
    # Save epsilon and set to 0 (greedy)
    original_epsilon = q_agent.epsilon
    q_agent.epsilon = 0.0
    
    eval_rewards = []
    eval_steps = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        state = q_agent.discretize_state(obs)
        episode_reward = 0
        episode_step_count = 0
        done = False
        
        while not done:
            # Select greedy action (no exploration)
            action = q_agent.select_action(state, training=False)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            next_state = q_agent.discretize_state(next_obs)
            
            state = next_state
            episode_reward += reward
            episode_step_count += 1
        
        eval_rewards.append(episode_reward)
        eval_steps.append(episode_step_count)
    
    # Restore epsilon
    q_agent.epsilon = original_epsilon
    
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    avg_steps = np.mean(eval_steps)
    
    print(f"Evaluation Results:")
    print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Min/Max Reward: {min(eval_rewards):.2f} / {max(eval_rewards):.2f}")
    print(f"\n{'='*60}\n")
    
    return eval_rewards, eval_steps


def plot_training(episode_rewards, episode_steps, title="Q-Learning Training Progress"):
    """Plot training progress"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6, linewidth=0.8, label='Episode Reward')
    
    # Add moving average
    window = 10
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'MA-{window}')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot steps
    ax2.plot(episode_steps, alpha=0.6, linewidth=0.8, label='Episode Steps')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent.parent.parent / 'training_progress.png'), dpi=100)
    print("Saved training progress plot to: training_progress.png")
    plt.show()


def main():
    """Main training script"""
    # Set global seed for reproducibility
    set_global_seed()
    
    # Get run number for this execution
    run_number = get_next_run_number()
    print(f"\n{'='*60}")
    print(f"ECOSIM TRAINING RUN #{run_number}")
    print(f"{'='*60}")
    
    # Save environment documentation
    save_environment_log(run_number)
    
    # Ensure results directory exists
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)    
    
    # Train prey agent
    print("\n" + "="*60)
    print("TRAINING PREY AGENT")
    print("="*60)
    
    agent_type = "PREY"
    prey_agent, prey_rewards, prey_steps = train_agent(
        num_episodes=100,
        agent_type=agent_type,
        num_prey=6,
        num_predators=2,
        run_number=run_number
    )
    
    # Evaluate prey agent
    prey_eval_rewards, prey_eval_steps = evaluate_agent(
        prey_agent,
        num_episodes=10,
        agent_type=agent_type,
        num_prey=6,
        num_predators=2
    )
    
    # Save trained agent using save_model method (properly handles pickling)
    # Save to src/models/ directory with run number
    model_dir = Path(__file__).parent.parent / "models"  # Go up to src, then into models
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"trained_{agent_type.lower()}_{run_number}.pkl"
    prey_agent.save_model(str(model_path))
    print(f"✓ Model saved: {model_path}")
    
    # Plot results
    plot_training(prey_rewards, prey_steps, title=f"Prey Agent Training Progress (Run #{run_number})")
    
    print("\n" + "="*60)
    print(f"Training run #{run_number} completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
