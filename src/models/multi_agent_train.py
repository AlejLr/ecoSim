"""Multi-agent training script - all agents learn via Q-learning"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import *
from environment.multi_agent_gym_env import MultiAgentEcoSimEnv


def train_multi_agent(num_episodes=100, num_prey=3, num_predators=2):
    """
    Train all agents (prey and predators) via Q-learning
    
    Args:
        num_episodes: Number of training episodes
        num_prey: Number of prey agents
        num_predators: Number of predator agents
    
    Returns:
        env: The trained environment with all agents' Q-tables
        episode_data: Dict with learning metrics per episode
    """
    
    print(f"\n{'='*70}")
    print(f"MULTI-AGENT Q-LEARNING TRAINING")
    print(f"{'='*70}")
    print(f"Prey agents: {num_prey} | Predator agents: {num_predators}")
    print(f"Episodes: {num_episodes}")
    print(f"Hyperparameters: LR={LEARNING_RATE}, DF={DISCOUNT_FACTOR}")
    print(f"{'='*70}\n")
    
    # Create multi-agent environment
    env = MultiAgentEcoSimEnv(num_prey=num_prey, num_predators=num_predators)
    
    # Track metrics
    episode_data = {
        "prey_rewards": defaultdict(list),  # agent_id -> [rewards per episode]
        "predator_rewards": defaultdict(list),
        "prey_survival": [],  # How many prey survive each episode
        "predator_survival": [],
        "num_species_interactions": [],
        "avg_episode_reward": [],
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        
        while not done:
            obs, reward, done, info = env.step(actions=None)
            step_count += 1
        
        # Collect episode stats
        prey_count = info["prey_count"]
        predator_count = info["predator_count"]
        
        episode_data["prey_survival"].append(prey_count)
        episode_data["predator_survival"].append(predator_count)
        
        # Track individual rewards
        for agent_id, reward in info["prey_rewards"].items():
            episode_data["prey_rewards"][agent_id].append(reward)
        
        for agent_id, reward in info["predator_rewards"].items():
            episode_data["predator_rewards"][agent_id].append(reward)
        
        # Overall episode reward
        if info["prey_rewards"] or info["predator_rewards"]:
            all_rewards = list(info["prey_rewards"].values()) + list(info["predator_rewards"].values())
            avg_reward = np.mean(all_rewards)
        else:
            avg_reward = 0
        
        episode_data["avg_episode_reward"].append(avg_reward)
        episode_data["num_species_interactions"].append(
            min(prey_count, predator_count)  # Potential interactions
        )
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward_recent = np.mean(episode_data["avg_episode_reward"][-10:])
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"Prey alive: {prey_count} | Predators alive: {predator_count} | "
                  f"Avg Reward (last 10): {avg_reward_recent:8.2f}")
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final prey survival: {episode_data['prey_survival'][-1]}")
    print(f"Final predator survival: {episode_data['predator_survival'][-1]}")
    print(f"Final avg reward: {episode_data['avg_episode_reward'][-1]:.2f}")
    print(f"{'='*70}\n")
    
    return env, episode_data


def evaluate_multi_agent(env, num_episodes=10):
    """
    Evaluate trained agents using greedy policy (no exploration)
    
    Args:
        env: Trained MultiAgentEcoSimEnv
        num_episodes: Number of evaluation episodes
    
    Returns:
        eval_data: Evaluation metrics
    """
    
    print(f"\n{'='*70}")
    print(f"MULTI-AGENT EVALUATION (Greedy Policy - No Exploration)")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    # Set all agents to greedy (epsilon=0)
    for agent in env.all_agents:
        agent.q_learning.epsilon = 0.0
    
    eval_data = {
        "prey_survival": [],
        "predator_survival": [],
        "avg_prey_reward": [],
        "avg_predator_reward": [],
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        # Set epsilon to 0 after reset
        for agent in env.all_agents:
            agent.q_learning.epsilon = 0.0
        
        while not done:
            obs, reward, done, info = env.step(actions=None)
        
        prey_count = info["prey_count"]
        predator_count = info["predator_count"]
        
        eval_data["prey_survival"].append(prey_count)
        eval_data["predator_survival"].append(predator_count)
        
        if info["prey_rewards"]:
            eval_data["avg_prey_reward"].append(np.mean(list(info["prey_rewards"].values())))
        
        if info["predator_rewards"]:
            eval_data["avg_predator_reward"].append(np.mean(list(info["predator_rewards"].values())))
        
        print(f"Eval Episode {episode+1:2d} | "
              f"Prey: {prey_count} | Predators: {predator_count} | "
              f"Prey Avg Reward: {eval_data['avg_prey_reward'][-1] if eval_data['avg_prey_reward'] else 0:7.2f}")
    
    print(f"\n{'='*70}")
    print(f"Evaluation Results:")
    print(f"  Avg Prey Survival: {np.mean(eval_data['prey_survival']):.1f}")
    print(f"  Avg Predator Survival: {np.mean(eval_data['predator_survival']):.1f}")
    if eval_data["avg_prey_reward"]:
        print(f"  Avg Prey Reward: {np.mean(eval_data['avg_prey_reward']):.2f}")
    if eval_data["avg_predator_reward"]:
        print(f"  Avg Predator Reward: {np.mean(eval_data['avg_predator_reward']):.2f}")
    print(f"{'='*70}\n")
    
    return eval_data


def plot_multi_agent_learning(episode_data, title="Multi-Agent Learning Progress"):
    """Plot multi-agent learning metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Species survival over time
    ax = axes[0, 0]
    ax.plot(episode_data["prey_survival"], label="Prey", linewidth=2, marker='o', markersize=3)
    ax.plot(episode_data["predator_survival"], label="Predators", linewidth=2, marker='s', markersize=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_title("Species Population Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Average episode reward
    ax = axes[0, 1]
    ax.plot(episode_data["avg_episode_reward"], linewidth=1, alpha=0.7)
    window = 10
    moving_avg = np.convolve(episode_data["avg_episode_reward"], np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(episode_data["avg_episode_reward"])), moving_avg, 'r-', linewidth=2, label=f'MA-{window}')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Potential predator-prey interactions
    ax = axes[1, 0]
    ax.plot(episode_data["num_species_interactions"], linewidth=1.5, color='purple')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Interactions (min of both)")
    ax.set_title("Predator-Prey Interaction Potential")
    ax.grid(alpha=0.3)
    
    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    MULTI-AGENT LEARNING STATISTICS
    
    Training Episodes: {len(episode_data['prey_survival'])}
    
    PREY:
    - Final Population: {episode_data['prey_survival'][-1]}
    - Avg Population: {np.mean(episode_data['prey_survival']):.1f}
    - Population Std: {np.std(episode_data['prey_survival']):.1f}
    
    PREDATORS:
    - Final Population: {episode_data['predator_survival'][-1]}
    - Avg Population: {np.mean(episode_data['predator_survival']):.1f}
    - Population Std: {np.std(episode_data['predator_survival']):.1f}
    
    LEARNING:
    - Avg Episode Reward: {np.mean(episode_data['avg_episode_reward']):.2f}
    - Final Avg Reward: {episode_data['avg_episode_reward'][-1]:.2f}
    - Best Episode Reward: {max(episode_data['avg_episode_reward']):.2f}
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(__file__).parent.parent.parent / 'multi_agent_learning.png'
    plt.savefig(str(save_path), dpi=100)
    print(f"Saved learning plot to: {save_path}")
    plt.show()


def save_trained_agents(env, filepath):
    """Save all trained agents' Q-tables"""
    
    agents_data = {}
    for agent in env.all_agents:
        agents_data[agent.agent_id] = {
            "type": agent.agent_type,
            "q_table": dict(agent.q_learning.q_table),
            "epsilon": agent.q_learning.epsilon,
            "episode_reward": agent.episode_reward
        }
    
    with open(filepath, 'wb') as f:
        pickle.dump(agents_data, f)
    
    print(f"Saved {len(agents_data)} trained agents to: {filepath}")


def main():
    """Main multi-agent training script"""
    
    print("\n" + "="*70)
    print("MULTI-AGENT ECOSYSTEM SIMULATION WITH Q-LEARNING")
    print("="*70)
    
    # Train multi-agent system
    env, episode_data = train_multi_agent(num_episodes=50, num_prey=3, num_predators=2)
    
    # Evaluate trained agents
    eval_data = evaluate_multi_agent(env, num_episodes=10)
    
    # Save trained agents
    model_path = Path(__file__).parent / "multi_agent_model.pkl"
    save_trained_agents(env, str(model_path))
    
    # Plot results
    plot_multi_agent_learning(episode_data, title="Multi-Agent Ecosystem Learning")
    
    print("\n" + "="*70)
    print("✓ Multi-agent training completed successfully!")
    print("='="*70 + "\n")


if __name__ == "__main__":
    main()
