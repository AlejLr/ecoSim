"""
EcoSim Q-Learning Training Script

Simple training loop for single-agent Q-learning in a multi-agent environment.
Usage: 
    python train.py                         # Train PREY with random opponents
    python train.py PREDATOR                # Train PREDATOR with random opponents
    python train.py PREDATOR 100            # Train PREDATOR for 100 episodes
    python train.py PREDATOR 100 memory     # Train PREDATOR with pre-trained opponents
"""

from environment.gym_env import EcoSimEnv
from models.Q_learning import QLearningAgent, train_agent, evaluate_agent
import numpy as np
from pathlib import Path
import sys


def main():

    agent_type = "PREY"
    num_episodes = 500
    memory = False
    
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].upper()
        if agent_type not in ["PREY", "PREDATOR"]:
            print("Invalid agent type. Use 'PREY' or 'PREDATOR'")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            num_episodes = int(sys.argv[2])
        except ValueError:
            print("Invalid number of episodes")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        if sys.argv[3].lower() in ["--memory", "memory", "true", "1"]:
            memory = True
    
    # Create environment and agent
    print(f"Training {agent_type} agent for {num_episodes} episodes...")
    if memory:
        print("  (Using pre-trained models for other agents)")
    print("Initializing environment...")
    env = EcoSimEnv(agent_id=0, num_prey=8, num_predators=2, map_path=None, agent_type=agent_type, memory=memory)
    
    print("Initializing Q-Learning agent...")
    agent = QLearningAgent(agent_id=0, num_actions=11, num_states=5400)
    
    # Train agent
    print("\n" + "="*60)
    print(f"TRAINING PHASE - {agent_type}")
    print("="*60)
    episode_rewards, episode_steps = train_agent(env, agent, num_episodes=num_episodes)
    
    # Plot results (optional)
    try:
        import matplotlib.pyplot as plt
        
        # Smooth rewards for visualization using rolling average
        window = min(50, max(5, len(episode_rewards) // 2))
        # Use a proper rolling average to avoid edge effects
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        # Adjust x-axis for smoothed data (starts at window-1)
        x_smooth = np.arange(window - 1, len(episode_rewards))
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.3, label='Raw')
        plt.plot(x_smooth, smoothed_rewards, label=f'{window}-episode average', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(episode_steps)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Length')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = Path(__file__).resolve().parent / "training_results.png"
        plt.savefig(plot_path)
        print(f"\nTraining plots saved to '{plot_path}'")
    except ImportError:
        print("(Matplotlib not installed, skipping plots)")
    
    # Save model
    print("\nSaving trained model...")
    output_dir = Path(__file__).resolve().parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"trained_{agent_type.lower()}.pkl"
    agent.save_model(str(model_path))
    print(f"Model saved to '{model_path}'")
    
    # Evaluate agent
    print("\n" + "="*60)
    print(f"EVALUATION PHASE - {agent_type}")
    print("="*60)
    avg_reward, avg_steps = evaluate_agent(env, agent, num_episodes=20)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Agent type: {agent_type}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total states explored: {len(agent.q_table)}")
    print(f"Evaluation reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
