"""
EcoSim Q-Learning Training Script

Simple training loop for single-agent Q-learning in a multi-agent environment.
"""

from environment.gym_env import EcoSimEnv
from models.Q_learning import QLearningAgent, train_agent, evaluate_agent
import numpy as np
from pathlib import Path


def main():
    # Create environment and agent
    print("Initializing environment...")
    env = EcoSimEnv(agent_id=0, num_prey=4, num_predators=2, map_path=None)
    
    print("Initializing Q-Learning agent...")
    agent = QLearningAgent(agent_id=0, num_actions=11, num_states=675)
    
    # Train agent
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    episode_rewards, episode_steps = train_agent(env, agent, num_episodes=500)
    
    # Plot results (optional)
    try:
        import matplotlib.pyplot as plt
        
        # Smooth rewards for visualization
        window = 50
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.3, label='Raw')
        plt.plot(range(window-1, len(episode_rewards)), smoothed_rewards, label=f'{window}-episode average')
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
    model_path = output_dir / "trained_agent.pkl"
    agent.save_model(str(model_path))
    print(f"Model saved to '{model_path}'")
    
    # Evaluate agent
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    avg_reward, avg_steps = evaluate_agent(env, agent, num_episodes=20)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total states explored: {len(agent.q_table)}")
    print(f"Evaluation reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
