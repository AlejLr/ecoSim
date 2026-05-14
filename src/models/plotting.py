"""
Plotting utilities for thesis visualization.
Generates population dynamics and training curves from simulation logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_population_dynamics(results_dir="src/models/results", output_filename="population_dynamics.png"):
    """
    Generate population dynamics plots from saved agent episodes.
    
    Reads all saved_agents_episode_*.csv files and creates:
    1. Prey population over time
    2. Predator population over time
    3. Combined population plot with dual axes
    4. Optional: Moving average overlays
    
    Args:
        results_dir (str): Directory containing saved_agents_episode_*.csv files
        output_filename (str): Name of output PNG file
    """
    results_path = Path(results_dir)
    episode_files = sorted(results_path.glob("saved_agents_episode_*.csv"))
    
    if not episode_files:
        print(f"No episode files found in {results_dir}")
        return
    
    print(f"Found {len(episode_files)} episode files. Generating population plots...")
    
    # Load all episodes
    episodes_data = []
    for episode_file in episode_files:
        df = pd.read_csv(episode_file)
        episode_num = int(episode_file.stem.split("_")[-1])
        df["episode"] = episode_num
        episodes_data.append(df)
    
    # Combine all episodes
    combined_df = pd.concat(episodes_data, ignore_index=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Population Dynamics Over Time", fontsize=14, fontweight="bold")
    
    # Color scheme
    prey_color = "#2ecc71"  # Green
    predator_color = "#e74c3c"  # Red
    
    # --- Plot 1: Prey Population ---
    ax = axes[0]
    prey_by_episode = combined_df.groupby("episode")["prey_count"].apply(list).to_dict()
    
    for episode_num in sorted(prey_by_episode.keys()):
        prey_counts = prey_by_episode[episode_num]
        steps = np.arange(len(prey_counts))
        ax.plot(steps, prey_counts, label=f"Episode {episode_num}", alpha=0.7, linewidth=1.5)
    
    # Add moving average across all episodes
    ax.plot(combined_df["step"], combined_df["prey_count"].rolling(window=10, center=True).mean(),
            color=prey_color, linewidth=2.5, label="10-step moving avg", linestyle="--", alpha=0.8)
    
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Prey Population", fontsize=11, color=prey_color)
    ax.tick_params(axis="y", labelcolor=prey_color)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Prey Population", fontweight="bold")
    
    # --- Plot 2: Predator Population ---
    ax = axes[1]
    predator_by_episode = combined_df.groupby("episode")["predator_count"].apply(list).to_dict()
    
    for episode_num in sorted(predator_by_episode.keys()):
        predator_counts = predator_by_episode[episode_num]
        steps = np.arange(len(predator_counts))
        ax.plot(steps, predator_counts, label=f"Episode {episode_num}", alpha=0.7, linewidth=1.5)
    
    # Add moving average
    ax.plot(combined_df["step"], combined_df["predator_count"].rolling(window=10, center=True).mean(),
            color=predator_color, linewidth=2.5, label="10-step moving avg", linestyle="--", alpha=0.8)
    
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Predator Population", fontsize=11, color=predator_color)
    ax.tick_params(axis="y", labelcolor=predator_color)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Predator Population", fontweight="bold")
    
    # --- Plot 3: Combined Population (Dual Axes) ---
    ax1 = axes[2]
    ax2 = ax1.twinx()
    
    # Aggregate across all episodes (mean per step across episodes)
    mean_prey = combined_df.groupby("step")["prey_count"].mean()
    mean_predator = combined_df.groupby("step")["predator_count"].mean()
    
    line1 = ax1.plot(mean_prey.index, mean_prey.values, color=prey_color, linewidth=2.5,
                     label="Prey (mean)", marker="o", markersize=3, alpha=0.8)
    line2 = ax2.plot(mean_predator.index, mean_predator.values, color=predator_color, linewidth=2.5,
                     label="Predator (mean)", marker="s", markersize=3, alpha=0.8)
    
    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("Prey Population", fontsize=11, color=prey_color)
    ax2.set_ylabel("Predator Population", fontsize=11, color=predator_color)
    ax1.tick_params(axis="y", labelcolor=prey_color)
    ax2.tick_params(axis="y", labelcolor=predator_color)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Combined Population Dynamics", fontweight="bold")
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = results_path / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Population dynamics plot saved to: {output_path}")
    plt.close()


def plot_training_curves(training_log_file="src/models/results/training_log_prey.csv",
                         agent_type="PREY",
                         output_filename=None):
    """
    Generate training curves from training log CSV.
    
    Creates:
    1. Episode reward over training
    2. Moving average reward (shows learning trend)
    3. Epsilon decay (exploration rate)
    4. Optional: States visited per episode
    
    Args:
        training_log_file (str): Path to training_log_*.csv file
        agent_type (str): "PREY" or "PREDATOR" for title
        output_filename (str): Custom output filename; defaults to training_curves_PREY.png
    """
    log_path = Path(training_log_file)
    
    if not log_path.exists():
        print(f"Training log file not found: {training_log_file}")
        return
    
    # Load training log
    df = pd.read_csv(log_path)
    
    if df.empty:
        print("Training log is empty")
        return
    
    print(f"Loading training log for {agent_type}. Found {len(df)} episodes.")
    
    # Determine output filename
    if output_filename is None:
        output_filename = f"training_curves_{agent_type}.png"
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves - {agent_type} Agent", fontsize=14, fontweight="bold")
    
    # Color scheme
    reward_color = "#3498db"  # Blue
    avg_color = "#e74c3c"  # Red
    epsilon_color = "#f39c12"  # Orange
    
    # --- Plot 1: Episode Reward ---
    ax = axes[0, 0]
    ax.plot(df["episode"], df["episode_reward"], marker="o", color=reward_color,
            linewidth=1.5, markersize=4, alpha=0.7, label="Episode reward")
    
    # Moving average (20-episode window)
    ma_window = min(20, len(df) // 3)
    if ma_window >= 2:
        moving_avg = df["episode_reward"].rolling(window=ma_window, center=True).mean()
        ax.plot(df["episode"], moving_avg, color=avg_color, linewidth=2.5,
                label=f"{ma_window}-episode moving avg", linestyle="--", alpha=0.8)
    
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_title("Episode Reward Over Training", fontweight="bold")
    
    # --- Plot 2: Mean Agent Reward (species-level) ---
    ax = axes[0, 1]
    reward_col = "mean_prey_reward" if agent_type == "PREY" else "mean_predator_reward"
    
    if reward_col in df.columns:
        ax.plot(df["episode"], df[reward_col], marker="o", color=reward_color,
                linewidth=1.5, markersize=4, alpha=0.7, label=f"Mean {agent_type.lower()} reward")
        
        # Moving average
        if ma_window >= 2:
            moving_avg = df[reward_col].rolling(window=ma_window, center=True).mean()
            ax.plot(df["episode"], moving_avg, color=avg_color, linewidth=2.5,
                    label=f"{ma_window}-episode moving avg", linestyle="--", alpha=0.8)
        
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(f"Mean Reward ({agent_type})", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.set_title(f"Mean {agent_type} Agent Reward", fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"Column '{reward_col}' not found", ha="center", va="center")
    
    # --- Plot 3: Epsilon Decay (Exploration Rate) ---
    ax = axes[1, 0]
    if "epsilon" in df.columns:
        ax.plot(df["episode"], df["epsilon"], marker="s", color=epsilon_color,
                linewidth=2, markersize=4, alpha=0.8, label="Epsilon (exploration rate)")
        ax.axhline(y=0.01, color="gray", linestyle=":", linewidth=1.5, label="Min epsilon (0.01)")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Epsilon", fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.set_title("Epsilon (Exploration Rate) Decay", fontweight="bold")
    
    # --- Plot 4: Population at End of Episode ---
    ax = axes[1, 1]
    pop_col = "prey_population_end" if agent_type == "PREY" else "predator_population_end"
    energy_col = "average_prey_energy" if agent_type == "PREY" else "average_predator_energy"
    
    if pop_col in df.columns:
        ax.bar(df["episode"], df[pop_col], color=reward_color, alpha=0.7, label=f"{agent_type} population")
        
        if energy_col in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df["episode"], df[energy_col], color=avg_color, linewidth=2.5,
                     marker="D", markersize=5, label=f"Avg {agent_type.lower()} energy", alpha=0.8)
            ax2.set_ylabel(f"Average {agent_type} Energy", fontsize=11, color=avg_color)
            ax2.tick_params(axis="y", labelcolor=avg_color)
            ax2.set_ylim([0, 105])
        
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(f"{agent_type} Population", fontsize=11, color=reward_color)
        ax.tick_params(axis="y", labelcolor=reward_color)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(f"{agent_type} Population & Energy Trend", fontweight="bold")
        
        # Combined legend
        if energy_col in df.columns:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = log_path.parent / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Training curves saved to: {output_path}")
    plt.close()


def plot_both_species_training(results_dir="src/models/results"):
    """
    Generate side-by-side training curves for both prey and predator agents.
    
    Args:
        results_dir (str): Directory containing training_log_*.csv files
    """
    results_path = Path(results_dir)
    
    prey_log = results_path / "training_log_prey.csv"
    predator_log = results_path / "training_log_predator.csv"
    
    prey_exists = prey_log.exists()
    predator_exists = predator_log.exists()
    
    if not prey_exists and not predator_exists:
        print("No training logs found")
        return
    
    if prey_exists and predator_exists:
        # Load both
        prey_df = pd.read_csv(prey_log)
        pred_df = pd.read_csv(predator_log)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training Comparison: Prey vs Predator", fontsize=14, fontweight="bold")
        
        # Prey training curve
        ax = axes[0]
        ax.plot(prey_df["episode"], prey_df["episode_reward"], marker="o", color="#2ecc71",
                linewidth=1.5, markersize=4, alpha=0.7)
        ma_window = min(20, len(prey_df) // 3)
        if ma_window >= 2:
            moving_avg = prey_df["episode_reward"].rolling(window=ma_window, center=True).mean()
            ax.plot(prey_df["episode"], moving_avg, color="#27ae60", linewidth=2.5,
                    linestyle="--", alpha=0.8)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_title("Prey Training", fontweight="bold")
        
        # Predator training curve
        ax = axes[1]
        ax.plot(pred_df["episode"], pred_df["episode_reward"], marker="s", color="#e74c3c",
                linewidth=1.5, markersize=4, alpha=0.7)
        ma_window = min(20, len(pred_df) // 3)
        if ma_window >= 2:
            moving_avg = pred_df["episode_reward"].rolling(window=ma_window, center=True).mean()
            ax.plot(pred_df["episode"], moving_avg, color="#c0392b", linewidth=2.5,
                    linestyle="--", alpha=0.8)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_title("Predator Training", fontweight="bold")
        
        plt.tight_layout()
        output_path = results_path / "training_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training comparison plot saved to: {output_path}")
        plt.close()
    
    elif prey_exists:
        plot_training_curves(str(prey_log), "PREY", "training_curves_prey.png")
    
    elif predator_exists:
        plot_training_curves(str(predator_log), "PREDATOR", "training_curves_predator.png")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("GENERATING THESIS PLOTS")
    print("=" * 60)
    
    # Plot population dynamics
    plot_population_dynamics()
    
    # Plot training curves
    plot_training_curves(agent_type="PREY")
    
    # Optionally plot both species if both logs exist
    plot_both_species_training()
    
    print("=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
