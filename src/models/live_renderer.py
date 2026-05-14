"""
Live environment renderer showing actual agent positions and behavior.
Can render episodes in real-time or save snapshots from running simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


class LiveEnvironmentRenderer:
    """Renders live environment state with actual agent positions."""
    
    def __init__(self, grid_size=150, figsize=(12, 10)):
        self.grid_size = grid_size
        self.figsize = figsize
    
    def render_environment(self,
                          env,
                          step=0,
                          show_vision=False,
                          title="",
                          save_path=None):
        """
        Render current environment state.
        
        Args:
            env: EcoSimEnv or MultiAgentEcoSimEnv instance
            step: Current step number
            show_vision: Whether to show agent vision circles
            title: Custom title
            save_path: If provided, save to file instead of displaying
        
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Background
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        ax.add_patch(patches.Rectangle((0, 0), self.grid_size, self.grid_size,
                                       linewidth=1, facecolor='#f5f5f5', edgecolor='black'))
        
        # Render grass (sample sparse representation)
        grass_positions = []
        water_positions = []
        
        if hasattr(env, 'grid'):
            # Sample grid for visualization
            sample_rate = 10  # Show every 10th tile to avoid clutter
            for x in range(0, self.grid_size, sample_rate):
                for y in range(0, self.grid_size, sample_rate):
                    if x < self.grid_size and y < self.grid_size:
                        tile = env.grid[y][x]
                        if hasattr(tile, 'tile_type'):
                            if tile.tile_type == 'grass':
                                grass_positions.append((x, y))
                            elif tile.tile_type == 'water':
                                water_positions.append((x, y))
        
        # Draw resources
        if grass_positions:
            gx, gy = zip(*grass_positions)
            ax.scatter(gx, gy, c='#90EE90', s=40, marker='s',
                      label='Grass (sampled)', alpha=0.4, edgecolors='none')
        
        if water_positions:
            wx, wy = zip(*water_positions)
            ax.scatter(wx, wy, c='#87CEEB', s=40, marker='o',
                      label='Water (sampled)', alpha=0.3, edgecolors='none')
        
        # Collect agents
        all_agents = []
        prey_agents = []
        pred_agents = []
        
        if hasattr(env, 'agents'):
            all_agents = env.agents
        elif hasattr(env, 'agent') and env.agent:
            all_agents = [env.agent]
            if hasattr(env, 'other_agents'):
                all_agents.extend(env.other_agents)
        
        # Separate by type
        for agent in all_agents:
            if agent.is_alive():
                if hasattr(agent, 'agent_type'):
                    if agent.agent_type == 'PREY':
                        prey_agents.append(agent)
                    elif agent.agent_type == 'PREDATOR':
                        pred_agents.append(agent)
        
        # Draw prey
        if prey_agents:
            prey_x = [a.x for a in prey_agents]
            prey_y = [a.y for a in prey_agents]
            ax.scatter(prey_x, prey_y, c='#2ecc71', s=200, marker='^',
                      label='Prey', edgecolors='darkgreen', linewidths=2, zorder=5)
            
            # Show vision if requested
            if show_vision:
                for agent in prey_agents:
                    vision_circle = patches.Circle((agent.x, agent.y), agent.vision_radius,
                                                   fill=False, edgecolor='green',
                                                   linestyle='--', linewidth=1, alpha=0.3)
                    ax.add_patch(vision_circle)
        
        # Draw predators
        if pred_agents:
            pred_x = [a.x for a in pred_agents]
            pred_y = [a.y for a in pred_agents]
            ax.scatter(pred_x, pred_y, c='#e74c3c', s=250, marker='v',
                      label='Predators', edgecolors='darkred', linewidths=2, zorder=5)
            
            # Show vision if requested
            if show_vision:
                for agent in pred_agents:
                    vision_circle = patches.Circle((agent.x, agent.y), agent.vision_radius,
                                                   fill=False, edgecolor='red',
                                                   linestyle='--', linewidth=1, alpha=0.3)
                    ax.add_patch(vision_circle)
        
        # Labels
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"Live Environment - Step {step}\n"
                        f"Prey: {len(prey_agents)} | Predators: {len(pred_agents)}",
                        fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        
        # Statistics
        stats_text = f"Step: {step}\nPrey: {len(prey_agents)}\nPredators: {len(pred_agents)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax
    
    def capture_episode_snapshots(self,
                                 env,
                                 max_steps=None,
                                 snapshot_interval=50,
                                 output_dir="src/models/results/snapshots",
                                 show_vision=False):
        """
        Run environment and capture snapshots at intervals.
        
        Args:
            env: Environment instance
            max_steps: Maximum steps to run (None for full episode)
            snapshot_interval: Capture every Nth step
            output_dir: Directory to save snapshots
            show_vision: Show agent vision circles
        
        Returns:
            List of saved snapshot paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        snapshots = []
        obs = env.reset()
        step = 0
        done = False
        
        print(f"Capturing episode snapshots to {output_path}...")
        
        while not done and (max_steps is None or step < max_steps):
            # Capture snapshot
            if step % snapshot_interval == 0:
                snapshot_path = output_path / f"snapshot_step_{step:04d}.png"
                self.render_environment(env, step=step, show_vision=show_vision,
                                      save_path=snapshot_path)
                snapshots.append(snapshot_path)
                print(f"  ✓ Captured step {step}")
            
            # Step environment
            action = np.random.randint(0, env.action_space.n) if hasattr(env, 'action_space') else 0
            obs, reward, done, info = env.step(action)
            step += 1
        
        print(f"✓ Captured {len(snapshots)} snapshots")
        return snapshots


def render_saved_episode_with_positions(episode_num,
                                       results_dir="src/models/results",
                                       output_file=None):
    """
    Render a saved episode with trajectory visualization.
    
    Args:
        episode_num: Episode number to load
        results_dir: Directory with saved_agents_episode_*.csv
        output_file: Path to save visualization
    
    Returns:
        Path to saved figure
    """
    import pandas as pd
    
    episode_file = Path(results_dir) / f"saved_agents_episode_{episode_num}.csv"
    
    if not episode_file.exists():
        print(f"Episode file not found: {episode_file}")
        return None
    
    df = pd.read_csv(episode_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Episode {episode_num} Population Dynamics", fontsize=14, fontweight='bold')
    
    # Left: Prey population
    ax = axes[0]
    ax.plot(df['step'], df['prey_count'], color='#2ecc71', linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(df['step'], df['prey_count'], alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Prey Count', fontsize=11)
    ax.set_title('Prey Population Over Episode', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Right: Predator population
    ax = axes[1]
    ax.plot(df['step'], df['predator_count'], color='#e74c3c', linewidth=2.5, marker='s', markersize=4)
    ax.fill_between(df['step'], df['predator_count'], alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Predator Count', fontsize=11)
    ax.set_title('Predator Population Over Episode', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = Path(results_dir) / f"episode_{episode_num}_populations.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Episode visualization saved to {output_file}")
    return output_file


if __name__ == "__main__":
    print("Live environment renderer test")
    print("Use render_environment() to visualize a running simulation")
    print("Use capture_episode_snapshots() to record behavioral snapshots")
    
    # Test: render saved episode data
    print("\nGenerating population visualization for Episode 0...")
    output = render_saved_episode_with_positions(0, "src/models/results")
    print(f"Saved to: {output}")
