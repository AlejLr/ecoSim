"""
Visual simulation renderer for qualitative agent behavior inspection.
Renders environment state showing:
- Prey agents (green)
- Predator agents (red)
- Grass resources (light green)
- Water resources (blue)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class EnvironmentVisualizer:
    """Renders ecosystem state at specific timesteps."""
    
    def __init__(self, grid_size=150):
        """
        Initialize visualizer.
        
        Args:
            grid_size: Size of the environment grid (default 150x150)
        """
        self.grid_size = grid_size
        self.fig = None
        self.ax = None
    
    def render_frame(self, 
                     prey_positions,
                     predator_positions,
                     grass_positions,
                     water_positions,
                     step=0,
                     title="",
                     show_grid=False):
        """
        Render a single frame of the environment.
        
        Args:
            prey_positions: List of (x, y) tuples for prey
            predator_positions: List of (x, y) tuples for predators
            grass_positions: List of (x, y) tuples for grass tiles
            water_positions: List of (x, y) tuples for water tiles
            step: Step number for title
            title: Optional custom title
            show_grid: Whether to show grid lines
        
        Returns:
            matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create background (empty tiles in light gray)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Flip Y so (0,0) is top-left
        
        # Background
        ax.add_patch(patches.Rectangle((0, 0), self.grid_size, self.grid_size,
                                       linewidth=0, facecolor='#f5f5f5'))
        
        # Draw resources
        if grass_positions:
            grass_x, grass_y = zip(*grass_positions)
            ax.scatter(grass_x, grass_y, c='#90EE90', s=50, marker='s', 
                      label='Grass', alpha=0.6, edgecolors='none')
        
        if water_positions:
            water_x, water_y = zip(*water_positions)
            ax.scatter(water_x, water_y, c='#87CEEB', s=50, marker='o',
                      label='Water', alpha=0.5, edgecolors='none')
        
        # Draw prey
        if prey_positions:
            prey_x, prey_y = zip(*prey_positions)
            ax.scatter(prey_x, prey_y, c='#2ecc71', s=120, marker='^',
                      label='Prey', edgecolors='darkgreen', linewidths=1.5, zorder=5)
        
        # Draw predators
        if predator_positions:
            pred_x, pred_y = zip(*predator_positions)
            ax.scatter(pred_x, pred_y, c='#e74c3c', s=150, marker='v',
                      label='Predator', edgecolors='darkred', linewidths=1.5, zorder=5)
        
        # Add grid if requested
        if show_grid:
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        # Labels and legend
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            prey_count = len(prey_positions) if prey_positions else 0
            pred_count = len(predator_positions) if predator_positions else 0
            ax.set_title(f"Ecosystem State - Step {step}\n"
                        f"Prey: {prey_count} | Predators: {pred_count}",
                        fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        
        # Statistics box
        stats_text = f"Step: {step}\nPrey: {len(prey_positions) if prey_positions else 0}\nPredators: {len(predator_positions) if predator_positions else 0}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
    
    def save_frame(self, fig, output_path, dpi=100):
        """Save figure to file."""
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def extract_positions_from_csv(csv_file, env_class=None):
    """
    Extract agent and resource positions from environment state.
    
    For now, returns simulated positions based on population counts.
    In future, can be enhanced to log actual positions.
    
    Args:
        csv_file: Path to CSV file with step data
        env_class: Optional environment class instance
    
    Returns:
        List of (prey_pos, predator_pos, grass_pos, water_pos, step) tuples
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    frames = []
    
    for idx, row in df.iterrows():
        step = int(row['step'])
        prey_count = int(row['prey_count'])
        predator_count = int(row['predator_count'])
        
        # Generate random positions (in production, load from actual env state)
        prey_pos = [(np.random.randint(0, 150), np.random.randint(0, 150)) 
                   for _ in range(prey_count)]
        pred_pos = [(np.random.randint(0, 150), np.random.randint(0, 150)) 
                   for _ in range(predator_count)]
        
        # Grass: roughly 70% of 22500 tiles = ~15750 grass tiles
        # Sample some for visualization (too many to plot)
        grass_count = min(500, max(50, prey_count * 2))
        grass_pos = [(np.random.randint(0, 150), np.random.randint(0, 150)) 
                    for _ in range(grass_count)]
        
        # Water: roughly 20% of tiles = ~4500, sample ~200
        water_count = min(200, max(30, prey_count // 2))
        water_pos = [(np.random.randint(0, 150), np.random.randint(0, 150)) 
                    for _ in range(water_count)]
        
        frames.append((prey_pos, pred_pos, grass_pos, water_pos, step))
    
    return frames


def create_animation_frames(episode_csv,
                           output_dir="src/models/results/frames",
                           frame_skip=10,
                           dpi=100):
    """
    Generate and save individual frame images from episode data.
    
    Args:
        episode_csv: Path to saved_agents_episode_N.csv
        output_dir: Directory to save PNG frames
        frame_skip: Save every Nth frame (default: 10)
        dpi: Resolution of saved images
    
    Returns:
        List of saved frame paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data from CSV
    frames = extract_positions_from_csv(episode_csv)
    
    visualizer = EnvironmentVisualizer(grid_size=150)
    saved_frames = []
    
    print(f"Generating frames from {episode_csv}...")
    
    for i, (prey_pos, pred_pos, grass_pos, water_pos, step) in enumerate(frames):
        if i % frame_skip != 0:
            continue
        
        fig, _ = visualizer.render_frame(
            prey_pos, pred_pos, grass_pos, water_pos,
            step=step,
            show_grid=False
        )
        
        frame_path = output_path / f"frame_{step:04d}.png"
        visualizer.save_frame(fig, frame_path, dpi=dpi)
        saved_frames.append(frame_path)
        
        if (len(saved_frames)) % 10 == 0:
            print(f"  ✓ Saved {len(saved_frames)} frames...")
    
    print(f"✓ Generated {len(saved_frames)} frames to {output_path}")
    return saved_frames


def create_animation_gif(frame_dir, output_file, duration=100, loop=0):
    """
    Create animated GIF from PNG frames.
    
    Args:
        frame_dir: Directory containing frame_*.png files
        output_file: Path to output GIF
        duration: Duration per frame in milliseconds
        loop: 0 = infinite loop
    
    Returns:
        Path to created GIF
    """
    frame_path = Path(frame_dir)
    frame_files = sorted(frame_path.glob("frame_*.png"))
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return None
    
    print(f"Creating GIF from {len(frame_files)} frames...")
    
    images = [Image.open(f) for f in frame_files]
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )
    
    print(f"✓ GIF saved to {output_file}")
    return output_file


def create_animation_mp4(frame_dir, output_file, fps=10):
    """
    Create MP4 video from PNG frames using imageio.
    
    Args:
        frame_dir: Directory containing frame_*.png files
        output_file: Path to output MP4
        fps: Frames per second
    
    Returns:
        Path to created MP4 or None if imageio not available
    """
    try:
        import imageio
    except ImportError:
        print("⚠️ imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return None
    
    frame_path = Path(frame_dir)
    frame_files = sorted(frame_path.glob("frame_*.png"))
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return None
    
    print(f"Creating MP4 from {len(frame_files)} frames...")
    
    writer = imageio.get_writer(output_file, fps=fps)
    
    for frame_file in frame_files:
        image = imageio.imread(frame_file)
        writer.append_data(image)
    
    writer.close()
    print(f"✓ MP4 saved to {output_file}")
    return output_file


def visualize_episode(episode_num, 
                     results_dir="src/models/results",
                     output_dir=None,
                     frame_skip=10,
                     create_gif=False,
                     create_mp4=False):
    """
    Generate all visualizations for a single episode.
    
    Args:
        episode_num: Episode number (0-indexed)
        results_dir: Directory containing saved_agents_episode_*.csv
        output_dir: Custom output directory (default: results_dir/frames)
        frame_skip: Save every Nth frame
        create_gif: Whether to create animated GIF
        create_mp4: Whether to create MP4 video
    
    Returns:
        Dictionary with paths to generated files
    """
    results_path = Path(results_dir)
    episode_file = results_path / f"saved_agents_episode_{episode_num}.csv"
    
    if not episode_file.exists():
        print(f"Episode file not found: {episode_file}")
        return {}
    
    if output_dir is None:
        output_dir = results_path / "frames" / f"episode_{episode_num}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Visualizing Episode {episode_num}")
    print(f"{'='*70}\n")
    
    # Generate frames
    frame_files = create_animation_frames(
        str(episode_file),
        output_dir=str(output_path),
        frame_skip=frame_skip,
        dpi=100
    )
    
    results = {"frames_dir": str(output_path), "frame_count": len(frame_files)}
    
    # Create GIF if requested
    if create_gif and frame_files:
        gif_path = output_path / f"episode_{episode_num}_animation.gif"
        create_animation_gif(str(output_path), str(gif_path), duration=100)
        results["gif"] = str(gif_path)
    
    # Create MP4 if requested
    if create_mp4 and frame_files:
        mp4_path = output_path / f"episode_{episode_num}_animation.mp4"
        mp4_result = create_animation_mp4(str(output_path), str(mp4_path), fps=10)
        if mp4_result:
            results["mp4"] = str(mp4_result)
    
    print(f"\n{'='*70}")
    print(f"Visualization complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Generating visualizations for Episode 0...")
    
    # Generate frames
    results = visualize_episode(
        episode_num=0,
        results_dir="src/models/results",
        frame_skip=10,
        create_gif=True,
        create_mp4=False  # Set to True if imageio installed
    )
    
    print("\nGenerated files:")
    for key, value in results.items():
        print(f"  {key}: {value}")
