#!/usr/bin/env python3
"""
Unified visualization pipeline for ecosystem simulations.
Combines all visualization capabilities (plots, animations, snapshots).

Usage:
    python -m src.models.visualize_simulation [--episode N] [--all-episodes] [--with-plots] [--with-animation]
    
    python -m src.models.visualize_simulation                    # Visualize episode 0
    python -m src.models.visualize_simulation --episode 2        # Specific episode
    python -m src.models.visualize_simulation --all-episodes     # All episodes
    python -m src.models.visualize_simulation --with-plots       # Include training curves
    python -m src.models.visualize_simulation --with-animation   # Generate GIFs
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

from src.models.visualizer import visualize_episode, create_animation_frames
from src.models.live_renderer import render_saved_episode_with_positions
from src.models.plotting import plot_population_dynamics, plot_training_curves


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive visualizations for ecosystem simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.models.visualize_simulation                 # Episode 0 visualizations
  python -m src.models.visualize_simulation --episode 2     # Specific episode
  python -m src.models.visualize_simulation --all-episodes  # All episodes
  python -m src.models.visualize_simulation --with-plots    # Include training curves
  python -m src.models.visualize_simulation --with-animation # Generate animations
        """
    )
    
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode number to visualize (default: 0)"
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Visualize all available episodes"
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="Generate training curves and population plots"
    )
    parser.add_argument(
        "--with-animation",
        action="store_true",
        help="Create animated GIFs from frames"
    )
    parser.add_argument(
        "--with-mp4",
        action="store_true",
        help="Create MP4 videos (requires imageio)"
    )
    parser.add_argument(
        "--results-dir",
        default="src/models/results",
        help="Directory containing simulation data (default: src/models/results)"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Save every Nth frame (default: 10)"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("ECOSYSTEM SIMULATION VISUALIZATION PIPELINE")
    print("=" * 80)
    print(f"Results directory: {results_dir.resolve()}")
    print()
    
    # Determine which episodes to process
    episodes_to_process = []
    
    if args.all_episodes:
        episode_files = sorted(results_dir.glob("saved_agents_episode_*.csv"))
        episodes_to_process = [int(f.stem.split("_")[-1]) for f in episode_files]
        print(f"Found {len(episodes_to_process)} episodes")
    else:
        episodes_to_process = [args.episode]
    
    # Process each episode
    for episode_num in episodes_to_process:
        episode_file = results_dir / f"saved_agents_episode_{episode_num}.csv"
        
        if not episode_file.exists():
            print(f"⚠️ Episode {episode_num} not found, skipping")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing Episode {episode_num}")
        print(f"{'='*80}\n")
        
        # 1. Generate population visualization
        print("📊 Generating population snapshot...")
        try:
            output_path = results_dir / f"episode_{episode_num}_populations.png"
            render_saved_episode_with_positions(episode_num, str(results_dir), str(output_path))
        except Exception as e:
            print(f"⚠️ Population snapshot failed: {e}")
        
        # 2. Generate animation frames
        print("🎬 Generating animation frames...")
        try:
            results = visualize_episode(
                episode_num,
                results_dir=str(results_dir),
                output_dir=str(results_dir / "frames" / f"episode_{episode_num}"),
                frame_skip=args.frame_skip,
                create_gif=args.with_animation,
                create_mp4=args.with_mp4
            )
            
            if "gif" in results:
                print(f"  ✓ GIF: {results['gif']}")
            if "mp4" in results:
                print(f"  ✓ MP4: {results['mp4']}")
        except Exception as e:
            print(f"⚠️ Animation generation failed: {e}")
    
    # 3. Generate global plots (only once, not per episode)
    if args.with_plots:
        print(f"\n{'='*80}")
        print("📈 Generating Training Curves")
        print(f"{'='*80}\n")
        
        try:
            prey_log = results_dir / "training_log_prey.csv"
            if prey_log.exists():
                plot_training_curves(str(prey_log), "PREY")
        except Exception as e:
            print(f"⚠️ Prey training curves failed: {e}")
        
        try:
            predator_log = results_dir / "training_log_predator.csv"
            if predator_log.exists():
                plot_training_curves(str(predator_log), "PREDATOR")
        except Exception as e:
            print(f"⚠️ Predator training curves failed: {e}")
        
        print(f"\n{'='*80}")
        print("📊 Generating Population Dynamics")
        print(f"{'='*80}\n")
        
        try:
            plot_population_dynamics(str(results_dir))
        except Exception as e:
            print(f"⚠️ Population dynamics plot failed: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ VISUALIZATION PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 All visualizations saved to: {results_dir.resolve()}\n")
    
    # List generated files
    print("Generated Files:")
    png_files = sorted(results_dir.glob("*.png"))
    gif_files = sorted(results_dir.glob("**/*.gif"))
    mp4_files = sorted(results_dir.glob("**/*.mp4"))
    
    for f in png_files:
        print(f"  📄 {f.name}")
    for f in gif_files:
        print(f"  🎬 {f.name}")
    for f in mp4_files:
        print(f"  🎥 {f.name}")
    
    print()


if __name__ == "__main__":
    main()
