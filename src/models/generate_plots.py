#!/usr/bin/env python3
"""
Command-line interface for generating thesis plots.

Usage:
    python -m src.models.generate_plots [--population] [--training] [--both]
    
    python -m src.models.generate_plots                 # Generate all plots
    python -m src.models.generate_plots --population    # Population dynamics only
    python -m src.models.generate_plots --training      # Training curves only
"""

import sys
import argparse
from pathlib import Path
from models.plotting import plot_population_dynamics, plot_training_curves, plot_both_species_training


def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis visualization plots from simulation and training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.models.generate_plots                 # All plots
  python -m src.models.generate_plots --population    # Population only
  python -m src.models.generate_plots --training      # Training curves only
  python -m src.models.generate_plots --both          # Comparison plots
        """
    )
    
    parser.add_argument(
        "--population",
        action="store_true",
        help="Generate population dynamics plots from saved episodes"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Generate training curves from training logs"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate side-by-side training comparison (prey vs predator)"
    )
    parser.add_argument(
        "--results-dir",
        default="src/models/results",
        help="Directory containing data files (default: src/models/results)"
    )
    
    args = parser.parse_args()
    
    # If no specific plot type specified, generate all
    generate_all = not (args.population or args.training or args.both)
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("THESIS PLOT GENERATOR")
    print("=" * 70)
    print(f"Results directory: {results_dir.resolve()}")
    print()
    
    # Generate requested plots
    if generate_all or args.population:
        print("📊 Generating population dynamics plots...")
        try:
            plot_population_dynamics(str(results_dir), "population_dynamics.png")
            print()
        except Exception as e:
            print(f"⚠️ Population plot generation failed: {e}")
            print()
    
    if generate_all or args.training:
        print("📈 Generating training curves...")
        try:
            plot_training_curves(
                str(results_dir / "training_log_prey.csv"),
                agent_type="PREY",
                output_filename="training_curves_prey.png"
            )
            print()
        except FileNotFoundError:
            print("⚠️ training_log_prey.csv not found; skipping prey training curves")
            print()
        except Exception as e:
            print(f"⚠️ Training curves generation failed: {e}")
            print()
    
    if generate_all or args.both:
        print("🔀 Generating training comparison plots...")
        try:
            plot_both_species_training(str(results_dir))
            print()
        except Exception as e:
            print(f"⚠️ Comparison plot generation failed: {e}")
            print()
    
    print("=" * 70)
    print("✅ Plot generation complete!")
    print(f"📁 Plots saved to: {results_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
