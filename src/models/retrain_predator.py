"""Retrain the predator with a stronger hunting reward.

This helper temporarily increases HUNTING_SUCCESS_BONUS in the runtime modules
that consume it, trains a predator with the existing single-agent trainer, and
saves a dedicated checkpoint.

Example:
    python -m src.models.retrain_predator --episodes 2000 --hunt-bonus 5.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random

import numpy as np

# Add src to path so this can be run as `python -m src.models.retrain_predator`
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.utils import get_next_run_number
from models.train import train_agent, evaluate_agent


def _patch_hunt_bonus(hunt_bonus: float):
    """Apply a temporary hunting bonus override to modules that read it at runtime."""
    import config.config as config_module
    import agents.agent as agent_module
    import models.training_protocols as protocols_module

    original_values = {
        "config": config_module.HUNTING_SUCCESS_BONUS,
        "agent": agent_module.HUNTING_SUCCESS_BONUS,
        "protocols": protocols_module.HUNTING_SUCCESS_BONUS,
    }

    config_module.HUNTING_SUCCESS_BONUS = hunt_bonus
    agent_module.HUNTING_SUCCESS_BONUS = hunt_bonus
    protocols_module.HUNTING_SUCCESS_BONUS = hunt_bonus

    return original_values


def _restore_hunt_bonus(original_values):
    import config.config as config_module
    import agents.agent as agent_module
    import models.training_protocols as protocols_module

    config_module.HUNTING_SUCCESS_BONUS = original_values["config"]
    agent_module.HUNTING_SUCCESS_BONUS = original_values["agent"]
    protocols_module.HUNTING_SUCCESS_BONUS = original_values["protocols"]


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain predator with a stronger hunting reward.")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes for the predator")
    parser.add_argument("--hunt-bonus", type=float, default=5.0, help="Temporary hunting reward bonus to use during training")
    parser.add_argument("--num-prey", type=int, default=6, help="Number of background prey agents")
    parser.add_argument("--num-predators", type=int, default=2, help="Number of background predator agents")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes after training")
    parser.add_argument("--output", type=str, default=None, help="Optional output checkpoint path")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed override")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"ASCII-safe seed set to {args.seed}")
    run_number = get_next_run_number()

    print("\n" + "=" * 70)
    print("PREDATOR RETRAINING WITH STRONGER HUNTING REWARD")
    print("=" * 70)
    print(f"Episodes: {args.episodes}")
    print(f"Hunting bonus override: {args.hunt_bonus}")
    print(f"Background prey: {args.num_prey}")
    print(f"Background predators: {args.num_predators}")
    print("=" * 70 + "\n")

    original_values = _patch_hunt_bonus(args.hunt_bonus)
    try:
        predator_agent, episode_rewards, episode_steps = train_agent(
            num_episodes=args.episodes,
            agent_type="PREDATOR",
            num_prey=args.num_prey,
            num_predators=args.num_predators,
            run_number=run_number,
        )

        evaluate_agent(
            predator_agent,
            num_episodes=args.eval_episodes,
            agent_type="PREDATOR",
            num_prey=args.num_prey,
            num_predators=args.num_predators,
        )

        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(exist_ok=True)

        if args.output:
            model_path = Path(args.output)
            if not model_path.is_absolute():
                model_path = model_dir / model_path
        else:
            bonus_tag = str(args.hunt_bonus).replace(".", "p")
            model_path = model_dir / f"trained_predator_{run_number}_hunt{bonus_tag}.pkl"

        predator_agent.save_model(str(model_path))
        print(f"Saved retrained predator model to: {model_path}")
        print(f"Final mean episode reward (last 10): {sum(episode_rewards[-10:]) / min(10, len(episode_rewards)):.2f}")
        print(f"Final mean episode length (last 10): {sum(episode_steps[-10:]) / min(10, len(episode_steps)):.2f}")
    finally:
        _restore_hunt_bonus(original_values)


if __name__ == "__main__":
    main()