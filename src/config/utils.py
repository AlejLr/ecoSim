"""Configuration utilities - seed control, run numbering, and reproducibility"""

import random
import numpy as np
from pathlib import Path
from config.config import SEED


def set_global_seed(seed=None):
    """Set seed for all random sources for reproducibility
    
    Args:
        seed: Seed value (int) or None for non-deterministic behavior
              If None, uses SEED from config.py
    """
    if seed is None:
        seed = SEED
    
    if seed is not None:
        # Seed Python's random module
        random.seed(seed)
        
        # Seed NumPy's random module
        np.random.seed(seed)
        
        # Note: gym/gymnasium seeding is done per-env in reset()
        print(f"[OK] Global seed set to {seed}")
    else:
        print("[OK] Non-deterministic mode (seed=None)")


def seed_env(env, seed=None):
    """Seed a gymnasium environment
    
    Args:
        env: Gymnasium environment
        seed: Seed value or None
    """
    if seed is None:
        seed = SEED
    
    if seed is not None:
        env.action_space.seed(seed)
        # For gym/gymnasium, seeding happens via env.reset(seed=seed)
        return seed
    return None


def get_next_run_number():
    """Get the next run number by finding the highest existing run number and incrementing.
    
    Returns:
        int: Next run number (1 if no runs exist yet)
    """
    results_dir = Path(__file__).parent.parent / "models" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all log files matching pattern log_run_*.txt
    log_files = list(results_dir.glob("log_run_*.txt"))
    
    if not log_files:
        return 1
    
    # Extract run numbers and find max
    run_numbers = []
    for f in log_files:
        try:
            # Extract number from log_run_N.txt
            num_str = f.stem.replace("log_run_", "")
            run_numbers.append(int(num_str))
        except (ValueError, AttributeError):
            pass
    
    return max(run_numbers) + 1 if run_numbers else 1


def get_latest_run_number():
    """Get the latest (highest) run number that exists.
    
    Returns:
        int or None: Latest run number, or None if no runs exist
    """
    results_dir = Path(__file__).parent.parent / "models" / "results"
    
    if not results_dir.exists():
        return None
    
    log_files = list(results_dir.glob("log_run_*.txt"))
    
    if not log_files:
        return None
    
    run_numbers = []
    for f in log_files:
        try:
            num_str = f.stem.replace("log_run_", "")
            run_numbers.append(int(num_str))
        except (ValueError, AttributeError):
            pass
    
    return max(run_numbers) if run_numbers else None


def get_latest_model_path(agent_type, run_number=None):
    """Get path to the latest trained model for an agent type.
    
    Args:
        agent_type: "PREY" or "PREDATOR"
        run_number: Specific run number to load, or None for latest
        
    Returns:
        Path: Path to model file, or None if not found
    """
    models_dir = Path(__file__).parent.parent / "models"
    
    if run_number is not None:
        # Look for specific run model
        model_file = models_dir / f"trained_{agent_type.lower()}_{run_number}.pkl"
        return model_file if model_file.exists() else None
    
    # Find latest model
    model_pattern = models_dir.glob(f"trained_{agent_type.lower()}_*.pkl")
    model_files = list(model_pattern)
    
    if not model_files:
        return None
    
    # Extract run numbers and find latest
    run_numbers = []
    for f in model_files:
        try:
            # Extract number from trained_prey_N.pkl or trained_prey_N_protocol_X.pkl
            # Split: trained_prey_6_protocol2_cycle1 → parts = ["trained", "prey", "6", "protocol2", "cycle1"]
            parts = f.stem.split("_")
            # The run number is always at position 2 (0=trained, 1=prey/predator, 2=run_number)
            num_str = parts[2]
            run_numbers.append(int(num_str))
        except (ValueError, IndexError):
            pass
    
    if run_numbers:
        latest_run = max(run_numbers)
        # Try exact match first (simple naming)
        exact_match = models_dir / f"trained_{agent_type.lower()}_{latest_run}.pkl"
        if exact_match.exists():
            return exact_match
        # Otherwise find any file with this run number
        for f in model_files:
            try:
                parts = f.stem.split("_")
                if int(parts[2]) == latest_run:
                    return f
            except (ValueError, IndexError):
                pass
    
    return None
