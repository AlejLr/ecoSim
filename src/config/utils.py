"""Configuration utilities - seed control and reproducibility"""

import random
import numpy as np
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
        print(f"✓ Global seed set to {seed}")
    else:
        print("✓ Non-deterministic mode (seed=None)")


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
