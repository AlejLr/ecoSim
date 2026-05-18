from __future__ import annotations

from pathlib import Path
import numpy as np

# Results directory for ODE outputs
HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


ODE_CONFIG = {
    "lotka_volterra": {
        "alpha": 0.6,
        "beta": 0.02,
        "delta": 0.01,
        "gamma": 0.4,
    },
    "logistic_predator_prey": {
        "r": 0.8,
        "K": 200,
        "beta": 0.015,
        "delta": 0.01,
        "gamma": 0.35,
    },
}


# Default initial conditions and time grid
INITIAL_CONDITIONS = [80.0, 20.0]
T_SPAN = (0.0, 200.0)
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 1000)


__all__ = ["ODE_CONFIG", "INITIAL_CONDITIONS", "T_SPAN", "T_EVAL", "RESULTS_DIR"]
