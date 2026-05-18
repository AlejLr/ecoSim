from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .models import lotka_volterra, logistic_predator_prey
from .config import T_SPAN, T_EVAL, INITIAL_CONDITIONS, RESULTS_DIR, ODE_CONFIG


def simulate_ode(model_fn: Callable, params: Dict[str, float], y0: list[float],
                 t_span: tuple[float, float] = T_SPAN, t_eval: np.ndarray = T_EVAL) -> pd.DataFrame:
    """Simulate an ODE model and return a trajectory DataFrame.

    Returns columns: time, prey, predator
    """
    sol = solve_ivp(fun=lambda t, y: model_fn(t, y, **params), t_span=t_span, y0=y0, t_eval=t_eval, vectorized=False)
    if not sol.success:
        raise RuntimeError("ODE solver failed: " + (sol.message or "unspecified"))

    df = pd.DataFrame({"time": sol.t, "prey": sol.y[0], "predator": sol.y[1]})
    return df


def simulate_named(model: str, scenario_params: Dict[str, Any] | None = None,
                   y0: list[float] | None = None, t_eval: np.ndarray | None = None) -> pd.DataFrame:
    """Convenience: simulate by model name using configuration defaults."""
    model = model.lower()
    if model.startswith("lotka"):
        model_fn = lotka_volterra
        params = ODE_CONFIG.get("lotka_volterra", {})
    elif model.startswith("logistic"):
        model_fn = logistic_predator_prey
        params = ODE_CONFIG.get("logistic_predator_prey", {})
    else:
        raise ValueError(f"Unknown model: {model}")

    if scenario_params:
        params = {**params, **scenario_params}
    if y0 is None:
        y0 = INITIAL_CONDITIONS
    if t_eval is None:
        t_eval = T_EVAL

    return simulate_ode(model_fn, params, y0=y0, t_eval=t_eval)


def save_trajectory(df: pd.DataFrame, name: str) -> Path:
    """Save trajectory CSV to results dir and return path."""
    out = RESULTS_DIR / f"trajectory_{name}.csv"
    df.to_csv(out, index=False)
    return out


def export_dict(df: pd.DataFrame) -> Dict[str, list]:
    return {"time": df["time"].tolist(), "prey": df["prey"].tolist(), "predator": df["predator"].tolist()}


__all__ = ["simulate_ode", "simulate_named", "save_trajectory", "export_dict"]
