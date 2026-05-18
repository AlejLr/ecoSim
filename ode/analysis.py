from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from .config import RESULTS_DIR


def plot_time_series(df: pd.DataFrame, title: str, out_path: Path | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["time"], df["prey"], label="Prey", color="#2ecc71")
    ax.plot(df["time"], df["predator"], label="Predator", color="#e74c3c")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if out_path is None:
        out_path = RESULTS_DIR / (title.replace(" ", "_").lower() + "_timeseries.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_phase(df: pd.DataFrame, title: str, out_path: Path | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df["prey"], df["predator"], color="#34495e")
    ax.set_xlabel("Prey population")
    ax.set_ylabel("Predator population")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if out_path is None:
        out_path = RESULTS_DIR / (title.replace(" ", "_").lower() + "_phase.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def summarize_trajectory(df: pd.DataFrame, extinct_threshold: float = 1.0) -> Dict[str, Any]:
    prey = df["prey"].to_numpy()
    pred = df["predator"].to_numpy()

    def approx_period(series: np.ndarray, t: np.ndarray) -> float | None:
        try:
            peaks, _ = find_peaks(series)
            if len(peaks) > 1:
                times = t[peaks]
                periods = np.diff(times)
                return float(np.mean(periods))
        except Exception:
            return None
        return None

    period_prey = approx_period(prey, df["time"].to_numpy())

    summary = {
        "prey_mean": float(np.mean(prey)),
        "predator_mean": float(np.mean(pred)),
        "prey_min": float(np.min(prey)),
        "prey_max": float(np.max(prey)),
        "predator_min": float(np.min(pred)),
        "predator_max": float(np.max(pred)),
        "prey_extinct": bool((prey < extinct_threshold).any()),
        "predator_extinct": bool((pred < extinct_threshold).any()),
        "prey_oscillation_period": period_prey,
    }
    return summary


def save_summary(summary: Dict[str, Any], name: str) -> Path:
    out = RESULTS_DIR / f"summary_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out


def save_csv(df: pd.DataFrame, name: str) -> Path:
    out = RESULTS_DIR / f"trajectory_{name}.csv"
    df.to_csv(out, index=False)
    return out


__all__ = ["plot_time_series", "plot_phase", "summarize_trajectory", "save_summary", "save_csv"]
