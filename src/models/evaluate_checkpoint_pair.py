"""Official long-horizon evaluation for predator-prey checkpoint pairs.

Runs frozen Protocol-2 policies inside MultiAgentEcoSimEnv and records
population dynamics at every step. This is the single authoritative
evaluation script for the thesis.

Usage
-----
    python -m src.models.evaluate_checkpoint_pair \
        --prey-model src/models/trained_prey_43_protocol2_cycle5.pkl \
        --predator-model src/models/trained_predator_43_protocol2_cycle5.pkl \
        --run-name run43_cycle5 \
        --steps 2000 \
        --seed-offsets 0 1 2

Output
------
Per-run CSV (one row per step):
    step, prey_population, predator_population, total_alive,
    system_coexistence_score, system_extinct,
    avg/min/max_prey_energy, avg/min/max_predator_energy,
    cumulative_prey_births, cumulative_predator_births,
    cumulative_predation_events, cumulative_prey_deaths, cumulative_predator_deaths,
    reward, cumulative_reward,
    reference_agent_type, reference_agent_energy,
    reference_agent_context, reference_agent_target_detected,
    prey_model_path, predator_model_path, run_name, seed, long_episode_steps

Summary CSV (one row per seed, all derived metrics):
    --- Population dynamics ---
    final_prey_population, final_predator_population
    mean_prey_population, mean_predator_population
    prey_population_cv, predator_population_cv
    prey_oscillation_period, predator_oscillation_period
    --- Stability ---
    system_survival_fraction
    system_extinction_step          (0 = no extinction)
    coexistence_score_mean, coexistence_score_min, coexistence_score_final
    --- Demographic rates (per step) ---
    prey_birth_rate, predator_birth_rate
    predation_rate, prey_death_rate, predator_death_rate
    --- Mean individual lifespan (steps, approximated) ---
    mean_prey_lifespan, mean_predator_lifespan
    --- Raw event totals ---
    cumulative_predation_events, cumulative_prey_births, cumulative_predator_births
    cumulative_prey_deaths, cumulative_predator_deaths
    --- Metadata ---
    run_name, prey_model_path, predator_model_path, seed, steps_ran, csv_path
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.agent import Predator, Prey
from config.config import COEXISTENCE_PREY_FLOOR, COEXISTENCE_PREDATOR_FLOOR, SEED
from environment.multi_agent_gym_env import MultiAgentEcoSimEnv
from models.Q_learning import QLearningAgent
from models.coexistence_metrics import coexistence_score


# ---------------------------------------------------------------------------
# Event tracking
# ---------------------------------------------------------------------------

@dataclass
class EventTracker:
    prey_births: int = 0
    predator_births: int = 0
    predation_events: int = 0
    cumulative_prey_deaths: int = 0
    cumulative_predator_deaths: int = 0


@contextlib.contextmanager
def track_ecosim_events(tracker: EventTracker):
    """Monkey-patch Prey/Predator methods to count births, deaths, predation."""
    original_prey_reproduce = Prey.reproduce
    original_predator_reproduce = Predator.reproduce
    original_predator_eat = Predator.eat

    def wrapped_prey_reproduce(self, *args, **kwargs):
        offspring = original_prey_reproduce(self, *args, **kwargs)
        if offspring is not None:
            tracker.prey_births += 1
        return offspring

    def wrapped_predator_reproduce(self, *args, **kwargs):
        offspring = original_predator_reproduce(self, *args, **kwargs)
        if offspring is not None:
            tracker.predator_births += 1
        return offspring

    def wrapped_predator_eat(self, *args, **kwargs):
        reward = original_predator_eat(self, *args, **kwargs)
        if reward > 0:
            tracker.predation_events += 1
        return reward

    Prey.reproduce = wrapped_prey_reproduce
    Predator.reproduce = wrapped_predator_reproduce
    Predator.eat = wrapped_predator_eat
    try:
        yield
    finally:
        Prey.reproduce = original_prey_reproduce
        Predator.reproduce = original_predator_reproduce
        Predator.eat = original_predator_eat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clone_frozen_agent(model_path: Path, agent_id: int = 0) -> QLearningAgent:
    """Load a saved Q-table and set epsilon=0 for greedy evaluation."""
    agent = QLearningAgent.load_model_from_file(str(model_path))
    agent.agent_id = agent_id
    agent.epsilon = 0.0
    return agent


def _mean_or_zero(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _detect_oscillation_period(series: np.ndarray, min_peak_height_ratio: float = 0.5) -> float:
    """Return mean distance between peaks in a population series (in steps).

    Uses a simple local-maximum scan: a point is a peak if it is strictly
    greater than both immediate neighbours AND above min_peak_height_ratio *
    max(series). Returns 0.0 if fewer than two peaks are found.
    """
    if len(series) < 5:
        return 0.0
    threshold = float(np.max(series)) * min_peak_height_ratio
    peaks = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1] and series[i] >= threshold:
            peaks.append(i)
    if len(peaks) < 2:
        return 0.0
    gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# Per-step snapshot
# ---------------------------------------------------------------------------

def build_step_snapshot(
    env,
    step_index: int,
    reward: float,
    cumulative_reward: float,
    tracker: EventTracker,
    previous_alive_ids: set,
    known_agent_types: Dict[int, str],
) -> tuple:
    all_agents = list(env.all_agents)
    alive_agents = [a for a in all_agents if a.is_alive()]
    prey_agents = [a for a in alive_agents if a.agent_type == "PREY"]
    predator_agents = [a for a in alive_agents if a.agent_type == "PREDATOR"]

    current_alive_ids = {a.agent_id for a in alive_agents}
    for a in all_agents:
        known_agent_types.setdefault(a.agent_id, a.agent_type)

    deaths = previous_alive_ids - current_alive_ids
    prey_deaths = sum(1 for aid in deaths if known_agent_types.get(aid) == "PREY")
    predator_deaths = sum(1 for aid in deaths if known_agent_types.get(aid) == "PREDATOR")
    tracker.cumulative_prey_deaths += prey_deaths
    tracker.cumulative_predator_deaths += predator_deaths

    ref = alive_agents[0] if alive_agents else None
    ref_obs = ref.get_observation(env.env) if ref is not None else np.zeros(6, dtype=np.float32)

    snapshot = {
        # Core population counts
        "step": step_index,
        "prey_population": len(prey_agents),
        "predator_population": len(predator_agents),
        "total_alive": len(alive_agents),
        # Coexistence
        "system_coexistence_score": coexistence_score(len(prey_agents), len(predator_agents)),
        "system_extinct": int(len(prey_agents) == 0 or len(predator_agents) == 0),
        # Energy distribution
        "avg_prey_energy": _mean_or_zero(a.energy for a in prey_agents),
        "avg_predator_energy": _mean_or_zero(a.energy for a in predator_agents),
        "min_prey_energy": float(min((a.energy for a in prey_agents), default=0.0)),
        "max_prey_energy": float(max((a.energy for a in prey_agents), default=0.0)),
        "min_predator_energy": float(min((a.energy for a in predator_agents), default=0.0)),
        "max_predator_energy": float(max((a.energy for a in predator_agents), default=0.0)),
        # Cumulative event counts
        "cumulative_prey_births": tracker.prey_births,
        "cumulative_predator_births": tracker.predator_births,
        "cumulative_predation_events": tracker.predation_events,
        "cumulative_prey_deaths": tracker.cumulative_prey_deaths,
        "cumulative_predator_deaths": tracker.cumulative_predator_deaths,
        # Reward
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        # Reference agent diagnostics (first alive agent — species varies)
        "reference_agent_type": ref.agent_type if ref is not None else "NONE",
        "reference_agent_energy": float(ref.energy) if ref is not None else 0.0,
        # obs[5] = predator_detected (prey) or prey_density (predator)
        "reference_agent_context": int(ref_obs[5]),
        # obs[4] = food_detected (prey) or prey_detected (predator)
        "reference_agent_target_detected": int(ref_obs[4]),
    }
    return snapshot, current_alive_ids


# ---------------------------------------------------------------------------
# Derived summary metrics
# ---------------------------------------------------------------------------

def compute_derived_metrics(df: pd.DataFrame) -> Dict:
    """Compute all thesis evaluation metrics from a per-step DataFrame.

    Metrics
    -------
    Coefficient of variation (CV):
        std / mean of population counts over steps where the species is alive.
        CV < 0.3 suggests stable; 0.3–0.6 oscillating; >0.6 volatile/crashing.

    Oscillation period:
        Mean distance (in steps) between consecutive population peaks.
        0 if fewer than two peaks are detected.

    System survival fraction:
        Fraction of steps where neither species is extinct.

    Coexistence score (mean / min / final):
        Composite score in [0,1]; 1.0 requires prey>=4, predators>=2, ratio~2:1.

    Demographic rates (per step):
        birth_rate = total_births / steps_ran
        death_rate = total_deaths / steps_ran
        predation_rate = total_predation_events / steps_ran

    Mean individual lifespan (approximated, in steps):
        lifespan ≈ mean_population / death_rate   (Little's law approximation)
        Falls back to steps_ran / total_deaths when death_rate is available.
    """
    if df.empty:
        return {}

    steps_ran = int(df["step"].max())

    prey_series = df["prey_population"].values.astype(float)
    pred_series = df["predator_population"].values.astype(float)

    # Population CV — computed only over steps where species is alive
    prey_alive = prey_series[prey_series > 0]
    pred_alive = pred_series[pred_series > 0]
    prey_cv = float(np.std(prey_alive) / np.mean(prey_alive)) if len(prey_alive) > 1 else 0.0
    pred_cv = float(np.std(pred_alive) / np.mean(pred_alive)) if len(pred_alive) > 1 else 0.0

    # Oscillation period
    prey_period = _detect_oscillation_period(prey_series)
    pred_period = _detect_oscillation_period(pred_series)

    # Stability
    survival_fraction = float((df["system_extinct"] == 0).mean())
    extinct_steps = df.loc[df["system_extinct"] == 1, "step"]
    extinction_step = int(extinct_steps.iloc[0]) if not extinct_steps.empty else 0

    # Coexistence score
    score_mean = float(df["system_coexistence_score"].mean())
    score_min = float(df["system_coexistence_score"].min())
    score_final = float(df["system_coexistence_score"].iloc[-1])

    # Demographic rates
    total_prey_births = int(df["cumulative_prey_births"].iloc[-1])
    total_pred_births = int(df["cumulative_predator_births"].iloc[-1])
    total_predation = int(df["cumulative_predation_events"].iloc[-1])
    total_prey_deaths = int(df["cumulative_prey_deaths"].iloc[-1])
    total_pred_deaths = int(df["cumulative_predator_deaths"].iloc[-1])

    prey_birth_rate = total_prey_births / steps_ran if steps_ran > 0 else 0.0
    pred_birth_rate = total_pred_births / steps_ran if steps_ran > 0 else 0.0
    predation_rate = total_predation / steps_ran if steps_ran > 0 else 0.0
    prey_death_rate = total_prey_deaths / steps_ran if steps_ran > 0 else 0.0
    pred_death_rate = total_pred_deaths / steps_ran if steps_ran > 0 else 0.0

    # Mean individual lifespan — Little's law: L = lambda * W → W = L / lambda
    # where L = mean population size, lambda = death rate
    mean_prey_pop = float(np.mean(prey_alive)) if len(prey_alive) > 0 else 0.0
    mean_pred_pop = float(np.mean(pred_alive)) if len(pred_alive) > 0 else 0.0
    mean_prey_lifespan = mean_prey_pop / prey_death_rate if prey_death_rate > 0 else float(steps_ran)
    mean_pred_lifespan = mean_pred_pop / pred_death_rate if pred_death_rate > 0 else float(steps_ran)

    return {
        # Population dynamics
        "mean_prey_population": float(np.mean(prey_series)),
        "mean_predator_population": float(np.mean(pred_series)),
        "final_prey_population": int(prey_series[-1]),
        "final_predator_population": int(pred_series[-1]),
        "prey_population_cv": round(prey_cv, 4),
        "predator_population_cv": round(pred_cv, 4),
        "prey_oscillation_period": round(prey_period, 1),
        "predator_oscillation_period": round(pred_period, 1),
        # Stability
        "system_survival_fraction": round(survival_fraction, 4),
        "system_extinction_step": extinction_step,
        "coexistence_score_mean": round(score_mean, 4),
        "coexistence_score_min": round(score_min, 4),
        "coexistence_score_final": round(score_final, 4),
        # Demographic rates (per step)
        "prey_birth_rate": round(prey_birth_rate, 6),
        "predator_birth_rate": round(pred_birth_rate, 6),
        "predation_rate": round(predation_rate, 6),
        "prey_death_rate": round(prey_death_rate, 6),
        "predator_death_rate": round(pred_death_rate, 6),
        # Mean individual lifespan (approx)
        "mean_prey_lifespan": round(mean_prey_lifespan, 1),
        "mean_predator_lifespan": round(mean_pred_lifespan, 1),
        # Raw event totals
        "cumulative_predation_events": total_predation,
        "cumulative_prey_births": total_prey_births,
        "cumulative_predator_births": total_pred_births,
        "cumulative_prey_deaths": total_prey_deaths,
        "cumulative_predator_deaths": total_pred_deaths,
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_long_episode_experiment(
    prey_model_path: Path,
    predator_model_path: Path,
    run_name: str,
    seed_offset: int,
    long_episode_steps: int,
    eval_seed: int,
    num_prey: int,
    num_predators: int,
    map_path,
    results_dir: Path,
) -> tuple:
    """Run one long-horizon evaluation and return (per_step_df, summary_dict)."""
    prey_policy = clone_frozen_agent(prey_model_path, agent_id=0)
    predator_policy = clone_frozen_agent(predator_model_path, agent_id=1)

    tracker = EventTracker()
    records: List[Dict] = []

    with track_ecosim_events(tracker):
        env = MultiAgentEcoSimEnv(num_prey=num_prey, num_predators=num_predators, map_path=map_path)
        env.reset(seed=eval_seed + seed_offset)

        previous_alive_ids = {a.agent_id for a in env.all_agents if a.is_alive()}
        known_agent_types = {a.agent_id: a.agent_type for a in env.all_agents}
        cumulative_reward = 0.0

        for step_index in range(1, long_episode_steps + 1):
            actions: Dict[int, int] = {}
            for agent in env.all_agents:
                if not agent.is_alive():
                    continue
                policy = prey_policy if agent.agent_type == "PREY" else predator_policy
                obs = agent.get_observation(env.env)
                state = policy.discretize_state(obs)
                actions[agent.agent_id] = policy.select_action(state, training=False)

            _, reward, done, _ = env.step(actions)
            cumulative_reward += reward

            snapshot, previous_alive_ids = build_step_snapshot(
                env, step_index, reward, cumulative_reward,
                tracker, previous_alive_ids, known_agent_types,
            )
            records.append(snapshot)

            if done:
                break

    df = pd.DataFrame(records)
    df["prey_model_path"] = str(prey_model_path)
    df["predator_model_path"] = str(predator_model_path)
    df["run_name"] = run_name
    df["seed"] = eval_seed + seed_offset
    df["long_episode_steps"] = long_episode_steps

    csv_path = results_dir / f"{run_name}__{prey_model_path.stem}__{predator_model_path.stem}_{len(df)}steps.csv"
    df.to_csv(csv_path, index=False)

    derived = compute_derived_metrics(df)
    summary = {
        "run_name": run_name,
        "prey_model_path": str(prey_model_path),
        "predator_model_path": str(predator_model_path),
        "seed": eval_seed + seed_offset,
        "steps_ran": len(df),
        "csv_path": str(csv_path),
        **derived,
    }
    return df, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official long-horizon evaluation for a prey/predator checkpoint pair."
    )
    parser.add_argument("--prey-model", required=True, type=Path)
    parser.add_argument("--predator-model", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "src" / "models" / "results" / "long_horizon_evaluation")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--num-prey", type=int, default=30)
    parser.add_argument("--num-predators", type=int, default=10)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=[0, 1, 2],
                        help="Multiple seed offsets produce multiple independent runs.")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Patch STEPS_PER_EPISODE globally so the environment doesn't terminate early
    import config.config as _cfg
    import environment.gym_env as _gym
    import environment.multi_agent_gym_env as _menv
    import models.Q_learning as _ql
    for _mod in (_cfg, _gym, _menv, _ql):
        _mod.STEPS_PER_EPISODE = args.steps

    results_dir = args.output_dir / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for seed_offset in args.seed_offsets:
        run_name = f"seed{seed_offset}"
        print(f"\nRunning {args.run_name} | seed_offset={seed_offset}")
        _, summary = run_long_episode_experiment(
            prey_model_path=args.prey_model,
            predator_model_path=args.predator_model,
            run_name=run_name,
            seed_offset=seed_offset,
            long_episode_steps=args.steps,
            eval_seed=args.seed,
            num_prey=args.num_prey,
            num_predators=args.num_predators,
            map_path=None,
            results_dir=results_dir,
        )
        all_summaries.append(summary)

        print(
            f"  steps={summary['steps_ran']} | "
            f"prey={summary['final_prey_population']} | predator={summary['final_predator_population']}\n"
            f"  coexistence_mean={summary['coexistence_score_mean']:.4f} | "
            f"survival_fraction={summary['system_survival_fraction']:.3f} | "
            f"extinction_step={summary['system_extinction_step']}\n"
            f"  prey_cv={summary['prey_population_cv']:.3f} | "
            f"predator_cv={summary['predator_population_cv']:.3f} | "
            f"prey_period={summary['prey_oscillation_period']:.1f} steps\n"
            f"  prey_lifespan~{summary['mean_prey_lifespan']:.1f} steps | "
            f"pred_lifespan~{summary['mean_predator_lifespan']:.1f} steps | "
            f"predation_rate={summary['predation_rate']:.4f}/step"
        )

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = results_dir / f"{args.run_name}_all_seeds_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nAll seeds complete. Summary: {summary_csv}")
