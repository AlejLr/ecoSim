"""Long-horizon evaluation for run 47 (Protocol 2, 5 cycles).

Evaluates every completed cycle pair (both prey and predator .pkl must exist).
Imports all evaluation infrastructure from evaluate_checkpoint_pair — that module
is the single authoritative evaluator for this thesis.

Usage
-----
    python -m src.models.run_eval_run47

Run after each Protocol 2 cycle completes, or after all cycles finish:
    python -m src.models.run_training_protocol 2 5 350

Output is written to:
    src/models/results/long_horizon_evaluation/run47/
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from config.config import SEED
from models.evaluate_checkpoint_pair import run_long_episode_experiment

# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------
RUN_NUMBER = 47
NUM_CYCLES = 5
SEED_OFFSETS = [0, 1, 2]
LONG_EPISODE_STEPS = 2000
NUM_PREY = 30
NUM_PREDATORS = 10

MODELS_DIR = ROOT / "src" / "models"
RESULTS_DIR = ROOT / "src" / "models" / "results" / "long_horizon_evaluation" / f"run{RUN_NUMBER}"


def main():
    # Patch global step cap so environments don't terminate early
    import config.config as _cfg
    import environment.gym_env as _gym
    import environment.multi_agent_gym_env as _menv
    import models.Q_learning as _ql
    for _mod in (_cfg, _gym, _menv, _ql):
        _mod.STEPS_PER_EPISODE = LONG_EPISODE_STEPS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for cycle_id in range(1, NUM_CYCLES + 1):
        prey_path = MODELS_DIR / f"trained_prey_{RUN_NUMBER}_protocol2_cycle{cycle_id}.pkl"
        pred_path = MODELS_DIR / f"trained_predator_{RUN_NUMBER}_protocol2_cycle{cycle_id}.pkl"

        if not prey_path.exists():
            print(f"Skipping cycle {cycle_id}: prey model not found ({prey_path.name})")
            continue
        if not pred_path.exists():
            print(f"Skipping cycle {cycle_id}: predator model not found ({pred_path.name})")
            continue

        print(f"\n{'='*60}")
        print(f"Cycle {cycle_id} — {prey_path.name} / {pred_path.name}")
        print(f"{'='*60}")

        for seed_offset in SEED_OFFSETS:
            run_name = f"cycle{cycle_id}_seed{seed_offset}"
            print(f"  [{run_name}] running...")
            _, summary = run_long_episode_experiment(
                prey_model_path=prey_path,
                predator_model_path=pred_path,
                run_name=run_name,
                seed_offset=seed_offset,
                long_episode_steps=LONG_EPISODE_STEPS,
                eval_seed=SEED,
                num_prey=NUM_PREY,
                num_predators=NUM_PREDATORS,
                map_path=None,
                results_dir=RESULTS_DIR,
            )
            all_summaries.append({"cycle": cycle_id, **summary})

            print(
                f"  steps={summary['steps_ran']} | "
                f"prey={summary['final_prey_population']} pred={summary['final_predator_population']}\n"
                f"  coex_mean={summary['coexistence_score_mean']:.4f} | "
                f"survival={summary['system_survival_fraction']:.3f} | "
                f"extinct_step={summary['system_extinction_step']}\n"
                f"  prey_cv={summary['prey_population_cv']:.3f} | "
                f"pred_cv={summary['predator_population_cv']:.3f} | "
                f"period={summary['prey_oscillation_period']:.1f} steps\n"
                f"  prey_lifespan~{summary['mean_prey_lifespan']:.1f} | "
                f"pred_lifespan~{summary['mean_predator_lifespan']:.1f} | "
                f"predation={summary['predation_rate']:.4f}/step"
            )

    if not all_summaries:
        print("\nNo complete cycle pairs found. Is training still running?")
        return

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = RESULTS_DIR / f"run{RUN_NUMBER}_all_cycles_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nAll evaluations complete. Summary: {summary_csv}")

    # Print aggregate across cycles for quick thesis reference
    print(f"\n{'='*60}")
    print("Cross-cycle aggregates (mean ± std across seeds per cycle):")
    print(f"{'='*60}")
    for cycle_id in summary_df["cycle"].unique():
        sub = summary_df[summary_df["cycle"] == cycle_id]
        print(
            f"  Cycle {cycle_id}: "
            f"coex={sub['coexistence_score_mean'].mean():.4f}±{sub['coexistence_score_mean'].std():.4f} | "
            f"survival={sub['system_survival_fraction'].mean():.3f} | "
            f"prey_cv={sub['prey_population_cv'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
