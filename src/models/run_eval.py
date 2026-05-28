"""Generic long-horizon evaluator — works for any run number and any cycles.

Usage
-----
    # Evaluate all completed cycles for run 47:
    python -m src.models.run_eval 47

    # Evaluate a specific cycle only:
    python -m src.models.run_eval 47 --cycle 2

    # Evaluate specific cycles:
    python -m src.models.run_eval 47 --cycles 2 3

    # Override defaults:
    python -m src.models.run_eval 47 --steps 2000 --seeds 0 1 2 --num-pred 10 --num-prey 30

Output
------
    src/models/results/long_horizon_evaluation/run<N>/run<N>_all_cycles_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from config.config import SEED
from models.evaluate_checkpoint_pair import run_long_episode_experiment

MODELS_DIR  = ROOT / "src" / "models"


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-horizon evaluator for any run/cycle pair.")
    p.add_argument("run", type=int, help="Run number (e.g. 47)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--cycle",  type=int,        help="Evaluate a single cycle")
    g.add_argument("--cycles", type=int, nargs="+", help="Evaluate specific cycles")
    p.add_argument("--max-cycles", type=int, default=5,
                   help="Scan up to this many cycles (default 5); ignored if --cycle/--cycles set")
    p.add_argument("--steps",    type=int,        default=2000)
    p.add_argument("--seeds",    type=int, nargs="+", default=[0, 1, 2],
                   help="Seed offsets (added to global SEED=42)")
    p.add_argument("--num-prey", type=int,        default=30)
    p.add_argument("--num-pred", type=int,        default=10)
    p.add_argument("--results-dir", type=Path,    default=None)
    return p.parse_args()


def main():
    args = _parse()

    # Determine which cycles to evaluate
    if args.cycle is not None:
        cycles = [args.cycle]
    elif args.cycles is not None:
        cycles = args.cycles
    else:
        cycles = list(range(1, args.max_cycles + 1))

    results_dir = args.results_dir or (
        ROOT / "src" / "models" / "results" / "long_horizon_evaluation" / f"run{args.run}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Patch global step cap so the environment doesn't terminate early
    import config.config as _cfg
    import environment.gym_env as _gym
    import environment.multi_agent_gym_env as _menv
    import models.Q_learning as _ql
    for _mod in (_cfg, _gym, _menv, _ql):
        _mod.STEPS_PER_EPISODE = args.steps

    all_summaries = []

    for cycle_id in cycles:
        prey_path = MODELS_DIR / f"trained_prey_{args.run}_protocol2_cycle{cycle_id}.pkl"
        pred_path = MODELS_DIR / f"trained_predator_{args.run}_protocol2_cycle{cycle_id}.pkl"

        if not prey_path.exists():
            print(f"Skipping cycle {cycle_id}: {prey_path.name} not found")
            continue
        if not pred_path.exists():
            print(f"Skipping cycle {cycle_id}: {pred_path.name} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Run {args.run}  Cycle {cycle_id}")
        print(f"{'='*60}")

        for seed_offset in args.seeds:
            run_name = f"cycle{cycle_id}_seed{seed_offset}"
            print(f"  [{run_name}] running …")

            _, summary = run_long_episode_experiment(
                prey_model_path=prey_path,
                predator_model_path=pred_path,
                run_name=run_name,
                seed_offset=seed_offset,
                long_episode_steps=args.steps,
                eval_seed=SEED,
                num_prey=args.num_prey,
                num_predators=args.num_pred,
                map_path=None,
                results_dir=results_dir,
            )
            all_summaries.append({"cycle": cycle_id, **summary})

            print(
                f"  steps={summary['steps_ran']} | "
                f"prey={summary['final_prey_population']} pred={summary['final_predator_population']}\n"
                f"  coex={summary['coexistence_score_mean']:.4f} | "
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
        print("\nNo complete cycle pairs found.")
        return

    # Merge with any existing summary so reruns don't wipe previous cycles
    summary_csv = results_dir / f"run{args.run}_all_cycles_summary.csv"
    new_df = pd.DataFrame(all_summaries)

    if summary_csv.exists():
        existing = pd.read_csv(summary_csv)
        evaluated_cycles = set(new_df["cycle"].unique())
        existing = existing[~existing["cycle"].isin(evaluated_cycles)]
        combined = pd.concat([existing, new_df], ignore_index=True).sort_values(
            ["cycle", "seed"]
        )
    else:
        combined = new_df

    combined.to_csv(summary_csv, index=False)
    print(f"\nSummary written -> {summary_csv}")

    # Cross-cycle aggregate
    print(f"\n{'='*60}")
    print("Cross-cycle summary (mean across seeds):")
    print(f"{'='*60}")
    for cycle_id in sorted(combined["cycle"].unique()):
        sub = combined[combined["cycle"] == cycle_id]
        print(
            f"  Cycle {cycle_id}: "
            f"coex={sub['coexistence_score_mean'].mean():.4f}±{sub['coexistence_score_mean'].std():.4f} | "
            f"survival={sub['system_survival_fraction'].mean():.3f} | "
            f"prey_cv={sub['prey_population_cv'].mean():.3f} | "
            f"pred_cv={sub['predator_population_cv'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
