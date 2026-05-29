"""Visualization tool for a long-horizon evaluation run.

Generates:
  animation.gif        — 20-second flythrough at 10 FPS (frame every 10 steps)
  density_prey.png     — spatial heatmap of cumulative prey presence
  density_predator.png — spatial heatmap of cumulative predator presence
  density_kills.png    — spatial heatmap of prey death locations
  population.png       — time series, phase portrait, energy distributions
  summary_panel.png    — compact single-figure overview

Usage
-----
    # Auto-locate PKLs from run/cycle numbers:
    python -m src.models.run_viz_eval --run 47 --cycle 1 --seed 0

    # Or pass PKL paths directly:
    python -m src.models.run_viz_eval \\
        --prey-model src/models/trained_prey_47_protocol2_cycle1.pkl \\
        --predator-model src/models/trained_predator_47_protocol2_cycle1.pkl \\
        --seed 0
"""
from __future__ import annotations

import argparse
import sys
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from agents.agent import Predator, Prey
from config.config import SEED
from environment.multi_agent_gym_env import MultiAgentEcoSimEnv
from models.Q_learning import QLearningAgent
from models.coexistence_metrics import coexistence_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LONG_STEPS   = 2000
NUM_PREY     = 30
NUM_PRED     = 10
GIF_FPS      = 10          # frames per second in output GIF
GIF_DURATION = 20          # seconds
GRID_SIZE    = 150

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_frozen(path: Path) -> QLearningAgent:
    agent = QLearningAgent.load_model_from_file(str(path))
    agent.epsilon = 0.0
    return agent

# ---------------------------------------------------------------------------
# Kill-position tracker (monkey-patch Predator.eat)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def track_kill_positions(kill_list: list):
    """Append (x, y) of killed prey to kill_list by patching Predator.eat."""
    original_eat = Predator.eat

    def wrapped_eat(self, *args, **kwargs):
        reward = original_eat(self, *args, **kwargs)
        if reward > 0:
            kill_list.append(self.position)
        return reward

    Predator.eat = wrapped_eat
    try:
        yield
    finally:
        Predator.eat = original_eat

# ---------------------------------------------------------------------------
# Single-frame renderer
# ---------------------------------------------------------------------------

def render_frame(env, step: int, prey_counts: list, pred_counts: list,
                 grid_size: int = GRID_SIZE) -> np.ndarray:
    """Render current env state -> RGB numpy array (H, W, 3)."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.invert_yaxis()
    ax.axis("off")

    # Grass overlay (sampled, best-effort — skip silently if structure differs)
    try:
        tiles = env.env.tiles
        gx, gy = [], []
        for x in range(0, grid_size, 4):
            for y in range(0, grid_size, 4):
                tile = tiles[x][y]
                if getattr(tile, "tile_type", None) == "grass":
                    gx.append(x)
                    gy.append(y)
        if gx:
            ax.scatter(gx, gy, c="#2d6a2d", s=8, marker="s",
                       alpha=0.25, edgecolors="none")
    except Exception:
        pass

    # Agents
    prey_x, prey_y, pred_x, pred_y = [], [], [], []
    for a in env.all_agents:
        if not a.is_alive():
            continue
        if a.agent_type == "PREY":
            prey_x.append(a.position[0]); prey_y.append(a.position[1])
        else:
            pred_x.append(a.position[0]); pred_y.append(a.position[1])

    if prey_x:
        ax.scatter(prey_x, prey_y, c="#00e676", s=30, marker="^",
                   edgecolors="none", alpha=0.9, zorder=5)
    if pred_x:
        ax.scatter(pred_x, pred_y, c="#ff1744", s=40, marker="v",
                   edgecolors="none", alpha=0.9, zorder=5)

    # HUD
    ax.text(0.02, 0.97,
            f"Step {step:4d}   Prey {len(prey_x):3d}   Pred {len(pred_x):3d}",
            transform=ax.transAxes, fontsize=8, color="white",
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]  # RGBA -> RGB
    plt.close(fig)
    return img

# ---------------------------------------------------------------------------
# Main evaluation + visualization loop
# ---------------------------------------------------------------------------

def run_viz(prey_path: Path, pred_path: Path, seed: int,
            steps: int, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {output_dir}")

    prey_policy = load_frozen(prey_path)
    pred_policy = load_frozen(pred_path)

    # How often to capture a frame
    total_frames = GIF_FPS * GIF_DURATION   # 200 frames
    frame_every  = max(1, steps // total_frames)

    # Accumulators
    prey_counts:  list[int]   = []
    pred_counts:  list[int]   = []
    avg_prey_e:   list[float] = []
    avg_pred_e:   list[float] = []
    prey_pos_all: list        = []   # (x, y) of all living prey at each step
    pred_pos_all: list        = []
    kill_pos:     list        = []   # (x, y) of predation events
    gif_frames:   list        = []

    # Patch global step cap
    import config.config as _cfg
    import environment.gym_env as _gym
    import environment.multi_agent_gym_env as _menv
    import models.Q_learning as _ql
    for _mod in (_cfg, _gym, _menv, _ql):
        _mod.STEPS_PER_EPISODE = steps

    with track_kill_positions(kill_pos):
        env = MultiAgentEcoSimEnv(num_prey=NUM_PREY, num_predators=NUM_PRED)
        env.reset(seed=seed)

        print(f"Running {steps} steps …")
        for step in range(1, steps + 1):
            # Action selection
            actions: dict[int, int] = {}
            for agent in env.all_agents:
                if not agent.is_alive():
                    continue
                policy = prey_policy if agent.agent_type == "PREY" else pred_policy
                obs    = agent.get_observation(env.env)
                state  = policy.discretize_state(obs)
                actions[agent.agent_id] = policy.select_action(state, training=False)

            _, _, done, _ = env.step(actions)

            # Count alive agents
            alive = [a for a in env.all_agents if a.is_alive()]
            prey  = [a for a in alive if a.agent_type == "PREY"]
            pred  = [a for a in alive if a.agent_type == "PREDATOR"]

            prey_counts.append(len(prey))
            pred_counts.append(len(pred))
            avg_prey_e.append(float(np.mean([a.energy for a in prey]))  if prey else 0.0)
            avg_pred_e.append(float(np.mean([a.energy for a in pred]))  if pred else 0.0)

            # Spatial accumulation — position is a (x, y) tuple
            prey_pos_all.extend(a.position for a in prey)
            pred_pos_all.extend(a.position for a in pred)

            # GIF frame
            if step % frame_every == 0:
                frame = render_frame(env, step, prey_counts, pred_counts)
                gif_frames.append(frame)
                if len(gif_frames) % 20 == 0:
                    print(f"  frame {len(gif_frames)}/{total_frames}  step {step}")

            if done:
                print(f"  Episode ended at step {step}")
                break

    steps_ran = len(prey_counts)
    print(f"Done — {steps_ran} steps, {len(gif_frames)} frames, "
          f"{len(kill_pos)} predation events.")

    # -----------------------------------------------------------------------
    # 1. GIF
    # -----------------------------------------------------------------------
    gif_path = output_dir / "animation.gif"
    _save_gif(gif_frames, gif_path, fps=GIF_FPS)
    print(f"Saved: {gif_path}")

    # -----------------------------------------------------------------------
    # 2. Density maps
    # -----------------------------------------------------------------------
    _save_density_maps(prey_pos_all, pred_pos_all, kill_pos, output_dir)
    print(f"Saved: density maps")

    # -----------------------------------------------------------------------
    # 3. Population dynamics (time series + phase portrait + energy)
    # -----------------------------------------------------------------------
    _save_population_plot(prey_counts, pred_counts, avg_prey_e, avg_pred_e, output_dir)
    print(f"Saved: population.png")

    # -----------------------------------------------------------------------
    # 4. Summary panel
    # -----------------------------------------------------------------------
    _save_summary_panel(prey_counts, pred_counts, avg_prey_e, avg_pred_e,
                        prey_pos_all, pred_pos_all, kill_pos, output_dir)
    print(f"Saved: summary_panel.png")
    print("All done.")


# ---------------------------------------------------------------------------
# GIF writer
# ---------------------------------------------------------------------------

def _save_gif(frames: list, path: Path, fps: int):
    if not frames:
        print("  No frames to write.")
        return
    try:
        import imageio
        with imageio.get_writer(str(path), mode="I", fps=fps, loop=0) as writer:
            for f in frames:
                writer.append_data(f)
    except ImportError:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / fps)
        imgs[0].save(str(path), save_all=True, append_images=imgs[1:],
                     duration=duration_ms, loop=0, optimize=True)


# ---------------------------------------------------------------------------
# Density maps
# ---------------------------------------------------------------------------

def _save_density_maps(prey_pos, pred_pos, kill_pos, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _heatmap(ax, positions, title, cmap):
        if positions:
            xs, ys = zip(*positions)
            h, _, _ = np.histogram2d(xs, ys,
                                     bins=GRID_SIZE // 3,
                                     range=[[0, GRID_SIZE], [0, GRID_SIZE]])
            im = ax.imshow(h.T, origin="upper", cmap=cmap,
                           norm=LogNorm(vmin=max(1, h[h > 0].min()), vmax=h.max()),
                           extent=[0, GRID_SIZE, GRID_SIZE, 0])
            plt.colorbar(im, ax=ax, label="log(visits)")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("X"); ax.set_ylabel("Y")

    _heatmap(axes[0], prey_pos,  "Prey presence density",      "Greens")
    _heatmap(axes[1], pred_pos,  "Predator presence density",  "Reds")
    _heatmap(axes[2], kill_pos,  "Predation event locations",  "Oranges")

    plt.tight_layout()
    fig.savefig(out / "density_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Population dynamics
# ---------------------------------------------------------------------------

def _save_population_plot(prey_c, pred_c, prey_e, pred_e, out: Path):
    steps = np.arange(1, len(prey_c) + 1)

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Top-left: population time series ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, prey_c, color="#2ecc71", lw=1.5, label="Prey")
    ax1b = ax1.twinx()
    ax1b.plot(steps, pred_c, color="#e74c3c", lw=1.5, label="Predator")
    ax1b.set_ylabel("Predator count", color="#e74c3c")
    ax1b.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_ylabel("Prey count", color="#2ecc71")
    ax1.tick_params(axis="y", labelcolor="#2ecc71")
    ax1.set_xlabel("Step")
    ax1.set_title("Population dynamics")
    l1, ll1 = ax1.get_legend_handles_labels()
    l2, ll2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, ll1 + ll2, fontsize=8)

    # ── Top-right: phase portrait coloured by time ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(prey_c, pred_c, c=steps, cmap="viridis",
                     s=4, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax2, label="Step")
    ax2.scatter([prey_c[0]],  [pred_c[0]],  color="green", s=80, zorder=6,
                label="Start")
    ax2.scatter([prey_c[-1]], [pred_c[-1]], color="red",   s=80, zorder=6,
                label="End")
    ax2.set_xlabel("Prey"); ax2.set_ylabel("Predator")
    ax2.set_title("Phase portrait (colour = time)")
    ax2.legend(fontsize=8)

    # ── Bottom-left: mean energy over time ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, prey_e, color="#2ecc71", lw=1.2, label="Avg prey energy")
    ax3.plot(steps, pred_e, color="#e74c3c", lw=1.2, label="Avg predator energy")
    ax3.axhline(55, color="#2ecc71", ls=":", lw=1, alpha=0.6,
                label="Prey repro threshold (55)")
    ax3.axhline(70, color="#e74c3c", ls=":", lw=1, alpha=0.6,
                label="Pred repro threshold (70)")
    ax3.set_xlabel("Step"); ax3.set_ylabel("Mean energy")
    ax3.set_title("Mean energy per species")
    ax3.legend(fontsize=7)

    # ── Bottom-right: ratio over time ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ratio = [p / max(d, 1) for p, d in zip(prey_c, pred_c)]
    ax4.plot(steps, ratio, color="#9b59b6", lw=1.2)
    ax4.axhline(2.0, color="gray", ls="--", lw=1, alpha=0.7,
                label="Target ratio 2:1")
    ax4.set_xlabel("Step"); ax4.set_ylabel("Prey / Predator")
    ax4.set_title("Prey-to-predator ratio")
    ax4.legend(fontsize=8)

    fig.savefig(out / "population.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary panel (compact, thesis-ready)
# ---------------------------------------------------------------------------

def _save_summary_panel(prey_c, pred_c, prey_e, pred_e,
                         prey_pos, pred_pos, kill_pos, out: Path):
    steps = np.arange(1, len(prey_c) + 1)
    fig   = plt.figure(figsize=(18, 5))
    gs    = GridSpec(1, 4, figure=fig, wspace=0.35)

    # Population time series (dual axis)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, prey_c, color="#2ecc71", lw=1.5, label="Prey")
    ax1b = ax1.twinx()
    ax1b.plot(steps, pred_c, color="#e74c3c", lw=1.5, label="Pred")
    ax1b.set_ylabel("Predator", color="#e74c3c", fontsize=8)
    ax1b.tick_params(axis="y", labelcolor="#e74c3c", labelsize=7)
    ax1.set_ylabel("Prey", color="#2ecc71", fontsize=8)
    ax1.tick_params(axis="y", labelcolor="#2ecc71", labelsize=7)
    ax1.set_xlabel("Step", fontsize=8); ax1.set_title("Population", fontsize=9)
    l1, ll1 = ax1.get_legend_handles_labels()
    l2, ll2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, ll1 + ll2, fontsize=7)

    # Phase portrait
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(prey_c, pred_c, c=steps, cmap="plasma", s=3, alpha=0.5)
    plt.colorbar(sc, ax=ax2, label="step", shrink=0.8)
    ax2.set_xlabel("Prey", fontsize=8); ax2.set_ylabel("Predator", fontsize=8)
    ax2.set_title("Phase portrait", fontsize=9)

    # Prey density
    ax3 = fig.add_subplot(gs[0, 2])
    if prey_pos:
        xs, ys = zip(*prey_pos)
        h, _, _ = np.histogram2d(xs, ys, bins=50,
                                 range=[[0, GRID_SIZE], [0, GRID_SIZE]])
        ax3.imshow(h.T, origin="upper", cmap="Greens",
                   norm=LogNorm(vmin=max(1, h[h > 0].min()), vmax=h.max()),
                   extent=[0, GRID_SIZE, GRID_SIZE, 0])
    ax3.set_title("Prey density", fontsize=9)
    ax3.set_xlabel("X", fontsize=8); ax3.set_ylabel("Y", fontsize=8)

    # Predator density
    ax4 = fig.add_subplot(gs[0, 3])
    if pred_pos:
        xs, ys = zip(*pred_pos)
        h, _, _ = np.histogram2d(xs, ys, bins=50,
                                 range=[[0, GRID_SIZE], [0, GRID_SIZE]])
        ax4.imshow(h.T, origin="upper", cmap="Reds",
                   norm=LogNorm(vmin=max(1, h[h > 0].min()), vmax=h.max()),
                   extent=[0, GRID_SIZE, GRID_SIZE, 0])
    ax4.set_title("Predator density", fontsize=9)
    ax4.set_xlabel("X", fontsize=8); ax4.set_ylabel("Y", fontsize=8)

    fig.savefig(out / "summary_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run",   type=int, help="Run number (auto-locates PKLs)")
    p.add_argument("--cycle", type=int, default=1)
    p.add_argument("--prey-model",      type=Path, default=None)
    p.add_argument("--predator-model",  type=Path, default=None)
    p.add_argument("--seed",  type=int, default=SEED)
    p.add_argument("--steps", type=int, default=LONG_STEPS)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    models_dir = ROOT / "src" / "models"

    if args.run is not None:
        prey_path = models_dir / f"trained_prey_{args.run}_protocol2_cycle{args.cycle}.pkl"
        pred_path = models_dir / f"trained_predator_{args.run}_protocol2_cycle{args.cycle}.pkl"
        tag = f"run{args.run}_cycle{args.cycle}_seed{args.seed}"
    else:
        if args.prey_model is None or args.predator_model is None:
            raise SystemExit("Provide --run or both --prey-model and --predator-model")
        prey_path = args.prey_model
        pred_path = args.predator_model
        tag = f"{prey_path.stem}__seed{args.seed}"

    for p in (prey_path, pred_path):
        if not p.exists():
            raise SystemExit(f"Model not found: {p}")

    out = args.output_dir or (ROOT / "src" / "models" / "results" / "visualizations" / tag)

    run_viz(prey_path, pred_path, seed=args.seed, steps=args.steps, output_dir=out)
