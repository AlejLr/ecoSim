"""
Fit Lotka-Volterra and Rosenzweig-MacArthur ODE models to RL population trajectories.
Used to produce the cross-model comparison table for §5.4.

Fits to Run 50 Cycle 3 (seed9=actual seed 51, best coexistence case) and
Run 48 Cycle 3 (seed0=actual seed 0, last 1500 steps only to avoid pre-collapse transient).
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

RESULT_DIR = Path(__file__).parent / 'ode_results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RL_DIR = Path(__file__).parent.parent / 'src' / 'models' / 'results' / 'long_horizon_evaluation'


# ── ODE definitions ──────────────────────────────────────────────────────────

def lv_ode(t, y, params):
    N, P = max(y[0], 0.0), max(y[1], 0.0)
    alpha, beta, delta, gamma = params
    return [alpha * N - beta * N * P,
            delta * beta * N * P - gamma * P]

def rm_ode(t, y, params):
    N, P = max(y[0], 0.0), max(y[1], 0.0)
    r, K, a, e, d = params
    return [r * N * (1.0 - N / K) - a * N * P,
            e * a * N * P - d * P]

def simulate(ode_fn, y0, t_span, params, n_points=400):
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    try:
        sol = solve_ivp(
            lambda t, y: ode_fn(t, y, list(params)),
            t_span, list(y0), method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-10,
        )
        if sol.success:
            return sol.t, sol.y[0], sol.y[1]
    except Exception:
        pass
    nan = np.full(n_points, np.nan)
    return t_eval, nan, nan.copy()


def fit_rl(ode_fn, t_data, N_obs, P_obs, bounds, label=''):
    """Fit ODE to RL trajectory. t_data, N_obs, P_obs are 1-D numpy arrays."""
    y0 = [float(N_obs[0]), float(P_obs[0])]
    t_span = (float(t_data[0]), float(t_data[-1]))
    N_sc = np.mean(N_obs)
    P_sc = np.mean(P_obs)

    def objective(params):
        t, N, P = simulate(ode_fn, y0, t_span, params, n_points=300)
        if np.any(np.isnan(N)):
            return 1e10
        N_s = np.clip(np.interp(t_data, t, N), 0, None)
        P_s = np.clip(np.interp(t_data, t, P), 0, None)
        return (np.mean(((N_s - N_obs) / N_sc) ** 2) +
                np.mean(((P_s - P_obs) / P_sc) ** 2))

    print(f'  Fitting {label}...')
    res = differential_evolution(
        objective, bounds, seed=42, maxiter=400, tol=1e-7,
        popsize=12, mutation=(0.5, 1.5), recombination=0.7, workers=1,
    )
    params = list(res.x)

    t_sim, N_sim, P_sim = simulate(ode_fn, y0, t_span, params, n_points=len(t_data) * 4)
    N_fit = np.interp(t_data, t_sim, N_sim)
    P_fit = np.interp(t_data, t_sim, P_sim)
    rmse_N = float(np.sqrt(np.mean((N_fit - N_obs) ** 2)))
    rmse_P = float(np.sqrt(np.mean((P_fit - P_obs) ** 2)))
    return params, rmse_N, rmse_P


def lv_equilibrium(params):
    alpha, beta, delta, gamma = params
    N_eq = gamma / (delta * beta)
    P_eq = alpha / beta
    return N_eq, P_eq

def rm_equilibrium(params):
    r, K, a, e, d = params
    N_eq = d / (e * a)
    P_eq = r / a * (1.0 - N_eq / K) if N_eq < K else 0.0
    return N_eq, P_eq

def lv_jacobian(N_eq, P_eq, p):
    alpha, beta, delta, gamma = p
    return np.array([[alpha - beta*P_eq,   -beta*N_eq],
                     [delta*beta*P_eq,      delta*beta*N_eq - gamma]])

def rm_jacobian(N_eq, P_eq, p):
    r, K, a, e, d = p
    return np.array([[r*(1 - 2*N_eq/K) - a*P_eq,  -a*N_eq],
                     [e*a*P_eq,                     e*a*N_eq - d]])


def load_rl_trajectory(filepath, start_step=1, stride=10):
    """Load per-step CSV, subsample, return numpy arrays (t, N, P)."""
    df = pd.read_csv(filepath)
    df = df[df['step'] >= start_step].copy()
    df = df.iloc[::stride].reset_index(drop=True)
    t = df['step'].values.astype(float) - df['step'].values[0]
    N = df['prey_population'].values.astype(float)
    P = df['predator_population'].values.astype(float)
    return t, N, P


# ── Run 50 Cycle 3 — seed9 (actual seed 51): best coexistence ────────────────

print('\n=== Run 50 Cycle 3 — seed9 (actual seed 51, best case) ===')
f50 = RL_DIR / 'run50' / 'cycle3_seed9__trained_prey_50_protocol2_cycle3__trained_predator_50_protocol2_cycle3_2000steps.csv'

# Use full trajectory but skip early transient (steps 1-300); stride=10
t50, N50, P50 = load_rl_trajectory(f50, start_step=300, stride=10)
print(f'  Trajectory: {len(t50)} points, prey {N50.min():.0f}–{N50.max():.0f}, pred {P50.min():.0f}–{P50.max():.0f}')
print(f'  Mean prey: {N50.mean():.1f}, mean pred: {P50.mean():.1f}')

# Bounds tuned for agent-count scale (not thousands)
lv_bounds_50 = [(0.001, 0.5), (1e-5, 0.01), (0.001, 1.0), (0.001, 0.5)]
rm_bounds_50 = [(0.001, 0.5), (200.0, 700.0), (1e-5, 0.01), (0.001, 1.0), (0.001, 0.5)]

lv_p50, lv_rN50, lv_rP50 = fit_rl(lv_ode, t50, N50, P50, lv_bounds_50, 'LV to Run50c3')
lv_N50, lv_P50 = lv_equilibrium(lv_p50)
eigs_lv50 = np.linalg.eigvals(lv_jacobian(lv_N50, lv_P50, lv_p50))
print(f'\n  LV: alpha={lv_p50[0]:.6f} beta={lv_p50[1]:.8f} delta={lv_p50[2]:.6f} gamma={lv_p50[3]:.6f}')
print(f'      RMSE prey={lv_rN50:.2f}  pred={lv_rP50:.2f}')
print(f'      N*={lv_N50:.1f}  P*={lv_P50:.1f}')
print(f'      eigenvalues = {eigs_lv50[0]:.5f}, {eigs_lv50[1]:.5f}')

rm_p50, rm_rN50, rm_rP50 = fit_rl(rm_ode, t50, N50, P50, rm_bounds_50, 'RM to Run50c3')
rm_N50, rm_P50 = rm_equilibrium(rm_p50)
eigs_rm50 = np.linalg.eigvals(rm_jacobian(rm_N50, rm_P50, rm_p50))
print(f'\n  RM: r={rm_p50[0]:.6f} K={rm_p50[1]:.1f} a={rm_p50[2]:.8f} e={rm_p50[3]:.6f} d={rm_p50[4]:.6f}')
print(f'      RMSE prey={rm_rN50:.2f}  pred={rm_rP50:.2f}')
print(f'      N*={rm_N50:.1f}  P*={rm_P50:.1f}')
print(f'      eigenvalues = {eigs_rm50[0]:.5f}, {eigs_rm50[1]:.5f}')

# Save immediately so results aren't lost
results_50 = {
    'run50_cycle3_seed9': {
        'trajectory_points': len(t50), 'start_step': 300, 'stride': 10,
        'mean_prey': float(N50.mean()), 'mean_pred': float(P50.mean()),
        'lv': {'params': dict(zip(['alpha','beta','delta','gamma'], lv_p50)),
               'rmse_prey': lv_rN50, 'rmse_pred': lv_rP50,
               'N_star': lv_N50, 'P_star': lv_P50,
               'eigenvalues': [str(eigs_lv50[0]), str(eigs_lv50[1])]},
        'rm': {'params': dict(zip(['r','K','a','e','d'], rm_p50)),
               'rmse_prey': rm_rN50, 'rmse_pred': rm_rP50,
               'N_star': rm_N50, 'P_star': rm_P50,
               'eigenvalues': [str(eigs_rm50[0]), str(eigs_rm50[1])]},
    }
}
with open(RESULT_DIR / 'rl_fitted_parameters.json', 'w') as f:
    json.dump(results_50, f, indent=2)
print('\n  Saved to ode_results/rl_fitted_parameters.json')


# ── Run 48 Cycle 3 — seed0: before extinction ─────────────────────────────────

print('\n=== Run 48 Cycle 3 — seed0 (pre-extinction phase) ===')
f48 = RL_DIR / 'run48' / 'cycle3_seed0__trained_prey_48_protocol2_cycle3__trained_predator_48_protocol2_cycle3_2000steps.csv'

if f48.exists():
    df48 = pd.read_csv(f48)
    print(f'  File loaded, {len(df48)} steps')
    # Find first extinction step
    extinct = df48[df48['system_extinct'] == True]
    if len(extinct) > 0:
        ext_step = extinct['step'].values[0]
        print(f'  Extinction at step {ext_step}')
        # Fit to only the coexistence window
        t48, N48, P48 = load_rl_trajectory(f48, start_step=1, stride=10)
        t48 = t48[t48 < ext_step - 1]
        N48 = N48[:len(t48)]
        P48 = P48[:len(t48)]
    else:
        t48, N48, P48 = load_rl_trajectory(f48, start_step=1, stride=10)

    print(f'  Fitting window: {len(t48)} points, prey {N48.min():.0f}–{N48.max():.0f}, pred {P48.min():.0f}–{P48.max():.0f}')

    lv_bounds_48 = [(0.001, 0.5), (1e-5, 0.01), (0.001, 1.0), (0.001, 0.5)]
    rm_bounds_48 = [(0.001, 0.5), (200.0, 1500.0), (1e-5, 0.01), (0.001, 1.0), (0.001, 0.5)]

    lv_p48, lv_rN48, lv_rP48 = fit_rl(lv_ode, t48, N48, P48, lv_bounds_48, 'LV to Run48c3')
    rm_p48, rm_rN48, rm_rP48 = fit_rl(rm_ode, t48, N48, P48, rm_bounds_48, 'RM to Run48c3')

    lv_N48, lv_P48 = lv_equilibrium(lv_p48)
    rm_N48, rm_P48 = rm_equilibrium(rm_p48)

    eigs_lv48 = np.linalg.eigvals(lv_jacobian(lv_N48, lv_P48, lv_p48))
    eigs_rm48 = np.linalg.eigvals(rm_jacobian(rm_N48, rm_P48, rm_p48))

    print(f'\n  LV: alpha={lv_p48[0]:.6f} beta={lv_p48[1]:.8f} delta={lv_p48[2]:.6f} gamma={lv_p48[3]:.6f}')
    print(f'      RMSE prey={lv_rN48:.2f}  pred={lv_rP48:.2f}')
    print(f'      N*={lv_N48:.1f}  P*={lv_P48:.1f}')
    print(f'      eigenvalues = {eigs_lv48[0]:.5f}, {eigs_lv48[1]:.5f}')

    print(f'\n  RM: r={rm_p48[0]:.6f} K={rm_p48[1]:.1f} a={rm_p48[2]:.8f} e={rm_p48[3]:.6f} d={rm_p48[4]:.6f}')
    print(f'      RMSE prey={rm_rN48:.2f}  pred={rm_rP48:.2f}')
    print(f'      N*={rm_N48:.1f}  P*={rm_P48:.1f}')
    print(f'      eigenvalues = {eigs_rm48[0]:.5f}, {eigs_rm48[1]:.5f}')
else:
    print(f'  File not found: {f48}')

# ── Summary table ─────────────────────────────────────────────────────────────

print('\n=== Summary ===')
print(f'{"Checkpoint":<20} {"Model":<5} {"RMSE prey":>10} {"RMSE pred":>10} {"N*":>8} {"P*":>8} {"Stability":>20}')
print('-' * 90)

def stability_label(eigs):
    re = np.real(eigs)
    im = np.imag(eigs)
    if np.all(np.abs(re) < 1e-6):
        return 'Neutral centre'
    elif np.all(re < 0) and np.any(np.abs(im) > 1e-6):
        return 'Stable spiral'
    elif np.all(re < 0):
        return 'Stable node'
    else:
        return 'Unstable'

print(f'{"Run50c3 seed9":<20} {"LV":<5} {lv_rN50:>10.2f} {lv_rP50:>10.2f} {lv_N50:>8.1f} {lv_P50:>8.1f} {stability_label(eigs_lv50):>20}')
print(f'{"Run50c3 seed9":<20} {"RM":<5} {rm_rN50:>10.2f} {rm_rP50:>10.2f} {rm_N50:>8.1f} {rm_P50:>8.1f} {stability_label(eigs_rm50):>20}')

if f48.exists():
    print(f'{"Run48c3 seed0":<20} {"LV":<5} {lv_rN48:>10.2f} {lv_rP48:>10.2f} {lv_N48:>8.1f} {lv_P48:>8.1f} {stability_label(eigs_lv48):>20}')
    print(f'{"Run48c3 seed0":<20} {"RM":<5} {rm_rN48:>10.2f} {rm_rP48:>10.2f} {rm_N48:>8.1f} {rm_P48:>8.1f} {stability_label(eigs_rm48):>20}')

print('\nDone.')
