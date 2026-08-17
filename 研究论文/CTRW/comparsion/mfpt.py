# MFPT experiment: SSA (space-discrete, improved Q_u) vs Tamed EM vs Truncated EM
# SDE: dX_t = -X_t^3 dt + sqrt(2) dW_t
# We compare |MFPT - t^e| and error ratios across large x.
#
# Notes:
# - Space-discrete SSA uses improved Q_u (Zu's modification):
#     Rates at location x with grid step h:
#       M = max( 0.5*(sigma^2 - |mu(x)|*h), 0 )
#       q_left  = max(-mu(x), 0)/h + M/h**2
#       q_right = max( mu(x), 0)/h + M/h**2
#   For mu(x) = -x^3 and sigma = sqrt(2), 0.5*sigma^2 = 1, so M = max(1 - 0.5*|mu|*h, 0)
# - Tamed EM:
#     X_{n+1} = X_n + mu(X_n)*dt / (1 + |mu(X_n)|*dt) + sigma*sqrt(dt)*N(0,1)
# - Truncated EM (Mao-style truncation via state clipping):
#     r_dt = C * dt^(-alpha); clip state: x_clip = clip(X_n, -r_dt, r_dt); mu_trunc = mu(x_clip)
#     X_{n+1} = X_n + mu_trunc*dt + sigma*sqrt(dt)*N(0,1)
#
# Crossing detection:
# - We stop when X <= L = x0 - delta; if the last Euler step crosses the level, use linear interpolation
#   to estimate the crossing time fraction within the step.
#
# Plots:
# - Absolute error vs x
# - Error ratio (time / space) vs x
#
# This script is moderately sized to run within the notebook constraints.
# Increase N paths for stronger statistical power as needed.

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict

rng = np.random.default_rng(12345)

# Problem setup
sigma = math.sqrt(2.0)

def mu(x: float) -> float:
    return -x**3

def t_e(x0: float, delta: float) -> float:
    L = x0 - delta
    # ∫_{L}^{x0} s^{-3} ds = (1/(2 L^2) - 1/(2 x0^2))
    return 0.5*(1.0/(L**2) - 1.0/(x0**2))

# ------------------ Space-discrete SSA with improved Q_u ------------------
def ssa_improved_Qu_MFPT(x0: float, delta: float, h: float, rng: np.random.Generator) -> float:
    """
    Simulate one MFPT sample using improved Q_u SSA on a uniform grid with spacing h.
    State space: {..., x0-2h, x0-h, x0, x0+h, ...}
    Target: L = x0 - delta (assumed to be aligned to grid, i.e., delta/h is integer)
    """
    L = x0 - delta
    # align checks
    steps_to_L = round(delta / h)
    L_aligned = x0 - steps_to_L*h
    if abs(L_aligned - L) > 1e-12:
        raise ValueError("delta must be an integer multiple of h for SSA grid alignment.")
    x = x0
    t = 0.0
    # Run until reach L (or below)
    while x > L + 1e-15:
        # Rates at current site x
        M = max(1.0 - 0.5*abs(mu(x))*h, 0.0)  # improved diffusion compensation
        q_left  = max(-mu(x), 0.0)/h + M/(h*h)
        q_right = max( mu(x), 0.0)/h + M/(h*h)
        lam = q_left + q_right
        if lam <= 0.0 or not np.isfinite(lam):
            # Fallback: if numerically degenerate, break (should not happen with our settings)
            # Treat as pure drift to the left with deterministic time h/|mu|
            dt = h / max(abs(mu(x)), 1e-16)
            t += dt
            x -= h
            continue
        # waiting time
        u = rng.random()
        dt = -math.log(u) / lam
        t += dt
        # choose direction
        v = rng.random()
        if v < q_left / lam:
            x -= h
        else:
            x += h
    return t

def ssa_batch(x0: float, delta: float, h: float, N: int, rng: np.random.Generator) -> float:
    ts = np.empty(N, dtype=float)
    for i in range(N):
        ts[i] = ssa_improved_Qu_MFPT(x0, delta, h, rng)
    return float(np.mean(ts))

# ------------------ Tamed EM ------------------
def mfpt_tamed(x0: float, delta: float, dt: float, rng: np.random.Generator, max_steps: int = 10_000_000) -> float:
    L = x0 - delta
    X = x0
    t = 0.0
    for n in range(max_steps):
        if X <= L:
            break
        drift = mu(X)
        drift_tamed = drift / (1.0 + abs(drift)*dt)
        dW = sigma*math.sqrt(dt)*rng.standard_normal()
        X_new = X + drift_tamed*dt + dW
        # crossing detection (linear interpolation within step)
        if X > L and X_new <= L:
            # fraction theta such that X + theta*(X_new - X) = L
            denom = (X_new - X)
            theta = (L - X)/denom if denom != 0 else 1.0
            theta = max(0.0, min(1.0, theta))
            t += theta*dt
            X = L
            break
        else:
            X = X_new
            t += dt
    return t

def tamed_batch(x0: float, delta: float, dt: float, N: int, rng: np.random.Generator) -> float:
    ts = np.empty(N, dtype=float)
    for i in range(N):
        ts[i] = mfpt_tamed(x0, delta, dt, rng)
    return float(np.mean(ts))

# ------------------ Truncated EM (state truncation) ------------------
def mfpt_truncated(x0: float, delta: float, dt: float, rng: np.random.Generator,
                   C: float = 5.0, alpha: float = 1.0/6.0, max_steps: int = 10_000_000) -> float:
    """
    Mao-style truncated EM: evaluate drift at clipped state x_clip with radius r_dt = C*dt^{-alpha}.
    For mu(x) = -x^3, this bounds |mu| <= r_dt^3 (large but finite).
    """
    L = x0 - delta
    X = x0
    t = 0.0
    r_dt = C * (dt ** (-alpha))
    for n in range(max_steps):
        if X <= L:
            break
        x_clip = max(-r_dt, min(r_dt, X))
        drift = mu(x_clip)  # truncated drift by state clipping
        dW = sigma*math.sqrt(dt)*rng.standard_normal()
        X_new = X + drift*dt + dW
        # crossing detection
        if X > L and X_new <= L:
            denom = (X_new - X)
            theta = (L - X)/denom if denom != 0 else 1.0
            theta = max(0.0, min(1.0, theta))
            t += theta*dt
            X = L
            break
        else:
            X = X_new
            t += dt
    return t

def truncated_batch(x0: float, delta: float, dt: float, N: int, rng: np.random.Generator,
                    C: float = 5.0, alpha: float = 1.0/6.0) -> float:
    ts = np.empty(N, dtype=float)
    for i in range(N):
        ts[i] = mfpt_truncated(x0, delta, dt, rng, C=C, alpha=alpha)
    return float(np.mean(ts))

# ------------------ Experiment runner ------------------
def run_experiment(x_list: List[float],
                   delta: float = 0.1,
                   h: float = 1e-3,
                   dt_tamed: float = 1e-5,
                   dt_trunc: float = 1e-5,
                   N_ssa: int = 3000,
                   N_time: int = 3000,
                   seed: int = 12345) -> Dict[str, np.ndarray]:
    """
    For each x in x_list, compute MFPT by SSA (space-discrete), tamed EM, truncated EM,
    then compare to t^e (deterministic time). Returns dict of arrays.
    """
    local_rng = np.random.default_rng(seed)
    xs = np.array(x_list, dtype=float)
    te = np.array([t_e(x, delta) for x in xs], dtype=float)
    tau_ssa = np.zeros_like(xs)
    tau_tamed = np.zeros_like(xs)
    tau_trunc = np.zeros_like(xs)

    # Ensure delta/h is integer for SSA grid alignment
    k = round(delta / h)
    if abs(k*h - delta) > 1e-12:
        raise ValueError("Choose h so that delta/h is an integer.")

    t0 = time.time()
    for idx, x0 in enumerate(xs):
        # SSA
        tau_ssa[idx] = ssa_batch(x0, delta, h, N_ssa, local_rng)
        # Tamed
        tau_tamed[idx] = tamed_batch(x0, delta, dt_tamed, N_time, local_rng)
        # Truncated
        tau_trunc[idx] = truncated_batch(x0, delta, dt_trunc, N_time, local_rng)
    t1 = time.time()

    # Errors
    err_ssa = np.abs(tau_ssa - te)
    err_tamed = np.abs(tau_tamed - te)
    err_trunc = np.abs(tau_trunc - te)

    results = {
        "x": xs,
        "t_e": te,
        "tau_ssa": tau_ssa,
        "tau_tamed": tau_tamed,
        "tau_trunc": tau_trunc,
        "err_ssa": err_ssa,
        "err_tamed": err_tamed,
        "err_trunc": err_trunc,
        "runtime_seconds": np.array([t1 - t0])
    }
    return results

# ------------------ Run a demo experiment ------------------
x_list = [5.0, 8.0, 12.0, 16.0, 20.0]   # large x
delta = 0.1
h = 1e-3              # SSA grid step; delta/h = 100 exactly
dt_tamed = 1e-5
dt_trunc = 1e-5
N_ssa = 3000          # increase for paper-quality plots (e.g., 2e4 or 5e4)
N_time = 3000

res = run_experiment(x_list, delta, h, dt_tamed, dt_trunc, N_ssa, N_time, seed=42)
print(res)
