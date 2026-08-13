# Re-execute the full experiment code after a reset.

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict

rng = np.random.default_rng(12345)

sigma = math.sqrt(2.0)

def mu(x: float) -> float:
    return -x**3

def t_e(x0: float, delta: float) -> float:
    L = x0 - delta
    return 0.5*(1.0/(L**2) - 1.0/(x0**2))

def ssa_improved_Qu_MFPT(x0: float, delta: float, h: float, rng: np.random.Generator) -> float:
    L = x0 - delta
    steps_to_L = round(delta / h)
    L_aligned = x0 - steps_to_L*h
    if abs(L_aligned - L) > 1e-12:
        raise ValueError("delta must be an integer multiple of h for SSA grid alignment.")
    x = x0
    t = 0.0
    while x > L + 1e-15:
        M = max(1.0 - 0.5*abs(mu(x))*h, 0.0)  # improved diffusion comp.
        q_left  = max(-mu(x), 0.0)/h + M/(h*h)
        q_right = max( mu(x), 0.0)/h + M/(h*h)
        lam = q_left + q_right
        if lam <= 0.0 or not np.isfinite(lam):
            dt = h / max(abs(mu(x)), 1e-16)
            t += dt
            x -= h
            continue
        u = rng.random()
        dt = -math.log(u) / lam
        t += dt
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

def tamed_batch(x0: float, delta: float, dt: float, N: int, rng: np.random.Generator) -> float:
    ts = np.empty(N, dtype=float)
    for i in range(N):
        ts[i] = mfpt_tamed(x0, delta, dt, rng)
    return float(np.mean(ts))

def mfpt_truncated(x0: float, delta: float, dt: float, rng: np.random.Generator,
                   C: float = 5.0, alpha: float = 1.0/6.0, max_steps: int = 10_000_000) -> float:
    L = x0 - delta
    X = x0
    t = 0.0
    r_dt = C * (dt ** (-alpha))
    for n in range(max_steps):
        if X <= L:
            break
        x_clip = max(-r_dt, min(r_dt, X))
        drift = mu(x_clip)
        dW = sigma*math.sqrt(dt)*rng.standard_normal()
        X_new = X + drift*dt + dW
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

def run_experiment(x_list: List[float],
                   delta: float = 0.1,
                   h: float = 1e-3,
                   dt_tamed: float = 1e-5,
                   dt_trunc: float = 1e-5,
                   N_ssa: int = 3000,
                   N_time: int = 3000,
                   seed: int = 12345) -> Dict[str, np.ndarray]:
    local_rng = np.random.default_rng(seed)
    xs = np.array(x_list, dtype=float)
    te = np.array([t_e(x, delta) for x in xs], dtype=float)
    tau_ssa = np.zeros_like(xs)
    tau_tamed = np.zeros_like(xs)
    tau_trunc = np.zeros_like(xs)

    k = round(delta / h)
    if abs(k*h - delta) > 1e-12:
        raise ValueError("Choose h so that delta/h is an integer.")

    t0 = time.time()
    for idx, x0 in enumerate(xs):
        tau_ssa[idx] = ssa_batch(x0, delta, h, N_ssa, local_rng)
        tau_tamed[idx] = tamed_batch(x0, delta, dt_tamed, N_time, local_rng)
        tau_trunc[idx] = truncated_batch(x0, delta, dt_trunc, N_time, local_rng)
    t1 = time.time()

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

# Run demo
x_list = [1.0, 50.0, 100.0, 150.0, 200.0, 250.0]
delta = 0.1
h = 1e-3
dt_tamed = 1e-5
dt_trunc = 1e-5
N_ssa = 1500   # lower N to ensure this demo runs fast; increase for paper-quality
N_time = 1500

res = run_experiment(x_list, delta, h, dt_tamed, dt_trunc, N_ssa, N_time, seed=2025)

# Create plots: (1) absolute error vs x (log y); (2) error ratio vs x.
xs = res["x"]
err_ssa = res["err_ssa"]
err_tamed = res["err_tamed"]
err_trunc = res["err_trunc"]
te = res["t_e"]

# Plot 1: absolute error vs x
plt.figure(figsize=(7,5))
plt.plot(xs, err_ssa, marker='o', label='Space SSA (improved Q_u)')
plt.plot(xs, err_tamed, marker='s', label='Tamed EM')
plt.plot(xs, err_trunc, marker='^', label='Truncated EM')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Absolute error |MFPT - t^e|')
plt.title('MFPT absolute error vs x')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: error ratios (time / space) vs x
eps = 1e-16
ratio_tamed = (err_tamed + eps) / (err_ssa + eps)
ratio_trunc = (err_trunc + eps) / (err_ssa + eps)

plt.figure(figsize=(7,5))
plt.plot(xs, ratio_tamed, marker='s', label='Tamed / Space error')
plt.plot(xs, ratio_trunc, marker='^', label='Truncated / Space error')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Error ratio (time / space)')
plt.title('Error ratio vs x')
plt.legend()
plt.tight_layout()
plt.show()

# Print a small table of results
import pandas as pd
df = pd.DataFrame({
    "x": xs,
    "t_e": te,
    "tau_ssa": res["tau_ssa"],
    "tau_tamed": res["tau_tamed"],
    "tau_trunc": res["tau_trunc"],
    "err_ssa": err_ssa,
    "err_tamed": err_tamed,
    "err_trunc": err_trunc,
    "ratio_tamed/space": ratio_tamed,
    "ratio_trunc/space": ratio_trunc
})
# from caas_jupyter_tools import display_dataframe_to_user
# display_dataframe_to_user("MFPT_Results", df)
