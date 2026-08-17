"""
Stochastic Canard system: SSA hybrid vs Euler-Maruyama comparison.

Theory (Berglund & Gentz 2006, Sec. 6.1, normal form 6.1.20):
    If sigma < eps^(1/3): paths track the slow manifold past the saddle node
                          (Canard phenomenon preserved).
    If sigma > eps^(1/3): paths escape at y_t ~ -sigma^(4/5) * eps^(2/5)
                          before reaching the saddle node.

System (global van der Pol form):
    eps * dx = (y - x^3/3 + x) dt        (fast, no noise)
    dy       = (a - x) dt + sigma dW      (slow, with noise)

Experiment:
    - Start on the positive stable branch at (x0=1.5, y0 = x0^3/3 - x0).
    - Record y_exit = y value when the path first crosses x = 1 (saddle node).
    - Canard preserved  =>  y_exit close to y_saddle = -2/3 (path stays on manifold).
    - Early escape      =>  y_exit >> y_saddle.

    Comparison metric: mean(y_exit) and std(y_exit) as functions of
    sigma / sigma_c, for different EM time steps and SSA grid spacings.

SSA grid design (n_x = n_y^2):
    h_x = span / n_x  (fine, fast variable, no noise)
    h_y = span / n_y  (coarse, slow variable, with noise)
    Jump rates (Chang-Cooper hybrid):
        mu_x = (y - x^3/3 + x) / eps
        mu_y = a - x
        m_y  = max(sigma^2 - |mu_y|*h_y, 0) / 2
        q_x+ = max(mu_x,  0) / h_x
        q_x- = max(-mu_x, 0) / h_x
        q_y+ = max(mu_y,  0) / h_y + m_y / h_y^2
        q_y- = max(-mu_y, 0) / h_y + m_y / h_y^2

Parameters:
    eps     = 0.1
    sigma^2 varies;  sigma_c = eps^(1/3)
    a = 1 - eps/8 - 3*eps^2/32 - 173*eps^3/1024 - 0.01
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from pathlib import Path


# ---------------------------------------------------------------------------
#  System parameters
# ---------------------------------------------------------------------------
EPS     = 0.1
A_PARAM = 1 - EPS/8 - 3*EPS**2/32 - 173*EPS**3/1024 - 0.01
SIGMA_C = EPS**(1.0/3.0)

# Experiment geometry
X0      = 1.5                        # start x (on positive slow manifold)
Y0      = X0**3/3 - X0              # start y (exactly on manifold)
X_EXIT  = 1.0                        # saddle node x = 1
Y_SADD  = X_EXIT**3/3 - X_EXIT      # saddle node y = -2/3
T_MAX   = 4.0                        # max simulation time per path
SPAN    = 6.0                        # domain [-3, 3] for SSA


# ---------------------------------------------------------------------------
#  Euler-Maruyama simulator
# ---------------------------------------------------------------------------

def em_y_exit(
    sigma: float,
    dt: float,
    n_paths: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate n_paths trajectories with EM step dt.
    Return array of y_exit values (y when path first crosses x = X_EXIT).
    Paths that do not cross within T_MAX are excluded.
    """
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(T_MAX / dt)
    eps, a  = EPS, A_PARAM

    y_exits = []
    for _ in range(n_paths):
        x, y   = X0, Y0
        prev_x = x
        for _ in range(n_steps):
            dW      = np.random.randn() * sqrt_dt
            x      += (y - x**3/3 + x) / eps * dt
            y      += (a - x) * dt + sigma * dW
            if prev_x > X_EXIT >= x:
                y_exits.append(float(y))
                break
            prev_x = x
            if x < 0.3:          # path jumped far from manifold
                break
    return np.array(y_exits)


# ---------------------------------------------------------------------------
#  SSA hybrid simulator  (n_x = n_y^2)
# ---------------------------------------------------------------------------

def ssa_y_exit(
    sigma: float,
    h_y: float,
    n_paths: int = 500,
    seed: int = 7777,
) -> np.ndarray:
    """
    Simulate n_paths trajectories with the hybrid SSA scheme.
    Grid: n_y slow-variable points, n_x = n_y^2 fast-variable points.
    Return array of y_exit values.
    """
    eps, a = EPS, A_PARAM
    sig2   = sigma**2

    lowx = lowy = -SPAN / 2       # grid origin = -3
    n_y  = max(int(SPAN / h_y), 2)
    h_y  = SPAN / n_y             # exact
    n_x  = n_y * n_y
    h_x  = SPAN / n_x

    np.random.seed(seed)
    y_exits = []

    for _ in range(n_paths):
        x, y   = X0, Y0
        t      = 0.0
        prev_x = x

        while t < T_MAX:
            # drift
            mu_x = (y - x**3/3 + x) / eps
            mu_y = a - x

            # Chang-Cooper mixing term (y direction only)
            m_y = 0.5 * max(sig2 - abs(mu_y) * h_y, 0.0)

            # jump rates
            q_xp = max(mu_x,  0.0) / h_x
            q_xm = max(-mu_x, 0.0) / h_x
            q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y)
            q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)
            lam  = q_xp + q_xm + q_yp + q_ym

            # Gillespie waiting time
            tau  = -np.log(1.0 - np.random.random()) / lam
            t   += tau

            # check crossing BEFORE jumping
            if prev_x > X_EXIT >= x:
                y_exits.append(float(y))
                break

            # jump direction
            r      = np.random.random() * lam
            prev_x = x
            if r < q_xp:
                x += h_x
            elif r < q_xp + q_xm:
                x -= h_x
            elif r < q_xp + q_xm + q_yp:
                y += h_y
            else:
                y -= h_y

            if x < 0.3:          # escaped far from manifold
                break

    return np.array(y_exits)


# ---------------------------------------------------------------------------
#  Run full comparison experiment
# ---------------------------------------------------------------------------

def run_experiment(
    sigma_ratios: np.ndarray,
    em_dt_factors: list,    # dt = eps * factor
    ssa_h_y_list: list,
    n_paths: int = 400,
) -> dict:
    """
    For each sigma, compute mean(y_exit) and std(y_exit) for each method.

    Returns dict with keys:
        'sigma_ratios', 'sigma_vals',
        'em_<factor>_mean', 'em_<factor>_std',
        'ssa_<h>_mean',     'ssa_<h>_std'
    """
    results = {
        "sigma_ratios": sigma_ratios,
        "sigma_vals":   sigma_ratios * SIGMA_C,
    }

    for factor in em_dt_factors:
        dt    = EPS * factor
        key   = f"em_{factor}"
        means, stds = [], []
        print(f"  EM dt={dt:.4f} (dt/eps={factor})", flush=True)
        for ratio in sigma_ratios:
            ye  = em_y_exit(ratio * SIGMA_C, dt, n_paths=n_paths)
            means.append(np.mean(ye) if len(ye) > 0 else np.nan)
            stds.append(np.std(ye)   if len(ye) > 0 else np.nan)
            print(f"    sigma/sc={ratio:.2f}: n={len(ye)}, "
                  f"mean={means[-1]:.4f}, std={stds[-1]:.4f}")
        results[key + "_mean"] = np.array(means)
        results[key + "_std"]  = np.array(stds)

    for h_y in ssa_h_y_list:
        key   = f"ssa_{h_y}"
        means, stds = [], []
        n_y  = int(SPAN / h_y)
        print(f"  SSA h_y={h_y:.3f} (n_y={n_y}, n_x={n_y**2})", flush=True)
        for ratio in sigma_ratios:
            ye  = ssa_y_exit(ratio * SIGMA_C, h_y, n_paths=n_paths)
            means.append(np.mean(ye) if len(ye) > 0 else np.nan)
            stds.append(np.std(ye)   if len(ye) > 0 else np.nan)
            print(f"    sigma/sc={ratio:.2f}: n={len(ye)}, "
                  f"mean={means[-1]:.4f}, std={stds[-1]:.4f}")
        results[key + "_mean"] = np.array(means)
        results[key + "_std"]  = np.array(stds)

    return results


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_results(res: dict, out_dir: Path) -> None:
    ratios = res["sigma_ratios"]

    # color palettes
    em_colors  = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#d62728"]
    ssa_colors = ["#2ca02c", "#98df8a", "#9467bd"]

    em_dt_factors = [0.01, 0.1, 0.5, 1.0, 2.0]
    ssa_h_y_list  = [0.06, 0.1, 0.3]

    em_labels = [
        r"EM  $dt = 0.01\varepsilon$",
        r"EM  $dt = 0.1\varepsilon$",
        r"EM  $dt = 0.5\varepsilon$",
        r"EM  $dt = 1.0\varepsilon$",
        r"EM  $dt = 2.0\varepsilon$",
    ]
    ssa_labels = [
        r"SSA $h_y = 0.06$  ($n_x = n_y^2$)",
        r"SSA $h_y = 0.10$  ($n_x = n_y^2$)",
        r"SSA $h_y = 0.30$  ($n_x = n_y^2$)",
    ]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # ---- panel (a): mean(y_exit) for EM ----
    ax1 = fig.add_subplot(gs[0, 0])
    for factor, color, label in zip(em_dt_factors, em_colors, em_labels):
        key  = f"em_{factor}"
        mean = res[key + "_mean"]
        ax1.plot(ratios, mean, "o-", color=color, lw=1.8, ms=5, label=label)
    ax1.axhline(Y_SADD, color="k", ls=":", lw=1.2, label=f"$y_{{saddle}}={Y_SADD:.3f}$")
    ax1.axvline(1.0, color="gray", ls="--", lw=1.0)
    ax1.set_xlabel(r"$\sigma\,/\,\sigma_c$", fontsize=12)
    ax1.set_ylabel(r"mean$(y_\mathrm{exit})$", fontsize=12)
    ax1.set_title("(a) EM: mean exit y vs sigma", fontsize=11)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.25)

    # ---- panel (b): mean(y_exit) for SSA ----
    ax2 = fig.add_subplot(gs[0, 1])
    for h_y, color, label in zip(ssa_h_y_list, ssa_colors, ssa_labels):
        key  = f"ssa_{h_y}"
        mean = res[key + "_mean"]
        ax2.plot(ratios, mean, "s-", color=color, lw=1.8, ms=5, label=label)
    # also plot best EM as reference
    ax2.plot(ratios, res["em_0.01_mean"], "k--", lw=1.5, ms=4,
             label=r"EM ref ($dt=0.01\varepsilon$)")
    ax2.axhline(Y_SADD, color="k", ls=":", lw=1.2)
    ax2.axvline(1.0, color="gray", ls="--", lw=1.0)
    ax2.set_xlabel(r"$\sigma\,/\,\sigma_c$", fontsize=12)
    ax2.set_ylabel(r"mean$(y_\mathrm{exit})$", fontsize=12)
    ax2.set_title("(b) SSA: mean exit y vs sigma", fontsize=11)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.25)

    # ---- panel (c): std(y_exit) for EM ----
    ax3 = fig.add_subplot(gs[1, 0])
    for factor, color, label in zip(em_dt_factors, em_colors, em_labels):
        key = f"em_{factor}"
        std = res[key + "_std"]
        ax3.plot(ratios, std, "o-", color=color, lw=1.8, ms=5, label=label)
    ax3.axvline(1.0, color="gray", ls="--", lw=1.0, label=r"$\sigma=\sigma_c$")
    ax3.set_xlabel(r"$\sigma\,/\,\sigma_c$", fontsize=12)
    ax3.set_ylabel(r"std$(y_\mathrm{exit})$", fontsize=12)
    ax3.set_title("(c) EM: spread of exit y vs sigma", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)

    # ---- panel (d): error |mean(y_exit) - reference| at sigma = sigma_c ----
    ax4 = fig.add_subplot(gs[1, 1])

    # reference = finest EM
    ref_mean = res["em_0.01_mean"]

    # EM error vs dt
    em_dts    = np.array(em_dt_factors) * EPS
    em_errors = [np.abs(res[f"em_{f}_mean"] - ref_mean).mean()
                 for f in em_dt_factors]
    ax4.loglog(em_dts / EPS, em_errors, "o-", color="#1f77b4",
               lw=2, ms=7, label="EM (avg over sigma)")

    # SSA error vs h_y
    ssa_hy     = np.array(ssa_h_y_list)
    ssa_errors = [np.abs(res[f"ssa_{h}_mean"] - ref_mean).mean()
                  for h in ssa_h_y_list]
    ax4.loglog(ssa_hy, ssa_errors, "s-", color="#2ca02c",
               lw=2, ms=7, label="SSA (avg over sigma)")

    # reference lines
    h_ref = np.logspace(-2, 0, 50)
    ax4.loglog(h_ref, 0.005 * h_ref,    "k:",  lw=1.2, label=r"$O(h)$")
    ax4.loglog(h_ref, 0.01  * h_ref**2, "k--", lw=1.2, label=r"$O(h^2)$")

    ax4.axvline(1.0, color="gray", ls="--", lw=1.0, label="dt or h = eps")
    ax4.set_xlabel(r"$dt/\varepsilon$  or  $h_y$", fontsize=12)
    ax4.set_ylabel(r"mean $|$error$|$ in $y_\mathrm{exit}$", fontsize=12)
    ax4.set_title("(d) Discretization error (ref = EM finest)", fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        f"Stochastic Canard: SSA vs EM  "
        f"($\\varepsilon={EPS}$, $\\sigma_c=\\varepsilon^{{1/3}}={SIGMA_C:.3f}$)\n"
        f"Metric: $y_{{\\mathrm{{exit}}}}$ = $y$ when path first crosses $x=1$ (saddle node)",
        fontsize=12
    )

    out_png = out_dir / "canard_ssa_vs_em.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent

    print("Stochastic Canard: SSA hybrid vs Euler-Maruyama")
    print(f"eps={EPS},  sigma_c={SIGMA_C:.4f},  a={A_PARAM:.6f}")
    print(f"Start: ({X0}, {Y0:.4f}),  Saddle: ({X_EXIT}, {Y_SADD:.4f})")
    print()

    sigma_ratios  = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0])

    # EM: vary dt/eps from accurate to coarse
    em_dt_factors = [0.01, 0.1, 0.5, 1.0, 2.0]

    # SSA: vary h_y (with n_x = n_y^2 fixed)
    ssa_h_y_list  = [0.06, 0.1, 0.3]

    n_paths = 500

    print(f"sigma values: {sigma_ratios} x sigma_c")
    print(f"EM   dt/eps:  {em_dt_factors}")
    print(f"SSA  h_y:     {ssa_h_y_list}")
    print(f"n_paths per run: {n_paths}")
    print()

    t0  = time.perf_counter()
    res = run_experiment(sigma_ratios, em_dt_factors, ssa_h_y_list, n_paths)
    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")

    plot_results(res, out_dir)


if __name__ == "__main__":
    main()