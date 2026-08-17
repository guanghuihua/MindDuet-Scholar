"""
Stochastic Canard system: SSA hybrid vs EM - four Canard-structure statistics.

Theory (Berglund & Gentz 2006, Sec. 6.1):
    sigma_c = eps^(1/3)
    sigma < sigma_c  =>  Canard preserved: paths track the slow manifold.
    sigma > sigma_c  =>  paths escape at y ~ -sigma^(4/5) * eps^(2/5).

System:
    eps * dx = (y - x^3/3 + x) dt        (fast, no noise)
    dy       = (a - x) dt + sigma dW      (slow, with noise)

Four statistics:
    1. Hitting probability  P_hit
       P(path enters tube A before crossing x=1)
       A = { |y - y_ref| <= eta,  |x - x_bar(y)| <= delta_hit }
       where x_bar(y) is the positive stable slow manifold.

    2. Premature escape probability  P_esc
       P(path leaves manifold tube before y drops to y_esc_thr)
       escape = |x - x_bar(y)| > delta_esc  while y > y_esc_thr

    3. Jump-location distribution
       y_jump = y value when path first crosses x = 1 (saddle node).
       Compare mean, std, and histogram across methods.

    4. Slow-manifold occupancy
       Fraction of time spent with |x - x_bar(y)| < delta_hit.

Grid (SSA):
    n_x = n_y^2  (fast variable, no noise, fine grid)
    n_y          (slow variable, with noise, coarse grid)
    Jump rates: Chang-Cooper hybrid scheme.
"""

from __future__ import annotations
import numpy as np
import numba as nb
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

X0 = 1.5                       # start on positive slow manifold
Y0 = X0**3/3 - X0

# Reference window on slow manifold (near saddle, but not too close)
Y_REF      = -0.60    # centre y of window
ETA        =  0.04    # half-width in y
DELTA_HIT  =  0.15    # x-distance from manifold for "hit" and "occupancy"

# Premature escape threshold
Y_ESC_THR  = -0.55    # escape is "premature" if y still > this value
DELTA_ESC  =  0.25    # x-distance from manifold that counts as escaped

T_MAX = 4.0
SPAN  = 6.0           # SSA domain [-3, 3]


# ---------------------------------------------------------------------------
#  Slow manifold helper
# ---------------------------------------------------------------------------

def x_manifold_py(y: float) -> float:
    """x on the positive stable branch: solve x^3/3 - x = y, x > 1."""
    coeffs = [1.0/3.0, 0.0, -1.0, -y]
    roots  = np.roots(coeffs)
    real_r = roots[np.isreal(roots)].real
    valid  = real_r[real_r > 1.0]
    return float(valid[0]) if len(valid) > 0 else np.nan


# Precompute manifold lookup table for numba use (avoid np.roots inside jit)
_Y_TABLE = np.linspace(-0.66, -0.30, 1000)
_X_TABLE = np.array([x_manifold_py(y) for y in _Y_TABLE])

@nb.njit(fastmath=True, cache=True)
def x_manifold_fast(y: float) -> float:
    """Fast piecewise-linear lookup for x_bar(y)."""
    if y < -0.66 or y > -0.30:
        return np.nan
    # linear interpolation
    n   = len(_Y_TABLE)
    idx = (y - _Y_TABLE[0]) / (_Y_TABLE[-1] - _Y_TABLE[0]) * (n - 1)
    i   = int(idx)
    if i >= n - 1:
        return _X_TABLE[-1]
    frac = idx - i
    return _X_TABLE[i] * (1.0 - frac) + _X_TABLE[i+1] * frac


# ---------------------------------------------------------------------------
#  EM simulator (numba JIT)
# ---------------------------------------------------------------------------

@nb.njit(fastmath=True, cache=True)
def _em_single(sigma, dt, seed,
               y_ref, eta, delta_hit,
               y_esc_thr, delta_esc):
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(T_MAX / dt)
    eps, a  = EPS, A_PARAM

    x, y   = X0, Y0
    prev_x = x

    hit    = False
    prem   = False
    y_jump = np.nan
    t_mfd  = 0.0
    t_tot  = 0.0

    for _ in range(n_steps):
        dW  = np.random.randn() * sqrt_dt
        x  += (y - x**3/3 + x) / eps * dt
        y  += (a - x) * dt + sigma * dW
        t_tot += dt

        xm = x_manifold_fast(y)
        if not np.isnan(xm):
            dx = abs(x - xm)
            if abs(y - y_ref) <= eta and dx <= delta_hit:
                hit = True
            if (not prem) and y > y_esc_thr and dx > delta_esc:
                prem = True
            if dx < delta_hit:
                t_mfd += dt

        if prev_x > 1.0 >= x and np.isnan(y_jump):
            y_jump = y
        prev_x = x
        if x < 0.3:
            break

    occ = t_mfd / max(t_tot, 1e-14)
    return hit, prem, y_jump, occ


@nb.njit(fastmath=True, parallel=True, cache=True)
def em_batch(sigma, dt, n_paths,
             y_ref, eta, delta_hit, y_esc_thr, delta_esc):
    """Run n_paths EM trajectories in parallel."""
    hits  = np.zeros(n_paths, dtype=nb.boolean)
    prems = np.zeros(n_paths, dtype=nb.boolean)
    yjumps = np.full(n_paths, np.nan)
    occs   = np.zeros(n_paths)

    for i in nb.prange(n_paths):
        h, p, yj, oc = _em_single(sigma, dt, i,
                                   y_ref, eta, delta_hit,
                                   y_esc_thr, delta_esc)
        hits[i]   = h
        prems[i]  = p
        yjumps[i] = yj
        occs[i]   = oc

    return hits, prems, yjumps, occs


# ---------------------------------------------------------------------------
#  SSA simulator (numba JIT)
# ---------------------------------------------------------------------------

@nb.njit(fastmath=True, cache=True)
def _ssa_single(sigma, h_y, seed,
                y_ref, eta, delta_hit,
                y_esc_thr, delta_esc):
    np.random.seed(seed)
    sig2 = sigma * sigma
    eps, a = EPS, A_PARAM

    n_y = int(SPAN / h_y)
    h_y = SPAN / n_y
    n_x = n_y * n_y
    h_x = SPAN / n_x

    x, y   = X0, Y0
    prev_x = x
    t      = 0.0
    t_mfd  = 0.0
    t_tot  = 0.0

    hit    = False
    prem   = False
    y_jump = np.nan

    while t < T_MAX:
        mu_x = (y - x**3/3 + x) / eps
        mu_y = a - x
        m_y  = 0.5 * max(sig2 - abs(mu_y) * h_y, 0.0)

        q_xp = max(mu_x,  0.0) / h_x
        q_xm = max(-mu_x, 0.0) / h_x
        q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y)
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        lam  = q_xp + q_xm + q_yp + q_ym

        tau   = -np.log(1.0 - np.random.random()) / lam
        t    += tau
        t_tot += tau

        xm = x_manifold_fast(y)
        if not np.isnan(xm):
            dx = abs(x - xm)
            if abs(y - y_ref) <= eta and dx <= delta_hit:
                hit = True
            if (not prem) and y > y_esc_thr and dx > delta_esc:
                prem = True
            if dx < delta_hit:
                t_mfd += tau

        if prev_x > 1.0 >= x and np.isnan(y_jump):
            y_jump = y

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

        if x < 0.3:
            break

    occ = t_mfd / max(t_tot, 1e-14)
    return hit, prem, y_jump, occ


@nb.njit(fastmath=True, parallel=True, cache=True)
def ssa_batch(sigma, h_y, n_paths,
              y_ref, eta, delta_hit, y_esc_thr, delta_esc):
    """Run n_paths SSA trajectories in parallel."""
    hits   = np.zeros(n_paths, dtype=nb.boolean)
    prems  = np.zeros(n_paths, dtype=nb.boolean)
    yjumps = np.full(n_paths, np.nan)
    occs   = np.zeros(n_paths)

    for i in nb.prange(n_paths):
        h, p, yj, oc = _ssa_single(sigma, h_y, i + 5000,
                                    y_ref, eta, delta_hit,
                                    y_esc_thr, delta_esc)
        hits[i]   = h
        prems[i]  = p
        yjumps[i] = yj
        occs[i]   = oc

    return hits, prems, yjumps, occs


def summarise(hits, prems, yjumps, occs):
    """Compute summary statistics from raw arrays."""
    valid = yjumps[~np.isnan(yjumps)]
    return dict(
        P_hit  = float(np.mean(hits)),
        P_esc  = float(np.mean(prems)),
        yj_mean= float(np.mean(valid)) if len(valid) > 0 else np.nan,
        yj_std = float(np.std(valid))  if len(valid) > 0 else np.nan,
        yj_all = valid,
        occ    = float(np.mean(occs)),
    )


# ---------------------------------------------------------------------------
#  Run full experiment
# ---------------------------------------------------------------------------

def run_experiment(sigma_ratios, configs, n_paths=600):
    """
    configs : list of (label, method, param)
        method = "em"  => param = dt
        method = "ssa" => param = h_y
    """
    geo = (Y_REF, ETA, DELTA_HIT, Y_ESC_THR, DELTA_ESC)
    all_results = {}

    for label, method, param in configs:
        print(f"\n--- {label} ---", flush=True)
        rows = []
        for ratio in sigma_ratios:
            sigma = ratio * SIGMA_C
            t0    = time.perf_counter()
            if method == "em":
                h, p, yj, oc = em_batch(sigma, param, n_paths, *geo)
            else:
                h, p, yj, oc = ssa_batch(sigma, param, n_paths, *geo)
            s = summarise(h, p, yj, oc)
            elapsed = time.perf_counter() - t0
            print(f"  sigma/sc={ratio:.2f}: "
                  f"P_hit={s['P_hit']:.3f}  P_esc={s['P_esc']:.3f}  "
                  f"y_jump={s['yj_mean']:.4f}+/-{s['yj_std']:.4f}  "
                  f"occ={s['occ']:.4f}  ({elapsed:.1f}s)")
            rows.append(s)
        all_results[label] = rows

    return all_results


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_all(all_results, sigma_ratios, out_dir):
    ratios = sigma_ratios
    labels = list(all_results.keys())

    # color / style per method
    style = {
        "EM  dt=eps/100": dict(color="#1f77b4", ls="-",  marker="o", lw=2.0),
        "EM  dt=0.5*eps": dict(color="#aec7e8", ls="--", marker="o", lw=1.5),
        "EM  dt=1.0*eps": dict(color="#ff7f0e", ls="--", marker="s", lw=1.5),
        "EM  dt=2.0*eps": dict(color="#d62728", ls="--", marker="s", lw=1.5),
        "SSA hy=0.10":    dict(color="#2ca02c", ls="-",  marker="^", lw=2.0),
        "SSA hy=0.30":    dict(color="#98df8a", ls="--", marker="^", lw=1.5),
    }

    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])  # P_hit
    ax2 = fig.add_subplot(gs[0, 1])  # P_esc
    ax3 = fig.add_subplot(gs[0, 2])  # occupancy
    ax4 = fig.add_subplot(gs[1, 0])  # mean(y_jump)
    ax5 = fig.add_subplot(gs[1, 1])  # std(y_jump)
    ax6 = fig.add_subplot(gs[1, 2])  # histogram of y_jump at sigma = sigma_c

    for label in labels:
        rows = all_results[label]
        kw   = style.get(label, dict(color="gray", ls="-", marker="x", lw=1.0))
        ph   = [r["P_hit"]   for r in rows]
        pe   = [r["P_esc"]   for r in rows]
        oc   = [r["occ"]     for r in rows]
        yjm  = [r["yj_mean"] for r in rows]
        yjs  = [r["yj_std"]  for r in rows]

        for ax, vals in [(ax1, ph), (ax2, pe), (ax3, oc), (ax4, yjm), (ax5, yjs)]:
            ax.plot(ratios, vals, label=label, ms=5, **kw)

    # Vertical line at sigma_c
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axvline(1.0, color="gray", ls=":", lw=1.2)
        ax.set_xlabel(r"$\sigma\,/\,\sigma_c$", fontsize=11)
        ax.grid(True, alpha=0.25)

    ax1.set_ylabel("Hitting probability", fontsize=11)
    ax1.set_title(
        f"(1) P_hit: path enters tube A\n"
        f"A = {{|y-{Y_REF}|<{ETA}, |x-x*(y)|<{DELTA_HIT}}}",
        fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=7, loc="lower left")

    ax2.set_ylabel("Premature escape prob.", fontsize=11)
    ax2.set_title(
        f"(2) P_esc: leaves manifold before y={Y_ESC_THR}\n"
        f"|x - x*(y)| > {DELTA_ESC}",
        fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7)

    ax3.set_ylabel("Manifold occupancy", fontsize=11)
    ax3.set_title(
        f"(3) Occupancy: fraction of time\n"
        f"with |x - x*(y)| < {DELTA_HIT}",
        fontsize=10)
    ax3.legend(fontsize=7)

    ax4.set_ylabel(r"mean($y_\mathrm{jump}$)", fontsize=11)
    ax4.axhline(-2/3, color="k", ls=":", lw=1.0)
    ax4.set_title(
        r"(4) Jump location: mean($y$) at x=1 crossing"
        f"\ndotted = saddle $y_c = {-2/3:.3f}$",
        fontsize=10)
    ax4.legend(fontsize=7)

    ax5.set_ylabel(r"std($y_\mathrm{jump}$)", fontsize=11)
    ax5.set_title(r"(5) Jump location: std($y$) at x=1", fontsize=10)
    ax5.legend(fontsize=7)

    # Histogram at sigma ~= sigma_c (closest ratio to 1.0)
    idx_sc = int(np.argmin(np.abs(ratios - 1.0)))
    ax6.set_title(
        r"(6) Jump-location distribution at $\sigma \approx \sigma_c$",
        fontsize=10)
    for label in labels:
        yj_arr = all_results[label][idx_sc]["yj_all"]
        kw2    = {k: style.get(label, {}).get(k, None)
                  for k in ("color", "ls", "lw")}
        kw2    = {k: v for k, v in kw2.items() if v is not None}
        if len(yj_arr) > 5:
            ax6.hist(yj_arr, bins=25, density=True, alpha=0.35,
                     color=kw2.get("color", "gray"), label=label)
    ax6.axvline(-2/3, color="k", ls=":", lw=1.2, label=r"$y_c=-2/3$")
    ax6.set_xlabel(r"$y_\mathrm{jump}$", fontsize=11)
    ax6.set_ylabel("Density", fontsize=11)
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.25)

    fig.suptitle(
        f"Stochastic Canard: SSA hybrid vs EM  "
        f"($\\varepsilon={EPS}$,  $\\sigma_c=\\varepsilon^{{1/3}}={SIGMA_C:.3f}$)\n"
        f"Start: $x_0={X0}$ on slow manifold.  "
        f"Vertical dotted: $\\sigma=\\sigma_c$",
        fontsize=12
    )

    out_png = out_dir / "canard_four_stats.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent

    print(f"EPS={EPS}, SIGMA_C={SIGMA_C:.4f}, A={A_PARAM:.5f}")
    print(f"Start: ({X0}, {Y0:.4f})")
    print(f"Window A: y in [{Y_REF-ETA:.3f}, {Y_REF+ETA:.3f}], "
          f"|x - x*(y)| <= {DELTA_HIT}")
    print(f"Escape:   y > {Y_ESC_THR} and |x - x*(y)| > {DELTA_ESC}")
    print()

    # JIT warm-up
    print("Compiling JIT functions...")
    geo = (Y_REF, ETA, DELTA_HIT, Y_ESC_THR, DELTA_ESC)
    em_batch(SIGMA_C, EPS/100, 4, *geo)
    ssa_batch(SIGMA_C, 0.1, 4, *geo)
    print("Done.\n")

    sigma_ratios = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0])

    configs = [
        # (label,           method, param)
        ("EM  dt=eps/100",  "em",   EPS / 100),   # reference
        ("EM  dt=0.5*eps",  "em",   EPS * 0.5),
        ("EM  dt=1.0*eps",  "em",   EPS * 1.0),
        ("EM  dt=2.0*eps",  "em",   EPS * 2.0),   # expected to fail
        ("SSA hy=0.10",     "ssa",  0.10),         # expected to be robust
        ("SSA hy=0.30",     "ssa",  0.30),
    ]

    t0 = time.perf_counter()
    results = run_experiment(sigma_ratios, configs, n_paths=800)
    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")

    plot_all(results, sigma_ratios, out_dir)


if __name__ == "__main__":
    main()