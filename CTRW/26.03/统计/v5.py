"""
Stochastic Canard system: hybrid SSA trajectory simulation.

System equations:
    epsilon * dx = (y - x^3/3 + x) dt        (fast variable, no noise)
    dy = (a - x) dt + sigma * dW_t            (slow variable, with noise)

Grid design (n_x = n_y^2):
    x direction (no noise, fast): h_x = span / n_x,  n_x = n_y^2
    y direction (with noise, slow): h_y = span / n_y

Jump rates (Chang-Cooper hybrid scheme):
    mu_x = (y - x^3/3 + x) / eps
    mu_y = a - x
    m_y  = max(sigma^2 - |mu_y| * h_y, 0) / 2    (mixing term, y only)
    m_x  = 0                                       (no diffusion in x)

    q_x+ = max(mu_x,  0) / h_x
    q_x- = max(-mu_x, 0) / h_x
    q_y+ = max(mu_y,  0) / h_y + m_y / h_y^2
    q_y- = max(-mu_y, 0) / h_y + m_y / h_y^2

Parameters:
    delta (epsilon) = 0.1
    sigma^2         = 2.0
    a = 1 - delta/8 - 3*delta^2/32 - 173*delta^3/1024 - 0.01
"""

from __future__ import annotations
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time
from pathlib import Path


# -----------------------------------------------------------------------------
#  System parameters
# -----------------------------------------------------------------------------
DELTA    = 0.1
SIGMA_SQ = 2.0
A_PARAM  = 1 - DELTA/8 - 3*DELTA**2/32 - 173*DELTA**3/1024 - 0.01


# -----------------------------------------------------------------------------
#  Single-chain SSA (time-weighted density accumulation)
# -----------------------------------------------------------------------------

@nb.njit(fastmath=True, cache=True)
def ssa_canard_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n_y: int,
    eps: float,
    sigma_sq: float,
    a: float,
    sample_size: int,
) -> tuple:
    """
    Single SSA chain with time-weighted density accumulation.

    Grid:
        n_x = n_y^2  (x direction, no noise, fine grid)
        n_y          (y direction, with noise, coarse grid)

    Returns:
        counts    : shape (n_x, n_y), time-weighted hit counts
        oob_count : number of out-of-bounds events
    """
    n_x = n_y * n_y
    h_x = span / n_x
    h_y = span / n_y

    counts = np.zeros((n_x, n_y), dtype=np.float64)

    # random initial point
    x = lowx + np.random.random() * span
    y = lowy + np.random.random() * span

    valid_count = 0
    oob_count   = 0

    while valid_count < sample_size:

        # drift
        mu_x = (y - x * x * x / 3.0 + x) / eps   # O(1/eps), fast variable
        mu_y = a - x                                # O(1),     slow variable

        # Chang-Cooper mixing term: y direction only (has noise)
        m_y = 0.5 * max(sigma_sq - abs(mu_y) * h_y, 0.0)
        # x direction: m_x = 0 (no physical diffusion)

        # jump rates
        q_xp = max(mu_x,  0.0) / h_x                       # x right
        q_xm = max(-mu_x, 0.0) / h_x                       # x left
        q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y)  # y up
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)  # y down
        lam  = q_xp + q_xm + q_yp + q_ym

        # Gillespie waiting time
        tau = -np.log(1.0 - np.random.random()) / lam

        # grid index of current position
        ix = int((x - lowx) / h_x)
        iy = int((y - lowy) / h_y)

        if 0 <= ix < n_x and 0 <= iy < n_y:
            counts[ix, iy] += tau
            valid_count += 1
        else:
            oob_count += 1
            # reflect at boundary
            if ix < 0:      x = lowx + h_x * 0.5
            elif ix >= n_x: x = lowx + span - h_x * 0.5
            if iy < 0:      y = lowy + h_y * 0.5
            elif iy >= n_y: y = lowy + span - h_y * 0.5

        # choose jump direction
        r = np.random.random() * lam
        if r < q_xp:
            x += h_x
        elif r < q_xp + q_xm:
            x -= h_x
        elif r < q_xp + q_xm + q_yp:
            y += h_y
        else:
            y -= h_y

    return counts, oob_count


# -----------------------------------------------------------------------------
#  Ensemble runner (parallel chains)
# -----------------------------------------------------------------------------

@nb.njit(fastmath=True, parallel=True, cache=True)
def run_ensemble_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n_y: int,
    eps: float,
    sigma_sq: float,
    a: float,
    sample_size: int,
    loops: int,
    bin_factor_x: int,
    bin_factor_y: int,
) -> tuple:
    """
    Run `loops` independent SSA chains in parallel and average their densities.

    bin_factor_x: coarsen factor in x.
    bin_factor_y: coarsen factor in y.
    """
    n_x       = n_y * n_y
    flat_size = n_x * n_y

    # Store per-chain results first. Writing directly into shared arrays inside
    # prange would create race conditions.
    density_all = np.zeros((loops, flat_size), dtype=np.float64)
    oob_all     = np.zeros(loops, dtype=np.int64)

    for i in nb.prange(loops):
        c, oob = ssa_canard_timeweighted(
            lowx, lowy, span, n_y, eps, sigma_sq, a, sample_size
        )
        density_all[i, :] = c.ravel()
        oob_all[i] = oob

    density_sum = np.zeros(flat_size, dtype=np.float64)
    oob_total = 0
    for i in range(loops):
        density_sum += density_all[i, :]
        oob_total += oob_all[i]

    density_mean = density_sum / loops

    # optional coarsening
    if bin_factor_x > 1 or bin_factor_y > 1:
        out_nx = n_x // bin_factor_x
        out_ny = n_y // bin_factor_y
        fine   = density_mean.reshape((n_x, n_y))
        coarse = np.zeros((out_nx, out_ny), dtype=np.float64)
        for ix in range(out_nx):
            for iy in range(out_ny):
                s = 0.0
                for di in range(bin_factor_x):
                    for dj in range(bin_factor_y):
                        s += fine[ix * bin_factor_x + di,
                                  iy * bin_factor_y + dj]
                coarse[ix, iy] = s
        return coarse.ravel(), oob_total

    return density_mean, oob_total


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent

    # simulation parameters
    lowx        = -3.0
    lowy        = -3.0
    span        =  6.0
    n_y         = 600          # slow variable grid points
    sample_size = int(3e7)     # samples per chain
    loops       = 5            # number of parallel chains
    out_n       = 100          # output grid size (after binning)

    eps      = DELTA
    sigma_sq = SIGMA_SQ
    a        = A_PARAM

    # n_x = n_y^2, x and y need different coarsening factors to reach out_n x out_n.
    n_x = n_y * n_y
    if n_y % out_n != 0:
        raise ValueError(f"n_y={n_y} must be divisible by out_n={out_n}")
    if n_x % out_n != 0:
        raise ValueError(f"n_x={n_x} must be divisible by out_n={out_n}")
    bin_factor_y = n_y // out_n
    bin_factor_x = n_x // out_n

    h_x = span / n_x
    h_y = span / n_y

    print(f"System : eps={eps:.3f},  sigma={np.sqrt(sigma_sq):.4f},  a={a:.6f}")
    print(f"Grid   : n_y={n_y},  n_x=n_y^2={n_x}")
    print(f"         h_y={h_y:.5f},  h_x={h_x:.7f}")
    print(f"         numerical diffusion x ~ h_x/(2*eps) = {h_x/(2*eps):.6f}")
    print(f"Samples: {loops} chains x {sample_size:.0e} = {loops*sample_size:.0e}")
    print(
        f"Output : {out_n}x{out_n} grid "
        f"(bin_factor_x={bin_factor_x}, bin_factor_y={bin_factor_y})"
    )
    print()

    # JIT warm-up
    print("Compiling JIT functions...")
    run_ensemble_timeweighted(lowx, lowy, span, 10, eps, sigma_sq, a,
                               100, 1, 1, 1)
    print("Done.\n")

    # run simulation
    print("Running SSA ensemble...")
    t0 = time.perf_counter()
    data_flat, oob = run_ensemble_timeweighted(
        lowx, lowy, span, n_y, eps, sigma_sq, a,
        sample_size, loops, bin_factor_x, bin_factor_y
    )
    elapsed = time.perf_counter() - t0
    print(f"Finished in {elapsed:.2f}s,  out-of-bounds events: {oob}")

    # normalise to probability density
    h_eff = span / out_n
    data  = data_flat.reshape((out_n, out_n))
    total = data.sum()
    if total > 0:
        data /= (h_eff * h_eff * total)
    print(f"Density integral = {data.sum() * h_eff * h_eff:.6f}  (should be 1.0)")

    # grid coordinates
    x_arr = np.linspace(lowx + h_eff / 2.0, lowx + span - h_eff / 2.0, out_n)
    y_arr = np.linspace(lowy + h_eff / 2.0, lowy + span - h_eff / 2.0, out_n)
    X, Y  = np.meshgrid(x_arr, y_arr, indexing="ij")

    # slow manifold
    x_mf = np.linspace(-2.5, 2.5, 500)
    y_mf = x_mf**3 / 3.0 - x_mf

    # -------------------------------------------------------------------------
    #  Plot
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))

    # -- 3D surface --
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, data, cmap="viridis", alpha=0.9, linewidth=0)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("Density")
    ax1.set_title(
        f"SSA Canard  (n_x = n_y^2)\n"
        f"eps={eps},  sigma={np.sqrt(sigma_sq):.3f}"
    )

    # -- 2D contour --
    ax2 = fig.add_subplot(122)
    cf = ax2.contourf(X, Y, data, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax2, label="Density")
    ax2.plot(x_mf, y_mf, "r--", lw=1.5, label="Slow manifold")
    ax2.axvline(x= 1.0, color="orange", ls=":", lw=1.2, alpha=0.8)
    ax2.axvline(x=-1.0, color="orange", ls=":", lw=1.2, alpha=0.8,
                label="Saddle nodes x=+-1")
    ax2.set_xlim(lowx, lowx + span)
    ax2.set_ylim(lowy, lowy + span)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Stationary density (contour)", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    out_png = out_dir / "canard_hybrid_density.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
