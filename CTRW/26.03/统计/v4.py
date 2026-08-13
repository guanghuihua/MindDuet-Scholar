"""
SSA stationary density simulation for the stochastic Canard system.
Non-uniform grid: fast variable x (no noise) uses n_x = n_y^2 grid points,
                  slow variable y (with noise) uses n_y grid points.

System equations:
    epsilon * dx/dt = y - x^3/3 + x        (fast variable, no noise)
    dy/dt = (a - x) dt + sigma * dW_t       (slow variable, with noise)

Grid design (adapted from the Langevin code where n_x = n_v^2):
    Langevin: x has no noise, v has noise  ->  n_x = n_v^2
    Canard:   x has no noise (fast), y has noise (slow)  ->  n_x = n_y^2

    Rationale: x has no physical diffusion; upwind numerical diffusion ~ |mu_x|*h_x/2.
    In Canard, |mu_x| ~ O(1/eps), much larger than |v| ~ O(1) in Langevin.
    To suppress numerical diffusion we need h_x << h_y, so n_x = n_y^2 is a
    practical compromise between accuracy and computational cost.

Jump rates (Chang-Cooper hybrid scheme):
    mu_x = (y - x^3/3 + x) / eps    (x-drift, O(1/eps))
    mu_y = a - x                     (y-drift, O(1))
    m_y  = max(sigma^2 - |mu_y|*h_y, 0) / 2   (diffusion mixing term, y only)
    m_x  = 0                                    (no diffusion in x)

    q_x+ = max(mu_x, 0) / h_x
    q_x- = max(-mu_x, 0) / h_x
    q_y+ = max(mu_y, 0) / h_y + m_y / h_y^2
    q_y- = max(-mu_y, 0) / h_y + m_y / h_y^2

Parameters (identical to the original SSA Canard code):
    delta (epsilon) = 0.1
    sigma^2 = 2.0
    a = 1 - delta/8 - 3*delta^2/32 - 173*delta^3/1024 - 0.01
"""

from __future__ import annotations
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time
from pathlib import Path


# -----------------------------------------------------------------------------
#  System parameters (identical to the original SSA Canard code)
# -----------------------------------------------------------------------------
DELTA    = 0.1
SIGMA_SQ = 2
A_PARAM  = 1 - DELTA/8 - 3*DELTA**2/32 - 173*DELTA**3/1024 - 0.01


# -----------------------------------------------------------------------------
#  Core SSA function (numba JIT)
# -----------------------------------------------------------------------------

@nb.njit(fastmath=True, cache=True)
def ssa_canard_nonuniform(
    lowx: float,
    lowy: float,
    span: float,
    n_y: int,        # slow variable (with noise): n_y grid points
    n_x: int,        # fast variable (no noise):   n_x = n_y^2 grid points
    eps: float,
    sigma_sq: float,
    a: float,
    sample_size: int,
) -> tuple:
    """
    Single-chain time-weighted SSA.
    Returns (counts[n_x, n_y], out_of_bounds_count).
    """
    h_x = span / n_x
    h_y = span / n_y

    counts = np.zeros((n_x, n_y), dtype=np.float64)

    # random initial point
    x = lowx + np.random.random() * span
    y = lowy + np.random.random() * span

    valid_count = 0
    oob_count   = 0

    while valid_count < sample_size:

        # drift terms
        mu_x = (y - x * x * x / 3.0 + x) / eps   # O(1/eps)
        mu_y = a - x                                # O(1)

        # Chang-Cooper mixing term: y-direction only (has noise)
        # x-direction: m_x = 0 (no diffusion)
        m_y = 0.5 * max(sigma_sq - abs(mu_y) * h_y, 0.0)

        # jump rates for four directions
        q_xp = max(mu_x,  0.0) / h_x                      # x right
        q_xm = max(-mu_x, 0.0) / h_x                      # x left
        q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y) # y up
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y) # y down
        lam  = q_xp + q_xm + q_yp + q_ym

        # Gillespie waiting time
        tau = -np.log(1.0 - np.random.random()) / lam

        # grid indices
        ix = int((x - lowx) / h_x)
        iy = int((y - lowy) / h_y)

        if 0 <= ix < n_x and 0 <= iy < n_y:
            counts[ix, iy] += tau   # time-weighted accumulation
            valid_count += 1
        else:
            oob_count += 1
            # reflect at boundaries
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


@nb.njit(fastmath=True, parallel=True, cache=True)
def run_ensemble_nonuniform(
    lowx: float,
    lowy: float,
    span: float,
    n_y: int,
    n_x: int,
    eps: float,
    sigma_sq: float,
    a: float,
    sample_size: int,
    loops: int,
    bin_factor_x: int,
    bin_factor_y: int,
) -> tuple:
    """
    Run `loops` independent chains in parallel, average results,
    then coarsen the output grid by bin_factor_x / bin_factor_y.
    """
    flat_size   = n_x * n_y
    density_sum = np.zeros(flat_size, dtype=np.float64)
    oob_total   = 0

    for i in nb.prange(loops):
        c, oob = ssa_canard_nonuniform(
            lowx, lowy, span, n_y, n_x, eps, sigma_sq, a, sample_size
        )
        density_sum += c.ravel()
        oob_total   += oob

    density_mean = density_sum / loops

    # coarsen output grid
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
#  Comparison experiment: uniform grid vs non-uniform grid
# -----------------------------------------------------------------------------

def run_comparison(
    lowx=-3.0, lowy=-3.0, span=6.0,
    n_y=60, sample_size=int(3e6), loops=5,
    out_n=100,
):
    """
    Compare uniform grid (n_x = n_y) and non-uniform grid (n_x = n_y^2)
    under equal n_y.
    """
    eps      = DELTA
    sigma_sq = SIGMA_SQ
    a        = A_PARAM

    results = {}

    for label, n_x in [("uniform",    n_y),
                        ("nonuniform", n_y ** 2)]:

        # ensure divisibility for binning
        bx = max(n_x // out_n, 1)
        by = max(n_y // out_n, 1)
        nx = out_n * bx
        ny = out_n * by

        h_x = span / nx
        h_y = span / ny

        print(f"\n{'='*55}")
        print(f"{label}: n_x={nx}, n_y={ny}  "
              f"(h_x={h_x:.5f}, h_y={h_y:.4f})")
        print(f"  numerical diffusion ~ |mu_x|*h_x/2 = {h_x/(2*eps):.5f}")

        # JIT warm-up
        run_ensemble_nonuniform(
            lowx, lowy, span, ny, nx, eps, sigma_sq, a,
            100, 1, bx, by
        )

        t0 = time.perf_counter()
        data_flat, oob = run_ensemble_nonuniform(
            lowx, lowy, span, ny, nx, eps, sigma_sq, a,
            sample_size, loops, bx, by
        )
        elapsed = time.perf_counter() - t0

        # normalise
        h_eff = span / out_n
        data  = data_flat.reshape((out_n, out_n))
        total = data.sum()
        if total > 0:
            data /= (h_eff * h_eff * total)

        print(f"  time: {elapsed:.1f}s,  oob: {oob},  "
              f"density integral: {data.sum()*h_eff*h_eff:.6f}")
        results[label] = data

    return results, out_n, span, lowx, lowy


# -----------------------------------------------------------------------------
#  Plotting
# -----------------------------------------------------------------------------

def plot_results(results, out_n, span, lowx, lowy, out_dir):
    h     = span / out_n
    x_arr = np.linspace(lowx + h/2, lowx + span - h/2, out_n)
    y_arr = np.linspace(lowy + h/2, lowy + span - h/2, out_n)
    X, Y  = np.meshgrid(x_arr, y_arr, indexing="ij")

    # slow manifold curve
    x_mf = np.linspace(-2.5, 2.5, 500)
    y_mf = x_mf**3 / 3 - x_mf

    labels = list(results.keys())
    titles = {
        "uniform":    f"Uniform grid  (n_x = n_y)\n"
                      f"eps={DELTA}, sigma={np.sqrt(SIGMA_SQ):.3f}",
        "nonuniform": f"Non-uniform grid  (n_x = n_y^2)\n"
                      f"eps={DELTA}, sigma={np.sqrt(SIGMA_SQ):.3f}",
    }

    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 6))
    if len(labels) == 1:
        axes = [axes]

    vmax = max(d.max() for d in results.values())

    for ax, label in zip(axes, labels):
        data = results[label]
        cf   = ax.contourf(X, Y, data, levels=20, cmap="viridis",
                           vmin=0, vmax=vmax)
        plt.colorbar(cf, ax=ax, label="Density")
        ax.plot(x_mf, y_mf, 'r--', lw=1.5, label="Slow manifold")
        ax.axvline(x= 1.0, color="orange", ls=":", lw=1.2, alpha=0.8)
        ax.axvline(x=-1.0, color="orange", ls=":", lw=1.2, alpha=0.8,
                   label="Saddle nodes x=+-1")
        ax.set_xlim(lowx, lowx + span)
        ax.set_ylim(lowy, lowy + span)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(titles.get(label, label), fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.15)

    fig.tight_layout()
    p = out_dir / "canard_nonuniform_density.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)

    # difference plot
    if "uniform" in results and "nonuniform" in results:
        diff = results["nonuniform"] - results["uniform"]
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        vm  = np.abs(diff).max()
        cf2 = ax2.contourf(X, Y, diff, levels=20, cmap="RdBu_r",
                           vmin=-vm, vmax=vm)
        plt.colorbar(cf2, ax=ax2, label="Density difference")
        ax2.plot(x_mf, y_mf, "k--", lw=1.5, label="Slow manifold")
        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("y", fontsize=12)
        ax2.set_title(
            "Difference: non-uniform minus uniform\n"
            "(shows x-direction resolution effect)",
            fontsize=11
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.15)
        fig2.tight_layout()
        p2 = out_dir / "canard_density_diff.png"
        fig2.savefig(p2, dpi=180, bbox_inches="tight")
        print(f"Saved: {p2}")
        plt.close(fig2)


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent

    print("Stochastic Canard SSA: uniform grid vs non-uniform grid (n_x = n_y^2)")
    print(f"System: eps={DELTA:.3f}, sigma={np.sqrt(SIGMA_SQ):.4f}, a={A_PARAM:.6f}")
    print()

    # print grid scheme comparison
    span = 6.0
    n_y  = 60
    print(f"{'Scheme':22s} {'n_x':>8} {'n_y':>6} {'h_x':>10} {'h_y':>8} "
          f"{'num_diff_x':>12}  note")
    print("-" * 78)
    for label, nx, ny in [
        ("uniform (original)",  n_y,             n_y),
        ("non-uniform n_x=n_y^2", n_y**2,        n_y),
        ("strict CFL",          int(n_y**2/DELTA), n_y),
    ]:
        hx   = span / nx
        hy   = span / ny
        ndx  = hx / (2 * DELTA)   # |mu_x_typical| * h_x / 2 ~ h_x/(2*eps)
        note = "this work" if nx == n_y**2 else \
               ("too costly" if nx > n_y**2 else "original")
        print(f"{label:22s} {nx:8d} {ny:6d} {hx:10.6f} {hy:8.5f} "
              f"{ndx:12.5f}  {note}")

    print()

    # run comparison
    results, out_n, span, lowx, lowy = run_comparison(
        n_y=60,
        sample_size=int(3e6),
        loops=5,
        out_n=100,
    )

    # plot
    print("\nPlotting...")
    plot_results(results, out_n, span, lowx, lowy, out_dir)

    # summary statistics
    print("\nSummary:")
    h = span / out_n
    x_arr = np.linspace(lowx + h/2, lowx + span - h/2, out_n)
    y_arr = np.linspace(lowy + h/2, lowy + span - h/2, out_n)
    for label, data in results.items():
        d_x   = data.sum(axis=1) * h
        d_y   = data.sum(axis=0) * h
        ix_pk = np.argmax(d_x)
        iy_pk = np.argmax(d_y)
        print(f"  {label}: x_peak={x_arr[ix_pk]:.3f}, "
              f"y_peak={y_arr[iy_pk]:.3f}, "
              f"density_max={data.max():.4f}")


if __name__ == "__main__":
    main()