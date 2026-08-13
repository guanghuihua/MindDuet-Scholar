from __future__ import annotations

from pathlib import Path
import numba as nb

import matplotlib.pyplot as plt
import numpy as np

@nb.njit(cache=True)
def U_prime(x: float) -> float:
    # U(x) = x^2 / 2
    return x

@nb.njit(cache=True)
def bernoulli_b(z: float) -> float:
    # B(z) = z / (exp(z) - 1), stable near z=0
    az = abs(z)
    if az < 1e-8:
        return 1.0 - 0.5 * z + (z * z) / 12.0
    return z / np.expm1(z)

@nb.njit(cache=True)
def rates_mixed_qc_qu(
    x: float,
    v: float,
    h_x: float,
    h_v: float,
    gamma: float,
    sigma: float,
) -> tuple[float, float, float, float]:
    """
    Mixed rates for underdamped Langevin:
    - x direction (no noise): upwind
    - v direction (has noise): Q_c (Scharfetter-Gummel/Chang-Cooper style)
    """
    # ----- x direction (no noise): upwind -----
    a = v
    r_xp = 0.0
    r_xm = 0.0
    if a > 0.0:
        r_xp = a / h_x
    elif a < 0.0:
        r_xm = -a / h_x

    # ----- v direction (has noise): Q_c -----
    b = -(U_prime(x) + gamma * v)
    d_v = 0.5 * sigma * sigma
    z = b * h_v / d_v
    r_vp = (d_v / (h_v * h_v)) * bernoulli_b(-z)
    r_vm = (d_v / (h_v * h_v)) * bernoulli_b(z)

    return r_xp, r_xm, r_vp, r_vm


@nb.njit(cache=True)
def simulate_ssa_stationary_density(
    n_x: int,
    n_v: int,
    l_x: float,
    l_v: float,
    gamma: float,
    sigma: float,
    t_burn: float,
    t_sample: float,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    np.random.seed(seed)

    x_grid = np.linspace(-l_x, l_x, n_x)
    v_grid = np.linspace(-l_v, l_v, n_v)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    # Start from center
    ix = n_x // 2
    iv = n_v // 2

    occ = np.zeros((n_x, n_v), dtype=np.float64)

    t = 0.0
    t_end = t_burn + t_sample

    while t < t_end:
        x = x_grid[ix]
        v = v_grid[iv]

        r_xp, r_xm, r_vp, r_vm = rates_mixed_qc_qu(x, v, h_x, h_v, gamma, sigma)

        # Reflecting boundaries
        if ix == n_x - 1:
            r_xp = 0.0
        if ix == 0:
            r_xm = 0.0
        if iv == n_v - 1:
            r_vp = 0.0
        if iv == 0:
            r_vm = 0.0

        r_sum = r_xp + r_xm + r_vp + r_vm
        if r_sum <= 0.0:
            break

        u = np.random.random()
        dt = -np.log(max(u, 1e-15)) / r_sum

        # Time-weighted occupancy estimator
        if t >= t_burn:
            occ[ix, iv] += dt

        t += dt

        u2 = np.random.random() * r_sum
        if u2 < r_xp:
            ix += 1
        elif u2 < r_xp + r_xm:
            ix -= 1
        elif u2 < r_xp + r_xm + r_vp:
            iv += 1
        else:
            iv -= 1

    total = np.sum(occ)
    if total <= 0.0:
        return np.zeros((n_x, n_v), dtype=np.float64), h_x, h_v

    rho_hat = occ / (total * h_x * h_v)
    return rho_hat, h_x, h_v


def true_invariant_density(
    n_x: int,
    n_v: int,
    l_x: float,
    l_v: float,
    gamma: float,
    sigma: float,
) -> tuple[np.ndarray, float, float]:
    x = np.linspace(-l_x, l_x, n_x)
    v = np.linspace(-l_v, l_v, n_v)
    h_x = x[1] - x[0]
    h_v = v[1] - v[0]

    beta = 2.0 * gamma / (sigma * sigma)
    x_mesh, v_mesh = np.meshgrid(x, v, indexing="ij")
    u = 0.5 * x_mesh * x_mesh
    rho = np.exp(-beta * (u + 0.5 * v_mesh * v_mesh))
    rho /= np.sum(rho) * h_x * h_v

    return rho, h_x, h_v


def l1_error(rho_hat: np.ndarray, rho_true: np.ndarray, h_x: float, h_v: float) -> float:
    return float(np.sum(np.abs(rho_hat - rho_true)) * h_x * h_v)


def fit_order_tail(h_vals: np.ndarray, err_vals: np.ndarray, tail_points: int = 3) -> float:
    x = np.log(h_vals[-tail_points:])
    y = np.log(err_vals[-tail_points:])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def main() -> None:
    # Model params
    gamma = 1.0
    sigma = 1.0

    # Truncation domain
    l_x = 4.0
    l_v = 4.0

    # Refinement in noisy direction (v): h1.
    # Non-noisy direction (x): h2 with h1 = 10 * h2.
    # n_v_list = [41, 61, 81, 101, 121]
    n_v_list = [100, 200, 400, 800, 1600]

    # SSA horizon
    t_burn = 100.0
    t_sample = 10000.0
    n_rep = 10

    # Warm-up (JIT)
    _ = rates_mixed_qc_qu(0.0, 0.0, 0.01, 0.1, gamma, sigma)

    h1_vals = np.zeros(len(n_v_list), dtype=np.float64)
    err_vals = np.zeros(len(n_v_list), dtype=np.float64)

    print("Running mixed SSA: x-upwind (h2), v-Q_c (h1)")
    print(f"gamma={gamma}, sigma={sigma}, domain=[{-l_x},{l_x}]x[{-l_v},{l_v}]")
    print(f"t_burn={t_burn}, t_sample={t_sample}, n_rep={n_rep}, constraint: h2=h1^2")

    for i, n_v in enumerate(n_v_list):
        # h1 = 2*L_v/(n_v-1), enforce h1 = 10*h2 exactly by construction.
        h1 = (2.0 * l_v) / (n_v - 1)
        n_x = 10 * (n_v - 1) + 1
        if n_x < 3:
            n_x = 3

        rho_true, h_x, h_v = true_invariant_density(n_x, n_v, l_x, l_v, gamma, sigma)

        e_sum = 0.0
        for r in range(n_rep):
            rho_hat, _, _ = simulate_ssa_stationary_density(
                n_x=n_x,
                n_v=n_v,
                l_x=l_x,
                l_v=l_v,
                gamma=gamma,
                sigma=sigma,
                t_burn=t_burn,
                t_sample=t_sample,
                seed=10000 * n_v + 37 * r + 1,
            )
            e_sum += l1_error(rho_hat, rho_true, h_x, h_v)

        h1_vals[i] = h_v
        err_vals[i] = e_sum / n_rep
        print(
            f"Nv={n_v:4d}, Nx={n_x:4d}, h1={h_v:.6e}, h2={h_x:.6e}, "
            f"h1/h2={h_v/h_x:.3f}, L1 error={err_vals[i]:.6e}"
        )

    # Reference slopes
    i_anchor = len(h1_vals) - 2
    err_anchor = err_vals[i_anchor]
    h_anchor = h1_vals[i_anchor]
    
    ref1 = err_anchor * (h1_vals / h_anchor)        # O(h1): parallel reference
    ref2 = err_anchor * (h1_vals / h_anchor) ** 2   # O(h1²): parallel reference

    local_slopes = np.log(err_vals[1:] / err_vals[:-1]) / np.log(h1_vals[1:] / h1_vals[:-1])
    tail_order = fit_order_tail(h1_vals, err_vals, tail_points=3)

    print("Local slopes:", local_slopes)
    print("Tail fitted order (last 3 points):", tail_order)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.loglog(h1_vals, err_vals, "o-", color="#1f77b4", mfc="none", lw=1.0, ms=7,
              label="mixed scheme (x: upwind h2, v: Q_c h1)")
    ax.loglog(h1_vals, ref1, "--", color="#d95319", dashes=(7, 5), lw=0.9, label=r"$O(h_1)$")
    ax.loglog(h1_vals, ref2, "--", color="#7e2f8e", dashes=(7, 4), lw=0.9, label=r"$O(h_1^2)$")

    ax.set_title("Underdamped Langevin Invariant Density Accuracy (h2=h1²)")
    ax.set_xlabel("noisy-direction grid size h1")
    ax.set_ylabel(r"$L^1$ error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    out_png = Path(__file__).resolve().parent / "mix_v5_accuracy.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
