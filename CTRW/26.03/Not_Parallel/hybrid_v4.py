
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
from pathlib import Path


@nb.njit(cache=True, fastmath=True)
def drift(x: float, v: float, gamma: float) -> tuple[float, float]:
    # dX = V dt, dV = (-X - gamma V) dt + sigma dW
    mu_x = v
    mu_v = -x - gamma * v
    return mu_x, mu_v


def ground_true_density(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    # True invariant density for U(x)=x^2/2:
    # rho(x,v) proportional to exp(-beta * (x^2/2 + v^2/2)), beta = 2*gamma/sigma^2
    beta = 2.0 * gamma / (sigma * sigma)
    x_mesh, v_mesh = np.meshgrid(x_grid, v_grid, indexing="ij")
    energy = 0.5 * x_mesh * x_mesh + 0.5 * v_mesh * v_mesh
    rho = np.exp(-beta * energy)
    return rho


def normalize_density(rho: np.ndarray, h_x: float, h_v: float) -> np.ndarray:
    mass = np.sum(rho) * h_x * h_v
    if mass <= 0.0:
        return np.zeros_like(rho)
    return rho / mass




@nb.njit(cache=True, fastmath=True)
def discrete_density(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
    # max_iter: int = 8000,
    max_iter: int = 8000,
    tol: float = 1e-11,
) -> np.ndarray:
    n_x = len(x_grid)
    n_v = len(v_grid)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    rate_sum_max = 0.0
    for ix in range(n_x):
        for iv in range(n_v):
            x = x_grid[ix]
            v = v_grid[iv]
            mu_x, mu_v = drift(x, v, gamma)
            m_v = 0.5 * max(sigma * sigma - abs(mu_v) * h_v, 0.0)
            r_xp = max(mu_x, 0.0) / h_x
            r_xm = -min(mu_x, 0.0) / h_x
            r_vp = max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
            r_vm = -min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
            if ix == n_x - 1:
                r_xp = 0.0
            if ix == 0:
                r_xm = 0.0
            if iv == n_v - 1:
                r_vp = 0.0
            if iv == 0:
                r_vm = 0.0
            rate_sum = r_xp + r_xm + r_vp + r_vm
            if rate_sum > rate_sum_max:
                rate_sum_max = rate_sum

    # Uniformization constant
    lam = 1.05 * rate_sum_max + 1e-14

    p = np.full((n_x, n_v), 1.0 / (n_x * n_v), dtype=np.float64)
    p_next = np.zeros_like(p)

    for _ in range(max_iter):
        p_next.fill(0.0)
        for ix in range(n_x):
            for iv in range(n_v):
                x = x_grid[ix]
                v = v_grid[iv]
                mu_x, mu_v = drift(x, v, gamma)
                m_v = 0.5 * max(sigma * sigma - abs(mu_v) * h_v, 0.0)
                r_xp = max(mu_x, 0.0) / h_x
                r_xm = -min(mu_x, 0.0) / h_x
                r_vp = max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                r_vm = -min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                if ix == n_x - 1:
                    r_xp = 0.0
                if ix == 0:
                    r_xm = 0.0
                if iv == n_v - 1:
                    r_vp = 0.0
                if iv == 0:
                    r_vm = 0.0

                r_sum = r_xp + r_xm + r_vp + r_vm
                w = p[ix, iv]

                p_next[ix, iv] += w * (1.0 - r_sum / lam)
                if r_xp > 0.0:
                    p_next[ix + 1, iv] += w * (r_xp / lam)
                if r_xm > 0.0:
                    p_next[ix - 1, iv] += w * (r_xm / lam)
                if r_vp > 0.0:
                    p_next[ix, iv + 1] += w * (r_vp / lam)
                if r_vm > 0.0:
                    p_next[ix, iv - 1] += w * (r_vm / lam)

        p_next /= np.sum(p_next)
        max_diff = 0.0
        for ix in range(n_x):
            for iv in range(n_v):
                diff = abs(p_next[ix, iv] - p[ix, iv])
                if diff > max_diff:
                    max_diff = diff
        if max_diff < tol:
            p = p_next.copy()
            break
        p, p_next = p_next, p

    # Convert pmf on grid nodes to density
    rho = p / (h_x * h_v)
    return rho


def l1_error(rho_a: np.ndarray, rho_b: np.ndarray, h_x: float, h_v: float) -> float:
    return float(np.sum(np.abs(rho_a - rho_b)) * h_x * h_v)

def l1_error_result(n_v: int) -> tuple[float, float]:
    t_total_start = time.perf_counter()

    gamma = 1.0
    sigma = 1.0

    # Grid and truncated domain
    n_x = n_v**2
    # n_x = 10*(n_v-1)+1
    # n_x = n_v
    l_x = 4.0
    l_v = 4.0

    x_grid = np.linspace(-l_x, l_x, n_x)
    v_grid = np.linspace(-l_v, l_v, n_v)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    t0 = time.perf_counter()
    rho_true_raw = ground_true_density(x_grid, v_grid, gamma, sigma)
    rho_true = normalize_density(rho_true_raw, h_x, h_v)
    t_true = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho_disc_raw = discrete_density(x_grid, v_grid, gamma, sigma)
    rho_disc = normalize_density(rho_disc_raw, h_x, h_v)
    t_disc = time.perf_counter() - t0

    err_disc_true = l1_error(rho_disc, rho_true, h_x, h_v)

    t_total = time.perf_counter() - t_total_start

    print(f"n_v={n_v:4d}, h_v={h_v:.6e}, L1={err_disc_true:.6e}")
    print(f"Time true density:      {t_true:.6f} s")
    print(f"Time discrete density:  {t_disc:.6f} s")
    print(f"Total time:             {t_total:.6f} s")

    return h_v, err_disc_true

def main():
    out_dir = Path(__file__).resolve().parent
    # n_v_list = [10, 50, 100, 150, 200]
    n_v_list = [ 40, 50, 60, 70, 80]
    # n_v_list = [32, 64, 128, 256, 1024]

    # JIT warm-up: avoid counting first-time compilation in benchmark results.
    _ = discrete_density(
        np.linspace(-1.0, 1.0, 8),
        np.linspace(-1.0, 1.0, 8),
        1.0,
        1.0,
        max_iter=2,
        tol=1e-8,
    )

    h_v_vals = []
    err_vals = []
    for n_v in n_v_list:
        h_v, err_disc_true = l1_error_result(n_v)
        h_v_vals.append(h_v)
        err_vals.append(err_disc_true)

    h_v_vals = np.array(h_v_vals)
    err_vals = np.array(err_vals)

    # Sort by h_v for cleaner plotting and slope diagnostics.
    sort_idx = np.argsort(h_v_vals)
    h_v_vals = h_v_vals[sort_idx]
    err_vals = err_vals[sort_idx]

    # Build reference lines O(h_v) and O(h_v^2), anchored at one data point.
    i_anchor = max(len(h_v_vals) - 2, 0)
    h_anchor = h_v_vals[i_anchor]
    err_anchor = err_vals[i_anchor]
    ref_o1 = err_anchor * (h_v_vals / h_anchor)
    ref_o2 = err_anchor * (h_v_vals / h_anchor) ** 2

    # Empirical convergence orders.
    local_orders = np.log(err_vals[1:] / err_vals[:-1]) / np.log(h_v_vals[1:] / h_v_vals[:-1])
    global_order, _ = np.polyfit(np.log(h_v_vals), np.log(err_vals), 1)
    print("local orders:", local_orders)
    print(f"global fitted order: {global_order:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h_v_vals, err_vals, "o-", lw=1.2, ms=6, label="L1 error")
    ax.loglog(h_v_vals, ref_o1, "--", lw=1.0, label="O(h_v)")
    ax.loglog(h_v_vals, ref_o2, "--", lw=1.0, label="O(h_v^2)")
    ax.set_xlabel("h_v")
    ax.set_ylabel("L1 error")
    ax.set_title("L1 Error vs h_v (log-log)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = out_dir / "hybrid_v4_l1_error_vs_hv.png"
    fig.savefig(out_png, dpi=300)
    print(f"Saved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
