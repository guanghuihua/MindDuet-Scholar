
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb


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


def l1_error(rho_a: np.ndarray, rho_b: np.ndarray, h_x: float, h_v: float) -> float:
    return float(np.sum(np.abs(rho_a - rho_b)) * h_x * h_v)


@nb.njit(cache=True, fastmath=True)
def discrete_density(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
    # max_iter: int = 8000,
    max_iter: int = 800,
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


def main() -> None:
    t_total_start = time.perf_counter()

    gamma = 1.0
    sigma = 1.0

    # Grid and truncated domain
    # n_x = 121
    # n_v = 121
    n_v = 100
    # n_x = 100
    n_x = n_v**2
    l_x = 4.0
    l_v = 4.0

    x_grid = np.linspace(-l_x, l_x, n_x)
    v_grid = np.linspace(-l_v, l_v, n_v)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    # JIT warm-up: avoid counting first-time compilation in runtime stats.
    _ = discrete_density(
        np.linspace(-1.0, 1.0, 8),
        np.linspace(-1.0, 1.0, 8),
        gamma,
        sigma,
        max_iter=2,
        tol=1e-8,
    )

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

    print(f"L1 error (discrete density vs true density) = {err_disc_true:.6e}")
    print(f"Time true density:      {t_true:.6f} s")
    print(f"Time discrete density:  {t_disc:.6f} s")
    print(f"Total time:             {t_total:.6f} s")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = (x_grid[0], x_grid[-1], v_grid[0], v_grid[-1])

    im0 = axes[0].imshow(rho_true.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0].set_title("True Density")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rho_disc.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[1].set_title("Discrete Stationary Density")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("v")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
