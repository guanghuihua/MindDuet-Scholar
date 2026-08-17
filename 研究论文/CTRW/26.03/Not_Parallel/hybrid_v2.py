from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


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


def ssa_stationary_density(
    n_x: int,
    n_v: int,
    l_x: float,
    l_v: float,
    gamma: float,
    sigma: float,
    t_burn: float,
    t_sample: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)

    x_grid = np.linspace(-l_x, l_x, n_x)
    v_grid = np.linspace(-l_v, l_v, n_v)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    ix = n_x // 2
    iv = n_v // 2
    t = 0.0
    t_end = t_burn + t_sample
    occ = np.zeros((n_x, n_v), dtype=np.float64)

    while t < t_end:
        x = x_grid[ix]
        v = v_grid[iv]
        mu_x, mu_v = drift(x, v, gamma)

        m_v = 0.5 * max(sigma * sigma - abs(mu_v) * h_v, 0.0)
        r_xp = max(mu_x, 0.0) / h_x
        r_xm = -min(mu_x, 0.0) / h_x
        r_vp = max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
        r_vm = -min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)

        # Reflecting boundaries
        if ix == n_x - 1:
            r_xp = 0.0
        if ix == 0:
            r_xm = 0.0
        if iv == n_v - 1:
            r_vp = 0.0
        if iv == 0:
            r_vm = 0.0

        rate_sum = r_xp + r_xm + r_vp + r_vm
        if rate_sum <= 0.0:
            break

        u1 = rng.random()
        dt = -np.log(max(1.0 - u1, 1e-15)) / rate_sum
        if t >= t_burn:
            occ[ix, iv] += dt
        t += dt

        u2 = rng.random() * rate_sum
        if u2 < r_xp:
            ix += 1
        elif u2 < r_xp + r_xm:
            ix -= 1
        elif u2 < r_xp + r_xm + r_vp:
            iv += 1
        else:
            iv -= 1

    rho_hat = normalize_density(occ, h_x, h_v)
    return rho_hat, x_grid, v_grid, h_x, h_v


def stationary_density_discrete(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
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
        if np.max(np.abs(p_next - p)) < tol:
            p = p_next.copy()
            break
        p, p_next = p_next, p

    # Convert pmf on grid nodes to density
    rho = p / (h_x * h_v)
    return rho


def main() -> None:
    gamma = 1.0
    sigma = 1.0

    # Grid and truncated domain
    n_x = 121
    n_v = 121
    l_x = 4.0
    l_v = 4.0

    # SSA settings
    t_burn = 100.0
    t_sample = 10000.0
    seed = 20260311

    rho_ssa, x_grid, v_grid, h_x, h_v = ssa_stationary_density(
        n_x=n_x,
        n_v=n_v,
        l_x=l_x,
        l_v=l_v,
        gamma=gamma,
        sigma=sigma,
        t_burn=t_burn,
        t_sample=t_sample,
        seed=seed,
    )

    rho_true = ground_true_density(x_grid, v_grid, gamma, sigma)
    rho_true = normalize_density(rho_true, h_x, h_v)

    rho_stat = stationary_density_discrete(x_grid, v_grid, gamma, sigma)
    rho_stat = normalize_density(rho_stat, h_x, h_v)

    err_ssa_true = l1_error(rho_ssa, rho_true, h_x, h_v)
    err_stat_true = l1_error(rho_stat, rho_true, h_x, h_v)

    print(f"L1 error (SSA density vs true density) = {err_ssa_true:.6e}")
    print(f"L1 error (stationary density vs true density) = {err_stat_true:.6e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = (x_grid[0], x_grid[-1], v_grid[0], v_grid[-1])

    im0 = axes[0].imshow(rho_true.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0].set_title("True Density")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rho_ssa.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[1].set_title("SSA Density")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(rho_stat.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[2].set_title("Discrete Stationary Density")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("v")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()