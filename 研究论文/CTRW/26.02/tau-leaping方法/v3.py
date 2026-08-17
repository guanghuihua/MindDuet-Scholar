from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

nb.config.NUMBA_DEFAULT_NUM_THREADS = 28


@nb.njit(fastmath=True)
def ssa_canard_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
) -> tuple:
    counts = np.zeros((n + 1, n + 1), dtype=np.float64)
    h = span / n
    inv_h = 1.0 / h
    lowx_center = lowx + h / 2.0
    lowy_center = lowy + h / 2.0
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01

    out_of_bounds_counts = 0
    x = lowx + np.random.random() * span
    y = lowy + np.random.random() * span
    t = 0.0

    while t < t_stop:
        mu_1 = (y - x**3 / 3.0 + x) / delta
        mu_2 = a - x
        m1 = (eps**2) * max(2.0 - abs(mu_1) * h, 0.0) / 2.0
        m2 = (eps**2) * max(2.0 - abs(mu_2) * h, 0.0) / 2.0

        x = lowx_center + round((x - lowx_center) * inv_h) * h
        y = lowy_center + round((y - lowy_center) * inv_h) * h

        q0 = max(mu_1, 0.0) / h + m1 / (h**2)
        q1 = -min(mu_1, 0.0) / h + m1 / (h**2)
        q2 = max(mu_2, 0.0) / h + m2 / (h**2)
        q3 = -min(mu_2, 0.0) / h + m2 / (h**2)
        lam = q0 + q1 + q2 + q3
        if lam <= 0.0:
            break
        r1 = np.random.random()
        r2 = np.random.random()
        tau = -np.log(r2) / lam

        x_n = int(round((x - lowx_center) * inv_h)) + 1
        y_n = int(round((y - lowy_center) * inv_h)) + 1
        if t >= burn_time:
            if 1 <= x_n <= n + 1 and 1 <= y_n <= n + 1:
                counts[x_n - 1, y_n - 1] += tau
            else:
                out_of_bounds_counts += 1

        if q0 >= r1 * lam:
            x += h
        elif q0 + q1 >= r1 * lam:
            x -= h
        elif q0 + q1 + q2 >= r1 * lam:
            y += h
        else:
            y -= h

        t += tau

    return counts.T.ravel(), out_of_bounds_counts


@nb.njit(fastmath=True, parallel=True)
def run_ensemble_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
    loops: int,
    bin_factor: int,
) -> tuple:
    density_sum = np.zeros((n + 1) * (n + 1), dtype=np.float64)
    out_total = 0
    for i in nb.prange(loops):
        data, out_count = ssa_canard_timeweighted(
            lowx, lowy, span, n, eps, t_stop, burn_time
        )
        density_sum += data
        out_total += out_count
    density_mean = density_sum / loops

    if bin_factor > 1:
        out_n = n // bin_factor
        coarse = np.zeros((out_n, out_n), dtype=np.float64)
        fine = density_mean.reshape((n + 1, n + 1)).T
        for i in range(out_n):
            i0 = i * bin_factor
            for j in range(out_n):
                j0 = j * bin_factor
                s = 0.0
                for di in range(bin_factor):
                    for dj in range(bin_factor):
                        s += fine[i0 + di, j0 + dj]
                coarse[i, j] = s
        return coarse.T.ravel(), out_total

    return density_mean, out_total


def main() -> None:
    nb.set_num_threads(os.cpu_count() or 28)
    lowx = -3.0
    lowy = -3.0
    span = 6.0
    n = 6000
    eps = 0.3
    t_stop = 2.0e5
    burn_time = 2.0e4
    loops = nb.get_num_threads()
    out_n = 600
    if n % out_n != 0:
        raise ValueError("n must be divisible by out_n")
    bin_factor = n // out_n

    start_time = time.perf_counter()
    data, out_of_bounds_counts = run_ensemble_timeweighted(
        lowx, lowy, span, n, eps, t_stop, burn_time, loops, bin_factor
    )
    end_time = time.perf_counter()
    print(f"SSA Canard Simulation Time: {end_time - start_time:.4f} seconds")
    print(f"Out of bounds counts: {out_of_bounds_counts}")

    h = span / n
    if bin_factor > 1:
        h_eff = h * bin_factor
        data = data.reshape((out_n, out_n), order="F")
    else:
        h_eff = h
        data = data.reshape((n + 1, n + 1), order="F")
    total = data.sum()
    if total > 0:
        data /= (h_eff**2 * total)
    else:
        raise ValueError("No samples landed in the grid. Check domain or burn_time.")

    if bin_factor > 1:
        x = np.linspace(lowx + h_eff / 2.0, lowx + span - h_eff / 2.0, out_n)
        y = np.linspace(lowy + h_eff / 2.0, lowy + span - h_eff / 2.0, out_n)
    else:
        x = np.linspace(lowx, lowx + span, n + 1)
        y = np.linspace(lowy, lowy + span, n + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_surface(X, Y, data, cmap="viridis")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Density")
    ax1.set_title("SSA Canard Time-Weighted Density")
    fig1.tight_layout()
    fig1.savefig("26.02_v3_density_3d.png", dpi=300)
    fig1.show()

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    extent = (x[0], x[-1], y[0], y[-1])
    im = ax2.imshow(data.T, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_title("SSA Canard Density Heatmap")
    fig2.colorbar(im, ax=ax2, label="Density")
    fig2.tight_layout()
    fig2.savefig("26.02_v3_density_heatmap.png", dpi=300)
    fig2.show()


if __name__ == "__main__":
    main()

