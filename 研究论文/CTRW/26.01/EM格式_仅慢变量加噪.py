import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb

nb.config.NUMBA_DEFAULT_NUM_THREADS = 28


@nb.njit(fastmath=True)
def em_canard_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
    dt: float,
) -> tuple:
    counts = np.zeros((n, n), dtype=np.float64)
    h = span / n
    inv_h = 1.0 / h
    lowx_center = lowx + h / 2.0
    lowy_center = lowy + h / 2.0
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01

    out_of_bounds_counts = 0
    trajectory_point_x = lowx + np.random.random() * span
    trajectory_point_y = lowy + np.random.random() * span
    t = 0.0

    while t < t_stop:
        # Drift terms
        mu_1 = (trajectory_point_y - trajectory_point_x**3 / 3.0 + trajectory_point_x) / delta
        mu_2 = a - trajectory_point_x

        # Euler-Maruyama step: noise only on slow variable y
        trajectory_point_x += mu_1 * dt
        trajectory_point_y += mu_2 * dt + eps * np.sqrt(dt) * np.random.randn()

        ix = int(round((trajectory_point_x - lowx_center) * inv_h))
        iy = int(round((trajectory_point_y - lowy_center) * inv_h))

        if t >= burn_time:
            if 0 <= ix < n and 0 <= iy < n:
                counts[ix, iy] += dt
            else:
                out_of_bounds_counts += 1

        if ix < 0:
            trajectory_point_x = lowx_center
        elif ix >= n:
            trajectory_point_x = lowx + span - h / 2.0
        if iy < 0:
            trajectory_point_y = lowy_center
        elif iy >= n:
            trajectory_point_y = lowy + span - h / 2.0

        t += dt

    return counts.ravel(), out_of_bounds_counts


@nb.njit(fastmath=True, parallel=True)
def run_ensemble(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
    dt: float,
    loops: int,
) -> tuple:
    density_sum = np.zeros(n * n, dtype=np.float64)
    out_total = 0
    for i in nb.prange(loops):
        data, out_count = em_canard_timeweighted(lowx, lowy, span, n, eps, t_stop, burn_time, dt)
        density_sum += data
        out_total += out_count
    density_mean = density_sum / loops
    return density_mean, out_total


@nb.njit(fastmath=True)
def bin_coarse(fine: np.ndarray, n: int, bin_factor: int) -> np.ndarray:
    out_n = n // bin_factor
    coarse = np.zeros((out_n, out_n), dtype=np.float64)
    for i in range(out_n):
        i0 = i * bin_factor
        for j in range(out_n):
            j0 = j * bin_factor
            s = 0.0
            for di in range(bin_factor):
                for dj in range(bin_factor):
                    s += fine[i0 + di, j0 + dj]
            coarse[i, j] = s
    return coarse


def main():
    nb.set_num_threads(os.cpu_count() if hasattr(os, "cpu_count") else 24)
    lowx = -3.0
    lowy = -3.0
    span = 6.0
    n = 600
    eps = 0.3
    t_stop = 2.0e5
    burn_time = 2.0e3
    dt = 0.005
    loops = 2
    out_n = 100
    if n % out_n != 0:
        raise ValueError("n must be divisible by out_n")
    bin_factor = n // out_n

    start_time = time.perf_counter()
    data, out_of_bounds_counts = run_ensemble(
        lowx, lowy, span, n, eps, t_stop, burn_time, dt, loops
    )
    end_time = time.perf_counter()
    print(f"EM Canard Simulation Time: {end_time - start_time:.4f} seconds")
    print(f"Out of bounds counts: {out_of_bounds_counts}")

    fine = data.reshape((n, n))
    data = bin_coarse(fine, n, bin_factor)
    h_eff = span / out_n
    total = data.sum()
    if total > 0:
        data /= (h_eff**2 * total)

    x = np.linspace(lowx + h_eff / 2.0, lowx + span - h_eff / 2.0, out_n)
    y = np.linspace(lowy + h_eff / 2.0, lowy + span - h_eff / 2.0, out_n)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.plot_surface(X, Y, data, cmap="viridis")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Density")
    ax1.set_title("EM Canard Density (Noise on Slow Variable)")
    plt.show()


if __name__ == "__main__":
    import os
    main()
