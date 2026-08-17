import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import os
os.chdir(os.path.dirname(__file__))

nb.config.NUMBA_DEFAULT_NUM_THREADS = 28

@nb.njit(fastmath=True)
def tau_leaping_canard_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    sample_size: int,
    tau: float,
) -> tuple:
    counts = np.zeros((n + 1, n + 1), dtype=np.float64)
    h = span / n
    inv_h = 1.0 / h
    lowx_center = lowx + h / 2.0
    lowy_center = lowy + h / 2.0
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01
    valid_count = 0
    out_of_bounds_counts = 0

    trajectory_point_x = lowx + np.random.random() * span
    trajectory_point_y = lowy + np.random.random() * span

    while valid_count < sample_size:
        mu_1 = (trajectory_point_y - np.power(trajectory_point_x, 3) / 3 + trajectory_point_x) / delta
        mu_2 = a - trajectory_point_x

        # Wen_Lorenz-style tau-leaping: fixed tau, Poisson jumps
        m1 = max(2 - abs(mu_1) * h, 0.0) / 2.0
        m2 = max(2 - abs(mu_2) * h, 0.0) / 2.0

        trajectory_point_x = lowx_center + round((trajectory_point_x - lowx_center) * inv_h) * h
        trajectory_point_y = lowy_center + round((trajectory_point_y - lowy_center) * inv_h) * h

        q0 = tau * (max(mu_1, 0.0) / h + m1 / (h ** 2))
        q1 = tau * (-min(mu_1, 0.0) / h + m1 / (h ** 2))
        q2 = tau * (max(mu_2, 0.0) / h + m2 / (h ** 2))
        q3 = tau * (-min(mu_2, 0.0) / h + m2 / (h ** 2))

        k0 = np.random.poisson(q0)
        k1 = np.random.poisson(q1)
        k2 = np.random.poisson(q2)
        k3 = np.random.poisson(q3)

        trajectory_point_x += (k0 - k1) * h
        trajectory_point_y += (k2 - k3) * h

        x_n = int(round((trajectory_point_x - lowx_center) * inv_h)) + 1
        y_n = int(round((trajectory_point_y - lowy_center) * inv_h)) + 1
        if 1 <= x_n <= n + 1 and 1 <= y_n <= n + 1:
            counts[x_n - 1, y_n - 1] += tau
            valid_count += 1
        else:
            out_of_bounds_counts += 1
            if x_n < 1:
                trajectory_point_x = lowx_center
            elif x_n > n + 1:
                trajectory_point_x = lowx + span - h / 2.0
            if y_n < 1:
                trajectory_point_y = lowy_center
            elif y_n > n + 1:
                trajectory_point_y = lowy + span - h / 2.0

    return counts.T.ravel(), out_of_bounds_counts


@nb.njit(fastmath=True, parallel=True)
def run_ensemble_tau(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    sample_size: int,
    tau: float,
    loops: int,
    bin_factor: int,
) -> tuple:
    density_sum = np.zeros((n + 1) * (n + 1), dtype=np.float64)
    out_total = 0
    for i in nb.prange(loops):
        data, out_count = tau_leaping_canard_timeweighted(
            lowx, lowy, span, n, eps, sample_size, tau
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


def main():
    lowx = -3.0
    lowy = -3.0
    span = 6.0
    n = 3000
    eps = 0.3
    # sample_size = 100_000_000
    sample_size = int(3E9)
    tau = 0.01
    loops = nb.get_num_threads()
    out_n = 500
    if n % out_n != 0:
        raise ValueError("n must be divisible by out_n")
    bin_factor = n // out_n

    start_time = time.perf_counter()
    data, out_of_bounds_counts = run_ensemble_tau(
        lowx, lowy, span, n, eps, sample_size, tau, loops, bin_factor
    )
    end_time = time.perf_counter()
    print(f"Tau-leaping Canard Simulation Time: {end_time - start_time:.4f} seconds")
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

    if bin_factor > 1:
        x = np.linspace(lowx + h_eff / 2.0, lowx + span - h_eff / 2.0, out_n)
        y = np.linspace(lowy + h_eff / 2.0, lowy + span - h_eff / 2.0, out_n)
    else:
        x = np.linspace(lowx, lowx + span, n + 1)
        y = np.linspace(lowy, lowy + span, n + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.plot_surface(X, Y, data, cmap="viridis")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Density")
    ax1.set_title("Tau-leaping Canard Time-Weighted Density")

    fig2 = plt.figure(figsize=(7, 6))
    ax2 = fig2.add_subplot(111)
    extent = (x[0], x[-1], y[0], y[-1])
    im = ax2.imshow(data.T, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_title("Tau-leaping Canard Density Heatmap")
    fig2.colorbar(im, ax=ax2, label="Density")

    fig.savefig("v4_density_3d.png", dpi=300)
    fig2.savefig("v4_density_heatmap.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
