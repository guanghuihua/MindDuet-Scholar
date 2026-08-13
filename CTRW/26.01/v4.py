import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb

nb.config.NUMBA_DEFAULT_NUM_THREADS = 28

"""
次数跑
次数加权
"""

@nb.njit(fastmath=True)
def ssa_canard_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    sample_size: int,
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
        m1 = max(2 - abs(mu_1) * h, 0.0) / 2.0
        m2 = max(2 - abs(mu_2) * h, 0.0) / 2.0

        trajectory_point_x = lowx_center + round((trajectory_point_x - lowx_center) * inv_h) * h
        trajectory_point_y = lowy_center + round((trajectory_point_y - lowy_center) * inv_h) * h

        q0 = max(mu_1, 0.0) / h + m1 / (h ** 2)
        q1 = -min(mu_1, 0.0) / h + m1 / (h ** 2)
        q2 = max(mu_2, 0.0) / h + m2 / (h ** 2)
        q3 = -min(mu_2, 0.0) / h + m2 / (h ** 2)
        lam = q0 + q1 + q2 + q3
        r1 = np.random.random()
        r2 = np.random.random()
        tau = -np.log(1.0 - r2) / lam

        x_n = int(round((trajectory_point_x - lowx_center) * inv_h)) + 1
        y_n = int(round((trajectory_point_y - lowy_center) * inv_h)) + 1
        if 1 <= x_n <= n + 1 and 1 <= y_n <= n + 1:
            counts[x_n - 1, y_n - 1] += 1.0
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

        if q0 >= r1 * lam:
            trajectory_point_x += h
        elif q0 + q1 >= r1 * lam:
            trajectory_point_x -= h
        elif q0 + q1 + q2 >= r1 * lam:
            trajectory_point_y += h
        else:
            trajectory_point_y -= h

    return counts.T.ravel(), out_of_bounds_counts


@nb.njit(fastmath=True, parallel=True)
def run_ensemble_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    sample_size: int,
    loops: int,
    bin_factor: int,
) -> tuple:
    density_sum = np.zeros((n + 1) * (n + 1), dtype=np.float64)
    out_total = 0
    for i in nb.prange(loops):
        data, out_count = ssa_canard_timeweighted(lowx, lowy, span, n, eps, sample_size)
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
    n = 6000
    eps = 0.3
    # sample_size = 1_000_000
    sample_size = int(5E9)
    loops = 28
    out_n = 1000
    if n % out_n != 0:
        raise ValueError("n must be divisible by out_n")
    bin_factor = n // out_n

    start_time = time.perf_counter()
    data, out_of_bounds_counts = run_ensemble_timeweighted(
        lowx, lowy, span, n, eps, sample_size, loops, bin_factor
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
    ax1.set_title("SSA Canard Time-Weighted Density")

    plt.show()


if __name__ == "__main__":
    main()

"""
1. 使用Numba加速
2. 使用多尺度方法
3. 使用ensemble方法
4. 使用时间权重
"""
