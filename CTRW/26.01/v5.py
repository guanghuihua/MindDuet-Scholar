import os
import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

nb.config.NUMBA_DEFAULT_NUM_THREADS = 28

"""
时间跑
次数加权
"""


@nb.njit(fastmath=True)
def ssa_canard_coarse_stat(
    lowx: float,
    lowy: float,
    span: float,
    n_fine: int,
    out_n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
) -> tuple:
    counts = np.zeros((out_n, out_n), dtype=np.float64)
    h = span / n_fine
    inv_h = 1.0 / h
    lowx_center = lowx + h / 2.0
    lowy_center = lowy + h / 2.0
    h_eff = span / out_n
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01
    out_of_bounds_counts = 0

    x = lowx + np.random.random() * span
    y = lowy + np.random.random() * span
    t = 0.0

    while t < t_stop:
        mu_1 = (y - np.power(x, 3) / 3 + x) / delta
        mu_2 = a - x
        m1 = max(2 - abs(mu_1) * h, 0.0) / 2.0
        m2 = max(2 - abs(mu_2) * h, 0.0) / 2.0
        # m1 = (eps ** 2) * max(2 - abs(mu_1) * h, 0.0) / 2.0
        # m2 = (eps ** 2) * max(2 - abs(mu_2) * h, 0.0) / 2.0
        ix = int(round((x - lowx_center) * inv_h))
        iy = int(round((y - lowy_center) * inv_h))
        x = lowx_center + ix * h
        y = lowy_center + iy * h

        q0 = max(mu_1, 0.0) / h + m1 / (h ** 2)
        q1 = -min(mu_1, 0.0) / h + m1 / (h ** 2)
        q2 = max(mu_2, 0.0) / h + m2 / (h ** 2)
        q3 = -min(mu_2, 0.0) / h + m2 / (h ** 2)
        lam = q0 + q1 + q2 + q3
        r1 = np.random.random()
        r2 = np.random.random()
        if lam <= 0.0:
            break
        tau = -np.log(r2) / lam

        if t >= burn_time:
            cx = int((x - lowx) / h_eff)
            cy = int((y - lowy) / h_eff)
            if 0 <= cx < out_n and 0 <= cy < out_n:
                counts[cx, cy] += 1.0
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

        ix = int(round((x - lowx_center) * inv_h))
        iy = int(round((y - lowy_center) * inv_h))
        if ix < 0:
            x = lowx_center
        elif ix >= n_fine:
            x = lowx + span - h / 2.0
        if iy < 0:
            y = lowy_center
        elif iy >= n_fine:
            y = lowy + span - h / 2.0

        t += tau

    return counts.ravel(), out_of_bounds_counts


@nb.njit(fastmath=True, parallel=True)
def run_ensemble(
    lowx: float,
    lowy: float,
    span: float,
    n_fine: int,
    out_n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
    loops: int,
) -> tuple:
    density_sum = np.zeros(out_n * out_n, dtype=np.float64)
    out_total = 0
    for i in nb.prange(loops):
        data, out_count = ssa_canard_coarse_stat(
            lowx, lowy, span, n_fine, out_n, eps, t_stop, burn_time
        )
        density_sum += data
        out_total += out_count
    density_mean = density_sum / loops
    return density_mean, out_total


def smooth2d(data: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float
    )
    kernel /= kernel.sum()
    pad = np.pad(data, 1, mode="edge")
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = np.sum(pad[i : i + 3, j : j + 3] * kernel)
    return out


def main():
    nb.set_num_threads(os.cpu_count() or 28)
    lowx = -3.0
    lowy = -3.0
    span = 6.0
    n_fine = 600
    out_n = 100
    eps = 0.4
    t_stop = 2E5
    burn_time = 2E3
    loops = nb.get_num_threads()
    do_smooth = False

    start_time = time.perf_counter()
    data, out_of_bounds_counts = run_ensemble(
        lowx, lowy, span, n_fine, out_n, eps, t_stop, burn_time, loops
    )
    end_time = time.perf_counter()
    print(f"SSA Canard Simulation Time: {end_time - start_time:.4f} seconds")
    print(f"Out of bounds counts: {out_of_bounds_counts}")

    h_eff = span / out_n
    data = data.reshape((out_n, out_n), order="F")
    if do_smooth:
        data = smooth2d(data)
    total = data.sum()
    if total > 0:
        data /= (h_eff**2 * total)
    else:
        raise ValueError("No samples landed in the grid. Check domain or burn_time.")

    x = np.linspace(lowx + h_eff / 2.0, lowx + span - h_eff / 2.0, out_n)
    y = np.linspace(lowy + h_eff / 2.0, lowy + span - h_eff / 2.0, out_n)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.plot_surface(X, Y, data, cmap="viridis")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Density")
    ax1.set_title("SSA Canard Density Estimation (Coarse Stat)")
    plt.show()


if __name__ == "__main__":
    main()
