import os
os.chdir(os.path.dirname(__file__))
import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

nb.config.NUMBA_DEFAULT_NUM_THREADS = 28


@nb.njit(fastmath=True)
def tau_leap_canard_timeweighted(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
    tau: float,
    max_events: int,
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
    n_events = 0

    while t < t_stop:
        mu_1 = (trajectory_point_y - trajectory_point_x**3 / 3.0 + trajectory_point_x) / delta
        mu_2 = a - trajectory_point_x
        m1 = (eps**2) * max(2.0 - abs(mu_1) * h, 0.0) / 2.0
        m2 = (eps**2) * max(2.0 - abs(mu_2) * h, 0.0) / 2.0

        ix = int(round((trajectory_point_x - lowx_center) * inv_h))
        iy = int(round((trajectory_point_y - lowy_center) * inv_h))
        trajectory_point_x = lowx_center + ix * h
        trajectory_point_y = lowy_center + iy * h

        q0 = max(mu_1, 0.0) / h + m1 / (h**2)
        q1 = -min(mu_1, 0.0) / h + m1 / (h**2)
        q2 = max(mu_2, 0.0) / h + m2 / (h**2)
        q3 = -min(mu_2, 0.0) / h + m2 / (h**2)
        lam = q0 + q1 + q2 + q3
        if lam <= 0.0:
            break

        tau_eff = tau
        cap = 0.2 / lam
        if tau_eff > cap:
            tau_eff = cap

        if t >= burn_time:
            if 0 <= ix < n and 0 <= iy < n:
                counts[ix, iy] += tau_eff
            else:
                out_of_bounds_counts += 1

        k0 = np.random.poisson(q0 * tau_eff)
        k1 = np.random.poisson(q1 * tau_eff)
        k2 = np.random.poisson(q2 * tau_eff)
        k3 = np.random.poisson(q3 * tau_eff)
        trajectory_point_x += h * (k0 - k1)
        trajectory_point_y += h * (k2 - k3)

        ix = int(round((trajectory_point_x - lowx_center) * inv_h))
        iy = int(round((trajectory_point_y - lowy_center) * inv_h))
        if ix < 0:
            trajectory_point_x = lowx_center
        elif ix >= n:
            trajectory_point_x = lowx + span - h / 2.0
        if iy < 0:
            trajectory_point_y = lowy_center
        elif iy >= n:
            trajectory_point_y = lowy + span - h / 2.0

        t += tau_eff
        n_events += 1
        if n_events >= max_events:
            break

    return counts.ravel(), out_of_bounds_counts, n_events


@nb.njit(fastmath=True, parallel=True)
def run_ensemble(
    lowx: float,
    lowy: float,
    span: float,
    n: int,
    eps: float,
    t_stop: float,
    burn_time: float,
    tau: float,
    max_events: int,
    loops: int,
) -> tuple:
    density_sum = np.zeros(n * n, dtype=np.float64)
    out_total = 0
    events_total = 0
    for i in nb.prange(loops):
        data, out_count, n_events = tau_leap_canard_timeweighted(
            lowx, lowy, span, n, eps, t_stop, burn_time, tau, max_events
        )
        density_sum += data
        out_total += out_count
        events_total += n_events
    density_mean = density_sum / loops
    return density_mean, out_total, events_total


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
    nb.set_num_threads(os.cpu_count() or 24)
    lowx = -3.0
    lowy = -3.0
    span = 6.0
    n = 1800
    eps = 0.3
    t_stop = 5.0e4
    burn_time = 5.0e3
    tau = 1.0e-2
    samples = 1000_000_000
    loops = nb.get_num_threads()
    out_n = 300
    if n % out_n != 0:
        raise ValueError("n must be divisible by out_n")
    bin_factor = n // out_n

    start_time = time.perf_counter()
    data, out_of_bounds_counts, events_total = run_ensemble(
        lowx, lowy, span, n, eps, t_stop, burn_time, tau, samples, loops
    )
    end_time = time.perf_counter()
    print(f"Tau-leaping Canard Simulation Time: {end_time - start_time:.4f} seconds")
    print(f"Out of bounds counts: {out_of_bounds_counts}")
    print(f"Total events: {events_total}")

    fine = data.reshape((n, n))
    data = bin_coarse(fine, n, bin_factor)
    h_eff = span / out_n
    total = data.sum()
    if total > 0:
        data /= (h_eff**2 * total)
    else:
        raise ValueError("No samples landed in the grid. Check domain or burn_time.")

    x = np.linspace(lowx + h_eff / 2.0, lowx + span - h_eff / 2.0, out_n)
    y = np.linspace(lowy + h_eff / 2.0, lowy + span - h_eff / 2.0, out_n)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot_surface(X, Y, data, cmap="viridis")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Density")
    ax1.set_title("Tau-leaping Canard Time-Weighted Density")
    fig1.tight_layout()
    fig1.savefig("tau_leaping_density_3d.png", dpi=300)
    fig1.show()

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    extent = (x[0], x[-1], y[0], y[-1])
    im = ax2.imshow(data.T, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_title("Tau-leaping Canard Density Heatmap")
    fig2.colorbar(im, ax=ax2, label="Density")
    fig2.tight_layout()
    fig2.savefig("tau_leaping_density_heatmap.png", dpi=300)
    fig2.show()


if __name__ == "__main__":
    main()
