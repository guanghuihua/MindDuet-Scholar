from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


nb.set_num_threads(20)


EPS = 0.1
A_PARAM = 1.0 - EPS / 8.0 - 3.0 * EPS**2 / 32.0 - 173.0 * EPS**3 / 1024.0 - 0.01
SIGMA_C = EPS ** (1.0 / 3.0)

X0 = 1.5
Y0 = X0**3 / 3.0 - X0
X_EXIT = 1.0
SPAN = 6.0
LOWX = -3.0
LOWY = -3.0
T_MAX = 4.0

# Event window near the stable slow manifold, just before the saddle node.
X_TARGET = 1.15
Y_TARGET = X_TARGET**3 / 3.0 - X_TARGET
X_TOL = 0.06
Y_TOL = 0.06


@nb.njit(fastmath=True)
def drift_x(x: float, y: float) -> float:
    return (y - x * x * x / 3.0 + x) / EPS


@nb.njit(fastmath=True)
def drift_y(x: float) -> float:
    return A_PARAM - x


@nb.njit(fastmath=True)
def classify_state(x: float, y: float) -> int:
    if abs(x - X_TARGET) <= X_TOL and abs(y - Y_TARGET) <= Y_TOL:
        return 1  # hit target window
    if x <= X_EXIT:
        return 2  # escaped before target
    return 0


@nb.njit(fastmath=True)
def em_single_path(sigma: float, dt: float) -> int:
    x = X0
    y = Y0
    t = 0.0

    while t < T_MAX:
        state = classify_state(x, y)
        if state != 0:
            return state

        x = x + drift_x(x, y) * dt
        y = y + drift_y(x) * dt + sigma * np.sqrt(dt) * np.random.randn()
        t += dt

    return 0


@nb.njit(fastmath=True)
def snap_to_grid(value: float, low: float, h: float, n: int) -> float:
    idx = int(np.round((value - low - 0.5 * h) / h))
    if idx < 0:
        idx = 0
    elif idx >= n:
        idx = n - 1
    return low + 0.5 * h + idx * h


@nb.njit(fastmath=True)
def ssa_single_path(sigma: float, n_y: int) -> int:
    n_x = n_y * n_y
    h_x = SPAN / n_x
    h_y = SPAN / n_y

    x = snap_to_grid(X0, LOWX, h_x, n_x)
    y = snap_to_grid(Y0, LOWY, h_y, n_y)
    t = 0.0

    while t < T_MAX:
        state = classify_state(x, y)
        if state != 0:
            return state

        mu_x = drift_x(x, y)
        mu_y = drift_y(x)
        m_y = 0.5 * max(sigma * sigma - abs(mu_y) * h_y, 0.0)

        q_xp = max(mu_x, 0.0) / h_x
        q_xm = max(-mu_x, 0.0) / h_x
        q_yp = max(mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        lam = q_xp + q_xm + q_yp + q_ym
        if lam <= 0.0:
            return 0

        tau = -np.log(max(1.0 - np.random.random(), 1e-15)) / lam
        t += tau

        r = np.random.random() * lam
        if r < q_xp:
            x += h_x
        elif r < q_xp + q_xm:
            x -= h_x
        elif r < q_xp + q_xm + q_yp:
            y += h_y
        else:
            y -= h_y

        if x < LOWX:
            x = LOWX + 0.5 * h_x
        elif x > LOWX + SPAN:
            x = LOWX + SPAN - 0.5 * h_x
        if y < LOWY:
            y = LOWY + 0.5 * h_y
        elif y > LOWY + SPAN:
            y = LOWY + SPAN - 0.5 * h_y

    return 0


@nb.njit(parallel=True, fastmath=True)
def em_event_counts(sigma: float, dt: float, n_paths: int) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int64)
    out = np.zeros(n_paths, dtype=np.int64)
    for i in nb.prange(n_paths):
        out[i] = em_single_path(sigma, dt)
    for i in range(n_paths):
        counts[out[i]] += 1
    return counts


@nb.njit(parallel=True, fastmath=True)
def ssa_event_counts(sigma: float, n_y: int, n_paths: int) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int64)
    out = np.zeros(n_paths, dtype=np.int64)
    for i in nb.prange(n_paths):
        out[i] = ssa_single_path(sigma, n_y)
    for i in range(n_paths):
        counts[out[i]] += 1
    return counts


def summarize_counts(counts: np.ndarray) -> tuple[float, float, float]:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    return counts[1] / total, counts[2] / total, counts[0] / total


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    sigma = 0.8 * SIGMA_C
    n_paths = 2000
    dt_list = np.array([5e-4, 1e-3, 2e-3, 5e-3], dtype=np.float64)
    n_y_list = np.array([30, 40, 60, 80, 120], dtype=np.int64)

    print(f"eps={EPS:.3f}, sigma_c={SIGMA_C:.6f}, sigma={sigma:.6f}")
    print(
        f"target window: x in [{X_TARGET - X_TOL:.3f}, {X_TARGET + X_TOL:.3f}], "
        f"y in [{Y_TARGET - Y_TOL:.3f}, {Y_TARGET + Y_TOL:.3f}]"
    )
    print(f"n_paths={n_paths}, numba threads={nb.get_num_threads()}")

    # JIT warm-up
    _ = em_event_counts(sigma, dt_list[0], 4)
    _ = ssa_event_counts(sigma, int(n_y_list[0]), 4)

    em_hit = np.zeros(len(dt_list), dtype=np.float64)
    em_escape = np.zeros(len(dt_list), dtype=np.float64)
    em_noevent = np.zeros(len(dt_list), dtype=np.float64)

    ssa_hit = np.zeros(len(n_y_list), dtype=np.float64)
    ssa_escape = np.zeros(len(n_y_list), dtype=np.float64)
    ssa_noevent = np.zeros(len(n_y_list), dtype=np.float64)
    h_y_vals = SPAN / n_y_list.astype(np.float64)

    print("\nEM scan")
    for i, dt in enumerate(dt_list):
        t0 = time.perf_counter()
        counts = em_event_counts(sigma, float(dt), n_paths)
        elapsed = time.perf_counter() - t0
        em_hit[i], em_escape[i], em_noevent[i] = summarize_counts(counts)
        print(
            f"dt={dt:.5e}, hit={em_hit[i]:.4f}, "
            f"early_escape={em_escape[i]:.4f}, no_event={em_noevent[i]:.4f}, "
            f"time={elapsed:.2f}s"
        )

    print("\nHybrid SSA scan")
    for i, n_y in enumerate(n_y_list):
        t0 = time.perf_counter()
        counts = ssa_event_counts(sigma, int(n_y), n_paths)
        elapsed = time.perf_counter() - t0
        ssa_hit[i], ssa_escape[i], ssa_noevent[i] = summarize_counts(counts)
        print(
            f"n_y={int(n_y):4d}, h_y={h_y_vals[i]:.5f}, hit={ssa_hit[i]:.4f}, "
            f"early_escape={ssa_escape[i]:.4f}, no_event={ssa_noevent[i]:.4f}, "
            f"time={elapsed:.2f}s"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(dt_list, em_hit, "o-", label="hit target window")
    axes[0].plot(dt_list, em_escape, "s--", label="early escape")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("EM step size dt")
    axes[0].set_ylabel("probability")
    axes[0].set_title("Euler-Maruyama")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(h_y_vals, ssa_hit, "o-", label="hit target window")
    axes[1].plot(h_y_vals, ssa_escape, "s--", label="early escape")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("SSA grid size h_y")
    axes[1].set_ylabel("probability")
    axes[1].set_title("Hybrid SSA")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        f"Canard Event Comparison, sigma/sigma_c={sigma / SIGMA_C:.2f}",
        fontsize=12,
    )
    fig.tight_layout()

    out_png = out_dir / "canard_event_compare.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
