"""
Python translation of the MATLAB script SSA_Canard.m.
Implements the Monte Carlo (MC) sampler, the stochastic simulation algorithm
(SSA) on a uniform grid, and the sparse linear system construction used to
combine SSA data with a PDE-based correction.

Dependencies: numpy, matplotlib, scipy (for sparse matrices).
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import lsqr


# ---------------------------- Vector field --------------------------------- #
def canard(x: np.ndarray) -> np.ndarray:
    """Vector field for the 2D ODE."""
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01
    y = np.zeros(2, dtype=float)
    y[0] = (x[0] + x[1] - x[0] ** 3 / 3.0) / delta
    y[1] = a - x[0]
    return y


# ---------------------------- Monte Carlo EM ------------------------------- #
def mc_2d(lowx: float, lowy: float, span: float, n: int, eps: float, sample: int, dt: float) -> np.ndarray:
    """Euler-Maruyama Monte Carlo approximation of the invariant density."""
    h = span / n
    data = np.zeros(n * n, dtype=float)
    counts = np.zeros((n, n), dtype=float)
    x_old = np.array([1.0, 1.0], dtype=float)
    sqrt_dt = math.sqrt(dt)

    for _ in range(sample):
        x_new = x_old + dt * canard(x_old) + eps * sqrt_dt * np.random.randn(2)
        xx, yy = x_new
        x_n = math.ceil((xx - lowx) / h)
        y_n = math.ceil((yy - lowy) / h)
        if 1 <= x_n <= n and 1 <= y_n <= n:
            counts[x_n - 1, y_n - 1] += 1
        x_old = x_new

    # flatten counts into data
    data[:] = counts.ravel(order="F")  # MATLAB is column-major; match it
    total = data.sum()
    if total > 0:
        data /= (h**2 * total)
    print(f"Sample Size = {sample}")
    return data


# ---------------------------- SSA on grid ---------------------------------- #
def ssa_2d(lowx: float, lowy: float, span: float, n: int, eps: float, sample: int) -> np.ndarray:
    """CTRW/SSA approximation of the invariant density on a uniform grid."""
    h = span / n
    data = np.zeros((n + 1) * (n + 1), dtype=float)
    counts = np.zeros((n + 1, n + 1), dtype=float)

    x = 1.0
    y = 1.0
    t = 0.0
    t_stop = 50000.0
    n_events = 0
    delta = 0.1
    a = 1 - delta / 8 - 3 * delta**2 / 32 - 173 * delta**3 / 1024 - 0.01
    sigma = eps
    m11 = sigma**2 / 2.0
    m22 = sigma**2 / 2.0
    c1 = m11 / h**2
    c2 = m22 / h**2

    while t < t_stop:
        mu_1 = (y - (x**3) / 3.0 + x) / delta
        mu_2 = a - x

        q1 = max(mu_1, 0.0) / h + c1
        q2 = -min(mu_1, 0.0) / h + c1
        q3 = max(mu_2, 0.0) / h + c2
        q4 = -min(mu_2, 0.0) / h + c2

        lam = q1 + q2 + q3 + q4
        r1, r2 = np.random.rand(2)
        tau = -math.log(r1) / lam

        # Mimic MATLAB's reversed cumulative selection
        cum = np.cumsum([q1, q2, q3, q4])
        mu_number = int(np.sum(r2 * lam <= cum))

        t += tau
        if mu_number == 4:
            x += h
        elif mu_number == 3:
            x -= h
        elif mu_number == 2:
            y += h
        elif mu_number == 1:
            y -= h

        x_n = round((x - lowx) / h)
        y_n = round((y - lowy) / h)
        if 1 <= x_n <= n + 1 and 1 <= y_n <= n + 1:
            counts[x_n - 1, y_n - 1] += 1

        n_events += 1
        if n_events == sample:
            break

    data[:] = counts.ravel(order="F")
    total = data.sum()
    if total > 0:
        data /= (h**2 * total)
    print(f"Sample Size = {n_events}")
    return data


# ---------------------------- Matrix assembly ------------------------------ #
def matrix_2d(lowx: float, lowy: float, span: float, n: int, eps0: float) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse matrix and RHS construction mirroring the MATLAB Matrix_2D."""
    eps = eps0**2 / 2.0
    h = span / n
    f1 = lambda x, y: canard(np.array([x, y]))[0]
    f2 = lambda x, y: canard(np.array([x, y]))[1]

    b = np.zeros((n - 1) * (n - 1) + 1, dtype=float)
    b[-1] = 1 / h**2
    num_entries = 5 * (n - 1) * (n - 1) + (n + 1) * (n + 1)
    rows = np.zeros(num_entries, dtype=int)
    cols = np.zeros(num_entries, dtype=int)
    vals = np.zeros(num_entries, dtype=float)
    count = 0

    # Interior points: i,j in 1..n-1 (0-based)
    for i0 in range(1, n):
        for j0 in range(1, n):
            xx = i0 * h + lowx
            yy = j0 * h + lowy
            row = (i0 - 1) * (n - 1) + (j0 - 1)

            # +x neighbor
            rows[count] = row
            cols[count] = (i0 + 1) * (n + 1) + j0
            vals[count] = -f1(xx + h, yy) / (2 * h) + eps / (h**2)
            count += 1

            # -x neighbor
            rows[count] = row
            cols[count] = (i0 - 1) * (n + 1) + j0
            vals[count] = f1(xx - h, yy) / (2 * h) + eps / (h**2)
            count += 1

            # +y neighbor
            rows[count] = row
            cols[count] = i0 * (n + 1) + (j0 + 1)
            vals[count] = -f2(xx, yy + h) / (2 * h) + eps / (h**2)
            count += 1

            # -y neighbor
            rows[count] = row
            cols[count] = i0 * (n + 1) + (j0 - 1)
            vals[count] = f2(xx, yy - h) / (2 * h) + eps / (h**2)
            count += 1

            # center
            rows[count] = row
            cols[count] = i0 * (n + 1) + j0
            vals[count] = -4 * eps / (h**2)
            count += 1

    # Normalization row (all ones)
    for i0 in range(n + 1):
        for j0 in range(n + 1):
            rows[count] = (n - 1) * (n - 1)
            cols[count] = i0 * (n + 1) + j0
            vals[count] = 1.0
            count += 1

    A = sparse.coo_matrix(
        (vals[:count], (rows[:count], cols[:count])),
        shape=((n - 1) * (n - 1) + 1, (n + 1) * (n + 1)),
    ).tocsr()
    return A, b


# ---------------------------- Main driver ---------------------------------- #
def main() -> None:
    # Parameters (matching the MATLAB script)
    n = 600  # grid resolution
    eps = 0.3  # noise strength
    lowx = -3.0
    lowy = -3.0
    span = 6.0
    sample = 10_000_000  # This is very large; reduce for quicker tests
    dt = 0.001

    # SSA sampler (Monte Carlo section is commented out in the MATLAB code)
    t1 = time.perf_counter()
    data2 = ssa_2d(lowx, lowy, span, n, eps, sample)
    t2 = time.perf_counter()
    print("SSA_2D data generated")
    print(f"Elapsed time: {t2 - t1:.2f} seconds")

    # Build matrix and solve the least-norm problem
    A, b = matrix_2d(lowx, lowy, span, n, eps)
    print("matrix built")
    b = b - A @ data2
    # least-norm solution for the sparse system
    x = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=10_000)[0]
    y = x + data2

    V = y.reshape((n + 1, n + 1), order="F")
    W = data2.reshape((n + 1, n + 1), order="F")

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    ax1.plot_surface(
        *np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij"),
        W,
        cmap="viridis",
    )
    ax1.set_title("Monte Carlo estimate W")

    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    ax2.plot_surface(
        *np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij"),
        V,
        cmap="viridis",
    )
    ax2.set_title("Corrected solution V")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
