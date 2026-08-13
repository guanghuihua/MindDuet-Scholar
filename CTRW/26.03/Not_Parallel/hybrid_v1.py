from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time
nb.config.NUMBA_DEFAULT_NUM_THREADS = 28

"""
Hybrid SSA for underdamped Langevin dynamics

$(x,v)$ 空间中的欠阻尼朗之万（Underdamped Langevin / Hypoelliptic）动力学：

$$
\begin{cases}
dX_t = V_t  dt \\
dV_t = -U'(X_t)  dt - \gamma V_t  dt + \sigma  dW_t
\end{cases}
\qquad (\text{仅在 } v \text{ 分量有噪声})
$$

dX_t     V_t                               0
      =                            dt  +             dW_t
dV_t     -U'(X_t) - \gamma V_t             \sigma

由于在 $x$ 方向没有扩散项，这是典型的**退化噪声**
尽管如此，其不变密度具有闭合形式：

U(x) = 0.5 * x*x
U'(x) = -x
$$
\rho_\infty(x,v) = Z^{-1} \exp \left( -\beta \left( U(x) + \frac{1}{2} v^2 \right) \right), \qquad \beta = \frac{2\gamma}{\sigma^2}
$$

"""

def drift(x):
    y = np.zeros(2)
    y[0] = x[1]
    y[1] = -x[0] + gamma * x[1]
    return y

@nb.njit(fastmath=True)
def Hybrid_SSA(gamma, sigma, N, span, lowx, lowy, valid_number):
    h = span / N
    trajectory = np.zeros((N,N))


    sample1 = np.array([lowx + h / 2.0 + round((lowx + span * random.random() - lowx - h / 2.0) / h) * h,
                        lowy + h / 2.0 + round((lowy + span * random.random() - lowy - h / 2.0) / h) * h])
    sample1_x_n = round((sample1[0] - lowx - h / 2.0) / h)
    sample1_y_n = round((sample1[1] - lowy - h / 2.0) / h)

    # sample2 = np.array([lowx + h / 2.0 + round((lowx + span * random.random() - lowx - h / 2.0) / h) * h,
    #                     lowy + h / 2.0 + round((lowy + span * random.random() - lowy - h / 2.0) / h) * h])
    # sample2_x_n = round((sample2[0] - lowx - h / 2.0) / h)
    # sample2_y_n = round((sample2[1] - lowy - h / 2.0) / h)

    mu = drift(sample1)
    M =  0.5 * np.max(0,(sigma ** 2 - np.abs(mu[1] * h)))

    q = np.zeros(4)
    q[0] = np.max(0, mu[0]) / h  #?????有问题这个地方
    q[1] = -np.min(0, mu[0]) / h 
    q[2] = np.max(0, mu[1]) /h + M / h**2
    q[3] = -np.min(0, mu[1]) /h + M / h**2

    Lambda = q[0] + q[1] + q[2] + q[3]
    t_end = t_burn + t_sample

    while t < t_end:
        sample1_time = 0.0

        rnd1 = random.random()
        rnd2 = random.random()

        tau = -np.log(1 - rnd1) / Lambda
        t += tau

        if q[0] < Lambda * rnd2:
            sample[0] += h
        elif q[0] + q[1] < Lambda * rnd2:
            sample[0] -= h
        elif q[0] + q[1] + q[2] < Lambda * rnd2:
            sample[1] += h
        else:
            sample[1] -= h

        # check if out of bound
        x_n = round((Sample[0] - low - h / 2.0) / h)
        y_n = round((Sample[1] - low - h / 2.0) / h)
        if x_n >= 0 and x_n < N and y_n >= 0 and y_n < N:
            trajectory[x_n, y_n] += 1
            # n += 1
        else:
            raise ValueError("x_n out of scope")
        
        return trajectory

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

def main():
    out_png = Path(__file__).resolve().parent / "Hybrid_SSA.png"

    # Basic setup for timing
    gamma = 1.0
    sigma = 1.0
    n_x = 121
    n_v = 121
    l_x = 4.0
    l_v = 4.0
    x_grid = np.linspace(-l_x, l_x, n_x)
    v_grid = np.linspace(-l_v, l_v, n_v)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    rho_true = ground_true_density(x_grid, v_grid, gamma, sigma)
    t_ground = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho_true = normalize_density(rho_true, h_x, h_v)
    t_norm = time.perf_counter() - t0

    # Optional timing entry for SSA (uncomment after Hybrid_SSA path is fully runnable)
    # t0 = time.perf_counter()
    # rho_ssa = Hybrid_SSA(gamma, sigma, N=120, span=8.0, lowx=-4.0, lowy=-4.0, valid_number=100000)
    # t_ssa = time.perf_counter() - t0

    t_total = time.perf_counter() - total_t0

    print(f"ground_true_density time: {t_ground:.6f} s")
    print(f"normalize_density time:   {t_norm:.6f} s")
    # print(f"Hybrid_SSA time:          {t_ssa:.6f} s")
    print(f"total time:               {t_total:.6f} s")

    plt.figure(figsize=(5.6, 4.6))
    extent = (x_grid[0], x_grid[-1], v_grid[0], v_grid[-1])
    plt.imshow(rho_true.T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    plt.title("Ground True Density")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.colorbar(label="density")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
