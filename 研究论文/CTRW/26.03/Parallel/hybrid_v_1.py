from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
from pathlib import Path

# 使用 fastmath 提升向量化效率
@nb.njit(cache=True, fastmath=True)
def drift(x: float, v: float, gamma: float) -> tuple[float, float]:
    mu_x = v
    mu_v = -x - gamma * v
    return mu_x, mu_v


@nb.njit(parallel=True, cache=True, fastmath=True)
def discrete_density(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
    # max_iter: int = 8000,
    max_iter: int = 50000,
    tol: float = 1e-11,
) -> np.ndarray:
    n_x = len(x_grid)
    n_v = len(v_grid)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]
    sig2 = sigma * sigma

    # --- 1. 计算 lambda ---
    # 预计算所有格点的流出速率和，避免循环中重复计算
    rate_out = np.zeros((n_x, n_v), dtype=np.float64)
    for ix in nb.prange(n_x):
        x = x_grid[ix]
        for iv in range(n_v):
            v = v_grid[iv]
            # 手动内联 drift: mu_x = v, mu_v = -x - gamma*v
            mu_x = v
            mu_v = -x - gamma * v
            m_v = 0.5 * max(sig2 - abs(mu_v) * h_v, 0.0)
            
            r_xp = max(mu_x, 0.0) / h_x if ix < n_x - 1 else 0.0
            r_xm = -min(mu_x, 0.0) / h_x if ix > 0 else 0.0
            r_vp = (max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)) if iv < n_v - 1 else 0.0
            r_vm = (-min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)) if iv > 0 else 0.0
            
            rate_out[ix, iv] = r_xp + r_xm + r_vp + r_vm
    
    lam = 1.05 * np.max(rate_out) + 1e-14

    # --- 2. 迭代 ---
    p = np.full((n_x, n_v), 1.0 / (n_x * n_v), dtype=np.float64)
    p_next = np.zeros_like(p)

    for it in range(max_iter):
        # Pull 模式，避免竞争
        for ix in nb.prange(n_x):
            x = x_grid[ix]
            for iv in range(n_v):
                # 1. 自身的贡献 (留在原地的概率)
                val = p[ix, iv] * (1.0 - rate_out[ix, iv] / lam)

                # 2. 邻居的贡献 (流入本格点的概率)
                # 从左侧 (ix-1) 过来: 只有当 mu_x[ix-1] > 0
                if ix > 0:
                    v_here = v_grid[iv]
                    # mu_x_m 就是 v_here
                    r_xp_m = max(v_here, 0.0) / h_x
                    val += p[ix - 1, iv] * (r_xp_m / lam)
                
                # 从右侧 (ix+1) 过来: 只有当 mu_x[ix+1] < 0
                if ix < n_x - 1:
                    v_here = v_grid[iv]
                    r_xm_p = -min(v_here, 0.0) / h_x
                    val += p[ix + 1, iv] * (r_xm_p / lam)
                
                # 从下方 (iv-1) 过来
                if iv > 0:
                    x_here = x_grid[ix]
                    v_m = v_grid[iv - 1]
                    mu_v_m = -x_here - gamma * v_m
                    m_v_m = 0.5 * max(sig2 - abs(mu_v_m) * h_v, 0.0)
                    r_vp_m = (max(mu_v_m, 0.0) / h_v + m_v_m / (h_v * h_v))
                    val += p[ix, iv - 1] * (r_vp_m / lam)
                
                # 从上方 (iv+1) 过来
                if iv < n_v - 1:
                    x_here = x_grid[ix]
                    v_p = v_grid[iv + 1]
                    mu_v_p = -x_here - gamma * v_p
                    m_v_p = 0.5 * max(sig2 - abs(mu_v_p) * h_v, 0.0)
                    r_vm_p = (-min(mu_v_p, 0.0) / h_v + m_v_p / (h_v * h_v))
                    val += p[ix, iv + 1] * (r_vm_p / lam)
                
                p_next[ix, iv] = val

        # 收敛检查
        p_next /= np.sum(p_next)
        
        # 优化：每 10 次迭代检查一次 max_diff 以减少并行同步开销
        if it % 10 == 0:
            max_diff = 0.0
            # 注意：在 prange 中更新标量需要使用原子操作或特定的 reduction 语法
            # 这里简单起见，使用 np.max 替代手动循环
            max_diff = np.max(np.abs(p_next - p))
            if max_diff < tol:
                p[:] = p_next[:]
                break
        
        p[:] = p_next[:]

    return p / (h_x * h_v)

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

def l1_error_result(n_v: int) -> tuple[float, float]:
    t_total_start = time.perf_counter()

    gamma = 1.0
    sigma = 1.0

    # Grid and truncated domain
    # n_x = n_v**2
    # n_x = 10*(n_v-1)+1
    n_x = n_v
    l_x = 4.0
    l_v = 4.0

    x_grid = np.linspace(-l_x, l_x, n_x)
    v_grid = np.linspace(-l_v, l_v, n_v)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    t0 = time.perf_counter()
    rho_true_raw = ground_true_density(x_grid, v_grid, gamma, sigma)
    rho_true = normalize_density(rho_true_raw, h_x, h_v)
    t_true = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho_disc_raw = discrete_density(x_grid, v_grid, gamma, sigma)
    rho_disc = normalize_density(rho_disc_raw, h_x, h_v)
    t_disc = time.perf_counter() - t0

    err_disc_true = l1_error(rho_disc, rho_true, h_x, h_v)

    t_total = time.perf_counter() - t_total_start

    print(f"n_v={n_v:4d}, h_v={h_v:.6e}, L1={err_disc_true:.6e}")
    print(f"Time true density:      {t_true:.6f} s")
    print(f"Time discrete density:  {t_disc:.6f} s")
    print(f"Total time:             {t_total:.6f} s")

    return h_v, err_disc_true

def main():
    out_dir = Path(__file__).resolve().parent
    # n_v_list = [64, 128, 256, 512, 1024]
    # n_v_list = [50, 100, 150, 200]
    # n_v_list = [ 40, 50, 60, 70, 80]
    n_v_list = [32, 64, 128, 256, 400]

    h_v_vals = []
    err_vals = []
    for n_v in n_v_list:
        h_v, err_disc_true = l1_error_result(n_v)
        h_v_vals.append(h_v)
        err_vals.append(err_disc_true)

    h_v_vals = np.array(h_v_vals)
    err_vals = np.array(err_vals)

    # Sort by h_v for cleaner plotting and slope diagnostics.
    sort_idx = np.argsort(h_v_vals)
    h_v_vals = h_v_vals[sort_idx]
    err_vals = err_vals[sort_idx]

    # Build reference lines O(h_v) and O(h_v^2), anchored at one data point.
    i_anchor = max(len(h_v_vals) - 2, 0)
    h_anchor = h_v_vals[i_anchor]
    err_anchor = err_vals[i_anchor]
    ref_o1 = err_anchor * (h_v_vals / h_anchor)
    ref_o2 = err_anchor * (h_v_vals / h_anchor) ** 2

    # Empirical convergence orders.
    local_orders = np.log(err_vals[1:] / err_vals[:-1]) / np.log(h_v_vals[1:] / h_v_vals[:-1])
    global_order, _ = np.polyfit(np.log(h_v_vals), np.log(err_vals), 1)
    print("local orders:", local_orders)
    print(f"global fitted order: {global_order:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h_v_vals, err_vals, "o-", lw=1.2, ms=6, label="L1 error")
    ax.loglog(h_v_vals, ref_o1, "--", lw=1.0, label="O(h_v)")
    ax.loglog(h_v_vals, ref_o2, "--", lw=1.0, label="O(h_v^2)")
    ax.set_xlabel("h_v")
    ax.set_ylabel("L1 error")
    ax.set_title("L1 Error vs h_v (log-log)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = out_dir / "hybrid_v4_l1_error_vs_hv.png"
    fig.savefig(out_png, dpi=300)
    print(f"Saved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
