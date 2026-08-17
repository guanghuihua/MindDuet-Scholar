from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
from pathlib import Path

# --- 核心计算引擎 (针对 28 核极致并行) ---
@nb.njit(parallel=True, cache=True, fastmath=True)
def discrete_density(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
    max_iter: int = 50000, # 增加上限以应对细网格
    tol: float = 1e-11,
) -> np.ndarray:
    n_x, n_v = len(x_grid), len(v_grid)
    h_x, h_v = x_grid[1] - x_grid[0], v_grid[1] - v_grid[0]
    sig2 = sigma * sigma

    # 1. 预计算速率 (充分利用并行)
    rate_out = np.zeros((n_x, n_v), dtype=np.float64)
    for ix in nb.prange(n_x):
        x = x_grid[ix]
        for iv in range(n_v):
            v = v_grid[iv]
            mu_v = -x - gamma * v
            m_v = 0.5 * max(sig2 - abs(mu_v) * h_v, 0.0)
            
            r_xp = max(v, 0.0) / h_x if ix < n_x - 1 else 0.0
            r_xm = -min(v, 0.0) / h_x if ix > 0 else 0.0
            r_vp = (max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)) if iv < n_v - 1 else 0.0
            r_vm = (-min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)) if iv > 0 else 0.0
            rate_out[ix, iv] = r_xp + r_xm + r_vp + r_vm
    
    lam = 1.05 * np.max(rate_out) + 1e-14

    # 2. 迭代演化
    p = np.full((n_x, n_v), 1.0 / (n_x * n_v), dtype=np.float64)
    p_next = np.zeros_like(p)

    for it in range(max_iter):
        for ix in nb.prange(n_x):
            x = x_grid[ix]
            for iv in range(n_v):
                # 自身贡献
                val = p[ix, iv] * (1.0 - rate_out[ix, iv] / lam)
                v_curr = v_grid[iv]
                
                # 邻居流入 (Pull 模式)
                if ix > 0: val += p[ix-1, iv] * (max(v_curr, 0.0) / h_x / lam)
                if ix < n_x - 1: val += p[ix+1, iv] * (-min(v_curr, 0.0) / h_x / lam)
                if iv > 0:
                    v_m = v_grid[iv-1]
                    mu_v_m = -x - gamma * v_m
                    m_v_m = 0.5 * max(sig2 - abs(mu_v_m) * h_v, 0.0)
                    val += p[ix, iv-1] * ((max(mu_v_m, 0.0) / h_v + m_v_m / (h_v * h_v)) / lam)
                if iv < n_v - 1:
                    v_p = v_grid[iv+1]
                    mu_v_p = -x - gamma * v_p
                    m_v_p = 0.5 * max(sig2 - abs(mu_v_p) * h_v, 0.0)
                    val += p[ix, iv+1] * ((-min(mu_v_p, 0.0) / h_v + m_v_p / (h_v * h_v)) / lam)
                p_next[ix, iv] = val

        # 归一化 (向量化操作)
        p_next /= np.sum(p_next)
        
        # 每 50 次迭代检查一次收敛，减少 CPU 同步开销
        if it % 50 == 0:
            diff = np.max(np.abs(p_next - p))
            if diff < tol:
                p[:] = p_next[:]
                break
        p[:] = p_next[:]

    return p / (h_x * h_v)

# --- 辅助函数 ---
def ground_true_density(x_grid, v_grid, gamma, sigma):
    beta = 2.0 * gamma / (sigma**2)
    x_mesh, v_mesh = np.meshgrid(x_grid, v_grid, indexing="ij")
    rho = np.exp(-beta * (0.5 * x_mesh**2 + 0.5 * v_mesh**2))
    h_x, h_v = x_grid[1]-x_grid[0], v_grid[1]-v_grid[0]
    return rho / (np.sum(rho) * h_x * h_v)

def main():
    # n_v_list = [32, 64, 128, 256, 400] 
    n_v_list = [200, 400, 600, 800]
    h_v_vals, err_vals = [], []
    gamma, sigma = 1.0, 1.0

    print("Starting computation on 28 cores...")
    for n_v in n_v_list:
        n_x =  n_v**2 # 建议设为 2*n_v 保证稳定
        x_grid = np.linspace(-4.5, 4.5, n_x) # 稍微扩大边界减少截断误差
        v_grid = np.linspace(-4.5, 4.5, n_v)
        
        t0 = time.perf_counter()
        rho_true = ground_true_density(x_grid, v_grid, gamma, sigma)
        rho_disc = discrete_density(x_grid, v_grid, gamma, sigma)
        
        h_x, h_v = x_grid[1]-x_grid[0], v_grid[1]-v_grid[0]
        err = np.sum(np.abs(rho_disc - rho_true)) * h_x * h_v
        
        print(f"n_v={n_v:4d} | h_v={h_v:.4f} | error={err:.2e} | time={time.perf_counter()-t0:.2f}s")
        h_v_vals.append(h_v); err_vals.append(err)

    # --- 绘图逻辑 (解决相交问题) ---
    h_v_vals, err_vals = np.array(h_v_vals), np.array(err_vals)
    sort_idx = np.argsort(h_v_vals)
    h_v_vals, err_vals = h_v_vals[sort_idx], err_vals[sort_idx]

    plt.figure(figsize=(8, 6))
    plt.loglog(h_v_vals, err_vals, 'o-', label='$Q_u$ (Numeric)', ms=8, mfc='none', lw=2)

    # 1. 设置 O(h) 参考线 (顶到上方)
    c_o1 = 2.0 
    plt.loglog(h_v_vals, c_o1 * h_v_vals, '--', label='$O(h)$', color='tab:orange')

    # 2. 设置 O(h^2) 参考线 (贴近数据线下缘)
    # 使用锚点法：让线经过数据中误差最小的点，并向下偏移一点
    c_o2 = (err_vals[0] / (h_v_vals[0]**2)) * 0.5 
    plt.loglog(h_v_vals, c_o2 * (h_v_vals**2), '--', label='$O(h^2)$', color='tab:purple')

    plt.ylim([1e-5, 10]) # 关键：给上方线条留出呼吸空间
    plt.xlabel('spatial stepsize h', fontsize=12)
    plt.ylabel('$l^1$-error', fontsize=12)
    plt.title('Stationary Density Accuracy (Parallel Boosted)', fontsize=14)
    plt.grid(True, which="both", alpha=0.2)
    plt.legend(loc='lower right')
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "accuracy_plot_styled.png"
    fig.savefig(out_png, dpi=300)
    print(f"Saved figure: {out_png}")
    plt.show()
if __name__ == "__main__":
    main()