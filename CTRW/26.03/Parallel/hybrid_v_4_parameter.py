from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import numba as nb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

# ── 设置 BLAS/OpenMP 线程数，充分利用 20 核 ──────────────────────────────────
N_THREADS = os.cpu_count() or 20
os.environ.setdefault("OMP_NUM_THREADS",       str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS",  str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",       str(N_THREADS))
os.environ.setdefault("NUMBA_NUM_THREADS",     str(N_THREADS))


# ─────────────────────────────────────────────────────────────────────────────
#  构建转移矩阵（稀疏 CSR），只做一次
# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(parallel=True, cache=True, fastmath=True)
def _build_triplets(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    返回 (rows, cols, vals, lam) 用于构建转移矩阵 P，满足：
        p_{j->i} = rate_{j->i} / lam
        p_{j->j} = 1 - sum_{i≠j} rate_{j->i} / lam
    矩阵按"列随机"（Markov chain）存储，即 P[:,j] 对应从 j 出发的转移概率。
    但我们实际做行向量左乘：p_new = p_old @ P，所以需要行随机版本，
    因此这里构造的是转置后的列随机矩阵，对应 p_new = P^T p_old（列向量乘法）。
    
    实际：我们把所有 off-diagonal 和 diagonal 都存成 (i, j, val) 三元组，
    其中 i=目标格点，j=来源格点（即 rate j→i / lam），之后用 csr_matrix 构建。
    这样 p_new = P @ p_old 直接可用。
    """
    n_x = len(x_grid)
    n_v = len(v_grid)
    N   = n_x * n_v
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]
    sig2 = sigma * sigma

    # 每个格点最多 4 个邻居 + 自身，上界为 5*N 个三元组
    max_nnz = 5 * N
    rows = np.empty(max_nnz, dtype=np.int64)
    cols = np.empty(max_nnz, dtype=np.int64)
    vals = np.empty(max_nnz, dtype=np.float64)

    # --- pass 1: 计算所有流出速率，求 lam ---
    rate_out = np.zeros(N, dtype=np.float64)
    for ix in nb.prange(n_x):
        x = x_grid[ix]
        for iv in range(n_v):
            v   = v_grid[iv]
            mu_x = v
            mu_v = -x - gamma * v
            m_v  = 0.5 * max(sig2 - abs(mu_v) * h_v, 0.0)

            r_xp = max(mu_x,  0.0) / h_x if ix < n_x - 1 else 0.0
            r_xm = -min(mu_x, 0.0) / h_x if ix > 0       else 0.0
            r_vp = (max(mu_v,  0.0) / h_v + m_v / (h_v * h_v)) if iv < n_v - 1 else 0.0
            r_vm = (-min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)) if iv > 0       else 0.0

            rate_out[ix * n_v + iv] = r_xp + r_xm + r_vp + r_vm

    lam = 1.05 * np.max(rate_out) + 1e-14

    # --- pass 2: 填充三元组 ---
    # 注意 prange 会并行，但写入 rows/cols/vals 时下标计算需要确定性
    # 为避免竞争，对每个格点 j，写入位置从 5*j 开始（最多 5 项）
    for ix in nb.prange(n_x):
        x = x_grid[ix]
        for iv in range(n_v):
            j    = ix * n_v + iv
            base = 5 * j          # 每个 j 占 5 个槽位
            cnt  = 0

            v_   = v_grid[iv]
            mu_x = v_
            mu_v = -x - gamma * v_
            m_v  = 0.5 * max(sig2 - abs(mu_v) * h_v, 0.0)

            # x+ 邻居
            if ix < n_x - 1:
                r = max(mu_x, 0.0) / h_x
                if r > 0.0:
                    rows[base + cnt] = (ix + 1) * n_v + iv  # 目标
                    cols[base + cnt] = j                     # 来源
                    vals[base + cnt] = r / lam
                    cnt += 1

            # x- 邻居
            if ix > 0:
                r = -min(mu_x, 0.0) / h_x
                if r > 0.0:
                    rows[base + cnt] = (ix - 1) * n_v + iv
                    cols[base + cnt] = j
                    vals[base + cnt] = r / lam
                    cnt += 1

            # v+ 邻居
            if iv < n_v - 1:
                r = max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                if r > 0.0:
                    rows[base + cnt] = ix * n_v + (iv + 1)
                    cols[base + cnt] = j
                    vals[base + cnt] = r / lam
                    cnt += 1

            # v- 邻居
            if iv > 0:
                r = -min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                if r > 0.0:
                    rows[base + cnt] = ix * n_v + (iv - 1)
                    cols[base + cnt] = j
                    vals[base + cnt] = r / lam
                    cnt += 1

            # 对角（留在原位）
            diag_val = 1.0 - rate_out[j] / lam
            rows[base + cnt] = j
            cols[base + cnt] = j
            vals[base + cnt] = diag_val
            cnt += 1

            # 未使用的槽位清零（避免垃圾值）
            for k in range(cnt, 5):
                rows[base + k] = 0
                cols[base + k] = 0
                vals[base + k] = 0.0

    return rows, cols, vals, lam


def build_transition_matrix(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
) -> sp.csr_matrix:
    n_x, n_v = len(x_grid), len(v_grid)
    N = n_x * n_v
    rows, cols, vals, lam = _build_triplets(x_grid, v_grid, gamma, sigma)
    P = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return P


# ─────────────────────────────────────────────────────────────────────────────
#  稀疏幂迭代（主循环）
# ─────────────────────────────────────────────────────────────────────────────
def power_iteration_sparse(
    P: sp.csr_matrix,
    max_iter: int = 50000,
    tol: float = 1e-11,
    check_every: int = 10,
) -> np.ndarray:
    """
    p_{k+1} = P @ p_k，收敛到平稳分布。
    scipy sparse matvec 自动调用多线程 BLAS（MKL/OpenBLAS），20 核全跑满。
    """
    N = P.shape[0]
    p = np.full(N, 1.0 / N, dtype=np.float64)

    for it in range(max_iter):
        p_next = P @ p                       # ← 这里真正多核并行
        p_next /= p_next.sum()

        if it % check_every == 0:
            diff = np.max(np.abs(p_next - p))
            if diff < tol:
                print(f"  收敛于第 {it} 轮，max_diff={diff:.3e}")
                return p_next

        p = p_next

    print(f"  未收敛（达到 max_iter={max_iter}），返回当前结果")
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  也可以用 ARPACK 直接求最大特征向量（更快，适合大矩阵）
# ─────────────────────────────────────────────────────────────────────────────
def stationary_via_eigs(P: sp.csr_matrix) -> np.ndarray:
    """
    用 ARPACK 求转移矩阵的特征值为 1 的特征向量，即平稳分布。
    对大矩阵比幂迭代快得多，且利用多线程 BLAS。
    """
    vals, vecs = spla.eigs(P, k=1, which='LM', tol=1e-12, maxiter=10000)
    pi = np.real(vecs[:, 0])
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi


# ─────────────────────────────────────────────────────────────────────────────
#  接口函数（替换原来的 discrete_density）
# ──────────────────────────────────────────────────────────────────────────
def discrete_density(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
    method: str = "eigs",       # "eigs" 更快；"power" 更可控
    max_iter: int = 50000,
    tol: float = 1e-11,
) -> np.ndarray:
    n_x, n_v = len(x_grid), len(v_grid)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    P = build_transition_matrix(x_grid, v_grid, gamma, sigma)

    if method == "eigs":
        p_flat = stationary_via_eigs(P)
    else:
        p_flat = power_iteration_sparse(P, max_iter=max_iter, tol=tol)

    return p_flat.reshape(n_x, n_v) / (h_x * h_v)


# ─────────────────────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────────────────────
def ground_true_density(x_grid, v_grid, gamma, sigma):
    beta = 2.0 * gamma / (sigma * sigma)
    x_mesh, v_mesh = np.meshgrid(x_grid, v_grid, indexing="ij")
    energy = 0.5 * x_mesh ** 2 + 0.5 * v_mesh ** 2
    return np.exp(-beta * energy)


def normalize_density(rho, h_x, h_v):
    mass = np.sum(rho) * h_x * h_v
    return rho / mass if mass > 0 else np.zeros_like(rho)


def l1_error(rho_a, rho_b, h_x, h_v):
    return float(np.sum(np.abs(rho_a - rho_b)) * h_x * h_v)


def l1_error_result(n_v: int) -> tuple[float, float]:
    t_total_start = time.perf_counter()

    gamma = 1.0
    sigma = 1.0
    n_x   = n_v ** 2
    l_x, l_v = 4.0, 4.0

    x_grid = np.linspace(-l_x, l_x, n_x)
    v_grid = np.linspace(-l_v, l_v, n_v)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    t0 = time.perf_counter()
    rho_true = normalize_density(ground_true_density(x_grid, v_grid, gamma, sigma), h_x, h_v)
    t_true = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho_disc = normalize_density(discrete_density(x_grid, v_grid, gamma, sigma, method="eigs"), h_x, h_v)
    t_disc = time.perf_counter() - t0

    err = l1_error(rho_disc, rho_true, h_x, h_v)
    t_total = time.perf_counter() - t_total_start

    # print(f"n_v={n_v:4d}, h_v={h_v:.6e}, L1={err:.6e}")
    # print(f"  Time true density:      {t_true:.4f} s")
    # print(f"  Time discrete density:  {t_disc:.4f} s")
    # print(f"  Total time:             {t_total:.4f} s")
    print(f"n_v={n_v:4d} | h_v={h_v:.4f} | error={err:.2e} | time={time.perf_counter()-t0:.2f}s")
    return h_v, err


# ─────────────────────────────────────────────────────────────────────────────
#  多个 n_v 并发运行（用 ThreadPoolExecutor，因为主要是 BLAS 多线程）
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir   = Path(__file__).resolve().parent
    # n_v_list  = [10, 20, 30, 40, 50]
    # n_v_list  = [30, 40, 50, 60, 70]
    n_v_list  = [20, 40, 60, 80, 100]

    # 顺序跑（每个问题内部已经多核了）
    # 如果问题规模小，也可以外层并行；但因为 BLAS 已用全部核，外层并行反而会争抢
    results = [l1_error_result(n_v) for n_v in n_v_list]

    h_v_vals = np.array([r[0] for r in results])
    err_vals  = np.array([r[1] for r in results])

    sort_idx  = np.argsort(h_v_vals)
    h_v_vals  = h_v_vals[sort_idx]
    err_vals  = err_vals[sort_idx]

    C_o1 = 0.1
    h_anchor, err_anchor = h_v_vals[0], err_vals[0]
    C_o2    = (err_anchor / h_anchor**2) * 1.0
    ref_o1  = C_o1 * h_v_vals
    ref_o2  = 0.6 * C_o2 * h_v_vals**2

    plt.rcParams.update({'text.usetex': False, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_v_vals, err_vals,  'o-',  color='#1f77b4', label='$hybrid$',  lw=1.5, ms=7, mfc='none')
    ax.loglog(h_v_vals, ref_o1,    '--',  color='#ff7f0e', label='$O(h)$',    lw=1.2)
    ax.loglog(h_v_vals, ref_o2,    '--',  color='#9467bd', label='$O(h^2)$',  lw=1.2)
    ax.set_xlabel('spatial stepsize h', fontsize=12)
    ax.set_ylabel('$l^1$-error',         fontsize=12)
    ax.set_title('Stationary Density Accuracy', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    fig.tight_layout()

    out_png = out_dir / "accuracy_plot_styled.png"
    fig.savefig(out_png, dpi=300)
    print(f"Saved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()