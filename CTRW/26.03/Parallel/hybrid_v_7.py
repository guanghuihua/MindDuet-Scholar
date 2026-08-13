from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import numba as nb
from pathlib import Path
import os

# ── 设置多线程，充分利用 ───────────────────────────────────────────────
N_THREADS = os.cpu_count() 
os.environ.setdefault("OMP_NUM_THREADS",      str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",      str(N_THREADS))
os.environ.setdefault("NUMBA_NUM_THREADS",    str(N_THREADS))


# ─────────────────────────────────────────────────────────────────────────────
#  用 Numba 并行构建转移矩阵的 COO 三元组（只做一次）
# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(parallel=True, cache=True, fastmath=True)
def _build_triplets(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_x  = len(x_grid)
    n_v  = len(v_grid)
    h_x  = x_grid[1] - x_grid[0]
    h_v  = v_grid[1] - v_grid[0]
    sig2 = sigma * sigma

    # --- pass 1: 计算每个格点的总流出速率，用于求 lam 和对角元 ---
    rate_out = np.zeros(n_x * n_v, dtype=np.float64)
    for ix in nb.prange(n_x):
        x = x_grid[ix]
        for iv in range(n_v):
            v    = v_grid[iv]
            mu_x = v
            mu_v = -x - gamma * v
            m_v  = 0.5 * max(sig2 - abs(mu_v) * h_v, 0.0)

            r_xp = max(mu_x,  0.0) / h_x if ix < n_x - 1 else 0.0
            r_xm = -min(mu_x, 0.0) / h_x if ix > 0       else 0.0
            r_vp = (max(mu_v,  0.0) / h_v + m_v / (h_v * h_v)) if iv < n_v - 1 else 0.0
            r_vm = (-min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)) if iv > 0       else 0.0

            rate_out[ix * n_v + iv] = r_xp + r_xm + r_vp + r_vm

    lam = 1.05 * np.max(rate_out) + 1e-14

    # --- pass 2: 填充三元组，每个格点最多 5 个槽位（4 邻居 + 自身）---
    N       = n_x * n_v
    rows    = np.zeros(5 * N, dtype=np.int64)
    cols    = np.zeros(5 * N, dtype=np.int64)
    vals    = np.zeros(5 * N, dtype=np.float64)

    for ix in nb.prange(n_x):
        x = x_grid[ix]
        for iv in range(n_v):
            j    = ix * n_v + iv
            base = 5 * j
            cnt  = 0

            mu_x = v_grid[iv]
            mu_v = -x - gamma * v_grid[iv]
            m_v  = 0.5 * max(sig2 - abs(mu_v) * h_v, 0.0)

            # x+ 方向
            if ix < n_x - 1:
                r = max(mu_x, 0.0) / h_x
                if r > 0.0:
                    rows[base + cnt] = (ix + 1) * n_v + iv
                    cols[base + cnt] = j
                    vals[base + cnt] = r / lam
                    cnt += 1

            # x- 方向
            if ix > 0:
                r = -min(mu_x, 0.0) / h_x
                if r > 0.0:
                    rows[base + cnt] = (ix - 1) * n_v + iv
                    cols[base + cnt] = j
                    vals[base + cnt] = r / lam
                    cnt += 1

            # v+ 方向
            if iv < n_v - 1:
                r = max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                if r > 0.0:
                    rows[base + cnt] = ix * n_v + (iv + 1)
                    cols[base + cnt] = j
                    vals[base + cnt] = r / lam
                    cnt += 1

            # v- 方向
            if iv > 0:
                r = -min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                if r > 0.0:
                    rows[base + cnt] = ix * n_v + (iv - 1)
                    cols[base + cnt] = j
                    vals[base + cnt] = r / lam
                    cnt += 1

            # 对角元（留在原位的概率）
            rows[base + cnt] = j
            cols[base + cnt] = j
            vals[base + cnt] = 1.0 - rate_out[j] / lam

    return rows, cols, vals


def build_transition_matrix(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
) -> sp.csr_matrix:
    N    = len(x_grid) * len(v_grid)
    rows, cols, vals = _build_triplets(x_grid, v_grid, gamma, sigma)
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


# ─────────────────────────────────────────────────────────────────────────────
#  稀疏幂迭代：p_{k+1} = P @ p_k
#  scipy sparse matvec 底层调用 MKL/OpenBLAS，20 核真正并行
# ─────────────────────────────────────────────────────────────────────────────
def power_iteration(
    P: sp.csr_matrix,
    # max_iter: int = 200000,
    max_iter: int = 50000,
    tol: float = 1e-12,
    check_every: int = 10,
) -> np.ndarray:
    """单线程 SpMV 幂迭代，作为兜底备用。"""
    N = P.shape[0]
    p = np.full(N, 1.0 / N, dtype=np.float64)

    for it in range(max_iter):
        p_next = P @ p
        p_next /= p_next.sum()

        if it % check_every == 0:
            diff = np.max(np.abs(p_next - p))
            if diff < tol:
                print(f"  幂迭代收敛于第 {it} 轮，max_diff={diff:.3e}")
                return p_next

        p = p_next

    print(f"  警告：幂迭代未收敛（达到 max_iter={max_iter}），返回当前结果")
    return p


def stationary_via_eigs(P: sp.csr_matrix) -> np.ndarray:
    """
    ARPACK 求平稳分布，能充分利用多核。
    动态调整 ncv 和容差，失败时回退到幂迭代。
    """
    import scipy.sparse.linalg as spla
    N   = P.shape[0]
    # ncv = min(max(50, N // 20), N - 1)
    ncv = min(50, N - 1)

    try:
        _, vecs = spla.eigs(P, k=1, which='LM', tol=1e-10, maxiter=N * 5, ncv=ncv)
        pi = np.abs(np.real(vecs[:, 0]))
        pi /= pi.sum()
        print(f"  ARPACK 收敛（N={N}）")
        return pi
    except Exception as e:
        if hasattr(e, 'eigenvectors') and len(e.eigenvectors) > 0:
            print(f"  ARPACK 部分收敛，使用已有结果（N={N}）")
            pi = np.abs(np.real(e.eigenvectors[:, 0]))
            pi /= pi.sum()
            return pi
        print(f"  ARPACK 失败（N={N}），回退到幂迭代...")
        return power_iteration(P)


# ─────────────────────────────────────────────────────────────────────────────
#  主接口
# ─────────────────────────────────────────────────────────────────────────────
def discrete_density(
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    n_x, n_v = len(x_grid), len(v_grid)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    P      = build_transition_matrix(x_grid, v_grid, gamma, sigma)
    p_flat = stationary_via_eigs(P)

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

    rho_true = normalize_density(ground_true_density(x_grid, v_grid, gamma, sigma), h_x, h_v)

    t0 = time.perf_counter()
    rho_disc = normalize_density(discrete_density(x_grid, v_grid, gamma, sigma), h_x, h_v)  
    t_disc   = time.perf_counter() - t0

    err     = l1_error(rho_disc, rho_true, h_x, h_v)
    t_total = time.perf_counter() - t_total_start

    print(f"n_v={n_v:4d} | h_v={h_v:.4f} | error={err:.2e} | disc_time={t_disc:.2f}s | total={t_total:.2f}s")
    return h_v, err


# ─────────────────────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir  = Path(__file__).resolve().parent
    n_v_list = [20, 40, 60, 80, 99]
    # n_v_list = [20, 40, 60, 80]
    # n_v_list = [10, 20, 30, 40, 50 ]

    results = [l1_error_result(n_v) for n_v in n_v_list]

    h_v_vals = np.array([r[0] for r in results])
    err_vals  = np.array([r[1] for r in results])

    sort_idx = np.argsort(h_v_vals)
    h_v_vals = h_v_vals[sort_idx]
    err_vals  = err_vals[sort_idx]

    C_o1 = 0.5
    h_anchor, err_anchor = h_v_vals[0], err_vals[0]
    C_o2    = (err_anchor / h_anchor**2) * 1.0
    ref_o1  = C_o1 * h_v_vals
    ref_o2  = 0.6 * C_o2 * h_v_vals**2

    plt.rcParams.update({'text.usetex': False, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_v_vals, err_vals, 'o-',  color='#1f77b4', label='$hybrid$', lw=1.5, ms=7, mfc='none')
    ax.loglog(h_v_vals, ref_o1,   '--',  color='#ff7f0e', label='$O(h)$',   lw=1.2)
    ax.loglog(h_v_vals, ref_o2,   '--',  color='#9467bd', label='$O(h^2)$', lw=1.2)
    ax.set_xlabel('spatial stepsize h', fontsize=12)
    ax.set_ylabel('$l^1$-error',        fontsize=12)
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