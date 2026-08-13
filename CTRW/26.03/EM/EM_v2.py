from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import numba as nb
from pathlib import Path
import os

# ── 多线程设置 ────────────────────────────────────────────────────────────────
N_THREADS = os.cpu_count() or 20
os.environ.setdefault("OMP_NUM_THREADS",      str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",      str(N_THREADS))
os.environ.setdefault("NUMBA_NUM_THREADS",    str(N_THREADS))


# ─────────────────────────────────────────────────────────────────────────────
#  离散化方法（你的原始方法）
# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(parallel=True, cache=True, fastmath=True)
def _build_triplets(x_grid, v_grid, gamma, sigma):
    n_x  = len(x_grid)
    n_v  = len(v_grid)
    h_x  = x_grid[1] - x_grid[0]
    h_v  = v_grid[1] - v_grid[0]
    sig2 = sigma * sigma

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

    lam  = 1.05 * np.max(rate_out) + 1e-14
    N    = n_x * n_v
    rows = np.zeros(5 * N, dtype=np.int64)
    cols = np.zeros(5 * N, dtype=np.int64)
    vals = np.zeros(5 * N, dtype=np.float64)

    for ix in nb.prange(n_x):
        x = x_grid[ix]
        for iv in range(n_v):
            j    = ix * n_v + iv
            base = 5 * j
            cnt  = 0
            mu_x = v_grid[iv]
            mu_v = -x - gamma * v_grid[iv]
            m_v  = 0.5 * max(sig2 - abs(mu_v) * h_v, 0.0)

            if ix < n_x - 1:
                r = max(mu_x, 0.0) / h_x
                if r > 0.0:
                    rows[base+cnt] = (ix+1)*n_v+iv; cols[base+cnt] = j
                    vals[base+cnt] = r/lam; cnt += 1
            if ix > 0:
                r = -min(mu_x, 0.0) / h_x
                if r > 0.0:
                    rows[base+cnt] = (ix-1)*n_v+iv; cols[base+cnt] = j
                    vals[base+cnt] = r/lam; cnt += 1
            if iv < n_v - 1:
                r = max(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                if r > 0.0:
                    rows[base+cnt] = ix*n_v+(iv+1); cols[base+cnt] = j
                    vals[base+cnt] = r/lam; cnt += 1
            if iv > 0:
                r = -min(mu_v, 0.0) / h_v + m_v / (h_v * h_v)
                if r > 0.0:
                    rows[base+cnt] = ix*n_v+(iv-1); cols[base+cnt] = j
                    vals[base+cnt] = r/lam; cnt += 1

            rows[base+cnt] = j; cols[base+cnt] = j
            vals[base+cnt] = 1.0 - rate_out[j] / lam

    return rows, cols, vals


def build_transition_matrix(x_grid, v_grid, gamma, sigma):
    N = len(x_grid) * len(v_grid)
    rows, cols, vals = _build_triplets(x_grid, v_grid, gamma, sigma)
    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


def stationary_via_eigs(P):
    N   = P.shape[0]
    ncv = min(50, N - 1)
    try:
        _, vecs = spla.eigs(P, k=1, which='LM', tol=1e-10,
                            maxiter=N * 5, ncv=ncv)
        pi = np.abs(np.real(vecs[:, 0]))
        pi /= pi.sum()
        return pi
    except Exception as e:
        if hasattr(e, 'eigenvectors') and len(e.eigenvectors) > 0:
            pi = np.abs(np.real(e.eigenvectors[:, 0]))
            pi /= pi.sum()
            return pi
        return power_iteration(P)


def power_iteration(P, max_iter=200000, tol=1e-12, check_every=10):
    N = P.shape[0]
    p = np.full(N, 1.0 / N, dtype=np.float64)
    for it in range(max_iter):
        p_next = P @ p
        p_next /= p_next.sum()
        if it % check_every == 0:
            if np.max(np.abs(p_next - p)) < tol:
                return p_next
        p = p_next
    return p


def discrete_density(x_grid, v_grid, gamma, sigma):
    n_x, n_v = len(x_grid), len(v_grid)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]
    P      = build_transition_matrix(x_grid, v_grid, gamma, sigma)
    p_flat = stationary_via_eigs(P)
    return p_flat.reshape(n_x, n_v) / (h_x * h_v)


# ─────────────────────────────────────────────────────────────────────────────
#  EM 方法：给定时间预算 time_budget 秒，尽量多跑步数
# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=True)
def _em_simulate(gamma, sigma, dt, n_burnin, n_steps, seed):
    """纯 Numba 单线程 EM 积分，速度比 Python 循环快 ~100x。"""
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    x, v    = 0.0, 0.0

    # 预热
    for _ in range(n_burnin):
        x = x + v * dt
        v = v + (-x - gamma * v) * dt + sigma * sqrt_dt * np.random.randn()

    # 采样
    x_arr = np.empty(n_steps, dtype=np.float64)
    v_arr = np.empty(n_steps, dtype=np.float64)
    for i in range(n_steps):
        x = x + v * dt
        v = v + (-x - gamma * v) * dt + sigma * sqrt_dt * np.random.randn()
        x_arr[i] = x
        v_arr[i] = v

    return x_arr, v_arr


def em_density(
    x_grid, v_grid, gamma, sigma,
    dt=1e-3,
    time_budget=None,   # 秒；None 则用 n_steps
    n_steps=5_000_000,
    n_burnin=100_000,
    seed=42,
):
    """
    用 EM 方法估计不变密度。
    若指定 time_budget，则根据单步耗时动态估算最大步数。
    """
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    if time_budget is not None:
        # 用足够大的批量估算单步耗时，避免 CPU 缓存热效应导致低估
        # 100k 步太少，Numba 循环在热缓存下比真实大批量快很多
        probe = 5_000_000
        t0    = time.perf_counter()
        _em_simulate(gamma, sigma, dt, 0, probe, seed)
        t_per_step = (time.perf_counter() - t0) / probe

        # 预留 15% 给 burnin 与直方图后处理，避免实际超出预算
        budget_effective = time_budget * 0.85
        n_steps = max(int(budget_effective / t_per_step), 500_000)
        print(f"  EM 时间预算 {time_budget:.1f}s → 单步 {t_per_step*1e9:.1f}ns → n_steps={n_steps:,}")

    x_traj, v_traj = _em_simulate(gamma, sigma, dt, n_burnin, n_steps, seed)

    # 二维直方图投影到网格
    x_edges = np.concatenate([[x_grid[0]  - h_x/2],
                               (x_grid[:-1] + x_grid[1:]) / 2,
                               [x_grid[-1] + h_x/2]])
    v_edges = np.concatenate([[v_grid[0]  - h_v/2],
                               (v_grid[:-1] + v_grid[1:]) / 2,
                               [v_grid[-1] + h_v/2]])
    counts, _, _ = np.histogram2d(x_traj, v_traj, bins=[x_edges, v_edges])

    mass = counts.sum() * h_x * h_v
    return counts / mass if mass > 0 else np.zeros_like(counts)


# ─────────────────────────────────────────────────────────────────────────────
#  解析真值与误差
# ─────────────────────────────────────────────────────────────────────────────
def ground_true_density(x_grid, v_grid, gamma, sigma):
    beta   = 2.0 * gamma / sigma ** 2
    x_mesh, v_mesh = np.meshgrid(x_grid, v_grid, indexing="ij")
    return np.exp(-beta * 0.5 * (x_mesh**2 + v_mesh**2))


def normalize_density(rho, h_x, h_v):
    mass = rho.sum() * h_x * h_v
    return rho / mass if mass > 0 else np.zeros_like(rho)


def l1_error(rho_a, rho_b, h_x, h_v):
    return float(np.sum(np.abs(rho_a - rho_b)) * h_x * h_v)


# ─────────────────────────────────────────────────────────────────────────────
#  公平对比：对每个 n_v，两种方法使用相同的计算时间
# ─────────────────────────────────────────────────────────────────────────────
def compare_one(n_v, gamma=1.0, sigma=1.0, dt=1e-3):
    n_x      = n_v ** 2
    x_grid   = np.linspace(-4.0, 4.0, n_x)
    v_grid   = np.linspace(-4.0, 4.0, n_v)
    h_x      = x_grid[1] - x_grid[0]
    h_v      = v_grid[1] - v_grid[0]

    rho_true = normalize_density(
        ground_true_density(x_grid, v_grid, gamma, sigma), h_x, h_v
    )

    # ── 离散化方法，记录耗时 ──────────────────────────────────────────────────
    t0      = time.perf_counter()
    rho_disc = normalize_density(discrete_density(x_grid, v_grid, gamma, sigma), h_x, h_v)
    t_disc  = time.perf_counter() - t0
    err_disc = l1_error(rho_disc, rho_true, h_x, h_v)

    # ── EM 方法，使用相同时间预算 ─────────────────────────────────────────────
    t0     = time.perf_counter()
    rho_em = normalize_density(
        em_density(x_grid, v_grid, gamma, sigma, dt=dt, time_budget=t_disc), h_x, h_v
    )
    t_em   = time.perf_counter() - t0
    err_em = l1_error(rho_em, rho_true, h_x, h_v)

    print(f"n_v={n_v:4d} | h_v={h_v:.4f} | "
          f"disc: err={err_disc:.2e} t={t_disc:.2f}s | "
          f"EM:   err={err_em:.2e}   t={t_em:.2f}s")

    return h_v, err_disc, err_em


# ─────────────────────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir  = Path(__file__).resolve().parent
    n_v_list = [20, 40] #, 60, 80, 100]

    # Numba JIT 预热（避免第一个 n_v 的时间被编译污染）
    print("JIT 预热中...")
    _build_triplets(np.linspace(-4, 4, 25), np.linspace(-4, 4, 5), 1.0, 1.0)
    _em_simulate(1.0, 1.0, 1e-3, 100, 1000, 0)
    print("预热完成，开始计算...\n")

    results = [compare_one(n_v) for n_v in n_v_list]

    h_v_vals  = np.array([r[0] for r in results])
    err_disc  = np.array([r[1] for r in results])
    err_em    = np.array([r[2] for r in results])

    idx      = np.argsort(h_v_vals)
    h_v_vals = h_v_vals[idx]
    err_disc = err_disc[idx]
    err_em   = err_em[idx]

    # 参考线
    ref_o1 = 0.1  * h_v_vals
    ref_o2 = (err_disc[0] / h_v_vals[0]**2) * h_v_vals**2

    plt.rcParams.update({'text.usetex': False, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_v_vals, err_disc, 'o-',  color='#1f77b4', label='Discrete', lw=1.5, ms=7, mfc='none')
    ax.loglog(h_v_vals, err_em,   's--', color='#d62728', label='EM',       lw=1.5, ms=7, mfc='none')
    ax.loglog(h_v_vals, ref_o1,   '--',  color='#ff7f0e', label='$O(h)$',   lw=1.2)
    ax.loglog(h_v_vals, ref_o2,   '--',  color='#9467bd', label='$O(h^2)$', lw=1.2)
    ax.set_xlabel('spatial stepsize $h_v$', fontsize=12)
    ax.set_ylabel('$l^1$-error',             fontsize=12)
    ax.set_title('Discrete vs EM (equal compute time)', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    fig.tight_layout()

    out_png = out_dir / "comparison_disc_vs_EM.png"
    fig.savefig(out_png, dpi=300)
    print(f"\nSaved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()