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
#  图1：Discrete 收敛阶  —  err vs h_v
# ─────────────────────────────────────────────────────────────────────────────
def run_discrete_convergence(n_v_list, gamma=1.0, sigma=1.0):
    """对每个 n_v 运行空间离散格式，返回 (h_v, err, time)。"""
    results = []
    for n_v in n_v_list:
        n_x    = n_v ** 2
        x_grid = np.linspace(-4.0, 4.0, n_x)
        v_grid = np.linspace(-4.0, 4.0, n_v)
        h_x    = x_grid[1] - x_grid[0]
        h_v    = v_grid[1] - v_grid[0]

        rho_true = normalize_density(
            ground_true_density(x_grid, v_grid, gamma, sigma), h_x, h_v)

        t0       = time.perf_counter()
        rho_disc = normalize_density(
            discrete_density(x_grid, v_grid, gamma, sigma), h_x, h_v)
        t_disc   = time.perf_counter() - t0
        err      = l1_error(rho_disc, rho_true, h_x, h_v)

        print(f"  [Discrete] n_v={n_v:4d} h_v={h_v:.4f} err={err:.3e} t={t_disc:.2f}s")
        results.append((h_v, err, t_disc))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  图2：EM 收敛阶  —  err vs Δt（固定足够长的轨迹消除统计误差）
# ─────────────────────────────────────────────────────────────────────────────
def run_em_convergence(dt_list, gamma=1.0, sigma=1.0,
                       n_v_ref=40, T_total=20_000.0, n_burnin=200_000):
    """
    固定总物理时间 T_total，扫描不同 dt，
    展示 EM 误差随 Δt 的收敛行为。

    关键：n_steps = T_total / dt
    dt 越小 → 步数越多 → 统计误差保持稳定 → 只有时间离散误差单调下降。
    若固定 n_steps 而缩小 dt，总物理时间缩短，统计误差反而上升，
    会掩盖甚至逆转时间收敛趋势。

    网格固定为 n_v_ref（细网格），使空间投影误差远小于时间离散误差。
    """
    n_x    = n_v_ref ** 2
    x_grid = np.linspace(-4.0, 4.0, n_x)
    v_grid = np.linspace(-4.0, 4.0, n_v_ref)
    h_x    = x_grid[1] - x_grid[0]
    h_v    = v_grid[1] - v_grid[0]
    rho_true = normalize_density(
        ground_true_density(x_grid, v_grid, gamma, sigma), h_x, h_v)

    results = []
    for dt in dt_list:
        n_steps = int(T_total / dt)   # 固定总物理时间
        t0      = time.perf_counter()
        rho_em  = normalize_density(
            em_density(x_grid, v_grid, gamma, sigma,
                       dt=dt, n_steps=n_steps, n_burnin=n_burnin), h_x, h_v)
        t_em    = time.perf_counter() - t0
        err     = l1_error(rho_em, rho_true, h_x, h_v)
        print(f"  [EM conv]  dt={dt:.1e}  n_steps={n_steps:>12,}  err={err:.3e}  t={t_em:.2f}s")
        results.append((dt, err, t_em))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  图3：Work-Error  —  相同计算代价下的公平比较
# ─────────────────────────────────────────────────────────────────────────────
def run_work_error(n_v_list, gamma=1.0, sigma=1.0, dt=1e-3):
    """
    对每个 n_v：
      - Discrete：记录 (计算时间, err)
      - EM：在相同时间预算内跑尽量多步，记录 (实际时间, err)
    """
    results = []
    for n_v in n_v_list:
        n_x    = n_v ** 2
        x_grid = np.linspace(-4.0, 4.0, n_x)
        v_grid = np.linspace(-4.0, 4.0, n_v)
        h_x    = x_grid[1] - x_grid[0]
        h_v    = v_grid[1] - v_grid[0]

        rho_true = normalize_density(
            ground_true_density(x_grid, v_grid, gamma, sigma), h_x, h_v)

        # Discrete
        t0       = time.perf_counter()
        rho_disc = normalize_density(
            discrete_density(x_grid, v_grid, gamma, sigma), h_x, h_v)
        t_disc   = time.perf_counter() - t0
        err_disc = l1_error(rho_disc, rho_true, h_x, h_v)

        # EM：相同时间预算
        t0     = time.perf_counter()
        rho_em = normalize_density(
            em_density(x_grid, v_grid, gamma, sigma,
                       dt=dt, time_budget=t_disc), h_x, h_v)
        t_em   = time.perf_counter() - t0
        err_em = l1_error(rho_em, rho_true, h_x, h_v)

        print(f"  [Work-Err] n_v={n_v:4d} | "
              f"Disc: err={err_disc:.2e} t={t_disc:.2f}s | "
              f"EM:   err={err_em:.2e} t={t_em:.2f}s")
        results.append((t_disc, err_disc, t_em, err_em))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  绘图函数
# ─────────────────────────────────────────────────────────────────────────────
def plot_discrete_convergence(res_disc, out_path):
    h_v  = np.array([r[0] for r in res_disc])
    err  = np.array([r[1] for r in res_disc])
    idx  = np.argsort(h_v)
    h_v, err = h_v[idx], err[idx]

    # 拟合收敛阶
    slope, _ = np.polyfit(np.log(h_v), np.log(err), 1)

    ref_o1 = err[0] / h_v[0]    * h_v
    ref_o2 = err[0] / h_v[0]**2 * h_v**2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h_v, err,    'o-',  color='#1f77b4',
              label=f'Discrete (slope≈{slope:.2f})', lw=1.8, ms=7, mfc='none')
    ax.loglog(h_v, ref_o1, '--',  color='#ff7f0e', label='$O(h)$',   lw=1.2)
    ax.loglog(h_v, ref_o2, '--',  color='#9467bd', label='$O(h^2)$', lw=1.2)
    ax.set_xlabel('spatial stepsize $h_v$', fontsize=12)
    ax.set_ylabel('$L^1$ error',             fontsize=12)
    ax.set_title('Discrete method: spatial convergence', fontsize=13)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_em_convergence(res_em, out_path):
    dt_arr  = np.array([r[0] for r in res_em])
    err_arr = np.array([r[1] for r in res_em])
    idx     = np.argsort(dt_arr)
    dt_arr, err_arr = dt_arr[idx], err_arr[idx]

    slope, _ = np.polyfit(np.log(dt_arr), np.log(err_arr), 1)

    ref_o1 = err_arr[0] / dt_arr[0] * dt_arr

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dt_arr, err_arr, 's-',  color='#d62728',
              label=f'EM (slope≈{slope:.2f})', lw=1.8, ms=7, mfc='none')
    ax.loglog(dt_arr, ref_o1,  '--',  color='#ff7f0e', label='$O(\\Delta t)$', lw=1.2)
    ax.set_xlabel('time stepsize $\\Delta t$', fontsize=12)
    ax.set_ylabel('$L^1$ error',               fontsize=12)
    ax.set_title('EM method: temporal convergence\n'
                 f'(fixed $T=20000$, $n_{{steps}}=T/\\Delta t$, grid $n_v=40$)', fontsize=13)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_work_error(res_work, out_path):
    t_disc   = np.array([r[0] for r in res_work])
    err_disc = np.array([r[1] for r in res_work])
    t_em     = np.array([r[2] for r in res_work])
    err_em   = np.array([r[3] for r in res_work])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(t_disc, err_disc, 'o-',  color='#1f77b4',
              label='Discrete ($\\widetilde{Q}_u$)', lw=1.8, ms=7, mfc='none')
    ax.loglog(t_em,   err_em,   's--', color='#d62728',
              label='EM (same time budget)', lw=1.8, ms=7, mfc='none')
    ax.set_xlabel('wall-clock time (s)', fontsize=12)
    ax.set_ylabel('$L^1$ error',          fontsize=12)
    ax.set_title('Work–error: equal compute time comparison', fontsize=13)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir  = Path(__file__).resolve().parent
    gamma, sigma = 1.0, 1.0

    # ── JIT 预热 ──────────────────────────────────────────────────────────────
    print("JIT 预热中...")
    _build_triplets(np.linspace(-4, 4, 25), np.linspace(-4, 4, 5), gamma, sigma)
    _em_simulate(gamma, sigma, 1e-3, 100, 10_000, 0)
    print("预热完成\n")

    # ── 图1：Discrete 空间收敛 ────────────────────────────────────────────────
    print("=== 图1：Discrete 空间收敛 ===")
    n_v_list  = [20, 40, 60, 80, 100]
    res_disc  = run_discrete_convergence(n_v_list, gamma, sigma)
    plot_discrete_convergence(res_disc, out_dir / "fig1_discrete_convergence.png")

    # ── 图2：EM 时间收敛 ──────────────────────────────────────────────────────
    print("\n=== 图2：EM 时间收敛（固定总物理时间 T=20000，扫描 Δt）===")
    # 固定总物理时间 T_total，dt 越小步数越多，统计误差保持稳定
    # 从粗到细覆盖约两个数量级
    dt_list  = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]
    res_em   = run_em_convergence(dt_list, gamma, sigma,
                                  n_v_ref=40, T_total=20_000.0)
    plot_em_convergence(res_em, out_dir / "fig2_em_convergence.png")

    # ── 图3：Work-Error 公平对比 ──────────────────────────────────────────────
    print("\n=== 图3：Work-Error 公平对比 ===")
    res_work = run_work_error(n_v_list, gamma, sigma, dt=1e-3)
    plot_work_error(res_work, out_dir / "fig3_work_error.png")

    print("\n全部完成。")
    plt.show()


if __name__ == "__main__":
    main()