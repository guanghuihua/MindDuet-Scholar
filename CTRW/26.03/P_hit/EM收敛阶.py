"""
Euler-Maruyama 方法对 underdamped Langevin 方程不变密度的 L1 误差收敛阶实验

系统:
    dq =  p  dt
    dp = (-q - gamma*p) dt + sigma dW

精确不变密度 (Gibbs):
    pi(q, p) ∝ exp(-beta * (q^2/2 + p^2/2)),  beta = 2*gamma/sigma^2

误差指标: L1 误差 (不变密度的直方图估计 vs 解析解)
收敛阶:   log-log 斜率 (对步长 Delta_t)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os

N_THREADS = os.cpu_count()
os.environ.setdefault("OMP_NUM_THREADS",      str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",      str(N_THREADS))


# ─────────────────────────────────────────────────────────────────────────────
#  Euler-Maruyama 长时间模拟，估计不变密度直方图
# ─────────────────────────────────────────────────────────────────────────────
def run_em(
    dt: float,
    T_total: float,
    T_burn: float,
    gamma: float,
    sigma: float,
    n_traj: int,
    x_grid: np.ndarray,
    v_grid: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    运行 EM 格式，收集 (q, p) 样本后直方图估计不变密度。

    格式:
        q_{n+1} = q_n + p_n * dt
        p_{n+1} = p_n + (-q_n - gamma*p_n)*dt + sigma*sqrt(dt)*xi_n

    thin 策略: 每隔约 0.1 个时间单位采一次样，消除自相关。
    """
    rng = np.random.default_rng(seed)

    n_burn  = int(T_burn  / dt)
    n_total = int(T_total / dt)
    n_stat  = n_total - n_burn

    # 初始化：从标准正态出发
    q = rng.standard_normal(n_traj)
    p = rng.standard_normal(n_traj)

    # ── 预热阶段（burn-in） ────────────────────────────────────────────────
    sqrt_dt = np.sqrt(dt)
    for _ in range(n_burn):
        xi    = rng.standard_normal(n_traj)
        q_new = q + p * dt
        p_new = p + (-q - gamma * p) * dt + sigma * sqrt_dt * xi
        q, p  = q_new, p_new

    # ── 统计阶段：每隔 thin 步采样一次 ────────────────────────────────────
    thin = max(1, int(0.1 / dt))
    q_samples = []
    p_samples = []

    step_count = 0
    for _ in range(n_stat):
        xi    = rng.standard_normal(n_traj)
        q_new = q + p * dt
        p_new = p + (-q - gamma * p) * dt + sigma * sqrt_dt * xi
        q, p  = q_new, p_new
        step_count += 1
        if step_count % thin == 0:
            q_samples.append(q.copy())
            p_samples.append(p.copy())

    q_all = np.concatenate(q_samples)
    p_all = np.concatenate(p_samples)

    # ── 直方图估计密度 ─────────────────────────────────────────────────────
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]
    x_edges = np.concatenate([[x_grid[0]  - h_x/2],
                               (x_grid[:-1] + x_grid[1:]) / 2,
                               [x_grid[-1] + h_x/2]])
    v_edges = np.concatenate([[v_grid[0]  - h_v/2],
                               (v_grid[:-1] + v_grid[1:]) / 2,
                               [v_grid[-1] + h_v/2]])

    hist, _, _ = np.histogram2d(q_all, p_all, bins=[x_edges, v_edges])
    rho = hist / (hist.sum() * h_x * h_v)
    return rho


# ─────────────────────────────────────────────────────────────────────────────
#  解析不变密度与辅助函数
# ─────────────────────────────────────────────────────────────────────────────
def ground_true_density(x_grid, v_grid, gamma, sigma):
    beta = 2.0 * gamma / (sigma ** 2)
    x_mesh, v_mesh = np.meshgrid(x_grid, v_grid, indexing="ij")
    energy = 0.5 * x_mesh ** 2 + 0.5 * v_mesh ** 2
    return np.exp(-beta * energy)


def normalize_density(rho, h_x, h_v):
    mass = np.sum(rho) * h_x * h_v
    return rho / mass if mass > 0 else np.zeros_like(rho)


def l1_error(rho_a, rho_b, h_x, h_v):
    return float(np.sum(np.abs(rho_a - rho_b)) * h_x * h_v)


# ─────────────────────────────────────────────────────────────────────────────
#  单个步长的误差计算
# ─────────────────────────────────────────────────────────────────────────────
def l1_error_em(
    dt: float,
    gamma: float = 1.0,
    sigma: float = 1.0,
    T_total: float = 5e5,
    T_burn:  float = 1e4,
    n_traj:  int   = 20,
    n_grid:  int   = 80,
    l_x:     float = 4.0,
    l_v:     float = 4.0,
) -> tuple[float, float]:
    x_grid = np.linspace(-l_x, l_x, n_grid)
    v_grid = np.linspace(-l_v, l_v, n_grid)
    h_x = x_grid[1] - x_grid[0]
    h_v = v_grid[1] - v_grid[0]

    rho_true = normalize_density(
        ground_true_density(x_grid, v_grid, gamma, sigma), h_x, h_v
    )

    t0 = time.perf_counter()
    rho_em = run_em(
        dt=dt, T_total=T_total, T_burn=T_burn,
        gamma=gamma, sigma=sigma, n_traj=n_traj,
        x_grid=x_grid, v_grid=v_grid,
    )
    t_em = time.perf_counter() - t0

    err = l1_error(rho_em, rho_true, h_x, h_v)
    print(f"  dt={dt:.5f} | error={err:.3e} | time={t_em:.1f}s")
    return dt, err


# ─────────────────────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(__file__).resolve().parent

    gamma   = 1.0
    sigma   = 1.0
    T_total = 5e5
    T_burn  = 1e4
    n_traj  = 20
    n_grid  = 80

    # 步长序列：约 1.5 个数量级，log 空间均匀分布
    dt_list = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]

    print("=" * 60)
    print("EM 方法不变密度 L1 误差收敛阶实验")
    print(f"系统: underdamped Langevin,  gamma={gamma}, sigma={sigma}")
    print(f"T_total={T_total:.0e}, T_burn={T_burn:.0e}, n_traj={n_traj}")
    print(f"密度估计网格: {n_grid}x{n_grid}, 域: [-4,4]^2")
    print("=" * 60)

    results = []
    for dt in dt_list:
        res = l1_error_em(
            dt=dt, gamma=gamma, sigma=sigma,
            T_total=T_total, T_burn=T_burn,
            n_traj=n_traj, n_grid=n_grid,
        )
        results.append(res)

    dt_vals  = np.array([r[0] for r in results])
    err_vals = np.array([r[1] for r in results])

    # ── log-log 斜率（全局收敛阶） ────────────────────────────────────────
    log_dt  = np.log(dt_vals)
    log_err = np.log(err_vals)
    slope, _ = np.polyfit(log_dt, log_err, 1)
    local_slopes = np.diff(log_err) / np.diff(log_dt)

    print(f"\n全局 log-log 斜率 (估计收敛阶): {slope:.3f}")
    print("相邻点局部斜率:", [f"{s:.3f}" for s in local_slopes])

    # ── 参考线 ───────────────────────────────────────────────────────────
    h0, e0 = dt_vals[0], err_vals[0]
    ref_o05 = e0 * (dt_vals / h0) ** 0.5
    ref_o1  = e0 * (dt_vals / h0) ** 1.0

    # ── 绘图 ──────────────────────────────────────────────────────────────
    plt.rcParams.update({'text.usetex': False, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(dt_vals, err_vals, 'o-', color='#1f77b4',
              label=r'EM ($\gamma=1,\;\sigma=1$)', lw=1.5, ms=7, mfc='none')
    ax.loglog(dt_vals, ref_o05,  '--', color='#ff7f0e',
              label=r'$O(\Delta t^{1/2})$', lw=1.2)
    ax.loglog(dt_vals, ref_o1,   '--', color='#9467bd',
              label=r'$O(\Delta t)$', lw=1.2)

    ax.text(0.35, 0.72,
            f'estimated order $\\approx {slope:.2f}$',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.set_xlabel(r'time stepsize $\Delta t$', fontsize=13)
    ax.set_ylabel(r'$L^1$-error (stationary density)', fontsize=13)
    ax.set_title('EM Convergence: Underdamped Langevin\n'
                 r'$dq=p\,dt,\quad dp=(-q-\gamma p)\,dt+\sigma\,dW$',
                 fontsize=13)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    fig.tight_layout()

    out_png = out_dir / "em_convergence_L1.png"
    fig.savefig(out_png, dpi=300)
    print(f"\n图像已保存: {out_png}")
    plt.show()

    # ── 汇总表 ────────────────────────────────────────────────────────────
    print("\n汇总表:")
    print(f"{'Delta_t':>10}  {'L1 error':>12}  {'local order':>12}")
    print("-" * 38)
    for i, (dt, err) in enumerate(zip(dt_vals, err_vals)):
        if i == 0:
            print(f"{dt:>10.5f}  {err:>12.3e}  {'—':>12}")
        else:
            s = (np.log(err_vals[i]) - np.log(err_vals[i-1])) / \
                (np.log(dt_vals[i])  - np.log(dt_vals[i-1]))
            print(f"{dt:>10.5f}  {err:>12.3e}  {s:>12.3f}")
    print(f"\n全局 log-log 斜率 = {slope:.3f}")


if __name__ == "__main__":
    main()