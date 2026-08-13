"""
van der Pol 随机慢-快系统：SSA 方法 vs Euler-Maruyama 方法比较
基于全局方程，验证理论预测 sigma_c ~ epsilon^(1/3)

全局方程（van der Pol）：
    epsilon * dx = (y - x^3/3 + x) dt
    dy = (a - x) dt + sigma dW_t

物理设置：
    路径从正值慢流形稳定支 (x0=1.5) 出发，沿稳定支向鞍结点 (1, -2/3) 爬行
    测量路径首次穿越 x=1（离开正值支）时对应的 y 值

核心观测量：
    y_cross = 路径首次穿越 x=1 时的 y 值
    确定性：y_det（固定值）
    带噪声：y_cross 是随机变量，均值 < y_det，标准差随 sigma 增大

理论预测（书 p.198）：
    std(x at y=y_check) ~ sigma * sqrt(epsilon) / |y_check - y_c|^(3/4)
    sigma_c = epsilon^(1/3) 是散布达到 O(1) 的临界值
    对应全局量：std(y_cross) ~ sigma * sqrt(epsilon) ~ epsilon^(5/6) at sigma=sigma_c

SSA 方法：
    对慢变量 y 用 SSA（Gillespie），对快变量 x 用 EM（在等待时间内积分）
    体现慢-快分离的数值格式

EM 方法：
    对两个变量统一用 Euler-Maruyama，步长 dt = epsilon/500
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 避免中文字体警告
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  全局参数
# ─────────────────────────────────────────────────────────────────────────────
A_PARAM = 0.8    # van der Pol 参数 a
X_START = 1.5   # 初始 x（正值慢流形稳定支）


def y_manifold(x):
    """正值慢流形 y = x^3/3 - x"""
    return x**3 / 3.0 - x


def get_det_trajectory(epsilon, dt=None):
    """计算确定性轨道：从 (X_START, y_manifold(X_START)) 出发
    到首次 x 下穿 1 的时刻及 y 值"""
    if dt is None:
        dt = epsilon / 500.0
    x0 = X_START
    y0 = y_manifold(x0)
    x, y = x0, y0
    a = A_PARAM
    for k in range(10_000_000):
        px = x
        x += (y - x**3/3 + x) / epsilon * dt
        y += (a - x) * dt
        if px >= 1.0 and x < 1.0:
            return (k+1)*dt, y, x0, y0
    return None, None, x0, y0


# ─────────────────────────────────────────────────────────────────────────────
#  Part 1: EM 方法（向量化）
# ─────────────────────────────────────────────────────────────────────────────

def run_em(
    epsilon: float,
    sigma: float,
    n_paths: int = 600,
    dt: float = None,
    T_factor: float = 4.0,
    record_n: int = 8,
) -> dict:
    """
    向量化 EM：模拟 n_paths 条路径。
    记录每条路径首次 x 下穿 1 时的 y 值（y_cross）。
    """
    if dt is None:
        dt = epsilon / 500.0

    t_det, y_det, x0, y0 = get_det_trajectory(epsilon, dt)
    T_max   = t_det * T_factor
    n_steps = int(T_max / dt)
    sqrt_dt = np.sqrt(dt)
    a       = A_PARAM

    t_start = time.perf_counter()

    x = np.full(n_paths, x0)
    y = np.full(n_paths, y0)
    y_cross  = np.full(n_paths, np.nan)
    t_cross  = np.full(n_paths, T_max)
    crossed  = np.zeros(n_paths, dtype=bool)
    prev_x   = x.copy()

    # 稀疏记录部分路径
    rec_x = [[x0] for _ in range(record_n)]
    rec_y = [[y0] for _ in range(record_n)]
    rec_t = [0.0]
    rec_ev = max(n_steps // 300, 1)

    for k in range(n_steps):
        dW     = np.random.randn(n_paths) * sqrt_dt
        prev_x[:] = x
        x     += (y - x**3/3 + x) / epsilon * dt
        y     += (a - x) * dt + sigma * dW

        t_now  = (k + 1) * dt
        # 首次从 x>=1 穿越到 x<1
        c = (~crossed) & (prev_x >= 1.0) & (x < 1.0)
        y_cross[c] = y[c]
        t_cross[c] = t_now
        crossed |= c

        if (k+1) % rec_ev == 0:
            rec_t.append(t_now)
            for i in range(record_n):
                rec_x[i].append(float(x[i]))
                rec_y[i].append(float(y[i]))

    elapsed  = time.perf_counter() - t_start
    valid    = y_cross[~np.isnan(y_cross)]
    mean_yc  = float(np.mean(valid))  if len(valid) > 0 else np.nan
    std_yc   = float(np.std(valid))   if len(valid) > 0 else np.nan
    # 提前跳变：y_cross > y_det（比确定性更早，y 更高时就穿越了）
    # 注意：y_det < y_c，噪声使路径更早穿越意味着 y_cross > y_det
    early_prob = float(np.mean(valid > y_det + 0.01)) if len(valid)>0 else np.nan

    return {
        "y_cross":   valid,
        "mean_yc":   mean_yc,
        "std_yc":    std_yc,
        "early_prob": early_prob,
        "y_det":     y_det,
        "elapsed":   elapsed,
        "rec_x":     [np.array(rec_x[i]) for i in range(record_n)],
        "rec_y":     [np.array(rec_y[i]) for i in range(record_n)],
        "rec_t":     np.array(rec_t),
        "x0": x0, "y0": y0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Part 2: SSA + EM 混合方法
#  慢变量 y 用 SSA（Gillespie），快变量 x 用 EM（在每个等待时间内积分）
# ─────────────────────────────────────────────────────────────────────────────

def ssa_em_single(
    epsilon: float,
    sigma: float,
    h_y: float,
    dt_fast: float,
    x0: float,
    y0: float,
    T_max: float,
) -> float:
    """单条路径的 SSA+EM 混合模拟，返回首次 x 下穿 1 时的 y 值"""
    sig2 = sigma**2
    x, y = x0, y0
    t = 0.0
    a = A_PARAM
    prev_x = x

    while t < T_max:
        # 慢变量 SSA 步骤
        mu_y = a - x
        m_y  = max(sig2 - abs(mu_y) * h_y, 0.0) / 2.0
        q_up = max(mu_y, 0.0) / h_y + m_y / h_y**2
        q_dn = max(-mu_y, 0.0) / h_y + m_y / h_y**2
        lam  = q_up + q_dn

        if lam < 1e-14:
            break

        tau = -np.log(np.random.random()) / lam
        if t + tau > T_max:
            tau = T_max - t

        # 快变量 EM 积分（在 tau 时间内，y 固定）
        n_fast = max(int(tau / dt_fast), 1)
        dt_sub = tau / n_fast
        for _ in range(n_fast):
            prev_x = x
            x += (y - x**3/3 + x) / epsilon * dt_sub
            # 检测穿越
            if prev_x >= 1.0 and x < 1.0:
                return y   # 返回穿越时的 y 值

        t += tau

        # 慢变量跳跃
        r = np.random.random() * lam
        if r < q_up:
            y += h_y
        else:
            y -= h_y

        # 边界截断
        y = np.clip(y, y0 - 2.0, y0 + 3.0)

    return np.nan


def run_ssa(
    epsilon: float,
    sigma: float,
    n_paths: int = 600,
    h_y: float = 0.05,
    dt_fast: float = None,
    T_factor: float = 4.0,
) -> dict:
    """运行 SSA+EM 混合方法"""
    if dt_fast is None:
        dt_fast = epsilon / 500.0

    t_det, y_det, x0, y0 = get_det_trajectory(epsilon, dt_fast)
    T_max = t_det * T_factor

    t_start  = time.perf_counter()
    y_cross  = np.empty(n_paths)

    for i in range(n_paths):
        y_cross[i] = ssa_em_single(
            epsilon, sigma, h_y, dt_fast, x0, y0, T_max
        )

    elapsed  = time.perf_counter() - t_start
    valid    = y_cross[~np.isnan(y_cross)]
    mean_yc  = float(np.mean(valid))  if len(valid) > 0 else np.nan
    std_yc   = float(np.std(valid))   if len(valid) > 0 else np.nan
    early_prob = float(np.mean(valid > y_det + 0.01)) if len(valid)>0 else np.nan

    return {
        "y_cross":    valid,
        "mean_yc":    mean_yc,
        "std_yc":     std_yc,
        "early_prob": early_prob,
        "y_det":      y_det,
        "elapsed":    elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Part 3: 扫描实验
# ─────────────────────────────────────────────────────────────────────────────

def experiment_sigma_scan(
    epsilon: float = 0.1,
    n_sigma: int = 12,
    n_paths: int = 600,
    h_y: float = 0.05,
) -> dict:
    """固定 epsilon，扫描 sigma，比较 SSA 和 EM 的 std(y_cross)"""
    sigma_c    = epsilon**(1.0/3.0)
    sigma_list = np.logspace(
        np.log10(sigma_c * 0.02),
        np.log10(sigma_c * 6.0),
        n_sigma
    )

    t_det, y_det, x0, y0 = get_det_trajectory(epsilon)
    print(f"\n{'='*65}")
    print(f"epsilon={epsilon:.3f},  sigma_c={sigma_c:.4f}")
    print(f"初始点: ({x0:.2f}, {y0:.4f})")
    print(f"确定性 y_cross = {y_det:.5f}")
    print(f"{'='*65}")

    ssa_std, em_std   = [], []
    ssa_ep,  em_ep    = [], []
    ssa_t,   em_t     = [], []
    theory_std_list   = []

    for sigma in sigma_list:
        ratio = sigma / sigma_c
        print(f"\n--- sigma={sigma:.4f}  (sigma/sigma_c={ratio:.2f}) ---")

        res_ssa = run_ssa(epsilon=epsilon, sigma=sigma,
                          n_paths=n_paths, h_y=h_y)
        ssa_std.append(res_ssa["std_yc"])
        ssa_ep.append(res_ssa["early_prob"])
        ssa_t.append(res_ssa["elapsed"])
        print(f"  SSA: std={res_ssa['std_yc']:.5f}, "
              f"early={res_ssa['early_prob']:.3f}, "
              f"time={res_ssa['elapsed']:.1f}s")

        res_em = run_em(epsilon=epsilon, sigma=sigma, n_paths=n_paths)
        em_std.append(res_em["std_yc"])
        em_ep.append(res_em["early_prob"])
        em_t.append(res_em["elapsed"])
        print(f"  EM : std={res_em['std_yc']:.5f}, "
              f"early={res_em['early_prob']:.3f}, "
              f"time={res_em['elapsed']:.1f}s")

        # 理论预测 std(y_cross) ~ sigma * sqrt(epsilon)
        theory_std_list.append(sigma * np.sqrt(epsilon))

    return {
        "sigma_list":   sigma_list,
        "sigma_c":      sigma_c,
        "epsilon":      epsilon,
        "y_det":        y_det,
        "ssa_std":      np.array(ssa_std),
        "em_std":       np.array(em_std),
        "ssa_ep":       np.array(ssa_ep),
        "em_ep":        np.array(em_ep),
        "theory_std":   np.array(theory_std_list),
        "ssa_t":        np.array(ssa_t),
        "em_t":         np.array(em_t),
    }


def experiment_epsilon_scaling(
    epsilon_list=None,
    n_paths: int = 500,
    h_y: float = 0.05,
) -> dict:
    """扫描 epsilon，验证 sigma_c ~ epsilon^(1/3) 的标度律
    通过测量 std(y_cross) at sigma=sigma_c 随 epsilon 的变化"""
    if epsilon_list is None:
        epsilon_list = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]

    sigma_c_list = np.array([eps**(1/3) for eps in epsilon_list])
    ssa_std_list = []
    em_std_list  = []

    for eps, sc in zip(epsilon_list, sigma_c_list):
        print(f"\n=== epsilon={eps:.3f}, sigma_c={sc:.4f} ===")
        sigma = sc

        r_ssa = run_ssa(epsilon=eps, sigma=sigma, n_paths=n_paths, h_y=h_y)
        r_em  = run_em(epsilon=eps, sigma=sigma, n_paths=n_paths)
        ssa_std_list.append(r_ssa["std_yc"])
        em_std_list.append(r_em["std_yc"])
        print(f"  SSA std={r_ssa['std_yc']:.5f}, EM std={r_em['std_yc']:.5f}")

    return {
        "epsilon_list":  np.array(epsilon_list),
        "sigma_c_list":  sigma_c_list,
        "ssa_std":       np.array(ssa_std_list),
        "em_std":        np.array(em_std_list),
        # 理论：std ~ sigma_c * sqrt(eps) = eps^(1/3+1/2) = eps^(5/6)
        "theory":        np.array([eps**(5/6) for eps in epsilon_list]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Part 4: 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_paths(epsilon: float, sigma: float, out_dir: Path):
    """比较 SSA 和 EM 的样本路径"""
    sigma_c = epsilon**(1/3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    t_det, y_det, x0, y0 = get_det_trajectory(epsilon)

    for ax, label, color in zip(axes, ['EM', 'SSA'], ['#1f77b4', '#d62728']):
        # 慢流形
        x_mf  = np.linspace(1.0, 2.0, 200)
        y_mf  = y_manifold(x_mf)
        ax.plot(x_mf, y_mf, 'k--', lw=1.8, alpha=0.7, label='Slow manifold')
        ax.axvline(x=1.0, color='orange', ls='--', lw=1.5,
                   label='x=1 (saddle node)')
        ax.axhline(y=-2/3, color='gray', ls=':', lw=1.2,
                   label=f'y_c = {-2/3:.3f}')
        ax.plot(x0, y0, 'go', ms=10, zorder=5, label='Start')
        ax.plot(1.0, -2/3, 'k*', ms=12, zorder=5, label='Saddle node')

        n_show = 8
        if label == 'EM':
            res = run_em(epsilon=epsilon, sigma=sigma,
                         n_paths=n_show, record_n=n_show)
            for i in range(n_show):
                ax.plot(res["rec_x"][i], res["rec_y"][i],
                        color=color, alpha=0.5, lw=0.9)
        else:
            dt_fast = epsilon / 500.0
            T_max   = t_det * 4
            for _ in range(n_show):
                xi, yi = x0, y0
                xp, yp = [xi], [yi]
                sig2 = sigma**2
                t = 0.0
                h_y = 0.05
                a = A_PARAM
                while t < T_max and len(xp) < 2000:
                    mu_y = a - xi
                    m_y  = max(sig2 - abs(mu_y)*h_y, 0)/2
                    q_up = max(mu_y,0)/h_y + m_y/h_y**2
                    q_dn = max(-mu_y,0)/h_y + m_y/h_y**2
                    lam  = q_up + q_dn
                    if lam < 1e-14: break
                    tau = -np.log(np.random.random())/lam
                    n_f = max(int(tau/dt_fast), 1)
                    dts = tau/n_f
                    done = False
                    for _ in range(n_f):
                        px = xi
                        xi += (yi - xi**3/3 + xi)/epsilon * dts
                        if px >= 1.0 and xi < 1.0:
                            done = True
                            break
                    xp.append(xi)
                    yp.append(yi)
                    t += tau
                    if done: break
                    r = np.random.random()*lam
                    yi = yi + h_y if r < q_up else yi - h_y
                    yi = np.clip(yi, y0-2, y0+3)
                ax.plot(xp, yp, color=color, alpha=0.5, lw=0.9)

        ax.set_xlim(0.5, 2.2)
        ax.set_ylim(-1.2, 0.2)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        ax.set_title(
            f'{label} sample paths\n'
            fr'$\varepsilon={epsilon}$, $\sigma/\sigma_c={sigma/sigma_c:.2f}$',
            fontsize=11
        )
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        'van der Pol: paths from positive stable branch toward saddle node\n'
        'Orange line x=1: first crossing defines y_cross',
        fontsize=11
    )
    fig.tight_layout()
    p = out_dir / "sample_paths.png"
    fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close()


def plot_sigma_scan(results: dict, out_dir: Path):
    """三张图：std, early prob, 计算时间"""
    sl     = results["sigma_list"]
    sc     = results["sigma_c"]
    ratio  = sl / sc
    eps    = results["epsilon"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 图1：std(y_cross) vs sigma
    ax = axes[0]
    ax.loglog(ratio, results["ssa_std"], 'o-',
              color='#d62728', lw=2, ms=8, label='SSA+EM')
    ax.loglog(ratio, results["em_std"],  's--',
              color='#1f77b4', lw=2, ms=8, label='EM')
    ax.loglog(ratio, results["theory_std"], 'k--', lw=1.8,
              label=r'Theory: $\sigma\sqrt{\varepsilon}$')
    ax.axvline(x=1.0, color='#333', ls=':', lw=2.0,
               label=r'$\sigma=\sigma_c$')
    ax.set_xlabel(r'$\sigma/\sigma_c$ (log scale)', fontsize=12)
    ax.set_ylabel(r'std$(y_\mathrm{cross})$', fontsize=12)
    ax.set_title(
        fr'Crossing y spread vs $\sigma$  ($\varepsilon={eps}$)'
        '\nShould scale as ' + r'$\sigma\sqrt{\varepsilon}$',
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # 图2：early prob vs sigma
    ax = axes[1]
    ax.semilogx(ratio, results["ssa_ep"], 'o-',
                color='#d62728', lw=2, ms=8, label='SSA+EM')
    ax.semilogx(ratio, results["em_ep"],  's--',
                color='#1f77b4', lw=2, ms=8, label='EM')
    ax.axvline(x=1.0, color='#333', ls=':', lw=2.0,
               label=r'$\sigma=\sigma_c$')
    ax.set_xlabel(r'$\sigma/\sigma_c$ (log scale)', fontsize=12)
    ax.set_ylabel(r'$P(y_\mathrm{cross} > y_\mathrm{det}+0.01)$', fontsize=12)
    ax.set_title(
        'Probability of early crossing\n'
        r'(cross at higher $y$ than deterministic)',
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    # 图3：计算时间
    ax = axes[2]
    xp = np.arange(len(sl))
    w  = 0.35
    ax.bar(xp-w/2, results["ssa_t"], w, color='#d62728', alpha=0.85, label='SSA+EM')
    ax.bar(xp+w/2, results["em_t"],  w, color='#1f77b4', alpha=0.85, label='EM')
    ax.set_xticks(xp)
    ax.set_xticklabels([f'{r:.2f}' for r in ratio], rotation=45, fontsize=8)
    ax.set_xlabel(r'$\sigma/\sigma_c$', fontsize=12)
    ax.set_ylabel('Computation time (s)', fontsize=12)
    ax.set_title('Computation time: SSA vs EM', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    p = out_dir / "sigma_scan.png"
    fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close()


def plot_epsilon_scaling(results: dict, out_dir: Path):
    """std(y_cross) vs epsilon 的标度律图"""
    eps_arr  = results["epsilon_list"]
    sc_arr   = results["sigma_c_list"]
    ssa_std  = results["ssa_std"]
    em_std   = results["em_std"]
    theory   = results["theory"]

    # 归一化到第一个点
    norm_ssa = ssa_std / ssa_std[0] * theory[0]
    norm_em  = em_std  / em_std[0]  * theory[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(eps_arr, ssa_std,  'o-',  color='#d62728', lw=2, ms=9,
              label='SSA+EM')
    ax.loglog(eps_arr, em_std,   's-',  color='#1f77b4', lw=2, ms=9,
              label='EM')
    ax.loglog(eps_arr, theory,   'k--', lw=2.0,
              label=r'Theory: $\varepsilon^{5/6}$')

    # 标注 1/3 斜率参考线
    x_ref = eps_arr
    y_ref = ssa_std[0] * (x_ref/eps_arr[0])**(1/3)
    ax.loglog(x_ref, y_ref, 'b:', lw=1.2,
              label=r'Slope 1/3 reference')

    ax.set_xlabel(r'$\varepsilon$', fontsize=13)
    ax.set_ylabel(r'std$(y_\mathrm{cross})$ at $\sigma=\sigma_c$', fontsize=13)
    ax.set_title(
        r'Scaling: std$(y_\mathrm{cross}) \sim \varepsilon^{5/6}$ at $\sigma=\sigma_c$'
        '\n(log-log, theory slope = 5/6)',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # 拟合斜率
    for label, vals, color in [('SSA+EM', ssa_std, '#d62728'),
                                ('EM',    em_std,  '#1f77b4')]:
        valid = np.isfinite(vals) & (vals > 0)
        if valid.sum() >= 3:
            slope = np.polyfit(np.log(eps_arr[valid]),
                               np.log(vals[valid]), 1)[0]
            print(f"  {label}: fitted slope = {slope:.3f}  "
                  f"(theory 5/6 = {5/6:.3f})")

    fig.tight_layout()
    p = out_dir / "epsilon_scaling.png"
    fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close()


def plot_ycross_distribution(epsilon: float, out_dir: Path):
    """比较不同 sigma 下 y_cross 的分布（SSA vs EM）"""
    sigma_c = epsilon**(1/3)
    _, y_det, _, _ = get_det_trajectory(epsilon)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, ratio in zip(axes, [0.3, 2.0]):
        sigma = ratio * sigma_c
        r_em  = run_em(epsilon=epsilon, sigma=sigma, n_paths=1000)
        r_ssa = run_ssa(epsilon=epsilon, sigma=sigma, n_paths=1000)

        bins = np.linspace(y_det - 0.6, y_det + 0.2, 40)
        ax.hist(r_em["y_cross"],  bins=bins, density=True,
                alpha=0.5, color='#1f77b4', label='EM')
        ax.hist(r_ssa["y_cross"], bins=bins, density=True,
                alpha=0.5, color='#d62728', label='SSA+EM')
        ax.axvline(x=y_det, color='k', ls='--', lw=2,
                   label=f'Deterministic y_det={y_det:.3f}')

        ax.set_xlabel(r'$y_\mathrm{cross}$', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(
            fr'$\sigma/\sigma_c={ratio}$,  $\varepsilon={epsilon}$'
            f'\nEM std={r_em["std_yc"]:.4f}, SSA std={r_ssa["std_yc"]:.4f}',
            fontsize=11
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        r'Distribution of $y_\mathrm{cross}$ (y value when x crosses 1)'
        '\nShift and spread increase with noise',
        fontsize=12
    )
    fig.tight_layout()
    p = out_dir / "ycross_distribution.png"
    fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close()


def print_summary(results: dict):
    sc = results["sigma_c"]
    yd = results["y_det"]
    print(f"\n{'='*75}")
    print(f"epsilon={results['epsilon']:.3f}, sigma_c={sc:.4f}, y_det={yd:.5f}")
    print(f"\n{'sigma':>10} {'s/sc':>6} "
          f"{'SSA std':>9} {'EM std':>9} "
          f"{'theory':>9} {'SSA ep':>8} {'EM ep':>7} "
          f"{'SSA(s)':>7} {'EM(s)':>6}")
    print("-"*75)
    for s, ss, es, ts, sep, eep, st, et in zip(
        results["sigma_list"],
        results["ssa_std"], results["em_std"],
        results["theory_std"],
        results["ssa_ep"], results["em_ep"],
        results["ssa_t"], results["em_t"],
    ):
        print(f"{s:10.4f} {s/sc:6.2f} "
              f"{ss:9.5f} {es:9.5f} "
              f"{ts:9.5f} {sep:8.3f} {eep:7.3f} "
              f"{st:7.1f} {et:6.1f}")


# ─────────────────────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).resolve().parent
    epsilon = 0.1
    sigma_c = epsilon**(1/3)

    print(f"van der Pol: epsilon={epsilon}, a={A_PARAM}")
    print(f"sigma_c = epsilon^(1/3) = {sigma_c:.4f}")

    # Step 1: 样本路径
    print("\nStep 1: Sample paths (sigma = sigma_c)")
    plot_sample_paths(epsilon=epsilon, sigma=sigma_c, out_dir=out_dir)

    # Step 2: sigma 扫描
    print("\nStep 2: Sigma scan")
    results = experiment_sigma_scan(
        epsilon=epsilon,
        n_sigma=12,
        n_paths=600,
        h_y=0.05,
    )
    plot_sigma_scan(results, out_dir)
    print_summary(results)

    # Step 3: y_cross 分布
    print("\nStep 3: y_cross distribution")
    plot_ycross_distribution(epsilon=epsilon, out_dir=out_dir)

    # Step 4: epsilon 标度律
    print("\nStep 4: Epsilon scaling")
    results_eps = experiment_epsilon_scaling(
        epsilon_list=[0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        n_paths=500,
    )
    plot_epsilon_scaling(results_eps, out_dir)


if __name__ == "__main__":
    main()