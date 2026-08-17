"""
van der Pol 随机慢-快系统：SSA 方法 vs Euler-Maruyama 方法比较
验证理论预测：临界噪声强度 sigma_c ~ epsilon^(1/3)

使用书中 (6.1.20) 的局部坐标方程（在鞍结点附近）：
    epsilon * d x_tilde = (-y_tilde - x_tilde^2) dt
    d y_tilde = dt + sigma dW_t

坐标含义：
    x_tilde = x - 1        (x=1 是鞍结点)
    y_tilde = -(y + 2/3)   (y=-2/3 是鞍结点，取负号使y_tilde从负增到正)

初始点：xt0 < 0, yt0 < 0（鞍结点左下方，路径从这里出发爬向鞍结点）
提前跳变定义：在 y_tilde < 0（还未到鞍结点）时，x_tilde 就已经 > 0
              即路径在到达鞍结点前就越过了 x=1

理论预测（书 p.198-199）：
    路径散布 ~ sigma * sqrt(epsilon) / |y_tilde|^(3/4)
    sigma_c = epsilon^(1/3) 是散布在 y_tilde -> 0 时达到 O(1) 的临界值

SSA 方法：
    对慢变量 y_tilde 用 SSA（Gillespie 格式）
    对快变量 x_tilde 用 EM（步长 dt_fast << epsilon）
    这种混合格式体现了慢-快分离的思想

EM 方法：
    对两个变量均用统一的 EM，步长 dt = epsilon/200
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  全局参数
# ─────────────────────────────────────────────────────────────────────────────
XT0   = -0.5    # 初始 x_tilde（鞍结点左方）
YT0   = -0.5    # 初始 y_tilde（鞍结点下方）
X_TH  =  0.0   # 跳变阈值（x_tilde > 0 即 x > 1）
Y_TH  =  0.0   # y 方向判定（y_tilde < 0 才算提前跳变）
T_MAX =  0.8   # 最大模拟时间（y_tilde 从 YT0 走到 0 约需 |YT0| = 0.5）


# ─────────────────────────────────────────────────────────────────────────────
#  Part 1: EM 方法（向量化）
# ─────────────────────────────────────────────────────────────────────────────

def run_em(
    epsilon: float,
    sigma: float,
    n_paths: int = 500,
    dt: float = None,
    record_paths: int = 8,
) -> dict:
    """
    向量化 Euler-Maruyama，模拟 n_paths 条路径。

    时间步 dt 取 epsilon/200（保证快变量精度）。
    提前跳变：x_tilde > X_TH 且 y_tilde < Y_TH（还未到鞍结点）。
    """
    if dt is None:
        dt = epsilon / 200.0

    n_steps = int(T_MAX / dt)
    sqrt_dt = np.sqrt(dt)
    t_start = time.perf_counter()

    xt = np.full(n_paths, XT0, dtype=np.float64)
    yt = np.full(n_paths, YT0, dtype=np.float64)

    jump_times = np.full(n_paths, T_MAX)
    jumped     = np.zeros(n_paths, dtype=bool)

    # 稀疏记录几条路径用于画图
    rec_xt = [[XT0] for _ in range(record_paths)]
    rec_yt = [[YT0] for _ in range(record_paths)]
    rec_t  = [0.0]
    rec_ev = max(n_steps // 200, 1)

    for k in range(n_steps):
        dW   = np.random.randn(n_paths) * sqrt_dt
        xt  += (-yt - xt**2) / epsilon * dt
        yt  += dt + sigma * dW
        xt   = np.clip(xt, -5.0, 5.0)

        t_now = (k + 1) * dt
        new_j = (~jumped) & (xt > X_TH) & (yt < Y_TH)
        jump_times[new_j] = t_now
        jumped |= new_j

        if (k + 1) % rec_ev == 0:
            rec_t.append(t_now)
            for i in range(record_paths):
                rec_xt[i].append(float(xt[i]))
                rec_yt[i].append(float(yt[i]))

    elapsed   = time.perf_counter() - t_start
    jump_prob = float(jumped.mean())
    mean_jt   = float(jump_times[jumped].mean()) if jumped.any() else np.nan

    return {
        "jump_prob":  jump_prob,
        "mean_jt":    mean_jt,
        "jump_times": jump_times,
        "elapsed":    elapsed,
        "rec_xt":     [np.array(rec_xt[i]) for i in range(record_paths)],
        "rec_yt":     [np.array(rec_yt[i]) for i in range(record_paths)],
        "rec_t":      np.array(rec_t),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Part 2: SSA + EM 混合方法（慢变量用SSA，快变量用EM）
#
#  慢-快分离思想：
#    y_tilde 是慢变量，漂移速率 O(1)，扩散 sigma
#    x_tilde 是快变量，弛豫时间 O(epsilon)
#
#  混合格式：
#    (1) 慢变量 y_tilde 用 SSA Gillespie 步骤
#        步长 h_y，速率 q_up = mu_y/h + m_y/h^2, q_dn = m_y/h^2
#        其中混合项 m_y = max(sigma^2 - |mu_y|*h, 0)/2
#    (2) 快变量 x_tilde 在每个 SSA 步内用 EM 积分
#        在慢变量跳跃等待时间 tau 内，对 x_tilde 做 tau/dt_fast 步 EM
#        y_tilde 在此期间保持不变（慢变量假设固定）
# ─────────────────────────────────────────────────────────────────────────────

def ssa_em_single(
    epsilon: float,
    sigma: float,
    h_y: float,
    dt_fast: float,
) -> tuple[float, list, list]:
    """
    单条路径的 SSA+EM 混合模拟。

    返回：
        jump_time : 跳变时刻（未跳变则为 T_MAX）
        xts, yts  : 路径记录（稀疏）
    """
    sig2 = sigma ** 2
    xt, yt = XT0, YT0
    t = 0.0

    xts, yts = [xt], [yt]
    record_dt   = T_MAX / 200
    last_record = 0.0
    jump_time   = T_MAX

    while t < T_MAX:
        # ── 慢变量 SSA 步骤 ──────────────────────────────────────────────
        mu_y = 1.0
        m_y  = max(sig2 - abs(mu_y) * h_y, 0.0) / 2.0
        q_up = mu_y / h_y + m_y / h_y ** 2
        q_dn = m_y / h_y ** 2
        lam  = q_up + q_dn

        if lam < 1e-14:
            break

        # Gillespie 等待时间
        tau = -np.log(np.random.random()) / lam
        if t + tau > T_MAX:
            tau = T_MAX - t

        # ── 快变量 EM 积分（在 tau 时间内） ──────────────────────────────
        n_fast = max(int(tau / dt_fast), 1)
        dt_sub = tau / n_fast
        sqrt_sub = np.sqrt(dt_sub)

        for _ in range(n_fast):
            # x 方向无噪声（噪声只在 y 上）
            xt += (-yt - xt**2) / epsilon * dt_sub
            xt = np.clip(xt, -5.0, 5.0)

        t += tau

        # 慢变量跳跃
        r = np.random.random() * lam
        if r < q_up:
            yt += h_y
        else:
            yt -= h_y

        # 边界
        if yt > 1.0:
            yt = 1.0
        if yt < YT0 - 1.0:
            yt = YT0

        # 稀疏记录
        if t - last_record >= record_dt:
            xts.append(xt)
            yts.append(yt)
            last_record = t

        # 检测提前跳变
        if xt > X_TH and yt < Y_TH:
            jump_time = t
            break

    return jump_time, xts, yts


def run_ssa(
    epsilon: float,
    sigma: float,
    n_paths: int = 500,
    h_y: float = 0.05,
    dt_fast: float = None,
) -> dict:
    """运行 SSA+EM 混合方法"""
    if dt_fast is None:
        dt_fast = epsilon / 200.0

    t_start    = time.perf_counter()
    jump_times = np.empty(n_paths)
    sample_xts, sample_yts = [], []

    for i in range(n_paths):
        jt, xts, yts = ssa_em_single(epsilon, sigma, h_y, dt_fast)
        jump_times[i] = jt
        if i < 8:
            sample_xts.append(xts)
            sample_yts.append(yts)

    elapsed   = time.perf_counter() - t_start
    jumped    = jump_times < T_MAX
    jump_prob = float(jumped.mean())
    mean_jt   = float(jump_times[jumped].mean()) if jumped.any() else np.nan

    return {
        "jump_prob":   jump_prob,
        "mean_jt":     mean_jt,
        "jump_times":  jump_times,
        "elapsed":     elapsed,
        "sample_xts":  sample_xts,
        "sample_yts":  sample_yts,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Part 3: 扫描实验
# ─────────────────────────────────────────────────────────────────────────────

def experiment_sigma_scan(
    epsilon: float = 0.3,
    n_sigma: int = 12,
    n_paths: int = 500,
    h_y: float = 0.05,
) -> dict:
    """固定 epsilon，扫描 sigma，比较 SSA 和 EM 的跳变概率"""
    sigma_c    = epsilon ** (1.0 / 3.0)
    sigma_list = np.logspace(
        np.log10(sigma_c * 0.03),
        np.log10(sigma_c * 6.0),
        n_sigma
    )

    print(f"\n{'='*65}")
    print(f"epsilon={epsilon:.3f},  sigma_c=epsilon^(1/3)={sigma_c:.4f}")
    print(f"初始点: ({XT0}, {YT0})")
    print(f"跳变条件: x_tilde > {X_TH}  AND  y_tilde < {Y_TH}")
    print(f"{'='*65}")

    ssa_probs, em_probs = [], []
    ssa_mjt,   em_mjt   = [], []
    ssa_times, em_times = [], []

    for sigma in sigma_list:
        ratio = sigma / sigma_c
        print(f"\n--- sigma={sigma:.4f}  (sigma/sigma_c={ratio:.2f}) ---")

        res_ssa = run_ssa(epsilon=epsilon, sigma=sigma,
                          n_paths=n_paths, h_y=h_y)
        ssa_probs.append(res_ssa["jump_prob"])
        ssa_mjt.append(res_ssa["mean_jt"])
        ssa_times.append(res_ssa["elapsed"])
        print(f"  SSA: prob={res_ssa['jump_prob']:.3f}, "
              f"mean_t={res_ssa['mean_jt']:.4f}, "
              f"time={res_ssa['elapsed']:.1f}s")

        res_em = run_em(epsilon=epsilon, sigma=sigma, n_paths=n_paths)
        em_probs.append(res_em["jump_prob"])
        em_mjt.append(res_em["mean_jt"])
        em_times.append(res_em["elapsed"])
        print(f"  EM : prob={res_em['jump_prob']:.3f}, "
              f"mean_t={res_em['mean_jt']:.4f}, "
              f"time={res_em['elapsed']:.1f}s")

    return {
        "sigma_list": sigma_list,
        "sigma_c":    sigma_c,
        "epsilon":    epsilon,
        "ssa_probs":  np.array(ssa_probs),
        "em_probs":   np.array(em_probs),
        "ssa_mjt":    np.array(ssa_mjt),
        "em_mjt":     np.array(em_mjt),
        "ssa_times":  np.array(ssa_times),
        "em_times":   np.array(em_times),
    }


def experiment_epsilon_scaling(
    epsilon_list: list = None,
    n_sigma: int = 8,
    n_paths: int = 400,
) -> dict:
    """
    扫描 epsilon，验证 sigma_c ~ epsilon^(1/3) 标度律。
    对每个 epsilon，找到跳变概率 = 0.2 的临界 sigma。
    """
    if epsilon_list is None:
        epsilon_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

    sc_theory  = np.array([eps**(1/3) for eps in epsilon_list])
    sc_ssa_arr = []
    sc_em_arr  = []

    def find_critical(probs, sigmas, target=0.20):
        if np.all(probs < target):
            return float(sigmas[-1])
        if np.all(probs > target):
            return float(sigmas[0])
        idx = int(np.searchsorted(probs, target))
        idx = max(1, min(idx, len(probs) - 1))
        s0, s1 = sigmas[idx-1], sigmas[idx]
        p0, p1 = probs[idx-1],  probs[idx]
        if abs(p1 - p0) < 1e-10:
            return float(s0)
        return float(s0 + (target - p0) / (p1 - p0) * (s1 - s0))

    for eps, sc_th in zip(epsilon_list, sc_theory):
        print(f"\n=== epsilon={eps:.3f},  sigma_c_theory={sc_th:.4f} ===")
        sigma_list = np.logspace(
            np.log10(sc_th * 0.02),
            np.log10(sc_th * 8.0),
            n_sigma
        )
        sp_ssa, sp_em = [], []
        for sigma in sigma_list:
            r_ssa = run_ssa(epsilon=eps, sigma=sigma, n_paths=n_paths)
            r_em  = run_em(epsilon=eps, sigma=sigma, n_paths=n_paths)
            sp_ssa.append(r_ssa["jump_prob"])
            sp_em.append(r_em["jump_prob"])
            print(f"  sigma/sc={sigma/sc_th:.2f}: "
                  f"SSA={r_ssa['jump_prob']:.3f}, "
                  f"EM={r_em['jump_prob']:.3f}")

        sp_ssa = np.array(sp_ssa)
        sp_em  = np.array(sp_em)

        sc_s = find_critical(sp_ssa, sigma_list)
        sc_e = find_critical(sp_em,  sigma_list)
        sc_ssa_arr.append(sc_s)
        sc_em_arr.append(sc_e)
        print(f"  -> SSA sigma_c={sc_s:.4f}, "
              f"EM sigma_c={sc_e:.4f}, "
              f"theory={sc_th:.4f}")

    return {
        "epsilon_list":   np.array(epsilon_list),
        "sigma_c_theory": sc_theory,
        "sigma_c_ssa":    np.array(sc_ssa_arr),
        "sigma_c_em":     np.array(sc_em_arr),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Part 4: 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_portrait(epsilon: float, sigma: float, out_dir: Path):
    """画 SSA 和 EM 的相平面路径"""
    sigma_c = epsilon**(1/3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, label, color in zip(axes, ['EM', 'SSA'], ['#1f77b4', '#d62728']):
        # 标注
        ax.axvline(x=X_TH, color='orange', ls='--', lw=1.5,
                   label=f'跳变阈值 $\\tilde{{x}}={X_TH}$')
        ax.axhline(y=Y_TH, color='gray', ls='--', lw=1.2, alpha=0.7,
                   label=f'鞍结点 $\\tilde{{y}}={Y_TH}$')
        ax.plot(XT0, YT0, 'go', ms=10, zorder=5, label='初始点')
        ax.plot(0, 0, 'k*', ms=12, zorder=5, label='鞍结点')

        n_show = 10
        if label == 'EM':
            res = run_em(epsilon=epsilon, sigma=sigma,
                         n_paths=n_show, record_paths=n_show)
            for i in range(n_show):
                ax.plot(res["rec_xt"][i], res["rec_yt"][i],
                        color=color, alpha=0.5, lw=0.9)
        else:
            for _ in range(n_show):
                jt, xts, yts = ssa_em_single(
                    epsilon, sigma, h_y=0.05, dt_fast=epsilon/200
                )
                ax.plot(xts, yts, color=color, alpha=0.5, lw=0.9)

        ax.set_xlim(-1.5, 1.2)
        ax.set_ylim(-0.8, 0.4)
        ax.set_xlabel(r'$\tilde{x}$', fontsize=13)
        ax.set_ylabel(r'$\tilde{y}$', fontsize=13)
        ax.set_title(
            f'{label} — 相平面路径\n'
            fr'$\varepsilon={epsilon}$, $\sigma/\sigma_c={sigma/sigma_c:.2f}$',
            fontsize=11
        )
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        '局部坐标方程 (6.1.20) 的样本路径\n'
        '从鞍结点左下方出发，橙色虚线为提前跳变阈值',
        fontsize=12
    )
    fig.tight_layout()
    p = out_dir / "phase_portrait.png"
    fig.savefig(p, dpi=180)
    print(f"Saved: {p}")
    plt.close()


def plot_sigma_scan(results: dict, out_dir: Path):
    """绘制跳变概率、平均时刻、计算时间的比较图"""
    sigma_list = results["sigma_list"]
    sigma_c    = results["sigma_c"]
    ratio      = sigma_list / sigma_c
    epsilon    = results["epsilon"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 图1：跳变概率
    ax = axes[0]
    ax.semilogx(ratio, results["ssa_probs"], 'o-',
                color='#d62728', lw=2, ms=8,
                label='SSA+EM 混合')
    ax.semilogx(ratio, results["em_probs"],  's--',
                color='#1f77b4', lw=2, ms=8,
                label='EM（Euler-Maruyama）')
    ax.axvline(x=1.0, color='#333', ls=':', lw=2.0,
               label=r'$\sigma=\sigma_c=\varepsilon^{1/3}$')
    ax.set_xlabel(r'$\sigma\,/\,\sigma_c$（对数轴）', fontsize=12)
    ax.set_ylabel(
        r'$P(\tilde{x}>0 \; \mathrm{before} \; \tilde{y}=0)$',
        fontsize=12
    )
    ax.set_title(
        fr'提前跳变概率 vs $\sigma$  ($\varepsilon={epsilon}$)'
        '\n理论：$\\sigma_c=\\varepsilon^{1/3}$ 处开始上升',
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(-0.02, 0.65)

    # 图2：平均跳变时刻
    ax = axes[1]
    vs = np.isfinite(results["ssa_mjt"])
    ve = np.isfinite(results["em_mjt"])
    if vs.any():
        ax.semilogx(ratio[vs], results["ssa_mjt"][vs],
                    'o-', color='#d62728', lw=2, ms=8, label='SSA+EM')
    if ve.any():
        ax.semilogx(ratio[ve], results["em_mjt"][ve],
                    's--', color='#1f77b4', lw=2, ms=8, label='EM')
    ax.axvline(x=1.0, color='#333', ls=':', lw=2.0)
    ax.set_xlabel(r'$\sigma\,/\,\sigma_c$（对数轴）', fontsize=12)
    ax.set_ylabel(r'平均跳变时刻 $\bar{t}$', fontsize=12)
    ax.set_title('平均提前跳变时刻\n噪声越强 → 越早跳', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # 图3：计算时间
    ax = axes[2]
    xp = np.arange(len(sigma_list))
    w  = 0.35
    ax.bar(xp - w/2, results["ssa_times"], w,
           color='#d62728', alpha=0.85, label='SSA+EM')
    ax.bar(xp + w/2, results["em_times"],  w,
           color='#1f77b4', alpha=0.85, label='EM')
    ax.set_xticks(xp)
    ax.set_xticklabels([f'{r:.2f}' for r in ratio], rotation=45, fontsize=8)
    ax.set_xlabel(r'$\sigma\,/\,\sigma_c$', fontsize=12)
    ax.set_ylabel('计算时间 (s)', fontsize=12)
    ax.set_title('计算时间比较', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    p = out_dir / "sigma_scan.png"
    fig.savefig(p, dpi=180)
    print(f"Saved: {p}")
    plt.close()


def plot_epsilon_scaling(results: dict, out_dir: Path):
    """绘制 sigma_c vs epsilon 的标度律图"""
    eps_arr   = results["epsilon_list"]
    sc_theory = results["sigma_c_theory"]
    sc_ssa    = results["sigma_c_ssa"]
    sc_em     = results["sigma_c_em"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(eps_arr, sc_theory, 'k--', lw=2.0,
              label=r'理论：$\sigma_c = \varepsilon^{1/3}$')
    ax.loglog(eps_arr, sc_ssa, 'o-',
              color='#d62728', lw=2, ms=9, label='SSA+EM')
    ax.loglog(eps_arr, sc_em,  's-',
              color='#1f77b4', lw=2, ms=9, label='EM')

    ax.set_xlabel(r'$\varepsilon$', fontsize=13)
    ax.set_ylabel(r'$\sigma_c$', fontsize=13)
    ax.set_title(
        r'标度律验证：$\sigma_c \sim \varepsilon^{1/3}$'
        '\n对数坐标，理论斜率 = 1/3',
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)

    for label, sc_vals, color in [('SSA+EM', sc_ssa, '#d62728'),
                                   ('EM',     sc_em,  '#1f77b4')]:
        valid = np.isfinite(sc_vals) & (sc_vals > 0) & (sc_vals < 10)
        if valid.sum() >= 2:
            slope = np.polyfit(np.log(eps_arr[valid]),
                               np.log(sc_vals[valid]), 1)[0]
            print(f"  {label} 拟合斜率 = {slope:.3f}  (理论 = {1/3:.3f})")

    fig.tight_layout()
    p = out_dir / "epsilon_scaling.png"
    fig.savefig(p, dpi=180)
    print(f"Saved: {p}")
    plt.close()


def print_summary(results: dict):
    sc = results["sigma_c"]
    print("\n" + "="*70)
    print(f"{'sigma':>10} {'σ/σ_c':>7} "
          f"{'SSA prob':>10} {'EM prob':>10} "
          f"{'|diff|':>8} {'SSA(s)':>7} {'EM(s)':>6}")
    print("-"*70)
    for s, sp, ep, st, et in zip(
        results["sigma_list"],
        results["ssa_probs"], results["em_probs"],
        results["ssa_times"], results["em_times"],
    ):
        print(f"{s:10.4f} {s/sc:7.2f} "
              f"{sp:10.3f} {ep:10.3f} "
              f"{abs(sp-ep):8.3f} {st:7.1f} {et:6.1f}")
    print(f"\n理论 sigma_c = {sc:.4f}")
    print(f"SSA+EM 总时间: {results['ssa_times'].sum():.1f}s  |  "
          f"EM 总时间: {results['em_times'].sum():.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).resolve().parent
    epsilon = 0.3
    sigma_c = epsilon ** (1.0/3.0)

    print(f"epsilon={epsilon},  sigma_c={sigma_c:.4f}")
    print(f"初始点: ({XT0}, {YT0})")
    print(f"跳变条件: x_tilde > {X_TH}  AND  y_tilde < {Y_TH}")

    # Step 1: 相平面路径
    print("\nStep 1: 画相平面路径（sigma = sigma_c）")
    plot_phase_portrait(epsilon=epsilon, sigma=sigma_c, out_dir=out_dir)

    # Step 2: 扫描 sigma
    print("\nStep 2: 扫描 sigma，比较跳变概率")
    results = experiment_sigma_scan(
        epsilon=epsilon,
        n_sigma=12,
        n_paths=500,
        h_y=0.05,
    )
    plot_sigma_scan(results, out_dir)
    print_summary(results)

    # Step 3: 标度律验证
    print("\nStep 3: 验证 sigma_c ~ epsilon^(1/3)")
    results_eps = experiment_epsilon_scaling(
        epsilon_list=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        n_sigma=8,
        n_paths=300,
    )
    plot_epsilon_scaling(results_eps, out_dir)


if __name__ == "__main__":
    main()