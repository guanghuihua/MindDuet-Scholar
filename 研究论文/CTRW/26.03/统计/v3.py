"""
随机 Canard 系统的 SSA 平稳密度模拟
非均匀网格：x 方向（快变量，无噪声）n_x = n_y^2，y 方向（慢变量，有噪声）n_y

系统方程：
    epsilon * dx/dt = y - x^3/3 + x        （快变量，无噪声）
    dy/dt = (a - x) dt + sigma * dW_t       （慢变量，有噪声）

网格设计思路（借鉴 Langevin 代码中 n_x = n_v^2）：
    ┌─────────────────────────────────────────────────────────────┐
    │ Langevin：x无噪声，v有噪声 → n_x = n_v^2                    │
    │ Canard：  x无噪声（快），y有噪声（慢） → n_x = n_y^2        │
    └─────────────────────────────────────────────────────────────┘

    原因：x 方向无物理扩散，迎风格式的数值扩散 ~ |mu_x|*h_x/2
    在 Canard 中 |mu_x| ~ O(1/eps)，远大于 Langevin 中 |v| ~ O(1)
    要控制数值扩散，需要 h_x 比 h_y 小，n_x = n_y^2 是合理折中

速率（Chang-Cooper 混合格式）：
    mu_x = (y - x^3/3 + x) / eps    （x方向漂移，O(1/eps)）
    mu_y = a - x                     （y方向漂移，O(1)）
    m_y  = max(sigma^2 - |mu_y|*h_y, 0) / 2   （仅y方向有混合扩散项）
    m_x  = 0                         （x方向无扩散）

    q_x+ = max(mu_x, 0) / h_x
    q_x- = max(-mu_x, 0) / h_x
    q_y+ = max(mu_y, 0) / h_y + m_y / h_y^2
    q_y- = max(-mu_y, 0) / h_y + m_y / h_y^2

参数（与原始 SSA Canard 代码保持一致）：
    delta（epsilon）= 0.1
    sigma^2 = 2.0
    a = 1 - delta/8 - 3*delta^2/32 - 173*delta^3/1024 - 0.01
"""

from __future__ import annotations
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  系统参数（与原始 SSA Canard 代码完全一致）
# ─────────────────────────────────────────────────────────────────────────────
DELTA    = 0.1
SIGMA_SQ = 2.0
A_PARAM  = 1 - DELTA/8 - 3*DELTA**2/32 - 173*DELTA**3/1024 - 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  核心 SSA 函数（numba JIT 加速）
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(fastmath=True, cache=True)
def ssa_canard_nonuniform(
    lowx: float,
    lowy: float,
    span: float,
    n_y: int,        # 慢变量（有噪声）方向：n_y 格点
    n_x: int,        # 快变量（无噪声）方向：n_x = n_y^2 格点
    eps: float,
    sigma_sq: float,
    a: float,
    sample_size: int,
) -> tuple:
    """
    单条链的时间加权 SSA，返回 (counts[n_x, n_y], oob_count)。
    """
    h_x = span / n_x
    h_y = span / n_y

    counts = np.zeros((n_x, n_y), dtype=np.float64)

    # 随机初始点
    x = lowx + np.random.random() * span
    y = lowy + np.random.random() * span

    valid_count = 0
    oob_count   = 0

    while valid_count < sample_size:

        # ── 漂移 ──────────────────────────────────────────────────────────
        mu_x = (y - x * x * x / 3.0 + x) / eps   # O(1/eps)
        mu_y = a - x                                # O(1)

        # ── 混合项：仅 y 方向（有噪声） ───────────────────────────────────
        m_y = 0.5 * max(sigma_sq - abs(mu_y) * h_y, 0.0)
        # x 方向无噪声 → m_x = 0（无混合项）

        # ── 四方向跳跃速率 ────────────────────────────────────────────────
        q_xp = max(mu_x,  0.0) / h_x                      # x 向右
        q_xm = max(-mu_x, 0.0) / h_x                      # x 向左
        q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y) # y 向上
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y) # y 向下
        lam  = q_xp + q_xm + q_yp + q_ym

        # ── Gillespie 等待时间 ────────────────────────────────────────────
        tau = -np.log(1.0 - np.random.random()) / lam

        # ── 时间加权统计 ──────────────────────────────────────────────────
        ix = int((x - lowx) / h_x)
        iy = int((y - lowy) / h_y)

        if 0 <= ix < n_x and 0 <= iy < n_y:
            counts[ix, iy] += tau
            valid_count += 1
        else:
            oob_count += 1
            # 边界反弹
            if ix < 0:      x = lowx + h_x * 0.5
            elif ix >= n_x: x = lowx + span - h_x * 0.5
            if iy < 0:      y = lowy + h_y * 0.5
            elif iy >= n_y: y = lowy + span - h_y * 0.5

        # ── 选择跳跃方向 ──────────────────────────────────────────────────
        r = np.random.random() * lam
        if r < q_xp:
            x += h_x
        elif r < q_xp + q_xm:
            x -= h_x
        elif r < q_xp + q_xm + q_yp:
            y += h_y
        else:
            y -= h_y

    return counts, oob_count


@nb.njit(fastmath=True, parallel=True, cache=True)
def run_ensemble_nonuniform(
    lowx: float,
    lowy: float,
    span: float,
    n_y: int,
    n_x: int,
    eps: float,
    sigma_sq: float,
    a: float,
    sample_size: int,
    loops: int,
    bin_factor_x: int,
    bin_factor_y: int,
) -> tuple:
    """
    并行集成 loops 条独立链，结果取平均后做网格粗化输出。
    """
    flat_size   = n_x * n_y
    density_sum = np.zeros(flat_size, dtype=np.float64)
    oob_total   = 0

    for i in nb.prange(loops):
        c, oob = ssa_canard_nonuniform(
            lowx, lowy, span, n_y, n_x, eps, sigma_sq, a, sample_size
        )
        density_sum += c.ravel()
        oob_total   += oob

    density_mean = density_sum / loops

    # 网格粗化（bin）
    if bin_factor_x > 1 or bin_factor_y > 1:
        out_nx = n_x // bin_factor_x
        out_ny = n_y // bin_factor_y
        fine   = density_mean.reshape((n_x, n_y))
        coarse = np.zeros((out_nx, out_ny), dtype=np.float64)
        for ix in range(out_nx):
            for iy in range(out_ny):
                s = 0.0
                for di in range(bin_factor_x):
                    for dj in range(bin_factor_y):
                        s += fine[ix * bin_factor_x + di,
                                  iy * bin_factor_y + dj]
                coarse[ix, iy] = s
        return coarse.ravel(), oob_total

    return density_mean, oob_total


# ─────────────────────────────────────────────────────────────────────────────
#  对比实验：均匀网格 vs 非均匀网格
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    lowx=-3.0, lowy=-3.0, span=6.0,
    n_y=60, sample_size=int(3e6), loops=5,
    out_n=100,
):
    """
    同等计算量下，比较均匀网格和非均匀网格（n_x=n_y^2）的结果。
    """
    eps      = DELTA
    sigma_sq = SIGMA_SQ
    a        = A_PARAM
    n_x_nu   = n_y ** 2   # 非均匀：n_x = n_y^2

    # 均匀网格：n_x = n_y（原始方案）
    n_x_uni = n_y

    results = {}

    for label, n_x in [("uniform",     n_x_uni),
                        ("nonuniform",  n_x_nu)]:

        # 确保能整除
        bx = max(n_x // out_n, 1)
        by = max(n_y // out_n, 1)
        nx = out_n * bx
        ny = out_n * by

        h_x = span / nx
        h_y = span / ny

        print(f"\n{'='*55}")
        print(f"{label}: n_x={nx}, n_y={ny}  "
              f"(h_x={h_x:.5f}, h_y={h_y:.4f})")
        print(f"  数值扩散 ~ |mu_x|*h_x/2 = {1/eps * h_x / 2:.5f}")

        # 预热
        run_ensemble_nonuniform(
            lowx, lowy, span, ny, nx, eps, sigma_sq, a,
            100, 1, bx, by
        )

        t0 = time.perf_counter()
        data_flat, oob = run_ensemble_nonuniform(
            lowx, lowy, span, ny, nx, eps, sigma_sq, a,
            sample_size, loops, bx, by
        )
        elapsed = time.perf_counter() - t0

        # 归一化
        h_eff = span / out_n
        data  = data_flat.reshape((out_n, out_n))
        total = data.sum()
        if total > 0:
            data /= (h_eff * h_eff * total)

        print(f"  时间: {elapsed:.1f}s,  越界: {oob},  "
              f"密度积分: {data.sum()*h_eff*h_eff:.6f}")
        results[label] = data

    return results, out_n, span, lowx, lowy


# ─────────────────────────────────────────────────────────────────────────────
#  绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results, out_n, span, lowx, lowy, out_dir):
    h = span / out_n
    x_arr = np.linspace(lowx + h/2, lowx + span - h/2, out_n)
    y_arr = np.linspace(lowy + h/2, lowy + span - h/2, out_n)
    X, Y  = np.meshgrid(x_arr, y_arr, indexing="ij")

    # 慢流形
    x_mf = np.linspace(-2.5, 2.5, 500)
    y_mf = x_mf**3 / 3 - x_mf

    labels    = list(results.keys())
    n_methods = len(labels)
    titles    = {
        "uniform":    f"Uniform grid\n$n_x = n_y$, {DELTA:.0%} scale",
        "nonuniform": f"Non-uniform grid\n$n_x = n_y^2$",
    }

    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 6))
    if n_methods == 1:
        axes = [axes]

    vmax = max(d.max() for d in results.values())

    for ax, label in zip(axes, labels):
        data = results[label]
        cf = ax.contourf(X, Y, data, levels=20, cmap="viridis",
                         vmin=0, vmax=vmax)
        plt.colorbar(cf, ax=ax, label="Density")
        ax.plot(x_mf, y_mf, 'r--', lw=1.5, label='Slow manifold')
        ax.axvline(x=1.0,  color='orange', ls=':', lw=1.2, alpha=0.8)
        ax.axvline(x=-1.0, color='orange', ls=':', lw=1.2, alpha=0.8,
                   label='Saddle nodes $x=\\pm1$')
        ax.set_xlim(lowx, lowx+span)
        ax.set_ylim(lowy, lowy+span)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(
            titles.get(label, label) +
            f"\n$\\varepsilon={DELTA}$, "
            f"$\\sigma={np.sqrt(SIGMA_SQ):.3f}$",
            fontsize=11
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.15)

    if n_methods == 2:
        # 差分图
        diff = results["nonuniform"] - results["uniform"]
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        vm = np.abs(diff).max()
        cf2 = ax2.contourf(X, Y, diff, levels=20, cmap="RdBu_r",
                           vmin=-vm, vmax=vm)
        plt.colorbar(cf2, ax=ax2, label="Density difference")
        ax2.plot(x_mf, y_mf, 'k--', lw=1.5, label='Slow manifold')
        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("y", fontsize=12)
        ax2.set_title(
            "Difference: non-uniform − uniform\n"
            r"(shows $x$-direction resolution effect)",
            fontsize=11
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.15)
        fig2.tight_layout()
        p2 = out_dir / "canard_density_diff.png"
        fig2.savefig(p2, dpi=180, bbox_inches='tight')
        print(f"Saved: {p2}")
        plt.close(fig2)

    fig.tight_layout()
    p = out_dir / "canard_nonuniform_density.png"
    fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).resolve().parent

    print("随机 Canard 系统 SSA：均匀网格 vs 非均匀网格（n_x = n_y^2）")
    print(f"系统参数: eps={DELTA:.3f}, sigma={np.sqrt(SIGMA_SQ):.4f}, a={A_PARAM:.6f}")
    print()

    # 打印网格方案对比
    span = 6.0
    n_y  = 60
    print(f"{'方案':20s} {'n_x':>8} {'n_y':>6} {'h_x':>10} {'h_y':>8} "
          f"{'数值扩散x':>12} {'说明'}")
    print("-" * 80)
    for label, nx, ny in [
        ("均匀（原始）",   n_y,     n_y),
        ("非均匀 n_x=n_y^2", n_y**2, n_y),
        ("严格CFL",   int(n_y**2/DELTA), n_y),
    ]:
        hx = span / nx
        hy = span / ny
        num_diff = hx / (2 * DELTA)   # |mu_x_typ| * h_x / 2 ≈ h_x/(2*eps)
        note = "本方案" if nx == n_y**2 else ("太大" if nx > n_y**2 else "原始")
        print(f"{label:20s} {nx:8d} {ny:6d} {hx:10.6f} {hy:8.5f} "
              f"{num_diff:12.5f}  {note}")

    print()

    # 运行比较实验
    results, out_n, span, lowx, lowy = run_comparison(
        n_y=60,
        sample_size=int(3e6),
        loops=5,
        out_n=100,
    )

    # 绘图
    print("\n绘图...")
    plot_results(results, out_n, span, lowx, lowy, out_dir)

    # 输出关键统计
    print("\n关键统计量：")
    for label, data in results.items():
        h = span / out_n
        x_arr = np.linspace(lowx + h/2, lowx + span - h/2, out_n)
        y_arr = np.linspace(lowy + h/2, lowy + span - h/2, out_n)
        # x 方向边缘密度
        d_x = data.sum(axis=1) * h
        ix_peak = np.argmax(d_x)
        # y 方向边缘密度
        d_y = data.sum(axis=0) * h
        iy_peak = np.argmax(d_y)
        print(f"  {label}: x_peak={x_arr[ix_peak]:.3f}, "
              f"y_peak={y_arr[iy_peak]:.3f}, "
              f"density_max={data.max():.4f}")


if __name__ == "__main__":
    main()
