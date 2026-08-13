"""
viz.py — 单纯复形 / 曲面粘合的可视化
配合 simplicial.py 与主 Notebook 使用。所有图内文字用 ASCII/数字，
中文说明放在 Notebook 的 markdown 里（避免 matplotlib 缺中文字体的方块问题）。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow


# ---------------------------------------------------------------------
# 平面单纯复形：画顶点、边、（可选）填充三角形、（可选）高亮一条链
# ---------------------------------------------------------------------

def draw_complex(K, coords, ax=None, fill=True, highlight=None,
                 title="", labels=True):
    """
    K        : SimplicialComplex
    coords   : {顶点号: (x, y)}
    highlight: 要高亮的边的列表，如 [(0,1),(1,2)]（画一条 1-链/闭链）
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    highlight = {tuple(sorted(e)) for e in (highlight or [])}

    # 2-单形：浅色填充
    if fill:
        for tri in K.simplices.get(2, []):
            poly = Polygon([coords[v] for v in tri], closed=True,
                           facecolor="#cfe8ff", edgecolor="none",
                           alpha=0.7, zorder=1)
            ax.add_patch(poly)

    # 1-单形：黑线，高亮的画粗红线
    for e in K.simplices.get(1, []):
        (x0, y0), (x1, y1) = coords[e[0]], coords[e[1]]
        if e in highlight:
            ax.plot([x0, x1], [y0, y1], color="#d62728", lw=3.5, zorder=3)
        else:
            ax.plot([x0, x1], [y0, y1], color="#333333", lw=1.4, zorder=2)

    # 0-单形：点 + 标号
    for v in K.simplices.get(0, []):
        x, y = coords[v[0]]
        ax.plot(x, y, "o", color="#222222", ms=8, zorder=4)
        if labels:
            ax.annotate(str(v[0]), (x, y), textcoords="offset points",
                        xytext=(7, 6), fontsize=12, zorder=5)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    return ax


# ---------------------------------------------------------------------
# 正方形粘合示意图：把商空间那一章和同调连起来
# 同色同向的箭头表示该对边按箭头方向粘合
# ---------------------------------------------------------------------

def draw_gluing_square(kind="torus", ax=None):
    """kind ∈ {'torus','klein','rp2','cylinder','mobius'}"""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))

    # 四条边的箭头方向：+1 表示沿坐标轴正向，-1 表示反向
    # (颜色, 标签)
    A, B = ("#1f77b4", "a"), ("#d62728", "b")
    specs = {
        # 边: (颜色标签, 底边方向, 顶边方向, 左边方向, 右边方向)
        "torus":    (A, B, +1, +1, +1, +1),   # 上下同向、左右同向 → 都直接粘
        "klein":    (A, B, +1, +1, +1, -1),   # 左右反向 → 一个翻转
        "rp2":      (A, B, +1, -1, +1, -1),   # 上下反向且左右反向 → 对径粘
        "cylinder": (A, B, +1, +1, +1, +1),   # 只粘左右（上下不标）
        "mobius":   (A, B, +1, +1, +1, -1),
    }
    colA, colB, bot, top, lft, rgt = specs[kind]
    glue_horizontal = kind not in ("cylinder", "mobius")  # 是否粘上下边

    def arrow(p0, p1, color):
        x0, y0 = p0; x1, y1 = p1
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=3))

    def mid_arrow(side, direction, color, label):
        """在一条边的中点画方向箭头，并标字母。side: 'bot/top/lft/rgt'"""
        pts = {"bot": ((0, 0), (1, 0)), "top": ((0, 1), (1, 1)),
               "lft": ((0, 0), (0, 1)), "rgt": ((1, 0), (1, 1))}
        (x0, y0), (x1, y1) = pts[side]
        if direction < 0:
            (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # 短箭头表示方向
        dx, dy = (x1 - x0), (y1 - y0)
        L = np.hypot(dx, dy); dx, dy = dx / L * 0.18, dy / L * 0.18
        ax.annotate("", xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        off = {"bot": (0, -0.1), "top": (0, 0.1),
               "lft": (-0.1, 0), "rgt": (0.1, 0)}[side]
        ax.text(mx + off[0], my + off[1], label, color=color,
                fontsize=14, ha="center", va="center", fontweight="bold")

    # 正方形四条边（底色）
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#aaaaaa", lw=1)

    # 左右边：字母 a（蓝）
    mid_arrow("lft", lft, colA[0], colA[1])
    mid_arrow("rgt", rgt, colA[0], colA[1])
    # 上下边：字母 b（红），圆柱/莫比乌斯不粘上下，用淡灰
    if glue_horizontal:
        mid_arrow("bot", bot, colB[0], colB[1])
        mid_arrow("top", top, colB[0], colB[1])
    else:
        ax.text(0.5, -0.08, "(open)", color="#999", ha="center", fontsize=10)
        ax.text(0.5, 1.08, "(open)", color="#999", ha="center", fontsize=10)

    titles = {"torus": "Torus  T²   (a,b both straight)",
              "klein": "Klein bottle  (b flipped)",
              "rp2":   "Projective plane  RP²  (both flipped)",
              "cylinder": "Cylinder  (glue a only)",
              "mobius": "Mobius band  (glue a flipped)"}
    ax.set_title(titles[kind], fontsize=11)
    ax.set_xlim(-0.25, 1.25); ax.set_ylim(-0.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")
    return ax


# ---------------------------------------------------------------------
# 3D：球面、环面的参数曲面（直觉用，非三角剖分）
# ---------------------------------------------------------------------

def draw_torus_3d(ax=None, R=2.0, r=0.8):
    if ax is None:
        fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 60); v = np.linspace(0, 2 * np.pi, 30)
    u, v = np.meshgrid(u, v)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    ax.plot_surface(x, y, z, color="#cfe8ff", edgecolor="#5599cc",
                    linewidth=0.2, alpha=0.9)
    # 两个生成圈：H₁(T²)=Z⊕Z 的两个生成元
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot((R + r) * np.cos(t), (R + r) * np.sin(t), 0, color="#d62728", lw=3)
    ax.plot(R + r * np.cos(t), 0 * t, r * np.sin(t), color="#2ca02c", lw=3)
    ax.set_box_aspect((1, 1, 0.4)); ax.set_axis_off()
    ax.set_title("T²:  two independent loops  (H₁ = Z ⊕ Z)", fontsize=10)
    return ax


def draw_sphere_3d(ax=None):
    if ax is None:
        fig = plt.figure(figsize=(4.5, 4)); ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 50); v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#cfe8ff", edgecolor="#5599cc",
                    linewidth=0.15, alpha=0.9)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title("S²:  hollow, no 1-holes  (H₁=0, H₂=Z)", fontsize=10)
    return ax
