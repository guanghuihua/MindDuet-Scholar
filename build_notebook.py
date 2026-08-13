"""生成 topology_homology.ipynb —— 同调群的可视化导览。"""
import json

def md(*lines):  return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _src(lines)}
def _src(lines):
    flat = []
    for ln in lines:
        flat.extend(ln.split("\n"))
    return [l + "\n" for l in flat[:-1]] + [flat[-1]]

cells = []

cells.append(md(
"# 用代码看见同调群 $H_n$",
"",
"> 配合 **尤承业《基础拓扑学》第三章（单纯同调）**。",
"",
"光看书时，$H_n$ 的定义 $H_n = Z_n / B_n = \\ker\\partial_n / \\operatorname{im}\\partial_{n+1}$ "
"很抽象。这个 Notebook 的思路是：**每个概念都既看图、又用代码算出来**。",
"",
"一句话直觉 —— 同调群在数「洞」：",
"",
"| 群 | 几何意义 | 例子 |",
"|---|---|---|",
"| $H_0$ | 连通分支的个数 | 两个分开的圆 $\\Rightarrow H_0=\\mathbb Z^2$ |",
"| $H_1$ | 一维「洞」（套不掉的环）的个数 | 圆周、环面 |",
"| $H_2$ | 二维「空腔」（包住的空心）的个数 | 球面、环面 |",
"| 挠率 torsion | 「扭一圈才回来」的洞 | $\\mathbb{RP}^2$ 的 $\\mathbb Z/2$ |",
"",
"主线：$C_n \\xrightarrow{\\ \\partial_n\\ } C_{n-1}$，闭链 $Z_n=\\ker\\partial_n$，"
"边缘链 $B_n=\\operatorname{im}\\partial_{n+1}$，再取商 $H_n=Z_n/B_n$。",
))

cells.append(code(
"import numpy as np",
"import matplotlib.pyplot as plt",
"import simplicial as S        # 单纯复形 + 同调（手写 Smith 标准形）",
"import viz                    # 画图",
"%matplotlib inline",
"print('就绪。目录里的标准空间：')",
"for name in S.CATALOG: print('  -', name)",
))

# ---- §1 单纯复形 ----
cells.append(md(
"## 1. 单纯复形：拓扑空间的「积木」",
"",
"一个 $n$-单形就是 $n+1$ 个顶点张成的最小凸块：0-单形=点，1-单形=线段，2-单形=三角形面，"
"3-单形=四面体。把它们沿着面粘起来就得到**单纯复形**。",
"",
"代码里一个 $n$-单形 = 一个升序顶点元组，例如 `(0,1,2)` 是一个三角形面。"
"给出「最大单形」即可，所有低维面会自动补全。",
))
cells.append(code(
"# 一个实心三角形：唯一的最大单形是 2-单形 (0,1,2)",
"K = S.SimplicialComplex([(0,1,2)])",
"for n in range(K.dim+1):",
"    print(f'{n}-单形 ({K.chain_rank(n)} 个):', K.simplices[n])",
"",
"coords = {0:(0,0), 1:(1,0), 2:(0.5,0.87)}",
"viz.draw_complex(K, coords, title='disk = solid 2-simplex (0,1,2)')",
"plt.show()",
))

# ---- §2 链群与边缘算子 ----
cells.append(md(
"## 2. 链群 $C_n$ 与边缘算子 $\\partial$",
"",
"$n$-**链群** $C_n$ 是所有 $n$-单形的整系数形式和，是自由 Abel 群 $\\mathbb Z^{(\\#\\,n\\text{-单形})}$。",
"",
"**边缘算子** $\\partial_n:C_n\\to C_{n-1}$ 把一个单形送到它的边界（带定向的正负号）：",
"$$\\partial[v_0v_1\\cdots v_n]=\\sum_i(-1)^i[v_0\\cdots\\widehat{v_i}\\cdots v_n].$$",
"例如 $\\partial[0,1,2]=[1,2]-[0,2]+[0,1]$ —— 正是绕三角形一圈的有向边界。",
"",
"代码把 $\\partial_n$ 表示成整数矩阵：**行 = 低维面，列 = 高维单形**。",
))
cells.append(code(
"d2 = K.boundary_matrix(2)   # ∂₂: C₂ -> C₁",
"d1 = K.boundary_matrix(1)   # ∂₁: C₁ -> C₀",
"print('1-单形（行）:', K.simplices[1])",
"print('2-单形（列）:', K.simplices[2])",
"print('∂₂ =\\n', d2)",
"print('\\n关键恒等式 ∂∘∂ = 0 （边界没有边界）：')",
"print('∂₁ ∂₂ =', (d1 @ d2).ravel())",
))
cells.append(md(
"$\\partial_1\\partial_2=0$ 不是巧合，而是同调论的**基石**：它保证 "
"$\\operatorname{im}\\partial_{n+1}\\subseteq\\ker\\partial_n$，即「边缘链一定是闭链」，"
"取商 $H_n=\\ker\\partial_n/\\operatorname{im}\\partial_{n+1}$ 才有意义。",
))

# ---- §3 闭链 / 边缘链 / H_1 ----
cells.append(md(
"## 3. 闭链、边缘链与 $H_1$：圆周为什么有个洞",
"",
"- **闭链** $Z_n=\\ker\\partial_n$：边界为零的链（首尾相接的圈）。",
"- **边缘链** $B_n=\\operatorname{im}\\partial_{n+1}$：恰好是某个高维块的边界的链。",
"- $H_n=Z_n/B_n$：把「能被填掉的圈」看成 0 之后，**剩下的本质的圈**。",
"",
"对比两个空间：实心三角形（圆盘 $D^2$）vs 空心三角形（圆周 $S^1$）。"
"两者的 1-闭链都是那条绕一圈的边 $[0,1]+[1,2]-[0,2]$；"
"区别在于：圆盘里它 = $\\partial[0,1,2]$ 是边缘链（被面填掉 $\\Rightarrow H_1=0$），"
"而圆周里没有这个面，圈填不掉 $\\Rightarrow H_1=\\mathbb Z$。",
))
cells.append(code(
"disk   = S.disk()      # 实心三角形",
"circle = S.circle()    # 空心三角形 = S¹",
"loop = [(0,1),(1,2),(0,2)]   # 绕一圈的那条 1-链（高亮成红色）",
"",
"fig, axes = plt.subplots(1, 2, figsize=(9,4.3))",
"viz.draw_complex(disk,   coords, ax=axes[0], highlight=loop,",
"                 title='D²: loop is filled  ->  H1 = 0')",
"viz.draw_complex(circle, coords, ax=axes[1], highlight=loop,",
"                 title='S1: loop survives  ->  H1 = Z')",
"plt.show()",
"",
"S.describe(disk,   'D² 圆盘')",
"S.describe(circle, 'S¹ 圆周')",
))

# ---- §4 Smith 标准形与挠率 ----
cells.append(md(
"## 4. Smith 标准形：挠率（$\\mathbb Z/2$ 这种「扭洞」）从哪来",
"",
"要从整数矩阵 $\\partial$ 读出商群 $Z_n/B_n$ 的结构，需要 **Smith 标准形**："
"任何整数矩阵都能经可逆整数行列变换化成对角阵 $\\mathrm{diag}(d_1,d_2,\\dots)$，"
"且 $d_1\\mid d_2\\mid\\cdots$。",
"",
"- 非零 $d_i$ 的**个数** = 矩阵的秩 → 决定 Betti 数（自由部分）；",
"- 其中 $d_i>1$ 的，每个贡献一个 $\\mathbb Z/d_i$ → **挠率**。",
"",
"最经典的例子是射影平面 $\\mathbb{RP}^2$：$\\partial_2$ 的 Smith 形里出现一个 $2$，"
"于是 $H_1(\\mathbb{RP}^2)=\\mathbb Z/2$ —— 「绕两圈才成为边界」的扭洞，**没法画进三维欧氏空间**，"
"只能在代码/抽象里看见。",
))
cells.append(code(
"rp2 = S.projective_plane()",
"d2 = rp2.boundary_matrix(2)",
"print('∂₂ 形状:', d2.shape, ' （15 条边 × 10 个三角面）')",
"print('∂₂ 的 Smith 标准形对角元:', S.smith_normal_form(d2))",
"print('  -> 出现一个 2，即挠率 Z/2\\n')",
"S.describe(rp2, 'ℝP² 射影平面')",
))

# ---- §5 经典曲面巡礼 ----
cells.append(md(
"## 5. 经典曲面巡礼：正方形粘边 → 同调群",
"",
"书里商空间那一章的核心图：一个正方形，把对边按不同方式粘起来得到不同曲面。"
"**同向同字母**的箭头表示按方向粘合。下面把「粘法」和「算出的同调群」并排看。",
))
cells.append(code(
"fig, axes = plt.subplots(1, 3, figsize=(12, 4))",
"for ax, kind in zip(axes, ['torus', 'klein', 'rp2']):",
"    viz.draw_gluing_square(kind, ax=ax)",
"plt.tight_layout(); plt.show()",
))
cells.append(code(
"# 把目录里 6 个空间的同调群全部算一遍",
"for name, (build, _) in S.CATALOG.items():",
"    S.describe(build(), name)",
"    print()",
))
cells.append(md(
"**怎么读这张总表**（注意 Euler 示性数 $\\chi=\\beta_0-\\beta_1+\\beta_2$）：",
"",
"| 空间 | $H_0$ | $H_1$ | $H_2$ | 读法 |",
"|---|---|---|---|---|",
"| $S^1$ | $\\mathbb Z$ | $\\mathbb Z$ | – | 连通，1 个圈 |",
"| $S^2$ | $\\mathbb Z$ | $0$ | $\\mathbb Z$ | 连通，无圈，包住 1 个空腔 |",
"| $T^2$ | $\\mathbb Z$ | $\\mathbb Z^2$ | $\\mathbb Z$ | 2 个独立圈 + 1 个空腔 |",
"| 克莱因瓶 | $\\mathbb Z$ | $\\mathbb Z\\oplus\\mathbb Z/2$ | $0$ | 1 个普通圈 + 1 个扭洞，不可定向 |",
"| $\\mathbb{RP}^2$ | $\\mathbb Z$ | $\\mathbb Z/2$ | $0$ | 只有 1 个扭洞，不可定向 |",
"",
"$H_2=0$（克莱因瓶、$\\mathbb{RP}^2$）正对应**不可定向** —— 没有整体一致的「内外」，"
"故而不存在基本的 2-闭链。这也是为什么它们无法无自交地嵌入 $\\mathbb R^3$。",
))
cells.append(code(
"# 环面的两个生成圈 / 球面的空腔（3D 直觉图）",
"fig = plt.figure(figsize=(10,4))",
"ax1 = fig.add_subplot(121, projection='3d'); viz.draw_torus_3d(ax1)",
"ax2 = fig.add_subplot(122, projection='3d'); viz.draw_sphere_3d(ax2)",
"plt.show()",
))

# ---- §6 动手 ----
cells.append(md(
"## 6. 自己动手",
"",
"1. **改剖分、看不变性**：`S.torus(4,5)` 用更细的网格，同调群应当不变（同调是拓扑不变量）。",
"2. **两个圆的并**：`S.SimplicialComplex([(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)])`，"
"   验证 $H_0=\\mathbb Z^2,\\ H_1=\\mathbb Z^2$。",
"3. **挖一个 8 字形**：两个三角形共享一个顶点，看 $H_1$。",
"4. **自己拼一个曲面**：给 `SimplicialComplex` 一组三角形面，让 `S.describe` 告诉你它是什么。",
"",
"下面留个空白格子，随便试：",
))
cells.append(code(
"# 例：更细的环面剖分，同调群不变",
"S.describe(S.torus(4, 5), 'T² (4x5 网格)')",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open("topology_homology.ipynb", "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"已写出 topology_homology.ipynb，共 {len(cells)} 个 cell")
