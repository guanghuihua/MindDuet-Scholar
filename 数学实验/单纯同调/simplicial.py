"""
simplicial.py — 单纯复形与单纯同调的最小实现
=================================================

配合尤承业《基础拓扑学》第三章（单纯同调）使用。

设计目标：让"链群 Cₙ → 边缘算子 ∂ → 闭链 Zₙ=ker∂ → 边缘链 Bₙ=im∂ → Hₙ=Zₙ/Bₙ"
这条主线的每一步都能用代码看见、算出来，包括挠率（torsion，如 ℝP² 的 ℤ/2）。

所有同调群都在整数 ℤ 上计算，靠的是整数矩阵的 Smith 标准形（手写，透明）。
"""

from itertools import combinations
import numpy as np


# =====================================================================
# 第一部分：抽象单纯复形
# =====================================================================

class SimplicialComplex:
    """
    抽象单纯复形：由一组"最大单形"（顶点元组）生成，自动补全所有面。

    例：三角形的边界 = SimplicialComplex([(0,1),(1,2),(0,2)])
        实心三角形      = SimplicialComplex([(0,1,2)])
    顶点用整数标号。一个 n-单形用长度 n+1 的**升序**元组表示。
    """

    def __init__(self, maximal_faces):
        faces_by_dim = {}
        for face in maximal_faces:
            face = tuple(sorted(set(face)))          # 去掉退化（重复顶点）的情形
            for r in range(1, len(face) + 1):
                for sub in combinations(face, r):    # 所有非空子集都是面
                    faces_by_dim.setdefault(len(sub) - 1, set()).add(sub)

        self.dim = max(faces_by_dim) if faces_by_dim else -1
        # simplices[n] = 第 n 维所有单形的有序列表（固定顺序 = 矩阵的行/列下标）
        self.simplices = {
            n: sorted(faces_by_dim.get(n, set()))
            for n in range(self.dim + 1)
        }

    def chain_rank(self, n):
        """链群 Cₙ 的秩 = n-单形的个数（Cₙ 是自由 Abel 群 ℤ^rank）。"""
        return len(self.simplices.get(n, []))

    def euler(self):
        """Euler 示性数 χ = Σ (-1)^n · (n-单形个数)。"""
        return sum((-1) ** n * self.chain_rank(n) for n in range(self.dim + 1))

    def boundary_matrix(self, n):
        """
        边缘算子 ∂ₙ : Cₙ → Cₙ₋₁ 的整数矩阵。
        形状 = (#(n-1)-单形) × (#n-单形)，即"行=低维面，列=高维单形"。

        带定向的定义：∂[σ] = Σ_i (-1)^i [σ 去掉第 i 个顶点]
        """
        if n <= 0 or n > self.dim:
            rows = self.chain_rank(n - 1) if n - 1 >= 0 else 0
            return np.zeros((rows, self.chain_rank(n)), dtype=np.int64)

        faces_lo = self.simplices[n - 1]
        idx = {f: i for i, f in enumerate(faces_lo)}    # 低维面 -> 行号
        cols = self.simplices[n]
        M = np.zeros((len(faces_lo), len(cols)), dtype=np.int64)
        for j, simplex in enumerate(cols):
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1:]     # 去掉第 i 个顶点
                M[idx[face], j] += (-1) ** i
        return M


# =====================================================================
# 第二部分：整数 Smith 标准形
# =====================================================================
# 任何整数矩阵 A 都能写成 U A V = D = diag(d1, d2, ...)，U,V 行列式 ±1，
# 且 d1 | d2 | ...。这些 dᵢ（初等因子）正是读出挠率的关键。

def smith_normal_form(A):
    """返回对角元列表 [d1, d2, ...]（满足整除链，已去掉 0）。"""
    import math
    M = A.copy().astype(np.int64)
    rows, cols = M.shape
    diag = []
    t = 0

    def pivot(t):
        for i in range(t, rows):
            for j in range(t, cols):
                if M[i, j] != 0:
                    M[[t, i]] = M[[i, t]]              # 换行
                    M[:, [t, j]] = M[:, [j, t]]        # 换列
                    return True
        return False

    while t < min(rows, cols):
        if not pivot(t):
            break
        done = False
        while not done:
            done = True
            for i in range(t + 1, rows):              # 消去 (t,t) 下方
                if M[i, t] != 0:
                    q = M[i, t] // M[t, t]
                    M[i, :] -= q * M[t, :]
                    if M[i, t] != 0:                  # 余数非零 -> 换上来辗转相除
                        M[[t, i]] = M[[i, t]]
                        done = False
            for j in range(t + 1, cols):              # 消去 (t,t) 右方
                if M[t, j] != 0:
                    q = M[t, j] // M[t, t]
                    M[:, j] -= q * M[:, t]
                    if M[t, j] != 0:
                        M[:, [t, j]] = M[:, [j, t]]
                        done = False
        diag.append(abs(int(M[t, t])))
        t += 1

    diag = [d for d in diag if d != 0]
    i = 0                                              # 强制整除链 d1|d2|...
    while i < len(diag) - 1:
        a, b = diag[i], diag[i + 1]
        if b % a != 0:
            g = math.gcd(a, b)
            diag[i], diag[i + 1] = g, a * b // g
            if i > 0:
                i -= 1
                continue
        i += 1
    return diag


def rank_int(A):
    """整数矩阵的秩 = Smith 标准形里非零对角元的个数。"""
    if A.size == 0:
        return 0
    return len(smith_normal_form(A))


# =====================================================================
# 第三部分：同调群 Hₙ = ker∂ₙ / im∂ₙ₊₁
# =====================================================================

def homology(K, n):
    """
    计算 Hₙ(K; ℤ)，返回 (betti, torsion)：
      betti   = 自由部分的秩（Betti 数 βₙ，几何上 = n 维"洞"的个数）
      torsion = 挠系数列表，如 [2] 表示一个 ℤ/2（"扭洞"）

    βₙ      = dim Zₙ − rank∂ₙ₊₁ = (rank Cₙ − rank∂ₙ) − rank∂ₙ₊₁
    torsion = ∂ₙ₊₁ 的 Smith 形里 > 1 的初等因子
    """
    cn = K.chain_rank(n)
    rank_dn = rank_int(K.boundary_matrix(n))           # ∂ₙ:   Cₙ → Cₙ₋₁
    Bnp1 = K.boundary_matrix(n + 1)                    # ∂ₙ₊₁: Cₙ₊₁ → Cₙ
    rank_dn1 = rank_int(Bnp1)

    betti = cn - rank_dn - rank_dn1
    torsion = [d for d in smith_normal_form(Bnp1) if d > 1]
    return betti, torsion


def homology_str(betti, torsion):
    """写成人类可读的群，如 'Z + Z/2' 或 '0'。"""
    parts = ["Z"] * betti + [f"Z/{d}" for d in torsion]
    return " + ".join(parts) if parts else "0"


def describe(K, name=""):
    """打印一个复形从 0 维到顶维的全部同调群。"""
    head = f" {name} " if name else " "
    print(f"=== 同调群{head}===")
    print(f"  dim K = {K.dim},  Euler χ = {K.euler()},  "
          f"单形数 = {[K.chain_rank(n) for n in range(K.dim + 1)]}")
    for n in range(K.dim + 1):
        betti, tors = homology(K, n)
        meaning = {0: "连通分支数", 1: "一维洞（环）", 2: "二维空腔"}.get(n, "")
        tag = f"   ← {meaning}" if meaning else ""
        print(f"  H_{n} = {homology_str(betti, tors):<12}"
              f"(β_{n}={betti}" + (f", 挠率={tors}" if tors else "") + f"){tag}")


# =====================================================================
# 第四部分：常见空间的三角剖分
# =====================================================================

def circle():
    """S¹ ≈ 三角形边界。 期望 H₀=Z, H₁=Z。"""
    return SimplicialComplex([(0, 1), (1, 2), (0, 2)])

def disk():
    """实心三角形（可缩）。 期望 H₀=Z, 其余 0。"""
    return SimplicialComplex([(0, 1, 2)])

def sphere():
    """S² ≈ 四面体边界（4 个三角面）。 期望 H₀=Z, H₁=0, H₂=Z。"""
    return SimplicialComplex([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])


def _grid_surface(m, n, flip_rows=False, flip_cols=False):
    """
    用 m×n 网格 + 粘合生成闭曲面（正方形粘边模型，呼应商空间那一章）。
      flip_cols=False: 左右边直接粘   -> 否则带翻转（莫比乌斯式）
      flip_rows=False: 上下边直接粘   -> 否则带翻转
    两者都 False = 环面 T²；恰一个 True = 克莱因瓶；都 True = 射影平面 ℝP²。

    返回 (复形, 顶点二维坐标字典)。顶点用网格基础块 i∈[0,m), j∈[0,n) 标号 i*n+j。
    """
    def vid(i, j):
        # 列方向（横向，参数 j）粘合
        if j >= n:
            j -= n
            if flip_rows:                 # 横向粘合翻转的是"纵向 i"
                i = (m - 1 - i) % m
        if j < 0:
            j += n
            if flip_rows:
                i = (m - 1 - i) % m
        # 行方向（纵向，参数 i）粘合
        if i >= m:
            i -= m
            if flip_cols:                 # 纵向粘合翻转的是"横向 j"
                j = (n - 1 - j) % n
        if i < 0:
            i += m
            if flip_cols:
                j = (n - 1 - j) % n
        return (i % m) * n + (j % n)

    faces = []
    for i in range(m):
        for j in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
            faces.append((a, b, c))       # 每个方格切成两个三角形
            faces.append((b, d, c))
    K = SimplicialComplex(faces)
    coords = {i * n + j: (j, i) for i in range(m) for j in range(n)}
    return K, coords


def torus(m=3, n=3):
    """环面 T² = m×n 网格上下、左右都直接粘。 期望 H₀=Z, H₁=Z+Z, H₂=Z。"""
    return _grid_surface(m, n)[0]

def klein_bottle(m=4, n=4):
    """克莱因瓶 = 一个方向翻转粘。 期望 H₀=Z, H₁=Z+Z/2, H₂=0。"""
    return _grid_surface(m, n, flip_rows=True)[0]

def projective_plane():
    """
    射影平面 ℝP² = 6 顶点极小三角剖分（icosahedron 的对径商，10 个三角形）。
    期望 H₀=Z, H₁=Z/2, H₂=0。
    """
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
             (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5)]
    return SimplicialComplex(faces)


CATALOG = {
    "S¹  圆周":        (circle,            "Z, Z"),
    "D²  圆盘":        (disk,              "Z, 0, 0"),
    "S²  球面":        (sphere,            "Z, 0, Z"),
    "T²  环面":        (torus,             "Z, Z + Z, Z"),
    "K   克莱因瓶":    (klein_bottle,      "Z, Z + Z/2, 0"),
    "ℝP² 射影平面":    (projective_plane,  "Z, Z/2, 0"),
}


if __name__ == "__main__":
    # 自检：把目录里每个空间都算一遍，和期望值对照
    print("单纯同调自检\n" + "=" * 50)
    for name, (build, expected) in CATALOG.items():
        K = build()
        got = ", ".join(homology_str(*homology(K, n)) for n in range(K.dim + 1))
        ok = "✓" if got == expected else "✗ 期望 " + expected
        print(f"{name:14s} χ={K.euler():+d}  H* = {got:18s} {ok}")
