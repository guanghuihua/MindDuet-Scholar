from numba import njit, prange
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 8


# ====================== 漂移函数 ======================
@njit(fastmath=True)
def eqn321Fun(xx):
    """
    漂移项 mu(x) = -x^3
    """
    return -xx**3


# ====================== SSA / CTRW 模拟 ======================
@njit(fastmath=True)
def SSA(N, h, valid_number, low, Sp):
    """
    空间离散 CTRW / SSA 方法，模拟一维立方振子在网格上的平稳分布。
    返回的是区间 [low, low+Sp] 上 N 个网格单元对应的“概率密度近似”。
    """
    data = np.zeros(N)
    # 初始点：在区间 [low, low+Sp] 上均匀
    X0 = low + Sp * np.random.random()
    # 对齐到最近的网格中心
    Sample_h = low + h / 2.0 + round((X0 - low - h / 2.0) / h) * h

    n = 0
    time_now = 0.0

    while n < valid_number:
        q = np.zeros(2)
        tmp = eqn321Fun(Sample_h)
        # M 是补偿项，保证跳率非负
        M = 0.5 * max(2.0 - abs(tmp) * h, 0.0)

        # 右跳 / 左跳的跳率
        q[0] = max(tmp, 0.0) / h + M / (h * h)   # 向右
        q[1] = -min(tmp, 0.0) / h + M / (h * h)  # 向左

        Lambda = q[0] + q[1]
        rnd1 = np.random.random()
        rnd2 = np.random.random()

        # 指数等待时间
        tau = -math.log(1.0 - rnd1) / Lambda
        time_now += tau

        # 选择跳向左还是向右
        mu_number = 0
        amu = q[mu_number]
        while amu < rnd2 * Lambda:
            mu_number += 1
            amu += q[mu_number]

        if mu_number == 0:
            Sample_h = Sample_h + h
        elif mu_number == 1:
            Sample_h = Sample_h - h
        else:
            raise ValueError("mu_number is wrong")

        # 将位置映射回网格索引 i = 0,...,N-1
        x_n = round((Sample_h - low - h / 2.0) / h)
        if 0 <= x_n < N:
            data[x_n] += 1
            n += 1
        else:
            # 如果跳出 [low, low+Sp]，说明 Sp 选得不够大
            raise ValueError("x_n out of scope")

    # 归一化：计数 -> 概率密度近似 rho_SSA(x_i)
    # 总质量 ≈ (n * h)，所以除以 (n*h) 得到密度
    data /= (n * h)
    return data


# ====================== L1 范数 ======================
@njit(fastmath=True)
def l1_norm(u, h):
    """
    计算离散向量 u 在网格步长 h 下的 L1 范数：
        ||u||_{L1} ≈ sum_i |u_i| * h
    在这里 u = rho_num - rho_true。
    """
    accumulate = 0.0
    for ii in range(len(u)):
        accumulate += abs(u[ii]) * h
    return accumulate


# ====================== 精确平稳分布的归一化常数 ======================
@njit(fastmath=True, parallel=True)
def KK():
    """
    数值近似归一化常数：
        Z = ∫_R exp(-x^4/4) dx
      ≈ 2 * ∫_0^{100} exp(-x^4/4) dx
    和你前面 EM 代码里的 KK() 一致。
    """
    local_N = 100000000
    upper_bound = 100.0
    lower_bound = 0.0
    local_h = (upper_bound - lower_bound) / local_N
    local_K = 0.0
    for ii in prange(local_N):
        x = (ii + 0.5) * local_h
        local_K += math.exp(-0.25 * x**4)
    return 2.0 * local_K * local_h


# ====================== 多次循环平均 ======================
@njit(fastmath=True, parallel=True)
def my_fun(loop, N, h, valid_number, low, Sp):
    """
    重复做 loop 次 SSA 模拟，每次得到一条数值平稳密度，
    返回 loop × N 的矩阵。
    """
    data = np.zeros((loop, N))
    for ii in prange(loop):
        data[ii, :] = SSA(N, h, valid_number, low, Sp)
    return data


# ====================== 主程序 ======================
if __name__ == "__main__":
    start = time.time()

    # 网格参数：与 EM 代码保持一致
    N = 120
    low = -3.0
    Sp = 6.0
    h = Sp / N   # = 0.05

    # 重复次数和样本数
    loop = 5
    valid_number = int(1e7)

    # 计算精确平稳密度 nu(x) = Z^{-1} exp(-x^4/4) 在格点上的值
    Z = KK()
    xs = low + h / 2.0 + np.arange(N) * h   # 网格中心点：[-2.975, 2.975]
    data_true = np.exp(-0.25 * xs**4) / Z   # 精确平稳密度

    # SSA 数值平稳密度，做 loop 次模拟取平均
    data_matrix = my_fun(loop, N, h, valid_number, low, Sp)
    output_data = np.sum(data_matrix, axis=0)
    rho_SSA = output_data / loop            # SSA 平稳密度近似

    # 计算 L1 误差： ||rho_SSA - nu||_{L1}
    diff = rho_SSA - data_true
    l1_err = l1_norm(diff, h)
    print("l1 norm of SSA_Qu (sim vs exact) = %f" % l1_err)
    print("h = %f" % h)

    end = time.time()
    print("Total time (with compilation) = %s s" % (end - start))

    # ====================== 绘图：与 EM 代码保持一致 ======================
    # 和前面 EM 代码一样，用物理坐标 xs 作为横坐标
    plt.figure(figsize=(8, 5))
    plt.plot(xs, data_true, label="True stationary density", linewidth=2)
    plt.plot(xs, rho_SSA, "--", label="SSA/CTRW stationary density", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.xlim(low, low + Sp)
    plt.legend()
    plt.title("Cubic oscillator: stationary density\n(True vs SSA/CTRW)")
    plt.tight_layout()
    plt.show()
