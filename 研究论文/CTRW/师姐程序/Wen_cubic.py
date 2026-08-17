from numba import njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 8





@njit(fastmath=True)
def eqn321Fun(xx):
    return - pow(xx, 3)

@njit(fastmath=True)
def SSA(N, h, valid_number, low, Sp):
    data = np.zeros(N)
    X0 = low + Sp * np.random.random()  # 使用 numpy 的随机数生成器
    Sample_h = low + h / 2.0 + round((X0 - low - h / 2.0) / h) * h
    n = 0
    time = 0.0

    # 新增列表用于记录轨迹
    trajectory = np.zeros(N)
    time_steps = np.zeros(N)

    while n < valid_number:
        q = np.zeros(2)
        tmp = eqn321Fun(Sample_h)
        M = 0.5 * max(2-abs(tmp) * h, 0.0)

        q[0] = max(tmp, 0.0) / h + M / (h * h)
        q[1] = -min(tmp, 0.0) / h + M / (h * h)

        Lambda = q[0] + q[1]
        rnd1 = np.random.random()  # 使用 numpy 随机数生成
        rnd2 = np.random.random()

        tau = -math.log(1 - rnd1) / Lambda
        time += tau

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

        x_n = round((Sample_h - low - h / 2.0) / h)
        if x_n >= 0 and x_n < N:
            data[x_n] += 1
            n += 1
        else:
            raise ValueError("x_n out of scope")
    data /= n * h
    return data

@njit(fastmath=True)
def l1_norm(u, h):
    accumulate = 0.0
    for ii in range(len(u)):
        accumulate += abs(u[ii]) * h
    return accumulate

@njit(fastmath=True, parallel=True)
def KK():
    local_N = 100000000
    upper_bound = 100.0
    lower_bound = 0.0
    local_h = (upper_bound - lower_bound) / local_N
    local_K = 0.0
    for ii in prange(local_N):  # 并行化此处循环
        local_K += math.exp(-0.25 * pow((ii + 0.5) * local_h, 4))
    return 2 * local_K * local_h

# 计算参数
@njit(fastmath=True, parallel=True)
def my_fun(loop, N, h, valid_number, low, Sp):
    data = np.zeros((loop, N))
    for ii in prange(loop):  # 并行化外层循环
        data[ii, :] = SSA(N, h, valid_number, low, Sp)
    return data

# 主程序
if __name__ == "__main__":
    start = time.time()

    N = 120
    # loop = 1000
    loop = 5
    valid_number = int(1e7)

    low = -3.0
    Sp = 6
    h = Sp / N

    Z = KK()
    data1 = np.zeros(N)
    for i in range(N):
        x = low + h / 2.0 + i * h
        data1[i] = 1 / Z * math.exp(-0.25 * pow(x, 4))

    data_matrix = my_fun(loop, N, h, valid_number, low, Sp)
    output_data = np.sum(data_matrix, axis=0)
    Final_data = output_data / loop

    data2 = Final_data - data1
    print("l1 norm of SSA_Qu is %f " % l1_norm(data2, h))
    print("h is %f " % h)

    end = time.time()
    print("Total time is (with compilation) = %s" % (end - start))

    plt.plot(Final_data, label="Qu distribution")
    plt.plot(data1, label="True distribution")
    plt.legend()
    plt.show()

