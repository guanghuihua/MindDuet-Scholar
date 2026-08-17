import numpy as np
import math
import time
import matplotlib.pyplot as plt
from numba import njit, prange
import numba

# Optional: set number of threads for numba, like in Wen_cubic.py
numba.config.NUMBA_DEFAULT_NUM_THREADS = 8

# ------------------------------
# 1. SDE definition and exact stationary density
# ------------------------------

def drift(x):
    """Drift term mu(x) = -x^3."""
    return -x**3


sigma = math.sqrt(2.0)  # diffusion coefficient


@njit(fastmath=True, parallel=True)
def KK():
    """
    Numerical approximation of Z = ∫_R exp(-x^4/4) dx
    using symmetric integration on [0, 100] and doubling.
    This is exactly the same idea as in Wen_cubic.py.
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


def true_stationary_density(low, Sp, N):
    """
    Compute the true stationary density on the grid centers.
    Domain: [low, low + Sp], N cells, grid spacing h = Sp / N.
    """
    h = Sp / N
    xs = low + h / 2.0 + np.arange(N) * h
    Z = KK()
    dens = np.exp(-0.25 * xs**4) / Z
    return xs, dens, h


# ------------------------------
# 2. Histogram from samples
# ------------------------------

def histogram_density(samples, low, Sp, N):
    """
    Convert samples into an empirical density on [low, low+Sp]
    using N bins. Result is normalized to integrate to 1 on this interval.
    """
    h = Sp / N
    bins = np.linspace(low, low + Sp, N + 1)
    counts, _ = np.histogram(samples, bins=bins)
    # normalize to obtain density on [low, low+Sp]
    total_mass = counts.sum() * h
    if total_mass == 0.0:
        raise RuntimeError("No samples fell into the histogram interval.")
    density = counts.astype(np.float64) / total_mass
    centers = low + h / 2.0 + np.arange(N) * h
    return centers, density


def l1_norm(u, v, h):
    """Compute L1 norm of difference of two densities on same grid."""
    return np.sum(np.abs(u - v)) * h


# ------------------------------
# 3. Tamed EM sampler
# ------------------------------

def sample_tamed_em(
    valid_number,
    dt,
    burn_in_steps,
    sample_stride,
    x0=0.0,
    low=-3.0,
    Sp=6.0,
):
    """
    Generate 'valid_number' samples from the tamed EM chain
    (approximately stationary) and return them.

    - First run 'burn_in_steps' steps to reach stationarity.
    - Then collect one sample every 'sample_stride' steps.
    - Only keep samples that fall into [low, low+Sp].
      If you want exactly 'valid_number' usable samples, this
      while-loop is appropriate; it may run a bit longer if
      many samples are outside the interval.
    """
    samples = np.empty(valid_number, dtype=np.float64)
    x = x0

    # burn-in
    for _ in range(burn_in_steps):
        dW = math.sqrt(dt) * np.random.randn()
        mu = drift(x)
        # tamed drift: mu / (1 + dt |mu|)
        x = x + (dt * mu) / (1.0 + dt * abs(mu)) + sigma * dW

    # sampling
    n_valid = 0
    left = low
    right = low + Sp

    while n_valid < valid_number:
        # do 'sample_stride' steps between samples to reduce correlation
        for _ in range(sample_stride):
            dW = math.sqrt(dt) * np.random.randn()
            mu = drift(x)
            x = x + (dt * mu) / (1.0 + dt * abs(mu)) + sigma * dW

        # record if inside the window
        if left <= x <= right:
            samples[n_valid] = x
            n_valid += 1

    return samples


# ------------------------------
# 4. Truncated EM sampler
# ------------------------------

def truncate(x, R):
    """Project x onto [-R, R]."""
    if x > R:
        return R
    elif x < -R:
        return -R
    else:
        return x


def sample_truncated_em(
    valid_number,
    dt,
    burn_in_steps,
    sample_stride,
    x0=0.0,
    low=-3.0,
    Sp=6.0,
    alpha=0.25,
):
    """
    Generate 'valid_number' samples from the truncated EM chain.

    Truncation radius R depends on dt via R(dt) = dt^{-alpha},
    which is a standard choice in Mao's truncated EM framework
    (radius grows as dt -> 0).

    Update rule:
      y_n = trunc(X_n; R)
      X_{n+1} = X_n + mu(y_n) dt + sigma dW_n

    Other parameters are as in sample_tamed_em.
    """
    samples = np.empty(valid_number, dtype=np.float64)
    x = x0
    R = dt ** (-alpha)

    # burn-in
    for _ in range(burn_in_steps):
        dW = math.sqrt(dt) * np.random.randn()
        y = truncate(x, R)
        mu = drift(y)
        x = x + mu * dt + sigma * dW

    # sampling
    n_valid = 0
    left = low
    right = low + Sp

    while n_valid < valid_number:
        for _ in range(sample_stride):
            dW = math.sqrt(dt) * np.random.randn()
            y = truncate(x, R)
            mu = drift(y)
            x = x + mu * dt + sigma * dW

        if left <= x <= right:
            samples[n_valid] = x
            n_valid += 1

    return samples


# ------------------------------
# 5. Main: compare stationary densities
# ------------------------------

if __name__ == "__main__":
    start = time.time()

    # Grid settings: same as Wen_cubic.py
    N = 120
    low = -3.0
    Sp = 6.0

    # Time-discretization parameters
    dt = 1e-3           # time step for EM methods
    burn_in_steps = 100000   # burn-in steps before sampling
    sample_stride = 10       # steps between two recorded samples
    valid_number = int(1e6)  # number of samples used for histogram (you can increase)

    # 1) True stationary density
    xs_true, dens_true, h = true_stationary_density(low, Sp, N)

    # 2) Tamed EM samples and density
    print("Sampling tamed EM...")
    samples_tamed = sample_tamed_em(
        valid_number=valid_number,
        dt=dt,
        burn_in_steps=burn_in_steps,
        sample_stride=sample_stride,
        x0=0.0,
        low=low,
        Sp=Sp,
    )
    xs_tamed, dens_tamed = histogram_density(samples_tamed, low, Sp, N)

    # 3) Truncated EM samples and density
    print("Sampling truncated EM...")
    samples_trunc = sample_truncated_em(
        valid_number=valid_number,
        dt=dt,
        burn_in_steps=burn_in_steps,
        sample_stride=sample_stride,
        x0=0.0,
        low=low,
        Sp=Sp,
        alpha=0.25,
    )
    xs_trunc, dens_trunc = histogram_density(samples_trunc, low, Sp, N)

    # 4) L1 errors
    l1_tamed = l1_norm(dens_tamed, dens_true, h)
    l1_trunc = l1_norm(dens_trunc, dens_true, h)

    print(f"L1 error of stationary density (tamed EM)     : {l1_tamed:.6e}")
    print(f"L1 error of stationary density (truncated EM) : {l1_trunc:.6e}")
    print(f"h = {h:.6f}, dt = {dt}")

    end = time.time()
    print(f"Total CPU time (Python script) = {end - start:.2f} s")




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
    plt.plot(xs_tamed, dens_tamed, "--", label="Tamed EM", linewidth=1.5)
    plt.plot(xs_trunc, dens_trunc, ":", label="Truncated EM", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.title("Cubic oscillator: stationary density")
    plt.tight_layout()
    plt.show()