from numba import jit, njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 4

def ExpDecayRatefun(A):
    T_step = 0.05
    B = np.sort(A)
    n = len(B)
    T = T_step
    j = 0
    k = 0
    countNum = 0
    dist_list = []
    Tspan_list = []
    while k <= n-1:
        if B[k] <= T:
            countNum = countNum + 1
            k = k + 1
        else:
            dist_list.append(countNum)
            Tspan_list.append(T)
            j = j + 1
            T = T + T_step
    l = len(Tspan_list)
    dist = np.array(dist_list)
    Tspan = np.array(Tspan_list)
    dist = np.log(np.ones(l) - dist / n)
    end_l = round(l*4/5)
    slope = np.polyfit(Tspan[:end_l], dist[:end_l], 1)
    return Tspan, dist, slope



@njit
def Ringdensity_Independent_Coupling():
    def RingDensity(x):
        y = [0.0, 0.0]
        y[0] = -4.0 * x[0] * (x[0] * x[0] + x[1] * x[1] - 1.0) + x[1]
        y[1] = -4.0 * x[1] * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0]
        return y

    couplingtime = 0

    sample1 = np.array([lowx + h / 2.0 + round((lowx + Sp * random.random() - lowx - h / 2.0) / h) * h,
                        lowy + h / 2.0 + round((lowy + Sp * random.random() - lowy - h / 2.0) / h) * h])

    sample2 = np.array([lowx + h / 2.0 + round((lowx + Sp * random.random() - lowx - h / 2.0) / h) * h,
                        lowy + h / 2.0 + round((lowy + Sp * random.random() - lowy - h / 2.0) / h) * h])

    sample1_x_n = round((sample1[0] - lowx - h/2.0)/h)
    sample1_y_n = round((sample1[1] - lowy - h/2.0)/h)
    sample2_x_n = round((sample2[0] - lowx - h/2.0)/h)
    sample2_y_n = round((sample2[1] - lowy - h/2.0)/h)

    sample1_count = 0
    sample2_count = 0
    sample1_time = 0
    sample2_time = 0
    q1 = np.zeros(4)
    q2 = np.zeros(4)
    tmp = RingDensity(sample1)
    M11 = 0.5 * max(eps0 * eps0 - abs(tmp[0]) * h, 0.0)
    M22 = 0.5 * max(eps0 * eps0 - abs(tmp[1]) * h, 0.0)
    q1[0] = max(tmp[0], 0.0) / h + M11 / (h * h)
    q1[1] = -min(tmp[0], 0.0) / h + M11 / (h * h)
    q1[2] = max(tmp[1], 0.0) / h + M22 / (h * h)
    q1[3] = -min(tmp[1], 0.0) / h + M22 / (h * h)
    Lambda1 = np.sum(q1)
    rnd1 = random.random()
    tau = -math.log(1 - rnd1) / Lambda1
    sample1_time += tau

    tmp = RingDensity(sample2)
    M11 = 0.5 * max(eps0 * eps0 - abs(tmp[0]) * h, 0.0)
    M22 = 0.5 * max(eps0 * eps0 - abs(tmp[1]) * h, 0.0)
    q2[0] = max(tmp[0], 0.0) / h + M11 / (h * h)
    q2[1] = -min(tmp[0], 0.0) / h + M11 / (h * h)
    q2[2] = max(tmp[1], 0.0) / h + M22 / (h * h)
    q2[3] = -min(tmp[1], 0.0) / h + M22 / (h * h)
    Lambda2 = np.sum(q2)
    rnd1 = random.random()
    tau = -math.log(1 - rnd1) / Lambda2
    sample2_time += tau

    while sample1_time < end_time:
    # step of sample1
        if sample1_time <= sample2_time:
            rnd2 = random.random()
            mu_number = 0
            amu = q1[mu_number]
            while amu < rnd2 * Lambda1:
                mu_number += 1
                amu += q1[mu_number]

            if mu_number == 0:
                sample1[0] = sample1[0] + h
            elif mu_number == 1:
                sample1[0] = sample1[0] - h
            elif mu_number == 2:
                sample1[1] = sample1[1] + h
            elif mu_number == 3:
                sample1[1] = sample1[1] - h
            else:
                print("nu_number is wrong")
                print("mu_number is ", mu_number)
                print("q1 is ", q1)
                print()
                print("rnd2 is", rnd2)

            sample1_x_n = round((sample1[0] - lowx - h / 2.0) / h)
            sample1_y_n = round((sample1[1] - lowy - h / 2.0) / h)

            if sample1_x_n == sample2_x_n and sample1_y_n == sample2_y_n:
                couplingtime = sample1_time
                break
            tmp = RingDensity(sample1)
            M11 = 0.5 * max(eps0 * eps0 - abs(tmp[0]) * h, 0.0)
            M22 = 0.5 * max(eps0 * eps0 - abs(tmp[1]) * h, 0.0)
            q1[0] = max(tmp[0], 0.0) / h + M11 / (h * h)
            q1[1] = -min(tmp[0], 0.0) / h + M11 / (h * h)
            q1[2] = max(tmp[1], 0.0) / h + M22 / (h * h)
            q1[3] = -min(tmp[1], 0.0) / h + M22 / (h * h)
            Lambda1 = np.sum(q1)
            rnd1 = random.random()
            tau = -math.log(1 - rnd1) / Lambda1
            sample1_time += tau
        else:
            rnd2 = random.random()
            mu_number = 0
            amu = q2[mu_number]
            while amu < rnd2 * Lambda2:
                mu_number += 1
                amu += q2[mu_number]

            if mu_number == 0:
                sample2[0] = sample2[0] + h
            elif mu_number == 1:
                sample2[0] = sample2[0] - h
            elif mu_number == 2:
                sample2[1] = sample2[1] + h
            elif mu_number == 3:
                sample2[1] = sample2[1] - h
            else:
                print("nu_number is wrong")

            sample2_x_n = round((sample2[0] - lowx - h / 2.0) / h)
            sample2_y_n = round((sample2[1] - lowy - h / 2.0) / h)
            if sample1_x_n == sample2_x_n and sample1_y_n == sample2_y_n:
                couplingtime = sample2_time
                break
            tmp = RingDensity(sample2)
            M11 = 0.5 * max(eps0 * eps0 - abs(tmp[0]) * h, 0.0)
            M22 = 0.5 * max(eps0 * eps0 - abs(tmp[1]) * h, 0.0)
            q2[0] = max(tmp[0], 0.0) / h + M11 / (h * h)
            q2[1] = -min(tmp[0], 0.0) / h + M11 / (h * h)
            q2[2] = max(tmp[1], 0.0) / h + M22 / (h * h)
            q2[3] = -min(tmp[1], 0.0) / h + M22 / (h * h)
            Lambda2 = np.sum(q2)
            rnd1 = random.random()
            tau = -math.log(1 - rnd1) / Lambda2
            sample2_time += tau
    return couplingtime

@njit
#@njit(parallel=True)
def my_fun(loop):
    data = np.zeros(loop)
    for i in range(loop):
        data[i] = Ringdensity_Independent_Coupling()
        print("Loop is", i)
    return data
##########################################################################################

start = time.time()

N = 512
loop = 10000
end_time = 200000

eps0 = 0.5
eps = 0.5 * eps0 * eps0

lowx = -2.0
lowy = -2.0
Sp = 4
h = Sp / N

data = my_fun(loop)
#print(data)

Tspan, dist, slope = ExpDecayRatefun(data)

end = time.time()
print("Total loop is ", loop)
print("Total time is = %s" % (end - start))

plt.plot(Tspan, dist, 'b')
plt.plot(Tspan, Tspan * slope[0] + slope[1], 'r--')
plt.title('N = %s, slope = %s' % (str(N), str(slope[0])))
plt.xlim([0, 200])
plt.ylim([-10, 0])
plt.legend(["the independent coupling time", "the linear fitting"])
plt.savefig("Independent_coupling_N_%s.eps" % str(N))
plt.savefig("Independent_coupling_N_%s.jpg" % str(N))
plt.show()