from numba import jit, njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 16

@njit
def SSA():
    def eqn321Fun(xx):
        yy = - pow(xx, 3)
        return yy
    data = np.zeros(N)
    # X0 = low + Sp * random.random()
    X0 = 100
    Sample_h = low + h / 2.0 + round((X0 - low - h / 2.0) / h) * h
    n = 0
    time = 0.0
    while n < valid_number:
        q = np.zeros(2)
        tmp = eqn321Fun(Sample_h)
        M = 0.5 * max(2-abs(tmp) * h, 0.0)
        #M = 0.5 * eps0 * eps0

        q[0] = max(tmp, 0.0) / h+M / (h * h)
        q[1] = -min(tmp, 0.0) / h+M / (h * h)

        Lambda = q[0] + q[1]
        rnd1 = random.random() #python中内置的Random类就是采用了MT19937算法
        rnd2 = random.random()

        #print("ran1 is %f, ran2 is %f" % (rnd1, rnd2))
        tau = -math.log(1-rnd1) / Lambda
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
            print("nu_number is wrong")

        x_n = round((Sample_h - low - h / 2.0) / h)
        if x_n >= 0 and x_n < N:
            data[x_n] = data[x_n] + 1
            n += 1
        else:
            print("x_n out of scope")
    data /= n * h
    #print("Valid N_sample is ", n)
    return data

@njit
def l1_norm(u):
    accum = 0.0
    #print(len(u))
    for ii in range(len(u)):
        accum += abs(u[ii]) * h
    return accum

@njit
def KK():
    local_N = 100000000
    upper_bound = 100.0
    lower_bound = 0.0
    local_h = (upper_bound - lower_bound) / local_N
    local_K = 0.0
    for ii in range(local_N):
        local_K = local_K + math.exp(-0.25 * pow((ii+0.5) * local_h, 4))
    return 2 * local_K * local_h

# @ njit
@njit(fastmath=True, parallel=True)
def my_fun(data):
    for ii in range(loop):
        data[ii, :] = SSA()
        print("Loop is", ii)
    return data
##########################################################################################

start = time.time()

N = 120
# loop = 1000
loop = 3
valid_number = 1e+7 #end_time = 200000

eps0 = math.sqrt(2)
eps = 0.5 * eps0 * eps0

low = -3.0
Sp = 6
h = Sp / N

Z = KK()
data1 = np.zeros(N)
for i in range(N):
    x = low + h / 2.0 + i * h
    data1[i] = 1 / Z * math.exp(-0.25 * pow(x, 4))

data_matrix = np.zeros((loop, N))
output_data = sum(my_fun(data_matrix))
Final_data = output_data / loop

data2 = Final_data - data1
print("l1 norm of SSA_Qu is %f " % l1_norm(data2))
print("h is %f " % h)
end = time.time()
print("Total time is (with compilation) = %s" % (end - start))

plt.plot(Final_data, label="Qu distribution")
plt.plot(data1, label="True distribution")
plt.legend()
plt.show()

if __name__ == '__main__':
    pass
