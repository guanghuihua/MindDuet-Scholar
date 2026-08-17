# The tau-leaping method to simulate the Ring density model with small epsilon
# March 1, 2022
# SDE:
# dx = [- 4 * x * (x * x + y * y - 1) + y]dt + eps0 * dWt;
# dy = [- 4 * y * (x * x + y * y - 1) - x]dt + eps0 * dWt;
# --------------------------------------------------------------------
# Two paths | transite rate
from numba import njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt

@njit
def qu_with_small_diffusion():
    def ring_density_fun(sample):
        tmp = np.zeros(2)
        tmp[0] = - 4 * sample[0] * (sample[0] * sample[0] + sample[1] * sample[1] - 1) + sample[1]
        tmp[1] = - 4 * sample[1] * (sample[0] * sample[0] + sample[1] * sample[1] - 1) - sample[0]
        return tmp

    data = np.zeros(N*N)  # The distribution calculated by tau-Leap method
    sample = np.zeros(2)
    sample[0] = lowx + h / 2.0 + round((lowx + Sp * random.random() - lowx - h / 2.0) / h) * h
    sample[1] = lowy + h / 2.0 + round((lowy + Sp * random.random() - lowy - h/ 2.0) / h) * h
    n = 0
    run_time = 0.0
    M = np.zeros(2)
    q = np.zeros(4)
    rnd = np.zeros(2)
    POINT_WITH_ONLY_1_ORDER = 0
    sample_out = 0
    total_number = 0
    time_count = 0.0
    while run_time < t_stop:
        tmp = ring_density_fun(sample)
        M[0] = 0.5* max(eps0 * eps0 - abs(tmp[0]) * h, 0)
        M[1] = 0.5* max(eps0 * eps0 - abs(tmp[1]) * h, 0)
        if M[0] == 0 or M[1] == 0:
            POINT_WITH_ONLY_1_ORDER += 1

        q[0] = (max(tmp[0], 0) / h + M[0] / (h * h))
        q[1] = (-min(tmp[0], 0) / h + M[0] / (h * h))
        q[2] = (max(tmp[1], 0) / h + M[1] / (h * h))
        q[3] = (-min(tmp[1], 0) / h + M[1] / (h* h))
        Lambda= q[0]+q[1]+q[2]+q[3]
        rnd[0] = random.random()
        rnd[1] = random.random()
        tau = -math.log(1 - rnd[0]) / Lambda
        #print("rnd is ", rnd)
        mu_number = 0
        amu = q[mu_number]
        while amu < rnd[1] * Lambda:
            mu_number += 1
            amu += q[mu_number]
        if mu_number == 0:
            sample[0] = sample[0] + h
        elif mu_number == 1:
            sample[0] = sample[0] - h
        elif mu_number == 2:
            sample[1] = sample[1] + h
        elif mu_number == 3:
            sample[1] = sample[1] - h
        else:
            print("nu_number is wrong")
            # print("rnd is ", rnd)
        # print("sample is ", sample)
        # plt.plot(sample[0], sample[1])

        x_n = round((sample[0] - lowx - h / 2.0) / h)
        y_n = round((sample[1] - lowy - h / 2.0) / h)
        #print("sample is ", sample)
        #plt.plot(sample[0], sample[1])


        if x_n >= 0 and x_n < N and y_n >= 0 and y_n < N:
            data[x_n*N + y_n] = data[x_n*N + y_n] + 1.0  # Count number of sample points in each bin
            n += 1
        else:
            #print("sample out of scope")
            sample_out += 1
        run_time += tau
        total_number += 1
        if run_time > time_count:
            print("time_count is ", time_count)
            time_count = time_count + 1000
    print("POINT_WITH_ONLY_1_ORDER is ", POINT_WITH_ONLY_1_ORDER)
    print("sample_out is ", sample_out)
    print("total_number is ", total_number)
    data = data / (h * h * n)  # Normalization
    return data


@njit
def l1_norm(u):
    accum = 0.0
    # print(len(u))
    for ii in range(len(u)):
        accum += abs(u[ii]) * h * h
    return accum


@njit
def K(epsilon):
    local_N = 10000000
    upper_bound = 1000.0
    lower_bound = -1.0
    local_h = (upper_bound - lower_bound) / local_N
    KK = 0.0
    for i in range(local_N):
        KK = KK + math.exp(-2 * pow(-1 + (i+0.5) * local_h, 2) / pow(epsilon, 2))
    return KK * local_h * 3.1415926

@njit
#@njit(fastmath=True,parallel=True)
def my_fun(data):
    for ii in range(loop):
        data[ii, :] = qu_with_small_diffusion()
        print("Loop is", ii)
    return data

##########################################################################################

start = time.time()# initial CPU time
loop = 1

t_stop = 20000.0  # the total reaction time
lowx = -2.0
lowy = -2.0

Sp = 4.0  # the span
N = 256
h = Sp / N
eps0 = 1
eps = 0.5 * eps0 * eps0
data_exact = np.zeros(N * N)  # The exact distribution

#
data_matrix = np.zeros((loop, N*N))
output_data = sum(my_fun(data_matrix))
final_data = output_data / loop

#final_data = tau_leaping_with_small_diffusion()

data_exact_plot = np.zeros((N, N)) #for plot
final_data_plot = np.zeros((N, N)) #for plot
# KK = 0.1969
KK = K(eps0)
for i in range(N):
    for j in range(N):
        x = lowx + h / 2 + i * h
        y = lowy + h / 2 + j * h
        V = (x * x + y * y - 1) * (x * x + y * y - 1)
        data_exact[i * N + j] = 1 / KK * math.exp(- V / eps)

        data_exact_plot[i, j] = 1 / KK * math.exp(-2 * V / eps)
        final_data_plot[i, j] = final_data[i*N + j]

data2 = final_data - data_exact
print("l1 norm of SSA_Qu is %f " % l1_norm(data2))
#print("l1 norm of final_data is %f " % l1_norm(final_data))
#print("l1 norm of data_exact is %f " % l1_norm(data_exact))
print("h is %f " % h)
end = time.time()
print("Total time is (with compilation) = %s" % (end - start))

#plt.plot(final_data_plot, label="Qu distribution")


fig = plt.figure()
# plt.plot(final_data)
#
sub = fig.add_subplot(111)

sub.imshow(final_data_plot, cmap='rainbow', interpolation="nearest")
# sub.heatmap
# #plt.legend()
plt.show()
