# The tau-leaping method to simulate the Lorenz model
# March 2, 2022
# SDE:
# dx = [10 * (y - x)]dt + eps0 * dWt;
# dy = [x * (28 - z) - y]dt + eps0 * dWt;
# dz = [x * y - 8.0/3 * z]dt + eps0 * dWt;
# --------------------------------------------------------------------
from numba import njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt

@njit
def tau_leaping_lorenz():
    def lorenz_fun(sample):
        tmp = np.zeros(3)
        tmp[0] = 10 * (sample[1] - sample[0])
        tmp[1] = sample[0] * (28 - sample[2]) - sample[1]
        tmp[2] = sample[0] * sample[1] - 8.0/3 * sample[2]
        return tmp

    data = np.zeros(N*N*N)  # The distribution calculated by tau-Leap method
    data_XY = np.zeros(N*N)
    sample = np.zeros(3)
    sample[0] = lowx + h_new / 2.0 + round((lowx + Sp * random.random() - lowx - h_new / 2.0) / h_new) * h_new
    sample[1] = lowy + h_new / 2.0 + round((lowy + Sp * random.random() - lowy - h_new / 2.0) / h_new) * h_new
    sample[1] = lowz + h_new / 2.0 + round((lowy + Sp * random.random() - lowy - h_new / 2.0) / h_new) * h_new
    n = 0
    total_number = 0
    sample_out = 0
    run_time = 0.0
    M = np.zeros(3)
    q = np.zeros(6)
    rnd = np.zeros(6)
    POINT_WITH_ONLY_1_ORDER = 0
    time_count = 0.0
    while run_time < t_stop:
        tmp = lorenz_fun(sample)
        M[0] = 0.5 * max(eps0 * eps0 - abs(tmp[0]) * h_new, 0)
        M[1] = 0.5 * max(eps0 * eps0 - abs(tmp[1]) * h_new, 0)
        M[2] = 0.5 * max(eps0 * eps0 - abs(tmp[2]) * h_new, 0)
        if M[0] == 0 or M[1] == 0 or M[2] == 0:
            POINT_WITH_ONLY_1_ORDER += 1

        q[0] = tau * (max(tmp[0], 0) / h_new + M[0] / (h_new * h_new))
        q[1] = tau * (-min(tmp[0], 0) / h_new + M[0] / (h_new * h_new))
        q[2] = tau * (max(tmp[1], 0) / h_new + M[1] / (h_new * h_new))
        q[3] = tau * (-min(tmp[1], 0) / h_new + M[1] / (h_new * h_new))
        q[4] = tau * (max(tmp[2], 0) / h_new + M[1] / (h_new * h_new))
        q[5] = tau * (-min(tmp[2], 0) / h_new + M[1] / (h_new * h_new))

        rnd[0] = np.random.poisson(q[0])
        rnd[1] = np.random.poisson(q[1])
        rnd[2] = np.random.poisson(q[2])
        rnd[3] = np.random.poisson(q[3])
        rnd[4] = np.random.poisson(q[4])
        rnd[5] = np.random.poisson(q[5])

        #print("rnd is ", rnd)
        sample[0] = sample[0] + (rnd[0] - rnd[1]) * h_new
        sample[1] = sample[1] + (rnd[2] - rnd[3]) * h_new
        sample[2] = sample[2] + (rnd[4] - rnd[5]) * h_new
        #print("sample is ", sample)
        #plt.plot(sample[0], sample[1])

        x_n = round((sample[0] - lowx - h / 2.0) / h)
        y_n = round((sample[1] - lowy - h / 2.0) / h)
        z_n = round((sample[2] - lowz - h / 2.0) / h)
        if x_n >= 0 and x_n < N and y_n >= 0 and y_n < N and z_n >= 0 and z_n < N:
            #data[x_n*N*N + y_n*N + z_n] = data[x_n*N*N + y_n*N + z_n] + 1.0  # Count number of sample points in each bin
            data_XY[x_n*N + z_n] = data_XY[x_n*N + z_n] +1.0
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
    data_XY = data_XY / (h * h * h * n)  # Normalization
    return data_XY
    #data = data / (h * h * h * n)  # Normalization
    #return data
@njit
#@njit(fastmath=True,parallel=True)
def my_fun(data):
    for ii in range(loop):
        data[ii, :] = tau_leaping_lorenz()
        print("Loop is", ii)
    return data

##########################################################################################

start = time.time()# initial CPU time
loop = 2

t_stop = 20000.0  # the total reaction time

tau = 0.002
lowx = -25.0
lowy = -25.0
lowz = 0.0

Sp = 50.0  # the span
N = 1024
h = Sp / N
eps0 = 10
eps = 0.5 * eps0 * eps0
scale = 20
h_new = h / scale

data_matrix = np.zeros((loop, N*N))
output_data = sum(my_fun(data_matrix))
final_data = output_data / loop

XZ_plot = np.zeros((N, N)) #for plot
print("Begin to plot XZ!!")
for i in range(N):
    for j in range(N):
            XZ_plot[N-j-1, i] = final_data[i*N + j]

print("h is ", h)
end = time.time()
print("Total time is (with compilation) = ", (end - start))

fig = plt.figure()
sub = plt.imshow(XZ_plot, cmap='rainbow', interpolation="nearest")
plt.colorbar(sub)
plt.show()