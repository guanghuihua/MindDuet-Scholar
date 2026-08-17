from scipy.stats import skellam
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math # isnan

def RingDensity(x):
    y = [0.0, 0.0]
    y[0] = -4.0 * x[0] * (x[0] * x[0] + x[1] * x[1] - 1.0) + x[1]
    y[1] = -4.0 * x[1] * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0]
    return y

N = 512  # NxN Grid
count = 2000
eps0 = 0.5  # Strength of noise
lowx = -2.0  # left end of numerical domain
lowy = -2.0  # bottom end of numerical domain
Sp = 4.0  # length (and height) of the numerical domain
end_time = 10.0
h = Sp / N
print('h is ' + str(h))
#tau = 200.0*h*h/(2.0*eps0*eps0)
tau = 0.1
print('tau is ' + str(tau))
totalerror = 0.0
begin_time = datetime.datetime.now()


sample_2h = np.array(
    [lowx + h + round((1.0 - lowx - h) / (2 * h)) * 2 * h, lowy + h + round((1.0 - lowy - h) / (2 * h)) * 2 * h])
sample_h = sample_2h.copy()
for i in range(count):
    time = 0
    while time < end_time:
        #print('sample_h is ' + str(sample_h) + ' sample_2h is ' + str(sample_2h))
        tmp = RingDensity(sample_h)
        M11 = 0.5 * max(eps0 * eps0 - abs(tmp[0]) * h, 0.0)
        M22 = 0.5 * max(eps0 * eps0 - abs(tmp[1]) * h, 0.0)
        q1 = tau * (max(tmp[0], 0.0) / h + M11 / (h * h))
        q2 = tau * (-min(tmp[0], 0.0) / h + M11 / (h * h))
        q3 = tau * (max(tmp[1], 0.0) / h + M22 / (h * h))
        q4 = tau * (-min(tmp[1], 0.0) / h + M22 / (h * h))
        if q1 == 0 or q2 == 0 or q3 == 0 or q4 == 0:
            M11 = 0.5 * eps0 * eps0
            M22 = 0.5 * eps0 * eps0
            q1 = tau * (max(tmp[0], 0.0) / h + M11 / (h * h))
            q2 = tau * (-min(tmp[0], 0.0) / h + M11 / (h * h))
            q3 = tau * (max(tmp[1], 0.0) / h + M22 / (h * h))
            q4 = tau * (-min(tmp[1], 0.0) / h + M22 / (h * h))

        tmp = RingDensity(sample_2h)
        M11 = 0.5 * max(eps0*eps0 - abs(tmp[0]) * 2 * h, 0.0)
        M22 = 0.5 * max(eps0*eps0 - abs(tmp[1]) * 2 * h, 0.0)
        qq1 = tau * (max(tmp[0], 0.0)/(2 * h) + M11/(4 * h * h))
        qq2 = tau * (-min(tmp[0], 0.0)/(2 * h) + M11/(4 * h * h))
        qq3 = tau * (max(tmp[1], 0.0)/(2 * h)+M22/(4 * h * h))
        qq4 = tau * (-min(tmp[1], 0.0)/(2 * h)+M22/(4 * h * h))
        if qq1 == 0 or qq2 == 0 or qq3 == 0 or qq4 == 0:
            M11 = 0.5 * eps0 * eps0
            M22 = 0.5 * eps0 * eps0
            qq1 = tau * (max(tmp[0], 0.0) / (2 * h) + M11 / (4 * h * h))
            qq2 = tau * (-min(tmp[0], 0.0) / (2 * h) + M11 / (4 * h * h))
            qq3 = tau * (max(tmp[1], 0.0) / (2 * h) + M22 / (4 * h * h))
            qq4 = tau * (-min(tmp[1], 0.0) / (2 * h) + M22 / (4 * h * h))

        a = np.random.random_sample()
        b = np.random.random_sample()
        rnd1 = skellam.ppf(a, q1, q2)
        rnd2 = skellam.ppf(b, q3, q4)
        sample_h += [rnd1 * h, rnd2 * h]
        rnd1 = skellam.ppf(a, qq1, qq2)
        rnd2 = skellam.ppf(b, qq3, qq4)
        sample_2h += [rnd1 * 2.0 *h, rnd2 * 2.0 *h]
        # rnd1 = skellam.rvs(q1,q2)
        # rnd2 = skellam.rvs(q3,q4)
        #plt.plot(sample_h[0], sample_h[1], 'ro')
        #plt.plot(sample_2h[0], sample_2h[1], 'bo')
        time += tau
    error = abs(sample_h[0]-sample_2h[0]) + abs(sample_h[1]-sample_2h[1])
    totalerror += error
    print('loop is ' + str(i) +' error is ' + str(error))

    sample_2h = np.array(
        [lowx + h + round((sample_2h[0] - lowx - h) / (2 * h)) * 2 * h,
         lowy + h + round((sample_2h[1] - lowy - h) / (2 * h)) * 2 * h])
    sample_h = sample_2h.copy()

totalerror /= count
print('the totalerror is ' + str(totalerror))

print(datetime.datetime.now() - begin_time)

plt.axis([-2,2,-2,2])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
