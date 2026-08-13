from numba import jit, njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
import numba


# numba.config.NUMBA_DEFAULT_NUM_THREADS = 4

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
    while k <= n - 1:
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
    end_l = round(l * 4 / 5)
    slope = np.polyfit(Tspan[:end_l], dist[:end_l], 1)
    return Tspan, dist, slope


@njit
def Ringdensity_reflection_Coupling():
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

    sample1_x_n = round((sample1[0] - lowx - h / 2.0) / h)
    sample1_y_n = round((sample1[1] - lowy - h / 2.0) / h)
    sample2_x_n = round((sample2[0] - lowx - h / 2.0) / h)
    sample2_y_n = round((sample2[1] - lowy - h / 2.0) / h)

    if abs(sample1_x_n - sample2_x_n) >= abs(sample1_y_n - sample2_y_n):
        flag_reflection = False
    else:
        flag_reflection = True

    flag_sample1_time = False
    flag_sample2_time = False
    flag_sample1_shift = False
    flag_sample2_shift = False
    flag_sample1_diffusion = False
    flag_sample2_diffusion = False

    # sample1_count = 0
    # sample2_count = 0
    sample1_time = 0.0
    sample2_time = 0.0

    rnd1 = random.random()
    rnd2 = random.random()
    rnd3 = random.random()
    rnd4 = random.random()

    # q1 = np.zeros(4)
    # q2 = np.zeros(4)
    tmp1 = RingDensity(sample1)
    M1 = np.zeros(2)
    M1[0] = 0.5 * max(eps0 * eps0 - abs(tmp1[0]) * h, 0.0)
    M1[1] = 0.5 * max(eps0 * eps0 - abs(tmp1[1]) * h, 0.0)
    shift1 = (abs(tmp1[0]) + abs(tmp1[1])) / h
    diffusion1 = 2 * (M1[0] + M1[1]) / (h * h)
    lambda1 = shift1 + diffusion1

    tmp2 = RingDensity(sample2)
    M2 = np.zeros(2)
    M2[0] = 0.5 * max(eps0 * eps0 - abs(tmp2[0]) * h, 0.0)
    M2[1] = 0.5 * max(eps0 * eps0 - abs(tmp2[1]) * h, 0.0)
    shift2 = (abs(tmp2[0]) + abs(tmp2[1])) / h
    diffusion2 = 2 * (M2[0] + M2[1]) / (h * h)
    lambda2 = shift2 + diffusion2
    # q1[0] = max(tmp[0], 0.0) / h + M11 / (h * h)
    # q1[1] = -min(tmp[0], 0.0) / h + M11 / (h * h)
    # q1[2] = max(tmp[1], 0.0) / h + M22 / (h * h)
    # q1[3] = -min(tmp[1], 0.0) / h + M22 / (h * h)
    # Lambda1 = np.sum(q1)
    # rnd1 = random.random()
    # tau = -math.log(1 - rnd1) / Lambda1
    # sample1_time += tau
    #
    # tmp = RingDensity(sample2)
    # M11 = 0.5 * max(eps0 * eps0 - abs(tmp[0]) * h, 0.0)
    # M22 = 0.5 * max(eps0 * eps0 - abs(tmp[1]) * h, 0.0)
    # q2[0] = max(tmp[0], 0.0) / h + M11 / (h * h)
    # q2[1] = -min(tmp[0], 0.0) / h + M11 / (h * h)
    # q2[2] = max(tmp[1], 0.0) / h + M22 / (h * h)
    # q2[3] = -min(tmp[1], 0.0) / h + M22 / (h * h)
    # Lambda2 = np.sum(q2)
    # rnd1 = random.random()
    # tau = -math.log(1 - rnd1) / Lambda2
    # sample2_time += tau

    q = np.zeros(4)

    while sample1_time < end_time:
        # step of sample1
        if sample1_time <= sample2_time:
            # print("sample1 step")
            if not flag_sample2_time:  # flag_sample2_time = False
                rnd1 = random.random()
                rnd2 = random.random()
                flag_sample1_time = True
            else:
                flag_sample1_time = False
                flag_sample2_time = False
            # end time flag
            if shift1 > rnd2 * lambda1:  # begin sample1 shift step
                q[0] = max(tmp1[0], 0.0) / h
                q[1] = -min(tmp1[0], 0.0) / h
                q[2] = max(tmp1[1], 0.0) / h
                q[3] = -min(tmp1[1], 0.0) / h
                lambda_q = q[0] + q[1] + q[2] + q[3]
                if not flag_sample2_shift:  # flag_sample2_shift False
                    rnd3 = random.random()
                    flag_sample1_shift = True
                else:
                    flag_sample1_shift = False
                    flag_sample2_shift = False
                # end shift flag

                mu_number = 0
                amu = q[mu_number]
                while amu < rnd3 * lambda_q:
                    mu_number += 1
                    amu += q[mu_number]

                if mu_number == 0:
                    sample1[0] = sample1[0] + h
                elif mu_number == 1:
                    sample1[0] = sample1[0] - h
                elif mu_number == 2:
                    sample1[1] = sample1[1] + h
                elif mu_number == 3:
                    sample1[1] = sample1[1] - h
                else:
                    print("nu_number is wrong, at shift step in sample1")
                    print("mu_number is ", mu_number)
            # end if sample1 shift
            else:  # begin sample1 diffusion step
                q[0] = M1[0] / (h * h)
                q[1] = M1[0] / (h * h)
                q[2] = M1[1] / (h * h)
                q[3] = M1[1] / (h * h)
                lambda_q = q[0] + q[1] + q[2] + q[3]
                if not flag_sample2_diffusion:  # flag_sample2_diffusion False,
                    # begin original diffusion step
                    rnd4 = random.random()
                    flag_sample1_diffusion = True
                    mu_number = 0
                    amu = q[mu_number]
                    while amu < rnd4 * lambda_q:
                        mu_number += 1
                        amu += q[mu_number]

                    if mu_number == 0:
                        sample1[0] = sample1[0] + h
                    elif mu_number == 1:
                        sample1[0] = sample1[0] - h
                    elif mu_number == 2:
                        sample1[1] = sample1[1] + h
                    elif mu_number == 3:
                        sample1[1] = sample1[1] - h
                    else:
                        print("nu_number is wrong, at original diffusion step in sample1")
                        print("mu_number is ", mu_number)
                #  end original diffusion step
                else:  # begin reflection diffusion step
                    flag_sample1_diffusion = False
                    flag_sample2_diffusion = False
                    mu_number = 0
                    amu = q[mu_number]
                    while amu < rnd4 * lambda_q:
                        mu_number += 1
                        amu += q[mu_number]
                    if not flag_reflection:  # reflection x
                        if mu_number == 0:
                            sample1[0] = sample1[0] - h
                        elif mu_number == 1:
                            sample1[0] = sample1[0] + h
                        elif mu_number == 2:
                            sample1[1] = sample1[1] + h
                        elif mu_number == 3:
                            sample1[1] = sample1[1] - h
                        else:
                            print("nu_number is wrong, at reflection x diffusion step in sample1")
                            print("mu_number is ", mu_number)
                    else:  # reflection y
                        if mu_number == 0:
                            sample1[0] = sample1[0] + h
                        elif mu_number == 1:
                            sample1[0] = sample1[0] - h
                        elif mu_number == 2:
                            sample1[1] = sample1[1] - h
                        elif mu_number == 3:
                            sample1[1] = sample1[1] + h
                        else:
                            print("nu_number is wrong, at reflection y diffusion step in sample1")
                            print("mu_number is ", mu_number)
                #  end reflection diffusion step
            # end if sample1 diffusion step
            sample1_x_n = round((sample1[0] - lowx - h / 2.0) / h)
            sample1_y_n = round((sample1[1] - lowy - h / 2.0) / h)

            if sample1_x_n == sample2_x_n and sample1_y_n == sample2_y_n:
                couplingtime = sample1_time
                break
            tmp1 = RingDensity(sample1)
            M1[0] = 0.5 * max(eps0 * eps0 - abs(tmp1[0]) * h, 0.0)
            M1[1] = 0.5 * max(eps0 * eps0 - abs(tmp1[1]) * h, 0.0)
            shift1 = (abs(tmp1[0]) + abs(tmp1[1])) / h
            diffusion1 = 2 * (M1[0] + M1[1]) / (h * h)
            lambda1 = shift1 + diffusion1

            tau = -math.log(1 - rnd1) / lambda1
            sample1_time += tau
        #  end the step of sample1
        else:   # begin the step of sample2
            # print("sample2 step")
            if not flag_sample1_time:  # flag_sample1_time = False
                rnd1 = random.random()
                rnd2 = random.random()
                flag_sample2_time = True
            else:
                flag_sample1_time = False
                flag_sample2_time = False
            # end time flag
            if shift2 > rnd2 * lambda2:  # begin sample2 shift step
                q[0] = max(tmp2[0], 0.0) / h
                q[1] = -min(tmp2[0], 0.0) / h
                q[2] = max(tmp2[1], 0.0) / h
                q[3] = -min(tmp2[1], 0.0) / h
                lambda_q = q[0] + q[1] + q[2] + q[3]
                if not flag_sample1_shift:  # flag_sample1_shift False
                    rnd3 = random.random()
                    flag_sample2_shift = True
                else:
                    flag_sample1_shift = False
                    flag_sample2_shift = False
                # end shift flag

                mu_number = 0
                amu = q[mu_number]
                while amu < rnd3 * lambda_q:
                    mu_number += 1
                    amu += q[mu_number]

                if mu_number == 0:
                    sample2[0] = sample2[0] + h
                elif mu_number == 1:
                    sample2[0] = sample2[0] - h
                elif mu_number == 2:
                    sample2[1] = sample2[1] + h
                elif mu_number == 3:
                    sample2[1] = sample2[1] - h
                else:
                    print("nu_number is wrong, at shift step in sample2")
                    print("mu_number is ", mu_number)
            # end if sample2 shift
            else:  # begin sample2 diffusion step
                q[0] = M2[0] / (h * h)
                q[1] = M2[0] / (h * h)
                q[2] = M2[1] / (h * h)
                q[3] = M2[1] / (h * h)
                lambda_q = q[0] + q[1] + q[2] + q[3]
                if not flag_sample1_diffusion:  # flag_sample1_diffusion False,
                    # begin original diffusion step
                    rnd4 = random.random()
                    flag_sample2_diffusion = True
                    mu_number = 0
                    amu = q[mu_number]
                    while amu < rnd4 * lambda_q:
                        mu_number += 1
                        amu += q[mu_number]

                    if mu_number == 0:
                        sample2[0] = sample2[0] + h
                    elif mu_number == 1:
                        sample2[0] = sample2[0] - h
                    elif mu_number == 2:
                        sample2[1] = sample2[1] + h
                    elif mu_number == 3:
                        sample2[1] = sample2[1] - h
                    else:
                        print("nu_number is wrong, at original diffusion step in sample2")
                        print("mu_number is ", mu_number)
                #  end original diffusion step
                else:  # begin reflection diffusion step
                    flag_sample1_diffusion = False
                    flag_sample2_diffusion = False
                    mu_number = 0
                    amu = q[mu_number]
                    while amu < rnd4 * lambda_q:
                        mu_number += 1
                        amu += q[mu_number]
                    if not flag_reflection:  # reflection x
                        if mu_number == 0:
                            sample2[0] = sample2[0] - h
                        elif mu_number == 1:
                            sample2[0] = sample2[0] + h
                        elif mu_number == 2:
                            sample2[1] = sample2[1] + h
                        elif mu_number == 3:
                            sample2[1] = sample2[1] - h
                        else:
                            print("nu_number is wrong, at reflection x diffusion step in sample2")
                            print("mu_number is ", mu_number)
                    else:  # reflection y
                        if mu_number == 0:
                            sample2[0] = sample2[0] + h
                        elif mu_number == 1:
                            sample2[0] = sample2[0] - h
                        elif mu_number == 2:
                            sample2[1] = sample2[1] - h
                        elif mu_number == 3:
                            sample2[1] = sample2[1] + h
                        else:
                            print("nu_number is wrong, at reflection y diffusion step in sample2")
                            print("mu_number is ", mu_number)
                #  end reflection diffusion step
            # end if sample1 diffusion step
            sample2_x_n = round((sample2[0] - lowx - h / 2.0) / h)
            sample2_y_n = round((sample2[1] - lowy - h / 2.0) / h)

            if sample1_x_n == sample2_x_n and sample1_y_n == sample2_y_n:
                couplingtime = sample1_time
                break
            tmp2 = RingDensity(sample2)
            M2[0] = 0.5 * max(eps0 * eps0 - abs(tmp2[0]) * h, 0.0)
            M2[1] = 0.5 * max(eps0 * eps0 - abs(tmp2[1]) * h, 0.0)
            shift2 = (abs(tmp2[0]) + abs(tmp2[1])) / h
            diffusion2 = 2 * (M2[0] + M2[1]) / (h * h)
            lambda2 = shift2 + diffusion2

            tau = -math.log(1 - rnd1) / lambda2
            sample2_time += tau
            #  end the step of sample2

        if abs(sample1_x_n - sample2_x_n) < abs(sample1_y_n - sample2_y_n) and not flag_reflection:
            flag_sample1_diffusion = False
            flag_sample2_diffusion = False
            flag_reflection = True

        if abs(sample1_y_n - sample2_y_n) < abs(sample1_x_n - sample2_x_n) and flag_reflection:
            flag_sample1_diffusion = False
            flag_sample2_diffusion = False
            flag_reflection = False
    # end while
    return couplingtime


@njit
#@njit(parallel=True)
def my_fun(loop):
    data = np.zeros(loop)
    for i in range(loop):
        data[i] = Ringdensity_reflection_Coupling()
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
# print(data)

Tspan, dist, slope = ExpDecayRatefun(data)

end = time.time()
print("Total loop is ", loop)
print("Total time is = %s" % (end - start))

plt.plot(Tspan, dist, 'b')
plt.plot(Tspan, Tspan * slope[0] + slope[1], 'r--')
plt.title('N = %s, slope = %s' % (str(N), str(slope[0])))
plt.xlim([0, 100])
plt.ylim([-10, 0])
plt.legend(["the reflection coupling time", "the linear fitting"])
plt.savefig("Reflection_coupling_N_%s.eps" % str(N))
plt.savefig("Reflection_coupling_N_%s.jpg" % str(N))
plt.show()
