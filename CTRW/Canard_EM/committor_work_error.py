# Work–error curves for committor on stochastic canard:
# Compare truncated-EM Monte Carlo vs CTRW (generator solve via Gauss–Seidel) 
# using the CTRW fine-grid solution as reference.
#
# This cell computes numeric tables and plots.

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

rng = np.random.default_rng(7)

# ------- Model --------
eps = 1e-2
a = 1.0
sigma_x = 0.0
sigma_y = 0.08

def drift(x, y):
    fx = y - (x**3/3.0 - x)
    fy = eps * (a - x)
    return fx, fy

# domain and sets
xmin, xmax = -2.5, 2.5
ymin, ymax = -1.0, 3.0
xA, xB = -0.2, 1.5  # A={x<=xA}, B={x>=xB}

# ------- Grid / rates (Qu) --------
def build_grid(xmin, xmax, ymin, ymax, hx, hy=None):
    if hy is None: hy = hx
    xs = np.arange(xmin, xmax + 1e-12, hx)
    ys = np.arange(ymin, ymax + 1e-12, hy)
    return xs, ys, hx, hy

def precompute_rates(xs, ys, hx, hy):
    nx, ny = len(xs), len(ys)
    rxp = np.zeros((nx, ny)); rxm = np.zeros((nx, ny))
    ryp = np.zeros((nx, ny)); rym = np.zeros((nx, ny))
    Mx = sigma_x**2/2.0; My = sigma_y**2/2.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            fx, fy = drift(x, y)
            rxp[i,j] = max(fx, 0.0)/hx + (Mx/(hx*hx) if Mx>0 else 0.0)
            rxm[i,j] = max(-fx,0.0)/hx + (Mx/(hx*hx) if Mx>0 else 0.0)
            ryp[i,j] = max(fy, 0.0)/hy + (My/(hy*hy) if My>0 else 0.0)
            rym[i,j] = max(-fy,0.0)/hy + (My/(hy*hy) if My>0 else 0.0)
    return rxp, rxm, ryp, rym

# masks
def boundary_masks(xs, ys):
    nx, ny = len(xs), len(ys)
    A = np.zeros((nx, ny), dtype=bool)
    B = np.zeros((nx, ny), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if x <= xA: A[i,j] = True
            if x >= xB: B[i,j] = True
    return A, B

# ------- Gauss–Seidel solver for Qq=0 and Qm=-1 (reflecting outer boundary) --------
def solve_committor_qu(xs, ys, rates, max_sweeps=2000, tol=1e-5, omega=1.4):
    rxp, rxm, ryp, rym = rates
    nx, ny = len(xs), len(ys)
    A, B = boundary_masks(xs, ys)
    q = np.zeros((nx, ny))
    q[B] = 1.0
    # precompute lambda and neighbor masks
    for sweep in range(max_sweeps):
        diff = 0.0
        # red-black or plain GS; we'll use in-place GS
        for i in range(nx):
            for j in range(ny):
                if A[i,j] or B[i,j]:
                    continue
                rpx = rxp[i,j]; rmx = rxm[i,j]; rpy = ryp[i,j]; rmy = rym[i,j]
                # reflect outer boundaries by nulling outward rates
                if i == nx-1: rpx = 0.0
                if i == 0:    rmx = 0.0
                if j == ny-1: rpy = 0.0
                if j == 0:    rmy = 0.0
                lam = rpx + rmx + rpy + rmy
                if lam <= 0:
                    # no movement; keep current
                    continue
                # neighbors
                q_right = q[i+1,j] if i+1 < nx else q[i,j]
                q_left  = q[i-1,j] if i-1 >= 0 else q[i,j]
                q_up    = q[i,j+1] if j+1 < ny else q[i,j]
                q_down  = q[i,j-1] if j-1 >= 0 else q[i,j]
                q_new = (rpx*q_right + rmx*q_left + rpy*q_up + rmy*q_down) / lam
                q_old = q[i,j]
                q[i,j] = (1-omega)*q_old + omega*q_new
                diff = max(diff, abs(q[i,j]-q_old))
        if diff < tol:
            return q, sweep+1, (nx*ny - A.sum() - B.sum())*(sweep+1)
    return q, max_sweeps, (nx*ny - A.sum() - B.sum())*max_sweeps

# ------- Nearest-neighbor interpolation --------
def nn_val(x, y, xs, ys, F):
    i = int(np.clip(round((x - xs[0])/(xs[1]-xs[0])), 0, len(xs)-1))
    j = int(np.clip(round((y - ys[0])/(ys[1]-ys[0])), 0, len(ys)-1))
    return F[i,j]

# ------- Evaluation nodes along slow manifold --------
def slow_y(x):
    return x**3/3.0 - x

def build_eval_nodes(xmin_eval=-0.6, xmax_eval=1.2, K=10):
    xsamp = np.linspace(xmin_eval, xmax_eval, K)
    pts = np.column_stack([xsamp, slow_y(xsamp)])
    return pts

eval_pts = build_eval_nodes(K=10)

# ------- Truncated EM Monte Carlo for committor --------
def step_truncated(z, dt, R0=2.0):
    x, y = z
    R = R0 * (dt**(-0.25))
    r = np.hypot(x, y); scale = min(1.0, R/max(r,1e-12))
    xt, yt = x*scale, y*scale
    fx, fy = drift(xt, yt)
    dWx = rng.normal(0.0, np.sqrt(dt))*sigma_x
    dWy = rng.normal(0.0, np.sqrt(dt))*sigma_y
    xn = x + dt*fx + dWx
    yn = y + dt*fy + dWy
    # reflecting box
    if xn < xmin: xn = xmin + (xmin - xn)
    if xn > xmax: xn = xmax - (xn - xmax)
    if yn < ymin: yn = ymin + (ymin - yn)
    if yn > ymax: yn = ymax - (yn - ymax)
    return np.array([xn, yn])

def mc_committor_trunc(z0, dt, N, T_max=30.0):
    hits_B = 0
    steps_total = 0
    for _ in range(N):
        z = np.array(z0, float)
        t = 0.0
        while t < T_max:
            if z[0] >= xB: hits_B += 1; break
            if z[0] <= xA: break
            z = step_truncated(z, dt)
            t += dt; steps_total += 1
    qhat = hits_B / max(N,1)
    return qhat, steps_total

# ------- Driver: compute CTRW ref, CTRW coarse (iterative), and truncated-EM MC -------
# 1) Fine CTRW (reference)
hx_ref = 0.06
xs_ref, ys_ref, hx, hy = build_grid(xmin, xmax, ymin, ymax, hx_ref)
rates_ref = precompute_rates(xs_ref, ys_ref, hx, hy)
q_ref, sweeps_ref, work_ref = solve_committor_qu(xs_ref, ys_ref, rates_ref, max_sweeps=1200, tol=5e-5, omega=1.4)

# Reference values at eval points
q_ref_pts = np.array([nn_val(x, y, xs_ref, ys_ref, q_ref) for x,y in eval_pts])

# 2) CTRW on coarser grids (measure error vs work)
hx_list = [0.12, 0.09, 0.075]
ct_work = []
ct_err = []
ct_sweeps = []
for hx_c in hx_list:
    xs_c, ys_c, _, _ = build_grid(xmin, xmax, ymin, ymax, hx_c)
    rates_c = precompute_rates(xs_c, ys_c, hx_c, hx_c)
    q_c, sweeps_c, work_c = solve_committor_qu(xs_c, ys_c, rates_c, max_sweeps=1500, tol=5e-5, omega=1.4)
    # interpolate to eval points, compare to ref
    q_c_pts = np.array([nn_val(x, y, xs_c, ys_c, q_c) for x,y in eval_pts])
    err = float(np.sqrt(np.mean((q_c_pts - q_ref_pts)**2)))
    ct_work.append(work_c)
    ct_err.append(err)
    ct_sweeps.append(sweeps_c)

# 3) Truncated-EM MC (vary N at fixed dt) to scan work
dt = 0.004
N_list = [16, 32, 64, 96]
tm_work = []
tm_err = []
for N in N_list:
    est = []
    work_sum = 0
    for (x,y), qtrue in zip(eval_pts, q_ref_pts):
        qhat, steps = mc_committor_trunc((x,y), dt, N, T_max=25.0)
        est.append(qhat - qtrue)
        work_sum += steps
    err = float(np.sqrt(np.mean(np.square(est))))
    tm_work.append(work_sum)
    tm_err.append(err)

# Assemble numeric tables
ct_table = list(zip(hx_list, ct_sweeps, ct_work, ct_err))
tm_table = list(zip(N_list, tm_work, tm_err))

# ------- Plots -------
plt.figure(figsize=(6,4.5))
plt.loglog(ct_work, ct_err, marker='^', label='CTRW (Q-solve, GS iteration)')
plt.loglog(tm_work, tm_err, marker='s', label='truncated EM (MC)')
plt.xlabel('work (proxy)')
plt.ylabel('RMS error vs CTRW fine reference')
plt.title('Committor: error vs work')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.savefig('Canard_EM/work_error_committor_trunc_vs_ctrw.png', bbox_inches='tight')
plt.show()

# Show numeric results
ct_table, tm_table
