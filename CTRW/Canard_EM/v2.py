# Re-run with lighter settings to ensure execution within limits.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

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

def boundary_masks(xs, ys):
    nx, ny = len(xs), len(ys)
    A = np.zeros((nx, ny), dtype=bool)
    B = np.zeros((nx, ny), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if x <= xA: A[i,j] = True
            if x >= xB: B[i,j] = True
    return A, B

# Gauss–Seidel iteration for Qq=0
def solve_committor_qu(xs, ys, rates, max_sweeps=700, tol=2e-4, omega=1.35):
    rxp, rxm, ryp, rym = rates
    nx, ny = len(xs), len(ys)
    A, B = boundary_masks(xs, ys)
    q = np.zeros((nx, ny)); q[B] = 1.0
    nfree = nx*ny - A.sum() - B.sum()
    for sweep in range(max_sweeps):
        diff = 0.0
        for i in range(nx):
            for j in range(ny):
                if A[i,j] or B[i,j]:
                    continue
                rpx = rxp[i,j]; rmx = rxm[i,j]; rpy = ryp[i,j]; rmy = rym[i,j]
                if i == nx-1: rpx = 0.0
                if i == 0:    rmx = 0.0
                if j == ny-1: rpy = 0.0
                if j == 0:    rmy = 0.0
                lam = rpx + rmx + rpy + rmy
                if lam <= 0: 
                    continue
                q_right = q[i+1,j] if i+1 < nx else q[i,j]
                q_left  = q[i-1,j] if i-1 >= 0 else q[i,j]
                q_up    = q[i,j+1] if j+1 < ny else q[i,j]
                q_down  = q[i,j-1] if j-1 >= 0 else q[i,j]
                q_new = (rpx*q_right + rmx*q_left + rpy*q_up + rmy*q_down) / lam
                q_old = q[i,j]
                q[i,j] = (1-omega)*q_old + omega*q_new
                diff = max(diff, abs(q[i,j]-q_old))
        if diff < tol:
            return q, sweep+1, nfree*(sweep+1)
    return q, max_sweeps, nfree*max_sweeps

def nn_val(x, y, xs, ys, F):
    i = int(np.clip(round((x - xs[0])/(xs[1]-xs[0])), 0, len(xs)-1))
    j = int(np.clip(round((y - ys[0])/(ys[1]-ys[0])), 0, len(ys)-1))
    return F[i,j]

def slow_y(x): return x**3/3.0 - x

def build_eval_nodes(xmin_eval=-0.6, xmax_eval=1.2, K=10):
    xsamp = np.linspace(xmin_eval, xmax_eval, K)
    pts = np.column_stack([xsamp, slow_y(xsamp)])
    return pts

# Truncated EM MC for committor
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

def mc_committor_trunc(z0, dt, N, T_max=20.0):
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

# ---------- Build reference (CTRW on fine grid) ----------
hx_ref = 0.08  # lighter
xs_ref, ys_ref, hx, hy = build_grid(xmin, xmax, ymin, ymax, hx_ref)
rates_ref = precompute_rates(xs_ref, ys_ref, hx, hy)
q_ref, sweeps_ref, work_ref = solve_committor_qu(xs_ref, ys_ref, rates_ref, max_sweeps=600, tol=3e-4, omega=1.35)

eval_pts = build_eval_nodes(K=10)
q_ref_pts = np.array([nn_val(x, y, xs_ref, ys_ref, q_ref) for x,y in eval_pts])

# ---------- CTRW at coarser grids ----------
hx_list = [0.14, 0.12, 0.10]
ct_work = []; ct_err = []; ct_sweeps = []
for hx_c in hx_list:
    xs_c, ys_c, _, _ = build_grid(xmin, xmax, ymin, ymax, hx_c)
    rates_c = precompute_rates(xs_c, ys_c, hx_c, hx_c)
    q_c, sweeps_c, work_c = solve_committor_qu(xs_c, ys_c, rates_c, max_sweeps=700, tol=3e-4, omega=1.35)
    q_c_pts = np.array([nn_val(x, y, xs_c, ys_c, q_c) for x,y in eval_pts])
    err = float(np.sqrt(np.mean((q_c_pts - q_ref_pts)**2)))
    ct_work.append(work_c); ct_err.append(err); ct_sweeps.append(sweeps_c)

# ---------- Truncated-EM MC (vary N) ----------
dt = 0.004
N_list = [12, 24, 48, 72]
tm_work = []; tm_err = []
for N in N_list:
    est = []; work_sum = 0
    for (x,y), qtrue in zip(eval_pts, q_ref_pts):
        qhat, steps = mc_committor_trunc((x,y), dt, N, T_max=18.0)
        est.append(qhat - qtrue); work_sum += steps
    err = float(np.sqrt(np.mean(np.square(est))))
    tm_work.append(work_sum); tm_err.append(err)

# ---------- Plot ----------
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

# ---------- Prepare numeric tables ----------
# import pandas as pd
# ct_df = pd.DataFrame({
#     "hx": hx_list,
#     "sweeps": ct_sweeps,
#     "work_proxy": ct_work,
#     "RMS_error": ct_err
# })
# tm_df = pd.DataFrame({
#     "N_paths_per_node": N_list,
#     "work_proxy": tm_work,
#     "RMS_error": tm_err
# })

# from caas_jupyter_tools import display_dataframe_to_user
# display_dataframe_to_user("CTRW (Q-solve) work–error (committor)", ct_df)
# display_dataframe_to_user("Truncated EM MC work–error (committor)", tm_df)

# print("Saved figure: /mnt/data/work_error_committor_trunc_vs_ctrw.pdf")

# Inspect values to diagnose the huge errors
import numpy as np

def summary(F, name):
    finite = np.isfinite(F)
    print(name, "shape=", getattr(F, "shape", None), 
          "min=", np.nanmin(F[finite]) if np.any(finite) else None,
          "max=", np.nanmax(F[finite]) if np.any(finite) else None)

summary(q_ref, "q_ref grid")
print("q_ref at eval pts:", q_ref_pts)

print("CTRW work, err:", list(zip(ct_work, ct_err)))
print("TruncEM work, err:", list(zip(tm_work, tm_err)))

# Recompute q_ref with safer omega=1.0 and check values
hx_ref = 0.08
xs_ref, ys_ref, hx, hy = build_grid(xmin, xmax, ymin, ymax, hx_ref)
rates_ref = precompute_rates(xs_ref, ys_ref, hx, hy)

q_ref, sweeps_ref, work_ref = solve_committor_qu(xs_ref, ys_ref, rates_ref, max_sweeps=1200, tol=1e-4, omega=1.0)
print("sweeps_ref=", sweeps_ref, "work_ref=", work_ref)
print("q_ref stats: min", q_ref.min(), "max", q_ref.max())
q_ref_pts = np.array([nn_val(x, y, xs_ref, ys_ref, q_ref) for x,y in eval_pts])
print("q_ref_pts:", q_ref_pts[:6])

# Recompute CTRW coarser grids and truncated-EM MC against the corrected reference
eval_pts = build_eval_nodes(K=10)
q_ref_pts = np.array([nn_val(x, y, xs_ref, ys_ref, q_ref) for x,y in eval_pts])

# CTRW coarse
hx_list = [0.14, 0.12, 0.10]
ct_work = []; ct_err = []; ct_sweeps = []
for hx_c in hx_list:
    xs_c, ys_c, _, _ = build_grid(xmin, xmax, ymin, ymax, hx_c)
    rates_c = precompute_rates(xs_c, ys_c, hx_c, hx_c)
    q_c, sweeps_c, work_c = solve_committor_qu(xs_c, ys_c, rates_c, max_sweeps=1000, tol=1e-4, omega=1.0)
    q_c_pts = np.array([nn_val(x, y, xs_c, ys_c, q_c) for x,y in eval_pts])
    err = float(np.sqrt(np.mean((q_c_pts - q_ref_pts)**2)))
    ct_work.append(work_c); ct_err.append(err); ct_sweeps.append(sweeps_c)

# Truncated-EM MC
dt = 0.004
N_list = [12, 24, 48, 72]
tm_work = []; tm_err = []
for N in N_list:
    est = []; work_sum = 0
    for (x,y), qtrue in zip(eval_pts, q_ref_pts):
        qhat, steps = mc_committor_trunc((x,y), dt, N, T_max=18.0)
        est.append(qhat - qtrue); work_sum += steps
    err = float(np.sqrt(np.mean(np.square(est))))
    tm_work.append(work_sum); tm_err.append(err)

# Plot
plt.figure(figsize=(6,4.5))
plt.loglog(ct_work, ct_err, marker='^', label='CTRW (Q-solve, GS iteration)')
plt.loglog(tm_work, tm_err, marker='s', label='truncated EM (MC)')
plt.xlabel('work (proxy)')
plt.ylabel('RMS error vs CTRW fine reference')
plt.title('Committor: error vs work')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.savefig('Canard_EM/work_error_committor_trunc_vs_ctrw.pdf', bbox_inches='tight')
plt.show()
