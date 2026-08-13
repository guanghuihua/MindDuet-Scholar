# ---------- Error vs work: common reference using very fine tamed EM (dt=1e-4) ----------
def generate_coupled_increments(T, dt_fine, Npaths):
    steps_fine = int(np.ceil(T/dt_fine))
    # Only y has noise
    dWy = rng.standard_normal((Npaths, steps_fine))*np.sqrt(dt_fine)*sigma_y
    dWx = rng.standard_normal((Npaths, steps_fine))*np.sqrt(dt_fine)*sigma_x  # possibly zero
    return dWx, dWy, steps_fine

def simulate_tamed_with_increments(dWx, dWy, dt_fine, dt, x0=-1.5, y0=slow_manifold_xnull(-1.5)):
    Npaths, steps_fine = dWy.shape
    steps = int(np.ceil(steps_fine*dt_fine/dt))
    # indices to aggregate
    ratio = int(round(dt/dt_fine))
    assert ratio >= 1 and abs(ratio - dt/dt_fine) < 1e-8
    X = np.zeros((Npaths, 2))
    X[:,0] = x0; X[:,1] = y0
    for k in range(steps):
        # aggregate increments
        sl = slice(k*ratio, (k+1)*ratio)
        dW_block = np.stack([dWx[:,sl].sum(axis=1), dWy[:,sl].sum(axis=1)], axis=1)
        mu = np.vstack(drift(X[:,0], X[:,1])).T
        denom = 1.0 + dt*np.linalg.norm(mu, axis=1, ord=2)
        incr = (dt*mu.T/denom).T + dW_block
        X = X + incr
    return X  # state at T

def simulate_trunc_with_increments(dWx, dWy, dt_fine, dt, x0=-1.5, y0=slow_manifold_xnull(-1.5), R0=2.0):
    Npaths, steps_fine = dWy.shape
    steps = int(np.ceil(steps_fine*dt_fine/dt))
    ratio = int(round(dt/dt_fine))
    assert ratio >= 1 and abs(ratio - dt/dt_fine) < 1e-8
    X = np.zeros((Npaths, 2))
    X[:,0] = x0; X[:,1] = y0
    R = R0 * (dt**(-0.25))
    for k in range(steps):
        sl = slice(k*ratio, (k+1)*ratio)
        dW_block = np.stack([dWx[:,sl].sum(axis=1), dWy[:,sl].sum(axis=1)], axis=1)
        r = np.linalg.norm(X, axis=1)
        scale = np.minimum(1.0, R/np.maximum(r, 1e-12))
        Xt = (X.T * scale).T
        mu = np.vstack(drift(Xt[:,0], Xt[:,1])).T
        X = X + dt*mu + dW_block
    return X

# Reference with dt_fine
T_end_ref = 40.0
dt_fine = 1e-4
Npaths_ref = 80
dWx, dWy, steps_f = generate_coupled_increments(T_end_ref, dt_fine, Npaths_ref)

# Reference (tamed with dt_fine)
X_ref = simulate_tamed_with_increments(dWx, dWy, dt_fine, dt_fine)

# Candidate step sizes
dts = [4e-3, 2e-3, 1e-3, 5e-4]
errs_tamed, work_tamed = [], []
errs_trunc, work_trunc = [], []

for dt in dts:
    X_t = simulate_tamed_with_increments(dWx, dWy, dt_fine, dt)
    X_s = simulate_trunc_with_increments(dWx, dWy, dt_fine, dt)
    # strong RMS error at T
    e_t = np.sqrt(np.mean(np.sum((X_t - X_ref)**2, axis=1)))
    e_s = np.sqrt(np.mean(np.sum((X_s - X_ref)**2, axis=1)))
    errs_tamed.append(e_t); errs_trunc.append(e_s)
    work_tamed.append(Npaths_ref * (T_end_ref/dt))  # proxy: #steps * #paths
    work_trunc.append(Npaths_ref * (T_end_ref/dt))

# For CTRW-SSA: measure error to reference by nearest-time snapshot at T
# We simulate SSA with multiple grid sizes (work ~ expected number of jumps)
hx_list = [0.1, 0.075, 0.05]
errs_ctrw, work_ctrw = [], []
for hx in hx_list:
    xs_, ys_, nx_, ny_, hx_, hy_ = build_grid(hx=hx, hy=hx)
    rates_ = precompute_rates(xs_, ys_, hx_, hy_)
    paths_ = ssa_paths(Npaths=Npaths_ref, T=T_end_ref, xs=xs_, ys=ys_, rates=rates_, hx=hx_, hy=hy_)
    # State at time T (last point per path)
    X_T = np.zeros((len(paths_),2))
    jumps = 0
    for k, p in enumerate(paths_):
        # p[:,0] is time; choose last time <= T_end_ref
        idx = np.searchsorted(p[:,0], T_end_ref, side='right') - 1
        idx = max(idx, 0)
        X_T[k,0] = p[idx,1]; X_T[k,1] = p[idx,2]
        jumps += (len(p)-1)
    # Need same number of paths as reference; if fewer (due to domain issues), pad by repeating
    if X_T.shape[0] < Npaths_ref:
        pad = np.repeat(X_T[-1:], Npaths_ref - X_T.shape[0], axis=0)
        X_T = np.vstack([X_T, pad])
    # Compare to reference
    e_c = np.sqrt(np.mean(np.sum((X_T[:Npaths_ref] - X_ref)**2, axis=1)))
    errs_ctrw.append(e_c)
    work_ctrw.append(jumps)  # proxy: total number of jumps

# Plot error vs work
plt.figure(figsize=(6,5))
plt.loglog(work_tamed, errs_tamed, marker='o', label='tamed EM (time)')
plt.loglog(work_trunc, errs_trunc, marker='s', label='truncated EM (time)')
plt.loglog(work_ctrw, errs_ctrw, marker='^', label='CTRW Qu (space)')
plt.xlabel("work (proxy)"); plt.ylabel("RMS strong error at T")
plt.title("Canard SDE: error vs work (common tamed-EM reference)")
plt.legend()
plt.tight_layout()
plt.show()

errs_tamed, errs_trunc, errs_ctrw, work_tamed[0], work_ctrw[0]


# Retry with a much lighter common reference to meet time limits
T_end_ref = 10.0
dt_fine = 5e-4
Npaths_ref = 30
dWx, dWy, steps_f = generate_coupled_increments(T_end_ref, dt_fine, Npaths_ref)
X_ref = simulate_tamed_with_increments(dWx, dWy, dt_fine, dt_fine)

dts = [4e-3, 2e-3, 1e-3]
errs_tamed, work_tamed = [], []
errs_trunc, work_trunc = [], []

for dt in dts:
    X_t = simulate_tamed_with_increments(dWx, dWy, dt_fine, dt)
    X_s = simulate_trunc_with_increments(dWx, dWy, dt_fine, dt)
    e_t = np.sqrt(np.mean(np.sum((X_t - X_ref)**2, axis=1)))
    e_s = np.sqrt(np.mean(np.sum((X_s - X_ref)**2, axis=1)))
    errs_tamed.append(e_t); errs_trunc.append(e_s)
    work_tamed.append(Npaths_ref * (T_end_ref/dt))
    work_trunc.append(Npaths_ref * (T_end_ref/dt))

hx_list = [0.12, 0.08, 0.06]
errs_ctrw, work_ctrw = [], []
for hx in hx_list:
    xs_, ys_, nx_, ny_, hx_, hy_ = build_grid(hx=hx, hy=hx)
    rates_ = precompute_rates(xs_, ys_, hx_, hy_)
    paths_ = ssa_paths(Npaths=Npaths_ref, T=T_end_ref, xs=xs_, ys=ys_, rates=rates_, hx=hx_, hy=hy_)
    X_T = np.zeros((len(paths_),2))
    jumps = 0
    for k, p in enumerate(paths_):
        idx = np.searchsorted(p[:,0], T_end_ref, side='right') - 1
        idx = max(idx, 0)
        X_T[k,0] = p[idx,1]; X_T[k,1] = p[idx,2]
        jumps += (len(p)-1)
    if X_T.shape[0] < Npaths_ref:
        pad = np.repeat(X_T[-1:], Npaths_ref - X_T.shape[0], axis=0)
        X_T = np.vstack([X_T, pad])
    e_c = np.sqrt(np.mean(np.sum((X_T[:Npaths_ref] - X_ref)**2, axis=1)))
    errs_ctrw.append(e_c)
    work_ctrw.append(jumps)

plt.figure(figsize=(6,5))
plt.loglog(work_tamed, errs_tamed, marker='o', label='tamed EM (time)')
plt.loglog(work_trunc, errs_trunc, marker='s', label='truncated EM (time)')
plt.loglog(work_ctrw, errs_ctrw, marker='^', label='CTRW Qu (space)')
plt.xlabel("work (proxy)"); plt.ylabel("RMS strong error at T")
plt.title("Canard SDE: error vs work (common tamed-EM reference)")
plt.legend()
plt.tight_layout()
plt.show()

list(zip(work_tamed, errs_tamed)), list(zip(work_ctrw, errs_ctrw))
