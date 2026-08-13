# Stochastic Canard system: time vs space discretization (tamed EM / truncated EM vs CTRW-SSA)
# NOTE: This notebook generates figures and basic stats for Section 6.3 experiments.
import numpy as np
import matplotlib.pyplot as plt

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

rng = np.random.default_rng(42)

# ---------- System definition (stochastic canard) ----------
delta = 0.1
a = 1 - delta/8 - 3*delta**2/32 - 173*delta**3/1024 - 0.01
sigma_x = 0.0
sigma_y = 0.08  # small noise in slow variable

def drift(x, y):
    fx = (x + y - x**3/3.0)/delta
    fy = a - x
    return np.array([fx, fy])

def diffusion():
    # constant additive noise
    return np.array([sigma_x, sigma_y])  # independent noises

# Slow manifold (cubic nullcline for x')
def slow_manifold_xnull(x):
    return x**3/3.0 - x

# ---------- Time-discretization: Tamed EM ----------
def simulate_tamed_py(Npaths=20, dt=1e-3, T=5.0, x0=-1.5, y0=None, seed=42):
    if y0 is None:
        y0 = slow_manifold_xnull(x0)
    steps = int(np.ceil(T/dt))
    t = np.linspace(0, steps*dt, steps+1)
    X = np.zeros((Npaths, steps+1, 2))
    X[:,0,0] = x0
    X[:,0,1] = y0
    sig = diffusion()
    rng_local = np.random.default_rng(seed)
    for n in range(steps):
        # independent gaussians per path for y only (sigma_x may be zero)
        dW = rng_local.standard_normal((Npaths,2))*np.sqrt(dt)
        dW *= sig  # scale by diffusion
        x = X[:,n,0]; y = X[:,n,1]
        mu = np.vstack(drift(x, y)).T  # (Npaths,2)
        denom = 1.0 + dt*np.linalg.norm(mu, axis=1, ord=2)
        incr = (dt*mu.T/denom).T + dW
        X[:,n+1,:] = X[:,n,:] + incr
    return t, X

# ---------- Time-discretization: Truncated EM (simplified, radius depending on dt) ----------
def simulate_truncated_py(Npaths=20, dt=1e-3, T=5.0, x0=-1.5, y0=None, R0=2.0, seed=42):
    if y0 is None:
        y0 = slow_manifold_xnull(x0)
    steps = int(np.ceil(T/dt))
    t = np.linspace(0, steps*dt, steps+1)
    X = np.zeros((Npaths, steps+1, 2))
    X[:,0,0] = x0
    X[:,0,1] = y0
    sig = diffusion()
    rng_local = np.random.default_rng(seed)
    # Mao's h(Δ) ~ Δ^{-1/4}; we emulate with RΔ = R0 * Δ^{-1/4}
    R = R0 * (dt**(-0.25))
    for n in range(steps):
        dW = rng_local.standard_normal((Npaths,2))*np.sqrt(dt)
        dW *= sig
        x = X[:,n,0]; y = X[:,n,1]
        # truncate the state before computing drift
        r = np.sqrt(x*x + y*y)
        scale = np.minimum(1.0, R/np.maximum(r, 1e-12))
        xtr = x*scale; ytr = y*scale
        mu = np.vstack(drift(xtr, ytr)).T
        X[:,n+1,:] = X[:,n,:] + dt*mu + dW
    return t, X

# ---------- Space-discretization: CTRW-SSA with Qu (finite-difference) ----------
# Uniform rectangular grid with reflecting boundaries
def build_grid(xmin=-2.5, xmax=2.5, ymin=-2.0, ymax=3.0, hx=0.05, hy=0.05):
    xs = np.arange(xmin, xmax+1e-12, hx)
    ys = np.arange(ymin, ymax+1e-12, hy)
    nx, ny = len(xs), len(ys)
    return xs, ys, nx, ny, hx, hy

def precompute_rates_py(xs, ys, hx, hy):
    # Qu rates for uniform grid: for each node (i,j), rates to (i+1,j), (i-1,j), (i,j+1), (i,j-1)
    nx, ny = len(xs), len(ys)
    rates = np.zeros((nx, ny, 4))  # [rx+, rx-, ry+, ry-]
    # constants
    Mx = sigma_x**2/2.0
    My = sigma_y**2/2.0
    hi_x = hx  # average step size
    hi_y = hy
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            mu = drift(x, y)  # [fx, fy]
            # x-direction
            rxp = max(mu[0], 0.0)/hx + (Mx/(hi_x*hx) if Mx>0 else 0.0)
            rxm = max(-mu[0], 0.0)/hx + (Mx/(hi_x*hx) if Mx>0 else 0.0)
            # y-direction
            ryp = max(mu[1], 0.0)/hy + (My/(hi_y*hy) if My>0 else 0.0)
            rym = max(-mu[1], 0.0)/hy + (My/(hi_y*hy) if My>0 else 0.0)
            rates[i,j,:] = [rxp, rxm, ryp, rym]
    return rates

if NUMBA_AVAILABLE:
    @njit
    def _drift_pair_numba(x, y):
        fx = (x + y - x**3/3.0)/delta
        fy = a - x
        return fx, fy

    @njit(parallel=True)
    def simulate_tamed_numba(Npaths, dt, T, x0, y0, seed):
        steps = int(np.ceil(T/dt))
        t = np.linspace(0.0, steps*dt, steps+1)
        X = np.zeros((Npaths, steps+1, 2))
        X[:,0,0] = x0
        X[:,0,1] = y0
        sqrt_dt = np.sqrt(dt)
        np.random.seed(seed)
        for n in range(steps):
            for p in prange(Npaths):
                x = X[p,n,0]
                y = X[p,n,1]
                mu0, mu1 = _drift_pair_numba(x, y)
                norm_mu = np.sqrt(mu0*mu0 + mu1*mu1)
                denom = 1.0 + dt*norm_mu
                dWx = np.random.standard_normal() * sqrt_dt * sigma_x
                dWy = np.random.standard_normal() * sqrt_dt * sigma_y
                X[p,n+1,0] = x + (dt*mu0)/denom + dWx
                X[p,n+1,1] = y + (dt*mu1)/denom + dWy
        return t, X

    @njit(parallel=True)
    def simulate_truncated_numba(Npaths, dt, T, x0, y0, R0, seed):
        steps = int(np.ceil(T/dt))
        t = np.linspace(0.0, steps*dt, steps+1)
        X = np.zeros((Npaths, steps+1, 2))
        X[:,0,0] = x0
        X[:,0,1] = y0
        R = R0 * (dt**(-0.25))
        sqrt_dt = np.sqrt(dt)
        np.random.seed(seed)
        for n in range(steps):
            for p in prange(Npaths):
                x = X[p,n,0]
                y = X[p,n,1]
                r = np.sqrt(x*x + y*y)
                scale = R/ r if r > 1e-12 else R
                if scale > 1.0:
                    scale = 1.0
                xtr = x * scale
                ytr = y * scale
                mu0, mu1 = _drift_pair_numba(xtr, ytr)
                dWx = np.random.standard_normal() * sqrt_dt * sigma_x
                dWy = np.random.standard_normal() * sqrt_dt * sigma_y
                X[p,n+1,0] = x + dt*mu0 + dWx
                X[p,n+1,1] = y + dt*mu1 + dWy
        return t, X

    @njit(parallel=True)
    def precompute_rates_numba(xs, ys, hx, hy):
        nx = xs.shape[0]
        ny = ys.shape[0]
        rates = np.zeros((nx, ny, 4))
        Mx = sigma_x**2/2.0
        My = sigma_y**2/2.0
        hi_x = hx
        hi_y = hy
        for i in prange(nx):
            for j in range(ny):
                x = xs[i]
                y = ys[j]
                fx = y - (x*x*x/3.0 - x)
                fy = eps*(a - x)
                if fx > 0.0:
                    rxp = fx/hx
                    rxm = 0.0
                else:
                    rxp = 0.0
                    rxm = -fx/hx
                if fy > 0.0:
                    ryp = fy/hy
                    rym = 0.0
                else:
                    ryp = 0.0
                    rym = -fy/hy
                if Mx > 0.0:
                    diffx = Mx/(hi_x*hx)
                    rxp += diffx
                    rxm += diffx
                if My > 0.0:
                    diffy = My/(hi_y*hy)
                    ryp += diffy
                    rym += diffy
                rates[i, j, 0] = rxp
                rates[i, j, 1] = rxm
                rates[i, j, 2] = ryp
                rates[i, j, 3] = rym
        return rates

def simulate_tamed(Npaths=20, dt=1e-3, T=5.0, x0=-1.5, y0=None, seed=42, use_numba=NUMBA_AVAILABLE):
    if y0 is None:
        y0 = slow_manifold_xnull(x0)
    if use_numba and NUMBA_AVAILABLE:
        return simulate_tamed_numba(Npaths, dt, T, x0, y0, seed)
    return simulate_tamed_py(Npaths, dt, T, x0, y0, seed)

def simulate_truncated(Npaths=20, dt=1e-3, T=5.0, x0=-1.5, y0=None, R0=2.0, seed=42, use_numba=NUMBA_AVAILABLE):
    if y0 is None:
        y0 = slow_manifold_xnull(x0)
    if use_numba and NUMBA_AVAILABLE:
        return simulate_truncated_numba(Npaths, dt, T, x0, y0, R0, seed)
    return simulate_truncated_py(Npaths, dt, T, x0, y0, R0, seed)

def precompute_rates(xs, ys, hx, hy, use_numba=NUMBA_AVAILABLE):
    if use_numba and NUMBA_AVAILABLE:
        return precompute_rates_numba(xs, ys, hx, hy)
    return precompute_rates_py(xs, ys, hx, hy)

def ssa_paths(Npaths=15, T=5.0, x0=-1.5, y0=None, xs=None, ys=None, rates=None, hx=0.05, hy=0.05):
    if y0 is None:
        y0 = slow_manifold_xnull(x0)
    nx, ny = len(xs), len(ys)
    # map start to nearest grid
    i0 = int(np.clip(round((x0 - xs[0])/hx), 0, nx-1))
    j0 = int(np.clip(round((y0 - ys[0])/hy), 0, ny-1))
    paths = []
    times = []
    for p in range(Npaths):
        i, j = i0, j0
        t = 0.0
        path = [(t, xs[i], ys[j])]
        while t < T:
            r = rates[i,j].copy()
            # reflect at boundaries by disabling outward jumps
            if i == nx-1: r[0] = 0.0  # no +x
            if i == 0:    r[1] = 0.0  # no -x
            if j == ny-1: r[2] = 0.0  # no +y
            if j == 0:    r[3] = 0.0  # no -y
            lam = r.sum()
            if lam <= 0:
                # stuck
                break
            # exponential waiting time
            u = rng.random()
            dt = -np.log(u)/lam
            t += dt
            # choose channel
            u2 = rng.random()*lam
            if u2 < r[0]:
                i = min(i+1, nx-1)
            elif u2 < r[0]+r[1]:
                i = max(i-1, 0)
            elif u2 < r[0]+r[1]+r[2]:
                j = min(j+1, ny-1)
            else:
                j = max(j-1, 0)
            path.append((t, xs[i], ys[j]))
            if len(path) > 20000:
                break
        paths.append(np.array(path))
        times.append(t)
    return paths

# ---------- Metrics ----------
def detect_fast_jump_times_traj(t, X, x_threshold=1.5):
    # First time x crosses threshold from below
    idx = np.argmax((X[:,0] < x_threshold) & (X[:,0] >= x_threshold))  # dummy to create same shape
    # Actually compute per path
    Npaths = X.shape[0]
    Tjump = np.full(Npaths, np.nan)
    for k in range(Npaths):
        x = X[k,:,0]
        hit = np.where(x >= x_threshold)[0]
        if len(hit)>0:
            Tjump[k] = t[hit[0]]
    return Tjump

def detect_fast_jump_times_ssa(paths, x_threshold=1.5):
    Tjump = np.full(len(paths), np.nan)
    for k, path in enumerate(paths):
        x = path[:,1]
        hit = np.where(x >= x_threshold)[0]
        if len(hit)>0:
            Tjump[k] = path[hit[0],0]
    return Tjump
# ---------- Run experiments ----------
# Parameters
# T_end = 5.0
T_end = 1000
dt_ref = 1e-3

# Time-discretization simulations
t_tamed, Xt_tamed = simulate_tamed(Npaths=30, dt=dt_ref, T=T_end)
t_trunc, Xt_trunc = simulate_truncated(Npaths=30, dt=dt_ref, T=T_end)

# Space-discretization setup and SSA paths
xs, ys, nx, ny, hx, hy = build_grid()
rates = precompute_rates(xs, ys, hx, hy)
paths_ssa = ssa_paths(Npaths=20, T=T_end, xs=xs, ys=ys, rates=rates, hx=hx, hy=hy)

# ---------- Figure 1: Phase-plane sample trajectories (one per method) ----------
plt.figure(figsize=(6,5))
xx = np.linspace(-2.5, 2.5, 400)
plt.plot(xx, slow_manifold_xnull(xx), label="x-nullcline y = x^3/3 - x")  # slow manifold
# pick one path from each
plt.plot(Xt_tamed[0,:,0], Xt_tamed[0,:,1], lw=1.0, label="Tamed EM (dt=1e-3)")
plt.plot(Xt_trunc[0,:,0], Xt_trunc[0,:,1], lw=1.0, label="Truncated EM (dt=1e-3)")
p0 = paths_ssa[0]
plt.plot(p0[:,1], p0[:,2], lw=1.0, label="CTRW-SSA (Qu)")
plt.xlim([-2.5, 2.5]); plt.ylim([-2.0, 3.0])
plt.xlabel("x"); plt.ylabel("y"); plt.title("Stochastic Canard: sample trajectories")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Figure 2: Distribution of fast jump times ----------
Tjump_tamed = detect_fast_jump_times_traj(t_tamed, Xt_tamed)
Tjump_trunc = detect_fast_jump_times_traj(t_trunc, Xt_trunc)
Tjump_ssa = detect_fast_jump_times_ssa(paths_ssa)

bins = np.linspace(0, T_end, 30)
plt.figure(figsize=(6,4))
plt.hist(Tjump_tamed[~np.isnan(Tjump_tamed)], bins=bins, alpha=0.5, label="Tamed EM")
plt.hist(Tjump_trunc[~np.isnan(Tjump_trunc)], bins=bins, alpha=0.5, label="Truncated EM")
plt.hist(Tjump_ssa[~np.isnan(Tjump_ssa)], bins=bins, alpha=0.5, label="CTRW-SSA")
plt.xlabel("first time x >= 1.5"); plt.ylabel("count")
plt.title("Fast jump time distribution")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Figure 3: Occupancy heatmaps (coarse 2D histograms) ----------
def occupancy_heat(X, xbins, ybins):
    # X: (Npaths, Tsteps, 2)
    pts = X.reshape(-1,2)
    H, xe, ye = np.histogram2d(pts[:,0], pts[:,1], bins=[xbins, ybins])
    return H.T, xe, ye

xbins = np.linspace(-2.5, 2.5, 60)
ybins = np.linspace(-2.0, 3.0, 60)

H_tamed, xe, ye = occupancy_heat(Xt_tamed, xbins, ybins)

# For SSA, sample points at event locations across all paths
ssa_pts = np.vstack([p[:,1:3] for p in paths_ssa])
H_ssa, _, _ = np.histogram2d(ssa_pts[:,0], ssa_pts[:,1], bins=[xbins, ybins])

plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.imshow(H_tamed, origin='lower', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto')
plt.title("Occupancy (Tamed EM)"); plt.xlabel("x"); plt.ylabel("y")
plt.subplot(1,2,2)
plt.imshow(H_ssa.T, origin='lower', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto')
plt.title("Occupancy (CTRW-SSA)"); plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.show()

# ---------- Basic stats printout ----------
def summarize(name, Tjump):
    arr = Tjump[~np.isnan(Tjump)]
    if arr.size == 0:
        return f"{name}: no jumps detected"
    return f"{name}: n={arr.size}, mean={arr.mean():.3f}, std={arr.std(ddof=1):.3f}, min={arr.min():.3f}, max={arr.max():.3f}"

print(summarize("Tamed EM", Tjump_tamed))
print(summarize("Truncated EM", Tjump_trunc))
print(summarize("CTRW-SSA", Tjump_ssa))
