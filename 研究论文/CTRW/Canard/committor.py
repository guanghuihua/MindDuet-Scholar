# Committor and MFPT on the CTRW (Qu) spatial generator for the stochastic canard system
# Reuse previously defined: build_grid, precompute_rates, slow_manifold_xnull, drift, etc.
import numpy as np
import matplotlib.pyplot as plt

# Try scipy for sparse linear algebra; fall back to dense if unavailable
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except Exception as e:
    SCIPY_AVAILABLE = False
    print("SciPy not available, will use dense linear algebra (may be slower).", e)

rng = np.random.default_rng(42)

# ---------- System definition (stochastic canard / van der Pol-FHN style) ----------
# Fast variable x, slow variable y
# dx = (y - (x^3/3 - x)) dt + sigma_x dW_x   (we set sigma_x = 0 by default)
# dy = eps*(a - x) dt + sigma_y dW_y
eps = 0.01
a = 1.0
sigma_x = 0.0
sigma_y = 0.08  # small noise in slow variable

def drift(x, y):
    fx = y - (x**3/3.0 - x)
    fy = eps*(a - x)
    return np.array([fx, fy])

def diffusion():
    # constant additive noise
    return np.array([sigma_x, sigma_y])  # independent noises

# Slow manifold (cubic nullcline for x')
def slow_manifold_xnull(x):
    return x**3/3.0 - x

# ---------- Time-discretization: Tamed EM ----------
def simulate_tamed(Npaths=20, dt=1e-3, T=5.0, x0=-1.5, y0=slow_manifold_xnull(-1.5)):
    steps = int(np.ceil(T/dt))
    t = np.linspace(0, steps*dt, steps+1)
    X = np.zeros((Npaths, steps+1, 2))
    X[:,0,0] = x0
    X[:,0,1] = y0
    sig = diffusion()
    for n in range(steps):
        # independent gaussians per path for y only (sigma_x may be zero)
        dW = rng.standard_normal((Npaths,2))*np.sqrt(dt)
        dW *= sig  # scale by diffusion
        x = X[:,n,0]; y = X[:,n,1]
        mu = np.vstack(drift(x, y)).T  # (Npaths,2)
        denom = 1.0 + dt*np.linalg.norm(mu, axis=1, ord=2)
        incr = (dt*mu.T/denom).T + dW
        X[:,n+1,:] = X[:,n,:] + incr
    return t, X

# ---------- Time-discretization: Truncated EM (simplified, radius depending on dt) ----------
def simulate_truncated(Npaths=20, dt=1e-3, T=5.0, x0=-1.5, y0=slow_manifold_xnull(-1.5), R0=2.0):
    steps = int(np.ceil(T/dt))
    t = np.linspace(0, steps*dt, steps+1)
    X = np.zeros((Npaths, steps+1, 2))
    X[:,0,0] = x0
    X[:,0,1] = y0
    sig = diffusion()
    # Mao's h(Δ) ~ Δ^{-1/4}; we emulate with RΔ = R0 * Δ^{-1/4}
    R = R0 * (dt**(-0.25))
    for n in range(steps):
        dW = rng.standard_normal((Npaths,2))*np.sqrt(dt)
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
def build_grid(xmin=-2.5, xmax=2.5, ymin=-1.0, ymax=3.0, hx=0.05, hy=0.05):
    xs = np.arange(xmin, xmax+1e-12, hx)
    ys = np.arange(ymin, ymax+1e-12, hy)
    nx, ny = len(xs), len(ys)
    return xs, ys, nx, ny, hx, hy

def precompute_rates(xs, ys, hx, hy):
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

# --- Assemble generator Q (reflecting outer boundary) ---
# Node indexing: k = i + j*nx (row-major in x fast)
def assemble_Q(xs, ys, rates):
    nx, ny = len(xs), len(ys)
    N = nx*ny
    rows = []
    cols = []
    vals = []
    def idx(i,j): return i + j*nx
    for j in range(ny):
        for i in range(nx):
            k = idx(i,j)
            rxp, rxm, ryp, rym = rates[i,j]  # +x, -x, +y, -y
            # reflect boundaries by zeroing outward rates
            if i == nx-1: rxp = 0.0
            if i == 0:    rxm = 0.0
            if j == ny-1: ryp = 0.0
            if j == 0:    rym = 0.0
            # off-diagonals (non-negative)
            if i+1 < nx and rxp>0:
                rows.append(k); cols.append(idx(i+1,j)); vals.append(rxp)
            if i-1 >= 0 and rxm>0:
                rows.append(k); cols.append(idx(i-1,j)); vals.append(rxm)
            if j+1 < ny and ryp>0:
                rows.append(k); cols.append(idx(i,j+1)); vals.append(ryp)
            if j-1 >= 0 and rym>0:
                rows.append(k); cols.append(idx(i,j-1)); vals.append(rym)
            # diagonal
            lam = rxp+rxm+ryp+rym
            rows.append(k); cols.append(k); vals.append(-lam)
    if SCIPY_AVAILABLE:
        Q = sp.csr_matrix((vals, (rows, cols)), shape=(N,N))
    else:
        Q = np.zeros((N,N))
        for r,c,v in zip(rows, cols, vals):
            Q[r,c] += v
    return Q

# --- Helper: Dirichlet elimination for linear systems with boundary sets ---
def solve_dirichlet(Q, rhs, fix_mask, g_dirichlet):
    if SCIPY_AVAILABLE:
        # Partition into interior vs fixed
        free = ~fix_mask
        Qii = Q[free][:, free]
        rhs_eff = rhs[free] - Q[free][:, fix_mask] @ g_dirichlet[fix_mask]
        sol = np.zeros(Q.shape[0])
        sol[fix_mask] = g_dirichlet[fix_mask]
        sol[free] = spla.spsolve(Qii, rhs_eff)
        return sol
    else:
        free = ~fix_mask
        Qii = Q[np.ix_(free, free)]
        rhs_eff = rhs[free] - Q[np.ix_(free, fix_mask)] @ g_dirichlet[fix_mask]
        sol = np.zeros(Q.shape[0])
        sol[fix_mask] = g_dirichlet[fix_mask]
        sol[free] = np.linalg.solve(Qii, rhs_eff)
        return sol

# --- Build grid and Q ---
# Use an anisotropic-friendly box and moderate resolution for speed
xs_c, ys_c, nx_c, ny_c, hx_c, hy_c = build_grid(xmin=-2.5, xmax=2.5, ymin=-1.0, ymax=3.0, hx=0.06, hy=0.06)
rates_c = precompute_rates(xs_c, ys_c, hx_c, hy_c)
Q = assemble_Q(xs_c, ys_c, rates_c)

nx, ny = nx_c, ny_c
N = nx*ny
def k2ij(k): return (k % nx, k // nx)

# --- Define A and B for committor; B for MFPT ---
# A: left set x <= xA; B: right set x >= xB
xA, xB = -0.2, 1.5
A_mask = np.zeros(N, dtype=bool)
B_mask = np.zeros(N, dtype=bool)
for j in range(ny):
    for i in range(nx):
        k = i + j*nx
        if xs_c[i] <= xA: A_mask[k] = True
        if xs_c[i] >= xB: B_mask[k] = True

# --- Solve Committor: Q q = 0; q|A=0, q|B=1 ---
rhs_q = np.zeros(N)
g_dir = np.zeros(N); g_dir[B_mask] = 1.0; fix_mask = A_mask | B_mask
q = solve_dirichlet(Q, rhs_q, fix_mask, g_dir)

# --- Solve MFPT to B: Q m = -1 on Ω\B ; m|B = 0 ---
rhs_m = -np.ones(N)
fix_m = B_mask.copy()
g_m = np.zeros(N)
m = solve_dirichlet(Q, rhs_m, fix_m, g_m)

# --- Visualization: heatmaps ---
Xg, Yg = np.meshgrid(xs_c, ys_c, indexing='xy')
q_grid = q.reshape(ny, nx)
m_grid = m.reshape(ny, nx)


plt.figure(figsize=(6.2,4.8))
plt.imshow(q_grid, origin='lower', extent=[xs_c[0], xs_c[-1], ys_c[0], ys_c[-1]], aspect='auto')
plt.colorbar(label='committor q(x,y)')
CS = plt.contour(Xg, Yg, q_grid, levels=[0.25,0.5,0.75], colors='w', linewidths=1.0)
plt.clabel(CS, inline=True, fontsize=8, fmt='%0.2f')
plt.title('Committor: Q q = 0, q|A=0, q|B=1')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()
plt.savefig('committor_heatmap.jpg', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6.2,4.8))
plt.imshow(m_grid, origin='lower', extent=[xs_c[0], xs_c[-1], ys_c[0], ys_c[-1]], aspect='auto')
plt.colorbar(label='MFPT m(x,y)')
CS2 = plt.contour(Xg, Yg, m_grid, levels=6, colors='w', linewidths=0.9)
plt.clabel(CS2, inline=True, fontsize=8)
plt.title('MFPT: Q m = -1 on Ω\\B, m|B=0')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()
plt.savefig('mfpt_heatmap.jpg', bbox_inches='tight')
plt.show()

# --- Optional sanity: slice along the slow manifold y = x^3/3 - x (project to closest y) ---
def interpolate_on_grid(xval, yval, xs, ys, F):
    # nearest-neighbor sampling for simplicity
    i = int(np.clip(round((xval - xs[0])/(xs[1]-xs[0])), 0, len(xs)-1))
    j = int(np.clip(round((yval - ys[0])/(ys[1]-ys[0])), 0, len(ys)-1))
    return F[j, i]  # note F is [ny, nx]

xx = np.linspace(-2.0, 2.0, 200)
yy = slow_manifold_xnull(xx)
q_slice = np.array([interpolate_on_grid(x, y, xs_c, ys_c, q_grid) for x,y in zip(xx,yy)])
m_slice = np.array([interpolate_on_grid(x, y, xs_c, ys_c, m_grid) for x,y in zip(xx,yy)])

# Recompute and plot slices (variables from previous cell)
plt.figure(figsize=(6.0,4.0))
plt.plot(xx, q_slice, label='q on slow manifold')
plt.axvline(xA, ls='--', lw=1, label='x_A')
plt.axvline(xB, ls='--', lw=1, label='x_B')
plt.xlabel('x (on slow manifold y=x^3/3 - x)'); plt.ylabel('q')
plt.title('Committor along the slow manifold')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.savefig('committor_slice.jpg', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6.0,4.0))
plt.plot(xx, m_slice, label='m on slow manifold')
plt.axvline(xB, ls='--', lw=1, label='x_B')
plt.xlabel('x (on slow manifold y=x^3/3 - x)'); plt.ylabel('m')
plt.title('MFPT to B along the slow manifold')
plt.grid(True, ls=':')
plt.legend()
plt.tight_layout()
plt.savefig('mfpt_slice.jpg', bbox_inches='tight')
plt.show()

# Print representative values again
def interpolate_on_grid(xval, yval, xs, ys, F):
    i = int(np.clip(round((xval - xs[0])/(xs[1]-xs[0])), 0, len(xs)-1))
    j = int(np.clip(round((yval - ys[0])/(ys[1]-ys[0])), 0, len(ys)-1))
    return F[j, i]

x0 = -1.5; y0 = slow_manifold_xnull(x0)
xf = 1.0; yf = slow_manifold_xnull(xf)

q_x0y0 = interpolate_on_grid(x0,y0,xs_c,ys_c,q_grid)
m_x0y0 = interpolate_on_grid(x0,y0,xs_c,ys_c,m_grid)
q_xfyf = interpolate_on_grid(xf,yf,xs_c,ys_c,q_grid)
m_xfyf = interpolate_on_grid(xf,yf,xs_c,ys_c,m_grid)

print("q(x0,y0)=", q_x0y0, "m(x0,y0)=", m_x0y0)
print("q(xf,yf)=", q_xfyf, "m(xf,yf)=", m_xfyf)

print("Saved: committor_heatmap.jpg")
print("Saved: mfpt_heatmap.jpg")
print("Saved: committor_slice.jpg")
print("Saved: mfpt_slice.jpg")
