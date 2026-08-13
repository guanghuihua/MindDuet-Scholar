# Time-discretization (tamed EM & truncated EM) committor/MFPT heatmaps + slow-manifold slices
# Same plotting style as CTRW: heatmaps with contour lines and 1D slices.
# Runtime-conscious: coarse grid + modest Monte Carlo per node.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

# -------------------------
# Model and parameters
# -------------------------
eps = 0.01
a = 1.0
sigma_x = 0.0
sigma_y = 0.08

def drift(x, y):
    fx = (y - (x**3/3.0 - x))  # note: no 1/eps here to match previous CTRW runs where eps was absorbed in y-drift
    fy = eps*(a - x)
    return fx, fy

def diffusion():
    return np.array([sigma_x, sigma_y])

# Domain and sets
xmin, xmax = -2.5, 2.5
ymin, ymax = -1.0, 3.0
# A: x <= xA; B: x >= xB (vertical strips)
xA, xB = -0.2, 1.5

# Grid for heatmaps (coarse for runtime)
nx, ny = 21, 17
xs = np.linspace(xmin, xmax, nx)
ys = np.linspace(ymin, ymax, ny)
hx = xs[1]-xs[0]; hy = ys[1]-ys[0]

def in_A(x, y): return x <= xA
def in_B(x, y): return x >= xB

def reflect_box(x, xmin, xmax):
    if x < xmin:
        return xmin + (xmin - x)
    if x > xmax:
        return xmax - (x - xmax)
    return x

def step_tamed(z, dt):
    x, y = z
    fx, fy = drift(x, y)
    normf = np.hypot(fx, fy)
    denom = 1.0 + dt*normf
    dW = rng.normal(0.0, np.sqrt(dt), size=2) * diffusion()
    xn = x + (dt*fx)/denom + dW[0]
    yn = y + (dt*fy)/denom + dW[1]
    xn = reflect_box(xn, xmin, xmax)
    yn = reflect_box(yn, ymin, ymax)
    return np.array([xn, yn])

def step_truncated(z, dt, R0=2.0):
    x, y = z
    R = R0 * (dt**(-0.25))
    r = np.hypot(x, y)
    scale = min(1.0, R/max(r, 1e-12))
    xt, yt = x*scale, y*scale
    fx, fy = drift(xt, yt)
    dW = rng.normal(0.0, np.sqrt(dt), size=2) * diffusion()
    xn = x + dt*fx + dW[0]
    yn = y + dt*fy + dW[1]
    xn = reflect_box(xn, xmin, xmax)
    yn = reflect_box(yn, ymin, ymax)
    return np.array([xn, yn])

def slow_manifold_y(x):
    return x**3/3.0 - x

# Monte Carlo estimators on the grid
def mc_committor_heatmap(method='tamed', dt=1e-2, T_max=12.0, N=12):
    Q = np.zeros((ny, nx))
    for j, y0 in enumerate(ys):
        for i, x0 in enumerate(xs):
            if in_A(x0,y0):
                Q[j,i] = 0.0
                continue
            if in_B(x0,y0):
                Q[j,i] = 1.0
                continue
            hitsB = 0; count = 0
            for r in range(N):
                z = np.array([x0, y0], float); t = 0.0
                while t < T_max:
                    if in_B(z[0], z[1]):
                        hitsB += 1; count += 1; break
                    if in_A(z[0], z[1]):
                        count += 1; break
                    z = step_tamed(z, dt) if method=='tamed' else step_truncated(z, dt)
                    t += dt
            Q[j,i] = hitsB / max(count,1)
    return Q

def mc_mfpt_heatmap(method='tamed', dt=1e-2, T_max=18.0, N=12):
    M = np.zeros((ny, nx))
    for j, y0 in enumerate(ys):
        for i, x0 in enumerate(xs):
            if in_B(x0,y0):
                M[j,i] = 0.0
                continue
            times = []
            for r in range(N):
                z = np.array([x0, y0], float); t = 0.0
                while t < T_max and not in_B(z[0], z[1]):
                    z = step_tamed(z, dt) if method=='tamed' else step_truncated(z, dt)
                    t += dt
                times.append(t)
            M[j,i] = float(np.mean(times))
    return M

# Compute heatmaps for tamed and truncated
dt_c = 1e-2
Q_tamed  = mc_committor_heatmap('tamed',    dt=dt_c, T_max=12.0, N=12)
Q_trunc  = mc_committor_heatmap('truncated',dt=dt_c, T_max=12.0, N=8)

M_tamed  = mc_mfpt_heatmap('tamed',    dt=dt_c, T_max=18.0, N=12)
M_trunc  = mc_mfpt_heatmap('truncated',dt=dt_c, T_max=18.0, N=8)

# Plot heatmaps (style matched: imshow + contour levels)
extent = [xmin, xmax, ymin, ymax]
Xg, Yg = np.meshgrid(xs, ys, indexing='xy')

def plot_heat_with_contours(F, title, fname, nlevels=6):
    plt.figure(figsize=(6.2,4.8))
    plt.imshow(F, origin='lower', extent=extent, aspect='auto')
    plt.colorbar(label='committor q(x,y)')
    CS = plt.contour(Xg, Yg, F, levels=nlevels, linewidths=0.9)
    plt.clabel(CS, inline=True, fontsize=8)
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"Canard_EM/{fname}", bbox_inches='tight')
    plt.show()

plot_heat_with_contours(Q_tamed, "Committor (tamed-EM MC)", "committor_tamed_heatmap.png")
plot_heat_with_contours(Q_trunc, "Committor (truncated-EM MC)", "committor_trunc_heatmap.png")
plot_heat_with_contours(M_tamed, "MFPT to B (tamed-EM MC)", "mfpt_tamed_heatmap.png")
plot_heat_with_contours(M_trunc, "MFPT to B (truncated-EM MC)", "mfpt_trunc_heatmap.png")

# Slices along the slow manifold y = x^3/3 - x
xx = np.linspace(-2.0, 2.0, 200)
yy = slow_manifold_y(xx)

def sample_grid_nearest(F, x, y):
    i = int(np.clip(np.round((x - xmin)/hx), 0, nx-1))
    j = int(np.clip(np.round((y - ymin)/hy), 0, ny-1))
    return F[j, i]

qt_slice  = np.array([sample_grid_nearest(Q_tamed, xi, yi) for xi, yi in zip(xx,yy)])
qr_slice  = np.array([sample_grid_nearest(Q_trunc, xi, yi) for xi, yi in zip(xx,yy)])
mt_slice  = np.array([sample_grid_nearest(M_tamed, xi, yi) for xi, yi in zip(xx,yy)])
mr_slice  = np.array([sample_grid_nearest(M_trunc, xi, yi) for xi, yi in zip(xx,yy)])

def plot_slice(xx, yvals1, yvals2, title, ylabel, fname):
    plt.figure(figsize=(6.0,4.0))
    plt.plot(xx, yvals1, label='tamed-EM MC')
    plt.plot(xx, yvals2, label='truncated-EM MC')
    plt.axvline(xA, ls='--', lw=1, label='x_A')
    plt.axvline(xB, ls='--', lw=1, label='x_B')
    plt.xlabel('x (on slow manifold y=x^3/3 - x)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Canard_EM/{fname}", bbox_inches='tight')
    plt.show()

plot_slice(xx, qt_slice, qr_slice, "Committor slice (time-MC)", "q", "committor_time_slice.png")
plot_slice(xx, mt_slice, mr_slice, "MFPT slice (time-MC)", "m", "mfpt_time_slice.png")

print("Saved files:")
print("Canard_EM/committor_tamed_heatmap.png")
print("Canard_EM/committor_trunc_heatmap.png")
print("Canard_EM/mfpt_tamed_heatmap.png")
print("Canard_EM/mfpt_trunc_heatmap.png")
print("Canard_EM/committor_time_slice.png")
print("Canard_EM/mfpt_time_slice.png")