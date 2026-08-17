import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_Q_hybrid_Qu_x_Qc_v(Nx, Nv, Lx, Lv, gamma=1.0, beta=1.0, theta_clip=50.0):
    sigma_phys = np.sqrt(2.0 * gamma / beta)
    D = 0.5 * sigma_phys**2  # = gamma/beta

    x = np.linspace(-Lx, Lx, Nx)
    v = np.linspace(-Lv, Lv, Nv)
    hx = x[1] - x[0]
    hv = v[1] - v[0]

    def Up(xx): return xx

    n = Nx * Nv
    rows, cols, data = [], [], []
    def add(i, j, val):
        rows.append(i); cols.append(j); data.append(val)

    base = D / (hv * hv)

    for ix in range(Nx):
        Upx = Up(x[ix])
        for jv in range(Nv):
            k = ix * Nv + jv
            rate_sum = 0.0

            # x: upwind Qu
            a = v[jv]
            if a > 0 and ix < Nx - 1:
                kk = (ix + 1) * Nv + jv
                r = a / hx
                add(k, kk, r); rate_sum += r
            elif a < 0 and ix > 0:
                kk = (ix - 1) * Nv + jv
                r = (-a) / hx
                add(k, kk, r); rate_sum += r

            # v: Qc exponential rates
            mu_v = -(Upx + gamma * v[jv])
            theta = (hv * mu_v) / (sigma_phys**2 + 1e-300)
            theta = np.clip(theta, -theta_clip, theta_clip)

            if jv < Nv - 1:
                r_plus = base * np.exp(+theta)
                kk = ix * Nv + (jv + 1)
                add(k, kk, r_plus); rate_sum += r_plus
            if jv > 0:
                r_minus = base * np.exp(-theta)
                kk = ix * Nv + (jv - 1)
                add(k, kk, r_minus); rate_sum += r_minus

            add(k, k, -rate_sum)

    Q = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return Q, x, v, hx, hv

def stationary_distribution(Q):
    n = Q.shape[0]
    A = Q.T.tolil()
    b = np.zeros(n)
    A[-1, :] = 1.0
    b[-1] = 1.0
    p = spla.spsolve(A.tocsr(), b)
    p = np.maximum(p, 0.0)
    p /= p.sum()
    return p

def true_stationary_mass(x, v, hx, hv, beta=1.0):
    X, V = np.meshgrid(x, v, indexing="ij")
    U = 0.5 * X**2
    rho = np.exp(-beta * (U + 0.5 * V**2))
    m = rho * hx * hv
    m /= m.sum()
    return m.reshape(-1)

def interior_mask(x, v, hx, hv, Lx, Lv, margin_factor=2.0):
    mx = margin_factor * hx
    mv = margin_factor * hv
    x_ok = np.abs(x) <= (Lx - mx + 1e-15)
    v_ok = np.abs(v) <= (Lv - mv + 1e-15)
    return (x_ok[:, None] & v_ok[None, :]).reshape(-1)

def L1_error_on_interior(p_num, p_true, mask):
    return np.sum(np.abs(p_num[mask] - p_true[mask]))

def fit_slope(hs, errs):
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]

# --- v refinement with much finer x grid to suppress x-error floor ---
Lx = Lv = 6.0
gamma = 1.0
beta = 1.0

Nx_fine = 801
Nv_list = [21, 31, 41, 61, 81, 101,201]

hv_list = []
err_v = []

for Nv in Nv_list:
    Q, x, v, hx, hv = build_Q_hybrid_Qu_x_Qc_v(Nx_fine, Nv, Lx, Lv, gamma=gamma, beta=beta)
    p_num = stationary_distribution(Q)
    p_true = true_stationary_mass(x, v, hx, hv, beta=beta)
    mask = interior_mask(x, v, hx, hv, Lx, Lv, margin_factor=2.0)
    hv_list.append(hv)
    err_v.append(L1_error_on_interior(p_num, p_true, mask))

hv_list = np.array(hv_list, dtype=float)
err_v = np.array(err_v, dtype=float)
slope_v = fit_slope(hv_list, err_v)

# reference lines anchored at finest hv
h_ref = hv_list[-1]
e_ref = err_v[-1]
ref1 = e_ref * (hv_list / h_ref) ** 1
ref2 = e_ref * (hv_list / h_ref) ** 2

plt.figure(figsize=(6.8, 5.0))
plt.loglog(hv_list, err_v, marker='s', linestyle='-', label=f'Refine v only (Nx={Nx_fine}), slope≈{slope_v:.3f}')
plt.loglog(hv_list, ref1, linestyle='--', label='slope 1 reference')
plt.loglog(hv_list, ref2, linestyle='--', label='slope 2 reference')
plt.gca().invert_xaxis()
plt.xlabel('v grid spacing h_v')
plt.ylabel('Interior L1 error of stationary mass')
plt.title('SSA hybrid: Qu in x, Qc in v  (refine v only, x very fine)')
plt.legend()
plt.tight_layout()
plt.show()

print("Refine v only (Nx fixed very fine):")
print("Nv, hv, interior_L1_error")
for Nv, hv, e in zip(Nv_list, hv_list, err_v):
    print(f"{Nv:>3d}, {hv:.6f}, {e:.6e}")
print("Observed slope in hv:", slope_v)

