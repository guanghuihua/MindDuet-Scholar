import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math

# ------------------------
# problem setup
# ------------------------
# Underdamped Langevin (hypoelliptic):
# dX = V dt
# dV = -U'(X) dt - gamma V dt + sigma dW
# We take U(x)=x^2/2 so U'(x)=x, and sigma^2 = 2 gamma / beta (FDR).
def true_mass(x, v, hx, hv, beta=1.0):
    X, V = np.meshgrid(x, v, indexing="ij")
    U = 0.5 * X**2
    rho = np.exp(-beta * (U + 0.5 * V**2))
    m = rho * hx * hv
    m /= m.sum()
    return m.reshape(-1)

def stationary_from_Q(Q):
    # Q has row-sum zero; stationary mass p solves Q^T p = 0 with sum(p)=1
    n = Q.shape[0]
    A = Q.T.tolil()
    b = np.zeros(n)
    A[-1, :] = 1.0
    b[-1] = 1.0
    p = spla.spsolve(A.tocsr(), b)
    p = np.maximum(p, 0.0)
    p /= p.sum()
    return p

def stationary_from_A(A):
    # Solve A rho = 0 with sum(rho)=1 (A is forward FP discretization matrix)
    n = A.shape[0]
    M = A.tolil()
    b = np.zeros(n)
    M[-1, :] = 1.0
    b[-1] = 1.0
    rho = spla.spsolve(M.tocsr(), b)
    rho = np.maximum(rho, 0.0)
    rho /= rho.sum()
    return rho

# ------------------------
# (1) SSA-realizable baseline generator Q
#     upwind in x and (x,v)-drift, symmetric v-diffusion
# ------------------------
def build_Q_baseline(Nx, Nv, Lx, Lv, gamma=1.0, beta=1.0):
    sigma = math.sqrt(2 * gamma / beta)
    D = 0.5 * sigma**2

    x = np.linspace(-Lx, Lx, Nx)
    v = np.linspace(-Lv, Lv, Nv)
    hx = x[1] - x[0]
    hv = v[1] - v[0]

    def Up(xx):  # U'(x)
        return xx

    n = Nx * Nv
    rows, cols, data = [], [], []

    def add(i, j, val):
        rows.append(i); cols.append(j); data.append(val)

    for ix in range(Nx):
        for jv in range(Nv):
            k = ix * Nv + jv
            rate_sum = 0.0

            # x-transport: v d/dx (upwind)
            a = v[jv]
            if a > 0 and ix < Nx - 1:
                kk = (ix + 1) * Nv + jv
                r = a / hx
                add(k, kk, r); rate_sum += r
            elif a < 0 and ix > 0:
                kk = (ix - 1) * Nv + jv
                r = (-a) / hx
                add(k, kk, r); rate_sum += r

            # v-drift: -(U'(x)+gamma v) d/dv (upwind)
            b = -(Up(x[ix]) + gamma * v[jv])
            if b > 0 and jv < Nv - 1:
                kk = ix * Nv + (jv + 1)
                r = b / hv
                add(k, kk, r); rate_sum += r
            elif b < 0 and jv > 0:
                kk = ix * Nv + (jv - 1)
                r = (-b) / hv
                add(k, kk, r); rate_sum += r

            # v-diffusion: symmetric jumps
            rdiff = D / (hv * hv)
            if jv < Nv - 1:
                kk = ix * Nv + (jv + 1)
                add(k, kk, rdiff); rate_sum += rdiff
            if jv > 0:
                kk = ix * Nv + (jv - 1)
                add(k, kk, rdiff); rate_sum += rdiff

            add(k, k, -rate_sum)

    Q = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return Q, x, v, hx, hv

# ------------------------
# (2) SSA-realizable hybrid: keep Hamiltonian transport upwind,
#     but discretize OU in v by 2nd-order divergence form
#     OU operator: -gamma v d/dv + D d^2/dv^2
# ------------------------
def build_Q_hybrid_ou(Nx, Nv, Lx, Lv, gamma=1.0, beta=1.0):
    sigma = math.sqrt(2 * gamma / beta)
    D = 0.5 * sigma**2  # = gamma/beta

    x = np.linspace(-Lx, Lx, Nx)
    v = np.linspace(-Lv, Lv, Nv)
    hx = x[1] - x[0]
    hv = v[1] - v[0]

    def Up(xx):
        return xx

    n = Nx * Nv
    rows, cols, data = [], [], []

    def add(i, j, val):
        rows.append(i); cols.append(j); data.append(val)

    for ix in range(Nx):
        Upx = Up(x[ix])
        for jv in range(Nv):
            k = ix * Nv + jv
            rate_sum = 0.0

            # (Hamiltonian) x-transport: v d/dx (upwind)
            a = v[jv]
            if a > 0 and ix < Nx - 1:
                kk = (ix + 1) * Nv + jv
                r = a / hx
                add(k, kk, r); rate_sum += r
            elif a < 0 and ix > 0:
                kk = (ix - 1) * Nv + jv
                r = (-a) / hx
                add(k, kk, r); rate_sum += r

            # (Hamiltonian) force transport in v: (-U'(x)) d/dv (upwind)
            c = -Upx
            if c > 0 and jv < Nv - 1:
                kk = ix * Nv + (jv + 1)
                r = c / hv
                add(k, kk, r); rate_sum += r
            elif c < 0 and jv > 0:
                kk = ix * Nv + (jv - 1)
                r = (-c) / hv
                add(k, kk, r); rate_sum += r

            # OU part in v in divergence form (2nd-order in v; exact discrete Maxwellian for OU):
            # L_OU f = D e^{β v^2/2} ∂_v( e^{-β v^2/2} ∂_v f )
            if jv < Nv - 1:
                vhalf = 0.5 * (v[jv] + v[jv + 1])
                w = math.exp(0.5 * beta * v[jv]**2 - 0.5 * beta * vhalf**2)
                r = (D / (hv * hv)) * w
                kk = ix * Nv + (jv + 1)
                add(k, kk, r); rate_sum += r
            if jv > 0:
                vhalf = 0.5 * (v[jv - 1] + v[jv])
                w = math.exp(0.5 * beta * v[jv]**2 - 0.5 * beta * vhalf**2)
                r = (D / (hv * hv)) * w
                kk = ix * Nv + (jv - 1)
                add(k, kk, r); rate_sum += r

            add(k, k, -rate_sum)

    Q = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return Q, x, v, hx, hv

# ------------------------
# (3) 2nd-order stationary Fokker–Planck discretization (NOT SSA-realizable)
#     0 = -∂_x(v ρ) + ∂_v((U'(x)+γ v) ρ) + D ∂_vv ρ
#     Use 2nd-order centered conservative fluxes, and no-flux at boundaries.
# ------------------------
def build_A_fp_second_order(Nx, Nv, Lx, Lv, gamma=1.0, beta=1.0):
    sigma = math.sqrt(2 * gamma / beta)
    D = 0.5 * sigma**2

    x = np.linspace(-Lx, Lx, Nx)
    v = np.linspace(-Lv, Lv, Nv)
    hx = x[1] - x[0]
    hv = v[1] - v[0]

    def Up(xx):
        return xx

    n = Nx * Nv
    rows, cols, data = [], [], []

    def add(r, c, val):
        rows.append(r); cols.append(c); data.append(val)

    def idx(ix, jv):
        return ix * Nv + jv

    for ix in range(Nx):
        Upx = Up(x[ix])
        for jv in range(Nv):
            k = idx(ix, jv)
            vv = v[jv]

            # x-flux: Jx = v * rho, centered at faces
            if ix < Nx - 1:
                coef = -(1.0 / hx) * vv * 0.5
                add(k, idx(ix, jv), coef)
                add(k, idx(ix + 1, jv), coef)
            if ix > 0:
                coef = +(1.0 / hx) * vv * 0.5
                add(k, idx(ix, jv), coef)
                add(k, idx(ix - 1, jv), coef)

            # v-flux: Jv = -b rho - D d_v rho, b = U'(x) + gamma v
            if jv < Nv - 1:
                vhalf = 0.5 * (v[jv] + v[jv + 1])
                bhalf = Upx + gamma * vhalf
                c_j  = (-bhalf * 0.5 + D / hv)
                c_jp = (-bhalf * 0.5 - D / hv)
                coef = -(1.0 / hv)
                add(k, idx(ix, jv),   coef * c_j)
                add(k, idx(ix, jv + 1), coef * c_jp)

            if jv > 0:
                vhalf = 0.5 * (v[jv - 1] + v[jv])
                bhalf = Upx + gamma * vhalf
                c_jm = (-bhalf * 0.5 + D / hv)
                c_j  = (-bhalf * 0.5 - D / hv)
                coef = +(1.0 / hv)
                add(k, idx(ix, jv - 1), coef * c_jm)
                add(k, idx(ix, jv),     coef * c_j)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return A, x, v, hx, hv

# ------------------------
# run convergence
# ------------------------
N_list = [21, 31, 41, 61]
L = 6.0
gamma = 1.0
beta = 1.0

hs = []
err_base = []
err_hyb = []
err_fp2 = []

for N in N_list:
    Qb, x, v, hx, hv = build_Q_baseline(N, N, L, L, gamma=gamma, beta=beta)
    Qh, _, _, _, _   = build_Q_hybrid_ou(N, N, L, L, gamma=gamma, beta=beta)
    A2, x2, v2, hx2, hv2 = build_A_fp_second_order(N, N, L, L, gamma=gamma, beta=beta)

    pb = stationary_from_Q(Qb)
    ph = stationary_from_Q(Qh)
    p2 = stationary_from_A(A2)

    ptr = true_mass(x, v, hx, hv, beta=beta)

    hs.append(hx)
    err_base.append(np.sum(np.abs(pb - ptr)))
    err_hyb.append(np.sum(np.abs(ph - ptr)))
    err_fp2.append(np.sum(np.abs(p2 - ptr)))

hs = np.array(hs)
err_base = np.array(err_base)
err_hyb  = np.array(err_hyb)
err_fp2  = np.array(err_fp2)

s_base = np.polyfit(np.log(hs), np.log(err_base), 1)[0]
s_hyb  = np.polyfit(np.log(hs), np.log(err_hyb),  1)[0]
s_fp2  = np.polyfit(np.log(hs), np.log(err_fp2),  1)[0]

# reference slope lines anchored at finest grid
h_ref = hs[-1]
e_ref = err_fp2[-1]
ref1 = e_ref * (hs / h_ref)**1
ref2 = e_ref * (hs / h_ref)**2

plt.figure(figsize=(7.2, 5.2))
plt.loglog(hs, err_base, marker='o', linestyle='-', label=f'SSA baseline slope≈{s_base:.2f}')
plt.loglog(hs, err_hyb,  marker='s', linestyle='-', label=f'SSA hybrid (better OU) slope≈{s_hyb:.2f}')
plt.loglog(hs, err_fp2,  marker='^', linestyle='-', label=f'2nd-order FP slope≈{s_fp2:.2f}')
plt.loglog(hs, ref1, linestyle='--', label='slope 1 reference')
plt.loglog(hs, ref2, linestyle='--', label='slope 2 reference')
plt.gca().invert_xaxis()
plt.xlabel('grid spacing h (uniform in x and v)')
plt.ylabel('L1 error of stationary probability mass')
plt.title('Underdamped Langevin: making the mixed scheme closer to 2nd order')
plt.legend()
plt.tight_layout()
plt.show()

print("N, h, err_baseline, err_hybridOU, err_FP2")
for N, h, eb, eh, ef in zip(N_list, hs, err_base, err_hyb, err_fp2):
    print(f"{N:>3d}, {h:.6f}, {eb:.6e}, {eh:.6e}, {ef:.6e}")
print("\nObserved slopes (log-log fit):")
print("  baseline :", s_base)
print("  hybridOU :", s_hyb)
print("  FP2      :", s_fp2)