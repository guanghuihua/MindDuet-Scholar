import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_Q_hybrid_Qu_x_Qc_v(Nx, Nv, Lx, Lv, gamma=1.0, sigma_phys=1.0, beta=1.0):
    """
    Underdamped Langevin on [-Lx,Lx]x[-Lv,Lv]:
        dX = V dt
        dV = -(U'(X)+gamma V) dt + sigma_phys dW
    with U(x)=x^2/2 (so U'(x)=x).

    Hybrid SSA-realizable generator:
      - x-direction (no noise): Qu (upwind) for v*∂_x
      - v-direction (noise):   Qc (exponential rates) for μ_v ∂_v + (sigma^2/2) ∂_vv

    Consistency with Eric's notation:
      Eric uses dY = μ dt + sqrt(2)*σ dW and Lf = Df^T μ + trace(D^2 f σσ^T).
      Our noise is sigma_phys dW, so take σ_eric = sigma_phys/sqrt(2) which makes
      diffusion term (sigma_phys^2/2) ∂_vv.

    Then, for uniform v-grid spacing hv, the 1D Qc jump rates become:
        D = sigma_phys^2/2
        r_+ = (D/hv^2) * exp( + hv * μ_v / sigma_phys^2 )
        r_- = (D/hv^2) * exp( - hv * μ_v / sigma_phys^2 )
    """
    x = np.linspace(-Lx, Lx, Nx)
    v = np.linspace(-Lv, Lv, Nv)
    hx = x[1] - x[0]
    hv = v[1] - v[0]

    def Up(xx):
        return xx  # U'(x) for U=x^2/2

    D = 0.5 * sigma_phys**2  # diffusion coefficient in v

    n = Nx * Nv
    rows, cols, data = [], [], []
    def add(i, j, val):
        rows.append(i); cols.append(j); data.append(val)

    for ix in range(Nx):
        Upx = Up(x[ix])
        for jv in range(Nv):
            k = ix * Nv + jv
            rate_sum = 0.0

            # ---- x-direction: Qu upwind for v * d/dx ----
            a = v[jv]
            if a > 0 and ix < Nx - 1:
                kk = (ix + 1) * Nv + jv
                r = a / hx
                add(k, kk, r); rate_sum += r
            elif a < 0 and ix > 0:
                kk = (ix - 1) * Nv + jv
                r = (-a) / hx
                add(k, kk, r); rate_sum += r

            # ---- v-direction: Qc exponential rates for μ_v ∂_v + D ∂_vv ----
            mu_v = -(Upx + gamma * v[jv])
            theta = (hv * mu_v) / (sigma_phys**2 + 1e-300)
            base = D / (hv * hv)

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
    """Solve Q^T p = 0 with sum(p)=1."""
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
    """Truncated Gibbs stationary density for U(x)=x^2/2: ρ ∝ exp(-β(U+v^2/2))."""
    X, V = np.meshgrid(x, v, indexing="ij")
    U = 0.5 * X**2
    rho = np.exp(-beta * (U + 0.5 * V**2))
    mass = rho * hx * hv
    mass /= mass.sum()
    return mass.reshape(-1)

def run_convergence(N_list, Lx=6.0, Lv=6.0, gamma=1.0, beta=1.0):
    # fluctuation-dissipation for underdamped Langevin: sigma^2 = 2 gamma / beta
    sigma_phys = np.sqrt(2.0 * gamma / beta)

    hs, errs = [], []
    for N in N_list:
        Q, x, v, hx, hv = build_Q_hybrid_Qu_x_Qc_v(
            N, N, Lx, Lv, gamma=gamma, sigma_phys=sigma_phys, beta=beta
        )
        p_num = stationary_distribution(Q)
        p_true = true_stationary_mass(x, v, hx, hv, beta=beta)
        errs.append(np.sum(np.abs(p_num - p_true)))
        hs.append(max(hx, hv))

    hs = np.array(hs, dtype=float)
    errs = np.array(errs, dtype=float)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    return hs, errs, slope

# -------------------- experiment --------------------
N_list = [21, 31, 61,101, 501]   # refine grid
hs, errs, slope = run_convergence(N_list, Lx=6.0, Lv=6.0, gamma=1.0, beta=1.0)

# slope reference lines anchored at finest grid
h_ref = hs[-1]
e_ref = errs[-1]
ref1 = e_ref * (hs / h_ref)**1
ref2 = e_ref * (hs / h_ref)**2

plt.figure(figsize=(6.8, 5.0))
plt.loglog(hs, errs, marker='o', linestyle='-',
           label=f'Hybrid: Qu in x, Qc in v (slope≈{slope:.3f})')
plt.loglog(hs, ref1, linestyle='--', label='slope 1 reference')
plt.loglog(hs, ref2, linestyle='--', label='slope 2 reference')
plt.gca().invert_xaxis()
plt.xlabel('grid spacing h')
plt.ylabel('L1 error of stationary probability mass')
plt.title('Underdamped Langevin: SSA-realizable hybrid generator')
plt.legend()
plt.tight_layout()
plt.show()

print("N, h, L1_error")
for N, h, e in zip(N_list, hs, errs):
    print(f"{N:>3d}, {h:.6f}, {e:.6e}")
print("\nObserved slope (log-log fit):", slope)
