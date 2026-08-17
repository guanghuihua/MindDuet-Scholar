import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except Exception as e:
    SCIPY_AVAILABLE = False
    SCIPY_ERR = str(e)

def B_bernoulli(x):
    """
    Bernoulli function B(x) = x/(exp(x)-1), with B(0)=1.
    Stable for small |x| via series.
    """
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-6
    # series: x/(exp(x)-1) = 1 - x/2 + x^2/12 - x^4/720 + ...
    xs = x[small]
    out[small] = 1.0 - xs/2.0 + (xs*xs)/12.0
    xl = x[~small]
    out[~small] = xl / (np.expm1(xl))
    return out

def build_Q_baseline_upwind(Nx, Nv, Lx, Lv, gamma=1.0, sigma=np.sqrt(2.0)):
    """
    Baseline mixed generator:
      x-transport upwind (1st)
      v-drift upwind (1st) + v-diffusion symmetric (2nd for diffusion)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy sparse is required for the sizes used here.")
    x = np.linspace(-Lx, Lx, Nx)
    v = np.linspace(-Lv, Lv, Nv)
    hx = x[1] - x[0]
    hv = v[1] - v[0]
    D = 0.5 * sigma**2
    
    def Up(xx):  # U'(x) for U(x)=x^2/2
        return xx
    
    n = Nx * Nv
    rows, cols, data = [], [], []
    def add(i,j,val):
        rows.append(i); cols.append(j); data.append(val)
    
    for ix in range(Nx):
        for jv in range(Nv):
            k = ix * Nv + jv
            rate_sum = 0.0
            
            # x-transport: a=v
            a = v[jv]
            if a > 0 and ix < Nx-1:
                kk = (ix+1)*Nv + jv
                r = a / hx
                add(k, kk, r); rate_sum += r
            elif a < 0 and ix > 0:
                kk = (ix-1)*Nv + jv
                r = (-a) / hx
                add(k, kk, r); rate_sum += r
            
            # v-drift: b=-(U'(x)+gamma v)
            b = -(Up(x[ix]) + gamma*v[jv])
            if b > 0 and jv < Nv-1:
                kk = ix*Nv + (jv+1)
                r = b / hv
                add(k, kk, r); rate_sum += r
            elif b < 0 and jv > 0:
                kk = ix*Nv + (jv-1)
                r = (-b) / hv
                add(k, kk, r); rate_sum += r
            
            # v-diffusion: symmetric jumps
            rdiff = D / (hv*hv)
            if jv < Nv-1:
                kk = ix*Nv + (jv+1)
                add(k, kk, rdiff); rate_sum += rdiff
            if jv > 0:
                kk = ix*Nv + (jv-1)
                add(k, kk, rdiff); rate_sum += rdiff
            
            add(k, k, -rate_sum)
    
    Q = sp.coo_matrix((data, (rows, cols)), shape=(n,n)).tocsr()
    return Q, x, v, hx, hv

def build_Q_sg_v(Nx, Nv, Lx, Lv, gamma=1.0, sigma=np.sqrt(2.0)):
    """
    Improved mixed generator:
      x-transport: upwind (1st, realizable)
      v-direction: Scharfetter-Gummel / Chang-Cooper type rates for drift-diffusion
                  (equilibrium-preserving; typically much closer to 2nd order for invariant measure)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy sparse is required for the sizes used here.")
    x = np.linspace(-Lx, Lx, Nx)
    v = np.linspace(-Lv, Lv, Nv)
    hx = x[1] - x[0]
    hv = v[1] - v[0]
    D = 0.5 * sigma**2
    
    def Up(xx):
        return xx
    
    n = Nx * Nv
    rows, cols, data = [], [], []
    def add(i,j,val):
        rows.append(i); cols.append(j); data.append(val)
    
    # precompute b(x_i, v_j)
    # b = -(U'(x)+gamma v)
    Bx = -Up(x)  # = -x
    for ix in range(Nx):
        # drift values at v nodes
        b_nodes = Bx[ix] - gamma * v  # since b=-(x+gamma v) = -x - gamma v
        for jv in range(Nv):
            k = ix * Nv + jv
            rate_sum = 0.0
            
            # x-transport
            a = v[jv]
            if a > 0 and ix < Nx-1:
                kk = (ix+1)*Nv + jv
                r = a / hx
                add(k, kk, r); rate_sum += r
            elif a < 0 and ix > 0:
                kk = (ix-1)*Nv + jv
                r = (-a) / hx
                add(k, kk, r); rate_sum += r
            
            # v drift-diffusion via SG rates
            # rate to j+1 uses midpoint b_{j+1/2}
            if jv < Nv-1:
                b_half = 0.5*(b_nodes[jv] + b_nodes[jv+1])
                xi = b_half * hv / D
                r_plus = (D/(hv*hv)) * B_bernoulli(xi)
                kk = ix*Nv + (jv+1)
                add(k, kk, r_plus); rate_sum += r_plus
            if jv > 0:
                b_half = 0.5*(b_nodes[jv-1] + b_nodes[jv])
                xi = b_half * hv / D
                # rate to j-1 corresponds to B(-xi)
                r_minus = (D/(hv*hv)) * B_bernoulli(-xi)
                kk = ix*Nv + (jv-1)
                add(k, kk, r_minus); rate_sum += r_minus
            
            add(k, k, -rate_sum)
    
    Q = sp.coo_matrix((data, (rows, cols)), shape=(n,n)).tocsr()
    return Q, x, v, hx, hv

def stationary_distribution(Q):
    """
    Solve Q^T p = 0 with sum(p)=1.
    """
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
    rho = np.exp(-beta*(U + 0.5*V**2))
    mass = rho * hx * hv
    mass /= mass.sum()
    return mass.reshape(-1)

def compute_errors_vs_hv(Nx_fixed=201, Nv_list=(21,31,41,61,81), Lx=6.0, Lv=6.0, gamma=1.0, beta=1.0):
    sigma = np.sqrt(2.0*gamma/beta)
    hv_list = []
    err_base = []
    err_sg = []
    
    for Nv in Nv_list:
        # baseline
        Qb, x, v, hx, hv = build_Q_baseline_upwind(Nx_fixed, Nv, Lx, Lv, gamma=gamma, sigma=sigma)
        pb = stationary_distribution(Qb)
        ptr = true_stationary_mass(x, v, hx, hv, beta=beta)
        err_base.append(np.sum(np.abs(pb - ptr)))
        
        # improved SG in v
        Qs, x2, v2, hx2, hv2 = build_Q_sg_v(Nx_fixed, Nv, Lx, Lv, gamma=gamma, sigma=sigma)
        ps = stationary_distribution(Qs)
        ptr2 = true_stationary_mass(x2, v2, hx2, hv2, beta=beta)
        err_sg.append(np.sum(np.abs(ps - ptr2)))
        
        hv_list.append(hv)
    
    hv_list = np.array(hv_list, dtype=float)
    err_base = np.array(err_base, dtype=float)
    err_sg = np.array(err_sg, dtype=float)
    
    slope_base = np.polyfit(np.log(hv_list), np.log(err_base), 1)[0]
    slope_sg = np.polyfit(np.log(hv_list), np.log(err_sg), 1)[0]
    return hv_list, err_base, err_sg, slope_base, slope_sg

# ---- experiment ----
Nx_fixed = 201
Nv_list = [21, 31, 41, 61, 81]
hv, e_base, e_sg, s_base, s_sg = compute_errors_vs_hv(Nx_fixed=Nx_fixed, Nv_list=Nv_list, Lx=6.0, Lv=6.0, gamma=1.0, beta=1.0)

# reference lines anchored at smallest hv point
h_ref = hv[-1]
ref_anchor_base = e_base[-1]
ref_anchor_sg = e_sg[-1]
ref1_base = ref_anchor_base * (hv/h_ref)**1
ref2_base = ref_anchor_base * (hv/h_ref)**2
ref1_sg = ref_anchor_sg * (hv/h_ref)**1
ref2_sg = ref_anchor_sg * (hv/h_ref)**2

plt.figure(figsize=(7.0, 5.2))
plt.loglog(hv, e_base, marker='o', linestyle='-', label=f'Baseline (upwind drift) slope≈{s_base:.2f}')
plt.loglog(hv, e_sg, marker='s', linestyle='-', label=f'Improved v (Scharfetter–Gummel) slope≈{s_sg:.2f}')
plt.loglog(hv, ref1_sg, linestyle='--', label='slope 1 reference')
plt.loglog(hv, ref2_sg, linestyle='--', label='slope 2 reference')
plt.gca().invert_xaxis()
plt.xlabel('v grid spacing h_v (x grid fixed fine)')
plt.ylabel('L1 error of stationary probability mass')
plt.title('Underdamped Langevin: closer-to-2nd-order via equilibrium-preserving v discretization')
plt.legend()
plt.tight_layout()
plt.show()

print("Nx fixed =", Nx_fixed)
print("Nv, h_v, err_baseline, err_SG")
for Nv, h, eb, es in zip(Nv_list, hv, e_base, e_sg):
    print(f"{Nv:>3d}, {h:.6f}, {eb:.6e}, {es:.6e}")
print("\nObserved slopes (log-log fit in h_v):")
print("  baseline:", s_base)
print("  SG-v    :", s_sg)
