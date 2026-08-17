import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================
#  Problem parameters (same as your Monte Carlo code)
# ============================================================
EPS = 0.1
A_PARAM = 1 - EPS/8 - 3*EPS**2/32 - 173*EPS**3/1024 - 0.01
SIGMA_C = EPS**(1.0/3.0)
SIGMA_RATIO = 0.90
SIGMA = SIGMA_RATIO * SIGMA_C

X0 = 1.5
Y0 = X0**3/3 - X0

Y_REF = -0.58
ETA = 0.04
DELTA_HIT = 0.15

T_MAX = 4.0


# ============================================================
#  Utility: positive stable branch x*(y), i.e. solve
#       y = x^3/3 - x   with x > 1
# ============================================================
def x_manifold_positive(y: float):
    """
    Return the positive stable branch x*(y) > 1 solving
        x^3/3 - x - y = 0
    If no such real root exists, return np.nan.
    """
    coeff = [1/3, 0.0, -1.0, -y]
    r = np.roots(coeff)
    rr = r[np.isreal(r)].real
    cand = rr[rr > 1.0]
    if len(cand) == 0:
        return np.nan
    # usually only one candidate > 1 in the relevant y-range
    return float(np.min(cand))


# ============================================================
#  Build target set A on a rectangular grid
# ============================================================
def build_hit_mask(x_grid, y_grid, y_ref=Y_REF, eta=ETA, delta_hit=DELTA_HIT):
    Nx = len(x_grid)
    Ny = len(y_grid)
    hit = np.zeros((Nx, Ny), dtype=bool)

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            xm = x_manifold_positive(y)
            if np.isnan(xm):
                continue
            if abs(y - y_ref) <= eta and abs(x - xm) <= delta_hit:
                hit[i, j] = True
    return hit


# ============================================================
#  Backward PDE solver
# ============================================================
def solve_phit_backward_pde(
    x_min=0.3, x_max=1.8,
    y_min=-1.2, y_max=0.2,
    Nx=241, Ny=281,
    Nt=800,
    verbose=True,
):
    """
    Solve backward PDE:
        u_t + b_x u_x + b_y u_y + (sigma^2/2) u_yy = 0
    backward in time on [0,T_MAX].

    Boundary / terminal conditions:
      - u = 1 on hit set A
      - u = 0 on x <= 1
      - u(T_MAX, x, y) = 0 outside A
      - outer artificial boundaries also set to 0
        (domain should be enlarged until result stabilizes)

    Returns:
      dict with fields:
        x_grid, y_grid, U0, p_hit_ref
    """

    # ----------------------------
    # grids
    # ----------------------------
    x_grid = np.linspace(x_min, x_max, Nx)
    y_grid = np.linspace(y_min, y_max, Ny)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dt = T_MAX / Nt

    if verbose:
        print(f"Nx={Nx}, Ny={Ny}, Nt={Nt}")
        print(f"dx={dx:.6f}, dy={dy:.6f}, dt={dt:.6f}")

    # ----------------------------
    # target set A and absorbing set x<=1
    # ----------------------------
    hit_mask = build_hit_mask(x_grid, y_grid)

    absorb_zero_mask = np.zeros((Nx, Ny), dtype=bool)
    for i, x in enumerate(x_grid):
        if x <= 1.0:
            absorb_zero_mask[i, :] = True

    # artificial outer boundaries -> 0
    outer_zero_mask = np.zeros((Nx, Ny), dtype=bool)
    outer_zero_mask[0, :] = True
    outer_zero_mask[-1, :] = True
    outer_zero_mask[:, 0] = True
    outer_zero_mask[:, -1] = True

    # boundary classification
    # hit states: fixed value 1
    # zero states: fixed value 0
    fixed_one_mask = hit_mask.copy()
    fixed_zero_mask = absorb_zero_mask | outer_zero_mask

    # if a point is both in hit and x<=1, let hit take precedence
    fixed_zero_mask[fixed_one_mask] = False

    interior_mask = ~(fixed_one_mask | fixed_zero_mask)

    # index mapping for unknowns
    unknown_id = -np.ones((Nx, Ny), dtype=int)
    unknown_points = []
    k = 0
    for i in range(Nx):
        for j in range(Ny):
            if interior_mask[i, j]:
                unknown_id[i, j] = k
                unknown_points.append((i, j))
                k += 1
    Nunk = k

    if verbose:
        print(f"unknown states = {Nunk}")
        print(f"hit states     = {fixed_one_mask.sum()}")
        print(f"zero states    = {fixed_zero_mask.sum()}")

    # ----------------------------
    # helper: fixed boundary values g(i,j)
    # ----------------------------
    g_fixed = np.zeros((Nx, Ny), dtype=float)
    g_fixed[fixed_one_mask] = 1.0
    g_fixed[fixed_zero_mask] = 0.0

    # ----------------------------
    # Build spatial operator L on interior:
    #     L u = b_x D_x^up u + b_y D_y^up u + (sigma^2/2) D_yy u
    #
    # We write:
    #     L_II * u_interior + L_IB * g_fixed
    # ----------------------------
    rows = []
    cols = []
    vals = []
    bvec = np.zeros(Nunk, dtype=float)   # this stores L_IB * g_fixed

    sig2_half = 0.5 * SIGMA**2

    for p, (i, j) in enumerate(unknown_points):
        x = x_grid[i]
        y = y_grid[j]

        bx = (y - x**3/3 + x) / EPS
        by = A_PARAM - x

        # start with diagonal contribution
        diag = 0.0

        # ------------------------
        # x-drift: upwind
        # ------------------------
        if bx >= 0:
            # bx * (u_{i,j} - u_{i-1,j}) / dx
            c_self = bx / dx
            c_nb = -bx / dx

            diag += c_self
            ni, nj = i - 1, j

            if interior_mask[ni, nj]:
                rows.append(p); cols.append(unknown_id[ni, nj]); vals.append(c_nb)
            else:
                bvec[p] += c_nb * g_fixed[ni, nj]
        else:
            # bx * (u_{i+1,j} - u_{i,j}) / dx
            c_self = -bx / dx
            c_nb = bx / dx

            diag += c_self
            ni, nj = i + 1, j

            if interior_mask[ni, nj]:
                rows.append(p); cols.append(unknown_id[ni, nj]); vals.append(c_nb)
            else:
                bvec[p] += c_nb * g_fixed[ni, nj]

        # ------------------------
        # y-drift: upwind
        # ------------------------
        if by >= 0:
            # by * (u_{i,j} - u_{i,j-1}) / dy
            c_self = by / dy
            c_nb = -by / dy

            diag += c_self
            ni, nj = i, j - 1

            if interior_mask[ni, nj]:
                rows.append(p); cols.append(unknown_id[ni, nj]); vals.append(c_nb)
            else:
                bvec[p] += c_nb * g_fixed[ni, nj]
        else:
            # by * (u_{i,j+1} - u_{i,j}) / dy
            c_self = -by / dy
            c_nb = by / dy

            diag += c_self
            ni, nj = i, j + 1

            if interior_mask[ni, nj]:
                rows.append(p); cols.append(unknown_id[ni, nj]); vals.append(c_nb)
            else:
                bvec[p] += c_nb * g_fixed[ni, nj]

        # ------------------------
        # y-diffusion: central
        # sig2_half * (u_{j+1} - 2u_j + u_{j-1}) / dy^2
        # ------------------------
        c_up = sig2_half / dy**2
        c_dn = sig2_half / dy**2
        c_self = -2.0 * sig2_half / dy**2

        diag += c_self

        # upper y neighbor
        ni, nj = i, j + 1
        if interior_mask[ni, nj]:
            rows.append(p); cols.append(unknown_id[ni, nj]); vals.append(c_up)
        else:
            bvec[p] += c_up * g_fixed[ni, nj]

        # lower y neighbor
        ni, nj = i, j - 1
        if interior_mask[ni, nj]:
            rows.append(p); cols.append(unknown_id[ni, nj]); vals.append(c_dn)
        else:
            bvec[p] += c_dn * g_fixed[ni, nj]

        # diagonal
        rows.append(p); cols.append(p); vals.append(diag)

    LII = sp.csr_matrix((vals, (rows, cols)), shape=(Nunk, Nunk))

    # backward Euler for:
    #   u_t + L u = 0
    # stepping from t_{n+1} -> t_n:
    #   (I + dt LII) u^n = u^{n+1} - dt * (L_IB g)
    A_mat = sp.eye(Nunk, format="csr") + dt * LII
    rhs_shift = -dt * bvec

    if verbose:
        print("Factorizing sparse matrix ...")
    solver = spla.factorized(A_mat.tocsc())

    # ----------------------------
    # terminal condition at T_MAX
    # u(T,x,y)=0 outside A; u=1 on A
    # ----------------------------
    u_in = np.zeros(Nunk, dtype=float)

    # backward time stepping
    for n in range(Nt):
        rhs = u_in + rhs_shift
        u_in = solver(rhs)

        # clip small numerical overshoot
        u_in = np.clip(u_in, 0.0, 1.0)

        if verbose and (n + 1) % max(1, Nt // 10) == 0:
            print(f"  backward step {n+1}/{Nt}")

    # reconstruct full grid solution at t=0
    U0 = np.zeros((Nx, Ny), dtype=float)
    U0[fixed_one_mask] = 1.0
    U0[fixed_zero_mask] = 0.0
    for p, (i, j) in enumerate(unknown_points):
        U0[i, j] = u_in[p]

    # interpolate at initial point
    p_hit_ref = bilinear_interp(x_grid, y_grid, U0, X0, Y0)

    return dict(
        x_grid=x_grid,
        y_grid=y_grid,
        U0=U0,
        p_hit_ref=float(p_hit_ref),
        hit_mask=hit_mask,
        interior_mask=interior_mask,
    )


# ============================================================
#  Bilinear interpolation
# ============================================================
def bilinear_interp(x_grid, y_grid, U, x, y):
    if x < x_grid[0] or x > x_grid[-1] or y < y_grid[0] or y > y_grid[-1]:
        return np.nan

    i = np.searchsorted(x_grid, x) - 1
    j = np.searchsorted(y_grid, y) - 1

    i = max(0, min(i, len(x_grid) - 2))
    j = max(0, min(j, len(y_grid) - 2))

    x1, x2 = x_grid[i], x_grid[i+1]
    y1, y2 = y_grid[j], y_grid[j+1]

    q11 = U[i,   j]
    q21 = U[i+1, j]
    q12 = U[i,   j+1]
    q22 = U[i+1, j+1]

    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    return ((1-tx)*(1-ty)*q11 +
            tx*(1-ty)*q21 +
            (1-tx)*ty*q12 +
            tx*ty*q22)


# ============================================================
#  Optional plot
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    out = solve_phit_backward_pde(
        x_min=0.3, x_max=1.8,
        y_min=-1.2, y_max=0.2,
        Nx=241, Ny=281,
        Nt=800,
        verbose=True,
    )

    print("\nReference P_hit from backward PDE:")
    print(f"P_hit_ref = {out['p_hit_ref']:.8f}")

    xg = out["x_grid"]
    yg = out["y_grid"]
    U0 = out["U0"]

    X, Y = np.meshgrid(yg, xg)

    plt.figure(figsize=(8, 5))
    im = plt.pcolormesh(yg, xg, U0, shading="auto", cmap="viridis")
    plt.colorbar(im, label="u(0,x,y)")
    plt.contour(yg, xg, out["hit_mask"].astype(float), levels=[0.5], colors="red", linewidths=1.5)
    plt.scatter([Y0], [X0], c="white", edgecolors="black", s=60, zorder=3, label="initial point")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title("Backward PDE reference solution for $P_{hit}$")
    plt.legend()
    plt.tight_layout()
    plt.show()