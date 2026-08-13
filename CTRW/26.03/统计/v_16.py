"""
Stochastic Canard: matched-work comparison of SSA hy=0.10 vs EM dt=eps/100.

Both methods have nearly equal EM-equivalent work per path (~370-390 steps),
so the comparison is fair at the flop level.

Key fix vs previous versions:
    Instead of   'if prev_x > 1.0 >= x'  (fragile under numba fastmath reordering),
    we use        'if was_above and x <= 1.0'  (flag set once, compiler-safe).

Four statistics:
    1. y_jump distribution  : y when path first crosses x=1 (saddle node)
    2. x_error profile      : x(path) - x*(y) at y-checkpoints along manifold
    3. tau_hit distribution : time to reach window A
    4. RMSE of P_hit vs N   : statistical efficiency curve
"""

from __future__ import annotations
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from pathlib import Path


# ---------------------------------------------------------------------------
#  System parameters
# ---------------------------------------------------------------------------
EPS     = 0.1
A_PARAM = 1 - EPS/8 - 3*EPS**2/32 - 173*EPS**3/1024 - 0.01
SIGMA_C = EPS**(1.0/3.0)
SIGMA   = 0.7 * SIGMA_C

X0, Y0  = 1.5, 1.5**3/3 - 1.5

Y_REF, ETA, DELTA_HIT = -0.60, 0.04, 0.15
Y_CHECKPOINTS = np.array([-0.40, -0.45, -0.50, -0.55, -0.60])

T_MAX = 4.0
SPAN  = 6.0


# ---------------------------------------------------------------------------
#  Slow manifold lookup table
# ---------------------------------------------------------------------------
def _build_manifold():
    ys = np.linspace(-0.665, -0.30, 3000)
    xs = np.empty_like(ys)
    for i, y in enumerate(ys):
        r  = np.roots([1/3, 0, -1, -y])
        rr = r[np.isreal(r)].real
        v  = rr[rr > 1.0]
        xs[i] = float(v[0]) if len(v) > 0 else np.nan
    return ys, xs

_Y_TAB, _X_TAB = _build_manifold()


@nb.njit(cache=True)
def x_manifold(y):
    """Linear interpolation on precomputed positive stable manifold."""
    if y < _Y_TAB[0] or y > _Y_TAB[-1]:
        return np.nan
    n   = len(_Y_TAB)
    idx = (y - _Y_TAB[0]) / (_Y_TAB[-1] - _Y_TAB[0]) * (n - 1)
    i   = int(idx)
    if i >= n - 1:
        return _X_TAB[-1]
    f = idx - i
    return _X_TAB[i] * (1.0 - f) + _X_TAB[i+1] * f


# ---------------------------------------------------------------------------
#  EM single path
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def _em_path(dt, seed, y_checkpoints):
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(T_MAX / dt)
    n_cp    = len(y_checkpoints)

    x, y    = X0, Y0
    prev_y  = y
    was_above = True     # x starts at 1.5 > 1.0

    y_jump    = np.nan
    tau_hit   = np.nan
    tau_window = np.nan

    x_err   = np.full(n_cp, np.nan)
    cp_done = np.zeros(n_cp, dtype=nb.boolean)

    for k in range(n_steps):
        dW  = np.random.randn() * sqrt_dt
        x  += (y - x**3/3 + x) / EPS * dt
        y  += (A_PARAM - x) * dt + SIGMA * dW
        t   = (k + 1) * dt

        # x=1 crossing: first time x drops to/below 1.0
        if was_above and x <= 1.0:
            y_jump    = y
            tau_hit   = t
            was_above = False

        # y-checkpoints (y decreasing)
        for ci in range(n_cp):
            if (not cp_done[ci]) and prev_y > y_checkpoints[ci] >= y:
                xm = x_manifold(y)
                if not np.isnan(xm):
                    x_err[ci] = x - xm
                    cp_done[ci] = True

        # window hit
        xm = x_manifold(y)
        if (not np.isnan(xm)) and np.isnan(tau_window):
            if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                tau_window = t

        prev_y = y
        if x < 0.3:
            break

    return y_jump, tau_hit, tau_window, x_err


@nb.njit(parallel=True, cache=True)
def em_full_ensemble(dt, n_paths, y_checkpoints):
    n_cp    = len(y_checkpoints)
    y_jumps = np.full(n_paths, np.nan)
    t_hits  = np.full(n_paths, np.nan)
    t_wins  = np.full(n_paths, np.nan)
    x_errs  = np.full((n_paths, n_cp), np.nan)

    for i in nb.prange(n_paths):
        yj, th, tw, xe = _em_path(dt, i, y_checkpoints)
        y_jumps[i] = yj
        t_hits[i]  = th
        t_wins[i]  = tw
        x_errs[i]  = xe

    return y_jumps, t_hits, t_wins, x_errs


# ---------------------------------------------------------------------------
#  SSA single path
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def _ssa_path(h_y, seed, y_checkpoints):
    np.random.seed(seed)
    sig2   = SIGMA * SIGMA
    n_cp   = len(y_checkpoints)

    n_y = int(SPAN / h_y)
    h_y = SPAN / n_y        # exact
    n_x = n_y * n_y         # n_x = n_y^2
    h_x = SPAN / n_x

    x, y   = X0, Y0
    prev_y = y
    t      = 0.0
    was_above = True         # x starts at 1.5 > 1.0

    y_jump    = np.nan
    tau_hit   = np.nan
    tau_window = np.nan

    x_err   = np.full(n_cp, np.nan)
    cp_done = np.zeros(n_cp, dtype=nb.boolean)

    while t < T_MAX:
        mu_x = (y - x**3/3 + x) / EPS
        mu_y = A_PARAM - x
        m_y  = 0.5 * max(sig2 - abs(mu_y) * h_y, 0.0)

        q_xp = max(mu_x,  0.0) / h_x
        q_xm = max(-mu_x, 0.0) / h_x
        q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y)
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        lam  = q_xp + q_xm + q_yp + q_ym

        tau   = -np.log(1.0 - np.random.random()) / lam
        t    += tau

        # jump
        r = np.random.random() * lam
        if r < q_xp:
            x += h_x
        elif r < q_xp + q_xm:
            x -= h_x
        elif r < q_xp + q_xm + q_yp:
            y += h_y
        else:
            y -= h_y

        # x=1 crossing: first time x drops to/below 1.0
        if was_above and x <= 1.0:
            y_jump    = y
            tau_hit   = t
            was_above = False

        # y-checkpoints
        for ci in range(n_cp):
            if (not cp_done[ci]) and prev_y > y_checkpoints[ci] >= y:
                xm = x_manifold(y)
                if not np.isnan(xm):
                    x_err[ci] = x - xm
                    cp_done[ci] = True

        # window hit
        xm = x_manifold(y)
        if (not np.isnan(xm)) and np.isnan(tau_window):
            if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                tau_window = t

        prev_y = y
        if x < 0.3:
            break

    return y_jump, tau_hit, tau_window, x_err


@nb.njit(parallel=True, cache=True)
def ssa_full_ensemble(h_y, n_paths, y_checkpoints):
    n_cp    = len(y_checkpoints)
    y_jumps = np.full(n_paths, np.nan)
    t_hits  = np.full(n_paths, np.nan)
    t_wins  = np.full(n_paths, np.nan)
    x_errs  = np.full((n_paths, n_cp), np.nan)

    for i in nb.prange(n_paths):
        yj, th, tw, xe = _ssa_path(h_y, i + 5000, y_checkpoints)
        y_jumps[i] = yj
        t_hits[i]  = th
        t_wins[i]  = tw
        x_errs[i]  = xe

    return y_jumps, t_hits, t_wins, x_errs


# ---------------------------------------------------------------------------
#  P_hit batch (for RMSE curve)
# ---------------------------------------------------------------------------
@nb.njit(parallel=True, cache=True)
def em_phit_batch(dt, n_paths):
    hits = np.zeros(n_paths, dtype=nb.boolean)
    for i in nb.prange(n_paths):
        np.random.seed(i)
        sqrt_dt = np.sqrt(dt)
        ns = int(T_MAX / dt)
        x, y = X0, Y0
        hit = False
        for _ in range(ns):
            x += (y - x**3/3 + x) / EPS * dt
            y += (A_PARAM - x) * dt + SIGMA * np.random.randn() * sqrt_dt
            xm = x_manifold(y)
            if not np.isnan(xm):
                if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                    hit = True
                    break
            if x < 0.3:
                break
        hits[i] = hit
    return hits


@nb.njit(parallel=True, cache=True)
def ssa_phit_batch(h_y, n_paths):
    hits = np.zeros(n_paths, dtype=nb.boolean)
    for i in nb.prange(n_paths):
        np.random.seed(i + 5000)
        sig2 = SIGMA * SIGMA
        n_y  = int(SPAN / h_y); hy2 = SPAN / n_y
        n_x  = n_y * n_y;       hx2 = SPAN / n_x
        x, y = X0, Y0; t = 0.0; hit = False
        while t < T_MAX:
            mu_x = (y - x**3/3 + x) / EPS; mu_y = A_PARAM - x
            m_y  = 0.5 * max(sig2 - abs(mu_y) * hy2, 0.0)
            q_xp = max(mu_x,  0.0) / hx2
            q_xm = max(-mu_x, 0.0) / hx2
            q_yp = max(mu_y,  0.0) / hy2 + m_y / (hy2 * hy2)
            q_ym = max(-mu_y, 0.0) / hy2 + m_y / (hy2 * hy2)
            lam  = q_xp + q_xm + q_yp + q_ym
            t   += -np.log(1.0 - np.random.random()) / lam
            r    = np.random.random() * lam
            if r < q_xp:              x += hx2
            elif r < q_xp + q_xm:    x -= hx2
            elif r < q_xp+q_xm+q_yp: y += hy2
            else:                     y -= hy2
            xm = x_manifold(y)
            if not np.isnan(xm):
                if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                    hit = True; break
            if x < 0.3: break
        hits[i] = hit
    return hits


# ---------------------------------------------------------------------------
#  RMSE of P_hit vs N
# ---------------------------------------------------------------------------
def rmse_vs_n(hits_full, n_list, n_boot=50):
    p_true = float(np.mean(hits_full))
    rmse   = np.empty(len(n_list))
    rng    = np.random.default_rng(0)
    for j, n in enumerate(n_list):
        ests = np.array([np.mean(hits_full[rng.choice(len(hits_full),
                         size=int(n), replace=False)]) for _ in range(n_boot)])
        rmse[j] = float(np.sqrt(np.mean((ests - p_true)**2)))
    return rmse


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def plot_all(yj_em, th_em, xe_em,
             yj_ssa, th_ssa, xe_ssa,
             hits_em_all, hits_ssa_all,
             n_em, n_ssa, out_dir):

    y_cp = Y_CHECKPOINTS
    n_cp = len(y_cp)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)
    ax_yj   = fig.add_subplot(gs[0, 0])
    ax_th   = fig.add_subplot(gs[0, 1])
    ax_rmse = fig.add_subplot(gs[0, 2])
    ax_xm   = fig.add_subplot(gs[1, :])
    ax_xs   = fig.add_subplot(gs[2, :])

    em_c, ssa_c = "#1f77b4", "#2ca02c"

    # (1) y_jump histogram
    yj_em_v  = yj_em[~np.isnan(yj_em)]
    yj_ssa_v = yj_ssa[~np.isnan(yj_ssa)]
    if len(yj_em_v) > 0 and len(yj_ssa_v) > 0:
        bins = np.linspace(min(yj_em_v.min(), yj_ssa_v.min()) - 0.02,
                           max(yj_em_v.max(), yj_ssa_v.max()) + 0.02, 40)
        ax_yj.hist(yj_em_v,  bins=bins, density=True, alpha=0.55,
                   color=em_c,  label=f"EM  (N={n_em})")
        ax_yj.hist(yj_ssa_v, bins=bins, density=True, alpha=0.55,
                   color=ssa_c, label=f"SSA (N={n_ssa})")
    ax_yj.axvline(-2/3, color="k", ls=":", lw=1.3, label=r"$y_c=-2/3$")
    em_m  = np.nanmean(yj_em);  em_s  = np.nanstd(yj_em)
    ssa_m = np.nanmean(yj_ssa); ssa_s = np.nanstd(yj_ssa)
    ax_yj.set_xlabel(r"$y_\mathrm{jump}$", fontsize=11)
    ax_yj.set_ylabel("Density", fontsize=11)
    ax_yj.set_title(
        f"(1) Jump-location distribution\n"
        f"EM  {em_m:.4f} +/- {em_s:.4f}\n"
        f"SSA {ssa_m:.4f} +/- {ssa_s:.4f}", fontsize=10)
    ax_yj.legend(fontsize=8); ax_yj.grid(True, alpha=0.25)

    # (2) tau_hit histogram
    th_em_v  = th_em[~np.isnan(th_em)]
    th_ssa_v = th_ssa[~np.isnan(th_ssa)]
    if len(th_em_v) == 0:  th_em_v  = np.array([T_MAX])
    if len(th_ssa_v) == 0: th_ssa_v = np.array([T_MAX])
    vmax_t = min(float(max(th_em_v.max(), th_ssa_v.max())), T_MAX)
    bins_t = np.linspace(0, vmax_t, 40)
    ax_th.hist(th_em_v,  bins=bins_t, density=True, alpha=0.55,
               color=em_c,  label="EM")
    ax_th.hist(th_ssa_v, bins=bins_t, density=True, alpha=0.55,
               color=ssa_c, label="SSA")
    ax_th.set_xlabel("Passage time to x=1", fontsize=11)
    ax_th.set_ylabel("Density", fontsize=11)
    ax_th.set_title(
        f"(2) Passage time distribution\n"
        f"EM  {np.mean(th_em_v):.4f} +/- {np.std(th_em_v):.4f}\n"
        f"SSA {np.mean(th_ssa_v):.4f} +/- {np.std(th_ssa_v):.4f}", fontsize=10)
    ax_th.legend(fontsize=8); ax_th.grid(True, alpha=0.25)

    # (3) RMSE vs N
    n_list = np.unique(np.logspace(
        np.log10(20),
        np.log10(min(len(hits_em_all), len(hits_ssa_all))),
        25).astype(int))
    rmse_em  = rmse_vs_n(hits_em_all,  n_list)
    rmse_ssa = rmse_vs_n(hits_ssa_all, n_list)
    p_ref = float(np.mean(hits_em_all))
    ref   = np.sqrt(p_ref * (1 - p_ref) / n_list)
    ax_rmse.loglog(n_list, rmse_em,  "o-", color=em_c,  lw=2, ms=6, label="EM")
    ax_rmse.loglog(n_list, rmse_ssa, "^-", color=ssa_c, lw=2, ms=6, label="SSA")
    ax_rmse.loglog(n_list, ref,      "k--", lw=1.2, label=r"$1/\sqrt{N}$")
    ax_rmse.set_xlabel("N paths", fontsize=11)
    ax_rmse.set_ylabel("RMSE of P_hit", fontsize=11)
    ax_rmse.set_title("(3) Statistical efficiency\n(RMSE vs number of paths)", fontsize=10)
    ax_rmse.legend(fontsize=8); ax_rmse.grid(True, which="both", alpha=0.25)

    # (4) x_error mean profile
    em_mean  = np.array([np.nanmean(xe_em[:, ci])  for ci in range(n_cp)])
    ssa_mean = np.array([np.nanmean(xe_ssa[:, ci]) for ci in range(n_cp)])
    n_em_cp  = np.array([np.sum(~np.isnan(xe_em[:, ci]))  for ci in range(n_cp)])
    n_ssa_cp = np.array([np.sum(~np.isnan(xe_ssa[:, ci])) for ci in range(n_cp)])
    em_se    = np.array([np.nanstd(xe_em[:, ci])  / max(np.sqrt(n_em_cp[ci]),  1)
                         for ci in range(n_cp)])
    ssa_se   = np.array([np.nanstd(xe_ssa[:, ci]) / max(np.sqrt(n_ssa_cp[ci]), 1)
                         for ci in range(n_cp)])

    ax_xm.errorbar(y_cp, em_mean,  yerr=2*em_se,  fmt="o-", color=em_c,
                   lw=2, ms=6, capsize=4, label="EM  mean(x - x*(y))")
    ax_xm.errorbar(y_cp, ssa_mean, yerr=2*ssa_se, fmt="^-", color=ssa_c,
                   lw=2, ms=6, capsize=4, label="SSA mean(x - x*(y))")
    ax_xm.axhline(0, color="k", ls=":", lw=1.2, label="exact manifold")
    ax_xm.set_xlabel("y checkpoint", fontsize=11)
    ax_xm.set_ylabel("mean(x - x*(y))  [bias]", fontsize=11)
    ax_xm.set_title(
        "(4) Manifold tracking bias\n"
        "Positive = path outside manifold   |   grows as y approaches saddle",
        fontsize=10)
    ax_xm.invert_xaxis()
    ax_xm.legend(fontsize=9); ax_xm.grid(True, alpha=0.25)

    # (5) x_error std profile
    em_std  = np.array([np.nanstd(xe_em[:, ci])  for ci in range(n_cp)])
    ssa_std = np.array([np.nanstd(xe_ssa[:, ci]) for ci in range(n_cp)])
    ax_xs.plot(y_cp, em_std,  "o-", color=em_c,  lw=2, ms=6, label="EM  std(x - x*(y))")
    ax_xs.plot(y_cp, ssa_std, "^-", color=ssa_c, lw=2, ms=6, label="SSA std(x - x*(y))")
    ax_xs.set_xlabel("y checkpoint", fontsize=11)
    ax_xs.set_ylabel("std(x - x*(y))  [spread]", fontsize=11)
    ax_xs.set_title(
        "(5) Manifold tracking spread\n"
        "Spread grows as path approaches saddle (y -> -2/3)",
        fontsize=10)
    ax_xs.invert_xaxis()
    ax_xs.legend(fontsize=9); ax_xs.grid(True, alpha=0.25)

    fig.suptitle(
        f"Matched-work comparison: SSA hy=0.10 vs EM dt=eps/100\n"
        f"eps={EPS},  sigma/sigma_c=0.70,  "
        f"EM equiv_work~392,  SSA equiv_work~367",
        fontsize=12)

    out_png = out_dir / "canard_matched_comparison.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


# ---------------------------------------------------------------------------
#  Summary table
# ---------------------------------------------------------------------------
def print_summary(yj_em, th_em, xe_em, yj_ssa, th_ssa, xe_ssa,
                  hits_em, hits_ssa, n_em, n_ssa):
    print()
    print("=" * 70)
    print(f"{'Statistic':38s} {'EM  dt=eps/100':>15} {'SSA hy=0.10':>13}")
    print("-" * 70)
    print(f"{'N paths':38s} {n_em:>15d} {n_ssa:>13d}")
    print(f"{'P_hit':38s} {float(np.mean(hits_em)):>15.4f} {float(np.mean(hits_ssa)):>13.4f}")
    print(f"{'P_hit SE':38s} "
          f"{float(np.std(hits_em.astype(float))/np.sqrt(n_em)):>15.4f} "
          f"{float(np.std(hits_ssa.astype(float))/np.sqrt(n_ssa)):>13.4f}")
    print()
    print(f"{'y_jump mean':38s} {np.nanmean(yj_em):>15.4f} {np.nanmean(yj_ssa):>13.4f}")
    print(f"{'y_jump std':38s} {np.nanstd(yj_em):>15.4f} {np.nanstd(yj_ssa):>13.4f}")
    print()
    print(f"{'tau_hit mean':38s} {np.nanmean(th_em):>15.4f} {np.nanmean(th_ssa):>13.4f}")
    print(f"{'tau_hit std':38s} {np.nanstd(th_em):>15.4f} {np.nanstd(th_ssa):>13.4f}")
    print()
    for ci, yc in enumerate(Y_CHECKPOINTS):
        em_v  = xe_em[:, ci][~np.isnan(xe_em[:, ci])]
        ssa_v = xe_ssa[:, ci][~np.isnan(xe_ssa[:, ci])]
        em_m  = np.mean(em_v)  if len(em_v)  > 0 else np.nan
        ssa_m = np.mean(ssa_v) if len(ssa_v) > 0 else np.nan
        em_s  = np.std(em_v)   if len(em_v)  > 0 else np.nan
        ssa_s = np.std(ssa_v)  if len(ssa_v) > 0 else np.nan
        print(f"{'x_err mean @ y='+f'{yc:.2f}':38s} {em_m:>+15.4f} {ssa_m:>+13.4f}")
        print(f"{'x_err std  @ y='+f'{yc:.2f}':38s} {em_s:>15.4f} {ssa_s:>13.4f}")
    print("=" * 70)

    # Winner per statistic
    print()
    stats = [
        ("y_jump std",    np.nanstd(yj_em),  np.nanstd(yj_ssa)),
        ("tau_hit std",   np.nanstd(th_em),  np.nanstd(th_ssa)),
        ("x_error bias",
         np.nanmean(np.abs([np.nanmean(xe_em[:, ci])
                             for ci in range(len(Y_CHECKPOINTS))])),
         np.nanmean(np.abs([np.nanmean(xe_ssa[:, ci])
                             for ci in range(len(Y_CHECKPOINTS))]))),
        ("x_error spread",
         np.nanmean([np.nanstd(xe_em[:, ci])
                     for ci in range(len(Y_CHECKPOINTS))]),
         np.nanmean([np.nanstd(xe_ssa[:, ci])
                     for ci in range(len(Y_CHECKPOINTS))])),
    ]
    print("Assessment (lower is better for std/bias/spread):")
    for name, em_v, ssa_v in stats:
        if np.isnan(em_v) or np.isnan(ssa_v):
            print(f"  {name}: inconclusive (nan)")
        elif em_v < ssa_v:
            print(f"  {name}: EM wins  ({em_v:.4f} < {ssa_v:.4f})")
        else:
            print(f"  {name}: SSA wins ({ssa_v:.4f} < {em_v:.4f})")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parent

    print(f"EPS={EPS},  SIGMA_C={SIGMA_C:.4f},  sigma={SIGMA:.4f} (0.70*sigma_c)")
    print(f"Start: ({X0}, {Y0:.4f}),  saddle: y_c={-2/3:.4f}")
    print(f"Window A: y in [{Y_REF-ETA:.3f},{Y_REF+ETA:.3f}], "
          f"|x-x*(y)|<={DELTA_HIT}")
    print()

    # JIT warm-up
    print("Compiling...")
    ycp = Y_CHECKPOINTS
    em_full_ensemble(EPS/100, 4, ycp)
    ssa_full_ensemble(0.10, 4, ycp)
    em_phit_batch(EPS/100, 4)
    ssa_phit_batch(0.10, 4)
    print("Done.\n")

    N_EM = N_SSA = 800

    print(f"Running EM  dt=eps/100  N={N_EM}...")
    t0 = time.perf_counter()
    yj_em, th_em, tw_em, xe_em = em_full_ensemble(EPS/100, N_EM, ycp)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print(f"Running SSA hy=0.10     N={N_SSA}...")
    t0 = time.perf_counter()
    yj_ssa, th_ssa, tw_ssa, xe_ssa = ssa_full_ensemble(0.10, N_SSA, ycp)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Diagnostics
    n_th_em  = int(np.sum(~np.isnan(th_em)))
    n_th_ssa = int(np.sum(~np.isnan(th_ssa)))
    n_yj_em  = int(np.sum(~np.isnan(yj_em)))
    n_yj_ssa = int(np.sum(~np.isnan(yj_ssa)))
    print(f"  tau_hit valid: EM={n_th_em}/{N_EM}, SSA={n_th_ssa}/{N_SSA}")
    print(f"  y_jump  valid: EM={n_yj_em}/{N_EM}, SSA={n_yj_ssa}/{N_SSA}")
    if n_th_em == 0 or n_th_ssa == 0:
        print("  WARNING: still getting nan. Delete __pycache__ and rerun.")

    hits_em  = ~np.isnan(tw_em)
    hits_ssa = ~np.isnan(tw_ssa)

    # RMSE curve
    N_LARGE = 4000
    print(f"\nRunning large batch for RMSE curve (N={N_LARGE} each)...")
    t0 = time.perf_counter()
    hits_em_all  = em_phit_batch(EPS/100, N_LARGE)
    hits_ssa_all = ssa_phit_batch(0.10,   N_LARGE)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print_summary(yj_em, th_em, xe_em, yj_ssa, th_ssa, xe_ssa,
                  hits_em, hits_ssa, N_EM, N_SSA)

    plot_all(yj_em, th_em, xe_em,
             yj_ssa, th_ssa, xe_ssa,
             hits_em_all, hits_ssa_all,
             N_EM, N_SSA, out_dir)


if __name__ == "__main__":
    main()