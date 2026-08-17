"""
Stochastic Canard: matched-work comparison of SSA hy=0.10 vs EM dt=eps/100.

Both methods have nearly equal EM-equivalent work per path (~370-390),
so the comparison is fair at the flop level.

Four statistics compared:
    1. y_jump distribution  : y value when path first crosses x=1
                               mean + std + histogram
    2. x_error profile       : x(path) - x*(y) at y-checkpoints along manifold
                               mean (bias) + std (spread) as function of y
    3. tau_hit distribution  : time to reach window A
                               mean + std + histogram
    4. RMSE of P_hit vs N    : statistical efficiency curve
                               how quickly does the P_hit estimate converge?

System:
    eps * dx = (y - x^3/3 + x) dt,   dy = (a-x) dt + sigma dW
    sigma = 0.7 * sigma_c  (subcritical, Canard preserved)
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

# Window A
Y_REF, ETA, DELTA_HIT = -0.60, 0.04, 0.15

# Checkpoints along the slow manifold (y decreasing from Y0 toward saddle)
Y_CHECKPOINTS = np.array([-0.40, -0.45, -0.50, -0.55, -0.60])

T_MAX = 4.0
SPAN  = 6.0

# ---------------------------------------------------------------------------
#  Slow manifold table
# ---------------------------------------------------------------------------
def _build_manifold():
    ys = np.linspace(-0.665, -0.30, 3000)
    xs = np.empty_like(ys)
    for i, y in enumerate(ys):
        r = np.roots([1/3, 0, -1, -y])
        rr = r[np.isreal(r)].real
        v  = rr[rr > 1.0]
        xs[i] = float(v[0]) if len(v) > 0 else np.nan
    return ys, xs

_Y_TAB, _X_TAB = _build_manifold()

@nb.njit(fastmath=True, cache=True)
def x_manifold(y):
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
#  EM single path: collect full statistics
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True, cache=True)
def _em_path_full(dt, seed, y_checkpoints):
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(T_MAX / dt)
    n_cp    = len(y_checkpoints)
    eps, a  = EPS, A_PARAM

    x, y   = X0, Y0
    prev_x = x
    prev_y = y

    y_jump    = np.nan
    tau_hit   = np.nan
    tau_window = np.nan

    # x_error at each checkpoint (one recording per path)
    x_err = np.full(n_cp, np.nan)
    cp_done = np.zeros(n_cp, dtype=nb.boolean)

    for k in range(n_steps):
        dW  = np.random.randn() * sqrt_dt
        x  += (y - x**3/3 + x) / eps * dt
        y  += (a - x) * dt + SIGMA * dW
        t   = (k + 1) * dt

        # checkpoint crossings
        for ci in range(n_cp):
            yc = y_checkpoints[ci]
            if (not cp_done[ci]) and prev_y > yc >= y:
                xm = x_manifold(y)
                if not np.isnan(xm):
                    x_err[ci] = x - xm
                    cp_done[ci] = True

        # window hit
        xm = x_manifold(y)
        if (not np.isnan(xm)) and np.isnan(tau_window):
            if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                tau_window = t

        # x=1 crossing
        if prev_x > 1.0 >= x and np.isnan(y_jump):
            y_jump  = y
            tau_hit = t

        prev_x = x
        prev_y = y
        if x < 0.3:
            break

    return y_jump, tau_hit, tau_window, x_err


@nb.njit(fastmath=True, parallel=True, cache=True)
def em_full_ensemble(dt, n_paths, y_checkpoints):
    n_cp    = len(y_checkpoints)
    y_jumps = np.full(n_paths, np.nan)
    t_hits  = np.full(n_paths, np.nan)
    t_wins  = np.full(n_paths, np.nan)
    x_errs  = np.full((n_paths, n_cp), np.nan)

    for i in nb.prange(n_paths):
        yj, th, tw, xe = _em_path_full(dt, i, y_checkpoints)
        y_jumps[i] = yj
        t_hits[i]  = th
        t_wins[i]  = tw
        x_errs[i]  = xe

    return y_jumps, t_hits, t_wins, x_errs


# ---------------------------------------------------------------------------
#  SSA single path: collect full statistics
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True, cache=True)
def _ssa_path_full(h_y, seed, y_checkpoints):
    np.random.seed(seed)
    sig2 = SIGMA * SIGMA
    eps, a = EPS, A_PARAM
    n_cp = len(y_checkpoints)

    n_y = int(SPAN / h_y)
    h_y = SPAN / n_y
    n_x = n_y * n_y
    h_x = SPAN / n_x

    x, y   = X0, Y0
    prev_x = x
    prev_y = y
    t      = 0.0

    y_jump    = np.nan
    tau_hit   = np.nan
    tau_window = np.nan

    x_err   = np.full(n_cp, np.nan)
    cp_done = np.zeros(n_cp, dtype=nb.boolean)

    while t < T_MAX:
        mu_x = (y - x**3/3 + x) / eps
        mu_y = a - x
        m_y  = 0.5 * max(sig2 - abs(mu_y) * h_y, 0.0)

        q_xp = max(mu_x,  0.0) / h_x
        q_xm = max(-mu_x, 0.0) / h_x
        q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y)
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        lam  = q_xp + q_xm + q_yp + q_ym

        tau   = -np.log(1.0 - np.random.random()) / lam
        t    += tau

        # jump first, then record all crossings with updated coordinates
        r      = np.random.random() * lam
        prev_x = x
        prev_y = y
        if r < q_xp:
            x += h_x
        elif r < q_xp + q_xm:
            x -= h_x
        elif r < q_xp + q_xm + q_yp:
            y += h_y
        else:
            y -= h_y

        # checkpoint crossings (y decreased through checkpoint)
        for ci in range(n_cp):
            yc = y_checkpoints[ci]
            if (not cp_done[ci]) and prev_y > yc >= y:
                xm = x_manifold(y)
                if not np.isnan(xm):
                    x_err[ci] = x - xm
                    cp_done[ci] = True

        # window hit
        xm = x_manifold(y)
        if (not np.isnan(xm)) and np.isnan(tau_window):
            if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                tau_window = t

        # x=1 crossing: prev_x was before jump, x is after
        if prev_x > 1.0 >= x and np.isnan(y_jump):
            y_jump  = y
            tau_hit = t

        if x < 0.3:
            break

    return y_jump, tau_hit, tau_window, x_err


@nb.njit(fastmath=True, parallel=True, cache=True)
def ssa_full_ensemble(h_y, n_paths, y_checkpoints):
    n_cp    = len(y_checkpoints)
    y_jumps = np.full(n_paths, np.nan)
    t_hits  = np.full(n_paths, np.nan)
    t_wins  = np.full(n_paths, np.nan)
    x_errs  = np.full((n_paths, n_cp), np.nan)

    for i in nb.prange(n_paths):
        yj, th, tw, xe = _ssa_path_full(h_y, i + 5000, y_checkpoints)
        y_jumps[i] = yj
        t_hits[i]  = th
        t_wins[i]  = tw
        x_errs[i]  = xe

    return y_jumps, t_hits, t_wins, x_errs


# ---------------------------------------------------------------------------
#  RMSE of P_hit vs N  (statistical efficiency)
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True, parallel=True, cache=True)
def em_phit_batch(dt, n_paths):
    """Return hit/miss array for RMSE analysis."""
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
            if x < 0.3 or x <= 1.0:
                break
        hits[i] = hit
    return hits


@nb.njit(fastmath=True, parallel=True, cache=True)
def ssa_phit_batch(h_y, n_paths):
    hits = np.zeros(n_paths, dtype=nb.boolean)
    for i in nb.prange(n_paths):
        np.random.seed(i + 5000)
        sig2 = SIGMA * SIGMA
        n_y  = int(SPAN / h_y); h_y2 = SPAN / n_y
        n_x  = n_y * n_y;       h_x  = SPAN / n_x
        x, y = X0, Y0; t = 0.0; hit = False
        while t < T_MAX:
            mu_x = (y - x**3/3 + x) / EPS; mu_y = A_PARAM - x
            m_y  = 0.5 * max(sig2 - abs(mu_y) * h_y2, 0.0)
            q_xp = max(mu_x,0)/h_x; q_xm = max(-mu_x,0)/h_x
            q_yp = max(mu_y,0)/h_y2 + m_y/h_y2**2
            q_ym = max(-mu_y,0)/h_y2 + m_y/h_y2**2
            lam  = q_xp+q_xm+q_yp+q_ym
            t   += -np.log(1-np.random.random())/lam
            # jump first, then check
            px = x
            r = np.random.random()*lam
            if r<q_xp: x+=h_x
            elif r<q_xp+q_xm: x-=h_x
            elif r<q_xp+q_xm+q_yp: y+=h_y2
            else: y-=h_y2
            xm = x_manifold(y)
            if not np.isnan(xm):
                if abs(y-Y_REF)<=ETA and abs(x-xm)<=DELTA_HIT:
                    hit=True; break
            if x<0.3 or px>1.0>=x: break
        hits[i] = hit
    return hits


def rmse_vs_n(hits_full: np.ndarray, n_list: np.ndarray) -> np.ndarray:
    """
    Given a large array of hit/miss outcomes, estimate P_hit using the first
    n samples for each n in n_list. Repeat with 30 bootstrap replications to
    get RMSE relative to the full-sample estimate.
    """
    p_true = float(np.mean(hits_full))
    rmse   = np.empty(len(n_list))
    rng    = np.random.default_rng(0)
    n_boot = 50

    for j, n in enumerate(n_list):
        estimates = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(len(hits_full), size=int(n), replace=False)
            estimates[b] = np.mean(hits_full[idx])
        rmse[j] = float(np.sqrt(np.mean((estimates - p_true)**2)))

    return rmse


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def plot_all(
    yj_em, th_em, xe_em,
    yj_ssa, th_ssa, xe_ssa,
    hits_em_all, hits_ssa_all,
    n_em, n_ssa,
    out_dir: Path,
):
    y_cp  = Y_CHECKPOINTS
    n_cp  = len(y_cp)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    ax_yj   = fig.add_subplot(gs[0, 0])   # y_jump histogram
    ax_th   = fig.add_subplot(gs[0, 1])   # tau_hit histogram
    ax_rmse = fig.add_subplot(gs[0, 2])   # RMSE vs N
    ax_xm   = fig.add_subplot(gs[1, :])   # x_error mean profile
    ax_xs   = fig.add_subplot(gs[2, :])   # x_error std profile

    em_kw  = dict(color="#1f77b4", alpha=0.55, density=True)
    ssa_kw = dict(color="#2ca02c", alpha=0.55, density=True)
    em_lkw = dict(color="#1f77b4", lw=2.0, marker="o", ms=6)
    ssa_lkw= dict(color="#2ca02c", lw=2.0, marker="^", ms=6)

    # (1) y_jump histogram
    _yj_em_v  = yj_em[~np.isnan(yj_em)]
    _yj_ssa_v = yj_ssa[~np.isnan(yj_ssa)]
    if len(_yj_em_v)  == 0: _yj_em_v  = np.array([-2/3])
    if len(_yj_ssa_v) == 0: _yj_ssa_v = np.array([-2/3])
    vmin = min(float(_yj_em_v.min()), float(_yj_ssa_v.min()))
    vmax = max(float(_yj_em_v.max()), float(_yj_ssa_v.max()))
    bins = np.linspace(vmin - 0.02, vmax + 0.02, 40)

    ax_yj.hist(yj_em[~np.isnan(yj_em)],  bins=bins, label=f"EM  (N={n_em})",  **em_kw)
    ax_yj.hist(yj_ssa[~np.isnan(yj_ssa)], bins=bins, label=f"SSA (N={n_ssa})", **ssa_kw)
    ax_yj.axvline(-2/3, color="k", ls=":", lw=1.3, label=r"$y_c=-2/3$")

    em_m,  em_s  = np.nanmean(yj_em),  np.nanstd(yj_em)
    ssa_m, ssa_s = np.nanmean(yj_ssa), np.nanstd(yj_ssa)
    ax_yj.set_xlabel(r"$y_\mathrm{jump}$",  fontsize=11)
    ax_yj.set_ylabel("Density", fontsize=11)
    ax_yj.set_title(
        f"(1) Jump-location distribution\n"
        f"EM  {em_m:.4f} +/- {em_s:.4f}\n"
        f"SSA {ssa_m:.4f} +/- {ssa_s:.4f}",
        fontsize=10)
    ax_yj.legend(fontsize=8)
    ax_yj.grid(True, alpha=0.25)

    # (2) tau_hit histogram
    th_em_v  = th_em[~np.isnan(th_em)]
    th_ssa_v = th_ssa[~np.isnan(th_ssa)]
    # guard against empty arrays (all paths timed out)
    if len(th_em_v)  == 0: th_em_v  = np.array([T_MAX])
    if len(th_ssa_v) == 0: th_ssa_v = np.array([T_MAX])
    vmax_t   = min(float(max(th_em_v.max(), th_ssa_v.max())), T_MAX)
    bins_t   = np.linspace(0, vmax_t, 40)

    ax_th.hist(th_em_v,  bins=bins_t, label=f"EM",  **em_kw)
    ax_th.hist(th_ssa_v, bins=bins_t, label=f"SSA", **ssa_kw)
    ax_th.set_xlabel("Passage time to x=1", fontsize=11)
    ax_th.set_ylabel("Density", fontsize=11)
    ax_th.set_title(
        f"(2) Passage time distribution\n"
        f"EM  {np.mean(th_em_v):.4f} +/- {np.std(th_em_v):.4f}\n"
        f"SSA {np.mean(th_ssa_v):.4f} +/- {np.std(th_ssa_v):.4f}",
        fontsize=10)
    ax_th.legend(fontsize=8)
    ax_th.grid(True, alpha=0.25)

    # (3) RMSE of P_hit vs N
    n_list = np.unique(np.logspace(
        np.log10(20), np.log10(min(len(hits_em_all), len(hits_ssa_all))),
        25).astype(int))

    rmse_em  = rmse_vs_n(hits_em_all,  n_list)
    rmse_ssa = rmse_vs_n(hits_ssa_all, n_list)

    ax_rmse.loglog(n_list, rmse_em,  **em_lkw,  label="EM")
    ax_rmse.loglog(n_list, rmse_ssa, **ssa_lkw, label="SSA")

    # theoretical 1/sqrt(N) reference
    p_ref   = float(np.mean(hits_em_all))
    ref     = np.sqrt(p_ref * (1 - p_ref) / n_list)
    ax_rmse.loglog(n_list, ref, "k--", lw=1.2, label=r"$1/\sqrt{N}$ ref")

    ax_rmse.set_xlabel("N paths", fontsize=11)
    ax_rmse.set_ylabel("RMSE of P_hit", fontsize=11)
    ax_rmse.set_title("(3) Statistical efficiency\n(RMSE vs number of paths)", fontsize=10)
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(True, which="both", alpha=0.25)

    # (4) x_error mean profile
    em_mean  = np.array([np.nanmean(xe_em[:, ci])  for ci in range(n_cp)])
    ssa_mean = np.array([np.nanmean(xe_ssa[:, ci]) for ci in range(n_cp)])
    em_se    = np.array([np.nanstd(xe_em[:, ci])  / np.sqrt(np.sum(~np.isnan(xe_em[:, ci])))
                         for ci in range(n_cp)])
    ssa_se   = np.array([np.nanstd(xe_ssa[:, ci]) / np.sqrt(np.sum(~np.isnan(xe_ssa[:, ci])))
                         for ci in range(n_cp)])

    ax_xm.errorbar(y_cp, em_mean,  yerr=2*em_se,  fmt="o-", **{k:v for k,v in em_lkw.items()},
                   capsize=4, label="EM  mean(x-x*)")
    ax_xm.errorbar(y_cp, ssa_mean, yerr=2*ssa_se, fmt="^-", **{k:v for k,v in ssa_lkw.items()},
                   capsize=4, label="SSA mean(x-x*)")
    ax_xm.axhline(0, color="k", ls=":", lw=1.2, label="x = x*(y)  (exact manifold)")
    ax_xm.set_xlabel("y checkpoint", fontsize=11)
    ax_xm.set_ylabel("mean(x - x*(y))  [bias]", fontsize=11)
    ax_xm.set_title(
        "(4) Manifold tracking bias along path\n"
        "EM: negative bias (slightly inside manifold)   "
        "SSA: positive bias (discrete grid overshoot)",
        fontsize=10)
    ax_xm.invert_xaxis()
    ax_xm.legend(fontsize=9)
    ax_xm.grid(True, alpha=0.25)

    # (5) x_error std profile
    em_std  = np.array([np.nanstd(xe_em[:, ci])  for ci in range(n_cp)])
    ssa_std = np.array([np.nanstd(xe_ssa[:, ci]) for ci in range(n_cp)])

    ax_xs.plot(y_cp, em_std,  "o-", **em_lkw,  label="EM  std(x-x*)")
    ax_xs.plot(y_cp, ssa_std, "^-", **ssa_lkw, label="SSA std(x-x*)")
    ax_xs.set_xlabel("y checkpoint", fontsize=11)
    ax_xs.set_ylabel("std(x - x*(y))  [spread]", fontsize=11)
    ax_xs.set_title(
        "(5) Manifold tracking spread along path\n"
        "Spread grows as path approaches saddle  (y -> -2/3)",
        fontsize=10)
    ax_xs.invert_xaxis()
    ax_xs.legend(fontsize=9)
    ax_xs.grid(True, alpha=0.25)

    fig.suptitle(
        f"Matched-work comparison: SSA hy=0.10 vs EM dt=eps/100\n"
        f"eps={EPS},  sigma/sigma_c=0.70,  "
        f"EM equiv_work~392,  SSA equiv_work~367",
        fontsize=12
    )

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

    def fmt(arr):
        v = arr[~np.isnan(arr)] if arr.dtype == float else arr
        return f"{np.mean(v):+.4f} +/- {np.std(v):.4f}"

    print(f"{'N paths':38s} {n_em:>15d} {n_ssa:>13d}")
    print(f"{'P_hit':38s} {np.mean(hits_em):>15.4f} {np.mean(hits_ssa):>13.4f}")
    print(f"{'P_hit SE':38s} "
          f"{np.std(hits_em.astype(float))/np.sqrt(n_em):>15.4f} "
          f"{np.std(hits_ssa.astype(float))/np.sqrt(n_ssa):>13.4f}")
    print()
    print(f"{'y_jump mean':38s} {np.nanmean(yj_em):>15.4f} {np.nanmean(yj_ssa):>13.4f}")
    print(f"{'y_jump std  (Canard quality)':38s} {np.nanstd(yj_em):>15.4f} {np.nanstd(yj_ssa):>13.4f}")
    print()
    print(f"{'tau_hit mean':38s} {np.nanmean(th_em):>15.4f} {np.nanmean(th_ssa):>13.4f}")
    print(f"{'tau_hit std':38s} {np.nanstd(th_em):>15.4f} {np.nanstd(th_ssa):>13.4f}")
    print()
    for ci, yc in enumerate(Y_CHECKPOINTS):
        em_v  = xe_em[:, ci][~np.isnan(xe_em[:, ci])]
        ssa_v = xe_ssa[:, ci][~np.isnan(xe_ssa[:, ci])]
        print(f"{'x_err mean @ y='+f'{yc:.2f}':38s} "
              f"{np.mean(em_v):>+15.4f} {np.mean(ssa_v):>+13.4f}")
        print(f"{'x_err std  @ y='+f'{yc:.2f}':38s} "
              f"{np.std(em_v):>15.4f} {np.std(ssa_v):>13.4f}")
    print("=" * 70)

    # Winner assessment
    print()
    print("Assessment:")
    if np.nanstd(yj_em) < np.nanstd(yj_ssa):
        print(f"  y_jump std:  EM wins  ({np.nanstd(yj_em):.4f} < {np.nanstd(yj_ssa):.4f})")
    else:
        print(f"  y_jump std:  SSA wins ({np.nanstd(yj_ssa):.4f} < {np.nanstd(yj_em):.4f})")

    if np.nanstd(th_em) < np.nanstd(th_ssa):
        print(f"  tau_hit std: EM wins  ({np.nanstd(th_em):.4f} < {np.nanstd(th_ssa):.4f})")
    else:
        print(f"  tau_hit std: SSA wins ({np.nanstd(th_ssa):.4f} < {np.nanstd(th_em):.4f})")

    em_bias  = np.nanmean(np.abs([np.nanmean(xe_em[:,ci])  for ci in range(len(Y_CHECKPOINTS))]))
    ssa_bias = np.nanmean(np.abs([np.nanmean(xe_ssa[:,ci]) for ci in range(len(Y_CHECKPOINTS))]))
    if em_bias < ssa_bias:
        print(f"  x_error bias: EM wins  ({em_bias:.4f} < {ssa_bias:.4f})")
    else:
        print(f"  x_error bias: SSA wins ({ssa_bias:.4f} < {em_bias:.4f})")

    em_spread  = np.nanmean([np.nanstd(xe_em[:,ci])  for ci in range(len(Y_CHECKPOINTS))])
    ssa_spread = np.nanmean([np.nanstd(xe_ssa[:,ci]) for ci in range(len(Y_CHECKPOINTS))])
    if em_spread < ssa_spread:
        print(f"  x_error spread: EM wins  ({em_spread:.4f} < {ssa_spread:.4f})")
    else:
        print(f"  x_error spread: SSA wins ({ssa_spread:.4f} < {em_spread:.4f})")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).resolve().parent

    print(f"EPS={EPS},  SIGMA_C={SIGMA_C:.4f},  sigma={SIGMA:.4f} (0.70*sigma_c)")
    print(f"Start: ({X0}, {Y0:.4f}),  saddle: y_c={-2/3:.4f}")
    print(f"Window A: y in [{Y_REF-ETA:.3f},{Y_REF+ETA:.3f}], |x-x*(y)|<={DELTA_HIT}")
    print()

    # JIT warm-up
    print("Compiling...")
    ycp = Y_CHECKPOINTS
    em_full_ensemble(EPS/100, 4, ycp)
    ssa_full_ensemble(0.10,   4, ycp)
    em_phit_batch(EPS/100, 4)
    ssa_phit_batch(0.10,   4)
    print("Done.\n")

    # Main run (N matched to equal equiv-work ~390)
    # From previous calibration: EM_fine ~392 equiv/path, SSA_0.10 ~367 equiv/path
    # Use N=800 for both (flop ratio ~ 1.07, close enough)
    N_EM  = 800
    N_SSA = 800

    print(f"Running EM dt=eps/100  N={N_EM}...")
    t0 = time.perf_counter()
    yj_em, th_em, tw_em, xe_em = em_full_ensemble(EPS/100, N_EM, ycp)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print(f"Running SSA hy=0.10    N={N_SSA}...")
    t0 = time.perf_counter()
    yj_ssa, th_ssa, tw_ssa, xe_ssa = ssa_full_ensemble(0.10, N_SSA, ycp)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Diagnostic: check for empty arrays before plotting
    n_th_em  = int(np.sum(~np.isnan(th_em)))
    n_th_ssa = int(np.sum(~np.isnan(th_ssa)))
    n_yj_em  = int(np.sum(~np.isnan(yj_em)))
    n_yj_ssa = int(np.sum(~np.isnan(yj_ssa)))
    print(f"  tau_hit valid: EM={n_th_em}/{N_EM}, SSA={n_th_ssa}/{N_SSA}")
    print(f"  y_jump  valid: EM={n_yj_em}/{N_EM}, SSA={n_yj_ssa}/{N_SSA}")
    if n_th_ssa == 0:
        print("  WARNING: SSA tau_hit all nan. Check numba cache: delete __pycache__.")

    # P_hit from full ensemble
    # Derive from tau_window being non-nan
    hits_em  = ~np.isnan(tw_em)
    hits_ssa = ~np.isnan(tw_ssa)

    # Large run for RMSE curve
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