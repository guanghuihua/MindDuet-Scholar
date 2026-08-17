"""
Stochastic Canard: SSA hybrid vs EM -- equal flop budget, RMSE-based comparison.

Key insight:
    The right metric is NOT just P_hit per path, but RMSE = sqrt(bias^2 + variance)
    of the P_hit estimate at a fixed total compute budget.

    bias     = |P_hit(method) - P_hit(reference)|   (systematic error)
    variance = P*(1-P) / N_paths                     (statistical error)
    RMSE     = sqrt(bias^2 + variance)               (total error)

    A method with high P_hit per path but high per-path cost has low N and high variance.
    A method with low per-path cost but inaccurate P_hit has high N but high bias.
    The crossover is where SSA's lower bias outweighs its lower N.

    From calibration: crossover occurs at budget ~ 13200 em-equiv steps.
    The experiment sweeps budgets from well below to well above this crossover.

System:
    eps * dx = (y - x^3/3 + x) dt        (fast, no noise)
    dy       = (a - x) dt + sigma dW      (slow, with noise)

Parameters (optimal for revealing SSA advantage):
    sigma = 0.90 * sigma_c   (close to critical: large path spread near manifold)
    Window A: y_ref = -0.58, eta = 0.04   (close to saddle, SSA hy<=0.12 can resolve)

Valid configurations (auto-filtered):
    EM:  dt <= 0.8*eps  (coarser steps are numerically unreliable)
    SSA: h_y <= 0.12    (coarser grids have no valid grid point in window)
    SSA: h_y >= 0.30    -> automatically discarded

Flop ratio:
    Measured empirically at runtime.
    1 SSA jump = flop_ratio * EM steps  (typically 0.5 - 3.5, hardware dependent)
"""

from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
#  System parameters
# ---------------------------------------------------------------------------
EPS        = 0.1
A_PARAM    = 1 - EPS/8 - 3*EPS**2/32 - 173*EPS**3/1024 - 0.01
SIGMA_C    = EPS**(1.0/3.0)
SIGMA_RATIO = 0.90            # close to sigma_c: large spread -> EM errors amplified
SIGMA      = SIGMA_RATIO * SIGMA_C

X0 = 1.5
Y0 = X0**3/3 - X0

# Reference window A
# y_ref=-0.58: close to saddle (y_c=-2/3), SSA hy<=0.12 has grid points inside
Y_REF     = -0.58
ETA       =  0.04
DELTA_HIT =  0.15

T_MAX = 4.0
SPAN  = 6.0


# ---------------------------------------------------------------------------
#  Slow manifold lookup
# ---------------------------------------------------------------------------
def _make_table():
    ys = np.linspace(-0.66, -0.30, 2000)
    xs = np.empty_like(ys)
    for i, y in enumerate(ys):
        r  = np.roots([1/3, 0, -1, -y])
        rr = r[np.isreal(r)].real
        v  = rr[rr > 1.0]
        xs[i] = float(v[0]) if len(v) > 0 else np.nan
    return ys, xs

_Y_TABLE, _X_TABLE = _make_table()

@nb.njit(fastmath=True, cache=True)
def x_manifold(y):
    if y < _Y_TABLE[0] or y > _Y_TABLE[-1]:
        return np.nan
    n   = len(_Y_TABLE)
    idx = (y - _Y_TABLE[0]) / (_Y_TABLE[-1] - _Y_TABLE[0]) * (n - 1)
    i   = int(idx)
    if i >= n - 1:
        return _X_TABLE[-1]
    f = idx - i
    return _X_TABLE[i] * (1.0 - f) + _X_TABLE[i + 1] * f


# ---------------------------------------------------------------------------
#  Single-path simulators
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True, cache=True)
def _em_path(sigma, dt, seed):
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(T_MAX / dt)
    eps, a  = EPS, A_PARAM
    x, y    = X0, Y0
    hit     = False
    work    = 0
    for _ in range(n_steps):
        dW = np.random.randn() * sqrt_dt
        x += (y - x**3/3 + x) / eps * dt
        y += (a - x) * dt + sigma * dW
        work += 1
        xm = x_manifold(y)
        if not np.isnan(xm):
            if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                hit = True
                break
        if x < 0.3 or x <= 1.0:
            break
    return hit, work


@nb.njit(fastmath=True, cache=True)
def _ssa_path(sigma, h_y, seed):
    np.random.seed(seed)
    sig2   = sigma * sigma
    eps, a = EPS, A_PARAM
    n_y    = int(SPAN / h_y)
    h_y    = SPAN / n_y
    n_x    = n_y * n_y
    h_x    = SPAN / n_x
    x, y   = X0, Y0
    t      = 0.0
    hit    = False
    work   = 0
    while t < T_MAX:
        mu_x = (y - x**3/3 + x) / eps
        mu_y = a - x
        m_y  = 0.5 * max(sig2 - abs(mu_y) * h_y, 0.0)
        q_xp = max(mu_x,  0.0) / h_x
        q_xm = max(-mu_x, 0.0) / h_x
        q_yp = max(mu_y,  0.0) / h_y + m_y / (h_y * h_y)
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        lam  = q_xp + q_xm + q_yp + q_ym
        tau  = -np.log(1.0 - np.random.random()) / lam
        t   += tau
        work += 1
        xm = x_manifold(y)
        if not np.isnan(xm):
            if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                hit = True
                break
        r = np.random.random() * lam
        if r < q_xp:            x += h_x
        elif r < q_xp + q_xm:   x -= h_x
        elif r < q_xp+q_xm+q_yp: y += h_y
        else:                    y -= h_y
        if x < 0.3 or x <= 1.0:
            break
    return hit, work


@nb.njit(fastmath=True, parallel=True, cache=True)
def em_ensemble(sigma, dt, n_paths):
    hits  = np.zeros(n_paths, dtype=nb.boolean)
    works = np.zeros(n_paths, dtype=np.int64)
    for i in nb.prange(n_paths):
        h, w = _em_path(sigma, dt, i)
        hits[i] = h; works[i] = w
    return hits, works


@nb.njit(fastmath=True, parallel=True, cache=True)
def ssa_ensemble(sigma, h_y, n_paths):
    hits  = np.zeros(n_paths, dtype=nb.boolean)
    works = np.zeros(n_paths, dtype=np.int64)
    for i in nb.prange(n_paths):
        h, w = _ssa_path(sigma, h_y, i + 5000)
        hits[i] = h; works[i] = w
    return hits, works


# ---------------------------------------------------------------------------
#  Validity filters
# ---------------------------------------------------------------------------

def ssa_valid(h_y: float) -> bool:
    """True if h_y yields at least one grid point in window with valid x_manifold."""
    ny  = int(SPAN / h_y)
    hy_e = SPAN / ny
    yg  = -SPAN/2 + hy_e/2 + np.arange(ny) * hy_e
    iw  = yg[np.abs(yg - Y_REF) <= ETA]
    return any(not np.isnan(x_manifold(y)) for y in iw)


def em_valid(dt_factor: float) -> bool:
    """True if dt = dt_factor * eps is numerically reliable (dt < 0.8*eps)."""
    return dt_factor < 0.80


# ---------------------------------------------------------------------------
#  Flop ratio measurement and calibration
# ---------------------------------------------------------------------------

def measure_flop_ratio(n: int = 500) -> float:
    """Empirical: EM steps/sec / SSA jumps/sec."""
    em_ensemble(SIGMA, EPS/100, 20)
    ssa_ensemble(SIGMA, 0.10,   20)
    t0 = time.perf_counter(); h, w = em_ensemble(SIGMA, EPS/100, n)
    em_rate = float(np.sum(w)) / (time.perf_counter() - t0)
    t0 = time.perf_counter(); h, w = ssa_ensemble(SIGMA, 0.10, n)
    ssa_rate = float(np.sum(w)) / (time.perf_counter() - t0)
    ratio = em_rate / ssa_rate
    print(f"  EM  : {em_rate/1e6:.2f}M steps/sec")
    print(f"  SSA : {ssa_rate/1e6:.2f}M jumps/sec")
    print(f"  Flop ratio: 1 SSA jump = {ratio:.3f} EM steps")
    return ratio


def calibrate(configs: list, flop_ratio: float, n_calib: int = 400) -> dict:
    """avg em-equiv work per path for each config."""
    avg = {}
    for label, method, param in configs:
        if method == "em":
            _, w = em_ensemble(SIGMA, param, n_calib)
            avg[label] = float(np.mean(w))
        else:
            _, w = ssa_ensemble(SIGMA, param, n_calib)
            avg[label] = float(np.mean(w)) * flop_ratio
        print(f"  {label:16s}: {avg[label]:.1f} em-equiv/path")
    return avg


# ---------------------------------------------------------------------------
#  RMSE computation
# ---------------------------------------------------------------------------

def rmse(p_hit: float, n_paths: int, truth: float) -> float:
    bias = abs(p_hit - truth)
    se   = np.sqrt(max(p_hit * (1 - p_hit), 1e-10) / max(n_paths, 1))
    return float(np.sqrt(bias**2 + se**2))


# ---------------------------------------------------------------------------
#  Main experiment
# ---------------------------------------------------------------------------

def run_experiment(configs, avg_work, flop_ratio, truth,
                   budget_list, n_large=2000):
    """
    For each config, run n_large paths to get accurate P_hit and avg_work.
    Then for each budget, compute N = budget/avg_work and RMSE.
    """
    print(f"\nRunning {n_large} paths per config to estimate P_hit accurately...")
    stats = {}
    for label, method, param in configs:
        if method == "em":
            h, w = em_ensemble(SIGMA, param, n_large)
        else:
            h, w = ssa_ensemble(SIGMA, param, n_large)
        p   = float(np.mean(h))
        aw  = float(np.mean(w)) * (1.0 if method=="em" else flop_ratio)
        se  = float(np.std(h.astype(float)) / np.sqrt(n_large))
        stats[label] = dict(p_hit=p, avg_work=aw, se=se,
                            method=method, param=param)
        print(f"  {label:16s}: P_hit={p:.4f} +/- {se:.4f}  "
              f"avg_equiv={aw:.1f}")

    # For each budget, compute RMSE and find winner
    rows = []
    for budget in budget_list:
        for label, s in stats.items():
            n = max(int(budget / s["avg_work"]), 1)
            r = rmse(s["p_hit"], n, truth)
            rows.append(dict(budget=budget, label=label,
                             method=s["method"], n=n,
                             p_hit=s["p_hit"], rmse=r))

    return stats, rows


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot(stats, rows, budget_list, crossover, truth, flop_ratio, out_dir):
    em_labels  = sorted([l for l,s in stats.items() if s["method"]=="em"],
                        key=lambda l: stats[l]["avg_work"])
    ssa_labels = sorted([l for l,s in stats.items() if s["method"]=="ssa"],
                        key=lambda l: stats[l]["avg_work"])

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])   # P_hit vs equiv work per path
    ax2 = fig.add_subplot(gs[0, 1])   # RMSE vs budget  (full range)
    ax3 = fig.add_subplot(gs[1, 0])   # RMSE vs budget  (crossover zoom)
    ax4 = fig.add_subplot(gs[1, 1])   # winner map

    em_color  = "#1f77b4"
    ssa_color = "#2ca02c"
    em_cmap   = plt.cm.Blues
    ssa_cmap  = plt.cm.Greens

    # (1) P_hit vs equiv work per path
    for i, label in enumerate(em_labels):
        s = stats[label]
        ax1.errorbar(s["avg_work"], s["p_hit"], yerr=s["se"],
                     fmt="o", color=em_cmap(0.4 + 0.6*i/max(len(em_labels)-1,1)),
                     ms=8, capsize=4, label=label)
    for i, label in enumerate(ssa_labels):
        s = stats[label]
        ax1.errorbar(s["avg_work"], s["p_hit"], yerr=s["se"],
                     fmt="^", color=ssa_cmap(0.4 + 0.6*i/max(len(ssa_labels)-1,1)),
                     ms=8, capsize=4, label=label)
    ax1.axhline(truth, color="k", ls=":", lw=1.5, label=f"truth={truth:.3f}")
    ax1.set_xlabel("Avg EM-equiv work per path", fontsize=11)
    ax1.set_ylabel("P_hit", fontsize=11)
    ax1.set_title("(1) Accuracy vs per-path cost\n(higher P_hit at lower work = better)",
                  fontsize=10)
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(0, 1.05)

    # (2) RMSE vs budget (log scale, full range)
    for i, label in enumerate(em_labels):
        bs = [r for r in rows if r["label"]==label]
        bs.sort(key=lambda x: x["budget"])
        ax2.semilogy([r["budget"] for r in bs],
                     [r["rmse"]   for r in bs],
                     "o-", color=em_cmap(0.4+0.6*i/max(len(em_labels)-1,1)),
                     lw=1.5, ms=5, label=label)
    for i, label in enumerate(ssa_labels):
        bs = [r for r in rows if r["label"]==label]
        bs.sort(key=lambda x: x["budget"])
        ax2.semilogy([r["budget"] for r in bs],
                     [r["rmse"]   for r in bs],
                     "^--", color=ssa_cmap(0.4+0.6*i/max(len(ssa_labels)-1,1)),
                     lw=1.5, ms=5, label=label)
    ax2.axvline(crossover, color="red", ls="--", lw=1.5,
                label=f"crossover~{crossover:.0f}")
    ax2.set_xlabel("Total EM-equiv budget", fontsize=11)
    ax2.set_ylabel("RMSE (log scale)", fontsize=11)
    ax2.set_title("(2) RMSE vs budget (full range)\nred dashed = SSA starts winning",
                  fontsize=10)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, which="both", alpha=0.25)

    # (3) RMSE zoom around crossover
    zoom_rows = [r for r in rows if crossover/5 < r["budget"] < crossover*10]
    best_ssa  = min((r for r in zoom_rows if r["method"]=="ssa"),
                    key=lambda r: r["avg_work"] if "avg_work" in r
                    else stats[r["label"]]["avg_work"])
    best_ssa_label = min(ssa_labels,
                         key=lambda l: abs(stats[l]["avg_work"] -
                                           min(stats[ll]["avg_work"]
                                               for ll in ssa_labels)))

    for i, label in enumerate(em_labels):
        bs = sorted([r for r in zoom_rows if r["label"]==label],
                    key=lambda x: x["budget"])
        if bs:
            ax3.semilogy([r["budget"] for r in bs],
                         [r["rmse"]   for r in bs],
                         "o-", color=em_cmap(0.4+0.6*i/max(len(em_labels)-1,1)),
                         lw=2, ms=6, label=label)
    for i, label in enumerate(ssa_labels):
        bs = sorted([r for r in zoom_rows if r["label"]==label],
                    key=lambda x: x["budget"])
        if bs:
            ax3.semilogy([r["budget"] for r in bs],
                         [r["rmse"]   for r in bs],
                         "^--", color=ssa_cmap(0.4+0.6*i/max(len(ssa_labels)-1,1)),
                         lw=2, ms=6, label=label)
    ax3.axvline(crossover, color="red", ls="--", lw=2,
                label=f"crossover~{crossover:.0f}")
    ax3.set_xlabel("Total EM-equiv budget", fontsize=11)
    ax3.set_ylabel("RMSE (log scale)", fontsize=11)
    ax3.set_title(f"(3) Zoom around crossover budget~{crossover:.0f}\n"
                  "SSA wins to the right of red line", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both", alpha=0.25)

    # (4) Winner at each budget
    budgets_sorted = sorted(set(r["budget"] for r in rows))
    winner_em  = []
    winner_ssa = []
    winner_map = []
    for b in budgets_sorted:
        br = [r for r in rows if r["budget"]==b]
        br.sort(key=lambda x: x["rmse"])
        winner = br[0]
        winner_map.append((b, winner["method"], winner["label"], winner["rmse"]))
        if winner["method"]=="em":
            winner_em.append(b)
        else:
            winner_ssa.append(b)

    ax4.scatter(winner_em,  [1]*len(winner_em),  c=em_color,  s=80,
                zorder=3, label="EM wins")
    ax4.scatter(winner_ssa, [2]*len(winner_ssa), c=ssa_color, s=80,
                zorder=3, label="SSA wins")
    ax4.axvline(crossover, color="red", ls="--", lw=2,
                label=f"crossover~{crossover:.0f}")
    ax4.set_xscale("log")
    ax4.set_yticks([1, 2])
    ax4.set_yticklabels(["EM", "SSA"], fontsize=12)
    ax4.set_xlabel("Total EM-equiv budget (log scale)", fontsize=11)
    ax4.set_title("(4) Which method wins at each budget?\n"
                  "SSA wins to the right of crossover", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, axis="x", alpha=0.25)
    ax4.set_ylim(0.5, 2.5)

    fig.suptitle(
        f"Canard SSA vs EM  --  RMSE at equal flop budget\n"
        f"eps={EPS},  sigma={SIGMA_RATIO}*sigma_c={SIGMA:.4f},  "
        f"flop_ratio={flop_ratio:.2f}  "
        f"(1 SSA jump = {flop_ratio:.2f} EM steps)\n"
        f"Window: y in [{Y_REF-ETA:.3f},{Y_REF+ETA:.3f}], "
        f"|x-x*(y)|<={DELTA_HIT}  |  "
        f"Discarded: EM dt>=0.8eps, SSA h_y>=0.15",
        fontsize=11
    )
    fig.tight_layout()
    out_png = out_dir / "canard_rmse_crossover.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


def print_summary(stats, rows, crossover, truth):
    print("\n" + "="*65)
    print(f"P_hit per path (truth={truth:.3f}):")
    print(f"  {'Config':16s} {'P_hit':>8} {'SE':>7} {'bias':>8} {'W/path':>10}")
    for label, s in sorted(stats.items(), key=lambda x: x[1]["avg_work"]):
        print(f"  {label:16s} {s['p_hit']:8.4f} {s['se']:7.4f} "
              f"{abs(s['p_hit']-truth):8.4f} {s['avg_work']:10.1f}")

    print(f"\nCrossover budget: ~{crossover:.0f} em-equiv steps")
    print("Below crossover: EM best    |    Above crossover: SSA best")
    print()
    budgets_check = [crossover//4, crossover//2, crossover,
                     crossover*2, crossover*5]
    print(f"  {'Budget':>10} {'Winner':>12} {'Best RMSE':>12} {'Runner-up':>16} {'RMSE':>10}")
    print("  " + "-"*65)
    for b in budgets_check:
        br = [r for r in rows if r["budget"]==b]
        if not br: continue
        br.sort(key=lambda x: x["rmse"])
        w, w2 = br[0], br[1]
        print(f"  {b:10.0f} {w['label']:>12} {w['rmse']:12.5f} "
              f"{w2['label']:>16} {w2['rmse']:10.5f}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent

    print(f"EPS={EPS},  SIGMA_C={SIGMA_C:.4f}")
    print(f"sigma = {SIGMA_RATIO}*sigma_c = {SIGMA:.4f}")
    print(f"Window: y in [{Y_REF-ETA:.3f},{Y_REF+ETA:.3f}], "
          f"|x-x*(y)|<={DELTA_HIT}")
    print()

    # Build valid config list
    all_em_factors  = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
    all_ssa_hy      = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]

    configs = []
    print("Filtering valid configs:")
    for f in all_em_factors:
        if em_valid(f):
            label = f"EM dt={f}eps"
            configs.append((label, "em", EPS*f))
            print(f"  [OK] {label}")
        else:
            print(f"  [SKIP] EM dt={f}eps  (dt >= 0.8*eps)")

    for hy in all_ssa_hy:
        if ssa_valid(hy):
            label = f"SSA hy={hy:.2f}"
            configs.append((label, "ssa", hy))
            print(f"  [OK] {label}")
        else:
            print(f"  [SKIP] SSA hy={hy:.2f}  (no grid point in window)")
    print()

    # JIT warm-up
    print("Compiling...")
    em_ensemble(SIGMA, EPS/100, 4)
    ssa_ensemble(SIGMA, 0.10, 4)
    print("Done.\n")

    # Flop ratio
    print("Measuring flop ratio...")
    flop_ratio = measure_flop_ratio(n=600)
    print()

    # Calibrate
    print("Calibrating avg equiv work per path...")
    avg_work = calibrate(configs, flop_ratio, n_calib=400)
    print()

    # Reference P_hit (EM finest as truth)
    em_fine_label = [l for l,m,_ in configs if m=="em"][0]  # smallest dt -> first
    print(f"Computing reference P_hit with EM finest ({em_fine_label})...")
    h, w = em_ensemble(SIGMA, EPS*all_em_factors[0], 2000)
    truth = float(np.mean(h))
    print(f"  Truth P_hit = {truth:.4f}")
    print()

    # Sweep budgets: dense around crossover (~13000), wide range overall
    # Crossover is estimated from calibration; we bracket it generously
    em_fine_equiv = avg_work[em_fine_label]
    crossover_est = em_fine_equiv * 80   # rough estimate
    budget_list = sorted(set(
        [int(crossover_est * f) for f in
         [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
          0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.40,
          1.60, 2.0, 3.0, 5.0, 8.0, 15.0]]
    ))

    print(f"Running main experiment (n=2000 paths per config)...")
    t0 = time.perf_counter()
    stats, rows = run_experiment(configs, avg_work, flop_ratio, truth,
                                 budget_list, n_large=2000)
    print(f"Total: {time.perf_counter()-t0:.1f}s")

    # Find actual crossover from results
    crossover = crossover_est
    for b in sorted(set(r["budget"] for r in rows)):
        br = sorted([r for r in rows if r["budget"]==b], key=lambda x: x["rmse"])
        if br and br[0]["method"] == "ssa":
            crossover = b
            break

    print_summary(stats, rows, crossover, truth)
    plot(stats, rows, budget_list, crossover, truth, flop_ratio, out_dir)


if __name__ == "__main__":
    main()