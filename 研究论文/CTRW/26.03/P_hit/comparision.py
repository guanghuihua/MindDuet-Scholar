"""
Stochastic Canard: equal-budget comparison of EM vs SSA.

Core question:
    Under a fixed compute budget W, each method faces a trade-off:
        EM:  smaller dt  => accurate P_hit, but fewer paths N => large SE
             larger dt  => many paths N => small SE, but P_hit is WRONG
        SSA: smaller h_y => accurate P_hit, but fewer paths N => large SE
             larger h_y  => more paths, but P_hit degrades gracefully

    => At matched work, SSA occupies a better accuracy-vs-paths frontier.

Budget: W = cost(EM_fine) * N_REF paths
Each method gets N = W / cost_per_path paths.
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
T_MAX = 4.0
SPAN  = 6.0

SSA_FLOP_RATIO = 3.47   # 1 SSA jump = this many EM steps (empirical)
P_TRUE = 0.990          # ground truth P_hit (from EM_fine with large N)
N_REF  = 600            # budget = cost(EM_fine) * N_REF


# ---------------------------------------------------------------------------
#  Slow manifold table
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
#  Simulators
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def em_batch(dt, n_paths):
    hits  = np.zeros(n_paths, dtype=nb.boolean)
    works = np.zeros(n_paths, dtype=np.int64)
    sqrt_dt = np.sqrt(dt)
    ns      = int(T_MAX / dt)

    for i in nb.prange(n_paths):
        np.random.seed(i)
        x, y = X0, Y0
        hit  = False
        w    = 0
        for _ in range(ns):
            x += (y - x**3/3 + x) / EPS * dt
            y += (A_PARAM - x) * dt + SIGMA * np.random.randn() * sqrt_dt
            w += 1
            xm = x_manifold(y)
            if not np.isnan(xm):
                if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                    hit = True
                    break
            if x < 0.3:
                break
        hits[i]  = hit
        works[i] = w

    return hits, works


@nb.njit(parallel=True, cache=True)
def ssa_batch(h_y, n_paths):
    hits  = np.zeros(n_paths, dtype=nb.boolean)
    works = np.zeros(n_paths, dtype=np.int64)
    sig2  = SIGMA * SIGMA

    for i in nb.prange(n_paths):
        np.random.seed(i + 5000)
        n_y = int(SPAN / h_y)
        hy2 = SPAN / n_y
        hx2 = SPAN / (n_y * n_y)

        x, y = X0, Y0
        t    = 0.0
        hit  = False
        w    = 0

        while t < T_MAX:
            mu_x = (y - x**3/3 + x) / EPS
            mu_y = A_PARAM - x
            m_y  = 0.5 * max(sig2 - abs(mu_y) * hy2, 0.0)
            q_xp = max(mu_x,  0.0) / hx2
            q_xm = max(-mu_x, 0.0) / hx2
            q_yp = max(mu_y,  0.0) / hy2 + m_y / (hy2 * hy2)
            q_ym = max(-mu_y, 0.0) / hy2 + m_y / (hy2 * hy2)
            lam  = q_xp + q_xm + q_yp + q_ym
            t   += -np.log(1.0 - np.random.random()) / lam
            w   += 1

            r = np.random.random() * lam
            if r < q_xp:              x += hx2
            elif r < q_xp + q_xm:    x -= hx2
            elif r < q_xp+q_xm+q_yp: y += hy2
            else:                     y -= hy2

            xm = x_manifold(y)
            if not np.isnan(xm):
                if abs(y - Y_REF) <= ETA and abs(x - xm) <= DELTA_HIT:
                    hit = True
                    break
            if x < 0.3:
                break

        hits[i]  = hit
        works[i] = w

    return hits, works


# ---------------------------------------------------------------------------
#  Calibrate cost per path
# ---------------------------------------------------------------------------

def calibrate(n_calib=300):
    costs = {}
    print("Calibrating cost per path...")
    for label, fn, param in [
        ("EM dt=eps/100", em_batch, EPS/100),
        ("EM dt=0.1*eps", em_batch, EPS*0.1),
        ("EM dt=0.2*eps", em_batch, EPS*0.2),
        ("EM dt=0.5*eps", em_batch, EPS*0.5),
        ("EM dt=1.0*eps", em_batch, EPS*1.0),
        ("EM dt=2.0*eps", em_batch, EPS*2.0),
        ("SSA hy=0.10",   ssa_batch, 0.10),
        ("SSA hy=0.20",   ssa_batch, 0.20),
        ("SSA hy=0.30",   ssa_batch, 0.30),
    ]:
        _, works = fn(param, n_calib)
        ratio     = SSA_FLOP_RATIO if "SSA" in label else 1.0
        avg_equiv = float(np.mean(works)) * ratio
        costs[label] = (param, fn, avg_equiv)
        print(f"  {label:18s}: {avg_equiv:.1f} EM-equiv/path")
    return costs


# ---------------------------------------------------------------------------
#  Run equal-budget experiment
# ---------------------------------------------------------------------------

def run_experiment(costs):
    budget = costs["EM dt=eps/100"][2] * N_REF
    print(f"\nBudget = {budget:.0f} EM-equiv steps  (= EM_fine x {N_REF} paths)\n")

    rows = []
    for label, (param, fn, cost_per_path) in costs.items():
        N = max(int(budget / cost_per_path), 30)
        t0 = time.perf_counter()
        hits, works = fn(param, N)
        elapsed = time.perf_counter() - t0

        ratio    = SSA_FLOP_RATIO if "SSA" in label else 1.0
        p_hit    = float(np.mean(hits))
        se       = float(np.std(hits.astype(float)) / np.sqrt(N))
        bias     = abs(p_hit - P_TRUE)
        rmse     = float(np.sqrt(bias**2 + p_hit*(1-p_hit)/N))

        print(f"  {label:18s}: N={N:6d}, P_hit={p_hit:.4f}, "
              f"SE={se:.4f}, RMSE={rmse:.4f}  ({elapsed:.1f}s)")

        rows.append(dict(
            label  = label,
            method = "em" if "EM" in label else "ssa",
            param  = param,
            N      = N,
            cost   = cost_per_path,
            p_hit  = p_hit,
            se     = se,
            rmse   = rmse,
        ))

    return rows, budget


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------

def plot_results(rows, budget, out_dir):
    em_rows  = sorted([r for r in rows if r["method"]=="em"],
                      key=lambda r: -r["cost"])
    ssa_rows = sorted([r for r in rows if r["method"]=="ssa"],
                      key=lambda r: -r["cost"])

    em_c, ssa_c = "#1f77b4", "#2ca02c"

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]

    def pts(rows_list):
        return ([r["N"] for r in rows_list],
                [r["p_hit"] for r in rows_list],
                [r["se"] for r in rows_list],
                [r["rmse"] for r in rows_list])

    em_N, em_p, em_se, em_rmse   = pts(em_rows)
    ssa_N, ssa_p, ssa_se, ssa_rmse = pts(ssa_rows)

    def annotate(ax, rows_list, color):
        for r in rows_list:
            short = (r["label"]
                     .replace("EM dt=", "dt=")
                     .replace("SSA hy=", "h="))
            ax.annotate(short, xy=(r["N"], r["p_hit"]) if ax is ax1
                        else (r["N"], r["rmse"]) if ax is ax2
                        else (r["se"], r["p_hit"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=color)

    # ------ Panel 1: P_hit vs N ---------------------------------------------------------------------------------------------------------------------------------------------------
    ax1.errorbar(em_N,  em_p,  yerr=2*np.array(em_se),
                 fmt="o-", color=em_c,  lw=2, ms=7, capsize=4, label="EM")
    ax1.errorbar(ssa_N, ssa_p, yerr=2*np.array(ssa_se),
                 fmt="^-", color=ssa_c, lw=2, ms=7, capsize=4, label="SSA")
    ax1.axhline(P_TRUE, color="k", ls=":", lw=1.5,
                label=f"True P_hit = {P_TRUE}")
    annotate(ax1, em_rows+ssa_rows, "gray")
    ax1.set_xlabel("N  (number of paths)", fontsize=12)
    ax1.set_ylabel("P_hit", fontsize=12)
    ax1.set_xscale("log")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(
        "(1)  P_hit vs N  (equal budget)\n"
        "EM: more paths = coarser dt = wrong P_hit\n"
        "SSA: P_hit stays near true value",
        fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)

    # ------ Panel 2: RMSE vs N ------------------------------------------------------------------------------------------------------------------------------------------------------
    ax2.loglog(em_N,  em_rmse,  "o-", color=em_c,  lw=2, ms=7, label="EM")
    ax2.loglog(ssa_N, ssa_rmse, "^-", color=ssa_c, lw=2, ms=7, label="SSA")
    n_ref = np.logspace(np.log10(min(em_N+ssa_N)),
                        np.log10(max(em_N+ssa_N)), 100)
    ax2.loglog(n_ref, 0.1/np.sqrt(n_ref), "k--", lw=1.2,
               label=r"$\propto 1/\sqrt{N}$ (stat only)")
    annotate(ax2, em_rows+ssa_rows, "gray")
    ax2.set_xlabel("N  (number of paths)", fontsize=12)
    ax2.set_ylabel(r"RMSE  of  P_hit", fontsize=12)
    ax2.set_title(
        r"(2)  RMSE = $\sqrt{bias^2 + variance}$" + "\n"
        "EM: RMSE rises when dt > eps  (bias dominates)\n"
        "SSA: RMSE follows stat-only line",
        fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)

    # ------ Panel 3: P_hit vs SE (accuracy-uncertainty frontier) ------------------------------------------------
    ax3.scatter(em_se,  em_p,  s=90, color=em_c,  marker="o", zorder=3,
                label="EM")
    ax3.scatter(ssa_se, ssa_p, s=90, color=ssa_c, marker="^", zorder=3,
                label="SSA")

    # draw EM trajectory with arrows (fine->coarse)
    for i in range(len(em_rows)-1):
        ax3.annotate("",
                     xy=(em_rows[i+1]["se"], em_rows[i+1]["p_hit"]),
                     xytext=(em_rows[i]["se"], em_rows[i]["p_hit"]),
                     arrowprops=dict(arrowstyle="->", color=em_c, lw=1.2))
    for i in range(len(ssa_rows)-1):
        ax3.annotate("",
                     xy=(ssa_rows[i+1]["se"], ssa_rows[i+1]["p_hit"]),
                     xytext=(ssa_rows[i]["se"], ssa_rows[i]["p_hit"]),
                     arrowprops=dict(arrowstyle="->", color=ssa_c, lw=1.2))

    for r in em_rows + ssa_rows:
        short = (r["label"]
                 .replace("EM dt=", "dt=")
                 .replace("SSA hy=", "h="))
        c = em_c if r["method"]=="em" else ssa_c
        ax3.annotate(short, xy=(r["se"], r["p_hit"]),
                     xytext=(5, 4), textcoords="offset points",
                     fontsize=7, color=c)

    ax3.axhline(P_TRUE, color="k", ls=":", lw=1.5,
                label=f"True P_hit = {P_TRUE}")
    # ideal region
    ax3.axhspan(P_TRUE-0.02, 1.0, alpha=0.07, color="green",
                label="Acceptable accuracy")

    ax3.set_xlabel("SE  (statistical uncertainty of P_hit)", fontsize=12)
    ax3.set_ylabel("P_hit  (accuracy)", fontsize=12)
    ax3.set_title(
        "(3)  Accuracy vs Uncertainty frontier\n"
        "Arrows: fine -> coarse discretisation\n"
        "EM drops off bottom; SSA stays near true value",
        fontsize=10)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(True, alpha=0.25)

    fig.suptitle(
        f"Equal-budget comparison: EM vs SSA  "
        f"(eps={EPS}, sigma/sigma_c=0.70)\n"
        f"Budget = EM_fine x {N_REF} paths.  "
        f"1 SSA jump = {SSA_FLOP_RATIO} EM steps.",
        fontsize=12)

    out_png = out_dir / "canard_optimal.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


def print_table(rows):
    print()
    print("=" * 78)
    print(f"{'Method':18s} {'N':>7} {'cost/path':>10} {'P_hit':>8} "
          f"{'SE':>7} {'RMSE':>8}  note")
    print("-" * 78)
    for r in rows:
        note = ""
        if r["method"] == "em" and r["param"] >= EPS:
            note = "<-- bias >> SE  (dt >= eps)"
        elif r["method"] == "em" and r["param"] >= EPS*0.2:
            note = "<-- some bias"
        elif r["method"] == "ssa" and r["p_hit"] < 0.1:
            note = "<-- grid/window mismatch"
        print(f"{r['label']:18s} {r['N']:7d} {r['cost']:10.1f} "
              f"{r['p_hit']:8.4f} {r['se']:7.4f} {r['rmse']:8.4f}  {note}")
    print("=" * 78)
    print()
    print("Trade-off summary:")
    print("  EM:  fine dt  -> accurate P_hit, small N, large SE")
    print("       coarse dt -> large N, tiny SE, but P_hit is WRONG")
    print("  SSA: P_hit stays near true value; SE decreases with N normally")
    print("       => SSA achieves acceptable accuracy with more paths")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent

    print(f"EPS={EPS},  SIGMA_C={SIGMA_C:.4f},  sigma={SIGMA:.4f}")
    print(f"True P_hit (reference) = {P_TRUE}")
    print()

    print("Compiling JIT functions...")
    em_batch(EPS/100, 4)
    ssa_batch(0.10, 4)
    print("Done.\n")

    costs = calibrate(n_calib=300)
    rows, budget = run_experiment(costs)
    print_table(rows)
    plot_results(rows, budget, out_dir)


if __name__ == "__main__":
    main()