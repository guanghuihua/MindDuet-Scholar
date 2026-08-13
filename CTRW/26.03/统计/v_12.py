"""
Stochastic Canard: SSA hybrid vs EM comparison under equal flop budget.

Each SSA jump is converted to equivalent EM steps using an empirical
timing ratio (measured at runtime), so that both methods are compared
at the same total floating-point work.

System:
    eps * dx = (y - x^3/3 + x) dt        (fast, no noise)
    dy       = (a - x) dt + sigma dW      (slow, with noise)

Fixed noise: sigma = SIGMA_RATIO * sigma_c   (subcritical, Canard should persist)

Metric: P_hit = probability that the path enters window A before x = 1.
    A = { |y - Y_REF| <= ETA,  |x - x*(y)| <= DELTA_HIT }

Equal-flop budget:
    1. Calibrate: for each method, run N_CALIB paths and measure
       - EM:  avg_em_steps  per path
       - SSA: avg_ssa_jumps per path
    2. Measure wall-clock throughput (steps/sec or jumps/sec).
    3. Compute: flop_ratio = EM_steps_per_sec / SSA_jumps_per_sec
       => 1 SSA jump = flop_ratio EM equivalent steps
    4. Convert all work to EM-equivalent units.
    5. Fix budget W = avg_em_equiv_per_path(EM_fine) * N_REF
    6. Each method gets n_paths = W / avg_em_equiv_per_path
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
SIGMA_RATIO = 0.90
SIGMA      = SIGMA_RATIO * SIGMA_C

X0 = 1.5
Y0 = X0**3/3 - X0

# Reference window A
Y_REF      = -0.58
ETA        =  0.04
DELTA_HIT  =  0.15

T_MAX = 4.0
SPAN  = 6.0

# Reference budget: equivalent to N_REF paths of EM_fine
N_REF    = 600
N_CALIB  = 300   # paths used for calibration


# ---------------------------------------------------------------------------
#  Slow manifold lookup table (precomputed for numba)
# ---------------------------------------------------------------------------
_Y_TABLE = np.linspace(-0.66, -0.30, 2000)
_X_TABLE = np.array([
    float(np.roots([1/3, 0, -1, -y])[np.isreal(np.roots([1/3, 0, -1, -y])).nonzero()[0]]
          [np.roots([1/3, 0, -1, -y])[np.isreal(np.roots([1/3, 0, -1, -y])).nonzero()[0]].real > 1.0][0].real)
    if len(np.roots([1/3, 0, -1, -y])[np.isreal(np.roots([1/3, 0, -1, -y])).nonzero()[0]]
           [np.roots([1/3, 0, -1, -y])[np.isreal(np.roots([1/3, 0, -1, -y])).nonzero()[0]].real > 1.0]) > 0
    else np.nan
    for y in _Y_TABLE
])

# Cleaner rebuild
def _make_table():
    ys = np.linspace(-0.66, -0.30, 2000)
    xs = np.empty_like(ys)
    for i, y in enumerate(ys):
        r = np.roots([1/3, 0, -1, -y])
        rr = r[np.isreal(r)].real
        v  = rr[rr > 1.0]
        xs[i] = float(v[0]) if len(v) > 0 else np.nan
    return ys, xs

_Y_TABLE, _X_TABLE = _make_table()


@nb.njit(fastmath=True, cache=True)
def x_manifold(y: float) -> float:
    """Linear interpolation on precomputed positive stable manifold."""
    if y < _Y_TABLE[0] or y > _Y_TABLE[-1]:
        return np.nan
    n   = len(_Y_TABLE)
    idx = (y - _Y_TABLE[0]) / (_Y_TABLE[-1] - _Y_TABLE[0]) * (n - 1)
    i   = int(idx)
    if i >= n - 1:
        return _X_TABLE[-1]
    f = idx - i
    return _X_TABLE[i] * (1.0 - f) + _X_TABLE[i+1] * f


# ---------------------------------------------------------------------------
#  Single-path simulators (numba JIT)
#  Return: (hit: bool, raw_work: int)
#  raw_work = number of EM steps  or  number of SSA jumps (NOT yet converted)
# ---------------------------------------------------------------------------

@nb.njit(fastmath=True, cache=True)
def _em_path(sigma, dt, seed):
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(T_MAX / dt)
    eps, a  = EPS, A_PARAM

    x, y = X0, Y0
    hit  = False
    work = 0

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
    sig2 = sigma * sigma
    eps, a = EPS, A_PARAM

    n_y = int(SPAN / h_y)
    h_y = SPAN / n_y          # exact grid spacing
    n_x = n_y * n_y           # n_x = n_y^2
    h_x = SPAN / n_x

    x, y = X0, Y0
    t    = 0.0
    hit  = False
    work = 0   # counts SSA jumps (raw, before flop conversion)

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
        if r < q_xp:
            x += h_x
        elif r < q_xp + q_xm:
            x -= h_x
        elif r < q_xp + q_xm + q_yp:
            y += h_y
        else:
            y -= h_y

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
#  Empirical flop-ratio calibration
# ---------------------------------------------------------------------------

def measure_throughput(n_warmup: int = 50, n_measure: int = 500) -> float:
    """
    Measure wall-clock throughput for EM and SSA.
    Returns flop_ratio = EM_steps_per_sec / SSA_jumps_per_sec.
    This means: 1 SSA jump costs flop_ratio EM steps.
    """
    sigma = SIGMA

    # EM throughput (use dt = eps/100 as reference)
    em_ensemble(sigma, EPS/100, n_warmup)           # warm-up
    t0  = time.perf_counter()
    h, w = em_ensemble(sigma, EPS/100, n_measure)
    t_em = time.perf_counter() - t0
    em_steps_per_sec = float(np.sum(w)) / t_em

    # SSA throughput (use h_y = 0.10 as reference)
    ssa_ensemble(sigma, 0.10, n_warmup)
    t0   = time.perf_counter()
    h, w = ssa_ensemble(sigma, 0.10, n_measure)
    t_ssa = time.perf_counter() - t0
    ssa_jumps_per_sec = float(np.sum(w)) / t_ssa

    ratio = em_steps_per_sec / ssa_jumps_per_sec
    print(f"  EM  throughput : {em_steps_per_sec/1e6:.2f}M steps/sec")
    print(f"  SSA throughput : {ssa_jumps_per_sec/1e6:.2f}M jumps/sec")
    print(f"  Flop ratio     : 1 SSA jump = {ratio:.2f} EM steps")
    return ratio


# ---------------------------------------------------------------------------
#  Calibrate avg work per path for each method
# ---------------------------------------------------------------------------

def calibrate_avg_work(
    configs: list,
    flop_ratio: float,
    n_calib: int = N_CALIB,
) -> dict:
    """
    For each method, run n_calib paths and record avg raw work.
    Convert SSA raw jumps to EM-equivalent using flop_ratio.
    Returns dict: label -> avg_em_equiv_work_per_path
    """
    avg_work = {}
    for label, method, param in configs:
        if method == "em":
            _, w = em_ensemble(SIGMA, param, n_calib)
            avg = float(np.mean(w))           # already in EM steps
        else:
            _, w = ssa_ensemble(SIGMA, param, n_calib)
            avg = float(np.mean(w)) * flop_ratio  # convert to EM-equiv
        avg_work[label] = avg
        print(f"  {label:16s}: avg_em_equiv = {avg:.1f}")
    return avg_work


# ---------------------------------------------------------------------------
#  Main experiment: equal flop budget
# ---------------------------------------------------------------------------

def run_equal_budget(
    configs: list,
    avg_work: dict,
    flop_ratio: float,
    n_ref: int = N_REF,
) -> list:
    """
    Fix budget = avg_work(EM_fine) * n_ref.
    Each method gets n_paths = budget / avg_work[label].
    Run and return results list.
    """
    # find EM_fine label
    em_fine_label = next(l for l, m, _ in configs if m == "em" and "100" in l)
    budget = avg_work[em_fine_label] * n_ref

    print(f"\nBudget = {budget:.0f} EM-equivalent steps  "
          f"({em_fine_label} x {n_ref} paths)")
    print()

    results = []
    for label, method, param in configs:
        n_paths = max(int(budget / avg_work[label]), 30)
        print(f"--- {label}  (N={n_paths}) ---", flush=True)

        t0 = time.perf_counter()
        if method == "em":
            hits, raw_works = em_ensemble(SIGMA, param, n_paths)
            equiv_works     = raw_works.astype(float)
        else:
            hits, raw_works = ssa_ensemble(SIGMA, param, n_paths)
            equiv_works     = raw_works.astype(float) * flop_ratio

        elapsed = time.perf_counter() - t0

        p_hit       = float(np.mean(hits))
        se_hit      = float(np.std(hits.astype(float)) / np.sqrt(n_paths))
        avg_equiv   = float(np.mean(equiv_works))

        print(f"  P_hit = {p_hit:.4f} +/- {se_hit:.4f}   "
              f"avg_equiv_work = {avg_equiv:.1f}   "
              f"({elapsed:.1f}s)")

        results.append(dict(
            label       = label,
            method      = method,
            param       = param,
            n_paths     = n_paths,
            p_hit       = p_hit,
            se_hit      = se_hit,
            avg_equiv   = avg_equiv,
            elapsed     = elapsed,
        ))

    return results, budget


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_results(results: list, budget: float, flop_ratio: float,
                 out_dir: Path) -> None:

    em_rows  = sorted([r for r in results if r["method"] == "em"],
                      key=lambda x: x["avg_equiv"])
    ssa_rows = sorted([r for r in results if r["method"] == "ssa"],
                      key=lambda x: x["avg_equiv"])

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Left: P_hit vs avg equivalent work
    ax1.errorbar(
        [r["avg_equiv"] for r in em_rows],
        [r["p_hit"]     for r in em_rows],
        yerr=[r["se_hit"] for r in em_rows],
        fmt="o-", color="#1f77b4", lw=2, ms=7, capsize=4,
        label="EM"
    )
    ax1.errorbar(
        [r["avg_equiv"] for r in ssa_rows],
        [r["p_hit"]     for r in ssa_rows],
        yerr=[r["se_hit"] for r in ssa_rows],
        fmt="^-", color="#2ca02c", lw=2, ms=7, capsize=4,
        label="SSA  (n_x = n_y^2)"
    )

    # annotate each point with its label
    for r in results:
        ax1.annotate(
            r["label"],
            xy=(r["avg_equiv"], r["p_hit"]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=7, color="#1f77b4" if r["method"]=="em" else "#2ca02c"
        )

    ax1.set_xlabel("Avg EM-equivalent work per path", fontsize=12)
    ax1.set_ylabel("P_hit", fontsize=12)
    ax1.set_title(
        "P_hit vs flop-equivalent work\n"
        f"(1 SSA jump = {flop_ratio:.2f} EM steps, empirical)",
        fontsize=11
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25)

    # Right: bar chart - equal-budget N comparison
    # match pairs by closest avg_equiv
    pairs = []
    used  = set()
    for er in em_rows:
        best = min(
            [s for s in ssa_rows if id(s) not in used],
            key=lambda s: abs(s["avg_equiv"] - er["avg_equiv"]),
            default=None
        )
        if best is not None:
            used.add(id(best))
            pairs.append((er, best))

    x      = np.arange(len(pairs))
    width  = 0.35
    em_ph  = [p[0]["p_hit"] for p in pairs]
    ssa_ph = [p[1]["p_hit"] for p in pairs]
    em_se  = [p[0]["se_hit"] for p in pairs]
    ssa_se = [p[1]["se_hit"] for p in pairs]

    bars_em  = ax2.bar(x - width/2, em_ph,  width,
                       color="#1f77b4", alpha=0.85, label="EM",
                       yerr=em_se, capsize=4, error_kw=dict(lw=1.5))
    bars_ssa = ax2.bar(x + width/2, ssa_ph, width,
                       color="#2ca02c", alpha=0.85, label="SSA",
                       yerr=ssa_se, capsize=4, error_kw=dict(lw=1.5))

    # x-tick labels: show both method names and N
    tick_labels = []
    for er, sr in pairs:
        gap_pct = abs(er["avg_equiv"] - sr["avg_equiv"]) / \
                  (0.5*(er["avg_equiv"]+sr["avg_equiv"])) * 100
        tick_labels.append(
            f"{er['label']} (N={er['n_paths']})\nvs\n"
            f"{sr['label']} (N={sr['n_paths']})\n"
            f"work gap {gap_pct:.0f}%"
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, fontsize=7)
    ax2.set_ylabel("P_hit", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(
        "Equal flop-budget pairs\n"
        f"(budget = EM_fine x {N_REF} paths)",
        fontsize=11
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Stochastic Canard: SSA (n_x=n_y^2) vs EM  --  Equal flop budget\n"
        f"eps={EPS},  sigma/sigma_c={SIGMA_RATIO:.2f},  "
        f"Window A: y in [{Y_REF-ETA:.3f}, {Y_REF+ETA:.3f}], "
        f"|x-x*(y)|<={DELTA_HIT}",
        fontsize=11
    )
    fig.tight_layout()

    out_png = out_dir / "canard_equal_flop_phit.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


def print_summary(results: list, flop_ratio: float) -> None:
    print("\n" + "="*72)
    print(f"{'Label':18s} {'Method':5s} {'N':>7} {'P_hit':>8} {'+-':>6} "
          f"{'equiv_work':>12}  winner")
    print("-"*72)

    # find pairs
    em_rows  = sorted([r for r in results if r["method"]=="em"],
                      key=lambda x: x["avg_equiv"])
    ssa_rows = sorted([r for r in results if r["method"]=="ssa"],
                      key=lambda x: x["avg_equiv"])
    used = set()
    for er in em_rows:
        best = min(
            [s for s in ssa_rows if id(s) not in used],
            key=lambda s: abs(s["avg_equiv"] - er["avg_equiv"]),
            default=None
        )
        if best is None:
            continue
        used.add(id(best))
        gap = abs(er["avg_equiv"] - best["avg_equiv"]) / \
              (0.5*(er["avg_equiv"]+best["avg_equiv"])) * 100
        w_em  = "**" if er["p_hit"]   > best["p_hit"] else "  "
        w_ssa = "**" if best["p_hit"] > er["p_hit"]   else "  "
        print(f"{er['label']:18s} {'em':5s} {er['n_paths']:>7} "
              f"{er['p_hit']:>8.4f} {er['se_hit']:>6.4f} "
              f"{er['avg_equiv']:>12.1f}  {w_em}EM")
        print(f"{best['label']:18s} {'ssa':5s} {best['n_paths']:>7} "
              f"{best['p_hit']:>8.4f} {best['se_hit']:>6.4f} "
              f"{best['avg_equiv']:>12.1f}  {w_ssa}SSA")
        print(f"  work gap = {gap:.1f}%")
        print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).resolve().parent

    print(f"EPS={EPS},  SIGMA_C={SIGMA_C:.4f},  A={A_PARAM:.5f}")
    print(f"sigma = {SIGMA_RATIO} * sigma_c = {SIGMA:.4f}")
    print(f"Window A: y in [{Y_REF-ETA:.3f}, {Y_REF+ETA:.3f}], "
          f"|x-x*(y)| <= {DELTA_HIT}")
    print()

    configs = [
        ("EM dt=eps/100",  "em",  EPS/100),
        ("EM dt=0.1*eps",  "em",  EPS*0.1),
        ("EM dt=0.5*eps",  "em",  EPS*0.5),
        ("EM dt=1.0*eps",  "em",  EPS*1.0),
        ("SSA hy=0.06",    "ssa", 0.06),
        ("SSA hy=0.10",    "ssa", 0.10),
        ("SSA hy=0.30",    "ssa", 0.30),
    ]

    # JIT warm-up
    print("Compiling JIT functions...")
    em_ensemble(SIGMA, EPS/100, 4)
    ssa_ensemble(SIGMA, 0.10,   4)
    print("Done.\n")

    # Step 1: measure empirical flop ratio
    print("Step 1: Measuring flop ratio (wall-clock throughput)...")
    flop_ratio = measure_throughput(n_warmup=100, n_measure=800)
    print()

    # Step 2: calibrate avg equiv work per path
    print("Step 2: Calibrating average EM-equivalent work per path...")
    avg_work = calibrate_avg_work(configs, flop_ratio, n_calib=N_CALIB)
    print()

    # Step 3: run equal-budget experiment
    print("Step 3: Running equal-budget experiment...")
    t0 = time.perf_counter()
    results, budget = run_equal_budget(configs, avg_work, flop_ratio, N_REF)
    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")

    print_summary(results, flop_ratio)
    plot_results(results, budget, flop_ratio, out_dir)


if __name__ == "__main__":
    main()