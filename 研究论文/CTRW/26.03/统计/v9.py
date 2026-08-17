"""
Stochastic Canard system: fixed subcritical noise, compare formats using one
event statistic under matched path count and nearly matched total work.

Goal:
    Choose sigma < sigma_c where theory predicts the canard should persist.
    Then compare schemes using only one indicator:

        P_hit = probability that the path reaches a reference tube on the
                positive stable slow manifold before crossing x = 1.

Computation matching:
    - All schemes use the same number of sample paths.
    - EM work is measured by the number of time steps taken.
    - SSA work is measured by the number of jump events taken.
    - Comparison is made by actual measured work, not just nominal dt or h_y.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


# ---------------------------------------------------------------------------
#  System parameters
# ---------------------------------------------------------------------------
EPS = 0.1
A_PARAM = 1 - EPS / 8 - 3 * EPS**2 / 32 - 173 * EPS**3 / 1024 - 0.01
SIGMA_C = EPS ** (1.0 / 3.0)
X0 = 1.5
Y0 = X0**3 / 3 - X0

# Use one fixed noise value below sigma_c: canard should still persist.
SIGMA_RATIO = 0.7
SIGMA = SIGMA_RATIO * SIGMA_C

# Reference tube near the positive stable slow manifold.
Y_REF = -0.60
ETA = 0.04
DELTA_HIT = 0.15

T_MAX = 4.0
SPAN = 6.0


# ---------------------------------------------------------------------------
#  Slow manifold helper
# ---------------------------------------------------------------------------
def x_manifold_py(y: float) -> float:
    """Positive stable branch: solve x^3/3 - x = y with x > 1."""
    coeffs = [1.0 / 3.0, 0.0, -1.0, -y]
    roots = np.roots(coeffs)
    real_r = roots[np.isreal(roots)].real
    valid = real_r[real_r > 1.0]
    return float(valid[0]) if len(valid) > 0 else np.nan


_Y_TABLE = np.linspace(-0.66, -0.30, 1000)
_X_TABLE = np.array([x_manifold_py(y) for y in _Y_TABLE])


@nb.njit(fastmath=True, cache=True)
def x_manifold_fast(y: float) -> float:
    if y < -0.66 or y > -0.30:
        return np.nan
    n = len(_Y_TABLE)
    idx = (y - _Y_TABLE[0]) / (_Y_TABLE[-1] - _Y_TABLE[0]) * (n - 1)
    i = int(idx)
    if i >= n - 1:
        return _X_TABLE[-1]
    frac = idx - i
    return _X_TABLE[i] * (1.0 - frac) + _X_TABLE[i + 1] * frac


# ---------------------------------------------------------------------------
#  EM simulator
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True, cache=True)
def _em_single(sigma, dt, seed, y_ref, eta, delta_hit):
    np.random.seed(seed)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(T_MAX / dt)

    x, y = X0, Y0

    hit = False
    work = 0

    for _ in range(n_steps):
        dW = np.random.randn() * sqrt_dt
        x += (y - x**3 / 3 + x) / EPS * dt
        y += (A_PARAM - x) * dt + sigma * dW
        work += 1

        xm = x_manifold_fast(y)
        if not np.isnan(xm):
            dx = abs(x - xm)
            if abs(y - y_ref) <= eta and dx <= delta_hit:
                hit = True
                break

        if x < 0.3:
            break

        if x <= 1.0:
            break

    return hit, work


@nb.njit(fastmath=True, parallel=True, cache=True)
def em_batch(sigma, dt, n_paths, y_ref, eta, delta_hit):
    hits = np.zeros(n_paths, dtype=nb.boolean)
    works = np.zeros(n_paths, dtype=np.int64)

    for i in nb.prange(n_paths):
        h, w = _em_single(sigma, dt, i, y_ref, eta, delta_hit)
        hits[i] = h
        works[i] = w

    return hits, works


# ---------------------------------------------------------------------------
#  SSA simulator
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True, cache=True)
def _ssa_single(sigma, h_y, seed, y_ref, eta, delta_hit):
    np.random.seed(seed)
    sig2 = sigma * sigma

    n_y = int(SPAN / h_y)
    h_y = SPAN / n_y
    n_x = n_y * n_y
    h_x = SPAN / n_x

    x, y = X0, Y0
    t = 0.0

    hit = False
    work = 0

    while t < T_MAX:
        mu_x = (y - x**3 / 3 + x) / EPS
        mu_y = A_PARAM - x
        m_y = 0.5 * max(sig2 - abs(mu_y) * h_y, 0.0)

        q_xp = max(mu_x, 0.0) / h_x
        q_xm = max(-mu_x, 0.0) / h_x
        q_yp = max(mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        q_ym = max(-mu_y, 0.0) / h_y + m_y / (h_y * h_y)
        lam = q_xp + q_xm + q_yp + q_ym

        tau = -np.log(1.0 - np.random.random()) / lam
        t += tau
        work += 1

        xm = x_manifold_fast(y)
        if not np.isnan(xm):
            dx = abs(x - xm)
            if abs(y - y_ref) <= eta and dx <= delta_hit:
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

        if x < 0.3:
            break

        if x <= 1.0:
            break

    return hit, work


@nb.njit(fastmath=True, parallel=True, cache=True)
def ssa_batch(sigma, h_y, n_paths, y_ref, eta, delta_hit):
    hits = np.zeros(n_paths, dtype=nb.boolean)
    works = np.zeros(n_paths, dtype=np.int64)

    for i in nb.prange(n_paths):
        h, w = _ssa_single(sigma, h_y, i + 5000, y_ref, eta, delta_hit)
        hits[i] = h
        works[i] = w

    return hits, works


def summarise(hits, works):
    total_work = int(np.sum(works))
    avg_work = float(np.mean(works))
    return dict(
        P_hit=float(np.mean(hits)),
        total_work=total_work,
        avg_work=avg_work,
    )


def compare_score(stats: dict, target_work: float) -> float:
    """
    Higher is better.
    Prefer larger hit probability, but penalize moving away from the target work.
    """
    rel_gap = abs(stats["avg_work"] - target_work) / max(target_work, 1e-12)
    return stats["P_hit"] - 0.15 * rel_gap


def run_experiment(configs, n_paths=800):
    geo = (Y_REF, ETA, DELTA_HIT)
    results = []

    for label, method, param in configs:
        print(f"\n--- {label} ---", flush=True)
        t0 = time.perf_counter()
        if method == "em":
            h, w = em_batch(SIGMA, param, n_paths, *geo)
        else:
            h, w = ssa_batch(SIGMA, param, n_paths, *geo)
        stats = summarise(h, w)
        stats["label"] = label
        stats["method"] = method
        stats["param"] = param
        stats["elapsed"] = time.perf_counter() - t0

        print(
            f"  P_hit={stats['P_hit']:.3f}  avg_work/path={stats['avg_work']:.1f}  "
            f"total_work={stats['total_work']}  "
            f"({stats['elapsed']:.1f}s)"
        )
        results.append(stats)

    em_avg = [r["avg_work"] for r in results if r["method"] == "em"]
    target_work = float(np.median(np.array(em_avg))) if em_avg else 0.0
    for row in results:
        row["work_gap"] = abs(row["avg_work"] - target_work)
        row["work_gap_rel"] = row["work_gap"] / max(target_work, 1e-12)
        row["score"] = compare_score(row, target_work)

    results.sort(key=lambda item: (-item["score"], item["work_gap"]))
    return results, target_work


def build_pairs(results):
    em_rows = [r for r in results if r["method"] == "em"]
    ssa_rows = [r for r in results if r["method"] == "ssa"]
    used = np.zeros(len(ssa_rows), dtype=np.bool_)
    pairs = []

    em_rows.sort(key=lambda item: item["avg_work"])
    ssa_rows.sort(key=lambda item: item["avg_work"])

    for em_row in em_rows:
        best_j = -1
        best_gap = np.inf
        for j, ssa_row in enumerate(ssa_rows):
            if used[j]:
                continue
            gap = abs(ssa_row["avg_work"] - em_row["avg_work"])
            if gap < best_gap:
                best_gap = gap
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            ssa_row = ssa_rows[best_j]
            rel_gap = best_gap / max(0.5 * (em_row["avg_work"] + ssa_row["avg_work"]), 1e-12)
            pairs.append((em_row, ssa_row, rel_gap))

    return pairs


def plot_results(results, pairs, target_work: float, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    em_rows = sorted([r for r in results if r["method"] == "em"], key=lambda item: item["avg_work"])
    ssa_rows = sorted([r for r in results if r["method"] == "ssa"], key=lambda item: item["avg_work"])

    axes[0].plot(
        [r["avg_work"] for r in em_rows],
        [r["P_hit"] for r in em_rows],
        "o-",
        color="#1f77b4",
        label="EM",
    )
    axes[0].plot(
        [r["avg_work"] for r in ssa_rows],
        [r["P_hit"] for r in ssa_rows],
        "^-",
        color="#2ca02c",
        label="SSA",
    )
    axes[0].axvline(target_work, color="gray", ls=":", lw=1.2, label="target work")
    axes[0].set_xlabel("Average work per path")
    axes[0].set_ylabel("P_hit")
    axes[0].set_title("Hit probability vs measured work")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if pairs:
        x = np.arange(len(pairs))
        em_hit = [pair[0]["P_hit"] for pair in pairs]
        ssa_hit = [pair[1]["P_hit"] for pair in pairs]
        labels = [
            f"{pair[0]['label']}\nvs\n{pair[1]['label']}\nΔ={100 * pair[2]:.1f}%"
            for pair in pairs
        ]
        width = 0.36
        axes[1].bar(x - width / 2, em_hit, width, color="#1f77b4", alpha=0.85, label="EM")
        axes[1].bar(x + width / 2, ssa_hit, width, color="#2ca02c", alpha=0.85, label="SSA")
        axes[1].set_xticks(x, labels, rotation=0)
    axes[1].set_ylabel("P_hit")
    axes[1].set_title("Closest-work EM vs SSA pairs")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        "Fixed-noise canard preservation by one indicator\n"
        f"sigma/sigma_c={SIGMA_RATIO:.2f}, same path count, work matched by measured updates",
        fontsize=13,
    )

    fig.tight_layout()
    out_png = out_dir / "canard_fixed_sigma_hit_compare.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    plt.show()


def print_ranking(results, target_work: float) -> None:
    print("\nRanking (higher hit with work near target is better):")
    print(f"Target average work per path = {target_work:.1f}")
    for i, row in enumerate(results, start=1):
        print(
            f"{i:2d}. {row['label']:<14} "
            f"score={row['score']:.4f}  "
            f"P_hit={row['P_hit']:.3f}  avg_work={row['avg_work']:.1f}  "
            f"gap={100 * row['work_gap_rel']:.1f}%"
        )


def print_pairs(pairs) -> None:
    if not pairs:
        return
    print("\nClosest-work EM vs SSA pairs:")
    for em_row, ssa_row, rel_gap in pairs:
        better = "EM" if em_row["P_hit"] > ssa_row["P_hit"] else "SSA"
        if np.isclose(em_row["P_hit"], ssa_row["P_hit"]):
            better = "tie"
        print(
            f"  {em_row['label']}  <->  {ssa_row['label']}  "
            f"work gap={100 * rel_gap:.1f}%  "
            f"P_hit: {em_row['P_hit']:.3f} vs {ssa_row['P_hit']:.3f}  "
            f"better={better}"
        )


def main():
    out_dir = Path(__file__).resolve().parent
    configs = [
        ("EM  dt=eps/100", "em", EPS / 100),
        ("EM  dt=0.1*eps", "em", EPS * 0.1),
        ("EM  dt=0.5*eps", "em", EPS * 0.5),
        ("EM  dt=1.0*eps", "em", EPS * 1.0),
        ("SSA hy=0.06", "ssa", 0.06),
        ("SSA hy=0.10", "ssa", 0.10),
        ("SSA hy=0.30", "ssa", 0.30),
    ]

    print(f"EPS={EPS}, SIGMA_C={SIGMA_C:.4f}, A={A_PARAM:.5f}")
    print(f"Start: ({X0}, {Y0:.4f})")
    print(f"Fixed sigma ratio: sigma/sigma_c={SIGMA_RATIO:.2f}, sigma={SIGMA:.4f}")
    print(
        f"Window A: y in [{Y_REF - ETA:.3f}, {Y_REF + ETA:.3f}], "
        f"|x - x*(y)| <= {DELTA_HIT}"
    )
    print("Only statistic used for comparison: P_hit")
    print("Work metric: EM steps / SSA jump events")
    print()

    print("Compiling JIT functions...")
    geo = (Y_REF, ETA, DELTA_HIT)
    em_batch(SIGMA, EPS / 100, 4, *geo)
    ssa_batch(SIGMA, 0.1, 4, *geo)
    print("Done.\n")

    t0 = time.perf_counter()
    results, target_work = run_experiment(configs, n_paths=800)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
    pairs = build_pairs(results)
    print_ranking(results, target_work)
    print_pairs(pairs)
    plot_results(results, pairs, target_work, out_dir)


if __name__ == "__main__":
    main()