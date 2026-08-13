from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


@nb.njit(cache=True)
def U(x: float) -> float:
    return 0.5 * x * x


@nb.njit(cache=True)
def U_prime(x: float) -> float:
    return x


@nb.njit(cache=True)
def _rates_mixed(x: float, v: float, hx: float, hv: float, gamma: float, sigma: float) -> tuple[float, float, float, float]:
    # x-direction (no diffusion): upwind
    r_xp = 0.0
    r_xm = 0.0
    if v > 0.0:
        r_xp = v / hx
    elif v < 0.0:
        r_xm = -v / hx

    # v-direction (with noise): Q_c (centered drift-diffusion)
    D = 0.5 * sigma * sigma
    # drift b = -(U'(x) + gamma v)
    # rates to v+- use half-node drift
    r_vp = 0.0
    r_vm = 0.0
    b_plus = -(U_prime(x) + gamma * (v + 0.5 * hv))
    b_minus = -(U_prime(x) + gamma * (v - 0.5 * hv))
    r_vp = D / (hv * hv) + b_plus / (2.0 * hv)
    r_vm = D / (hv * hv) - b_minus / (2.0 * hv)
    if r_vp < 0.0:
        r_vp = 0.0
    if r_vm < 0.0:
        r_vm = 0.0
    return r_xp, r_xm, r_vp, r_vm


@nb.njit(cache=True)
def simulate_ssa(
    n: int,
    Lx: float,
    Lv: float,
    gamma: float,
    sigma: float,
    t_burn: float,
    t_sample: float,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)

    x = np.linspace(-Lx, Lx, n)
    v = np.linspace(-Lv, Lv, n)
    hx = x[1] - x[0]
    hv = v[1] - v[0]

    # start at center
    ix = n // 2
    jv = n // 2

    counts = np.zeros((n, n), dtype=np.float64)

    t = 0.0
    t_end = t_burn + t_sample
    while t < t_end:
        xx = x[ix]
        vv = v[jv]
        r_xp, r_xm, r_vp, r_vm = _rates_mixed(xx, vv, hx, hv, gamma, sigma)

        # apply boundary caps
        if ix == n - 1:
            r_xp = 0.0
        if ix == 0:
            r_xm = 0.0
        if jv == n - 1:
            r_vp = 0.0
        if jv == 0:
            r_vm = 0.0

        r_sum = r_xp + r_xm + r_vp + r_vm
        if r_sum <= 0.0:
            # stuck; break
            break

        u = np.random.random()
        dt = -np.log(max(u, 1e-12)) / r_sum

        if t >= t_burn:
            counts[ix, jv] += dt

        t += dt
        u2 = np.random.random() * r_sum
        if u2 < r_xp:
            ix += 1
        elif u2 < r_xp + r_xm:
            ix -= 1
        elif u2 < r_xp + r_xm + r_vp:
            jv += 1
        else:
            jv -= 1

    total = np.sum(counts)
    if total <= 0.0:
        return np.zeros_like(counts)
    # convert to density
    return counts / (total * hx * hv)


def true_density(n: int, Lx: float, Lv: float, gamma: float, sigma: float) -> np.ndarray:
    x = np.linspace(-Lx, Lx, n)
    v = np.linspace(-Lv, Lv, n)
    hx = x[1] - x[0]
    hv = v[1] - v[0]
    beta = 2.0 * gamma / (sigma * sigma)
    X, V = np.meshgrid(x, v, indexing="ij")
    rho = np.exp(-beta * (0.5 * X * X + 0.5 * V * V))
    rho /= np.sum(rho) * hx * hv
    return rho


def l1_error(p_hat: np.ndarray, p_true: np.ndarray, h: float) -> float:
    return float(np.sum(np.abs(p_hat - p_true)) * h * h)


def tail_order(h: np.ndarray, e: np.ndarray, tail_points: int = 3) -> float:
    x = np.log(h[-tail_points:])
    y = np.log(e[-tail_points:])
    a, _b = np.polyfit(x, y, 1)
    return float(a)


def main() -> None:
    # Domain and parameters
    Lx = 3.0
    Lv = 3.0
    gamma = 1.0
    sigma = 1.0

    # grid sizes (h = 2L / (N-1))
    N_list = [41, 61, 81, 101,201]

    # SSA settings
    t_burn = 50.0
    t_sample = 200.0
    n_rep = 4

    h_list = []
    err_list = []

    print("Underdamped Langevin SSA (mixed Q_c in v, upwind in x)")
    print(f"Lx={Lx}, Lv={Lv}, gamma={gamma}, sigma={sigma}")
    print(f"t_burn={t_burn}, t_sample={t_sample}, n_rep={n_rep}")

    for n in N_list:
        h = (2.0 * Lx) / (n - 1)
        h_list.append(h)
        p_true = true_density(n, Lx, Lv, gamma, sigma)

        e_sum = 0.0
        for r in range(n_rep):
            p_hat = simulate_ssa(n, Lx, Lv, gamma, sigma, t_burn, t_sample, seed=1000 * n + r)
            e_sum += l1_error(p_hat, p_true, h)
        err = e_sum / n_rep
        err_list.append(err)
        print(f"N={n:4d}, h={h:.6f}, L1={err:.6e}")

    h_vals = np.array(h_list)
    e_vals = np.array(err_list)

    # reference lines anchored at the finest grid
    h_ref = h_vals[-1]
    e_ref = e_vals[-1]
    ref1 = e_ref * (h_vals / h_ref) ** 1
    ref2 = e_ref * (h_vals / h_ref) ** 2

    print("Tail fit order (last 3 points):", tail_order(h_vals, e_vals, 3))

    plt.figure(figsize=(7.2, 5.2))
    plt.loglog(h_vals, e_vals, "o-", label="mixed: x-upwind + v-Q_c")
    plt.loglog(h_vals, ref1, "--", label="O(h)")
    plt.loglog(h_vals, ref2, "--", label="O(h^2)")
    plt.xlabel("grid size h")
    plt.ylabel("L1 error of invariant density")
    plt.title("Underdamped Langevin: invariant density accuracy")
    plt.legend()
    plt.tight_layout()

    out = Path(__file__).resolve().parent / "mix_v4_accuracy.png"
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
