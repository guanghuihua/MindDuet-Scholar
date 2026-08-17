from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np


@nb.njit(cache=False, fastmath=True)
def potential_grad(x: float) -> float:
    # U(x) = x^4/4 - x^2/2, so U'(x) = x^3 - x
    return x * x * x - x


@nb.njit(cache=False, fastmath=True)
def underdamped_drift(x: float, v: float, gamma: float) -> tuple[float, float]:
    mu_x = v
    mu_v = -potential_grad(x) - gamma * v
    return mu_x, mu_v


@nb.njit(cache=False, fastmath=True)
def _project_to_grid(z: float, low: float, h: float) -> float:
    return low + h / 2.0 + round((z - low - h / 2.0) / h) * h


@nb.njit(cache=False, fastmath=True)
def _rates(mu_x: float, mu_v: float, h: float, sigma: float, scheme_id: int) -> tuple[float, float, float, float]:
    # Degenerate noise: diffusion only in v-direction.
    qx_p = max(mu_x, 0.0) / h
    qx_m = max(-mu_x, 0.0) / h

    m = 0.5 * sigma * sigma
    if scheme_id == 0:  # Q_u
        qv_p = max(mu_v, 0.0) / h + m / (h * h)
        qv_m = max(-mu_v, 0.0) / h + m / (h * h)
    elif scheme_id == 1:  # Q_c in v-direction + upwind in x-direction
        coef = m / (h * h)
        qv_p = coef * np.exp((mu_v * h) / (2.0 * m))
        qv_m = coef * np.exp(-(mu_v * h) / (2.0 * m))
    else:  # Q_tilde_u
        m_v = 0.5 * max(sigma * sigma - abs(mu_v) * h, 0.0)
        qv_p = max(mu_v, 0.0) / h + m_v / (h * h)
        qv_m = max(-mu_v, 0.0) / h + m_v / (h * h)
    return qx_p, qx_m, qv_p, qv_m


@nb.njit(cache=False, fastmath=True)
def simulate_stationary_histogram(
    n: int,
    scheme_id: int,
    gamma: float,
    sigma: float,
    lowx: float,
    lowv: float,
    span: float,
    tau: float,
    burn_steps: int,
    sample_steps: int,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)
    h = span / n
    x = lowx + np.random.random() * span
    v = lowv + np.random.random() * span
    x = _project_to_grid(x, lowx, h)
    v = _project_to_grid(v, lowv, h)

    counts = np.zeros((n, n), dtype=np.float64)
    total_steps = burn_steps + sample_steps

    for step in range(total_steps):
        mu_x, mu_v = underdamped_drift(x, v, gamma)
        qx_p, qx_m, qv_p, qv_m = _rates(mu_x, mu_v, h, sigma, scheme_id)

        dx = np.random.poisson(qx_p * tau) - np.random.poisson(qx_m * tau)
        dv = np.random.poisson(qv_p * tau) - np.random.poisson(qv_m * tau)

        x += dx * h
        v += dv * h

        x = min(max(x, lowx + h / 2.0), lowx + span - h / 2.0)
        v = min(max(v, lowv + h / 2.0), lowv + span - h / 2.0)
        x = _project_to_grid(x, lowx, h)
        v = _project_to_grid(v, lowv, h)

        if step >= burn_steps:
            ix = int(round((x - lowx - h / 2.0) / h))
            iv = int(round((v - lowv - h / 2.0) / h))
            if 0 <= ix < n and 0 <= iv < n:
                counts[ix, iv] += 1.0

    total = np.sum(counts)
    if total <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    return counts / (total * h * h)


def true_invariant_density(n: int, gamma: float, sigma: float, lowx: float, lowv: float, span: float) -> np.ndarray:
    h = span / n
    x = np.linspace(lowx + h / 2.0, lowx + span - h / 2.0, n)
    v = np.linspace(lowv + h / 2.0, lowv + span - h / 2.0, n)
    xx, vv = np.meshgrid(x, v, indexing="ij")

    u = 0.25 * xx**4 - 0.5 * xx**2
    beta = 2.0 * gamma / (sigma * sigma)
    rho = np.exp(-beta * (u + 0.5 * vv * vv))
    rho /= np.sum(rho) * h * h
    return rho


def l1_error(pi_hat: np.ndarray, pi_true: np.ndarray, h: float) -> float:
    return float(np.sum(np.abs(pi_hat - pi_true)) * h * h)


def local_slope(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])


def adaptive_sample_steps(
    n: int,
    n_ref: int,
    tau_ref: float,
    h_ref: float,
    h: float,
    t_sample: float,
    sample_cap: int,
) -> tuple[float, int]:
    # tau scales linearly with h; sample steps are additionally scaled by 1/h
    # so the effective sample size per bin is more stable across grids.
    tau = tau_ref * (h / h_ref)
    base_steps = int(round(t_sample / tau))
    extra_scale = n / n_ref
    sample_steps = int(round(base_steps * extra_scale))
    sample_steps = min(sample_cap, max(1, sample_steps))
    return tau, sample_steps


def plot_accuracy(h_vals: np.ndarray, e_qu: np.ndarray, e_qc: np.ndarray, e_qtu: np.ndarray, out: Path) -> None:
    i0 = 1
    c1 = 0.8 * e_qu[i0] / h_vals[i0]
    c2 = 0.7 * e_qc[i0] / (h_vals[i0] ** 2)

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.loglog(h_vals, e_qu, "o-", color="#0072BD", mfc="none", lw=0.9, ms=8, label=r"$Q_u$")
    ax.loglog(h_vals, c1 * h_vals, "--", color="#D95319", dashes=(7, 5), lw=0.9, label=r"$O(h)$")
    ax.loglog(h_vals, e_qc, "o-", color="#EDB120", mfc="none", lw=0.9, ms=8, label=r"$Q_c$")
    ax.loglog(h_vals, c2 * (h_vals**2), "--", color="#7E2F8E", dashes=(7, 4), lw=0.9, label=r"$O(h^2)$")
    ax.loglog(h_vals, e_qtu, "x-", color="#77AC30", lw=0.8, ms=7, mew=0.8, label=r"$\tilde{Q}_u$")

    ax.set_title("Underdamped Langevin: Stationary Density Accuracy")
    ax.set_xlabel("spatial stepsize h")
    ax.set_ylabel(r"$l^1$-error")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.show()


def main() -> None:
    gamma = 1.0
    sigma = 1.0
    lowx = -4.0
    lowv = -4.0
    span = 8.0
    n_list = [48, 96, 192, 384]
    h_vals = span / np.array(n_list, dtype=np.float64)

    tau_ref = 0.02
    t_burn = 1_000.0
    t_sample = 4_000.0
    sample_cap = 1_600_000
    n_rep = 2
    n_ref = n_list[0]
    h_ref = span / n_ref

    print("Underdamped Langevin test with v6 algorithm pipeline")
    print(f"gamma={gamma}, sigma={sigma}, beta={2.0 * gamma / (sigma * sigma):.3f}")
    print(f"domain=[{lowx},{lowx+span}]x[{lowv},{lowv+span}]")
    print(f"N={n_list}, h={h_vals}")
    print(f"tau_ref={tau_ref}, t_burn={t_burn}, t_sample={t_sample}, sample_cap={sample_cap}, n_rep={n_rep}")

    _ = simulate_stationary_histogram(16, 2, gamma, sigma, lowx, lowv, span, tau_ref, 10, 20, seed=1)

    err = np.zeros((3, len(n_list)), dtype=np.float64)
    for i, n in enumerate(n_list):
        h = span / n
        tau, sample_steps = adaptive_sample_steps(n, n_ref, tau_ref, h_ref, h, t_sample, sample_cap)
        burn_steps = int(round(t_burn / tau))
        print(f"N={n:4d}, tau={tau:.6f}, burn_steps={burn_steps}, sample_steps={sample_steps}")
        rho_true = true_invariant_density(n, gamma, sigma, lowx, lowv, span)
        for scheme_id in range(3):
            e_sum = 0.0
            for r in range(n_rep):
                pi_hat = simulate_stationary_histogram(
                    n,
                    scheme_id,
                    gamma,
                    sigma,
                    lowx,
                    lowv,
                    span,
                    tau,
                    burn_steps,
                    sample_steps,
                    seed=100000 * (scheme_id + 1) + 1000 * n + r,
                )
                e_sum += l1_error(pi_hat, rho_true, h)
            err[scheme_id, i] = e_sum / n_rep
            print(f"N={n:4d}, scheme={scheme_id}, L1={err[scheme_id, i]:.6e}")

    e_qu, e_qc, e_qtu = err[0], err[1], err[2]
    print("local slopes Qu:", local_slope(h_vals, e_qu))
    print("local slopes Qc:", local_slope(h_vals, e_qc))
    print("local slopes Qut:", local_slope(h_vals, e_qtu))

    out = Path(__file__).resolve().parent / "underdamped_Langevin_stationary_density_accuracy.png"
    plot_accuracy(h_vals, e_qu, e_qc, e_qtu, out)


if __name__ == "__main__":
    main()
