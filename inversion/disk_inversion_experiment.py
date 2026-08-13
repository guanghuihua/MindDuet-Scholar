import numpy as np
import matplotlib.pyplot as plt


def invert(z, R):
    # Geometric inversion (circle inversion)
    return (R ** 2) / np.conjugate(z)


def sample_disk(center, radius, n=80000, seed=0):
    rng = np.random.default_rng(seed)
    # Uniform sampling in disk by sqrt trick
    rho = radius * np.sqrt(rng.random(n))
    ang = 2 * np.pi * rng.random(n)
    return center + rho * np.exp(1j * ang)


def boundary_circle(center, radius, n=1500):
    t = np.linspace(0, 2 * np.pi, n)
    return center + radius * np.exp(1j * t)


def plot_case(ax_l, ax_r, center, radius, R, title):
    # Original disk
    z_in = sample_disk(center, radius, n=45000, seed=42)
    z_bd = boundary_circle(center, radius)

    # Avoid singularity at z=0 (maps to infinity)
    eps = 1e-4
    z_in = z_in[np.abs(z_in) > eps]
    z_bd = z_bd[np.abs(z_bd) > eps]

    w_in = invert(z_in, R)
    w_bd = invert(z_bd, R)

    theta = np.linspace(0, 2 * np.pi, 800)

    ax_l.scatter(z_in.real, z_in.imag, s=1, alpha=0.2, color='tab:blue')
    ax_l.plot(z_bd.real, z_bd.imag, color='tab:blue', lw=1.8)
    ax_l.plot(R * np.cos(theta), R * np.sin(theta), 'k--', lw=1.0)
    ax_l.scatter([0], [0], c='k', s=20)
    ax_l.set_title(f'Original: {title}')
    ax_l.set_aspect('equal', adjustable='box')
    ax_l.grid(True, alpha=0.25)

    ax_r.scatter(w_in.real, w_in.imag, s=1, alpha=0.2, color='tab:red')
    ax_r.plot(w_bd.real, w_bd.imag, color='tab:red', lw=1.8)
    ax_r.scatter([0], [0], c='k', s=20)
    ax_r.set_title('After inversion')
    ax_r.set_aspect('equal', adjustable='box')
    ax_r.grid(True, alpha=0.25)

    for ax in (ax_l, ax_r):
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')


def main():
    R = 3.0

    # Case A: disk strictly contains origin (|a| < r)
    center_a = 2.0 + 1.0j
    radius_a = 3.0

    # Case B: boundary passes through origin (|a| = r)
    center_b = 3.0 + 0.0j
    radius_b = 3.0

    fig, axs = plt.subplots(2, 2, figsize=(12, 11))

    plot_case(
        axs[0, 0],
        axs[0, 1],
        center=center_a,
        radius=radius_a,
        R=R,
        title='Disk contains origin (strictly)',
    )
    plot_case(
        axs[1, 0],
        axs[1, 1],
        center=center_b,
        radius=radius_b,
        R=R,
        title='Disk boundary passes origin',
    )

    fig.suptitle('Disk under geometric inversion: w = R^2 / conj(z),  R=3', fontsize=13)
    plt.tight_layout()

    out = 'disk_inversion_experiment.png'
    plt.savefig(out, dpi=180)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
