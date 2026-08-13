import os
import numpy as np
import matplotlib.pyplot as plt


def invert_points(z, R):
    # Geometric inversion: O, z, w are collinear.
    return (R ** 2) / np.conjugate(z)


def circle_points(center, radius, n=1000):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return center + radius * np.exp(1j * t)


def main():
    R = 3.0
    center = 6.0 + 0.0j
    ks = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(9, 9))

    theta = np.linspace(0, 2 * np.pi, 800)

    # Inversion reference circle
    ax.plot(R * np.cos(theta), R * np.sin(theta), color='black', lw=1.4, label=f'Inversion circle |z|={R:g}')

    cmap = plt.get_cmap('tab10')

    for i, k in enumerate(ks):
        color = cmap(i % 10)
        z = circle_points(center=center, radius=float(k), n=1500)
        w = invert_points(z, R)

        ax.plot(z.real, z.imag, color=color, lw=1.8, alpha=0.9, label=f'Original circle k={k}')
        ax.plot(w.real, w.imag, color=color, lw=1.8, ls='--', alpha=0.95, label=f'Inverted image k={k}')

    # Pointwise correspondence demo for k=3 (same-angle points)
    k_demo = 3.0
    z_demo = circle_points(center=center, radius=k_demo, n=8)
    w_demo = invert_points(z_demo, R)
    ax.scatter(z_demo.real, z_demo.imag, c='tab:blue', s=32, zorder=5)
    ax.scatter(w_demo.real, w_demo.imag, c='tab:red', s=32, zorder=5)
    for zp, wp in zip(z_demo, w_demo):
        ax.plot([zp.real, wp.real], [zp.imag, wp.imag], color='0.5', lw=0.8, alpha=0.7)

    ax.scatter([0], [0], color='black', s=24, zorder=6)
    ax.text(0.2, 0.2, 'O', color='black', fontsize=10)

    ax.set_title('Circle -> Circle under Geometric Inversion (k = 1..5)')
    ax.set_xlabel('Real axis')
    ax.set_ylabel('Imag axis')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    out = 'circle_to_circle_inversion_k1_to_k5.png'
    plt.savefig(out, dpi=180)
    print(f'Saved: {out}')

    if os.environ.get('DISPLAY') and plt.get_backend().lower() != 'agg':
        plt.show()


if __name__ == '__main__':
    main()
