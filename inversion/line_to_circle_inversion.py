import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Inversion circle |z| = R centered at origin.
    R = 3.0

    # A line that does NOT pass through the center: Im(z) = c, c != 0.
    c = 2.0
    x = np.linspace(-20, 20, 4000)
    z = x + 1j * c

    # Complex inversion with radius R: w = R^2 / z
    w = (R ** 2) / z

    fig, ax = plt.subplots(figsize=(8, 8))

    # Original line (z-plane coordinates)
    ax.plot(z.real, z.imag, color='tab:blue', lw=2, label=fr'Original line: Im(z)={c}')

    # Image curve after inversion (w-plane coordinates, overlaid for comparison)
    ax.plot(w.real, w.imag, color='tab:red', lw=2, label=fr'Inverted image: w={R**2:g}/z')

    # Theoretical image: circle through origin.
    # For Im(z)=c under w=R^2/z:
    # u^2 + (v + R^2/(2c))^2 = (R^2/(2c))^2
    theta = np.linspace(0, 2 * np.pi, 800)
    center = np.array([0.0, -(R ** 2) / (2 * c)])
    radius = (R ** 2) / (2 * abs(c))
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    ax.plot(circle_x, circle_y, 'k--', lw=1.5, label='Theoretical image circle')

    # Inversion reference circle |z|=R.
    ref_x = R * np.cos(theta)
    ref_y = R * np.sin(theta)
    ax.plot(ref_x, ref_y, color='gray', lw=1.2, alpha=0.8, label=fr'Inversion circle |z|={R:g}')

    ax.scatter([0], [0], color='black', s=20)
    ax.set_title('Line -> Circle under Complex Inversion (single graph)')
    ax.set_xlabel('Real axis')
    ax.set_ylabel('Imag axis')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = 'line_to_circle_inversion_R3_single_graph.png'
    plt.savefig(out, dpi=180)
    print(f'Saved: {out}')

    if os.environ.get('DISPLAY'):
        plt.show()


if __name__ == '__main__':
    main()
