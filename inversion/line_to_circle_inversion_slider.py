import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def invert_points(z, R):
    # Geometric circle inversion: O, z, w are collinear.
    return (R ** 2) / np.conjugate(z)


def line_points(c, xlim=(-20, 20), n=2000):
    x = np.linspace(xlim[0], xlim[1], n)
    return x + 1j * c


def sampled_points(c, x_samples):
    return np.array(x_samples) + 1j * c


def build_plot():
    # Initial parameters
    R0 = 3.0
    c0 = 2.0
    xlim = (-20, 20)

    # Sample x-positions on the line to show pointwise mapping z_k -> w_k
    x_samples = np.array([-12, -6, -3, 0, 3, 6, 12], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.22)

    # Base geometry
    theta = np.linspace(0, 2 * np.pi, 800)

    # Initialize curves
    z_line = line_points(c0, xlim=xlim)
    w_line = invert_points(z_line, R0)

    (line_z,) = ax.plot(z_line.real, z_line.imag, color='tab:blue', lw=2, label='Original line Im(z)=c')
    (line_w,) = ax.plot(w_line.real, w_line.imag, color='tab:red', lw=2, label='Mapped curve w=R^2/conj(z)')

    # Inversion circle
    ref_x = R0 * np.cos(theta)
    ref_y = R0 * np.sin(theta)
    (inv_circle,) = ax.plot(ref_x, ref_y, color='gray', lw=1.2, alpha=0.9, label='Inversion circle |z|=R')

    # Theoretical mapped circle
    center_y0 = (R0 ** 2) / (2 * c0)
    radius0 = (R0 ** 2) / (2 * abs(c0))
    cir_x = radius0 * np.cos(theta)
    cir_y = center_y0 + radius0 * np.sin(theta)
    (theory_circle,) = ax.plot(cir_x, cir_y, 'k--', lw=1.2, label='Theoretical image circle')

    # Sampled points and connectors
    z_pts = sampled_points(c0, x_samples)
    w_pts = invert_points(z_pts, R0)

    scat_z = ax.scatter(z_pts.real, z_pts.imag, c='tab:blue', s=45, zorder=5)
    scat_w = ax.scatter(w_pts.real, w_pts.imag, c='tab:red', s=45, zorder=5)

    connectors = []
    w_labels = []
    z_labels = []
    for i, (zp, wp) in enumerate(zip(z_pts, w_pts), start=1):
        (conn,) = ax.plot([zp.real, wp.real], [zp.imag, wp.imag], color='0.4', lw=0.9, alpha=0.8)
        connectors.append(conn)

        w_txt = ax.text(
            wp.real,
            wp.imag,
            f'w{i}',
            fontsize=8,
            color='tab:red',
            ha='left',
            va='bottom',
        )
        w_labels.append(w_txt)

        z_txt = ax.text(
            zp.real,
            zp.imag,
            f'z{i}',
            fontsize=8,
            color='tab:blue',
            ha='right',
            va='bottom',
        )
        z_labels.append(z_txt)

    # Origin
    ax.scatter([0], [0], color='black', s=20, zorder=6)

    # Axes style
    ax.set_title('Complex Inversion with Pointwise Mapping (single graph)')
    ax.set_xlabel('Real axis')
    ax.set_ylabel('Imag axis')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-22, 22)
    ax.set_ylim(-22, 22)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Sliders
    ax_R = plt.axes([0.16, 0.10, 0.70, 0.03])
    ax_c = plt.axes([0.16, 0.05, 0.70, 0.03])
    s_R = Slider(ax_R, 'R', 0.6, 8.0, valinit=R0, valstep=0.1)
    s_c = Slider(ax_c, 'c (Im line)', -8.0, 8.0, valinit=c0, valstep=0.1)

    note = ax.text(
        -21.5,
        -20.7,
        'Mapping: z_k -> w_k = R^2 / conj(z_k) (same index k)',
        fontsize=9,
        color='0.2',
    )
    mapping_text = fig.text(
        0.02,
        0.98,
        '',
        va='top',
        ha='left',
        fontsize=8,
        family='monospace',
    )

    def render_mapping_table(z_vals, w_vals, R):
        lines = [f'R={R:.2f}   mapping: w_k = R^2 / conj(z_k)']
        for idx, (zz, ww) in enumerate(zip(z_vals, w_vals), start=1):
            lines.append(
                f'k={idx}: z={zz.real:6.2f}{zz.imag:+6.2f}i -> '
                f'w={ww.real:7.3f}{ww.imag:+7.3f}i'
            )
        mapping_text.set_text('\\n'.join(lines))

    render_mapping_table(z_pts, w_pts, R0)

    def update(_):
        R = float(s_R.val)
        c = float(s_c.val)

        # Avoid c = 0 singular case for line through origin (maps to a line, not this circle formula)
        if abs(c) < 1e-8:
            c = 1e-8

        z_line_new = line_points(c, xlim=xlim)
        w_line_new = invert_points(z_line_new, R)
        line_z.set_data(z_line_new.real, z_line_new.imag)
        line_w.set_data(w_line_new.real, w_line_new.imag)

        inv_circle.set_data(R * np.cos(theta), R * np.sin(theta))

        center_y = (R ** 2) / (2 * c)
        radius = (R ** 2) / (2 * abs(c))
        theory_circle.set_data(radius * np.cos(theta), center_y + radius * np.sin(theta))

        z_new = sampled_points(c, x_samples)
        w_new = invert_points(z_new, R)
        scat_z.set_offsets(np.column_stack([z_new.real, z_new.imag]))
        scat_w.set_offsets(np.column_stack([w_new.real, w_new.imag]))

        for i, (zp, wp) in enumerate(zip(z_new, w_new)):
            connectors[i].set_data([zp.real, wp.real], [zp.imag, wp.imag])
            w_labels[i].set_position((wp.real, wp.imag))
            z_labels[i].set_position((zp.real, zp.imag))

        if abs(s_c.val) < 1e-8:
            note.set_text('c=0: line passes origin, inversion image is a line (not a circle)')
        else:
            note.set_text('Mapping: z_k -> w_k = R^2 / conj(z_k) (same index k)')

        render_mapping_table(z_new, w_new, R)
        fig.canvas.draw_idle()

    s_R.on_changed(update)
    s_c.on_changed(update)

    plt.show()


if __name__ == '__main__':
    build_plot()
