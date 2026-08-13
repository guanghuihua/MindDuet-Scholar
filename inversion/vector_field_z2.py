import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Grid on the complex plane: z = x + i y
    n = 29
    x = np.linspace(-2.0, 2.0, n)
    y = np.linspace(-2.0, 2.0, n)
    X, Y = np.meshgrid(x, y)

    # f(z) = z^2 = (x^2 - y^2) + i(2xy)
    U = X**2 - Y**2
    V = 2 * X * Y
    M = np.hypot(U, V)

    fig, ax = plt.subplots(figsize=(8, 8))
    q = ax.quiver(
        X,
        Y,
        U,
        V,
        M,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=10.0,
        width=0.004,
    )
    cb = plt.colorbar(q, ax=ax)
    cb.set_label(r"$|f(z)|$")

    ax.set_title(r"Vector Field of $f(z)=z^2$")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.grid(alpha=0.25)

    out = "vector_field_z2.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"Saved: {out}")

    if os.environ.get("DISPLAY") and plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
