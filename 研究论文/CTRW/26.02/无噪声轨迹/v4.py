import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import odeint

# 1. Base parameters
delta = 0.1
# Hopf bifurcation reference value
(a_base) = 1 - delta/8 - 3*(delta**2)/32 - 173*(delta**3)/1024

def derivatives(state, t, a):
    x, y = state
    dxdt = (y - x**3/3 + x) / delta
    dydt = a - x
    return [dxdt, dydt]

# 2. Figure setup
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.22)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot([], [], 'b-', lw=1.5, label='Trajectory')
point, = ax.plot([], [], 'go', label='Equilibrium')
a_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                 ha='left', va='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Static x-nullcline
x_vals = np.linspace(-2.2, 2.2, 200)
y_vals = x_vals**3/3 - x_vals
ax.plot(x_vals, y_vals, 'k--', alpha=0.2, label='x-nullcline')

# 3. Solve and update
def solve_and_update(a):
    t = np.linspace(0, 40, 2000)
    init_state = [a, a**3/3 - a + 0.01]
    sol = odeint(derivatives, init_state, t, args=(a,))
    line.set_data(sol[1000:, 0], sol[1000:, 1])
    point.set_data([a], [a**3/3 - a])
    ax.set_title(f'Canard Explosion (delta={delta})')
    a_text.set_text(f'a = {a:.6f}')
    fig.canvas.draw_idle()

# 4. Slider for a
ax_a = plt.axes([0.15, 0.08, 0.7, 0.04])
a_slider = Slider(
    ax=ax_a,
    label='a',
    valmin=a_base - 0.01,
    valmax=a_base + 0.01,
    valinit=a_base,
    valstep=0.0002
)

def on_a_change(val):
    solve_and_update(a_slider.val)

a_slider.on_changed(on_a_change)

# Initial draw
solve_and_update(a_base)

plt.xlabel('x (Fast Variable)')
plt.ylabel('y (Slow Variable)')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.show()
