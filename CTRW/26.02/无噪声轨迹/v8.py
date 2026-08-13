import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.integrate import odeint

"""
可以自己调这两个参数：

speed = 10 控制每帧推进的步数（越大越快）
interval=10 控制每帧间隔（毫秒，越小越快）
"""

# 1. Base parameters
delta = 0.1


def compute_a_base(d):
    return 1 - d/8 - 3*(d**2)/32 - 173*(d**3)/1024


# Hopf bifurcation reference value
a_base = compute_a_base(delta)
# Offset to move a into the relaxation oscillation regime
a_offset = 0.03


def derivatives(state, t, a):
    x, y = state
    dxdt = (y - x**3/3 + x) / delta
    dydt = a - x
    return [dxdt, dydt]


# 2. Figure setup
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.28)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot([], [], 'b-', lw=1.5, label='Trajectory')
current_point, = ax.plot([], [], 'bo', ms=4, alpha=0.8, label='Current state')
equilibrium, = ax.plot([], [], 'go', label='Equilibrium')
a_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                 ha='left', va='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Static x-nullcline
x_vals = np.linspace(-2.2, 2.2, 200)
y_vals = x_vals**3/3 - x_vals
ax.plot(x_vals, y_vals, 'k--', alpha=0.2, label='x-nullcline')

# 3. Precompute solution for current a
state = {
    'a': a_base + a_offset,
    't': np.linspace(0, 40, 2000),
    'sol': None,
    'i': 0,
}
speed = 1  # number of solution steps advanced per frame


def solve_for_a(a):
    t = state['t']
    init_state = [a, a**3/3 - a + 0.01]
    sol = odeint(derivatives, init_state, t, args=(a,))
    state['a'] = a
    state['sol'] = sol
    state['i'] = 0
    equilibrium.set_data([a], [a**3/3 - a])
    a_text.set_text(f'a = {a:.6f}\ndelta = {delta:.4f}')


# 4. Animation update (grows trajectory from t=0)
def update(_frame):
    sol = state['sol']
    if sol is None:
        return line, current_point, equilibrium, a_text

    i = state['i']
    if i >= len(sol):
        i = 0
    end_i = min(i + speed, len(sol) - 1)
    line.set_data(sol[:end_i + 1, 0], sol[:end_i + 1, 1])
    current_point.set_data([sol[end_i, 0]], [sol[end_i, 1]])
    ax.set_title(f'Canard Explosion (delta={delta})')
    state['i'] = end_i + 1
    return line, current_point, equilibrium, a_text


# 5. Sliders for a and delta
ax_a = plt.axes([0.15, 0.12, 0.7, 0.04])
a_slider = Slider(
    ax=ax_a,
    label='a',
    valmin=a_base - 0.1,
    valmax=a_base + 0.1,
    valinit=a_base + a_offset,
    valstep=0.0005
)

ax_delta = plt.axes([0.15, 0.06, 0.7, 0.04])
delta_slider = Slider(
    ax=ax_delta,
    label='delta',
    valmin=0.02,
    valmax=0.3,
    valinit=delta,
    valstep=0.005
)


def on_a_change(_val):
    solve_for_a(a_slider.val)


def on_delta_change(_val):
    global delta, a_base
    delta = float(delta_slider.val)
    a_base = compute_a_base(delta)

    # Keep the offset from a_base and shift slider range accordingly
    current_offset = a_slider.val - a_base
    a_slider.valmin = a_base - 0.1
    a_slider.valmax = a_base + 0.1
    a_slider.ax.set_xlim(a_slider.valmin, a_slider.valmax)

    new_a = a_base + current_offset
    new_a = max(a_slider.valmin, min(a_slider.valmax, new_a))
    a_slider.set_val(new_a)


a_slider.on_changed(on_a_change)
delta_slider.on_changed(on_delta_change)

# Initial solve
solve_for_a(a_base + a_offset)

# 6. Start animation
ani = FuncAnimation(fig, update, interval=1, blit=True)

plt.xlabel('x (Fast Variable)')
plt.ylabel('y (Slow Variable)')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.show()
