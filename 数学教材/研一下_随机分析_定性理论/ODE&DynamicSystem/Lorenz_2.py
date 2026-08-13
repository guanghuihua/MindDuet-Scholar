import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

t_span = (0, 50)  # Time interval
initial_state = [1.0, 1.0, 1.0]  # Initial conditions
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Time points to evaluate

sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = sol.y[0]
y = sol.y[1]
z = sol.y[2]

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.title("Lorenz Attractor")
plt.show()
'''

import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(num, x, y, z, line, step):
    line.set_data(x[:num*step], y[:num*step])
    line.set_3d_properties(z[:num*step])
    return line,

line, = ax.plot(x, y, z, lw=0.5)

step = 10  # Plot every 10th point
ani = animation.FuncAnimation(fig, update, frames=len(x)//step, fargs=[x, y, z, line, step])
# ani = animation.FuncAnimation(fig, update, frames=len(x)//step, fargs=[x, y, z, line, step], interval=10, blit=False)

plt.show()

