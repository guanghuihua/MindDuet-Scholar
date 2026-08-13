import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equations
def system(t, vars, r1, r2, c, a, b, d):
    x, y = vars
    dxdt = x * (r1 - a*x - b*y)
    dydt = y * (-r2 + c*x - d*y)
    return [dxdt, dydt]

# Define the parameters
r1 = 1
r2 = 1
c = 2
a = 1
b = 0.5
d = 1

# Define the initial conditions
initial_conditions = [[2.8, 0.25], [2.8, 0.5], [2.8, 1.0], [2.8, 1.8], [2.8, 2.4],
                      [1.5, 0.2], [0.1, 2.9], [1.2, 2.9], [0, 2.9], [2.9, 0],
                      [0.01, 0], [2.5, 2.9], [0.7, 2.9]]

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate through initial conditions
for init_cond in initial_conditions:
    # Solve the system of differential equations
    sol = solve_ivp(system, [0, 50], init_cond, args=(r1, r2, c, a, b, d), dense_output=True,
                    t_eval=np.linspace(0, 50, 100000))

    # Plot the trajectory
    ax.plot(sol.y[0], sol.y[1], '-')

# Set axis limits
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

# Set labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# Show grid
ax.grid(True)

plt.show()
