import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of differential equations
def system(t, z):
    x1, x2 = z
    dx1_dt = -x1 + x2
    dx2_dt = 2 * x1 - 3 * x2
    return [dx1_dt, dx2_dt]

# Create a grid of x1 and x2 values
x1 = np.linspace(-5, 5, 20)
x2 = np.linspace(-5, 5, 20)
X1, X2 = np.meshgrid(x1, x2)

# Compute the derivatives at each point in the grid
U = -X1 + X2
V = 2 * X1 - 3 * X2

# Plot the direction field
plt.figure(figsize=(8, 8))
plt.quiver(X1, X2, U, V, color='r')
plt.title('Phase Portrait')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid()

# Define initial conditions for specific trajectories
initial_conditions = [(-4, 0), (-3, 3), (0, 4), (3, 3), (4, 0)]

# Integrate the ODE to plot the trajectories
t_span = np.linspace(0, 10, 200)
for x0, y0 in initial_conditions:
    sol = solve_ivp(system, [0, 10], [x0, y0], t_eval=t_span)
    plt.plot(sol.y[0], sol.y[1])

plt.show()
