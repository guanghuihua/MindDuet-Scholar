# Riccati equation
import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation dy/dx = x^2 + y^2
def dy_dx(x, y):
    return x**2 + y**2

# Define the grid for plotting
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Calculate the direction at each grid point
U = 1  # Constant value for the x-component of the direction (for simplicity)
V = dy_dx(X, Y)

# Plot the direction field
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, scale=20)
plt.title('Direction Field')
plt.xlabel('x')
plt.ylabel('y')

# Optionally, plot some trajectories
# You can use a numerical solver like scipy's odeint to find the trajectories

plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the differential equation dy/dx = x^2 + y^2
def dy_dx(y, x):
    return [x**2 + y[0]**2]

# Define the x range for plotting trajectories
x_range = np.linspace(-3, 3, 100)

# Define initial conditions for the trajectories
initial_conditions = [[-3], [-2], [-1], [0], [1], [2], [3]]

# Plot the direction field
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, scale=20)
plt.title('Direction Field')
plt.xlabel('x')
plt.ylabel('y')

# Plot specific trajectories
for y0 in initial_conditions:
    y = odeint(dy_dx, y0, x_range)
    plt.plot(x_range, y[:, 0], label=f'y(0)={y0[0]}')

plt.legend()
plt.grid(True)
plt.show()
