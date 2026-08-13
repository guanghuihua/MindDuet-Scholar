import numpy as np
import matplotlib.pyplot as plt

# Define the augmented system of differential equations
def augmented_system(t, y):
    x, y, z, w = y
    dxdt = -y
    dydt = x
    dzdt = 1  # Additional equation for the augmented variable z
    dwdt = 0  # Additional equation for the augmented variable w
    return [dxdt, dydt, dzdt, dwdt]

# Define the time range for integration
t = np.linspace(0, 10, 1000)

# Integrate the system to obtain the trajectory
from scipy.integrate import solve_ivp
y0 = [1, 0, 0, 0]  # Initial conditions
sol = solve_ivp(augmented_system, [t[0], t[-1]], y0, t_eval=t)

# Plot the trajectory in the augmented phase space
fig = plt.figure(figsize=(10, 8))
ax = fig.gca(projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], label='Trajectory', color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Augmented Phase Space')
plt.show()
