import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ode41(a, b, c, d):
    # Define the matrix
    A = np.array([[a, b], [c, d]])
    print('所给的矩阵为:\n', A)
    
    # Define the differential equations
    print('所给的微分方程组为：')
    print(f'dx/dt = {a} * x(t) + {b} * y(t)')
    print(f'dy/dt = {c} * x(t) + {d} * y(t)')
    
    # Calculate p, q, and delta
    p = a + d
    q = a * d - b * c
    delta = p**2 - 4 * q
    print(f'p = {p}, q = {q}, p^2 - 4*q = {delta}')
    
    # Determine the nature of the singular point
    if q == 0:
        print('该线性系统的奇点是高阶奇点')
    if p == 0 and q > 0 and delta < 0:
        print('该线性系统的奇点是中心')
    if p < 0 and q > 0 and delta < 0:
        print('该线性系统的奇点是稳定焦点')
    if p > 0 and q > 0 and delta < 0:
        print('该线性系统的奇点是不稳定焦点')
    if p < 0:
        print('该线性系统的奇点是鞍点')
    if q > 0 and delta > 0 and p < 0:
        print('该线性系统的奇点是稳定结点')
    if q > 0 and delta > 0 and p > 0:
        print('该线性系统的奇点是不稳定结点')
    if q > 0 and delta == 0 and p < 0:
        print('该线性系统的奇点是稳定退化结点或临界结点')
    if q > 0 and delta == 0 and p > 0:
        print('该线性系统的奇点是不稳定退化结点或临界结点')

    # Define the system of ODEs
    def system(t, Y):
        x, y = Y
        dxdt = a * x + b * y
        dydt = c * x + d * y
        return [dxdt, dydt]
    
    # Plot phase portrait
    fig, ax = plt.subplots()
    
    # Set axis limits
    # xlim, ylim = [-10, 10], [-10, 10]
    xlim, ylim = [-20, 20], [-20, 20]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Create a grid of initial conditions
    x = np.linspace(xlim[0], xlim[1], 20)
    y = np.linspace(ylim[0], ylim[1], 20)
    X, Y = np.meshgrid(x, y)
    
    # Compute the direction field
    U = a * X + b * Y
    V = c * X + d * Y
    ax.quiver(X, Y, U, V, color='r')

    # Solve ODE for different initial conditions and plot trajectories
    for x0 in np.linspace(xlim[0], xlim[1], 5):
        for y0 in np.linspace(ylim[0], ylim[1], 5):
            sol = solve_ivp(system, [0, 10], [x0, y0], t_eval=np.linspace(0, 10, 100))
            ax.plot(sol.y[0], sol.y[1], 'b')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Portrait')
    plt.grid()
    plt.show()

# Example usage:
ode41(1, 1, 2, -3)
ode41(-0.05, 1, -1, -0.05)
ode41(1, 3, 1, -1)
ode41(0, 1, -1, 0)
