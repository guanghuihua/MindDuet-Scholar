import numpy as np
import matplotlib.pyplot as plt

def f(x, num_terms=100):
    # Calculate the constant term
    sum_result = np.pi**2 / 12
    
    # Sum the series up to num_terms
    for n in range(1, num_terms + 1):
        sum_result += np.cos(n * x) / n**2
    
    return sum_result

# Generate x values
x = np.linspace(0, 2 * np.pi, 1000)

def g(x):
    return (np.pi - x)**2/4

# Calculate y values for the function f(x)
y1 = f(x, num_terms=3)
y2 = g(x)

# Plot the function
plt.plot(x, y1)
plt.plot(x, y2)
plt.title(r'$f(x) = \frac{\pi^2}{12} + \sum_{n=1}^{\infty} \frac{\cos(nx)}{n^2}$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
