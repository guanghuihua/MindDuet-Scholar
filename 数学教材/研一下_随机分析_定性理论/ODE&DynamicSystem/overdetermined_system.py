# %%
import numpy as np
import matplotlib.pyplot as plt

n = 100
L = 4
x = np.linspace(0, L, n)
f = (x ** 2).reshape(-1, 1)  # parabola with 100 data points
M = 20  # polynomial degree

phi = np.zeros((n, M))
for j in range(M):
    phi[:, j] = x ** j  # build matrix A

plt.figure(figsize=(10, 8))
for j in range(1, 5):
    fn = (x ** 2 + 0.1 * np.random.randn(n)).reshape(-1, 1)
    an = np.linalg.pinv(phi) @ fn  # least-square fit
    fna = phi @ an
    En = np.linalg.norm(f - fna) / np.linalg.norm(f)
    plt.subplot(4, 2, 4 + j)
    plt.bar(range(M), an.flatten())
    plt.title(f'Fit {j}, Error: {En:.2e}')

plt.tight_layout()
plt.show()

# %%
import numpy as np
import cvxpy as cp

n = 500
m = 100
A = np.random.rand(n, m)
b = np.random.rand(n, 1)
xdag = np.linalg.pinv(A) @ b

lam = [0, 0.1, 0.5]
for j in range(3):
    x = cp.Variable(m)
    objective = cp.Minimize(cp.norm(A @ x - b, '2') + lam[j] * cp.norm(x, '1'))
    prob = cp.Problem(objective)
    prob.solve()
    # Optionally, print or store the result
    print(f'Lambda = {lam[j]}, x = {x.value}')

# %%
import numpy as np
import cvxpy as cp

n = 500
m = 100
A = np.random.rand(n, m)
b = np.random.rand(n)
xdag = np.linalg.pinv(A) @ b

lam = [0, 0.1, 0.5]
for j in range(3):
    x = cp.Variable(m)
    objective = cp.Minimize(cp.norm(A @ x - b, '2') + lam[j] * cp.norm(x, '1'))
    prob = cp.Problem(objective)
    prob.solve()
    # Optionally, print or store the result
    print(f'Lambda = {lam[j]}, x = {x.value}')

# %%
import numpy as np
import cvxpy as cp

n = 500
m = 100
A = np.random.rand(n, m)
b = np.random.rand(n)
xdag = np.linalg.pinv(A) @ b

lam = [0, 0.1, 0.5]
for j in range(3):
    x = cp.Variable(m)
    objective = cp.Minimize(cp.norm2(A @ x - b) + lam[j] * cp.norm1(x))
    prob = cp.Problem(objective)
    prob.solve()
    # Optionally, print or store the result
    print(f'Lambda = {lam[j]}, x = {x.value}')

# %%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

n = 500
m = 100
A = np.random.rand(n, m)
b = np.random.rand(n)
xdag = np.linalg.pinv(A) @ b

lam = [0, 0.1, 0.5]

# Initialize a list to store the solutions
solutions = []

for j in range(3):
    x = cp.Variable(m)
    objective = cp.Minimize(cp.norm2(A @ x - b) + lam[j] * cp.norm1(x))
    prob = cp.Problem(objective)
    prob.solve()
    
    # Store the result
    solutions.append(x.value)

# Convert the list to a NumPy array for easier handling
solutions = np.array(solutions)

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

for idx, lam_value in enumerate(lam):
    # Bar plot
    axes[idx, 0].bar(range(m), solutions[idx], color='b')
    axes[idx, 0].set_title(f'Bar Plot for Lambda = {lam_value}')
    axes[idx, 0].set_xlabel('Index')
    axes[idx, 0].set_ylabel('Value')

    # Histogram
    axes[idx, 1].hist(solutions[idx], bins=20, color='g', edgecolor='black')
    axes[idx, 1].set_title(f'Histogram for Lambda = {lam_value}')
    axes[idx, 1].set_xlabel('Value')
    axes[idx, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# %%
