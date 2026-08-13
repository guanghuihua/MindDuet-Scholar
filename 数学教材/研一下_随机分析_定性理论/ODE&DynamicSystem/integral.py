# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

X = np.linspace(0, np.pi, 1000)
Y = np.sin(X)/X
f = lambda x:np.sin(x)/x
v, err = integrate.quad(f, 0, 1000 )
print("%lf"%v)

fig, ax = plt.subplots(figsize=(8, 3))
x = np.linspace(-np.pi*10, np.pi*10, 10000)
ax.plot(x, f(x), lw=2)
ax.fill_between(x, f(x), color='green', alpha=0.5)
ax.set_xlabel("$x$", fontsize=18)
ax.set_ylabel("$f(x)$", fontsize=18)
ax.set_ylim(0, 25)
plt.show()