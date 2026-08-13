import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. 参数设置
delta = 0.1  # 你可以尝试不同的 delta 值
# 根据你提供的公式计算 a
a = 1 - delta/8 - 3*(delta**2)/32 - 173*(delta**3)/1024 - 0.01

def system(state, t):
    x, y = state
    # dx/dt = (y - x^3/3 + x) / delta
    dxdt = (y - x**3/3 + x) / delta
    # dy/dt = a - x
    dydt = a - x
    return [dxdt, dydt]

# 2. 初始条件与时间步长
initial_state = [2.0, 0.5]  # 你可以尝试不同的初始值
t = np.linspace(0, 50, 5000) # 模拟 50 个单位时间

# 3. 执行积分
solution = odeint(system, initial_state, t)
x = solution[:, 0]
y = solution[:, 1]

# 4. 可视化
plt.figure(figsize=(12, 5))

# 子图 1: 相平面图 (Phase Portrait)
plt.subplot(1, 2, 1)
plt.plot(x, y, label='Trajectory', color='blue')
# 绘制零解曲线 (Nullclines)
x_null = np.linspace(min(x)-0.5, max(x)+0.5, 400)
y_null = x_null**3/3 - x_null
plt.plot(x_null, y_null, 'r--', alpha=0.6, label='dx/dt=0 Nullcline')
plt.axvline(x=a, color='green', linestyle='--', alpha=0.6, label='dy/dt=0 Nullcline')

plt.title(f'Phase Portrait (delta={delta}, a={a:.4f})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 子图 2: 时间序列图 (Time Series)
plt.subplot(1, 2, 2)
plt.plot(t, x, label='x(t)', color='orange')
plt.plot(t, y, label='y(t)', color='purple')
plt.title('Time Series')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()