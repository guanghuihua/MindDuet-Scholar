import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# 1. 基础参数定义
delta = 0.1
# 理论上的 Hopf 分岔临界值附近的基准
a_base = 1 - delta/8 - 3*(delta**2)/32 - 173*(delta**3)/1024

def derivatives(state, t, a):
    x, y = state
    dxdt = (y - x**3/3 + x) / delta
    dydt = a - x
    return [dxdt, dydt]

# 2. 设置画布
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot([], [], 'b-', lw=1.5, label='Trajectory')
nullcline, = ax.plot([], [], 'r--', alpha=0.5, label='x-nullcline')
point, = ax.plot([], [], 'go', label='Equilibrium')

# 绘制静态的 x-nullcline
x_vals = np.linspace(-2.2, 2.2, 200)
y_vals = x_vals**3/3 - x_vals
ax.plot(x_vals, y_vals, 'k--', alpha=0.2)

# 3. 动画更新函数
def update(frame):
    # a 围绕 a_base 进行极其微小的变化
    # 范围从 a_base - 0.01 到 a_base + 0.01
    a = a_base + (frame - 50) * 0.0002 
    
    t = np.linspace(0, 40, 2000)
    # 初始点选在平衡点附近以快速进入极限环
    init_state = [a, a**3/3 - a + 0.01]
    sol = odeint(derivatives, init_state, t, args=(a,))
    
    # 只取后半段，观察稳定后的极限环
    line.set_data(sol[1000:, 0], sol[1000:, 1])
    point.set_data([a], [a**3/3 - a])
    ax.set_title(f'Canard Explosion: a = {a:.20f} (delta={delta})')
    return line, point

# 4. 创建动画
# frames=100 表示扫描 100 个不同的 a 值
ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

plt.xlabel('x (Fast Variable)')
plt.ylabel('y (Slow Variable)')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.show()