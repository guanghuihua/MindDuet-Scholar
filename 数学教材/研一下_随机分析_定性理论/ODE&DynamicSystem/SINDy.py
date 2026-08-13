import numpy as np
import pysindy as ps

t = np.linspace(0, 1, 100)
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1)

# 拟合模型
model = ps.SINDy(
    differentiation_method=ps.FiniteDifference(order=2),
    feature_library=ps.FourierLibrary(),
    optimizer=ps.STLSQ(threshold=0.2),
    feature_names=["x", "y"]
)
model.fit(X, t=t)

model.print()

'''

(x)' = 0.772 sin(1 x) + 2.097 cos(1 x) + -2.298 sin(1 y) + -3.115 cos(1 y)
(y)' = 1.362 sin(1 y) + -0.222 cos(1 y)

'''

import matplotlib.pyplot as plt

def plot_simulation(model, x0, y0):
    t_test = np.linspace(0, 1, 100)
    x_test = x0 * np.exp(-2 * t_test)
    y_test = y0 * np.exp(t_test)

    sim = model.simulate([x0, y0], t=t_test)

    plt.figure(figsize=(6, 4))
    plt.plot(x_test, y_test, label="Ground truth", linewidth=4)
    plt.plot(sim[:, 0], sim[:, 1], "--", label="SINDy estimate", linewidth=3)
    plt.plot(x0, y0, "ko", label="Initial condition", markersize=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

x0 = 6
y0 = -0.1
plot_simulation(model, x0, y0)



# ---------------------------------------------------------
## 自定义基函数

# 导入所需的库（假设 ps 是 PySINDy 的别名）  
# 注意：PySINDy 库的实际导入可能需要根据你的环境进行调整  
from pysindy import SINDy, FiniteDifference, PolynomialLibrary, STLSQ  
#from pysindy.utils import plot_simulation  
  
# 初始化一个 SINDy 模型，使用有限差分法（二阶）进行微分  
# 这里使用了多项式库（一阶）作为特征库  
# 使用了阈值为 0.2 的 STLSQ 优化器  
# 特征名称设置为 'x' 和 'y'  
model_1 = SINDy(  
    differentiation_method=FiniteDifference(order=2),  # 使用二阶有限差分法进行微分  
    feature_library=PolynomialLibrary(degree=1),       # 使用一阶多项式库作为特征库  
    optimizer=STLSQ(threshold=0.2),                    # 使用阈值为 0.2 的 STLSQ 优化器  
    feature_names=["x", "y"]                           # 特征名称设置为 'x' 和 'y'  
)  
  
# 使用数据 X 和时间 t 拟合模型  
model_1.fit(X, t=t)  
  
# 打印模型结果  
model_1.print()  
  
# 假设 plot_simulation 是一个自定义函数，用于绘制模拟结果  
# 这里设置初始条件 x0=6, y0=-0.1  
x0 = 6  
y0 = -0.1  
  
# 调用 plot_simulation 函数绘制模拟结果  
plot_simulation(model_1, x0, y0)


# 导入 PySINDy 相关的库（假设 ps 是 PySINDy 的别名）  
from pysindy import SINDy, FiniteDifference, PolynomialLibrary, STLSQ  
#from pysindy.utils import plot_simulation  # 假设 plot_simulation 函数已经定义或可用  
  

#-----------------------------------------------------------------
# 初始化一个 SINDy 模型  
# 使用二阶有限差分法进行微分  
# 使用四阶多项式库作为特征库  
# 使用阈值为 0.2 的 STLSQ 优化器  
# 特征名称设置为 'x' 和 'y'  
model_1 = ps.SINDy(  
    differentiation_method=ps.FiniteDifference(order=2),  # 使用二阶有限差分法进行微分  
    feature_library=ps.PolynomialLibrary(degree=4),       # 使用四阶多项式库作为特征库  
    optimizer=ps.STLSQ(threshold=0.2),                    # 使用阈值为 0.2 的 STLSQ 优化器  
    feature_names=["x", "y"]                              # 特征名称设置为 'x' 和 'y'  
)  
  
# 使用数据 X 和时间 t 拟合模型  
model_1.fit(X, t=t)  
  
# （注意：以下部分是错误的，因为不应该再次初始化 SINDy，并且没有缩进）  
# 正确的方式是上面已经初始化和拟合了 model_1，接下来直接使用它  
# SINDy(differentiation_method=FiniteDifference(),  
#       feature_library=PolynomialLibrary(degree=4), feature_names=['x', 'y'],  
#       optimizer=STLSQ(threshold=0.2))  
  
# 打印模型结果  
model_1.print()  
  
# 设置初始条件 x0=6, y0=-0.1  
x0 = 6  
y0 = -0.1  
  
# 调用 plot_simulation 函数绘制模拟结果  
plot_simulation(model_1, x0, y0)

#-----------------------------------------------------------------------
## 4. 洛伦兹吸引子实验

import numpy as np  
import matplotlib.pyplot as plt  
from scipy.integrate import odeint  
from mpl_toolkits.mplot3d import Axes3D  
# 假设 ps 是 PySINDy 的别名  
from pysindy import SINDy, STLSQ, PolynomialLibrary  
  
# Lorenz 吸引子的参数  
rho = 28.0  
sigma = 10.0  
beta = 8.0 / 3.0  
dt = 0.01  
  
# Lorenz 系统的动力学方程  
def f(state, t):  
    x, y, z = state  
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]  
  
# 初始状态  
state0 = [1.0, 1.0, 1.0]  
# 时间步长数组  
time_steps = np.arange(0.0, 40.0, dt)  
  
# 模拟 Lorenz 吸引子的轨迹  
x_train = odeint(f, state0, time_steps)  
  
# 初始化 SINDy 模型  
model = SINDy(  
    optimizer=STLSQ(threshold=0.05),  # 使用阈值为 0.05 的 STLSQ 优化器  
    feature_library=PolynomialLibrary(degree=2),  # 使用二阶多项式库  
)  
  
# 使用模拟的数据拟合 SINDy 模型  
model.fit(x_train, t=dt)  
  
# 使用 SINDy 模型进行模拟  
x_sim = model.simulate(x_train[0], time_steps)  
  
# 打印 SINDy 模型的识别结果  
model.print()  
  
# 绘制 Lorenz 吸引子的轨迹  
plt.figure(figsize=(6, 4))  
# 绘制真实轨迹  
plt.plot(x_train[:, 0], x_train[:, 2], label='ground truth')  
# 绘制 SINDy 估计的轨迹  
plt.plot(x_sim[:, 0], x_sim[:, 2], '--', label='SINDy estimate')  
# 绘制初始条件  
plt.plot(x_train[0, 0], x_train[0, 2], "ko", label="initial condition", markersize=8)  
plt.legend()  
plt.xlabel('x')  
plt.ylabel('z')  
plt.title('Lorenz Attractor')  
plt.draw()  
plt.show()


import matplotlib.pyplot as plt

# 定义一个函数，用于绘制特定维度的时间序列数据
def plot_dimension(dim, name):
    # 创建一个图形对象，设置图形大小为9x2英寸
    fig = plt.figure(figsize=(9, 2))
    # 获取当前的坐标轴对象
    ax = fig.gca()
    # 在坐标轴上绘制训练数据的时间序列
    ax.plot(time_steps, x_train[:, dim])
    # 在相同的坐标轴上绘制模拟数据的时间序列，使用虚线表示
    ax.plot(time_steps, x_sim[:, dim], "--")
    # 设置x轴标签为"时间"
    plt.xlabel("时间")
    # 设置y轴标签为维度名称
    plt.ylabel(name)
    plt.draw()
    plt.show()

# 调用函数绘制x、y、z三个维度的时间序列数据
plot_dimension(0, 'x')  # 绘制x维度
plot_dimension(1, 'y')  # 绘制y维度
plot_dimension(2, 'z')  # 绘制z维度
