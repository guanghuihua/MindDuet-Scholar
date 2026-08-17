# 反演几何实验

这里保存复平面向量场与圆反演的计算实验。静态脚本会把结果统一写入 `figures/`；交互脚本使用 Matplotlib slider 调节参数。

例如，从项目根目录运行：

```bash
MPLBACKEND=Agg .venv/bin/python 数学实验/反演几何/line_to_circle_inversion.py
.venv/bin/python 数学实验/反演几何/line_to_circle_inversion_slider.py
```

主要脚本：

- `line_to_circle_inversion.py`：直线在反演下变为过原点的圆。
- `line_to_circle_inversion_slider.py`：交互改变反演半径与直线位置。
- `circle_to_circle_inversion_k_series.py`：一组圆的反演像。
- `disk_inversion_experiment.py`：圆盘包含原点或边界经过原点时的对比。
- `vector_field_z2.py`：复函数 \(f(z)=z^2\) 的向量场。
