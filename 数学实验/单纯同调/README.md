# 单纯同调实验

这个目录是一套配合代数拓扑学习的最小可运行实验：

- `simplicial.py`：单纯复形、边缘矩阵、Smith 标准形和同调群计算。
- `viz.py`：单纯复形、粘合多边形和经典曲面的可视化。
- `topology_homology.ipynb`：按概念主线组织的交互式导览。
- `build_notebook.py`：重新生成 Notebook 的脚本。

从项目根目录重新生成 Notebook：

```bash
.venv/bin/python 数学实验/单纯同调/build_notebook.py
```

脚本总是把结果写回当前目录中的 `topology_homology.ipynb`，不会在运行命令的目录留下散落文件。

打开 Notebook 前先按上级 [`README.md`](../README.md) 安装 `math` 可选依赖，并以这个目录作为 Jupyter 工作目录，使 `simplicial` 和 `viz` 可以直接导入。
