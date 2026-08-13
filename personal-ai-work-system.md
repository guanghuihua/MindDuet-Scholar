# 个人 AI 开发工作系统

本文总结当前项目以及后续项目推荐采用的个人 AI 开发工作系统。这个系统的目标不是把所有事情交给一个模型，而是把项目资料、代码目录、agent 工具、API 密钥、项目规则和测试命令组织成一套稳定的工作方式。

当前核心项目是 MindDuet Math：一个面向高中数学学习的教师-AI 协作系统。项目代码位于 WSL 的 Linux 文件系统中，教学资料位于 D 盘的教师资料库中。AI agent 的任务是在不破坏资料边界和产品边界的前提下，帮助完成资料整理、题库导入、判题、诊断、训练推荐、报告生成和 Web 应用开发。

## 1. 项目目的

`/mnt/d/guanghui/teaching` 是教师长期积累的教学知识库，而 `mindduet-math` 是围绕这些资料建立的学习系统。

`teaching` 目录当前包含：

- 课程标准与总纲，例如课标索引、核心素养索引、教学内容映射表
- 知识点索引，例如三角恒等变换、圆锥曲线、复数、平面向量、解三角形
- 题库积累，例如三角函数、圆锥曲线、数列、最值问题、解三角形、高中好题
- 教学课件，例如讲评试卷、复数几何表示、三角恒等变换复习
- 高考真题和分类整理资料
- 自己学习和教学的思考，例如数学学习方法、费曼式理解、教育原则、成长与评价
- 参考书与外部资料
- 大量 TeX、PDF、Word、图片、Markdown 等文件

从文件类型看，资料库大致包含：

```text
TeX/PDF/Word/图片/Markdown/JSON/CSV/课件
```

因此，MindDuet Math 的真实目标不是简单做一个网页题库，而是把教师长期积累的资料逐步转化为一个可追溯、可审核、可反馈、可长期积累学生证据的学习系统。

系统应坚持：

- 原始教学资料保持只读
- 题目、知识点、答案、解析和来源必须可追溯
- 判题不能只依赖大语言模型
- 诊断和建议必须回到学生作答证据
- 教师可以审核、修正和覆盖自动判断
- AI 不可用时，核心应用仍然可运行

## 2. Agent 应该如何使用

Agent 不等于模型。一个 coding agent 至少包含四层：

```text
界面层
VS Code extension / CLI / Web UI

Agent 外壳
负责读文件、选择上下文、生成计划、调用工具、改代码、运行测试、查看错误

模型 API
OpenAI / DeepSeek / Claude / PackyAPI / 其他兼容 API

工具权限
文件读写、终端命令、git diff、测试命令、MCP、数据库等
```

替换 API 会影响 agent，因为不同 API 对工具调用、流式输出、长上下文、patch 编辑、Responses API、Anthropic API、多轮状态管理的支持程度不同。一个 API 能聊天，不代表它能稳定支撑复杂开发 agent。

因此，agent 应按任务风险分层使用。

### A 级任务：正式开发

使用：

```text
VS Code Remote WSL
+ Codex extension
+ PackyAPI / OpenAI 模型
```

适合：

- 多文件功能开发
- 数据库模型和 Alembic 迁移
- 路由、服务层、模板联动修改
- 判题逻辑、诊断逻辑、报告逻辑
- 学生数据隔离和权限边界
- 测试失败分析与修复
- 需要可视化 diff 和人工 review 的改动

要求：

- 从 WSL 项目目录打开 VS Code
- 修改前明确任务范围
- 修改后查看 `git diff`
- 能跑测试就跑测试
- 不让低成本模型直接承担高风险修改的最终落地

### B 级任务：普通修改

使用：

```text
CLI agent
+ DeepSeek API
```

适合：

- 解释一个文件
- 总结一个报错
- 起草测试用例
- 生成文档草稿
- 小范围单文件修改
- 初步 review `git diff`
- 梳理开发计划

要求：

- 控制上下文范围
- 不自动提交
- 修改后自己看 diff
- 必要时再交给 Codex extension 复查

### C 级任务：咨询和草稿

使用：

```text
CLI agent 或普通 chat
+ DeepSeek API
```

适合：

- 想法探索
- 资料摘要
- 方案比较
- prompt 草稿
- issue/TODO 草稿
- 教学材料初步分类

要求：

- 输出只作为参考
- 不直接覆盖原始教学资料
- 进入正式代码前必须经过项目规则和测试检查

## 3. 当前方案

主力开发使用 VS Code 界面：

```text
/home/guanghui/.codex/
    config.toml          # Codex extension/CLI 的全局配置，接 PackyAPI/OpenAI 模型
    models.json          # 可选：模型元数据和自定义模型说明
```

日常低成本命令行使用：

```text
/home/guanghui/.claude/
    settings.json        # Claude Code 的全局配置，如果它支持当前 DeepSeek 接法
```

API 密钥统一放全局私密环境文件或 shell 环境变量：

```text
/home/guanghui/.env
    DEEPSEEK_API_KEY=...
    PACKY_API_KEY=...
```

也可以在 shell 启动文件中导出：

```bash
export DEEPSEEK_API_KEY="..."
export PACKY_API_KEY="..."
```

但无论采用哪种方式，都不要把真实 API key 写入项目仓库、README、AGENTS.md、提交历史或教学资料目录。

推荐使用分工：

```text
正式开发、复杂修改、可视化 review
-> VS Code + Codex extension + PackyAPI/OpenAI

普通咨询、小改动、草稿生成、低成本探索
-> CLI + Claude Code/Aider/OpenCode + DeepSeek API
```

这套方案的好处是：

- 关键任务使用更可靠的模型和 VS Code 可视化界面
- 日常任务使用更低成本的 DeepSeek API
- 所有开发都发生在 WSL 中，减少路径、权限、虚拟环境和性能问题
- 每个项目通过 AGENTS.md 和 README 提供项目级记忆
- 全局配置只负责工具和模型，不混入项目规则

## 4. 文件存储方式

推荐存储边界：

```text
/home/guanghui/projects/
    mindduet-math/
    mindduet-scholar/
    其他开发项目/
```

WSL 项目目录保存：

- 源码
- `.git`
- `.venv`
- `pyproject.toml`
- `uv.lock`
- 测试代码
- 迁移文件
- 活跃 SQLite 数据库
- 项目生成的规范化内容和导入报告

教师资料库保存在 D 盘：

```text
/mnt/d/guanghui/teaching/
```

该目录保存：

- 原始 TeX
- 原始 PDF
- Word 文档
- 图片
- 课件
- 高考真题
- 教学反思
- 知识点索引
- 参考书和外部资料

这个目录应作为只读资料源。Agent 可以读取、盘点、引用、生成报告，但不应重命名、移动、删除、格式化或覆盖其中的文件。

项目外部数据保存在 D 盘独立数据目录：

```text
/mnt/d/guanghui/mindduet-math-data/
    student_uploads/
    ocr/
    reports/
    database_backups/
    exports/
```

这个目录保存运行时产生的大文件和用户数据，例如学生上传、OCR 中间产物、PDF 报告、数据库备份和导出文件。它不应混入源码仓库。

推荐总布局：

```text
代码和开发环境
-> /home/guanghui/projects/项目名

教师原始资料
-> /mnt/d/guanghui/teaching

项目运行数据和导出
-> /mnt/d/guanghui/项目名-data
```

## 5. 每个项目应具备的文件

每个项目都应有自己的项目级说明文件。它们共同构成 agent 的项目记忆。

```text
AGENTS.md
README.md
pyproject.toml / package.json
测试命令
项目边界说明
```

### AGENTS.md

用途：

- 写给 agent 的项目规则
- 定义可读写边界
- 定义不能违反的产品原则
- 定义开发工具和测试要求

MindDuet Math 中，AGENTS.md 应说明：

- `/mnt/d/guanghui/teaching` 是只读教师资料库
- 源码、`.venv`、活跃数据库放 WSL 项目目录
- 上传、OCR、报告、备份、导出放 `/mnt/d/guanghui/mindduet-math-data`
- AI 不能作为唯一判卷人
- 重要诊断必须链接证据
- 教师审核可以覆盖自动评分和诊断
- AI 不可用时核心应用仍可工作
- 使用 Python 3.12 和 `uv`
- 不提交 `.env`、API key、数据库、学生上传和生成报告

### README.md

用途：

- 写给人和 agent 的项目概览
- 说明项目目标、当前阶段、运行方式、测试命令
- 说明当前功能和未完成功能

README 应回答：

- 这个项目是什么
- 当前做到哪一步
- 如何安装依赖
- 如何初始化运行目录
- 如何启动服务
- 如何跑测试
- 当前数据和资料在哪里

### pyproject.toml / package.json

用途：

- 定义语言、依赖、测试工具、格式化工具
- 让 agent 知道项目技术栈和命令入口

Python 项目应至少包含：

```text
Python 版本
依赖
开发依赖
pytest 配置
ruff 配置
```

### 测试命令

测试命令应在 README、AGENTS.md 或项目文档中明确写出。

MindDuet Math 当前测试命令：

```bash
uv run ruff check .
uv run pytest
```

如果涉及数据库迁移或运行初始化，还应使用：

```bash
uv run python -m scripts.init_runtime
```

### 项目边界说明

项目边界说明可以在 AGENTS.md、README 或独立文档中维护。它应回答：

- 哪些目录可以改
- 哪些目录只能读
- 哪些文件不能提交
- 哪些判断必须由确定性规则完成
- 哪些结果必须教师审核
- 数据如何追溯到来源
- AI 失败时系统如何降级

## 6. 工作系统总览

最终工作系统由全局层和项目层组成。

全局层：

```text
~/.codex/config.toml        # 全局 agent 配置
~/.codex/models.json        # DeepSeek 或其他模型的元数据，可选
~/.claude/settings.json     # Claude Code 全局配置，可选
~/.aider.conf.yml           # Aider 全局配置，可选
~/.env 或 shell env         # API key
```

项目层：

```text
每个项目/AGENTS.md          # 项目专属规则
每个项目/README.md          # 项目说明
每个项目/pyproject.toml     # Python 项目依赖和工具配置
每个项目/package.json       # Node 项目依赖和脚本配置
每个项目测试命令             # pytest / ruff / npm test / pnpm test 等
每个项目边界说明             # 数据边界、产品边界、资料边界
```

资料层：

```text
/mnt/d/guanghui/teaching
    教师原始资料，只读

/mnt/d/guanghui/*-data
    项目运行数据、生成报告、上传、OCR、备份、导出
```

开发层：

```text
/home/guanghui/projects/*
    源码、git、虚拟环境、测试、活跃开发数据库
```

使用流程：

```bash
cd /home/guanghui/projects/mindduet-math
code .
```

在 VS Code 中使用 Codex extension 做正式开发。

日常低成本咨询：

```bash
cd /home/guanghui/projects/mindduet-math
claude
```

或使用当前配置好的其他 CLI agent。

每次修改后：

```bash
git status
git diff
uv run ruff check .
uv run pytest
```

## 7. 对 agent 的基本要求

无论使用哪个 agent，都应遵守：

- 先读项目规则，再改文件
- 不直接修改 `/mnt/d/guanghui/teaching`
- 不提交 API key、`.env`、数据库、学生上传和生成报告
- 修改代码后检查 diff
- 涉及核心逻辑时补测试
- 不让 LLM 单独决定数学判题结果
- 不把一次学生错误变成永久标签
- 诊断、建议和报告必须能追溯到证据
- 教师审核优先于自动评分和自动诊断

这套规则让 agent 成为长期的小助手，而不是一次性的聊天窗口。它通过全局配置获得工具和模型，通过项目文档获得边界和目标，通过测试命令获得反馈，通过 git diff 保持可审查。

## 8. 当前推荐结论

当前最适合的工作系统是：

```text
WSL 负责开发现场
D 盘负责资料库和大文件归档
VS Code + Codex extension + PackyAPI/OpenAI 负责正式开发
CLI + DeepSeek API 负责低成本日常任务
AGENTS.md + README.md + 测试命令负责项目记忆
git diff + pytest/ruff 负责最后的现实校验
```

这不是工具堆叠，而是一套分工明确的工作台。关键任务交给可靠链路，普通任务交给低成本链路，项目边界写在文件里，资料边界固定在目录结构里。这样 agent 才能真正成为跨项目、可持续、可审查的个人开发助手。
