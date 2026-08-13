# qmd-prover 介绍

## Source

- Project: `qmd-prover`
- Repository: https://github.com/powergiant/qmd-prover
- License: MIT
- 整理方式：根据项目 README 与前面关于 AI 数学科研的讨论整理。

## 一句话理解

`qmd-prover` 的核心思路是：把自然语言数学证明写成带有明确标签、明确引用、明确依赖关系的 `.qmd` 文件，然后让程序检查结构，让独立 AI 辅助检查局部证明。

它不是 Lean、Coq、Isabelle 那样的形式化证明器。它更像是介于普通数学笔记和形式化证明之间的一层纪律：

- 保留自然语言证明的可读性。
- 强迫每个 definition、lemma、theorem、proof 都有清楚的位置。
- 强迫证明在使用已有结果时显式引用。
- 用 dependency graph 追踪整个证明项目的结构。
- 用 mechanical checks 捕捉文件结构错误和引用错误。
- 可选地用 AI verifier 检查每个局部证明是否真的成立。

## 它想解决什么问题

普通数学笔记里经常会出现这类表达：

- 由定义可知。
- 由上面的 lemma 得到。
- 显然。
- 类似地可证。
- 根据前面的讨论。

这些写法对熟练读者很自然，但对 AI 和长期维护都很危险。因为它们隐藏了真实依赖：

- 到底用了哪个 definition？
- 到底用了哪个 theorem？
- 这个 lemma 是否已经证明？
- 是否偷用了后面才会证明的结论？
- 是否出现了循环依赖？
- AI 是否把一个不存在的结果当成了已知结果？

`qmd-prover` 的目标不是让数学完全机器化，而是让这些依赖显式化。它要求你或者 AI 写证明时，把“这一步靠什么站住”写出来。

## 基本单位：block

在 `qmd-prover` 里，数学内容写在 `.qmd` 文件中。`.qmd` 是 Quarto markdown 文件，本质上仍然是纯文本。

每个数学对象都写成一个独立 block。一个 block 用三行冒号包起来：

```markdown
::: {#def-even .definition name="Even number"}
An integer $n$ is even if $n = 2k$ for some integer $k$.
:::

::: {#thm-sum-even .theorem name="Sum of two even numbers"}
If $a$ and $b$ are even, then $a+b$ is even.
:::

::: {.proof of="thm-sum-even"}
By @def-even, write $a = 2k$ and $b = 2m$.
Then $a+b = 2(k+m)$, so $a+b$ is even by @def-even.
:::
```

这里有几个关键点：

- `#def-even` 是 definition 的唯一标签。
- `.definition` 表示这个 block 是一个 definition。
- `#thm-sum-even` 是 theorem 的唯一标签。
- `.theorem` 表示这个 block 是一个 theorem。
- `.proof of="thm-sum-even"` 表示这个 proof 证明的是 `thm-sum-even`。
- `@def-even` 表示证明中的这一步使用了 `def-even` 这个 definition。

所以证明不再是一整段散文，而是有结构的数学文本。

## 最重要的思想：显式依赖

`qmd-prover` 最重要的不是 `.qmd` 文件本身，而是显式依赖的思想。

例如一个代数拓扑证明可能有这样的依赖：

```text
definition of simplex
        ↓
definition of chain group
        ↓
definition of boundary operator
        ↓
boundary of boundary is zero
        ↓
cycles and boundaries
        ↓
homology group
```

在普通笔记中，这些关系可能只存在于人的脑子里。但在 `qmd-prover` 中，你需要把它们写成 `@id` 引用。

例如：

```markdown
::: {#def-chain-group .definition name="Chain group"}
...
:::

::: {#def-boundary-operator .definition name="Boundary operator"}
...
:::

::: {#lem-boundary-square-zero .lemma name="Boundary squared is zero"}
For every $n$, $\partial_n \circ \partial_{n+1} = 0$.
:::

::: {.proof of="lem-boundary-square-zero"}
Using @def-boundary-operator, the terms in the double boundary cancel in pairs.
:::
```

这里 `@def-boundary-operator` 就告诉读者和检查工具：这个 proof 依赖边缘算子的定义。

## Dependency graph

当所有证明都用 `@id` 显式引用以后，整个项目就会形成一张 dependency graph。

这张图可以回答：

- 当前目标 theorem 依赖哪些 lemma？
- 哪些 lemma 已经证明？
- 哪些 lemma 还没有证明？
- 是否有 theorem 建立在未证明的结果上？
- 是否有循环依赖？
- 哪个地方是整个证明真正卡住的位置？

这对大型证明尤其重要。因为一个长证明的困难通常不只是“某一步不会算”，而是：

- 不知道整个证明依赖了多少前置结果。
- 不知道哪个结果是关键瓶颈。
- 不知道某个 lemma 是否已经被其他地方使用。
- 修改一个定义以后，不知道哪些 theorem 会受影响。

`qmd-prover` 的思想是：把这些隐藏结构变成可检查的图。

## 两层检查

`qmd-prover` 的检查大致分为两层。

### 第一层：mechanical checks

这一层不需要 AI。程序只检查结构问题，例如：

- 每个 label 是否唯一。
- 每个 proof 是否指向一个存在的 theorem 或 lemma。
- 每个 `@id` 是否真的存在。
- 引用关系是否形成循环。
- 文件结构是否符合规范。

这一层不会判断数学是否正确。它只判断“线路有没有接错”。

这就像代码中的静态检查：它不保证程序一定符合你的数学意图，但可以先抓出很多低级错误。

### 第二层：AI verifier

第二层是可选的 AI verifier。它让一个独立 AI 检查单个 proof。

重点是：它不是让 AI 泛泛地回答“这个证明对不对”，而是给它一个局部任务：

```text
给定这个 theorem、这个 proof，以及 proof 明确引用的那些 definition/lemma/theorem，
判断这些引用是否真的足够推出结论。
```

这样做的好处是：

- verifier 和写证明的 AI 可以分开，避免自己给自己打分。
- verifier 只看局部上下文，减少被整篇长文本干扰。
- 每个 proof 都有明确输入：结论、证明、引用的前置结果。
- 如果失败，可以定位到具体 block，而不是笼统地说“整篇证明有问题”。

但要注意：AI verifier 不是形式化证明证书。它提高可信度，但不等于 Lean 那样的严格验证。

## 证明状态

`qmd-prover` 的另一个有用想法是给每个结果一个状态。

常见状态可以理解为：

- `open`：已经陈述，但还没有证明。
- `unverified`：有证明，但还没有通过 verifier。
- `rejected`：检查发现证明有问题。
- `blocked`：这个结果依赖的某个前置结果还没有完成。
- `broken`：文件结构或引用本身有问题。
- `verified`：这个结果及其依赖都通过检查。
- `disproved`：已经被反例否定。

这让数学项目像软件项目一样有了进度管理。你可以问：

```text
现在什么已经完成？
什么还 open？
什么 blocked？
整个目标 theorem 卡在哪个 lemma？
```

## 和普通 Obsidian 笔记的区别

普通 Obsidian 笔记强调连接：

```text
[[链群]] 连接到 [[边缘算子]]
[[边缘算子]] 连接到 [[闭链]]
[[闭链]] 连接到 [[同调群]]
```

这种连接适合学习和回忆，但它不一定表达严格依赖。

`qmd-prover` 更强调证明依赖：

```text
theorem A 的 proof 具体用了 definition B 和 lemma C
lemma C 的 proof 又具体用了 theorem D
```

所以可以这样区分：

- Obsidian link：帮助我理解概念之间有关联。
- `qmd-prover` dependency：说明一个证明在逻辑上依赖什么。

两者可以互补。Obsidian 适合学习地图，`qmd-prover` 适合证明工程。

## 和形式化证明的区别

Lean / Coq / Isabelle 的目标是严格形式化。每一步都必须符合机器可检查的逻辑规则。

`qmd-prover` 的目标更温和：

- 仍然使用自然语言。
- 不要求把所有数学对象形式化成类型系统中的对象。
- 不保证绝对正确。
- 但要求证明结构清晰、依赖明确、局部可检查。

所以它适合以下场景：

- 正在学习一个理论，希望整理概念依赖。
- 正在写论文草稿，希望管理 lemma 和 theorem。
- 正在和 AI 协作补证明细节。
- 不想直接进入 Lean 的高门槛，但又不满足于普通散文式证明。

## 对代数拓扑学习的意义

代数拓扑非常适合这种思路，因为它本来就有清楚的层级结构：

```text
topological space
open set
continuous map
simplicial complex
simplex
chain group
boundary operator
cycle
boundary
homology group
induced homomorphism
exact sequence
```

例如同调群的定义：

```text
H_n = Z_n / B_n
```

看起来很短，但背后依赖很多东西：

- $C_n$ 是 chain group。
- $\partial_n$ 是 boundary operator。
- $Z_n = \ker \partial_n$ 是 cycle group。
- $B_n = \operatorname{im} \partial_{n+1}$ 是 boundary group。
- 要让 $B_n \subseteq Z_n$，需要 $\partial_n \circ \partial_{n+1} = 0$。
- 要让 quotient $Z_n / B_n$ 合法，需要知道 $B_n$ 是 $Z_n$ 的 subgroup。

如果用 `qmd-prover` 思路整理，就不会只写“同调群定义为 $Z_n / B_n$”，而会把每个依赖都展开。

## 一个适合我的工作流

对当前学习来说，可以不用立刻安装 `qmd-prover`，但可以先吸收它的写法。

推荐工作流：

1. 普通学习笔记继续放在 Obsidian。
2. 每个重要 definition、lemma、theorem 都给一个稳定名字。
3. 写证明时尽量明确引用“用了哪个定义或定理”。
4. 对复杂主题画出依赖图。
5. 等某个主题积累到足够复杂时，再考虑把它迁移成 `.qmd` 项目。

例如整理同调群时，可以先在 Obsidian 中写：

```markdown
# Homology Group

## Depends on

- [[链群]]
- [[边缘算子]]
- [[闭链]]
- [[边缘链]]
- [[边缘算子#boundary squared is zero]]

## Definition

The $n$-th homology group is
$$
H_n = Z_n / B_n = \ker \partial_n / \operatorname{im} \partial_{n+1}.
$$

## Why this quotient makes sense

Because $\partial_n \circ \partial_{n+1} = 0$, every boundary is a cycle.
Hence $B_n \subseteq Z_n$.
```

这已经是在用 `qmd-prover` 的精神了：先把依赖说清楚，再写定义和证明。

## 我自己的理解

`qmd-prover` 最有价值的地方不是某个具体命令，而是一种数学写作习惯：

> 不让证明漂浮在自然语言里，而是让每个结论都有标签，每个引用都有出处，每个目标都有状态，每条路线都有依赖图。

这对 AI 协作尤其重要。因为 AI 最容易犯的错误之一，就是写出表面流畅、局部合理、但依赖不清的证明。`qmd-prover` 的思路正好约束这种问题。

## Related

- [[AI与数学科研]]
- [[同调群]]
- [[链群]]
- [[边缘算子]]
- [[闭链]]
- [[边缘链]]
