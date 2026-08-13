# Galois Theory：为什么五次方程无法用根式表达？

## Source

- Title: 【为什么五次方程无法用根式表达？】
- Link: <https://www.bilibili.com/video/BV1YaqJYVEr3/?share_source=copy_web&vd_source=1b2bf3aec3e6a122e5a737a94087fefb>
- Date watched: 2026-06-03

## Notes

我目前能访问到的是 Bilibili 页面公开 metadata 和视频简介；没有可用的公开字幕 transcript。因此下面是基于公开简介与相关数学背景整理的学习笔记，不是逐字视频记录。

核心问题：

为什么一般五次方程没有像二次、三次、四次方程那样的 radical formula？

关键回答来自 Galois theory：

> A polynomial is solvable by radicals only when its Galois group is a solvable group.

对于一般五次方程，根之间的 permutation symmetry 通常给出 $S_5$。而 $S_5$ is not solvable，因此一般五次方程 cannot be solved by radicals。

## Conceptual Path

1. 从 polynomial roots 出发。
   一个 polynomial 的 roots 可以被看成一组对象；不同 roots 之间存在 permutation symmetry。

2. 用 field extension 捕捉“加入 roots 之后”数域发生了什么变化。
   从 base field，例如 $\mathbb Q$，扩张到包含所有 roots 的 splitting field。

3. 用 Galois group 描述这些 roots 的 symmetry。
   Galois group 由保持 base field 不变的 automorphisms 构成。这些 automorphisms 会 permute roots。

4. radical expression 对应一种可逐层拆解的代数结构。
   如果 roots 可以通过 radicals 表达，那么对应的 Galois group 必须能通过一系列较简单的 quotient groups 分解；这就是 solvable group 的条件。

5. 一般五次方程的 obstruction 是 $S_5$。
   $S_5$ 的结构太复杂，not solvable，因此一般五次方程没有 radical formula。

## Key Takeaway

五次方程“无法用根式表达”不是因为人们还没找到足够聪明的公式，而是因为一般五次方程背后的 symmetry group 本身不允许这种公式存在。

Galois theory 把“求根公式是否存在”的问题转化成了 group theory 问题：

$$
\text{solvable by radicals}
\quad \Longleftrightarrow \quad
\text{Galois group is solvable}.
$$

## Questions

- Galois group 如何刻画方程根之间的对称性？
- “solvable by radicals” 和 “solvable group” 之间的精确关系是什么？
- 为什么一般五次方程对应的对称性会阻止根式表达？

## Follow-up Notes to Create

- [[field extension]]
- [[splitting field]]
- [[Galois group]]
- [[solvable group]]
- [[symmetric group]]
- [[Abel-Ruffini theorem]]

## Related

- [[Galois group]]
- [[solvable by radicals]]
- [[solvable group]]

## Reference

- Bilibili: 【为什么五次方程无法用根式表达？】 <https://www.bilibili.com/video/BV1YaqJYVEr3/>
- Original video listed in Bilibili description: <https://www.youtube.com/watch?v=zCU9tZ2VkWc>
