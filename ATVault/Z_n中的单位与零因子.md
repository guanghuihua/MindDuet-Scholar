# Z_n 中的单位与零因子

## 记号提醒

这里的

$$
\mathbb Z_n
$$

表示模 $n$ 整数：

$$
\mathbb Z_n = \{[0], [1], \ldots, [n-1]\}.
$$

注意：这和同调里的 $Z_n = \ker \partial_n$ 不是同一个记号。为了避免混淆，这里尽量写成 $\mathbb Z_n$。


## Theorem 2.8

设 $p > 1$ 是整数。下面三个条件等价：

1. $p$ 是素数。
2. 对任意 $\mathbb Z_p$ 中的非零元素 $a$，方程

   $$
   ax = 1
   $$

   在 $\mathbb Z_p$ 中有解。
3. 在 $\mathbb Z_p$ 中，只要

   $$
   bc = 0,
   $$

   就有

   $$
   b = 0
   \quad \text{or} \quad
   c = 0.
   $$
·
### 直觉

这个定理说明：

$$
p \text{ 是素数}
\quad \Longleftrightarrow \quad
\mathbb Z_p \text{ 中每个非零元素都有乘法逆元}
\quad \Longleftrightarrow \quad
\mathbb Z_p \text{ 中没有零因子}.
$$

也就是说，当模数是素数时，$\mathbb Z_p$ 的乘法性质特别好，很像普通数域里的运算。

例如在 $\mathbb Z_5$ 中：

$$
2 \cdot 3 = 6 \equiv 1 \pmod 5,
$$

所以

$$
2^{-1} = 3.
$$

## 单位和逆元

在 $\mathbb Z_n$ 中，如果方程

$$
ax = 1
$$

有解，那么元素 $a$ 称为一个单位。

换句话说，如果存在某个 $b \in \mathbb Z_n$，使得

$$
ab = 1,
$$

那么 $a$ 是单位。此时称 $b$ 是 $a$ 的逆元。

注意，$ab = 1$ 同时也说明 $b$ 是单位，并且 $a$ 是 $b$ 的逆元。

## Theorem 2.10

设 $a,n$ 是整数，且 $n > 1$。那么：

$$
[a] \text{ 是 } \mathbb Z_n \text{ 中的单位}
\quad \Longleftrightarrow \quad
\gcd(a,n) = 1.
$$

也就是说，$[a]$ 是单位，意思是存在某个 $[x] \in \mathbb Z_n$，使得

$$
[a][x] = [1].
$$

等价地说：

$$
ax \equiv 1 \pmod n.
$$

## 零因子

$\mathbb Z_n$ 中的非零元素 $a$ 称为零因子，如果方程

$$
ax = 0
$$

有非零解。也就是说，存在某个非零元素 $c \in \mathbb Z_n$，使得

$$
ac = 0.
$$

例如在 $\mathbb Z_6$ 中：

$$
2 \cdot 3 = 6 \equiv 0 \pmod 6.
$$

但是

$$
[2] \ne [0],
\qquad
[3] \ne [0].
$$

所以 $[2]$ 和 $[3]$ 都是零因子。

## 单位和零因子的关系

在 $\mathbb Z_n$ 中，一个非零元素恰好分成两种情况：

1. 如果 $\gcd(a,n)=1$，那么 $[a]$ 是单位。
2. 如果 $\gcd(a,n)>1$，那么 $[a]$ 是零因子。

所以：

$$
\mathbb Z_p \text{ 特别好，是因为 } p \text{ 是素数。}
$$

当 $p$ 是素数时，每个非零整数 $a$ 都满足：

$$
\gcd(a,p)=1.
$$

因此 $\mathbb Z_p$ 中每个非零元素都是单位，也就不会出现非零元素相乘等于 $0$ 的情况。

## 例子对比

### $\mathbb Z_5$

$5$ 是素数，所以 $\mathbb Z_5$ 中的非零元素

$$
[1], [2], [3], [4]
$$

全都是单位。

例如：

$$
[2][3] = [1],
\qquad
[4][4] = [1].
$$

### $\mathbb Z_6$

$6$ 不是素数，所以 $\mathbb Z_6$ 中会出现零因子：

$$
[2][3] = [0].
$$

$\mathbb Z_6$ 中的单位是：

$$
[1], [5].
$$

因为：

$$
\gcd(1,6)=1,
\qquad
\gcd(5,6)=1.
$$

而 $[2]$、$[3]$、$[4]$ 都是零因子。

## Related

- [[同余类]]
- [[Prime Numbers and Unique Factorization]]

## Reference

Hungerford, *Abstract Algebra: An Introduction*, pp. 39-41.
