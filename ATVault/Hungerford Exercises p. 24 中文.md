# Hungerford Exercises, p. 24 中文

## 来源

Hungerford, *Abstract Algebra: An Introduction*, p. 24.

英文原版笔记见：[[Hungerford Exercises p. 24]]。

## OCR 修正说明

你贴出的文本里有几处明显 OCR 错误。这里采用数学上成立的版本：

- Exercise 33: $2^p - 1$。
- Exercise 35(b): $r^n = a$。
- Exercise 36: $24 \mid (p^2 - q^2)$。如果写成 $24 \mid (p^2 - q)$，命题是假的；例如 $p = 5$，$q = 7$ 时，$25 - 7 = 18$，不能被 $24$ 整除。

## Exercise 32. Euclid

证明：素数有无穷多个。

提示：使用反证法。假设只有有限多个素数 $p_1, p_2, \ldots, p_k$，然后考虑数

$$
p_1p_2 \cdots p_k + 1
$$

并证明它不能被 $p_1, p_2, \ldots, p_k$ 中任何一个整除，从而得到矛盾。

### 解答

反设素数只有有限多个：

$$
p_1, p_2, \ldots, p_k.
$$

考虑整数

$$
N = p_1p_2 \cdots p_k + 1.
$$

对每个 $i$，用 $p_i$ 去除 $N$，余数都是 $1$，因为

$$
N = p_1p_2 \cdots p_k + 1.
$$

所以没有任何一个 $p_i$ 能整除 $N$。

另一方面，$N > 1$，由 [[Prime Numbers and Unique Factorization]] 可知，$N$ 至少有一个素因子。但这个素因子不可能是 $p_1, p_2, \ldots, p_k$ 中的任何一个，这与“这些就是全部素数”的假设矛盾。

因此，素数有无穷多个。

## Exercise 33

设 $p > 1$。如果 $2^p - 1$ 是素数，证明 $p$ 是素数。

提示：证明其逆否命题：如果 $p$ 是合数，那么 $2^p - 1$ 也是合数。

注：逆命题是假的，见 Exercise 2(b)。

### 解答

我们证明逆否命题。

假设 $p$ 是合数。那么存在整数 $a,b > 1$，使得

$$
p = ab.
$$

于是

$$
2^p - 1 = 2^{ab} - 1 = (2^a)^b - 1.
$$

利用因式分解公式

$$
x^b - 1 = (x - 1)(x^{b-1} + x^{b-2} + \cdots + x + 1),
$$

令 $x = 2^a$，得到

$$
2^p - 1
= (2^a - 1)(2^{a(b-1)} + 2^{a(b-2)} + \cdots + 2^a + 1).
$$

因为 $a > 1$ 且 $b > 1$，所以这两个因子都大于 $1$。因此 $2^p - 1$ 是合数。

所以，如果 $2^p - 1$ 是素数，那么 $p$ 必须是素数。

## Exercise 34

证明或否定：如果 $n$ 是整数且 $n > 2$，那么存在一个素数 $p$，使得

$$
n < p < n!.
$$

### 解答

这个命题是真的。

令

$$
N = n! - 1.
$$

因为 $n > 2$，所以 $N > 1$。因此 $N$ 有一个素因子，记作 $p$。

如果 $p \le n$，那么 $p$ 整除 $n!$。但 $p$ 也整除 $N = n! - 1$。于是 $p$ 整除

$$
n! - (n! - 1) = 1,
$$

这不可能。

因此 $p > n$。

另一方面，因为 $p$ 整除 $N$ 且 $N > 1$，所以

$$
p \le N = n! - 1 < n!.
$$

于是

$$
n < p < n!.
$$

## Exercise 35

### Exercise 35(a)

设 $a$ 是正整数。如果 $\sqrt a$ 是有理数，证明 $\sqrt a$ 是整数。

### 解答

设

$$
\sqrt a = \frac{m}{n},
$$

其中 $m,n \in \mathbb Z$，$n > 0$，并且 $\gcd(m,n) = 1$。

两边平方，得到

$$
a = \frac{m^2}{n^2},
$$

所以

$$
m^2 = an^2.
$$

因此 $n^2 \mid m^2$。由于 $\gcd(m,n) = 1$，可知 $n = 1$。

所以

$$
\sqrt a = m,
$$

因此 $\sqrt a$ 是整数。

### Exercise 35(b)

设 $r$ 是有理数，$a$ 是整数，并且

$$
r^n = a.
$$

证明 $r$ 是整数。

第 (a) 部分是 $n = 2$ 时的情形。

### 解答

将 $r$ 写成既约分数：

$$
r = \frac{m}{k},
$$

其中 $m,k \in \mathbb Z$，$k > 0$，并且 $\gcd(m,k) = 1$。

因为 $r^n = a$，所以

$$
\left(\frac{m}{k}\right)^n = a.
$$

因此

$$
m^n = ak^n.
$$

所以 $k^n \mid m^n$。但 $\gcd(m,k) = 1$，所以 $\gcd(m^n,k^n) = 1$。于是 $k^n = 1$。又因为 $k > 0$，所以 $k = 1$。

因此

$$
r = m,
$$

所以 $r$ 是整数。

## Exercise 36

设 $p,q$ 是素数，且 $p \ge 5$，$q \ge 5$。证明

$$
24 \mid (p^2 - q^2).
$$

### 解答

因为 $p$ 和 $q$ 都是大于等于 $5$ 的素数，所以它们都是奇数。因此 $p^2$ 和 $q^2$ 模 $8$ 都同余于 $1$：

$$
p^2 \equiv 1 \pmod 8,
\qquad
q^2 \equiv 1 \pmod 8.
$$

所以

$$
p^2 - q^2 \equiv 0 \pmod 8.
$$

另外，因为 $p$ 和 $q$ 都是大于等于 $5$ 的素数，所以它们都不能被 $3$ 整除。因此它们模 $3$ 同余于 $\pm 1$，从而

$$
p^2 \equiv 1 \pmod 3,
\qquad
q^2 \equiv 1 \pmod 3.
$$

于是

$$
p^2 - q^2 \equiv 0 \pmod 3.
$$

因此 $p^2 - q^2$ 同时被 $8$ 和 $3$ 整除。由于 $\gcd(8,3) = 1$，所以它被 $24$ 整除。

即

$$
24 \mid (p^2 - q^2).
$$

## Related

- [[Hungerford Exercises p. 24]]
- [[Prime Numbers and Unique Factorization]]
- [[The Fundamental Theorem of Arithmetic]]
