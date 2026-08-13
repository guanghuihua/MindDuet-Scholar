# Hungerford Exercises, p. 24

## Source

Hungerford, *Abstract Algebra: An Introduction*, p. 24.

## OCR Corrections

The pasted text appears to have a few OCR errors. I use the mathematically coherent forms below:

- Exercise 33: $2^p - 1$.
- Exercise 35(b): $r^n = a$.
- Exercise 36: $24 \mid (p^2 - q^2)$. The statement $24 \mid (p^2 - q)$ would be false, for example $p = 5$, $q = 7$ gives $25 - 7 = 18$.

## Exercise 32. Euclid

Prove that there are infinitely many primes.

Hint: Use proof by contradiction. Assume there are only finitely many primes $p_1, p_2, \ldots, p_k$ and reach a contradiction by showing that the number

$$
p_1p_2 \cdots p_k + 1
$$

is not divisible by any of $p_1, p_2, \ldots, p_k$.

### Solution

Assume, for contradiction, that there are only finitely many primes:

$$
p_1, p_2, \ldots, p_k.
$$

Consider the integer

$$
N = p_1p_2 \cdots p_k + 1.
$$

For each $i$, dividing $N$ by $p_i$ gives remainder $1$, because

$$
N = p_1p_2 \cdots p_k + 1.
$$

Thus no $p_i$ divides $N$.

Since $N > 1$, by [[Prime Numbers and Unique Factorization]], $N$ has a prime divisor. But this prime divisor cannot be any of $p_1, p_2, \ldots, p_k$, contradicting the assumption that these are all the primes.

Therefore, there are infinitely many primes.

## Exercise 33

Let $p > 1$. If $2^p - 1$ is prime, prove that $p$ is prime.

Hint: Prove the contrapositive: If $p$ is composite, so is $2^p - 1$.

Note: The converse is false by Exercise 2(b).

### Solution

We prove the contrapositive.

Assume $p$ is composite. Then

$$
p = ab
$$

for some integers $a,b > 1$.

Then

$$
2^p - 1 = 2^{ab} - 1 = (2^a)^b - 1.
$$

Using the factorization

$$
x^b - 1 = (x - 1)(x^{b-1} + x^{b-2} + \cdots + x + 1),
$$

with $x = 2^a$, we get

$$
2^p - 1
= (2^a - 1)(2^{a(b-1)} + 2^{a(b-2)} + \cdots + 2^a + 1).
$$

Since $a > 1$ and $b > 1$, both factors are greater than $1$. Hence $2^p - 1$ is composite.

Therefore, if $2^p - 1$ is prime, then $p$ must be prime.

## Exercise 34

Prove or disprove: If $n$ is an integer and $n > 2$, then there exists a prime $p$ such that

$$
n < p < n!.
$$

### Solution

The statement is true.

Let

$$
N = n! - 1.
$$

Since $n > 2$, we have $N > 1$. Therefore $N$ has a prime divisor; call it $p$.

If $p \le n$, then $p$ divides $n!$. But $p$ also divides $N = n! - 1$. Hence $p$ divides

$$
n! - (n! - 1) = 1,
$$

which is impossible.

Thus $p > n$.

Also, since $p$ divides $N$ and $N > 1$, we have

$$
p \le N = n! - 1 < n!.
$$

Therefore,

$$
n < p < n!.
$$

## Exercise 35

### Exercise 35(a)

Let $a$ be a positive integer. If $\sqrt a$ is rational, prove that $\sqrt a$ is an integer.

### Solution

Suppose

$$
\sqrt a = \frac{m}{n},
$$

where $m,n \in \mathbb Z$, $n > 0$, and $\gcd(m,n) = 1$.

Then

$$
a = \frac{m^2}{n^2},
$$

so

$$
m^2 = an^2.
$$

Thus $n^2 \mid m^2$. Since $\gcd(m,n) = 1$, this implies $n = 1$.

Therefore

$$
\sqrt a = m,
$$

so $\sqrt a$ is an integer.

### Exercise 35(b)

Let $r$ be a rational number and $a$ an integer such that

$$
r^n = a.
$$

Prove that $r$ is an integer.

Part (a) is the case when $n = 2$.

### Solution

Write

$$
r = \frac{m}{k},
$$

where $m,k \in \mathbb Z$, $k > 0$, and $\gcd(m,k) = 1$.

Since $r^n = a$, we have

$$
\left(\frac{m}{k}\right)^n = a.
$$

Hence

$$
m^n = ak^n.
$$

So $k^n \mid m^n$. But $\gcd(m,k) = 1$, so $\gcd(m^n,k^n) = 1$. Therefore $k^n = 1$, and since $k > 0$, we get $k = 1$.

Thus

$$
r = m,
$$

so $r$ is an integer.

## Exercise 36

Let $p,q$ be primes with $p \ge 5$, $q \ge 5$. Prove that

$$
24 \mid (p^2 - q^2).
$$

### Solution

Since $p$ and $q$ are primes greater than or equal to $5$, both are odd. Therefore $p^2$ and $q^2$ are both congruent to $1$ modulo $8$:

$$
p^2 \equiv 1 \pmod 8,
\qquad
q^2 \equiv 1 \pmod 8.
$$

Hence

$$
p^2 - q^2 \equiv 0 \pmod 8.
$$

Also, since $p$ and $q$ are primes greater than or equal to $5$, neither is divisible by $3$. Therefore each is congruent to $\pm 1$ modulo $3$, so

$$
p^2 \equiv 1 \pmod 3,
\qquad
q^2 \equiv 1 \pmod 3.
$$

Thus

$$
p^2 - q^2 \equiv 0 \pmod 3.
$$

So $p^2 - q^2$ is divisible by both $8$ and $3$. Since $\gcd(8,3) = 1$, it is divisible by $24$.

Therefore,

$$
24 \mid (p^2 - q^2).
$$

## Related

- [[Prime Numbers and Unique Factorization]]
- [[The Fundamental Theorem of Arithmetic]]
