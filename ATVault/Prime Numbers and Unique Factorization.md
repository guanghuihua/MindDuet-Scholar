# Prime Numbers and Unique Factorization

## Context

These are the main results about primes and unique factorization from Hungerford, *Abstract Algebra: An Introduction*, pp. 18-22.

## Theorem 1.5

Let $p$ be an integer with $p \ne 0, \pm 1$. Then $p$ is prime if and only if $p$ has this property:

whenever $p \mid bc$, then $p \mid b$ or $p \mid c$.

## Corollary 1.6

If $p$ is prime and

$$
p \mid a_1a_2 \cdots a_n,
$$

then $p$ divides at least one of the $a_i$.

## Theorem 1.7

Every integer $n$ except $0, \pm 1$ is a product of primes.

## Theorem 1.8: The Fundamental Theorem of Arithmetic

Every integer $n$ except $0, \pm 1$ is a product of primes. This prime factorization is unique in the following sense: If

$$
n = p_1p_2 \cdots p_r
$$

and

$$
n = q_1q_2 \cdots q_s
$$

with each $p_i$, $q_j$ prime, then $r = s$; that is, the number of factors is the same, and after reordering and relabeling the $q$'s,

$$
p_1 = \pm q_1,\quad
p_2 = \pm q_2,\quad
\ldots,\quad
p_r = \pm q_r.
$$

See also: [[The Fundamental Theorem of Arithmetic]].

## Corollary 1.9

Every integer $n > 1$ can be written in one and only one way in the form

$$
n = p_1p_2p_3 \cdots p_r,
$$

where the $p_i$ are positive primes such that

$$
p_1 \le p_2 \le p_3 \le \cdots \le p_r.
$$

## Theorem 1.10

Let $n > 1$. If $n$ has no positive prime factor less than or equal to $\sqrt n$, then $n$ is prime.

## How These Results Fit Together

- Theorem 1.5 gives the key divisibility property of prime numbers.
- Corollary 1.6 extends that property from a product of two integers to a finite product.
- Theorem 1.7 proves existence of prime factorizations.
- Theorem 1.8 proves uniqueness of prime factorizations, up to order and signs.
- Corollary 1.9 gives the familiar positive-prime version for $n > 1$.
- Theorem 1.10 gives a practical primality test: to check whether $n$ is prime, it is enough to test prime divisors up to $\sqrt n$.

## Reference

Hungerford, *Abstract Algebra: An Introduction*, Theorems 1.5-1.10, pp. 18-22.
