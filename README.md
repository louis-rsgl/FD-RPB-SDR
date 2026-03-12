**FD-RPB-SDR** implements the application of the **finite-temperature Fermi–Dirac operator**

$$
f_\beta(H) = \frac{1}{1 + \exp(\beta(H-\mu))}
$$

to vectors for **large sparse Hamiltonians**

$$
f_\beta(H)x
$$

without explicit diagonalization.

The method combines

* a **rational pole-based expansion of the Fermi–Dirac function**
* **sparse linear solves**
* **GMRES-SDR (sketched deflated restarted GMRES)**

making it suitable for **large electronic structure problems** and **material Hamiltonians**.


# Core References

The numerical method combines two algorithmic components.

### Pole expansion of the Fermi–Dirac function

**Lin, Lu, Ying, E**

*Pole-Based Approximation of the Fermi-Dirac Function*

Chinese Annals of Mathematics B **30**, 729–742 (2009)

[![DOI](https://img.shields.io/badge/DOI-10.1007/s11401--009--0145--9-blue)](https://doi.org/10.1007/s11401-009-0145-9)



### GMRES-SDR solver

**Burke, Güttel, Soodhalter**

*GMRES with Randomized Sketching and Deflated Restarting*

[![arXiv](https://img.shields.io/badge/arXiv-2311.14206-blue)](https://arxiv.org/abs/2311.14206)



# Physical Problem

We consider large **sparse Hermitian Hamiltonians**

$$
H \in \mathbb{C}^{n \times n}
$$

arising in electronic structure and material simulations.

The goal is to evaluate

$$
f_\beta(H)x
$$

for a vector (x) without diagonalizing (H).

This quantity corresponds to applying the **finite-temperature density operator**

$$
\rho = (1 + e^{\beta(H-\mu)})^{-1}
$$

# Mathematical Formulation

Define the shifted Hamiltonian

$$
B = H - \mu
$$

The Fermi operator can be rewritten using the identity

$$
f_\beta(B) =
\frac{1}{2}\left(1 - \tanh\left(\frac{\beta B}{2}\right)\right)
$$

The pole expansion approximates the function as

$$
f_\beta(B) \approx
\sum_{j=1}^{N_{\text{pole}}}
w_j (\xi_j I - B)^{-1}
$$

where

* ( \xi_j ) are complex poles
* ( w_j ) are weights.

Thus computing (f_\beta(H)x) reduces to solving

$$
(\xi_j I - B) y_j = x
$$

for each pole.



# Numerical Method

The computation consists of two stages.



## 1. Pole Expansion of the Fermi Function

The poles are generated using the **conformal contour construction** of

Lin–Lu–Ying–E.

The method maps a **dumbbell-shaped contour** using Jacobi elliptic functions and trapezoidal quadrature.

The resulting poles satisfy

$$
g(B) \approx
\sum_{j=1}^{Q} w_j (\xi_j I - B)^{-1}
$$

with exponential convergence in (Q).

The code computes

* conformal nodes
* quadrature weights
* stable evaluation of the Fermi function for complex arguments.



## 2. Shifted Linear Solves

Each pole requires solving

$$
(\xi I - B)y = x
$$

This package solves the systems using

**GMRES-SDR**

which combines

* GMRES Krylov iteration
* randomized sketching
* deflated restarting
* harmonic Ritz recycling.



### GMRES-SDR Sketching

The solver uses an **SRCT sketch operator**

$$
S = RFD
$$

where

* (D) is a random sign diagonal matrix
* (F) is a discrete cosine transform
* (R) is row sampling.

The sketch dimension

$$
s \ll n
$$

reduces orthogonalization cost.



### Recycling

Deflated restarting uses approximate invariant subspaces

$$
U
$$

constructed from harmonic Ritz vectors.

This improves convergence for sequences of shifted systems.



# Algorithm Overview

To compute

$$
f_\beta(H)x
$$

the algorithm performs:

1. Estimate spectral bounds

$$
E_{\min},E_{\max}
$$

2. Generate poles and weights

$$
{\xi_j,w_j}
$$

3. For each pole solve

$$
(\xi_j I - (H-\mu))y_j = x
$$

4. Accumulate

$$
f_\beta(H)x =
\sum_j w_j y_j
$$


# Dependencies

Python ≥ 3.13

Required packages

```
numpy
scipy
```


Install

```
pip install numpy scipy
```



# Author

Louis Rossignol, Zhouyin Zhanghao and Hong Guo

McGill University


## BibTeX

```bibtex
@software{FD-RPB-SDR,
  author = {Louis Rossignol, Zhanghao Zhouyin and Hong Guo},
  title = {FD-RPB-SDR: Sparse Fermi-Dirac Operator Expansion with GMRES-SDR},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/louis-rsgl/FD-RPB-SDR}
}
```