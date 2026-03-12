# FD-RPB-SDR

### Fermi–Dirac Rational Pole-Based Expansion with GMRES-SDR

**FD-RPB-SDR** evaluates the **finite-temperature Fermi–Dirac operator**

$$
f_\beta(H) = \frac{1}{1 + \exp!\left(\beta(H-\mu)\right)}
$$

applied to vectors

$$
y = f_\beta(H)x
$$

for **large sparse Hamiltonians** without performing a full diagonalization.

The method combines

* **rational pole expansions of the Fermi–Dirac function**
* **sparse shifted linear solves**
* **GMRES-SDR** (sketched deflated restarted GMRES)

making it suitable for **large electronic-structure Hamiltonians** and **tight-binding material simulations**.



# Core References

### Pole expansion of the Fermi–Dirac function

**Lin, Lu, Ying, E**

*Pole-Based Approximation of the Fermi-Dirac Function*

Chinese Annals of Mathematics B **30**, 729–742 (2009)

[![DOI](https://img.shields.io/badge/DOI-10.1007/s11401--009--0145--9-blue)](https://doi.org/10.1007/s11401-009-0145-9)



### GMRES-SDR solver

**Burke, Güttel, Soodhalter**

*GMRES with Randomized Sketching and Deflated Restarting*

[![arXiv](https://img.shields.io/badge/arXiv-2311.14206-blue)](https://arxiv.org/abs/2311.14206)



# Scientific Background

## Finite-Temperature Density Operator

For a Hamiltonian

$$
H \in \mathbb{C}^{n \times n}
$$

the **finite-temperature density operator** is

$$
\rho = \frac{1}{1 + e^{\beta(H-\mu)}} .
$$

where

* $\beta = (k_B T)^{-1}$
* $\mu$ is the chemical potential.

Applying the density operator to a vector gives

$$
y = f_\beta(H)x
$$

which is a fundamental operation in many **electronic-structure algorithms**.

Direct diagonalization scales as

$$
\mathcal{O}(n^3)
$$

and becomes prohibitive for large sparse systems.



# Mathematical Reformulation

Define the shifted Hamiltonian

$$
B = H - \mu I .
$$

The Fermi operator can be rewritten as

$$
f_\beta(B)
= \frac12\left(1-\tanh\left(\frac{\beta B}{2}\right)\right).
$$

Using a **rational pole expansion**

$$
f_\beta(B) \approx
\sum_{j=1}^{Q} w_j (\xi_j I - B)^{-1},
$$

where

* $\xi_j$ are complex poles
* $w_j$ are quadrature weights.

The computation reduces to solving **shifted linear systems**

$$
(\xi_j I - B) y_j = x
$$

and accumulating

$$
f_\beta(H)x \approx
\sum_{j=1}^{Q} w_j y_j .
$$



# Numerical Method

The algorithm has two main components.



# 1. Pole Expansion of the Fermi Function

The poles are generated using the **conformal contour construction** of

**Lin–Lu–Ying–E (2009)**.

The method maps a **dumbbell-shaped contour** using Jacobi elliptic functions and trapezoidal quadrature.

This produces poles and weights satisfying

$$
f_\beta(B) \approx
\sum_{j=1}^{Q} w_j (\xi_j I - B)^{-1}
$$

with **exponential convergence in $Q$**.

The implementation computes

* conformal quadrature nodes
* quadrature weights
* stable complex evaluation of the Fermi function.



# 2. Shifted Linear Solves

Each pole requires solving

$$
(\xi_j I - B)y_j = x .
$$

The systems are solved using **GMRES-SDR**, a randomized Krylov solver designed for large sparse matrices.



# GMRES-SDR Solver

The implementation follows

**Burke, Güttel, Soodhalter (2023)**
*GMRES with Randomized Sketching and Deflated Restarting*

The solver combines

* restarted GMRES
* randomized sketching
* truncated Arnoldi orthogonalization
* deflated restarting
* harmonic Ritz vector recycling.

This significantly reduces the cost of orthogonalization and improves convergence for sequences of shifted systems.



## Sketching Operator

GMRES-SDR uses a **Subsampled Randomized Cosine Transform (SRCT)** sketch

$$
S = \sqrt{\frac{n}{s}}, RFD
$$

where

* $D$ — random sign diagonal matrix
* $F$ — discrete cosine transform
* $R$ — row subsampling operator.

The sketch dimension satisfies

$$
s \ll n .
$$

This allows Krylov orthogonalization to be performed in the **sketched space** rather than the full dimension.



## Recycling and Deflation

After each restart cycle, GMRES-SDR extracts approximate invariant subspaces using **harmonic Ritz vectors**

$$
U \approx \text{invariant subspace of } A .
$$

These vectors are recycled across restart cycles and across shifted systems, improving convergence when solving

$$
(\xi_j I - B)y = x
$$

for many poles.



# Algorithm Overview

To compute

$$
y = f_\beta(H)x
$$

the algorithm performs

1. Estimate spectral bounds

$$
E_{\min}, E_{\max}
$$

2. Generate poles and weights

$$
{\xi_j, w_j}
$$

3. For each pole solve

$$
(\xi_j I - (H-\mu))y_j = x
$$

4. Accumulate

$$
y = \sum_j w_j y_j .
$$



# GPU Reference Benchmark

For benchmarking purposes the project can compute a **dense reference solution**

$$
f_\beta(H)x = V f_\beta(\Lambda) V^\dagger x
$$

using a full eigendecomposition

$$
H = V \Lambda V^\dagger .
$$

When available, this reference computation can be executed on the **GPU using CuPy** to accelerate

* dense Hermitian eigendecomposition
* spectral filtering.

This reference is used to measure the accuracy of the pole expansion and Krylov solves.



# Project Structure

```
FD-RPB-SDR
│
├── backend
│   ├── GMRES_SDR.py        # GMRES-SDR solver
│   ├── fermi_dirac.py      # pole expansion and conformal mapping
│   ├── engine.py           # main conformal application routines
│   └── benchmark_tools.py  # GMRES / SDR comparison helpers
│
├── graphene_builder.py     # sparse graphene Hamiltonian generator
├── benchmark.py            # benchmarking and parameter sweeps
├── test_gmres.py            # GMRES vs GMRES-SDR validation
│
└── README.md
```



# Dependencies

Python ≥ 3.12

Required

```
numpy
scipy
matplotlib
pandas
```

Optional (GPU reference)

```
cupy
```

Install with

```
pip install numpy scipy matplotlib pandas
```

GPU support

```
pip install cupy-cuda12x
```



# Benchmark Experiments

The repository includes benchmarking tools that compare

* **GMRES**
* **GMRES-SDR**

across

* system sizes
* temperatures
* solver parameters.

Typical sweeps include

* lattice sizes

$$
m=n \in {5,29,41,58}
$$

* temperatures

$$
T \in {100,;10,;1,;0.1},\mathrm{K}
$$

with

$$
\beta = \frac{1}{k_B T}.
$$

Metrics recorded

* runtime
* number of matrix-vector products
* pole solve failures
* relative error against dense reference.



# Authors

Louis Rossignol, Zhanghao Zhouyin and Hong Guo

McGill University



# Citation

```bibtex
@software{FD-RPB-SDR,
  author = {Louis Rossignol, Zhanghao Zhouyin and Hong Guo},
  title = {FD-RPB-SDR: Sparse Fermi-Dirac Operator Expansion with GMRES-SDR},
  year = {2026},
  url = {https://github.com/louis-rsgl/FD-RPB-SDR}
}
```
