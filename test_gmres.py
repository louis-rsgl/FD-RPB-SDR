#%%
import numpy as np
import scipy.sparse as sp

from backend.benchmark_tools import gmres_solve_true_residual, gmres_sdr_true_residual
from backend.engine import apply_reference_from_eigh, compare_gmres_vs_sdr_single_Q
from graphene_builder import (
    build_supercell,
    build_first_nn_bonds,
    k_from_grid_index,
    Hk_from_bonds_sparse,
)


def build_small_graphene_system(m=29, n=29, t=-2.7):
    """
    Build the sparse k-space graphene Hamiltonian used in the tests.
    """
    cell = build_supercell(m, n, a=2.46)
    bonds = build_first_nn_bonds(cell, t=t)
    kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)

    Hk = Hk_from_bonds_sparse(
        cell,
        bonds,
        kvec,
        onsite=0.0,
        make_hermitian=True,
    ).tocsr()

    return Hk


def test_direct_shifted_system():
    """
    Direct comparison on one shifted linear system:
        (z I - H) x = b
    """
    Hk_small = build_small_graphene_system(m=29, n=29, t=-2.7)

    Nat = Hk_small.shape[0]
    print("Nat =", Nat)
    print("nnz =", Hk_small.nnz, "  nnz/N =", Hk_small.nnz / Nat)

    I = sp.eye(Nat, format="csr", dtype=Hk_small.dtype)

    eta = 1e-3
    mu = 0.0
    z = mu + 1j * eta

    A = (z * I - Hk_small).tocsr()

    rng = np.random.default_rng(1234)
    b = rng.standard_normal(Nat) + 1j * rng.standard_normal(Nat)
    b /= np.linalg.norm(b)

    tol_abs = 1e-10
    max_it = 30
    max_restarts = 20

    sdr_param = {
        "verbose": 1,
        "ssa": 0,
        "t": 2,
        "k": 10,
        "d": 1,
        "pert": 0,
        "harmonic": 1,
        "ls_solve": "lstsq",
        "svd_tol": 1e-15,
    }

    x_gmres, info_gmres, mv_gmres, res_gmres, time_gmres = gmres_solve_true_residual(
        A=A,
        b=b,
        tol_abs=tol_abs,
        restart=max_it,
        maxiter_cycles=max_restarts,
        x0=None,
    )

    x_sdr, out_sdr, res_sdr, time_sdr = gmres_sdr_true_residual(
        A=A,
        b=b,
        tol_abs=tol_abs,
        max_it=max_it,
        max_restarts=max_restarts,
        x0=None,
        extra_param=sdr_param,
    )

    print("\n=== Direct shifted-system comparison: GMRES vs GMRES-SDR ===")
    print(
        f"SciPy GMRES   : info={info_gmres}, mv={mv_gmres}, "
        f"true_res={res_gmres:.3e}, time={time_gmres:.3f}s"
    )
    print(
        f"GMRES-SDR     : mv={out_sdr['mv']}, ip={out_sdr['ip']}, sv={out_sdr['sv']}, "
        f"true_res={res_sdr:.3e}, time={time_sdr:.3f}s"
    )

    print("\nGMRES-SDR cycle residuals:")
    print("  residuals =", out_sdr["residuals"])
    print("  iters     =", out_sdr["iters"])
    print("  sres      =", out_sdr["sres"])

    return {
        "gmres": dict(x=x_gmres, info=info_gmres, mv=mv_gmres, res=res_gmres, time=time_gmres),
        "sdr": dict(x=x_sdr, out=out_sdr, res=res_sdr, time=time_sdr),
    }


def test_conformal_apply_comparison():
    """
    Compare GMRES and GMRES-SDR inside the conformal Fermi-Dirac apply.
    """
    Hk_small = build_small_graphene_system(m=5, n=5, t=-2.7)

    Nat_small = Hk_small.shape[0]
    print("Nat =", Nat_small)
    print("nnz =", Hk_small.nnz, "  nnz/N =", Hk_small.nnz / Nat_small)

    beta = 50.0
    mu = 0.0

    rng = np.random.default_rng(1234)
    x = rng.standard_normal(Nat_small) + 1j * rng.standard_normal(Nat_small)
    x /= np.linalg.norm(x)

    Hd = Hk_small.toarray().astype(np.complex128)
    evals, V = np.linalg.eigh(Hd)
    Emin = float(np.min(evals))
    Emax = float(np.max(evals))

    print("\nSpectral window:")
    print("Emin =", Emin)
    print("Emax =", Emax)

    y_ref = apply_reference_from_eigh(evals, V, x, beta=beta, mu=mu)

    sdr_fixed = {
        "ssa": 0,
        "t": 2,
        "k": 12,
        "d": 1,
        "pert": 0,
        "harmonic": 1,
        "ls_solve": "lstsq",
        "verbose": 0,
    }

    results = compare_gmres_vs_sdr_single_Q(
        H_csr=Hk_small,
        x=x,
        y_ref=y_ref,
        beta=beta,
        mu=mu,
        Emin=Emin,
        Emax=Emax,
        Q=32,
        tol_abs_pole=1e-10,
        restart=40,
        maxiter_cycles=20,
        sdr_fixed=sdr_fixed,
        recycle_across_poles=True,
    )

    return results


if __name__ == "__main__":
    print("\nRunning direct shifted-system test...")
    test_direct_shifted_system()

    print("\nRunning conformal-apply comparison test...")
    test_conformal_apply_comparison()