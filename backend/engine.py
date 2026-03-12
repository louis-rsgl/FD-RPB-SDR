import time
import numpy as np
import scipy.sparse as sp

from backend.fermi_dirac import conformal_nodes_weights
from backend.benchmark_tools import solve_shifted_system


def apply_reference_from_eigh(evals, V, x, beta, mu):
    """
    Dense reference action of the Fermi-Dirac matrix function.
    """
    s = beta * (evals - mu)
    s = np.clip(s, -700.0, 700.0)
    f = 1.0 / (1.0 + np.exp(s))
    return V @ (f * (V.conj().T @ x))


def conformal_apply_sparse(
    H_csr,
    x,
    beta,
    mu,
    Emin,
    Emax,
    Q,
    solver="gmres",
    tol_abs=1e-10,
    restart=80,
    maxiter_cycles=50,
    sdr_fixed=None,
    recycle_across_poles=False,
):
    """
    Apply the conformal quadrature approximation using either GMRES or GMRES-SDR
    for the shifted linear systems.
    """
    Hc = H_csr.tocsc().astype(np.complex128, copy=False)
    n = Hc.shape[0]
    x = x.astype(np.complex128, copy=False)

    xis, ws = conformal_nodes_weights(Emin, Emax, Q, beta, mu)
    I = sp.eye(n, format="csc", dtype=np.complex128)

    gx = np.zeros(n, dtype=np.complex128)

    failures = 0
    max_true_res = 0.0
    total_mv = 0
    info_bad = 0

    pole_residuals = []
    pole_mvs = []

    sdr_state_shared = None
    if solver == "sdr" and recycle_across_poles:
        sdr_state_shared = dict(U=None, SU=None, SAU=None)

    for xi, w in zip(xis, ws):
        A = (xi + mu) * I - Hc

        if solver == "sdr":
            sdr_state = sdr_state_shared if recycle_across_poles else dict(U=None, SU=None, SAU=None)
        else:
            sdr_state = None

        v, out = solve_shifted_system(
            A=A,
            b=x,
            solver=solver,
            tol_abs=tol_abs,
            restart=restart,
            maxiter_cycles=maxiter_cycles,
            x0=None,
            sdr_fixed=sdr_fixed,
            sdr_state=sdr_state,
        )

        failures += out["failure"]
        max_true_res = max(max_true_res, out["true_res"])
        total_mv += out["mv"]
        info_bad += out.get("info_bad", 0)

        pole_residuals.append(out["true_res"])
        pole_mvs.append(out["mv"])

        if solver == "sdr" and recycle_across_poles:
            sdr_state_shared["U"] = out.get("U", None)
            sdr_state_shared["SU"] = out.get("SU", None)
            sdr_state_shared["SAU"] = out.get("SAU", None)
            if "sketch_distortion" in out:
                sdr_state_shared["sketch_distortion"] = out["sketch_distortion"]

        gx += w * v

    y = 0.5 * (x - gx)
    stats = dict(
        poles=len(xis),
        failures=failures,
        info_bad=info_bad,
        max_true_res=max_true_res,
        total_mv=total_mv,
        pole_residuals=np.array(pole_residuals, dtype=float),
        pole_mvs=np.array(pole_mvs, dtype=int),
    )
    return y, stats


def compare_gmres_vs_sdr_single_Q(
    H_csr,
    x,
    y_ref,
    beta,
    mu,
    Emin,
    Emax,
    Q,
    tol_abs_pole=1e-10,
    restart=80,
    maxiter_cycles=50,
    sdr_fixed=None,
    recycle_across_poles=False,
):
    """
    Compare GMRES and GMRES-SDR for one fixed conformal quadrature size Q.
    """
    ref_norm = np.linalg.norm(y_ref) + 1e-30

    print(f"\n=== Fixed-Q comparison at Q = {Q} ===")

    t0 = time.perf_counter()
    y_gmres, st_gmres = conformal_apply_sparse(
        H_csr=H_csr,
        x=x,
        beta=beta,
        mu=mu,
        Emin=Emin,
        Emax=Emax,
        Q=Q,
        solver="gmres",
        tol_abs=tol_abs_pole,
        restart=restart,
        maxiter_cycles=maxiter_cycles,
        sdr_fixed=None,
        recycle_across_poles=False,
    )
    dt_gmres = time.perf_counter() - t0
    err_gmres = np.linalg.norm(y_gmres - y_ref) / ref_norm

    t0 = time.perf_counter()
    y_sdr, st_sdr = conformal_apply_sparse(
        H_csr=H_csr,
        x=x,
        beta=beta,
        mu=mu,
        Emin=Emin,
        Emax=Emax,
        Q=Q,
        solver="sdr",
        tol_abs=tol_abs_pole,
        restart=restart,
        maxiter_cycles=maxiter_cycles,
        sdr_fixed=sdr_fixed,
        recycle_across_poles=recycle_across_poles,
    )
    dt_sdr = time.perf_counter() - t0
    err_sdr = np.linalg.norm(y_sdr - y_ref) / ref_norm

    print(
        f"GMRES     | err={err_gmres:.3e} | time={dt_gmres:.3f}s | "
        f"mv={st_gmres['total_mv']:7d} | fails={st_gmres['failures']:4d} | "
        f"info_bad={st_gmres['info_bad']:4d} | max||r||={st_gmres['max_true_res']:.2e}"
    )
    print(
        f"GMRES-SDR | err={err_sdr:.3e} | time={dt_sdr:.3f}s | "
        f"mv={st_sdr['total_mv']:7d} | fails={st_sdr['failures']:4d} | "
        f"max||r||={st_sdr['max_true_res']:.2e}"
    )

    return {
        "gmres": dict(y=y_gmres, rel_err=err_gmres, time=dt_gmres, stats=st_gmres),
        "sdr": dict(y=y_sdr, rel_err=err_sdr, time=dt_sdr, stats=st_sdr),
    }