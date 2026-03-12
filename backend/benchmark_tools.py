import time
import numpy as np
import scipy.sparse.linalg as spla

from backend.GMRES_SDR import gmres_sdr


class MatvecCounter(spla.LinearOperator):
    """
    LinearOperator wrapper that counts matrix-vector products.
    """

    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.dtype = np.dtype(getattr(A, "dtype", np.complex128))
        self.mv = 0

    def _matvec(self, x):
        self.mv += 1
        return self.A @ x


def gmres_solve_true_residual(A, b, tol_abs, restart, maxiter_cycles, x0=None):
    """
    Solve Ax=b with SciPy GMRES using absolute tolerance only.

    Returns
    -------
    x : ndarray
    info : int
    mv_count : int
    true_res_norm : float
    elapsed : float
    """
    Aop = MatvecCounter(A)
    t0 = time.perf_counter()
    x, info = spla.gmres(
        Aop,
        b,
        x0=x0,
        restart=restart,
        maxiter=maxiter_cycles,
        atol=tol_abs,
        rtol=0.0,
    )
    elapsed = time.perf_counter() - t0
    r = b - (A @ x)
    return x, info, Aop.mv, float(np.linalg.norm(r)), elapsed


def gmres_sdr_true_residual(A, b, tol_abs, max_it, max_restarts, x0=None, extra_param=None):
    """
    Solve Ax=b with GMRES-SDR and report the final true residual.

    Returns
    -------
    x : ndarray
    out : dict
        Raw GMRES-SDR diagnostics.
    true_res_norm : float
    elapsed : float
    """
    param = {} if extra_param is None else dict(extra_param)
    param["tol"] = tol_abs
    param["max_it"] = max_it
    param["max_restarts"] = max_restarts

    if x0 is not None:
        param["x0"] = x0

    t0 = time.perf_counter()
    x, out = gmres_sdr(A, b, param=param)
    elapsed = time.perf_counter() - t0

    r = b - (A @ x)
    return x, out, float(np.linalg.norm(r)), elapsed


def solve_shifted_system(
    A,
    b,
    solver="gmres",
    tol_abs=1e-10,
    restart=80,
    maxiter_cycles=50,
    x0=None,
    sdr_fixed=None,
    sdr_state=None,
):
    """
    Unified shifted-system solve for either SciPy GMRES or GMRES-SDR.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        Shifted system matrix.
    b : ndarray
        Right-hand side.
    solver : {"gmres", "sdr"}
        Linear solver choice.
    tol_abs : float
        Absolute residual tolerance.
    restart : int
        Restart length for GMRES and inner iteration budget for GMRES-SDR.
    maxiter_cycles : int
        Maximum restart cycles.
    x0 : ndarray or None
        Optional initial guess.
    sdr_fixed : dict or None
        Fixed GMRES-SDR parameter dictionary.
    sdr_state : dict or None
        Recycled GMRES-SDR state, e.g. U/SU/SAU/sketch_distortion.

    Returns
    -------
    x : ndarray
    out : dict
        Common fields:
            true_res
            mv
            elapsed
            failure
        GMRES-only:
            info_bad
        SDR-only:
            residuals, sres, iters, U, SU, SAU, sketch_distortion
    """
    if solver == "gmres":
        x, info, mv, true_res, elapsed = gmres_solve_true_residual(
            A=A,
            b=b,
            tol_abs=tol_abs,
            restart=restart,
            maxiter_cycles=maxiter_cycles,
            x0=x0,
        )
        out = dict(
            true_res=true_res,
            mv=int(mv),
            elapsed=float(elapsed),
            failure=int(true_res > tol_abs),
            info_bad=int(info != 0),
        )
        return x, out

    if solver == "sdr":
        extra_param = {}
        if sdr_fixed:
            extra_param.update(sdr_fixed)
        if sdr_state:
            extra_param.update(sdr_state)

        x, raw, true_res, elapsed = gmres_sdr_true_residual(
            A=A,
            b=b,
            tol_abs=tol_abs,
            max_it=restart,
            max_restarts=maxiter_cycles,
            x0=x0,
            extra_param=extra_param,
        )

        out = dict(
            true_res=true_res,
            mv=int(raw["mv"]),
            elapsed=float(elapsed),
            failure=int(true_res > tol_abs),
            residuals=raw.get("residuals", None),
            sres=raw.get("sres", None),
            iters=raw.get("iters", None),
        )
        for key in ("U", "SU", "SAU", "sketch_distortion"):
            if key in raw:
                out[key] = raw[key]

        return x, out

    raise ValueError("solver must be 'gmres' or 'sdr'")