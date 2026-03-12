import numpy as np
import scipy.linalg as sla
from scipy.fft import dct


# =============================================================================
# Sketch operator: SRCT (subsampled randomized cosine transform)
# =============================================================================
class SRCT:
    """
    Scaled SRCT sketch S in R^{s x n}:
        S x = sqrt(n/s) * R * DCT * D * x
    with random sign diagonal D and row sampling R.
    """

    def __init__(self, n: int, s: int, rng=None, sort_idx: bool = False):
        self.n = int(n)
        self.s = int(s)
        if not (1 <= self.s <= self.n):
            raise ValueError("Require 1 <= s <= n.")
        self.rng = np.random.default_rng() if rng is None else rng
        self.sign = self.rng.choice(np.array([-1.0, 1.0]), size=self.n)
        idx = self.rng.choice(self.n, size=self.s, replace=False)
        self.idx = np.sort(idx) if sort_idx else idx
        self.scale = np.sqrt(self.n / self.s)

    def __call__(self, X):
        X = np.asarray(X)
        flat = (X.ndim == 1)
        if flat:
            X = X.reshape(-1, 1)

        if X.shape[0] != self.n:
            raise ValueError(f"SRCT expected first dim {self.n}, got {X.shape[0]}")

        Y = dct((self.sign[:, None] * X), type=2, norm="ortho", axis=0)
        Y = self.scale * Y[self.idx, :]
        return Y[:, 0] if flat else Y


# =============================================================================
# Helpers
# =============================================================================
def _as_matvec(A):
    """Return matvec callable for A, supporting ndarray/sparse/LinearOperator/callable."""
    if callable(A):
        return A
    if hasattr(A, "matvec"):  # LinearOperator
        return lambda v: A.matvec(v)
    return lambda v: A @ v


def _apply_A_to_block(A_mv, U):
    """Compute AU columnwise robustly for callable matvec."""
    if U.size == 0:
        return U
    cols = [A_mv(U[:, i]) for i in range(U.shape[1])]
    return np.column_stack(cols)


def _solve_ls(SAW, Sr, method="lstsq"):
    """Solve min ||SAW y - Sr||_2."""
    if SAW.size == 0:
        return np.zeros((0,), dtype=Sr.dtype)

    if method in ("lstsq", "\\"):
        y, *_ = np.linalg.lstsq(SAW, Sr, rcond=None)
        return y
    if method == "pinv":
        return np.linalg.pinv(SAW) @ Sr
    if method == "qr":
        Q, R = np.linalg.qr(SAW, mode="reduced")
        return sla.solve_triangular(R, Q.conj().T @ Sr, lower=False)
    raise ValueError(f"Unknown ls_solve method: {method}")


def _ordered_qz(HH, Sig, k, harmonic=True):
    """
    Ordered QZ similar to MATLAB:
      - compute generalized eigs of (HH, Sig)
      - harmonic: keep largest |lambda|; non-harm: keep smallest |lambda|
      - use scipy.linalg.ordqz with a predicate based on a threshold.
    Returns (AA, BB, Z, keep) where keep includes 2x2 block safety.
    """
    n = HH.shape[0]
    if k <= 0 or n == 0:
        return None, None, np.eye(n, dtype=HH.dtype), 0

    # Determine ordering threshold from eigvals
    eigs = sla.eigvals(HH, Sig)
    mags = np.abs(eigs)
    mags_sorted = np.sort(mags)

    k_eff = min(k, n)
    if harmonic:
        thr = mags_sorted[-k_eff]
        def pred(alpha, beta):
            lam = alpha / beta
            return np.abs(lam) >= thr
    else:
        thr = mags_sorted[k_eff - 1]
        def pred(alpha, beta):
            lam = alpha / beta
            return np.abs(lam) <= thr

    output_type = "real" if (np.isrealobj(HH) and np.isrealobj(Sig)) else "complex"
    AA, BB, alpha, beta, Q, Z = sla.ordqz(HH, Sig, sort=pred, output=output_type)

    # Base keep: attempt to keep k vectors; predicate can select >k with ties.
    # We choose keep=k_eff, then apply MATLAB 2x2 block safety on AA/BB.
    keep = k_eff
    if keep > 0 and keep < AA.shape[0]:
        # MATLAB check: (AA(k+1,k) != 0 || BB(k+1,k) != 0) in 1-based indexing
        if (AA[keep, keep - 1] != 0) or (BB[keep, keep - 1] != 0):
            keep = min(keep + 1, AA.shape[0])

    return AA, BB, Z, keep


# =============================================================================
# One GMRES-SDR cycle
# =============================================================================
def srgmres_cycle(A_mv, r0, param):
    """
    One cycle for correction equation A e ≈ r0.
    Returns: e, r, out
    """
    max_it = int(param.get("max_it", 50))
    tol = float(param.get("tol", 1e-6))
    t = int(param.get("t", 2))
    k_target = int(param.get("k", min(10, max_it)))
    d = int(param.get("d", 1))
    ssa = int(param.get("ssa", 0))
    ls_solve = param.get("ls_solve", "lstsq")
    svd_tol = float(param.get("svd_tol", 1e-15))
    harmonic = bool(param.get("harmonic", 1))
    verbose = int(param.get("verbose", 1))
    pert = int(param.get("pert", 0))
    hS = param["hS"]
    sketch_distortion = float(param.get("sketch_distortion", 1.4))

    r0 = np.asarray(r0)
    n = r0.shape[0]
    Sr0 = hS(r0)
    sdim = Sr0.shape[0]

    # Recycling
    U = param.get("U", None)
    SU = param.get("SU", None)
    SAU = param.get("SAU", None)
    if U is None or U.size == 0:
        U = np.empty((n, 0), dtype=r0.dtype)
        SU = np.empty((sdim, 0), dtype=Sr0.dtype)
        SAU = np.empty((sdim, 0), dtype=Sr0.dtype)
    else:
        if SU is None:
            SU = hS(U)
        if SAU is None:
            AU = _apply_A_to_block(A_mv, U)
            SAU = hS(AU)

    # Counters
    mv = 0
    ip = 0
    sv = 0

    # Build sketches of recycling space for this cycle
    if U.shape[1] == 0:
        SW = SU
        SAW = SAU
    else:
        SW = SU
        if pert == 0:
            SAW = SAU
        else:
            AU = _apply_A_to_block(A_mv, U)
            SAW = hS(AU)
            mv += U.shape[1]  # equivalent matvec count
            sv += U.shape[1]

    # Initial residual sketch
    Sr = hS(r0); sv += 1
    if ssa in (1, 2):
        nrm = np.linalg.norm(Sr)
    else:
        nrm = np.linalg.norm(r0); ip += 1

    if nrm == 0:
        e = np.zeros_like(r0)
        out = dict(U=U, SU=SW, SAU=SAW, hS=hS, k=U.shape[1], m=0,
                   mv=mv, ip=ip, sv=sv, sres=np.array([], dtype=float),
                   sketch_distortion=sketch_distortion)
        return e, r0, out

    # Preallocate Krylov basis
    dtypeV = np.result_type(r0.dtype, np.complex128 if np.iscomplexobj(r0) else np.float64)
    V = np.zeros((n, max_it + 1), dtype=dtypeV)
    SV = np.zeros((sdim, max_it + 1), dtype=Sr.dtype)
    SAV = np.zeros((sdim, max_it), dtype=Sr.dtype)
    H = np.zeros((max_it + 1, max_it), dtype=dtypeV)

    V[:, 0] = r0 / nrm
    SV[:, 0] = Sr / nrm

    sres = []
    e = None
    r = None
    m_final = 0

    for j in range(1, max_it + 1):
        j0 = j - 1

        w = A_mv(V[:, j0]); mv += 1

        # -------------------------
        # Arnoldi variants
        # -------------------------
        if ssa == 0:
            i_start = max(j0 - t + 1, 0)
            for i in range(i_start, j0 + 1):
                hij = np.vdot(V[:, i], w)
                H[i, j0] = hij
                ip += 1
                w = w - V[:, i] * hij

            hnext = np.linalg.norm(w)
            H[j0 + 1, j0] = hnext
            ip += 1

            if hnext == 0:
                # Happy breakdown: Krylov subspace is invariant
                # We'll stop expanding basis; handle at checkpoint below.
                V[:, j0 + 1] = 0
                SV[:, j0 + 1] = 0
            else:
                V[:, j0 + 1] = w / hnext
                SV[:, j0 + 1] = hS(V[:, j0 + 1]); sv += 1

            SAV[:, j0] = SV[:, : (j0 + 2)] @ H[: (j0 + 2), j0]

        elif ssa == 1:
            sw = hS(w); sv += 1
            SAV[:, j0] = sw

            if U.shape[1] > 0:
                coeffs = np.linalg.lstsq(SU, sw, rcond=None)[0]
                w = w - U @ coeffs
                sw = sw - SU @ coeffs

            ind = np.arange(max(j0 - t + 1, 0), j0 + 1)
            coeffs = SV[:, ind].conj().T @ sw

            w = w - V[:, ind] @ coeffs
            sw = sw - SV[:, ind] @ coeffs

            nsw = np.linalg.norm(sw)
            if nsw == 0:
                V[:, j0 + 1] = 0
                SV[:, j0 + 1] = 0
            else:
                V[:, j0 + 1] = w / nsw
                SV[:, j0 + 1] = sw / nsw

            H[ind, j0] = coeffs
            H[j0 + 1, j0] = nsw

        elif ssa == 2:
            sw = hS(w); sv += 1
            SAV[:, j0] = sw

            coeffs_full = np.linalg.lstsq(SV[:, :j0 + 1], sw, rcond=None)[0] if j0 >= 0 else np.array([0.0])
            t_eff = min(t, coeffs_full.shape[0])
            sel = np.argpartition(np.abs(coeffs_full), -t_eff)[-t_eff:]
            coeffs = coeffs_full[sel]

            w = w - V[:, sel] @ coeffs
            sw = sw - SV[:, sel] @ coeffs

            nsw = np.linalg.norm(sw)
            if nsw == 0:
                V[:, j0 + 1] = 0
                SV[:, j0 + 1] = 0
            else:
                V[:, j0 + 1] = w / nsw
                SV[:, j0 + 1] = sw / nsw

            H[sel, j0] = coeffs
            H[j0 + 1, j0] = nsw

        else:
            raise ValueError("ssa must be 0, 1, or 2.")

        # -------------------------
        # Checkpoint every d iters
        # -------------------------
        if (j % d == 0) or (j == max_it) or (H[j0 + 1, j0] == 0):
            # Current bases include SV[:, :j] and SAV[:, :j]
            SW_cur = np.hstack([SW, SV[:, :j]]) if SW.size else SV[:, :j].copy()
            SAW_cur = np.hstack([SAW, SAV[:, :j]]) if SAW.size else SAV[:, :j].copy()

            y = _solve_ls(SAW_cur, Sr, method=ls_solve)
            sres_val = np.linalg.norm(Sr - SAW_cur @ y)
            sres.append(float(sres_val))

            if (sres_val < tol / sketch_distortion) or (j == max_it) or (H[j0 + 1, j0] == 0):
                nu = U.shape[1]
                if nu > 0:
                    e = U @ y[:nu] + V[:, :j] @ y[nu:]
                else:
                    e = V[:, :j] @ y

                r = r0 - A_mv(e); mv += 1
                nrmr = np.linalg.norm(r); ip += 1

                ratio = nrmr / max(sres_val, np.finfo(float).tiny)
                if ratio > sketch_distortion:
                    sketch_distortion = ratio
                    if verbose >= 1:
                        print(f"  sketch distortion increased to {sketch_distortion:g}")

                m_final = j
                if (nrmr < tol) or (j == max_it) or (H[j0 + 1, j0] == 0):
                    break

    # Final active sizes
    m = max(m_final, 1)

    # Build final SW/SAW for recycling update
    SW_cur = np.hstack([SW, SV[:, :m]]) if SW.size else SV[:, :m].copy()
    SAW_cur = np.hstack([SAW, SAV[:, :m]]) if SAW.size else SAV[:, :m].copy()

    # SVD for harmonic/standard extraction
    if SAW_cur.size == 0 and SW_cur.size == 0:
        U_new = np.empty((n, 0), dtype=r0.dtype)
        SU_new = np.empty((sdim, 0), dtype=Sr.dtype)
        SAU_new = np.empty((sdim, 0), dtype=Sr.dtype)
        keep = 0
    else:
        if harmonic:
            Lfull, Sigvals, Vh = sla.svd(SAW_cur, full_matrices=False)
        else:
            Lfull, Sigvals, Vh = sla.svd(SW_cur, full_matrices=False)

        if Sigvals.size == 0 or Sigvals[0] == 0:
            ell = 0
        else:
            ell = int(np.sum(Sigvals > svd_tol * Sigvals[0]))

        k_use = min(ell, k_target)

        if ell == 0 or k_use == 0:
            U_new = np.empty((n, 0), dtype=r0.dtype)
            SU_new = np.empty((sdim, 0), dtype=Sr.dtype)
            SAU_new = np.empty((sdim, 0), dtype=Sr.dtype)
            keep = 0
        else:
            L = Lfull[:, :ell]
            Sig = np.diag(Sigvals[:ell])
            J = Vh.conj().T[:, :ell]

            # MATLAB matching:
            # harmonic: HH = L'*SW*J ; non-harm: HH = L'*SAW*J
            if harmonic:
                HH = L.conj().T @ SW_cur @ J
            else:
                HH = L.conj().T @ SAW_cur @ J

            AA, BB, Z, keep = _ordered_qz(HH, Sig, k_use, harmonic=harmonic)

            JZ = J @ Z[:, :keep]

            nu = U.shape[1]
            if nu > 0:
                U_new = U @ JZ[:nu, :] + V[:, :m] @ JZ[nu:, :]
            else:
                U_new = V[:, :m] @ JZ

            SU_new = SW_cur @ JZ
            SAU_new = SAW_cur @ JZ

    out = dict(
        U=U_new, SU=SU_new, SAU=SAU_new, hS=hS,
        k=int(keep), m=int(m),
        mv=int(mv), ip=int(ip), sv=int(sv),
        sres=np.array(sres, dtype=float),
        sketch_distortion=float(sketch_distortion),
    )
    return e, r, out


# =============================================================================
# Top-level GMRES-SDR
# =============================================================================
def gmres_sdr(A, b, param=None):
    """
    Solve Ax=b with GMRES-SDR (sketched and deflated restarted GMRES).
    Parameters (param dict, optional):
      x0, max_it, max_restarts, tol
      U, SU, SAU, k, t
      hS (callable sketch), s
      verbose, ssa (0/1/2), d
      pert (0 reuse SAU; 1 recompute SAU each cycle)
      sketch_distortion, ls_solve ('lstsq'/'qr'/'pinv'), svd_tol, harmonic
      rng (np.random.Generator) for sketch reproducibility
    Returns: x, out
    """
    param = {} if param is None else dict(param)

    b = np.asarray(b)
    if b.ndim != 1:
        b = b.reshape(-1)
    n = b.shape[0]

    A_mv = _as_matvec(A)

    # Defaults
    verbose = int(param.get("verbose", 1))
    max_it = int(param.get("max_it", 50))
    max_restarts = int(param.get("max_restarts", min(int(np.ceil(n / max_it)), 10)))
    tol = float(param.get("tol", 1e-6))

    param.setdefault("ssa", 0)
    param.setdefault("t", 2)
    param.setdefault("k", min(10, max_it))
    param.setdefault("d", 1)
    param.setdefault("pert", 0)
    param.setdefault("sketch_distortion", 1.4)
    param.setdefault("ls_solve", "lstsq")
    param.setdefault("svd_tol", 1e-15)
    param.setdefault("harmonic", 1)

    s = int(param.get("s", min(n, 8 * (max_it + int(param["k"])))))
    param["s"] = s

    if verbose:
        print(f"  using sketching dimension of s = {s}")

    if "hS" not in param or param["hS"] is None:
        rng = param.get("rng", None)
        param["hS"] = SRCT(n, s, rng=rng)

    # Init x, r
    if "x0" in param and param["x0"] is not None:
        x = np.asarray(param["x0"]).reshape(-1)
        r = b - A_mv(x)
    else:
        x = np.zeros_like(b)
        r = b.copy()

    # Init recycling
    if "U" not in param or param["U"] is None:
        param["U"] = np.empty((n, 0), dtype=b.dtype)
        # Initialize sketch shapes correctly
        Sr = param["hS"](r)
        param["SU"] = np.empty((Sr.shape[0], 0), dtype=Sr.dtype)
        param["SAU"] = np.empty((Sr.shape[0], 0), dtype=Sr.dtype)

    residuals = [np.linalg.norm(b)]
    iters = [0]
    sres_all = [np.nan]

    mv_total = 0
    ip_total = 1  # for norm(b)
    sv_total = 0
    cycle_out = {}

    for _ in range(max_restarts):
        e, r, cycle_out = srgmres_cycle(A_mv, r, param)
        x = x + e

        resid = np.linalg.norm(r)
        residuals.append(float(resid))
        iters.append(int(cycle_out["m"]))
        sres_all.extend(list(cycle_out["sres"]))

        mv_total += int(cycle_out["mv"])
        ip_total += int(cycle_out["ip"])
        sv_total += int(cycle_out["sv"])

        # Update recycling for next cycle
        param["U"] = cycle_out["U"]
        param["SU"] = cycle_out["SU"]
        param["SAU"] = cycle_out["SAU"]
        param["sketch_distortion"] = cycle_out["sketch_distortion"]

        if resid < tol:
            break

    out = dict(cycle_out)
    out.update(
        mv=int(mv_total),
        ip=int(ip_total),
        sv=int(sv_total),
        residuals=np.array(residuals, dtype=float),
        iters=np.array(iters, dtype=int),
        sres=np.array(sres_all, dtype=float),
    )
    return x, out
