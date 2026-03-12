import numpy as np
import scipy.linalg as sla
from scipy.fft import dct

# =============================================================================
# Sketch operator
# =============================================================================
class SRCT:
    """
    Subsampled Randomized Cosine Transform (SRCT).

    This class implements a structured sketching operator S ∈ R^(s x n) of the form

        S x = sqrt(n / s) * R * C * D * x,

    where
      - D is a random diagonal matrix with ±1 entries,
      - C is the orthonormal DCT-II transform,
      - R selects s rows uniformly without replacement.

    The operator can be applied to a vector x ∈ R^n or to a dense block
    X ∈ R^(n x m). In the block case, the sketch is applied columnwise.

    Notes
    -----
    - This is an SRCT sketch, not an SRFT sketch.
    - For real-valued problems, SRCT is often convenient because it stays in
      real arithmetic while still acting as a subspace embedding.
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
        """
        Apply the sketch to a vector or dense block.

        Parameters
        ----------
        X : ndarray, shape (n,) or (n, m)
            Input vector or dense block.

        Returns
        -------
        Y : ndarray, shape (s,) or (s, m)
            Sketched vector or dense block.
        """
        X = np.asarray(X)
        was_vector = (X.ndim == 1)

        if was_vector:
            X = X.reshape(-1, 1)

        if X.shape[0] != self.n:
            raise ValueError(f"SRCT expected first dimension {self.n}, got {X.shape[0]}.")

        Y = dct(self.sign[:, None] * X, type=2, norm="ortho", axis=0)
        Y = self.scale * Y[self.idx, :]

        return Y[:, 0] if was_vector else Y


# =============================================================================
# Low-level helpers
# =============================================================================

def _as_matvec(A):
    """
    Convert A into a matrix-vector callable.

    Supported inputs
    ----------------
    - callable: assumed to already implement A(v)
    - LinearOperator-like: must provide `.matvec`
    - ndarray / sparse matrix: must support `A @ v`

    Returns
    -------
    A_mv : callable
        Function such that A_mv(v) returns A @ v.
    """
    if callable(A):
        return A
    if hasattr(A, "matvec"):
        return lambda v: A.matvec(v)
    return lambda v: A @ v

def _apply_A_to_block(A_mv, U):
    """
    Apply A to each column of a dense block U.

    This helper is written defensively so it works even when A is only
    available as a vector matvec, not as a block operator.

    Parameters
    ----------
    A_mv : callable
        Matrix-vector operator.
    U : ndarray, shape (n, k)

    Returns
    -------
    AU : ndarray, shape (n, k)
    """
    if U.size == 0:
        return U
    cols = [A_mv(U[:, j]) for j in range(U.shape[1])]
    return np.column_stack(cols)


def _solve_ls(SAW, Sr, method="lstsq"):
    """
    Solve the sketched least-squares problem

        min_y || SAW y - Sr ||_2.

    Parameters
    ----------
    SAW : ndarray, shape (s, m)
        Sketched basis of the current augmented search space.
    Sr : ndarray, shape (s,)
        Sketched right-hand side / sketched residual.
    method : {"lstsq", "\\", "pinv", "qr"}
        Dense least-squares backend.

    Returns
    -------
    y : ndarray, shape (m,)
        Least-squares coefficient vector.
    """
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
    Reorder the generalized Schur form of (HH, Sig) to keep the desired
    Ritz components in the leading block.

    Parameters
    ----------
    HH : ndarray, shape (ell, ell)
        Small projected matrix from the recycling extraction step.
    Sig : ndarray, shape (ell, ell)
        Diagonal singular-value matrix from the truncated SVD.
    k : int
        Target recycling-space dimension before 2x2 block protection.
    harmonic : bool, default=True
        If True, keep the k generalized eigenvalues of largest magnitude
        (harmonic Ritz extraction). Otherwise keep the smallest ones
        (standard Ritz extraction).

    Returns
    -------
    AA : ndarray
        Reordered generalized Schur A-factor.
    BB : ndarray
        Reordered generalized Schur B-factor.
    Z : ndarray
        Right Schur vectors.
    keep : int
        Number of vectors to keep after protecting 2x2 blocks.
    """
    n = HH.shape[0]
    if k <= 0 or n == 0:
        return None, None, np.eye(n, dtype=HH.dtype), 0

    eigs = sla.eigvals(HH, Sig)
    mags = np.abs(eigs)
    mags_sorted = np.sort(mags)

    k_eff = min(k, n)

    if harmonic:
        thr = mags_sorted[-k_eff]

        def pred(alpha, beta):
            alpha = np.asarray(alpha)
            beta = np.asarray(beta)
            out = np.ones_like(alpha, dtype=bool)   # beta=0 => keep (infinite magnitude)
            mask = (np.abs(beta) > 0)
            lam = np.empty_like(alpha, dtype=np.result_type(alpha, beta, np.complex128))
            lam[mask] = alpha[mask] / beta[mask]
            out[mask] = np.abs(lam[mask]) >= thr
            return out

    else:
        thr = mags_sorted[k_eff - 1]

        def pred(alpha, beta):
            alpha = np.asarray(alpha)
            beta = np.asarray(beta)
            out = np.zeros_like(alpha, dtype=bool)  # beta=0 => do not keep for smallest magnitude
            mask = (np.abs(beta) > 0)
            lam = np.empty_like(alpha, dtype=np.result_type(alpha, beta, np.complex128))
            lam[mask] = alpha[mask] / beta[mask]
            out[mask] = np.abs(lam[mask]) <= thr
            return out

    output_type = "real" if (np.isrealobj(HH) and np.isrealobj(Sig)) else "complex"
    AA, BB, alpha, beta, Q, Z = sla.ordqz(HH, Sig, sort=pred, output=output_type)

    keep = k_eff
    if 0 < keep < AA.shape[0]:
        if (AA[keep, keep - 1] != 0) or (BB[keep, keep - 1] != 0):
            keep = min(keep + 1, AA.shape[0])

    return AA, BB, Z, keep


# =============================================================================
# One restart cycle
# =============================================================================
def srgmres_cycle(A_mv, r0, param):
    """
    Perform one GMRES-SDR restart cycle for the correction equation

        A e ≈ r0.

    The cycle builds a truncated Krylov basis, periodically solves a small
    sketched least-squares problem to estimate convergence, and forms the
    true correction only when needed. At the end of the cycle, it updates the
    recycling subspace by harmonic or standard Ritz extraction.

    Parameters
    ----------
    A_mv : callable
        Matrix-vector operator.
    r0 : ndarray, shape (n,)
        Residual at the start of the cycle.
    param : dict
        Parameter dictionary.

    Returns
    -------
    e : ndarray, shape (n,)
        Correction computed in this cycle.
    r : ndarray, shape (n,)
        True residual after applying the correction.
    out : dict
        Cycle diagnostics and updated recycling data.
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

    r0 = np.asarray(r0).reshape(-1)
    n = r0.shape[0]

    # Sketch once up front to discover the sketch dimension and dtype.
    Sr0 = hS(r0)
    sdim = Sr0.shape[0]

    # -------------------------------------------------------------------------
    # Recycling subspace and its sketches
    # -------------------------------------------------------------------------
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

    # Cycle-local counters.
    mv = 0   # matrix-vector products
    ip = 0   # inner products / explicit norms counted like MATLAB
    sv = 0   # sketch applications

    # Reuse or recompute the sketch of A*U.
    SW = SU
    if U.shape[1] == 0:
        SAW = SAU
    else:
        if pert == 0:
            SAW = SAU
        else:
            AU = _apply_A_to_block(A_mv, U)
            SAW = hS(AU)
            mv += U.shape[1]
            sv += U.shape[1]

    # -------------------------------------------------------------------------
    # Initial residual and normalized first Krylov vector
    # -------------------------------------------------------------------------
    Sr = hS(r0)
    sv += 1

    # In the sketch-and-select variants, the MATLAB code normalizes with the
    # sketch norm rather than the true residual norm.
    if ssa in (1, 2):
        nrm = np.linalg.norm(Sr)
    else:
        nrm = np.linalg.norm(r0)
        ip += 1

    # Zero residual: nothing to do.
    if nrm == 0:
        e = np.zeros_like(r0)
        out = {
            "U": U,
            "SU": SW,
            "SAU": SAW,
            "hS": hS,
            "k": int(U.shape[1]),
            "m": 0,
            "mv": int(mv),
            "ip": int(ip),
            "sv": int(sv),
            "sres": np.array([], dtype=float),
            "sketch_distortion": float(sketch_distortion),
        }
        return e, r0.copy(), out

    # Basis arrays.
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

    # -------------------------------------------------------------------------
    # Inner Arnoldi / sketched-Arnoldi loop
    # -------------------------------------------------------------------------
    for j in range(1, max_it + 1):
        j0 = j - 1
        w = A_mv(V[:, j0])
        mv += 1

        # ---------------------------------------------------------------------
        # ssa = 0: standard t-truncated Arnoldi in the full space
        # ---------------------------------------------------------------------
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
                # Happy breakdown: the current Krylov space is A-invariant.
                V[:, j0 + 1] = 0
                SV[:, j0 + 1] = 0
            else:
                V[:, j0 + 1] = w / hnext
                SV[:, j0 + 1] = hS(V[:, j0 + 1])
                sv += 1

            # For standard Arnoldi, S*A*V(:,j) can be assembled from S*V and H.
            SAV[:, j0] = SV[:, : (j0 + 2)] @ H[: (j0 + 2), j0]

        # ---------------------------------------------------------------------
        # ssa = 1: sketched t-truncated Arnoldi
        # ---------------------------------------------------------------------
        elif ssa == 1:
            sw = hS(w)
            sv += 1
            SAV[:, j0] = sw

            # Quasi-orthogonalize against the recycled subspace using sketches.
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

        # ---------------------------------------------------------------------
        # ssa = 2: sketch-and-select
        # ---------------------------------------------------------------------
        elif ssa == 2:
            sw = hS(w)
            sv += 1
            SAV[:, j0] = sw

            coeffs_full = np.linalg.lstsq(SV[:, : j0 + 1], sw, rcond=None)[0]
            t_eff = min(t, coeffs_full.shape[0])

            # Select the t entries of largest magnitude.
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
            raise ValueError("ssa must be one of {0, 1, 2}.")

        # ---------------------------------------------------------------------
        # Checkpoint every d iterations, at max_it, or on happy breakdown.
        # At a checkpoint we solve the sketched least-squares problem and only
        # form the true correction/residual if the sketch says it is worthwhile.
        # ---------------------------------------------------------------------
        if (j % d == 0) or (j == max_it) or (H[j0 + 1, j0] == 0):
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

                r = r0 - A_mv(e)
                mv += 1

                nrmr = np.linalg.norm(r)
                ip += 1

                # Keep a conservative empirical safeguard between sketched and
                # true residual norms.
                ratio = nrmr / max(sres_val, np.finfo(float).tiny)
                if ratio > sketch_distortion:
                    sketch_distortion = ratio
                    if verbose >= 1:
                        print(f"  sketch distortion increased to {sketch_distortion:g}")

                m_final = j

                if (nrmr < tol) or (j == max_it) or (H[j0 + 1, j0] == 0):
                    break

    # If we never formed a correction inside the loop, do it now from the
    # latest available basis. This should be rare, but makes the routine safe.
    if e is None or r is None:
        m_final = max(m_final, max_it)
        SW_cur = np.hstack([SW, SV[:, :m_final]]) if SW.size else SV[:, :m_final].copy()
        SAW_cur = np.hstack([SAW, SAV[:, :m_final]]) if SAW.size else SAV[:, :m_final].copy()

        y = _solve_ls(SAW_cur, Sr, method=ls_solve)
        nu = U.shape[1]
        if nu > 0:
            e = U @ y[:nu] + V[:, :m_final] @ y[nu:]
        else:
            e = V[:, :m_final] @ y

        r = r0 - A_mv(e)
        mv += 1
        ip += 1

    m = max(m_final, 1)

    # -------------------------------------------------------------------------
    # Recycling-space update
    # -------------------------------------------------------------------------
    SW_cur = np.hstack([SW, SV[:, :m]]) if SW.size else SV[:, :m].copy()
    SAW_cur = np.hstack([SAW, SAV[:, :m]]) if SAW.size else SAV[:, :m].copy()

    if SAW_cur.size == 0 and SW_cur.size == 0:
        U_new = np.empty((n, 0), dtype=r0.dtype)
        SU_new = np.empty((sdim, 0), dtype=Sr.dtype)
        SAU_new = np.empty((sdim, 0), dtype=Sr.dtype)
        keep = 0
    else:
        # Harmonic extraction uses the SVD of SAW, while standard extraction
        # uses the SVD of SW.
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

            # Match the MATLAB logic:
            #   harmonic     -> HH = L^* SW  J
            #   non-harmonic -> HH = L^* SAW J
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

    out = {
        "U": U_new,
        "SU": SU_new,
        "SAU": SAU_new,
        "hS": hS,
        "k": int(keep),
        "m": int(m),
        "mv": int(mv),
        "ip": int(ip),
        "sv": int(sv),
        "sres": np.array(sres, dtype=float),
        "sketch_distortion": float(sketch_distortion),
    }
    return e, r, out


# =============================================================================
# Top-level solver
# =============================================================================
def gmres_sdr(A, b, param=None):
    """
    Solve Ax = b with GMRES-SDR:
    sketched, deflated, restarted GMRES with recycling.

    Parameters
    ----------
    A : ndarray, sparse matrix, LinearOperator, or callable
        Linear operator for the system. If callable, it must implement A(v).
    b : ndarray, shape (n,)
        Right-hand side.
    param : dict, optional
        Solver parameters. Supported entries include

        x0 : ndarray
            Initial guess.
        max_it : int
            Maximum number of inner iterations per restart cycle.
        max_restarts : int
            Maximum number of restart cycles.
        tol : float
            Absolute residual tolerance.
        U, SU, SAU : ndarrays
            Recycling basis and its sketches.
        k : int
            Target recycling-space dimension.
        t : int
            Truncation parameter in Arnoldi.
        hS : callable
            Sketch operator.
        s : int
            Sketch dimension, used only if hS is not supplied.
        verbose : int
            Verbosity level.
        ssa : int
            Arnoldi variant:
                0 -> standard t-truncated Arnoldi
                1 -> sketched t-truncated Arnoldi
                2 -> sketch-and-select
        d : int
            Checkpoint frequency for sketched residual tests.
        pert : int
            If 0, reuse SAU across cycles. If 1, recompute SAU each cycle.
        sketch_distortion : float
            Conservative bound relating true and sketched residual norms.
        ls_solve : str
            Least-squares backend: {"lstsq", "\\", "pinv", "qr"}.
        svd_tol : float
            Relative tolerance for truncating the recycling SVD.
        harmonic : bool or int
            If true, use harmonic Ritz extraction; otherwise standard Ritz.
        rng : np.random.Generator
            Random generator used when creating the default sketch.

    Returns
    -------
    x : ndarray, shape (n,)
        Approximate solution.
    out : dict
        Aggregate solver diagnostics with fields including

        residuals : ndarray
            True residual norms by restart cycle.
        iters : ndarray
            Inner iteration counts by cycle.
        sres : ndarray
            Sketched residual history collected at checkpoints.
        mv, ip, sv : int
            Counts of matrix-vector products, inner products, and sketches.
        U, SU, SAU : ndarrays
            Updated recycling basis and its sketches.
    """
    param = {} if param is None else dict(param)

    b = np.asarray(b).reshape(-1)
    n = b.shape[0]

    A_mv = _as_matvec(A)

    # -------------------------------------------------------------------------
    # Defaults
    # -------------------------------------------------------------------------
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

    if ("hS" not in param) or (param["hS"] is None):
        rng = param.get("rng", None)
        param["hS"] = SRCT(n, s, rng=rng)

    # -------------------------------------------------------------------------
    # Initial guess and residual
    # -------------------------------------------------------------------------
    if ("x0" in param) and (param["x0"] is not None):
        x = np.asarray(param["x0"]).reshape(-1)
        r = b - A_mv(x)
    else:
        x = np.zeros_like(b)
        r = b.copy()

    # -------------------------------------------------------------------------
    # Initialize recycling data if absent
    # -------------------------------------------------------------------------
    if ("U" not in param) or (param["U"] is None):
        param["U"] = np.empty((n, 0), dtype=b.dtype)
        Sr = param["hS"](r)
        param["SU"] = np.empty((Sr.shape[0], 0), dtype=Sr.dtype)
        param["SAU"] = np.empty((Sr.shape[0], 0), dtype=Sr.dtype)
    else:
        # Make sure sketch blocks exist if a recycle space is supplied.
        U = param["U"]
        Sr = param["hS"](r)
        if ("SU" not in param) or (param["SU"] is None):
            param["SU"] = param["hS"](U) if U.size else np.empty((Sr.shape[0], 0), dtype=Sr.dtype)
        if ("SAU" not in param) or (param["SAU"] is None):
            if U.size:
                param["SAU"] = param["hS"](_apply_A_to_block(A_mv, U))
            else:
                param["SAU"] = np.empty((Sr.shape[0], 0), dtype=Sr.dtype)

    # Diagnostics accumulated across restart cycles.
    residuals = [float(np.linalg.norm(r))]
    iters = [0]
    sres_all = [np.nan]

    mv_total = 0
    ip_total = 1  # for the initial true residual norm
    sv_total = 0

    cycle_out = {
        "U": param["U"],
        "SU": param["SU"],
        "SAU": param["SAU"],
        "hS": param["hS"],
        "k": int(param["U"].shape[1]),
        "m": 0,
        "mv": 0,
        "ip": 0,
        "sv": 0,
        "sres": np.array([], dtype=float),
        "sketch_distortion": float(param["sketch_distortion"]),
    }

    # Early exit if the initial guess already satisfies the tolerance.
    if residuals[0] < tol:
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

    # -------------------------------------------------------------------------
    # Outer restart loop
    # -------------------------------------------------------------------------
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

        # Carry the updated recycle space into the next cycle.
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