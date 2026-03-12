import numpy as np
import scipy.sparse as sp
from backend.fermi_dirac import conformal_nodes_weights
from backend.GMRES_SDR import gmres_sdr


# ----------------------------
# Geometry & supercell utilities (your code, trimmed minimally)
# ----------------------------
def graphene_prim_vectors(a=2.46):
    a1 = np.array([a, 0.0])
    a2 = np.array([0.5 * a, 0.5 * np.sqrt(3) * a])
    return a1, a2

def graphene_basis(a=2.46):
    a1, a2 = graphene_prim_vectors(a)
    d = (a1 + a2) / 3.0
    rA = np.array([0.0, 0.0])
    rB = d
    return rA, rB

def supercell_vectors(m, n, a=2.46):
    a1, a2 = graphene_prim_vectors(a)
    A1 = m * a1 + n * a2
    A2 = -n * a1 + (m + n) * a2
    return A1, A2, a1, a2

def reciprocal_vectors(A1, A2):
    M = np.stack([A1, A2], axis=1)
    BinvT = 2.0 * np.pi * np.linalg.inv(M).T
    B1 = BinvT[:, 0]
    B2 = BinvT[:, 1]
    return B1, B2

def frac_coords_in_supercell(r, A1, A2):
    M = np.stack([A1, A2], axis=1)
    return np.linalg.solve(M, r)

def enumerate_supercell_translations(m: int, n: int, R: int):
    """
    Generate candidate primitive translation indices (u,v) on a square window [-R, R]^2.

    Returns
    -------
    cand : list[tuple[int,int]]
        Candidate primitive-cell translation indices.
    Ncell : int
        Expected number of primitive cells in the (m,n) hex-preserving supercell:
            Ncell = m^2 + n^2 + m n
    """
    if R < 0:
        raise ValueError("R must be non-negative.")
    cand = [(u, v) for u in range(-R, R + 1) for v in range(-R, R + 1)]
    Ncell = m * m + n * n + m * n
    return cand, Ncell


def build_supercell(m: int, n: int, a: float = 2.46):
    """
    Build a graphene (m,n) hex-preserving supercell.

    Supercell vectors:
        A1 =  m a1 + n a2
        A2 = -n a1 + (m+n) a2

    We enumerate exactly Ncell = m^2 + n^2 + m n primitive translations (u,v) such that
    the corresponding real-space vector R_uv = u a1 + v a2 lies inside the supercell
    parallelogram spanned by (A1,A2), i.e. fractional coordinates satisfy:
        0 <= s < 1, 0 <= t < 1  where R_uv = s A1 + t A2.

    This implementation is robust: it grows the candidate window size R until it finds
    exactly Ncell translations.

    Returns
    -------
    dict containing:
        A1, A2 : (2,) ndarray
        B1, B2 : (2,) ndarray (reciprocal vectors)
        positions : (Nat,2) ndarray wrapped positions in the home supercell
        sublat : (Nat,) ndarray, 0 for A and 1 for B
        frac : (Nat,2) ndarray wrapped fractional coords in [0,1)
        a : float lattice constant used
    """
    A1, A2, a1, a2 = supercell_vectors(m, n, a)
    B1, B2 = reciprocal_vectors(A1, A2)
    rA, rB = graphene_basis(a)

    # Target number of primitive cells in the supercell
    Ncell_target = m * m + n * n + m * n

    # Enumerate translations robustly by expanding the search window until complete
    trans = []
    R = max(m, n) + 2
    max_R = 20 * (m + n + 2)

    while True:
        cand_uv, _ = enumerate_supercell_translations(m, n, R)
        trans = []

        for (u, v) in cand_uv:
            Ruv = u * a1 + v * a2
            s, t = frac_coords_in_supercell(Ruv, A1, A2)
            if (-1e-10 <= s < 1.0 - 1e-10) and (-1e-10 <= t < 1.0 - 1e-10):
                trans.append((u, v))

        if len(trans) == Ncell_target:
            break

        if len(trans) < Ncell_target:
            R += max(m, n)
            if R > max_R:
                raise RuntimeError(
                    f"Failed to enumerate supercell translations after expanding window. "
                    f"m={m}, n={n}, got {len(trans)} expected {Ncell_target}, last R={R}."
                )
            continue

        # len(trans) > Ncell_target should not happen if the fractional filter is correct
        raise RuntimeError(
            f"Enumeration produced too many translations: got {len(trans)} expected {Ncell_target}."
        )

    # Build positions for all A/B in each primitive translation
    positions = []
    sublat = []
    frac = []
    for (u, v) in trans:
        Ruv = u * a1 + v * a2
        for s_idx, rb in enumerate((rA, rB)):
            r = Ruv + rb
            positions.append(r)
            sublat.append(s_idx)
            frac.append(frac_coords_in_supercell(r, A1, A2))

    positions = np.array(positions, dtype=float)
    sublat = np.array(sublat, dtype=int)
    frac = np.array(frac, dtype=float)

    # Wrap fractional coords into [0,1)
    frac_wrapped = frac - np.floor(frac)

    # Recompute wrapped positions from supercell basis (numerically stable)
    positions_wrapped = frac_wrapped[:, 0:1] * A1 + frac_wrapped[:, 1:2] * A2

    return {
        "A1": A1, "A2": A2, "B1": B1, "B2": B2,
        "positions": positions_wrapped,
        "sublat": sublat,
        "frac": frac_wrapped,
        "a": a,
    }


def k_from_grid_index(cell, ik1, ik2, Nk1, Nk2, gamma_centered=True):
    B1, B2 = cell["B1"], cell["B2"]
    if gamma_centered:
        x = ik1 / Nk1
        y = ik2 / Nk2
    else:
        x = (ik1 + 0.5) / Nk1
        y = (ik2 + 0.5) / Nk2
    return x * B1 + y * B2


# ----------------------------
# Bonds: keep your routine, returns directed bonds with translation T and hop t
# ----------------------------
def build_first_nn_bonds(cell, t=-2.7, tol=2e-2):
    pos = cell["positions"]
    A1, A2 = cell["A1"], cell["A2"]
    a = cell["a"]
    d_nn = a / np.sqrt(3.0)

    Nat = pos.shape[0]
    bonds = []

    shifts = []
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            shifts.append((n1, n2, n1 * A1 + n2 * A2))

    for i in range(Nat):
        ri = pos[i]
        for (n1, n2, Tvec) in shifts:
            rj_all = pos + Tvec
            dr = rj_all - ri
            dist = np.sqrt((dr * dr).sum(axis=1))
            js = np.where((np.abs(dist - d_nn) < tol) & (dist > 1e-8))[0]
            for j in js:
                bonds.append((i, j, Tvec.copy(), t))

    # deduplicate using integer (n1,n2)
    uniq = {}
    M = np.stack([A1, A2], axis=1)
    for (i, j, Tvec, hop) in bonds:
        st = np.linalg.solve(M, Tvec)
        n1 = int(np.round(st[0]))
        n2 = int(np.round(st[1]))
        key = (i, j, n1, n2)
        uniq[key] = (i, j, n1, n2, hop)

    bonds_u = []
    for (i, j, n1, n2, hop) in uniq.values():
        T = n1 * A1 + n2 * A2
        bonds_u.append((i, j, T, hop))

    return bonds_u


# ----------------------------
# Sparse Bloch Hamiltonian builder
# ----------------------------
def Hk_from_bonds_sparse(cell, bonds, kvec, onsite=0.0, make_hermitian=True):
    """
    Build sparse CSR H(k) from bonds.

    bonds entries: (i, j, T, hop) meaning hopping i->j with translation T.

    We assemble H_ij += hop * exp(i k·T).
    If make_hermitian=True, we enforce Hermiticity by averaging:
        H <- 0.5*(H + H^H)
    which is robust even if bonds are not perfectly symmetric.

    Parameters
    ----------
    onsite : float or array-like
        Onsite term(s) added to diagonal.
    """
    Nat = cell["positions"].shape[0]
    rows = []
    cols = []
    data = []

    for (i, j, T, hop) in bonds:
        phase = np.exp(1j * float(np.dot(kvec, T)))
        rows.append(i)
        cols.append(j)
        data.append(hop * phase)

    H = sp.coo_matrix((np.array(data, dtype=complex), (rows, cols)), shape=(Nat, Nat)).tocsr()

    # onsite
    if np.isscalar(onsite):
        if onsite != 0.0:
            H = H + sp.identity(Nat, format="csr", dtype=complex) * complex(onsite)
    else:
        onsite = np.asarray(onsite, dtype=float)
        H = H + sp.diags(onsite, format="csr").astype(complex)

    if make_hermitian:
        H = 0.5 * (H + H.getH())

    return H



m = 29
n = 29
t = -2.7

cell = build_supercell(m, n, a=2.46)
bonds = build_first_nn_bonds(cell, t=t)
kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)

Hk_small = Hk_from_bonds_sparse(cell, bonds, kvec, onsite=0.0, make_hermitian=True).tocsr()

Nat_small = Hk_small.shape[0]
print("Nat =", Nat_small)
print("nnz =", Hk_small.nnz, "  nnz/N =", Hk_small.nnz / Nat_small)


m = 41
n = 41
t = -2.7

cell = build_supercell(m, n, a=2.46)
bonds = build_first_nn_bonds(cell, t=t)
kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)

Hk_medium = Hk_from_bonds_sparse(cell, bonds, kvec, onsite=0.0, make_hermitian=True).tocsr()

Nat_medium = Hk_medium.shape[0]
print("Nat =", Nat_medium)
print("nnz =", Hk_medium.nnz, "  nnz/N =", Hk_medium.nnz / Nat_medium)


m = 58
n = 58
t = -2.7

cell = build_supercell(m, n, a=2.46)
bonds = build_first_nn_bonds(cell, t=t)
kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)

Hk_big = Hk_from_bonds_sparse(cell, bonds, kvec, onsite=0.0, make_hermitian=True).tocsr()

Nat_big = Hk_big.shape[0]
print("Nat =", Nat_big)
print("nnz =", Hk_big.nnz, "  nnz/N =", Hk_big.nnz / Nat_big)


m = 82
n = 82
t = -2.7

cell = build_supercell(m, n, a=2.46)
bonds = build_first_nn_bonds(cell, t=t)
kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)

Hk_huge = Hk_from_bonds_sparse(cell, bonds, kvec, onsite=0.0, make_hermitian=True).tocsr()

Nat_huge = Hk_huge.shape[0]
print("Nat =", Nat_huge)
print("nnz =", Hk_huge.nnz, "  nnz/N =", Hk_huge.nnz / Nat_huge)


m = 100
n = 100
t = -2.7

cell = build_supercell(m, n, a=2.46)
bonds = build_first_nn_bonds(cell, t=t)
kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)

Hk_peta = Hk_from_bonds_sparse(cell, bonds, kvec, onsite=0.0, make_hermitian=True).tocsr()

Nat_peta = Hk_peta.shape[0]
print("Nat =", Nat_peta)
print("nnz =", Hk_peta.nnz, "  nnz/N =", Hk_peta.nnz / Nat_peta)


m = 130
n = 130
t = -2.7

cell = build_supercell(m, n, a=2.46)
bonds = build_first_nn_bonds(cell, t=t)
kvec = k_from_grid_index(cell, 0, 0, 1, 1, gamma_centered=True)

Hk_peta = Hk_from_bonds_sparse(cell, bonds, kvec, onsite=0.0, make_hermitian=True).tocsr()

Nat_peta = Hk_peta.shape[0]
print("Nat =", Nat_peta)
print("nnz =", Hk_peta.nnz, "  nnz/N =", Hk_peta.nnz / Nat_peta)


# =============================================================================
# Full benchmark cell: GMRES vs GMRES-SDR under Option A
#   - Same absolute true-residual tolerance per pole
#   - Comparable restart/maxiter: GMRES(restart=m,maxiter=R) <-> SDR(max_it=m,max_restarts=R)
#   - Reports: Q, poles, rel_err vs reference, time, failures, max true residual, (SDR matvecs)
# =============================================================================

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -------------------------------
# Utilities: true-residual GMRES
# -------------------------------
class MatvecCounter(spla.LinearOperator):
    """LinearOperator wrapper to count matvecs."""
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.dtype = np.dtype(np.complex128)
        self.mv = 0

    def _matvec(self, v):
        self.mv += 1
        return self.A @ v
    
def precompute_eigh(H_csr):
    H_dense = H_csr.toarray().astype(np.complex128, copy=False)
    evals, V = np.linalg.eigh(H_dense)
    return evals, V

def apply_reference_from_eigh(evals, V, x, beta, mu):
    s = beta * (evals - mu)
    s = np.clip(s, -700.0, 700.0)
    f = 1.0 / (1.0 + np.exp(s))
    return V @ (f * (V.conj().T @ x))


def gmres_solve_true_residual(A, b, tol_abs, restart, maxiter_cycles):
    """
    Solve Ax=b with SciPy GMRES using absolute tolerance only (rtol=0),
    return x, info, mv_count, true_res_norm.
    """
    Aop = MatvecCounter(A)
    x, info = spla.gmres(
        Aop, b,
        restart=restart,
        maxiter=maxiter_cycles,
        atol=tol_abs,
        rtol=0.0,
    )
    r = b - (A @ x)
    return x, info, Aop.mv, float(np.linalg.norm(r))


# -------------------------------
# Apply: conformal + GMRES
# -------------------------------
def conformal_apply_sparse_gmres(
    H_csr, x, beta, mu, Emin, Emax, Q,
    tol_abs=1e-10,
    restart=80,
    maxiter_cycles=50,
):
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

    for xi, w in zip(xis, ws):
        A = (xi + mu) * I - Hc
        v, info, mv, true_res = gmres_solve_true_residual(
            A, x, tol_abs=tol_abs, restart=restart, maxiter_cycles=maxiter_cycles
        )
        total_mv += mv
        max_true_res = max(max_true_res, true_res)

        # "failure" defined by not meeting absolute true residual tolerance
        if true_res > tol_abs:
            failures += 1
        if info != 0:
            info_bad += 1

        gx += w * v

    y = 0.5 * (x - gx)
    stats = dict(
        poles=len(xis),
        failures=failures,
        info_bad=info_bad,
        max_true_res=max_true_res,
        total_mv=total_mv,
    )
    return y, stats


# -------------------------------
# Apply: conformal + GMRES-SDR
# -------------------------------
def conformal_apply_sparse_gmressdr(
    H_csr, x, beta, mu, Emin, Emax, Q,
    tol_abs=1e-10,
    restart=80,            # maps to max_it
    maxiter_cycles=50,     # maps to max_restarts
    sdr_fixed=None,        # dict of fixed SDR params: k,t,s,ssa,d,pert,verbose,rng,ls_solve,harmonic,sketch_distortion,svd_tol
    recycle_across_poles=False,  # default OFF for strict per-pole fairness
):
    Hc = H_csr.tocsc().astype(np.complex128, copy=False)
    n = Hc.shape[0]
    x = x.astype(np.complex128, copy=False)

    xis, ws = conformal_nodes_weights(Emin, Emax, Q, beta, mu)
    I = sp.eye(n, format="csc", dtype=np.complex128)

    base = dict(
        tol=tol_abs,
        max_it=int(restart),
        max_restarts=int(maxiter_cycles),
        harmonic=1,
        ls_solve="lstsq",
        verbose=0,
    )
    if sdr_fixed:
        base.update(sdr_fixed)

    gx = np.zeros(n, dtype=np.complex128)

    failures = 0
    max_true_res = 0.0
    total_mv = 0

    # If recycling across poles is enabled, maintain one param dict across poles.
    param_shared = dict(base)
    if recycle_across_poles:
        param_shared["U"] = None
        param_shared["SU"] = None
        param_shared["SAU"] = None

    for xi, w in zip(xis, ws):
        A = (xi + mu) * I - Hc

        if recycle_across_poles:
            param = param_shared
        else:
            # strict per-pole: reset recycling each pole
            param = dict(base)
            param["U"] = None
            param["SU"] = None
            param["SAU"] = None

        v, out = gmres_sdr(A, x, param)

        true_res = float(out["residuals"][-1])
        max_true_res = max(max_true_res, true_res)
        total_mv += int(out["mv"])

        if true_res > tol_abs:
            failures += 1

        # if recycling across poles: carry updated subspace forward explicitly
        if recycle_across_poles:
            param_shared["U"] = out.get("U", None)
            param_shared["SU"] = out.get("SU", None)
            param_shared["SAU"] = out.get("SAU", None)
            if "sketch_distortion" in out:
                param_shared["sketch_distortion"] = out["sketch_distortion"]

        gx += w * v

    y = 0.5 * (x - gx)
    stats = dict(
        poles=len(xis),
        failures=failures,
        max_true_res=max_true_res,
        total_mv=total_mv,
    )
    return y, stats


# -------------------------------
# Q-sweep driver (to target global error)
# -------------------------------
def conformal_to_target_error_vs_ref(
    H_csr, x, y_ref, beta, mu, Emin, Emax,
    solver="gmres",                 # "gmres" or "sdr"
    Q0=16, max_Q=4096, tol_global=1e-12,
    tol_abs_pole=1e-10,
    restart=80,
    maxiter_cycles=50,
    sdr_fixed=None,
    recycle_across_poles=False,
):
    ref_norm = np.linalg.norm(y_ref) + 1e-30
    Q = Q0 if Q0 % 2 == 0 else Q0 + 1

    best = None
    while True:
        t0 = time.perf_counter()

        if solver == "gmres":
            y, st = conformal_apply_sparse_gmres(
                H_csr, x, beta, mu, Emin, Emax, Q,
                tol_abs=tol_abs_pole,
                restart=restart,
                maxiter_cycles=maxiter_cycles,
            )
        elif solver == "sdr":
            y, st = conformal_apply_sparse_gmressdr(
                H_csr, x, beta, mu, Emin, Emax, Q,
                tol_abs=tol_abs_pole,
                restart=restart,
                maxiter_cycles=maxiter_cycles,
                sdr_fixed=sdr_fixed,
                recycle_across_poles=recycle_across_poles,
            )
        else:
            raise ValueError("solver must be 'gmres' or 'sdr'")

        dt = time.perf_counter() - t0
        rel_err = np.linalg.norm(y - y_ref) / ref_norm

        best = (y, Q, rel_err, dt, st)

        # Print one line per Q
        mv_str = f"{st.get('total_mv', -1):9d}"
        if solver == "gmres":
            extra = f"mv={mv_str} info_bad={st.get('info_bad', 0):5d}"
        else:
            extra = f"mv={mv_str}"

        print(f"{Q:6d} | poles={st['poles']:7d} | err={rel_err:9.2e} | "
              f"t={dt:7.3f}s | fails={st['failures']:5d} | "
              f"max||r||={st['max_true_res']:.2e} | {extra}")

        if (rel_err < tol_global) or (Q >= max_Q):
            return best

        Q = min(4 * Q, max_Q)


Emin, Emax = -10.0, 10.0

T_kelvin = np.array([100.0, 10.0, 1.0, 0.1], dtype=float)
kB = 8.617333262145e-5
betas = 1.0 / (kB * T_kelvin)
mu = 0.0

Nat = Hk_small.shape[0]
rng = np.random.default_rng(0)
x = np.exp(1j * rng.uniform(0, 2 * np.pi, size=Nat)).astype(np.complex128)
x /= np.linalg.norm(x)

# Reference eigendecomposition ONCE (assumes Hk_small is small enough for dense eig)
evals, V = precompute_eigh(Hk_small)

# --- Solver comparability knobs (Option A) ---
tol_abs_pole = 1e-8     # absolute true residual tolerance per pole solve
tol_global   = 1e-8     # target relative error vs reference for overall f(H)x
restart_m    = 240        # GMRES restart length m  <-> SDR max_it
max_cycles_R = 200        # GMRES maxiter cycles R <-> SDR max_restarts
    
print("\nGMRES summary (Per-pole abs true residual tol):")
for T, beta in zip(T_kelvin, betas):
    print("\n" + "=" * 80)
    print(f"T={T:8.3f} K  beta={beta:12.6g} 1/eV  mu={mu}")
    print(f"GMRES: restart={restart_m}, maxiter(cycles)={max_cycles_R}, tol_abs_pole={tol_abs_pole:g}")
    print("=" * 80)

    y_ref = apply_reference_from_eigh(evals, V, x, beta, mu)

    best = conformal_to_target_error_vs_ref(
        Hk_small, x, y_ref, beta, mu, Emin, Emax,
        solver="gmres",
        Q0=16, max_Q=51200,
        tol_global=tol_global,
        tol_abs_pole=tol_abs_pole,
        restart=restart_m,
        maxiter_cycles=max_cycles_R,
    )

    y_best, Q_best, err_best, t_best, st_best = best
    print(f"  BEST: Q={Q_best} (poles={4*Q_best}), err={err_best:.3e}, t={t_best:.3f}s, "
          f"fails={st_best['failures']}, max||r||={st_best['max_true_res']:.2e}, mv={st_best['total_mv']}")
    
# Fixed SDR params (edit these as you like; keep fixed across runs for fairness)
sdr_fixed = dict(
    k=min(30, restart_m // 2),
    t=restart_m//3,
    s=min(Nat, 16 * (restart_m + min(30, restart_m // 2))),
    ssa=0,
    d=1,
    pert=0,
    verbose=0,
    sketch_distortion=5.0,
    rng=np.random.default_rng(0),
    ls_solve="qr",
    harmonic=1,
    svd_tol=1e-15,
)

print("\nGMRES-SDR summary (Per-pole abs true residual tol):")
for T, beta in zip(T_kelvin, betas):
    print("\n" + "=" * 80)
    print(f"T={T:8.3f} K  beta={beta:12.6g} 1/eV  mu={mu}")
    print(f"SDR: max_it={restart_m}, max_restarts={max_cycles_R}, tol_abs_pole={tol_abs_pole:g}")
    print("=" * 80)

    y_ref = apply_reference_from_eigh(evals, V, x, beta, mu)

    best = conformal_to_target_error_vs_ref(
        Hk_small, x, y_ref, beta, mu, Emin, Emax,
        solver="sdr",
        Q0=16, max_Q=51200,
        tol_global=tol_global,
        tol_abs_pole=tol_abs_pole,
        restart=restart_m,
        maxiter_cycles=max_cycles_R,
        sdr_fixed=sdr_fixed,
        recycle_across_poles=False,   # strict per-pole comparison; set True to test SDR recycling advantage
    )

    y_best, Q_best, err_best, t_best, st_best = best
    print(f"  BEST: Q={Q_best} (poles={4*Q_best}), err={err_best:.3e}, t={t_best:.3f}s, "
          f"fails={st_best['failures']}, max||r||={st_best['max_true_res']:.2e}, mv={st_best['total_mv']}")
    
Emin, Emax = -10.0, 10.0

T_kelvin = np.array([100.0, 10.0, 1.0, 0.1], dtype=float)
kB = 8.617333262145e-5
betas = 1.0 / (kB * T_kelvin)
mu = 0.0

Nat = Hk_medium.shape[0]
rng = np.random.default_rng(0)
x = np.exp(1j * rng.uniform(0, 2 * np.pi, size=Nat)).astype(np.complex128)
x /= np.linalg.norm(x)

# Reference eigendecomposition ONCE (assumes Hk_small is small enough for dense eig)
evals, V = precompute_eigh(Hk_medium)

# --- Solver comparability knobs (Option A) ---
tol_abs_pole = 1e-8     # absolute true residual tolerance per pole solve
tol_global   = 1e-10     # target relative error vs reference for overall f(H)x
restart_m    = 80        # GMRES restart length m  <-> SDR max_it
max_cycles_R = 50        # GMRES maxiter cycles R <-> SDR max_restarts

# Fixed SDR params (edit these as you like; keep fixed across runs for fairness)
sdr_fixed = dict(
    k=10,
    t=2,
    s=min(Nat, 8 * (restart_m + 10)),
    ssa=0,
    d=1,
    pert=0,
    verbose=0,
    rng=np.random.default_rng(0),
    ls_solve="lstsq",
    harmonic=1,
)

print("\nGMRES summary (Option A: per-pole abs true residual tol):")
for T, beta in zip(T_kelvin, betas):
    print("\n" + "=" * 80)
    print(f"T={T:8.3f} K  beta={beta:12.6g} 1/eV  mu={mu}")
    print(f"GMRES: restart={restart_m}, maxiter(cycles)={max_cycles_R}, tol_abs_pole={tol_abs_pole:g}")
    print("=" * 80)

    y_ref = apply_reference_from_eigh(evals, V, x, beta, mu)

    best = conformal_to_target_error_vs_ref(
        Hk_medium, x, y_ref, beta, mu, Emin, Emax,
        solver="gmres",
        Q0=16, max_Q=51200,
        tol_global=tol_global,
        tol_abs_pole=tol_abs_pole,
        restart=restart_m,
        maxiter_cycles=max_cycles_R,
    )

    y_best, Q_best, err_best, t_best, st_best = best
    print(f"  BEST: Q={Q_best} (poles={4*Q_best}), err={err_best:.3e}, t={t_best:.3f}s, "
          f"fails={st_best['failures']}, max||r||={st_best['max_true_res']:.2e}, mv={st_best['total_mv']}")

print("\nGMRES-SDR summary (Option A: per-pole abs true residual tol):")
for T, beta in zip(T_kelvin, betas):
    print("\n" + "=" * 80)
    print(f"T={T:8.3f} K  beta={beta:12.6g} 1/eV  mu={mu}")
    print(f"SDR: max_it={restart_m}, max_restarts={max_cycles_R}, tol_abs_pole={tol_abs_pole:g}")
    print("=" * 80)

    y_ref = apply_reference_from_eigh(evals, V, x, beta, mu)

    best = conformal_to_target_error_vs_ref(
        Hk_medium, x, y_ref, beta, mu, Emin, Emax,
        solver="sdr",
        Q0=16, max_Q=51200,
        tol_global=tol_global,
        tol_abs_pole=tol_abs_pole,
        restart=restart_m,
        maxiter_cycles=max_cycles_R,
        sdr_fixed=sdr_fixed,
        recycle_across_poles=False,   # strict per-pole comparison; set True to test SDR recycling advantage
    )

    y_best, Q_best, err_best, t_best, st_best = best
    print(f"  BEST: Q={Q_best} (poles={4*Q_best}), err={err_best:.3e}, t={t_best:.3f}s, "
          f"fails={st_best['failures']}, max||r||={st_best['max_true_res']:.2e}, mv={st_best['total_mv']}")