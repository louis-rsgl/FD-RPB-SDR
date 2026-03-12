import numpy as np
import scipy.sparse as sp


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