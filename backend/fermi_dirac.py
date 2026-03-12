import numpy as np
from scipy.special import ellipk, ellipj 

def fermi_dirac_stable(z: np.ndarray, beta: float, mu: float) -> np.ndarray:
    s = beta * (z - mu)
    re = s.real
    out = np.empty_like(s, dtype=np.complex128)

    m = re > 0
    t = np.exp(-s[m])
    out[m] = t / (1.0 + t)

    m2 = ~m
    t2 = np.exp(s[m2])
    out[m2] = 1.0 / (1.0 + t2)
    return out

def conformal_nodes_weights(Emin, Emax, Q, beta, mu):
    """
    Gapless (dumbbell) contour nodes/weights for g(B)=tanh(beta*B/2),
    using SciPy real ellipj + the iK'/2 shift identities.

    Returns
    -------
    xis : complex ndarray, shape (4Q,)
        Pole locations xi_j^+ followed by xi_j^-.
    ws  : complex ndarray, shape (4Q,)
        Corresponding scalar weights to approximate:
            g(B) ≈ sum_{p} ws[p] * (xis[p] I - B)^(-1)
        with B = H - mu.
    """
    # Spectral half-width around mu
    EM = max(abs(Emin - mu), abs(Emax - mu))

    # Paper's gapless parameters: m = (pi/beta)^2, M = EM^2 + m
    m = (np.pi / beta) ** 2
    M = EM**2 + m

    # Modulus k
    k = (np.sqrt(M / m) - 1.0) / (np.sqrt(M / m) + 1.0)

    # Complete elliptic integrals (SciPy uses parameter = k^2)
    K  = ellipk(k**2)
    Kp = ellipk(1.0 - k**2)  # K' = K(k'), k'^2=1-k^2

    # Real parts x_j, then t_j = x_j + i K'/2
    j = np.arange(1, Q + 1, dtype=float)
    x = -K + 2.0 * K * (j - 0.5) / Q

    # Real Jacobi elliptics at x
    snx, cnx, dnx, _ = ellipj(x, k**2)

    # Convert to values at t = x + iK'/2
    # (half imaginary-period shift identities)
    sqrtk = np.sqrt(k)
    A = 1.0 + k * snx * snx

    sn_t = ((1.0 + k) * snx + 1j * cnx * dnx) / (sqrtk * A)
    cn_t = (np.sqrt((1.0 + k) / k) * (cnx - 1j * snx * dnx)) / A
    dn_t = (np.sqrt(1.0 + k) * (dnx - 1j * k * snx * cnx)) / A

    # Möbius map z(t) (use sn_t, not snx)
    sqrt_mM = np.sqrt(m * M)
    invk = 1.0 / k
    z = sqrt_mM * (invk + sn_t) / (invk - sn_t)

    # Shifted sqrt to get xi = ±sqrt(z - m)
    xi_plus = np.sqrt(z - m + 0j)
    xi_minus = -xi_plus

    # g(xi) = tanh(beta*xi/2)
    g_plus = np.tanh(0.5 * beta * xi_plus)
    g_minus = np.tanh(0.5 * beta * xi_minus)

    # Common prefactor from (2.10)
    # gQ(B) = -(2K*sqrt(mM))/(pi*Q*k) * Im( sum_j ... )
    pref = -(2.0 * K * sqrt_mM) / (np.pi * Q * k)

    # Jacobian factor: cn(t) dn(t) / ( (k^{-1} - sn(t))^2 )
    J = cn_t * dn_t / (invk - sn_t)**2

    # The term inside Im[...] in (2.10) has extra division by xi
    alpha_plus = g_plus * (J / xi_plus)
    alpha_minus = g_minus * (J / xi_minus)

    # Convert Im( sum alpha * (xi I - B)^(-1) ) into a standard pole sum:
    # Im(Z) = (Z - conj(Z)) / (2i)
    #
    # Let S = sum alpha_j (xi_j I - B)^(-1).
    # Then gQ(B) = pref * Im(S) = pref/(2i) * (S - S^*).
    # Since B is real-symmetric/Hermitian, ( (xi I - B)^(-1) )^* = (conj(xi) I - B)^(-1).
    #
    # Therefore we can implement gQ(B) as a sum over poles at xi and conj(xi):
    #   gQ(B) = sum w_p (xi_p I - B)^(-1)
    # with weights:
    #   w(xi)      = pref/(2i) * alpha
    #   w(conj(xi))= -pref/(2i) * conj(alpha)
    #
    # This avoids taking Imag of matrices/vectors later.
    wfac = pref / (2.0j)

    xis = np.concatenate([xi_plus, xi_minus, np.conjugate(xi_plus), np.conjugate(xi_minus)])
    ws  = np.concatenate([wfac * alpha_plus,
                          wfac * alpha_minus,
                          -wfac * np.conjugate(alpha_plus),
                          -wfac * np.conjugate(alpha_minus)])

    return xis, ws

def estimate_Q(beta, EM, target_error=1e-6):
    """Estimate Q needed for target error."""
    if beta * EM < 10:
        Q = int(np.ceil(2 * np.log(1/target_error)))
    else:
        C = 1.0
        Q = int(np.ceil(np.log(beta*EM) * np.log(1/target_error) / C))
    return max(Q, 10)