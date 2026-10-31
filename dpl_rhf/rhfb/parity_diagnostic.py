"""Independent number parity from a pure HFB generalized density.

The real orthogonal representation of canonical transformations and its
Pfaffian congruence law follow Kasuya & Yoshida, arXiv:2005.03276, Eqs.16--22.
Here the flattening I-2R is used instead of a physical Hamiltonian. Dividing
by the empty particle vacuum Pfaffian fixes the convention for any dimension.
This is a diagnostic, not a differentiable training objective or a spectral
solver. Mixed densities, including EFA, have no definite pure-state parity.
"""
from __future__ import annotations

import numpy as np


def _pfaffian_real_skew(matrix):
    a = np.asarray(matrix, dtype=np.float64).copy()
    n = a.shape[0]
    result = 1.0
    for k in range(0, n - 1, 2):
        j = k + 1 + int(np.argmax(np.abs(a[k, k + 1:])))
        if j != k + 1:
            a[[k + 1, j], :] = a[[j, k + 1], :]
            a[:, [k + 1, j]] = a[:, [j, k + 1]]
            result = -result
        pivot = a[k, k + 1]
        if abs(pivot) < 1e-14:
            return 0.0
        result *= pivot
        first, second = a[k, k + 2:].copy(), a[k + 1, k + 2:].copy()
        a[k + 2:, k + 2:] -= (np.outer(first, second) - np.outer(second, first)) / pivot
    return float(result)


def number_parity_from_density(rho, chi, *, tolerance=1e-10):
    """Compute parity without trusting stored reference labels or eigensolves."""
    rho, chi = np.asarray(rho, dtype=np.complex128), np.asarray(chi, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or chi.shape != rho.shape:
        raise ValueError("rho and chi must be matching square matrices")
    if not np.isfinite(rho).all() or not np.isfinite(chi).all():
        raise ValueError("density contains nonfinite values")
    m = rho.shape[0]
    eye = np.eye(m)
    r = np.block([[rho, chi], [-chi.conj(), eye - rho.conj()]])
    hermiticity = float(np.linalg.norm(r - r.conj().T))
    purity = float(np.linalg.norm(r @ r - r))
    particle_hole = float(np.linalg.norm(chi + chi.T))
    if max(hermiticity, purity, particle_hole) > tolerance:
        raise ValueError("definite pure-state parity requires Hermitian, idempotent canonical R")
    x = np.block([[eye, eye], [1j * eye, -1j * eye]]) / np.sqrt(2)
    skew = -1j * x @ (np.eye(2 * m) - 2 * r) @ x.conj().T
    reality_error = float(np.linalg.norm(skew.imag))
    skew_error = float(np.linalg.norm(skew + skew.T))
    if max(reality_error, skew_error) > tolerance:
        raise ValueError("density has no real canonical skew representation")
    vacuum_pf = (-1.0) ** (m * (m + 1) // 2)
    value = _pfaffian_real_skew(skew.real) / vacuum_pf
    if abs(abs(value) - 1.0) > tolerance * max(1, m):
        raise ValueError("Pfaffian does not have unit magnitude for a pure HFB state")
    return {"number_parity": 1 if value > 0 else -1, "normalized_pfaffian": value,
            "particle_number": float(np.trace(rho).real), "dimension": m,
            "hermiticity_error": hermiticity, "purity_error": purity,
            "chi_antisymmetry_error": particle_hole,
            "majorana_reality_error": reality_error, "majorana_skew_error": skew_error}
