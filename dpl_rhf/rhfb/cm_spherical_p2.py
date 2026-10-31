"""Source-faithful spherical radial matrix elements of the one-body p^2.

This is an isolated helper for checking the one-body term in a microscopic
center-of-mass correction.  It implements the reduced-radial expression in
Geng and Long, arXiv:2202.00482, Appendix C, Eq. (C2), independently for the
upper and lower Dirac components.  It does not construct vector p_mu, does
not infer p^2 from a truncated sum of p_mu matrices, and is not an ADF/O17
operator or part of the self-consistent energy.

The caller supplies a radial second-derivative matrix with its intended box
boundary treatment and quadrature weights for dr.  The routine rejects a
non-Hermitian result instead of silently symmetrizing it; this makes failures
of the radial metric/boundary convention visible.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def spherical_reduced_radial_p2(
    radius_fm: Any,
    radial_weights_fm: Any,
    upper_values: Any,
    lower_values: Any,
    second_derivative_fm2: Any,
    *,
    upper_l: int,
    lower_l: int,
    hbar_c_mev_fm: float = 197.3269804,
    hermiticity_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    r"""Build a spherical-channel matrix of ``p^2`` in MeV^2.

    ``upper_values[i,a]`` and ``lower_values[i,a]`` are reduced radial
    spinor components ``G_a(r_i)`` and ``F_a(r_i)`` on the same radial grid;
    columns are radial states from one common spherical angular channel.  The
    radial quadrature integrates ``dr`` (not ``r^2 dr``), as appropriate for
    reduced radial components.  The discrete operator is the direct Eq. (C2)
    action

        -d^2/dr^2 + l(l+1)/r^2

    on each component, with its own orbital angular momentum.  ``second_derivative_fm2``
    maps sampled radial values to their second derivative at the supplied
    points.  Its boundary convention must be consistent with the quadrature.

    The result is only the matrix element of the scalar one-body p^2 term.
    The CoM normal-exchange and anomalous Wick terms require independently
    established Hermitian p_mu matrices and are evaluated separately by
    :mod:`center_of_mass`.
    """
    r = np.asarray(radius_fm, dtype=np.float64)
    w = np.asarray(radial_weights_fm, dtype=np.float64)
    upper = np.asarray(upper_values, dtype=np.complex128)
    lower = np.asarray(lower_values, dtype=np.complex128)
    d2 = np.asarray(second_derivative_fm2, dtype=np.float64)
    if r.ndim != 1 or r.size < 2 or not np.isfinite(r).all() or np.any(r <= 0.0):
        raise ValueError("radius_fm must be a finite positive one-dimensional grid")
    n_grid = r.size
    if w.shape != (n_grid,) or not np.isfinite(w).all() or np.any(w <= 0.0):
        raise ValueError("radial_weights_fm must be finite positive dr weights")
    if (upper.ndim != 2 or lower.shape != upper.shape
            or upper.shape[0] != n_grid or upper.shape[1] < 1):
        raise ValueError("upper_values and lower_values must have matching (N_grid,M) shape")
    if not np.isfinite(upper).all() or not np.isfinite(lower).all():
        raise ValueError("radial components must be finite")
    if d2.shape != (n_grid, n_grid) or not np.isfinite(d2).all():
        raise ValueError("second_derivative_fm2 must be a finite square grid operator")
    for name, value in (("upper_l", upper_l), ("lower_l", lower_l)):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not np.isfinite(hbar_c_mev_fm) or hbar_c_mev_fm <= 0:
        raise ValueError("hbar_c_mev_fm must be finite and positive")
    if not np.isfinite(hermiticity_tolerance) or hermiticity_tolerance < 0:
        raise ValueError("hermiticity_tolerance must be finite and nonnegative")

    upper_operator = -d2 + np.diag(float(upper_l * (upper_l + 1)) / r**2)
    lower_operator = -d2 + np.diag(float(lower_l * (lower_l + 1)) / r**2)
    # Eq. (C2): sum over G and F with their respective l; no 1/2 factor.
    p2_fm2 = upper.conj().T @ (w[:, None] * (upper_operator @ upper))
    p2_fm2 += lower.conj().T @ (w[:, None] * (lower_operator @ lower))
    p2_mev2 = float(hbar_c_mev_fm) ** 2 * p2_fm2
    hermiticity_abs = float(np.max(np.abs(p2_mev2 - p2_mev2.conj().T)))
    scale = max(1.0, float(np.max(np.abs(p2_mev2))))
    hermiticity_rel = hermiticity_abs / scale
    if hermiticity_rel > hermiticity_tolerance:
        raise ValueError(
            "discrete radial p^2 is not Hermitian under the supplied quadrature "
            f"and boundary operator (relative residual {hermiticity_rel:.3e})"
        )
    return {
        "p2_mev2": p2_mev2,
        "p2_fm2": p2_fm2,
        "upper_l": int(upper_l),
        "lower_l": int(lower_l),
        "hbar_c_mev_fm": float(hbar_c_mev_fm),
        "hermiticity_residual_absolute_mev2": hermiticity_abs,
        "hermiticity_residual_relative": hermiticity_rel,
        "source": "Geng-Long arXiv:2202.00482 Appendix C Eq. (C2)",
        "scope": "one spherical angular channel; direct reduced-radial p^2 only; not ADF p_mu or O17 CoM energy",
    }
