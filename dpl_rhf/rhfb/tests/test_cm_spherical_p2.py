"""Algebraic checks for source-derived spherical reduced-radial p^2."""
from __future__ import annotations

import numpy as np
import pytest

from dpl_rhf.rhfb.cm_spherical_p2 import spherical_reduced_radial_p2


def _gauss_grid(order: int, radius: float):
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * radius * (x + 1.0), 0.5 * radius * w


def _legendre_second_derivative_matrix(radius: float, x: np.ndarray) -> np.ndarray:
    """Differentiate the polynomial interpolant on the mapped Gauss nodes."""
    v = np.polynomial.legendre.legvander(x, x.size - 1)
    v2 = np.empty_like(v)
    for degree in range(x.size):
        coefficients = np.zeros(x.size)
        coefficients[degree] = 1.0
        d2 = np.polynomial.legendre.legder(coefficients, m=2)
        v2[:, degree] = np.polynomial.legendre.legval(x, d2)
    # x = 2r/R - 1, hence d^2/dr^2 = (2/R)^2 d^2/dx^2.
    return (2.0 / radius) ** 2 * v2 @ np.linalg.inv(v)


def test_direct_p2_matches_independent_radial_integration_for_both_dirac_components():
    radius = 12.0
    x, _wx = np.polynomial.legendre.leggauss(14)
    r, w = _gauss_grid(14, radius)
    d2 = _legendre_second_derivative_matrix(radius, x)
    # These reduced radial polynomials vanish at both box endpoints; their
    # second derivatives are also evaluated analytically below, independently
    # of the differentiation matrix used by the implementation.
    upper = np.column_stack((r * (radius-r), r**2 * (radius-r)))
    lower = np.column_stack((r**2 * (radius-r), r**3 * (radius-r)))
    result = spherical_reduced_radial_p2(
        r, w, upper, lower, d2, upper_l=0, lower_l=1,
        hbar_c_mev_fm=1.0, hermiticity_tolerance=1.0e-8,
    )

    upper_d2 = np.column_stack((
        np.full_like(r, -2.0), 2.0 * radius - 6.0*r,
    ))
    lower_d2 = np.column_stack((
        2.0*radius - 6.0*r, 6.0*radius*r - 12.0*r**2,
    ))
    upper_action = -upper_d2
    lower_action = -lower_d2 + (2.0 / r**2)[:, None] * lower
    expected = upper.conj().T @ (w[:, None] * upper_action)
    expected += lower.conj().T @ (w[:, None] * lower_action)
    assert np.allclose(result["p2_fm2"], expected, rtol=2e-10, atol=2e-9)
    assert np.allclose(result["p2_mev2"], expected, rtol=2e-10, atol=2e-9)
    assert np.allclose(result["p2_mev2"], result["p2_mev2"].conj().T,
                       rtol=0, atol=2e-9)


def test_p2_rejects_quadrature_boundary_pairing_that_breaks_hermiticity():
    radius = 4.0
    r, w = _gauss_grid(6, radius)
    x, _ = np.polynomial.legendre.leggauss(6)
    d2 = _legendre_second_derivative_matrix(radius, x)
    # Use a complete set of independent sampled radial columns so a weighted
    # nonsymmetric operator cannot collapse to a real 1x1 projected scalar.
    # This is deliberately a malformed boundary/operator fixture, not a
    # physical radial basis.
    radial_columns = np.eye(r.size, dtype=np.float64)
    bad_d2 = d2 + np.triu(np.ones_like(d2), 1)
    weighted_skew = np.diag(w) @ bad_d2 - bad_d2.T @ np.diag(w)
    assert np.linalg.norm(weighted_skew) > 1.0e-8
    with pytest.raises(ValueError, match="not Hermitian"):
        spherical_reduced_radial_p2(
            r, w, radial_columns, np.zeros_like(radial_columns), bad_d2,
            upper_l=0, lower_l=1, hermiticity_tolerance=1e-10,
        )
