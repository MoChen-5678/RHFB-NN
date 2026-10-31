"""Symmetric P1 radial product integration for the bare spherical Yukawa kernel.

This prototype integrates the l=0 radial energy bilinear form on [0,R] using
piecewise-linear finite-element hats and cellwise Gauss quadrature. It does not
use the production propagator. The matrix is assembled symmetrically so the
energy and its nodal-source derivative share exactly one bilinear form.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def p1_yukawa_product_matrix(radius_fm: float, spacing_fm: float,
                              inverse_range_fm: float,
                              quadrature_order: int = 8) -> np.ndarray:
    """Return symmetric A for I = c.T @ A @ c, c_i = r_i*rho(r_i).

    For spherical density rho, the exact continuum l=0 form is
    I = 4*pi int dr dr' (r rho(r))(r' rho(r'))
        sinh(mu*r_<) exp(-mu*r_>) / mu,
    with the mu=0 factor replaced by r_<.  A is the P1 Galerkin product
    integral of this same bare kernel over the finite interval.
    """
    radius = float(radius_fm)
    h = float(spacing_fm)
    mu = float(inverse_range_fm)
    if radius <= 0 or h <= 0 or mu < 0 or quadrature_order < 2:
        raise ValueError("invalid box, spacing, inverse range, or quadrature order")
    intervals = int(round(radius / h))
    if not np.isclose(intervals*h, radius, rtol=0.0, atol=1e-11):
        raise ValueError("spacing must divide the box radius")
    nodes, weights = leggauss(quadrature_order)
    unit = 0.5*(nodes + 1.0)
    unit_weights = 0.5*weights
    matrix = np.zeros((intervals+1, intervals+1), dtype=np.float64)

    def kernel_ordered(rlo, rhi):
        if mu == 0.0:
            return rlo
        return np.sinh(mu*rlo)*np.exp(-mu*rhi)/mu

    for cell_left in range(intervals):
        a = cell_left*h
        r_left = a + h*unit
        phi_left = np.stack((1.0-unit, unit), axis=1)
        w_left = h*unit_weights
        ids_left = np.array((cell_left, cell_left+1))

        # Off-diagonal cell pairs have a fixed radial ordering. Fill the
        # transposed block explicitly to enforce exact matrix symmetry.
        for cell_right in range(cell_left+1, intervals):
            b = cell_right*h
            r_right = b + h*unit
            phi_right = np.stack((1.0-unit, unit), axis=1)
            w_right = h*unit_weights
            k = kernel_ordered(r_left[:, None], r_right[None, :])
            block = 4.0*np.pi*np.einsum(
                "q,s,qi,sj,qs->ij", w_left, w_right,
                phi_left, phi_right, k, optimize=True)
            ids_right = np.array((cell_right, cell_right+1))
            matrix[np.ix_(ids_left, ids_right)] += block
            matrix[np.ix_(ids_right, ids_left)] += block.T

        # Split the diagonal cell square into two triangles. A Duffy map
        # integrates r<r' with Jacobian h^2(1-u); add its exchanged image.
        u = unit[:, None]
        v = unit[None, :]
        rp_unit = u + (1.0-u)*v
        r = a + h*u
        rp = a + h*rp_unit
        phi_r = np.stack((np.broadcast_to(1.0-u, rp_unit.shape),
                          np.broadcast_to(u, rp_unit.shape)), axis=-1)
        phi_rp = np.stack((1.0-rp_unit, rp_unit), axis=-1)
        jacobian_weights = (h*h*(1.0-u) * unit_weights[:, None]
                            * unit_weights[None, :])
        k = kernel_ordered(r, rp)
        block = 4.0*np.pi*np.einsum(
            "qv,qv,qva,qvb->ab", jacobian_weights, k,
            phi_r, phi_rp, optimize=True)
        block += block.T
        matrix[np.ix_(ids_left, ids_left)] += block

    return matrix


def gaussian_reduced_radial_source(radius_fm: float, spacing_fm: float,
                                   nucleon_number: float,
                                   width_fm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return radial nodes and unrenormalized P1 samples c_i=r_i*rho(r_i)."""
    radius, h, b = float(radius_fm), float(spacing_fm), float(width_fm)
    if radius <= 0 or h <= 0 or b <= 0:
        raise ValueError("box, spacing, and width must be positive")
    intervals = int(round(radius/h))
    if not np.isclose(intervals*h, radius, rtol=0.0, atol=1e-11):
        raise ValueError("spacing must divide the box radius")
    r = np.linspace(0.0, radius, intervals+1)
    rho = float(nucleon_number)/(np.pi**1.5*b**3)*np.exp(-(r/b)**2)
    return r, r*rho


def p1_source_norm(radius_fm: float, spacing_fm: float,
                   reduced_source: np.ndarray) -> float:
    """Integrate 4*pi*r*f(r) for the P1 interpolant f=r*rho, without rescaling."""
    radius, h = float(radius_fm), float(spacing_fm)
    intervals = int(round(radius/h))
    if reduced_source.shape != (intervals+1,):
        raise ValueError("source must have one P1 coefficient per radial node")
    # Four-point Gauss exactly integrates the cubic product r*f_h(r) on each
    # element (indeed a two-point rule is enough); no source renormalization.
    nodes, weights = leggauss(4)
    unit, unit_weights = 0.5*(nodes+1.0), 0.5*weights
    norm = 0.0
    for cell in range(intervals):
        t = unit
        r = (cell+t)*h
        f = (1.0-t)*reduced_source[cell] + t*reduced_source[cell+1]
        norm += h*np.sum(unit_weights*r*f)
    return float(4.0*np.pi*norm)
