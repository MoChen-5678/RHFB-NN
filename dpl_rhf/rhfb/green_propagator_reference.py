"""Independent spherical Gaussian reference for bare Yukawa/Coulomb energy.

The reference starts from the three-dimensional Fourier transform of
exp(-mu*r)/(4*pi*r), not from the production spherical kernel.  It is limited
to an l=0, constant-coupling source and says nothing about derivative/contact
vertices.
"""
from __future__ import annotations

from math import exp, pi, sqrt

from scipy.integrate import quad
from scipy.special import erfcx


def gaussian_yukawa_integral(nucleon_number: float, width_fm: float,
                             inverse_range_fm: float) -> float:
    """Return integral rho(x) D_mu(x-y) rho(y) d^3x d^3y in fm^-1 units.

    The density is N exp(-r^2/b^2)/(pi^(3/2) b^3), and
    D_mu(r)=exp(-mu*r)/(4*pi*r).  For mu=0 this returns the Coulomb limit.
    """
    if width_fm <= 0 or inverse_range_fm < 0:
        raise ValueError("width must be positive and inverse range nonnegative")
    a = width_fm**2 / 2.0
    mu = float(inverse_range_fm)
    bracket = (sqrt(pi) / (2.0 * sqrt(a)) if mu == 0.0 else
               sqrt(pi) / (2.0 * sqrt(a)) -
               0.5 * pi * mu * float(erfcx(mu * sqrt(a))))
    return float(nucleon_number)**2 * bracket / (2.0 * pi**2)


def gaussian_yukawa_fourier_quad(nucleon_number: float, width_fm: float,
                                 inverse_range_fm: float) -> float:
    """Independent 1-D Fourier quadrature of the same Gaussian integral."""
    if width_fm <= 0 or inverse_range_fm < 0:
        raise ValueError("width must be positive and inverse range nonnegative")
    a = width_fm**2 / 2.0
    mu = float(inverse_range_fm)
    radial, _ = quad(lambda q: exp(-a*q*q) if mu == 0.0 else
                     q*q*exp(-a*q*q)/(q*q + mu*mu),
                     0.0, float("inf"), epsabs=2e-13, epsrel=2e-13,
                     limit=250)
    return float(nucleon_number)**2 * radial / (2.0 * pi**2)


def gaussian_yukawa_box_integral(nucleon_number: float, width_fm: float,
                                 inverse_range_fm: float,
                                 radius_fm: float) -> float:
    """Independent exact-continuum integral with both Gaussian sources cut at R.

    Angular integration is analytic; the remaining nested radial quadrature
    uses the l=0 Yukawa Green function.  This estimates finite-box source-tail
    truncation separately from production mesh quadrature.
    """
    if width_fm <= 0 or inverse_range_fm < 0 or radius_fm <= 0:
        raise ValueError("invalid Gaussian, inverse-range, or box parameter")
    b = float(width_fm)
    mu = float(inverse_range_fm)
    radius = float(radius_fm)
    rho0 = float(nucleon_number) / (pi**1.5 * b**3)

    def density(r):
        return rho0 * exp(-(r / b)**2)

    def potential(r):
        if r == 0.0:
            # The finite limit follows by taking r -> 0 in sinh(mu*r_<)/(mu*r).
            if mu == 0.0:
                return quad(lambda rp: rp*density(rp), 0.0, radius,
                            epsabs=2e-13, epsrel=2e-13, limit=250)[0]
            return quad(lambda rp: rp*density(rp)*exp(-mu*rp), 0.0, radius,
                        epsabs=2e-13, epsrel=2e-13, limit=250)[0]
        if mu == 0.0:
            inner = quad(lambda rp: rp*rp*density(rp), 0.0, r,
                         epsabs=2e-13, epsrel=2e-13, limit=250)[0] / r
            outer = quad(lambda rp: rp*density(rp), r, radius,
                         epsabs=2e-13, epsrel=2e-13, limit=250)[0]
            return inner + outer
        left = quad(lambda rp: rp*density(rp)*exp(mu*(rp-r))*
                    (1.0-exp(-2.0*mu*rp)), 0.0, r,
                    epsabs=2e-13, epsrel=2e-13, limit=250)[0]
        right = quad(lambda rp: rp*density(rp)*exp(mu*(r-rp))*
                     (1.0-exp(-2.0*mu*r)), r, radius,
                     epsabs=2e-13, epsrel=2e-13, limit=250)[0]
        return (left + right) / (2.0*mu*r)

    value, _ = quad(lambda r: 4.0*pi*r*r*density(r)*potential(r),
                    0.0, radius, epsabs=2e-11, epsrel=2e-11, limit=250)
    return value
