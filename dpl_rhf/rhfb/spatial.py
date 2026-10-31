"""General multipole Green operators and ADF Cartesian spatial derivatives.

The free-space Yukawa/Coulomb kernel is projected on a declared angular
cutoff. No pointwise 1/|x-y| diagonal is sampled or replaced by an arbitrary
cell radius. Nonlinear density-dependent sources require independent angular
cutoff convergence. Radial integration uses the same metric as the factors.
"""
from __future__ import annotations

import numpy as np
import torch

from dpl_rhf.rhf.dirac import _first_derivative_coefficients
from dpl_rhf.rhf.kernels import SphericalPropagators
from .basis import AngularRadialBasis, _SPH_HARM_Y, _spherical_harmonic


def _ylm(l, m, theta, phi):
    return (_spherical_harmonic(l, m, theta, phi) if _SPH_HARM_Y
            else _spherical_harmonic(m, l, phi, theta))


def _scalar_adf(size: int, step: float, l: int, points: int) -> np.ndarray:
    """Parity-extended physical scalar radial stencil, with zero outer box.

    Odd scalar partial waves use forward ADF, even waves backward ADF.
    Even origin values are inferred from an even polynomial on interior
    samples, rather than setting a generally nonzero scalar f(0) to zero.
    This is distinct from reduced Dirac G/F amplitudes, which vanish there.
    """
    direction = 1 if l % 2 else -1
    offsets = [direction * i for i in range(points)]
    coeff = _first_derivative_coefficients(offsets) / step
    d = np.zeros((size, size))
    x = np.arange(1, points + 1, dtype=float) ** 2
    origin = np.array([np.prod(-np.delete(x, k) / (x[k] - np.delete(x, k)))
                       for k in range(points)])
    for row in range(size):
        for offset, c in zip(offsets, coeff):
            index = row + 1 + offset
            if index == 0:
                if l % 2 == 0:
                    d[row, :points] += c * origin
            elif abs(index) <= size:
                d[row, abs(index) - 1] += c * ((-1)**l if index < 0 else 1)
    return d


class GreenOperator:
    """Projected free-space Green action and its Cartesian derivative blocks.

    ``apply`` consumes physical sources shaped (points, ...), returning the
    potential with volume integration already included. ``kernel_block``
    contains no weights. Gradients act on physical functions; ``adjoint`` is
    the exact weighted discrete adjoint, so integration by parts is explicit.
    No spectral solution of a Dirac or Bogoliubov Hamiltonian is used.
    """
    def __init__(self, basis: AngularRadialBasis, *, lmax: int | None = None,
                 device="cpu"):
        self.basis = basis
        self.lmax = 2 * basis.kappa_max if lmax is None else int(lmax)
        if self.lmax < 0:
            raise ValueError("lmax must be nonnegative")
        if basis.n_theta < self.lmax + 1 or basis.n_phi < 2*self.lmax + 1:
            raise ValueError("angular quadrature cannot resolve requested Green cutoff")
        self.device = torch.device(device)
        self.orders = tuple((l, m) for l in range(self.lmax+1) for m in range(-l, l+1))
        theta, phi = basis.theta, basis.phi
        st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        er = np.stack((st*cp, st*sp, ct), axis=-1)
        et = np.stack((ct*cp, ct*sp, -st), axis=-1)
        ep = np.stack((-sp, cp, np.zeros_like(sp)), axis=-1)
        y, angular_gradient = [], []
        for l, m in self.orders:
            value = _ylm(l, m, theta, phi)
            dt = m*ct/st*value
            if m < l:
                dt += np.sqrt((l-m)*(l+m+1))*np.exp(-1j*phi)*_ylm(l, m+1, theta, phi)
            dp = 1j*m*value/st
            y.append(value)
            angular_gradient.append(et*dt[:, None] + ep*dp[:, None])
        self.y = torch.as_tensor(np.stack(y, axis=1), dtype=torch.complex128, device=device)
        self.angular_gradient = torch.as_tensor(np.stack(angular_gradient, axis=1),
                                               dtype=torch.complex128, device=device)
        self.er = torch.as_tensor(er, dtype=torch.complex128, device=device)
        self.angular_weights = torch.as_tensor(basis.angular_weights, dtype=torch.float64, device=device)
        self.radius = torch.as_tensor(basis.x, dtype=torch.float64, device=device)
        self.radial_weights = basis.spacing_fm*self.radius.square()
        points = 5 if basis.derivative_order in (4, 5) else 7
        if basis.n_radial < points:
            raise ValueError("radial mesh too small for ADF")
        self.derivatives = torch.as_tensor(np.stack([
            _scalar_adf(basis.n_radial, basis.spacing_fm, l, points) for l, _ in self.orders
        ]), dtype=torch.complex128, device=device)
        self.propagators = SphericalPropagators(self.radius, basis.hbar_c_mev_fm)
        self._radial_cache = {}

    def _input(self, source):
        if source.shape[0] != self.basis.point_count or source.device != self.device:
            raise ValueError("source points/device differ from Green operator")
        if source.dtype not in (torch.float64, torch.complex128):
            raise TypeError("source must use float64 or complex128")
        return source.to(torch.complex128)

    def _radial(self, mass_mev):
        mass = float(mass_mev)
        if mass < 0:
            raise ValueError("propagator mass must be nonnegative")
        if mass not in self._radial_cache:
            self._radial_cache[mass] = torch.stack([
                (self.propagators.coulomb(l) if mass == 0 else self.propagators.yukawa(l, mass))
                for l, _ in self.orders
            ]).to(torch.complex128)
        return self._radial_cache[mass]

    def _analyze(self, source):
        f = self._input(source).reshape(self.basis.n_radial, self.basis.n_angular_points, -1)
        return torch.einsum("al,rac->rlc", self.y.conj(), f*self.angular_weights[None, :, None])

    def apply(self, source: torch.Tensor, mass_mev: float) -> torch.Tensor:
        a = self._analyze(source)
        potential = torch.einsum("lij,jlc->ilc", self._radial(mass_mev), a*self.radial_weights[:, None, None])
        result = torch.einsum("al,rlc->rac", self.y, potential).reshape(source.shape)
        return result.real if source.dtype == torch.float64 else result

    def gradient(self, source: torch.Tensor) -> torch.Tensor:
        a = self._analyze(source)
        da = torch.einsum("lij,jlc->ilc", self.derivatives, a)
        radial = torch.einsum("al,rlc,ai->raic", self.y, da, self.er)
        angular = torch.einsum("ali,rlc->raic", self.angular_gradient, a)/self.radius[:, None, None, None]
        result = (radial+angular).reshape((source.shape[0], 3)+source.shape[1:])
        return result.real if source.dtype == torch.float64 else result

    def gradient_adjoint(self, vector: torch.Tensor) -> torch.Tensor:
        """G† in the physical r²dr dΩ metric; divergence is -G†."""
        if vector.ndim < 2 or vector.shape[1] != 3:
            raise ValueError("vector must have shape (points, 3, ...)")
        v = self._input(vector).reshape(self.basis.n_radial, self.basis.n_angular_points, 3, -1)
        v = v*self.angular_weights[None, :, None, None]
        b = torch.einsum("al,ai,raic->rlc", self.y.conj(), self.er.conj(), v)
        b = b*self.radial_weights[:, None, None]
        c = torch.einsum("lij,ilc->jlc", self.derivatives.conj(), b)
        c += torch.einsum("ali,raic->rlc", self.angular_gradient.conj(), v)*(self.radial_weights/self.radius)[:, None, None]
        result = (torch.einsum("al,rlc->rac", self.y, c)/self.radial_weights[:, None, None])
        result = result.reshape((vector.shape[0],)+vector.shape[2:])
        return result.real if vector.dtype == torch.float64 else result

    def divergence(self, vector: torch.Tensor) -> torch.Tensor:
        return -self.gradient_adjoint(vector)

    def kernel_block(self, start_x: int, stop_x: int, start_y: int, stop_y: int,
                     mass_mev: float, *, derivative_x=False, derivative_y=False):
        """Unweighted K, Gx K, Gy K, or Gx Gy K; trailing Cartesian axes.

        Derivatives differentiate the declared discrete projected kernel,
        including its coincident radial entry. Any phenomenological contact
        subtraction is an additional interaction term, not an omitted value.
        """
        if not (0 <= start_x <= stop_x <= self.basis.point_count and
                0 <= start_y <= stop_y <= self.basis.point_count):
            raise ValueError("invalid point block")
        ix = torch.arange(start_x, stop_x, device=self.device)
        iy = torch.arange(start_y, stop_y, device=self.device)
        rx, ax = ix//self.basis.n_angular_points, ix%self.basis.n_angular_points
        ry, ay = iy//self.basis.n_angular_points, iy%self.basis.n_angular_points
        k = self._radial(mass_mev)
        dx = torch.einsum("lij,ljk->lik", self.derivatives, k) if derivative_x else None
        dy = torch.einsum("lij,lkj->lik", k, self.derivatives) if derivative_y else None
        dxy = torch.einsum("lij,lkj->lik", dx, self.derivatives) if derivative_x and derivative_y else None
        yx, yy = self.y[ax], self.y[ay].conj()
        hx = self.angular_gradient[ax]/self.radius[rx, None, None]
        hy = self.angular_gradient[ay].conj()/self.radius[ry, None, None]
        erx, ery = self.er[ax], self.er[ay]
        select = lambda t: t[:, rx[:, None], ry[None, :]].permute(1, 2, 0)
        if derivative_x and derivative_y:
            result = torch.einsum("xyl,xl,yl,xi,yj->xyij", select(dxy), yx, yy, erx, ery)
            result += torch.einsum("xyl,xl,ylj,xi->xyij", select(dx), yx, hy, erx)
            result += torch.einsum("xyl,xli,yl,yj->xyij", select(dy), hx, yy, ery)
            result += torch.einsum("xyl,xli,ylj->xyij", select(k), hx, hy)
        elif derivative_x:
            result = torch.einsum("xyl,xl,yl,xi->xyi", select(dx), yx, yy, erx)
            result += torch.einsum("xyl,xli,yl->xyi", select(k), hx, yy)
        elif derivative_y:
            result = torch.einsum("xyl,xl,yl,yi->xyi", select(dy), yx, yy, ery)
            result += torch.einsum("xyl,xl,yli->xyi", select(k), yx, hy)
        else:
            result = torch.einsum("xyl,xl,yl->xy", select(k), yx, yy)
        return result.real
