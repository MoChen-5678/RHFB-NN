"""Exact multipole convolution for finite-range Gaussian pairing fields.

An isotropic Gaussian has a separable spherical expansion.  For a source with
angular bandwidth ``L``, only its coefficients through ``L`` contribute, even
though the Gaussian kernel itself has an infinite expansion.  This replaces a
quadratic point-pair loop by radial matrix products without changing the
Gaussian range or dropping angular channels.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ive
import torch
from torch.utils.checkpoint import checkpoint

from .basis import AngularRadialBasis, _SPH_HARM_Y, _spherical_harmonic
from .pairing import D1S, GognyD1S, _spin_exchange_indices


class GaussianGreen:
    r"""Apply ``exp(-|r-r'|²/a²)`` with exact multipoles to grid sources.

    The angular quadrature must resolve every requested ``(l,m)``.  Radial
    quadrature and angular weights are the basis's own discrete integration
    measure, so ``apply_gaussian`` is the angularly converged counterpart of
    the direct point-pair Gaussian sum.
    """

    def __init__(self, basis: AngularRadialBasis, *, lmax: int | None = None,
                 device: torch.device | str = "cpu"):
        self.basis = basis
        self.lmax = 2 * basis.kappa_max if lmax is None else int(lmax)
        if self.lmax < 0:
            raise ValueError("lmax must be nonnegative")
        if basis.n_theta < self.lmax + 1 or basis.n_phi < 2 * self.lmax + 1:
            raise ValueError("angular quadrature cannot resolve requested Gaussian multipoles")
        self.device = torch.device(device)
        self.orders = tuple((ell, m) for ell in range(self.lmax + 1)
                            for m in range(-ell, ell + 1))
        self.y = torch.as_tensor(
            np.stack([_spherical_harmonic(ell, m, basis.theta, basis.phi)
                      if _SPH_HARM_Y else
                      _spherical_harmonic(m, ell, basis.phi, basis.theta)
                      for ell, m in self.orders], axis=1),
            dtype=torch.complex128, device=self.device,
        )
        self.angular_weights = torch.as_tensor(
            basis.angular_weights, dtype=torch.float64, device=self.device
        )
        self.radius = torch.as_tensor(basis.x, dtype=torch.float64, device=self.device)
        self.radial_weights = basis.spacing_fm * self.radius.square()
        self._kernel_cache: dict[float, torch.Tensor] = {}

    def _analyze(self, source: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if source.shape[0] != self.basis.point_count or source.device != self.device:
            raise ValueError("source point count/device does not match GaussianGreen")
        if source.dtype not in (torch.float64, torch.complex128):
            raise TypeError("Gaussian source must use float64 or complex128")
        tail = source.shape[1:]
        values = source.to(torch.complex128).reshape(
            self.basis.n_radial, self.basis.n_angular_points, -1
        )
        return torch.einsum(
            "al,rac->rlc", self.y.conj(),
            values * self.angular_weights[None, :, None],
        ), tail

    def _kernel(self, range_fm: float) -> torch.Tensor:
        width = float(range_fm)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Gaussian range must be finite and positive")
        cached = self._kernel_cache.get(width)
        if cached is not None:
            return cached
        radius = self.basis.x[:, None]
        radius_prime = self.basis.x[None, :]
        z = 2.0 * radius * radius_prime / (width * width)
        radial_gaussian = np.exp(-((radius - radius_prime) / width) ** 2)
        blocks = []
        for ell, _ in self.orders:
            # e^(-r²-r'²) i_l(z) = e^(-(r-r')²) ive(l+1/2,z)*sqrt(pi/(2z))
            # keeps the Bessel factor stable for the largest box/range ratio.
            bessel_scaled = ive(ell + 0.5, z) * np.sqrt(np.pi / (2.0 * z))
            blocks.append(4.0 * np.pi * radial_gaussian * bessel_scaled)
        result = torch.as_tensor(np.stack(blocks), dtype=torch.float64, device=self.device)
        self._kernel_cache[width] = result
        return result

    def apply_gaussian(self, source: torch.Tensor, range_fm: float) -> torch.Tensor:
        """Return the Gaussian convolution, including full volume weights."""
        coefficients, tail = self._analyze(source)
        radial = self._kernel(range_fm)
        convolved = torch.einsum(
            "lij,jlc->ilc", radial.to(dtype=coefficients.dtype),
            coefficients * self.radial_weights[:, None, None],
        )
        result = torch.einsum("al,rlc->rac", self.y, convolved).reshape(
            (self.basis.point_count,) + tail
        )
        return result.real if source.dtype == torch.float64 else result


class GaussianPairingAction:
    """Apply the untruncated full-space D1S gap through Gaussian multipoles."""

    def __init__(self, basis: AngularRadialBasis, q: torch.Tensor,
                 chi: torch.Tensor, *, species: str, chunk_size: int = 32,
                 active_chunk_size: int = 4, lmax: int | None = None,
                 checkpoint_blocks: bool = True,
                 force: GognyD1S = D1S):
        if species not in {"n", "p"} or force.strength != 1.0:
            raise ValueError("separate nn/pp unscaled D1S is required")
        if q.dtype != torch.complex128 or chi.dtype != torch.complex128:
            raise TypeError("D1S factors must use complex128")
        if q.device != chi.device:
            raise ValueError("D1S factors must be on the same device")
        if q.shape[0] != basis.dimension or chi.shape != (q.shape[1], q.shape[1]):
            raise ValueError("D1S factors do not match basis")
        if not torch.allclose(chi, -chi.mT, atol=1.0e-10, rtol=0.0):
            raise ValueError("anomalous density must be antisymmetric")
        if chunk_size < 1 or active_chunk_size < 1:
            raise ValueError("hole-vector and active-index chunk sizes must be positive")
        effective_lmax = 2 * basis.kappa_max if lmax is None else int(lmax)
        if effective_lmax < 2 * basis.kappa_max:
            raise ValueError("full-space D1S pairing requires lmax>=2*kappa_max")
        self.basis, self.q, self.chi = basis, q, chi
        self.species, self.chunk_size = species, int(chunk_size)
        self.active_chunk_size = int(active_chunk_size)
        self.checkpoint_blocks, self.force = checkpoint_blocks, force
        self.green = GaussianGreen(basis, lmax=effective_lmax, device=q.device)

    def _apply_columns(self, q, chi, vector):
        basis = self.basis
        q_sampled = basis._sample_range(q, 0, basis.point_count)
        hole = basis._sample_range(vector.conj(), 0, basis.point_count).conj()
        spin_a, spin_b = _spin_exchange_indices(q.device)
        # Factor the x/y pair tensor before convolution.  The direct source is
        # Q_j(y) v_j(y), which has angular bandwidth 2K; Q_i(x) is applied
        # after the radial multipole action.  This is the identity that avoids
        # materializing a point-by-point kernel or an active-pair block.
        source_direct = torch.einsum("ab,pjb,pjk->pak", chi, q_sampled, hole)
        physical = torch.zeros(
            (basis.point_count, 4, vector.shape[1]),
            dtype=torch.complex128, device=q.device,
        )
        for range_fm, w, b, h, m in zip(
            self.force.ranges_fm, self.force.w_mev, self.force.b_mev,
            self.force.h_mev, self.force.m_mev,
        ):
            direct_strength = float(w - h)
            exchange_strength = float(b - m)
            if direct_strength:
                field = self.green.apply_gaussian(source_direct, range_fm)
                physical = physical + direct_strength * torch.einsum(
                    "pia,pak->pik", q_sampled, field
                )
        # P_sigma exchanges spin at fixed upper/lower component.  Contract the
        # anomalous active index before convolving and tile that index so peak
        # storage is independent of the full active-space pair count.
        for active_start in range(0, q.shape[1], self.active_chunk_size):
            active_stop = min(active_start + self.active_chunk_size, q.shape[1])
            weighted_q = torch.einsum(
                "ab,pjb->paj", chi[active_start:active_stop], q_sampled
            )
            source_exchange = weighted_q[:, :, spin_b, None] * hole[:, None, None, :, :]
            q_gamma = q_sampled[:, spin_a, active_start:active_stop]
            for range_fm, b, m in zip(self.force.ranges_fm,
                                      self.force.b_mev, self.force.m_mev):
                exchange_strength = float(b - m)
                if not exchange_strength:
                    continue
                field = self.green.apply_gaussian(source_exchange, range_fm)
                physical = physical + exchange_strength * torch.einsum(
                    "pija,paijk->pik", q_gamma, field
                )
        return basis.project_sample_range(physical, 0, basis.point_count)

    def apply(self, hole_vector: torch.Tensor) -> torch.Tensor:
        """Apply Delta to one or more arbitrary primitive hole vectors."""
        if (hole_vector.dtype != torch.complex128 or hole_vector.ndim not in {1, 2}
                or hole_vector.shape[0] != self.basis.dimension):
            raise ValueError("hole vector must match primitive dimension and use complex128")
        if hole_vector.device != self.q.device or hole_vector.ndim not in {1, 2}:
            raise ValueError("hole vectors must share the factor device and have shape (D,) or (D,K)")
        single = hole_vector.ndim == 1
        values = hole_vector[:, None] if single else hole_vector
        chunks = []
        for start in range(0, values.shape[1], self.chunk_size):
            vector = values[:, start:start + self.chunk_size]
            if self.checkpoint_blocks and torch.is_grad_enabled():
                piece = checkpoint(self._apply_columns, self.q, self.chi, vector,
                                   use_reentrant=False)
            else:
                piece = self._apply_columns(self.q, self.chi, vector)
            chunks.append(piece)
        result = torch.cat(chunks, dim=1)
        return result[:, 0] if single else result

    def energy_and_active_gap(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return D1S energy and ``Q† Delta Q*`` from the same full-space action."""
        gap = self.q.mH @ self.apply(self.q.conj())
        gap = 0.5 * (gap - gap.T)
        energy = 0.5 * torch.sum(self.chi.conj() * gap).real
        return energy, gap
