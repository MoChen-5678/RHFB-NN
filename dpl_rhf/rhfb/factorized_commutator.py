"""Exact full-space [H,R] using hQ once and untruncated pairing actions.

Normal-density and anomalous-density factors have finite active rank.  The
normal field therefore needs to act only on Q to form their commutators.
The remaining Delta term is retained on every primitive vector: pairing
wholly outside Q is never discarded.
"""
from __future__ import annotations

import torch

from .operators import DensityFactors


class FactorizedCommutator:
    def __init__(self, density: DensityFactors, hq: torch.Tensor, delta_apply,
                 *, chemical_potential=0., delta_q_star=None):
        if hq.shape != density.q.shape or hq.dtype != density.q.dtype:
            raise ValueError("hQ must have Q's shape and complex128 dtype")
        if hq.device != density.q.device:
            raise ValueError("hQ and Q must share device")
        self.density = density
        self.hq = hq
        self.delta_apply = delta_apply
        self.delta_q_star = (delta_apply(density.q.conj())
                             if delta_q_star is None else delta_q_star)
        if self.delta_q_star.shape != hq.shape:
            raise ValueError("Delta Q* must have Q's shape")
        self.chemical_potential = chemical_potential
        self.dimension = density.dimension

    def _normal(self, vector):
        q = self.density.q
        rho = self.density.rho_active
        chi = self.density.chi_active
        b = self.hq - self.chemical_potential * q
        a = self.delta_q_star
        return (b @ (rho @ (q.mH @ vector))
                - q @ (rho @ (b.mH @ vector))
                - a @ (chi.conj() @ (q.mH @ vector))
                - q @ (chi @ (a.mH @ vector)))

    def _anomalous(self, vector, delta_vector):
        q = self.density.q
        rho = self.density.rho_active
        chi = self.density.chi_active
        b = self.hq - self.chemical_potential * q
        a = self.delta_q_star
        return (b @ (chi @ (q.mT @ vector))
                + q @ (chi @ (b.mT @ vector)) + delta_vector
                - a @ (rho.conj() @ (q.mT @ vector))
                + q @ (rho @ (a.mT @ vector)))

    def apply(self, vector):
        if vector.ndim not in (1, 2) or vector.shape[0] != 2*self.dimension:
            raise ValueError("commutator vector must have primitive Nambu dimensions")
        single = vector.ndim == 1
        values = vector[:, None] if single else vector
        x, y = values[:self.dimension], values[self.dimension:]
        # Two conjugate pairing actions share one batch; no dense Delta or
        # dense primitive Nambu matrix is constructed.
        delta = self.delta_apply(torch.cat((y, x.conj()), dim=1))
        dy, dx_star = delta[:, :x.shape[1]], delta[:, x.shape[1]:]
        top = self._normal(x) + self._anomalous(y, dy)
        bottom = (self._anomalous(x.conj(), dx_star).conj()
                  + self._normal(y.conj()).conj())
        result = torch.cat((top, bottom), dim=0)
        return result[:, 0] if single else result

    def frobenius_squared(self, *, chunk_size: int = 32):
        """Differentiable exact squared residual in the full primitive space.

        With Hermitian h and skew Delta, [H,R] has blocks
        [[A,B],[B*,A*]]. Its norm squared is 2*(||A||²+||B||²).
        Stream primitive unit columns, retaining Delta between two omitted
        orbitals. No dense primitive Nambu matrix or spectral operation is
        needed. No field or density is detached: callers supplying fields
        that depend on the density retain those derivatives.

        This is an algebraic stationarity measure, not a no-sea branch test
        or a proof that a stationary solution is the desired nuclear state.
        The Hermitian/skew field conventions remain the caller's contract.
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        q = self.density.q
        normal_square = q.real.new_zeros(())
        anomalous_square = q.real.new_zeros(())
        for start in range(0, self.dimension, chunk_size):
            stop = min(start + chunk_size, self.dimension)
            units = q.new_zeros((self.dimension, stop-start))
            units[torch.arange(start, stop, device=q.device),
                  torch.arange(stop-start, device=q.device)] = 1.
            normal_square = normal_square + self._normal(units).abs().square().sum()
            anomalous_square = anomalous_square + self._anomalous(
                units, self.delta_apply(units)).abs().square().sum()
        return {
            "normal_squared_mev2": normal_square,
            "anomalous_squared_mev2": anomalous_square,
            "full_squared_mev2": 2.*(normal_square + anomalous_square),
        }
