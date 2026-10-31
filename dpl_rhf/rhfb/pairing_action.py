"""Full primitive-space D1S field action, including outside active Q.

The pairing energy/gap alone supplies only Q† Delta Q*. This operator applies
Delta to arbitrary hole vectors, permitting full-space stationarity checks.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .basis import AngularRadialBasis
from .pairing import D1S, GognyD1S, _spin_exchange_indices


class D1SPairingAction:
    def __init__(self, basis: AngularRadialBasis, q: torch.Tensor, chi: torch.Tensor,
                 *, species: str, chunk_size: int = 32, checkpoint_blocks=True,
                 force: GognyD1S = D1S):
        if species not in {'n', 'p'} or force.strength != 1.:
            raise ValueError('separate nn/pp unscaled D1S is required')
        if q.dtype != torch.complex128 or chi.dtype != torch.complex128:
            raise TypeError('D1S factors must use complex128')
        if q.shape[0] != basis.dimension or chi.shape != (q.shape[1], q.shape[1]):
            raise ValueError('D1S factors do not match basis')
        if not torch.allclose(chi, -chi.mT, atol=1e-10, rtol=0):
            raise ValueError('anomalous density must be antisymmetric')
        if chunk_size < 1:
            raise ValueError('chunk size must be positive')
        self.basis, self.q, self.chi = basis, q, chi
        self.species, self.chunk_size = species, chunk_size
        self.checkpoint_blocks, self.force = checkpoint_blocks, force

    def apply(self, hole_vector: torch.Tensor) -> torch.Tensor:
        if hole_vector.shape[0] != self.basis.dimension or hole_vector.dtype != torch.complex128:
            raise ValueError('hole vector must match primitive dimension and complex128')
        if hole_vector.device != self.q.device:
            raise ValueError('hole vector/factors devices differ')
        single = hole_vector.ndim == 1
        v = hole_vector[:, None] if single else hole_vector
        if v.ndim != 2:
            raise ValueError('hole vectors must have shape (D,) or (D,columns)')
        basis, force = self.basis, self.force
        spin_a, spin_b = _spin_exchange_indices(self.q.device)
        radii = torch.as_tensor(force.ranges_fm, device=self.q.device, dtype=torch.float64)
        c0 = torch.as_tensor(force.w_mev, device=self.q.device, dtype=torch.float64)-torch.as_tensor(force.h_mev, device=self.q.device, dtype=torch.float64)
        cs = torch.as_tensor(force.b_mev, device=self.q.device, dtype=torch.float64)-torch.as_tensor(force.m_mev, device=self.q.device, dtype=torch.float64)
        positions = torch.as_tensor(basis.positions, device=self.q.device, dtype=torch.float64)
        weights = torch.as_tensor(basis.weights, device=self.q.device, dtype=torch.float64)

        def first_block(q, chi, vector, p0, p1):
            qp = basis._sample_range(q, p0, p1)
            physical_result = qp.new_zeros((p1-p0, 4, vector.shape[1]))
            for y0 in range(0, basis.point_count, self.chunk_size):
                y1 = min(y0+self.chunk_size, basis.point_count)
                qq = basis._sample_range(q, y0, y1)
                # Hole coordinate amplitudes are S* v, not S v.
                vy = basis._sample_range(vector.conj(), y0, y1).conj()
                amplitude = torch.einsum('pia,ab,qjb->pqij', qp, chi, qq)
                distance2 = (positions[p0:p1, None]-positions[None, y0:y1]).square().sum(-1)
                gaussian = torch.exp(-distance2[:, :, None]/radii.square())
                direct = (gaussian*c0).sum(-1)
                spin = (gaussian*cs).sum(-1)
                field = direct[:, :, None, None]*amplitude + spin[:, :, None, None]*amplitude[:, :, spin_a, spin_b]
                physical_result = physical_result + torch.einsum('pqij,qjk->pik', field, vy*weights[y0:y1, None, None])
            return basis.project_sample_range(physical_result, p0, p1)

        result = v.new_zeros(v.shape)
        for p0 in range(0, basis.point_count, self.chunk_size):
            p1 = min(p0+self.chunk_size, basis.point_count)
            if self.checkpoint_blocks and torch.is_grad_enabled():
                partial = checkpoint(lambda q, chi, vector, lo=p0, hi=p1: first_block(q, chi, vector, lo, hi),
                                     self.q, self.chi, v, use_reentrant=False)
            else:
                partial = first_block(self.q, self.chi, v, p0, p1)
            result = result+partial
        return result[:, 0] if single else result
