"""Full-space normal Hamiltonian action from a scalar normal-density EDF.

An independent Hermitian tangent is inserted in the normal density. The
derivative gives hV on arbitrary primitive vectors, including vectors outside
the active factor space. No inverse of the occupation matrix, diagonalization,
or dense primitive Hamiltonian is required. Density-dependent vertex gradients
are retained, so the action includes all rearrangement from the supplied EDF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .basis import AngularRadialBasis
from .operators import DensityFactors


@dataclass
class _DensityTangentFactors:
    """Auxiliary EDF coordinates, deliberately outside the pure-state chart.

    Q rho Q† equals the physical density at zero probe. Away from zero it
    represents its arbitrary Hermitian tangent. This object must only enter
    the normal-density energy; it is not an HFB vacuum/checkpoint or a relaxed
    validation of DensityFactors.
    """
    q: torch.Tensor
    rho_active: torch.Tensor
    chi_active: torch.Tensor

    @property
    def dimension(self):
        return self.q.shape[0]

    def particle_number(self):
        return torch.trace(self.rho_active @ (self.q.mH @ self.q)).real


class AutogradNormalAction:
    """Matrix-free h=dE_ph/dρ, with the full EDF derivative.

    ``energy`` must evaluate a scalar *complete particle-hole* functional,
    including kinetic energy, on normal-factor coordinates. The caller owns
    its physical certification. An incomplete component energy must never be
    supplied as the PKA1 normal Hamiltonian. Pairing is a separate action.
    """
    def __init__(self, basis: AngularRadialBasis, states: dict[str, DensityFactors],
                 energy: Callable, *, species: str, vector_chunk: int = 8,
                 create_graph: bool = True):
        if set(states) != {'n', 'p'} or species not in states:
            raise ValueError('normal action requires independent neutron/proton states')
        if vector_chunk < 1:
            raise ValueError('vector chunk must be positive')
        self.basis, self.states, self.energy = basis, states, energy
        self.species, self.vector_chunk = species, vector_chunk
        self.create_graph = create_graph

    def apply(self, vector: torch.Tensor):
        state = self.states[self.species]
        if (vector.ndim not in (1, 2) or vector.shape[0] != self.basis.dimension
                or vector.dtype != torch.complex128 or vector.device != state.q.device):
            raise ValueError('normal action vectors must match basis, complex128 and device')
        single = vector.ndim == 1
        vectors = vector[:, None] if single else vector
        chunks = []
        # Field evaluation must remain differentiable even inside a diagnostics
        # no_grad context; create_graph controls whether field gradients persist.
        with torch.enable_grad():
            for start in range(0, vectors.shape[1], self.vector_chunk):
                v = vectors[:, start:start+self.vector_chunk]
                probe = torch.zeros_like(v, requires_grad=True)
                m, k = state.q.shape[1], v.shape[1]
                q = torch.cat((state.q, v, probe), dim=1)
                zero = state.rho_active.new_zeros
                eye = torch.eye(k, dtype=state.q.dtype, device=state.q.device)
                row0 = torch.cat((state.rho_active, zero((m,k)), zero((m,k))), dim=1)
                row1 = torch.cat((zero((k,m)), zero((k,k)), eye), dim=1)
                row2 = torch.cat((zero((k,m)), eye, zero((k,k))), dim=1)
                rho = torch.cat((row0, row1, row2), dim=0)
                # The anomalous density is not varied in a normal EDF derivative.
                chi = zero((m+2*k, m+2*k))
                augmented = dict(self.states)
                augmented[self.species] = _DensityTangentFactors(q, rho, chi)
                energy = self.energy(self.basis, augmented)
                if (not isinstance(energy, torch.Tensor) or energy.numel() != 1
                        or energy.dtype != torch.float64 or not energy.requires_grad):
                    raise TypeError('normal EDF must return a differentiable real float64 scalar')
                gradient, = torch.autograd.grad(energy, probe,
                                               create_graph=self.create_graph)
                # δE=2 Re Tr[(hV)† δprobe] for δρ=probe V†+V probe†.
                chunks.append(.5*gradient)
        result = torch.cat(chunks, dim=1)
        return result[:, 0] if single else result
