"""Exact normal-field action cache on a declared finite-dimensional domain.

This is a diagnostic/cache adapter around :class:`AutogradNormalAction`, not a
replacement for its EDF derivative.  Given orthonormal input columns U and the
oracle values hU, it applies the same linear action to vectors in span(U).
Inputs outside that span are rejected.  Passing a full primitive identity gives
a full dense field; passing saved Q and selected probes gives only that domain.

No density inversion, eigensolve, or approximation to the particle-hole energy
is used.  The caller owns the physical validity of the supplied oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch


def evaluate_two_species_normal_columns(basis, states: dict, energy, vectors: torch.Tensor,
                                        *, vector_chunk: int = 8,
                                        create_graph: bool = False) -> dict[str, torch.Tensor]:
    """Evaluate ``h_n V`` and ``h_p V`` from one complete PH energy per chunk.

    Both species receive an independent Hermitian tangent with the same probe
    columns. This lets a fixed-density diagnostic obtain two normal fields
    without evaluating the full EDF twice per primitive-column block. It uses
    the same tangent construction as ``AutogradNormalAction`` and requires a
    complete, differentiable scalar particle-hole energy callback.
    """
    if set(states) != {"n", "p"} or vector_chunk < 1:
        raise ValueError("two-species normal columns require n/p states and positive chunk")
    if (vectors.ndim != 2 or vectors.shape[0] != basis.dimension
            or vectors.dtype != torch.complex128):
        raise ValueError("vectors must be a complex128 primitive-column matrix")

    def augmented_state(state, v, probe):
        m, k = state.q.shape[1], v.shape[1]
        q = torch.cat((state.q, v, probe), dim=1)
        zero = state.rho_active.new_zeros
        eye = torch.eye(k, dtype=state.q.dtype, device=state.q.device)
        row0 = torch.cat((state.rho_active, zero((m, k)), zero((m, k))), dim=1)
        row1 = torch.cat((zero((k, m)), zero((k, k)), eye), dim=1)
        row2 = torch.cat((zero((k, m)), eye, zero((k, k))), dim=1)
        rho = torch.cat((row0, row1, row2), dim=0)
        chi = zero((m + 2*k, m + 2*k))
        return SimpleNamespace(q=q, rho_active=rho, chi_active=chi)

    chunks = {"n": [], "p": []}
    with torch.enable_grad():
        for start in range(0, vectors.shape[1], vector_chunk):
            v = vectors[:, start:start + vector_chunk]
            probes = {s: torch.zeros_like(v, requires_grad=True) for s in ("n", "p")}
            tangent_states = {s: augmented_state(states[s], v, probes[s]) for s in ("n", "p")}
            value = energy(basis, tangent_states)
            if (not isinstance(value, torch.Tensor) or value.numel() != 1
                    or value.dtype != torch.float64 or not value.requires_grad):
                raise TypeError("normal EDF must return a differentiable float64 scalar")
            gradients = torch.autograd.grad(value, tuple(probes[s] for s in ("n", "p")),
                                             create_graph=create_graph)
            for species, gradient in zip(("n", "p"), gradients):
                chunks[species].append(.5 * gradient)
    return {s: torch.cat(chunks[s], dim=1) for s in ("n", "p")}


@dataclass
class CachedNormalAction:
    """Linear action cache with an explicit orthonormal input-span contract."""

    input_basis: torch.Tensor
    output_basis: torch.Tensor
    domain_tolerance: float = 2.0e-10

    def __post_init__(self):
        u, hu = self.input_basis, self.output_basis
        if u.ndim != 2 or hu.ndim != 2 or u.shape != hu.shape:
            raise ValueError("cached input/output columns must have equal matrix shapes")
        if u.dtype != torch.complex128 or hu.dtype != torch.complex128 or u.device != hu.device:
            raise ValueError("cached columns must share complex128 dtype and device")
        if not torch.isfinite(u).all() or not torch.isfinite(hu).all():
            raise ValueError("cached columns must be finite")
        identity = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
        if not torch.allclose(u.mH @ u, identity, atol=2e-10, rtol=0):
            raise ValueError("input_basis columns must be orthonormal")
        if self.domain_tolerance <= 0:
            raise ValueError("domain_tolerance must be positive")

    @classmethod
    def from_oracle(cls, oracle, input_basis: torch.Tensor, *,
                    vector_chunk: int | None = None, domain_tolerance: float = 2e-10):
        """Evaluate hU once, then cache it for repeated actions on span(U).

        ``input_basis`` must already be orthonormal; its columns are kept in the
        supplied order.  ``vector_chunk`` may override an oracle's chunk size
        when that oracle exposes the ``vector_chunk`` attribute.
        """
        if input_basis.ndim != 2 or input_basis.dtype != torch.complex128:
            raise ValueError("input_basis must be a complex128 matrix")
        if vector_chunk is not None:
            if vector_chunk < 1:
                raise ValueError("vector_chunk must be positive")
            if not hasattr(oracle, "vector_chunk"):
                raise TypeError("oracle does not expose a configurable vector_chunk")
            oracle.vector_chunk = vector_chunk
        output = oracle.apply(input_basis).detach()
        return cls(input_basis.detach(), output, domain_tolerance)

    @classmethod
    def from_values(cls, input_basis: torch.Tensor, output_basis: torch.Tensor, *,
                    domain_tolerance: float = 2e-10):
        """Build from already measured hU columns (e.g. immutable run output)."""
        return cls(input_basis.detach(), output_basis.detach(), domain_tolerance)

    @property
    def dimension(self) -> int:
        return self.input_basis.shape[0]

    @property
    def cached_columns(self) -> int:
        return self.input_basis.shape[1]

    def apply(self, vector: torch.Tensor):
        if (vector.ndim not in (1, 2) or vector.shape[0] != self.dimension
                or vector.dtype != torch.complex128 or vector.device != self.input_basis.device):
            raise ValueError("input vectors must match cached dimension, dtype and device")
        single = vector.ndim == 1
        values = vector[:, None] if single else vector
        coordinates = self.input_basis.mH @ values
        remainder = values - self.input_basis @ coordinates
        error = float(torch.linalg.vector_norm(remainder).detach())
        scale = max(float(torch.linalg.vector_norm(values).detach()), 1.0)
        if error > self.domain_tolerance * scale:
            raise ValueError(f"input lies outside cached span (relative residual {error/scale:.3e})")
        result = self.output_basis @ coordinates
        return result[:, 0] if single else result
