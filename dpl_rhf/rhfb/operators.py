from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class DensityFactors:
    """Apply a pure generalized density without allocating a (2D,2D) matrix."""

    q: torch.Tensor
    rho_active: torch.Tensor
    chi_active: torch.Tensor

    def __post_init__(self):
        if self.q.ndim != 2 or self.q.dtype != torch.complex128:
            raise ValueError("Q must be a complex128 (D,M) factor")
        m = self.q.shape[1]
        if m == 0 or self.q.shape[0] < m:
            raise ValueError("Q must have nonempty independent columns")
        if self.rho_active.shape != (m, m) or self.chi_active.shape != (m, m):
            raise ValueError("active densities must both have shape (M,M)")
        for value in (self.q, self.rho_active, self.chi_active):
            if value.dtype != torch.complex128 or value.device != self.q.device:
                raise ValueError("density factors must share complex128 dtype and device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("density factors must be finite")
        identity = torch.eye(m, dtype=self.q.dtype, device=self.q.device)
        if not torch.allclose(self.q.mH @ self.q, identity, atol=1e-10, rtol=0):
            raise ValueError("Q columns must be orthonormal")
        if not torch.allclose(self.rho_active, self.rho_active.mH, atol=1e-10, rtol=0):
            raise ValueError("rho_active must be Hermitian")
        if not torch.allclose(self.chi_active, -self.chi_active.mT, atol=1e-10, rtol=0):
            raise ValueError("chi_active must be antisymmetric")
        # Purity identities avoid constructing even an active Nambu matrix.
        if (not torch.allclose(self.rho_active @ self.rho_active
                               - self.chi_active @ self.chi_active.conj(),
                               self.rho_active, atol=1e-10, rtol=0)
                or not torch.allclose(self.rho_active @ self.chi_active,
                                      self.chi_active @ self.rho_active.conj(),
                                      atol=1e-10, rtol=0)):
            raise ValueError("active generalized density must be pure")

    @property
    def dimension(self) -> int:
        return self.q.shape[0]

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.shape[0] != 2 * self.dimension:
            raise ValueError("Nambu vector dimension does not match factors")
        x, y = vector[:self.dimension], vector[self.dimension:]
        q = self.q
        px, py = q.mH @ x, q.mT @ y
        top = q @ (self.rho_active @ px + self.chi_active @ py)
        bottom = y - q.conj() @ (self.chi_active.conj() @ px + self.rho_active.conj() @ py)
        return torch.cat((top, bottom), dim=0)

    def particle_number(self) -> torch.Tensor:
        return torch.trace(self.rho_active @ (self.q.mH @ self.q)).real


@dataclass
class NambuAction:
    dimension: int
    h_apply: Callable[[torch.Tensor], torch.Tensor]
    delta_apply: Callable[[torch.Tensor], torch.Tensor]
    chemical_potential: torch.Tensor | float = 0.0

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.shape[0] != 2 * self.dimension:
            raise ValueError("Nambu vector dimension does not match Hamiltonian")
        x, y = vector[:self.dimension], vector[self.dimension:]
        top = self.h_apply(x) - self.chemical_potential * x + self.delta_apply(y)
        bottom = (-self.delta_apply(x.conj()).conj() - self.h_apply(y.conj()).conj()
                  + self.chemical_potential * y)
        return torch.cat((top, bottom), dim=0)


def commutator_action(hamiltonian: NambuAction, density: DensityFactors,
                      vectors: torch.Tensor) -> torch.Tensor:
    return hamiltonian.apply(density.apply(vectors)) - density.apply(hamiltonian.apply(vectors))


def stationarity_metrics(hamiltonian: NambuAction, density: DensityFactors,
                         *, chunk_size: int = 32) -> dict[str, torch.Tensor]:
    """Exact streamed Frobenius residual, including the omitted particle space.

    Checking C only on diag(Q,Q*) misses pairing between two omitted orbitals.
    Iterate all primitive unit vectors in bounded chunks to include that case.
    This expensive certification is separate from a cheaper per-step estimator.
    """
    if hamiltonian.dimension != density.dimension or chunk_size < 1:
        raise ValueError("inconsistent dimensions or nonpositive chunk size")
    return commutator_metrics(
        lambda vectors: commutator_action(hamiltonian, density, vectors), density,
        chunk_size=chunk_size,
    )


def commutator_metrics(apply_commutator, density: DensityFactors,
                       *, chunk_size: int = 32) -> dict[str, torch.Tensor]:
    """Stream an exact commutator action; distinguish element and column RMS."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    q = density.q
    d, m = q.shape
    z = q.new_zeros((2 * d, 2 * m))
    z[:d, :m] = q
    z[d:, m:] = q.conj()
    action = apply_commutator(z)
    active = z.mH @ action
    complement = action - z @ active
    full_square = q.real.new_zeros(())
    for start in range(0, 2 * d, chunk_size):
        stop = min(2 * d, start + chunk_size)
        vectors = q.new_zeros((2 * d, stop - start))
        vectors[torch.arange(start, stop, device=q.device),
                torch.arange(stop - start, device=q.device)] = 1.0
        full_square = full_square + apply_commutator(vectors).abs().square().sum()
    return {
        # Preserve the old column-RMS key for checkpoint/result compatibility.
        "full_rms_mev": torch.sqrt(full_square / (2 * d)),
        "full_column_rms_mev": torch.sqrt(full_square / (2 * d)),
        "full_element_rms_mev": torch.sqrt(full_square) / (2 * d),
        "full_frobenius_mev": torch.sqrt(full_square),
        "active_frobenius_mev": torch.linalg.vector_norm(active),
        "complement_frobenius_mev": torch.linalg.vector_norm(complement),
    }
