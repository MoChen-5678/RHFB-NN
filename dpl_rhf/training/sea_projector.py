"""Positive-energy (no-sea) projector of a Dirac block without diagonalization.

P_+ = (I + sign(h - W)) / 2 is formed with the Newton-Schulz iteration
X <- X (3I - X^2) / 2, which uses only matrix products (Higham, Functions of
Matrices, SIAM 2008, Sec. 5.3).  W lies in the gap between the Fermi sea and
the Dirac sea (W = -M with the rest mass removed from h).  The projector is
built from a detached Hamiltonian: the Dirac sea is defined by the current
self-consistent field, while its response is not varied (no-sea approximation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PositiveProjector:
    projector: torch.Tensor
    iterations: int
    sign_residual: float
    sea_dimension: int


def _spectral_norm_bound(a: torch.Tensor, steps: int = 30) -> torch.Tensor:
    vector = torch.ones(a.shape[0], dtype=a.dtype, device=a.device)
    vector = vector / torch.linalg.vector_norm(vector)
    value = torch.zeros((), dtype=a.dtype, device=a.device)
    for _ in range(steps):
        image = a @ vector
        value = torch.linalg.vector_norm(image)
        vector = image / value.clamp_min(1e-300)
    # Power iteration underestimates; the Frobenius norm is a hard upper bound.
    return torch.minimum(1.05 * value, torch.linalg.matrix_norm(a))


@torch.no_grad()
def positive_energy_projector(
    hamiltonian: torch.Tensor,
    shift: float,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-12,
) -> PositiveProjector:
    h = hamiltonian.detach()
    h = 0.5 * (h + h.T)
    identity = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    a = h - float(shift) * identity
    x = a / _spectral_norm_bound(a)
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        square = x @ x
        x = 0.5 * x @ (3.0 * identity - square)
        residual = float(torch.linalg.matrix_norm(square - identity) / h.shape[0] ** 0.5)
        if residual < tolerance:
            break
    else:
        raise RuntimeError(
            f"Newton-Schulz sign iteration did not converge (residual {residual:.3e}); "
            "the shift may lie on the spectrum"
        )
    x = 0.5 * (x + x.T)
    projector = 0.5 * (identity + x)
    sea_dimension = int(round(float(torch.trace(identity - projector))))
    return PositiveProjector(projector, iteration, residual, sea_dimension)
