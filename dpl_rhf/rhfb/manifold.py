"""Complex Bogoliubov manifold primitives for the blocked-RHFB prototype.

The convention is

    R = [[rho, chi], [-chi.conj(), I - rho.conj()]],

with ``rho.H == rho`` and ``chi.T == -chi``.  All routines here are finite
dimensional and contain no eigensolver calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeneralizedDensity:
    rho: torch.Tensor
    chi: torch.Tensor
    matrix: torch.Tensor
    fermion_parity: int


def _as_complex128(value: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if result.dtype != torch.complex128:
        raise TypeError(f"{name} must use complex128, got {result.dtype}")
    return result


def canonical_generator(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    tolerance: float = 1.0e-11,
) -> torch.Tensor:
    """Build an anti-Hermitian canonical generator from complex coordinates.

    ``A†=-A`` and ``Bᵀ=-B`` are checked instead of silently repaired so an
    optimizer cannot hide a malformed tangent behind a projection.
    """

    a = _as_complex128(a, "A")
    b = _as_complex128(b, "B")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be square")
    if b.shape != a.shape:
        raise ValueError("B must have the same square shape as A")
    if not torch.allclose(a.mH, -a, atol=tolerance, rtol=0.0):
        raise ValueError("A must be anti-Hermitian")
    if not torch.allclose(b.mT, -b, atol=tolerance, rtol=0.0):
        raise ValueError("B must be antisymmetric under transpose")
    return torch.cat(
        [torch.cat([a, b], dim=1), torch.cat([b.conj(), a.conj()], dim=1)],
        dim=0,
    )


def slater_reference(
    occupations: torch.Tensor,
    *,
    blocked_index: int | None = None,
    blocking_operation: str = "toggle",
) -> GeneralizedDensity:
    """Create an even/odd pure Slater reference by an explicit blocking swap.

    The supplied occupations define the even reference.  ``blocked_index``
    toggles one particle orbital, producing an odd reference for either
    particle addition (0->1) or hole blocking (1->0). The subsequent
    connected Bogoliubov transform preserves this parity.
    """

    occupations = torch.as_tensor(occupations)
    if occupations.dtype != torch.float64:
        raise TypeError("reference occupations must be float64 zero/one values")
    if occupations.ndim != 1 or occupations.numel() == 0:
        raise ValueError("occupations must be a nonempty vector")
    if not bool(torch.all((occupations == 0.0) | (occupations == 1.0))):
        raise ValueError("a pure Slater reference requires zero/one occupations")
    n = occupations.numel()
    if blocked_index is not None:
        if not 0 <= int(blocked_index) < n:
            raise IndexError("blocked_index lies outside the one-particle space")
        if blocking_operation not in {"toggle", "particle", "hole"}:
            raise ValueError("blocking_operation must be toggle, particle, or hole")
        occupied = float(occupations[int(blocked_index)])
        if blocking_operation == "particle" and occupied != 0.0:
            raise ValueError("particle blocking requires an initially empty orbital")
        if blocking_operation == "hole" and occupied != 1.0:
            raise ValueError("hole blocking requires an initially occupied orbital")
        occupations = occupations.clone()
        occupations[int(blocked_index)] = 1.0 - occupied
    rho = torch.diag(occupations).to(torch.complex128)
    chi = torch.zeros((n, n), dtype=torch.complex128, device=rho.device)
    parity = -1 if int(torch.sum(occupations).item()) % 2 else 1
    return GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), parity)


def generalized_density_matrix(rho: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """Assemble R and validate the normal/anomalous conjugation convention."""

    rho = _as_complex128(rho, "rho")
    chi = _as_complex128(chi, "chi")
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or chi.shape != rho.shape:
        raise ValueError("rho and chi must be square matrices of identical shape")
    return torch.cat(
        [
            torch.cat([rho, chi], dim=1),
            torch.cat([-chi.conj(), torch.eye(rho.shape[0], dtype=rho.dtype, device=rho.device) - rho.conj()], dim=1),
        ],
        dim=0,
    )


def transform_reference(
    reference: GeneralizedDensity,
    a: torch.Tensor,
    b: torch.Tensor,
) -> GeneralizedDensity:
    """Apply ``exp(K)`` to a pure even or blocked odd reference."""

    generator = canonical_generator(a, b)
    n = reference.rho.shape[0]
    if generator.shape != (2 * n, 2 * n):
        raise ValueError("generator dimension does not match reference")
    transform = torch.matrix_exp(generator)
    matrix = transform @ reference.matrix @ transform.mH
    rho = matrix[:n, :n]
    chi = matrix[:n, n:]
    return GeneralizedDensity(rho, chi, matrix, reference.fermion_parity)


def lift_active_density(
    q: torch.Tensor,
    rho_active: torch.Tensor,
    chi_active: torch.Tensor,
    *,
    tolerance: float = 1.0e-10,
    fermion_parity: int = 0,
) -> GeneralizedDensity:
    """Lift local normal/anomalous densities through a fixed primitive Q.

    The active anomalous tensor uses ``Q chi_active Qᵀ`` (never ``Q†``).
    QR is not done here because silently changing the coordinate chart would
    also change the supplied active density matrices.
    """

    q = _as_complex128(q, "Q")
    rho_active = _as_complex128(rho_active, "rho_active")
    chi_active = _as_complex128(chi_active, "chi_active")
    if q.ndim != 2 or rho_active.shape != (q.shape[1], q.shape[1]) or chi_active.shape != rho_active.shape:
        raise ValueError("active densities must match Q's column dimension")
    identity = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
    if not torch.allclose(q.mH @ q, identity, atol=tolerance, rtol=0.0):
        raise ValueError("Q columns must be orthonormal")
    if not torch.allclose(rho_active.mH, rho_active, atol=tolerance, rtol=0.0):
        raise ValueError("rho_active must be Hermitian")
    if not torch.allclose(chi_active.mT, -chi_active, atol=tolerance, rtol=0.0):
        raise ValueError("chi_active must be antisymmetric")
    rho = q @ rho_active @ q.mH
    chi = q @ chi_active @ q.mT
    if fermion_parity not in {-1, 0, 1}:
        raise ValueError("fermion_parity must be -1 (odd), 1 (even), or 0 (unspecified)")
    return GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), fermion_parity)


def blocking_overlap(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Return phase-insensitive overlap used to detect blocked-mode switching."""

    reference = _as_complex128(reference, "reference blocked mode")
    current = _as_complex128(current, "current blocked mode")
    if reference.ndim != 1 or current.shape != reference.shape:
        raise ValueError("blocked modes must be vectors of identical shape")
    return torch.abs(torch.vdot(reference, current))


def require_blocking_continuity(
    reference: torch.Tensor,
    current: torch.Tensor,
    *,
    minimum_overlap: float = 0.5,
) -> torch.Tensor:
    overlap = blocking_overlap(reference, current)
    if float(overlap.detach()) < minimum_overlap:
        raise RuntimeError(
            f"blocked-mode overlap {float(overlap.detach()):.6g} fell below "
            f"{minimum_overlap:.6g}; refusing a silent blocking switch"
        )
    return overlap


def orthonormalize_factors(q: torch.Tensor, *, tolerance: float = 1.0e-12) -> torch.Tensor:
    """Complex QR lift for normal/anomalous coordinate factors."""

    q = _as_complex128(q, "Q")
    if q.ndim != 2 or q.shape[0] < q.shape[1]:
        raise ValueError("Q must be a tall or square matrix")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("Q contains non-finite values")
    result, r = torch.linalg.qr(q, mode="reduced")
    if bool(torch.any(torch.abs(torch.diagonal(r)) <= tolerance)):
        raise RuntimeError("Q lost rank during QR")
    return result
