"""Candidate no-sea HFB construction in a positive-energy particle space.

This module implements an algebraically pure active-space HFB density and its
embedding in a larger Nambu space.  It does not assert that a moving
positive-energy projector is the unique nuclear no-sea prescription, nor
that a stationary point is a min-max solution.  For a field-dependent
projector, callers must solve the projector self-consistency condition and
include its response in the variational derivative.

Training helpers below do not diagonalize the Dirac or BdG operator.  The
explicit spectral derivative helper is marked reference-only for tiny-space
validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .manifold import GeneralizedDensity, generalized_density_matrix


@dataclass(frozen=True)
class ProjectorDiagnostics:
    hermiticity_error: float
    idempotency_error: float
    commutator_error: float


def positive_projector_from_sign(sign: torch.Tensor) -> torch.Tensor:
    """Form the positive spectral projector from an approximate matrix sign.

    The sign may come from ``RationalSignOperator``.  The returned matrix is
    only as accurate as that sign approximation; it is not silently
    re-diagonalized or repaired.  Use ``positive_frame_from_projector`` to
    create a differentiable orthonormal active chart.
    """

    sign = torch.as_tensor(sign)
    if sign.dtype != torch.complex128 or sign.ndim != 2 or sign.shape[0] != sign.shape[1]:
        raise TypeError("sign must be a square complex128 matrix")
    identity = torch.eye(sign.shape[0], dtype=sign.dtype, device=sign.device)
    return 0.5 * (identity + sign)


def positive_frame_from_projector(
    projector: torch.Tensor,
    reference_frame: torch.Tensor,
    *,
    rank_tolerance: float = 1.0e-10,
    projector_tolerance: float | None = 1.0e-6,
) -> torch.Tensor:
    """Build a differentiable active frame from ``Q @ reference_frame``.

    ``reference_frame`` is a fixed, full-rank chart seed with one column per
    active nucleon coordinate.  QR produces an orthonormal basis without
    eigensolving.  The discrete rank is held fixed, so this chart is local:
    it fails clearly if the projector loses overlap with the seed frame.
    """

    q = torch.as_tensor(projector)
    refs = torch.as_tensor(reference_frame, device=q.device)
    if q.dtype != torch.complex128 or q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise TypeError("projector must be a square complex128 matrix")
    if refs.dtype != torch.complex128 or refs.ndim != 2 or refs.shape[0] != q.shape[0]:
        raise TypeError("reference_frame must be complex128 with matching row dimension")
    if refs.shape[1] == 0 or refs.shape[1] > refs.shape[0]:
        raise ValueError("reference_frame must have between 1 and D columns")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("projector contains non-finite values")
    if projector_tolerance is not None:
        herm_error = torch.linalg.matrix_norm(q - q.mH)
        idem_error = torch.linalg.matrix_norm(q @ q - q)
        if float(herm_error.detach()) > projector_tolerance:
            raise ValueError(f"projector is not Hermitian: {float(herm_error.detach()):.3e}")
        if float(idem_error.detach()) > projector_tolerance:
            raise ValueError(f"projector is not idempotent: {float(idem_error.detach()):.3e}")
    projected = q @ refs
    frame, triangular = torch.linalg.qr(projected, mode="reduced")
    diagonal = torch.abs(torch.diagonal(triangular))
    if bool(torch.any(diagonal.detach() <= rank_tolerance)):
        raise RuntimeError("positive-space chart lost rank against its reference frame")
    return frame


def positive_frame_from_sign(
    sign: torch.Tensor,
    reference_frame: torch.Tensor,
    *,
    rank_tolerance: float = 1.0e-10,
    sign_hermiticity_tolerance: float = 1.0e-8,
) -> torch.Tensor:
    """Chart helper for a matrix-sign positive-space approximation."""

    sign = torch.as_tensor(sign)
    if sign.dtype != torch.complex128 or sign.ndim != 2 or sign.shape[0] != sign.shape[1]:
        raise TypeError("sign must be a square complex128 matrix")
    error = torch.linalg.matrix_norm(sign - sign.mH)
    if float(error.detach()) > sign_hermiticity_tolerance:
        raise ValueError(f"matrix sign is not Hermitian: {float(error.detach()):.3e}")
    projector = positive_projector_from_sign(sign)
    return positive_frame_from_projector(
        projector, reference_frame, rank_tolerance=rank_tolerance,
        projector_tolerance=None,
    )


def embed_active_density(
    frame: torch.Tensor,
    rho_active: torch.Tensor,
    chi_active: torch.Tensor,
    *,
    fermion_parity: int,
) -> GeneralizedDensity:
    """Embed a pure active HFB density; all orthogonal particles are empty.

    With ``B=frame``, the full blocks are ``rho=B rho_a B†`` and
    ``chi=B chi_a Bᵀ``.  The full lower-right Nambu block consequently has
    identity on the inactive hole complement.  The full ``R`` is exactly
    idempotent whenever the supplied active ``R_a`` is idempotent.
    """

    b = torch.as_tensor(frame)
    rho_a = torch.as_tensor(rho_active, device=b.device)
    chi_a = torch.as_tensor(chi_active, device=b.device)
    if b.dtype != torch.complex128 or b.ndim != 2 or b.shape[0] < b.shape[1]:
        raise TypeError("frame must be a tall or square complex128 matrix")
    m = b.shape[1]
    if rho_a.dtype != torch.complex128 or rho_a.shape != (m, m):
        raise TypeError("rho_active must be complex128 and match the frame rank")
    if chi_a.dtype != torch.complex128 or chi_a.shape != (m, m):
        raise TypeError("chi_active must be complex128 and match rho_active")
    if fermion_parity not in (-1, 1):
        raise ValueError("fermion_parity must be -1 (odd) or +1 (even)")
    identity = torch.eye(m, dtype=b.dtype, device=b.device)
    if not torch.allclose(b.mH @ b, identity, atol=1.0e-9, rtol=0.0):
        raise ValueError("frame columns must be orthonormal")
    if not torch.allclose(rho_a.mH, rho_a, atol=1.0e-9, rtol=0.0):
        raise ValueError("rho_active must be Hermitian")
    if not torch.allclose(chi_a.mT, -chi_a, atol=1.0e-9, rtol=0.0):
        raise ValueError("chi_active must be antisymmetric")
    rho = b @ rho_a @ b.mH
    chi = b @ chi_a @ b.mT
    matrix = generalized_density_matrix(rho, chi)
    return GeneralizedDensity(rho, chi, matrix, fermion_parity)


def active_nambu_isometry(frame: torch.Tensor) -> torch.Tensor:
    """Map active particle/hole coordinates into the full Nambu space."""

    b = torch.as_tensor(frame)
    if b.dtype != torch.complex128 or b.ndim != 2:
        raise TypeError("frame must be a rank-2 complex128 matrix")
    zero = torch.zeros_like(b)
    return torch.cat(
        (torch.cat((b, zero), dim=1), torch.cat((zero, b.conj()), dim=1)),
        dim=0,
    )


def active_nambu_hamiltonian(h_nambu: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Restrict a full BdG Hamiltonian to the active particle/hole chart."""

    h = torch.as_tensor(h_nambu)
    w = active_nambu_isometry(frame)
    if h.dtype != torch.complex128 or h.shape != (w.shape[0], w.shape[0]):
        raise ValueError("h_nambu dimension must be twice the primitive dimension")
    return w.mH @ h @ w


def active_stationarity_residual(
    h_nambu: torch.Tensor,
    density: GeneralizedDensity,
    frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active-space commutator RMS and the restricted commutator.

    This is the fixed-projector Euler-Lagrange residual.  It is distinct from
    the full commutator, which can remain nonzero because active-space
    variations cannot rotate into the excluded particle complement.
    """

    h_a = active_nambu_hamiltonian(h_nambu, frame)
    w = active_nambu_isometry(frame)
    r_a = w.mH @ density.matrix @ w
    commutator = h_a @ r_a - r_a @ h_a
    rms = torch.sqrt(torch.mean(torch.abs(commutator).square()))
    return rms, commutator


def density_purity_residual(density: GeneralizedDensity) -> torch.Tensor:
    """Frobenius error in ``R²=R`` for a full generalized density."""

    r = density.matrix
    return torch.linalg.matrix_norm(r @ r - r)


def projector_diagnostics(h: torch.Tensor, q: torch.Tensor) -> ProjectorDiagnostics:
    """No-eigensolver algebraic residuals for a candidate projector."""

    h = torch.as_tensor(h)
    q = torch.as_tensor(q, device=h.device)
    if h.dtype != torch.complex128 or q.dtype != torch.complex128 or h.shape != q.shape:
        raise ValueError("h and Q must be same-shaped complex128 square matrices")
    return ProjectorDiagnostics(
        hermiticity_error=float(torch.linalg.matrix_norm(q - q.mH).detach()),
        idempotency_error=float(torch.linalg.matrix_norm(q @ q - q).detach()),
        commutator_error=float(torch.linalg.matrix_norm(h @ q - q @ h).detach()),
    )


def projector_frechet_derivative_reference(
    h: torch.Tensor,
    perturbation: torch.Tensor,
    *,
    cutoff: float = 0.0,
    gap_tolerance: float = 1.0e-12,
) -> torch.Tensor:
    """Exact tiny-matrix derivative of ``1[h>cutoff]`` (reference-only).

    This helper uses ``eigh`` solely as an independent validation oracle.  It
    implements the spectral divided-difference formula and must not be used
    on a candidate/training path.
    """

    h = torch.as_tensor(h)
    dh = torch.as_tensor(perturbation, device=h.device)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("h must be square complex128")
    if dh.dtype != torch.complex128 or dh.shape != h.shape:
        raise TypeError("perturbation must be complex128 and match h")
    values, vectors = torch.linalg.eigh(h)
    positive = values > cutoff
    if bool(torch.any(torch.abs(values - cutoff) <= gap_tolerance)):
        raise ValueError("spectral projector derivative is undefined at the cutoff")
    basis_perturbation = vectors.mH @ dh @ vectors
    denominator = values[:, None] - values[None, :]
    crossing = positive[:, None] != positive[None, :]
    safe_denominator = torch.where(crossing, denominator, torch.ones_like(denominator))
    divided_difference = torch.where(
        crossing,
        (positive[:, None].to(torch.float64) - positive[None, :].to(torch.float64))
        / safe_denominator,
        torch.zeros_like(safe_denominator),
    )
    return vectors @ (divided_difference * basis_perturbation) @ vectors.mH
