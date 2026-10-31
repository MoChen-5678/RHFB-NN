"""Finite-dimensional matrix-sign projector with its field response.

This is an isolated mathematical prototype, not a selected nuclear no-sea
prescription.  It evaluates Q_+(h-W)=(I+sign(h-W))/2 using rational shifted
linear solves.  The caller must supply a valid spectral enclosure away from
W.  Autograd differentiates through the solves, so the derivative of Q with
respect to a field-dependent h is retained.  The result is only as accurate
as the supplied finite quadrature and enclosure.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .no_sea import RationalSignOperator, SignDiagnostics


@dataclass(frozen=True)
class SignProjectorResult:
    projector: torch.Tensor
    diagnostics: SignDiagnostics
    shift: float
    field_response_in_autograd_graph: bool
    physical_no_sea_prescription_certified: bool = False


def positive_energy_projector(
    h: torch.Tensor,
    *,
    shift: float,
    spectral_gap: float,
    spectral_bound: float,
    bound_certificate: str,
    quadrature_order: int = 65,
    log_step: float = 0.4,
    tolerance: float = 1.0e-8,
    solve_tolerance: float = 1.0e-11,
) -> SignProjectorResult:
    """Construct a finite-space positive spectral projector from shifted solves.

    No spectral diagonalization, mode selection, or guessed energy shift is
    performed.  ``shift`` and the enclosure are explicit inputs.  The bound
    certificate is caller-supplied evidence and is not independently checked
    by this small-space routine.
    """
    h = torch.as_tensor(h)
    if h.dtype != torch.complex128 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise TypeError("h must be a square complex128 matrix")
    identity = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    sign_operator = RationalSignOperator(
        h,
        h.shape[0],
        shift=shift,
        spectral_gap=spectral_gap,
        spectral_bound=spectral_bound,
        bound_certificate=bound_certificate,
        quadrature_order=quadrature_order,
        log_step=log_step,
        tolerance=tolerance,
        solve_tolerance=solve_tolerance,
    )
    sign = sign_operator.apply_sign(identity)
    projector = 0.5 * (identity + sign)
    return SignProjectorResult(
        projector=projector,
        diagnostics=sign_operator.diagnostics(),
        shift=float(shift),
        field_response_in_autograd_graph=bool(projector.requires_grad),
    )


def projector_source_contraction(
    h: torch.Tensor,
    source: torch.Tensor,
    **projector_options,
) -> tuple[torch.Tensor, SignProjectorResult]:
    """Return Re Tr(source† Q_+(h-W)) without detaching field dependence.

    If ``source`` is the external projector-response source from
    ``no_sea_variation``, differentiating the returned scalar with respect to
    the field parameters applies the matrix-sign response to that source.
    Any additional chain rule through the self-consistent EDF fields must
    still be supplied by the caller; this function does not solve that closure.
    """
    result = positive_energy_projector(h, **projector_options)
    source = torch.as_tensor(source, dtype=result.projector.dtype,
                             device=result.projector.device)
    if source.shape != result.projector.shape:
        raise ValueError("source must have the same shape as the projector")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("source contains non-finite values")
    value = torch.real(torch.sum(source.conj() * result.projector))
    return value, result
