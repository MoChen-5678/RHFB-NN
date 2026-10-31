"""Gauge-independent continuation of an occupied Nambu direction.

This module transports a physical Nambu vector by projecting it onto the
occupied plane of a pure generalized density.  The density is applied in
factorized form, so no primitive-space ``(2D, 2D)`` generalized-density matrix
is assembled.  The result is an occupied-plane continuation tag; it is not a
Hamiltonian quasiparticle eigenmode selector or a ground-state certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .manifold import GeneralizedDensity


@dataclass(frozen=True)
class BlockingTransport:
    """Diagnostics and the transported normalized Nambu tag."""

    mode: torch.Tensor | None
    tag_resolved: bool
    local_overlap: float
    initial_overlap: float | None
    projection_norm: float
    particle_hole_error: float | None
    physical_identity_certified: bool = False
    above_minimum_overlap: bool | None = None


def _require_complex128(value: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if result.dtype != torch.complex128:
        raise ValueError(f"{name} must use complex128")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _particle_hole(vector: torch.Tensor, dimension: int) -> torch.Tensor:
    """Apply the Nambu particle-hole conjugation ``tau_x K``."""

    return torch.cat((vector[dimension:].conj(), vector[:dimension].conj()))


def _active_density_matrix(rho: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    m = rho.shape[0]
    lower = torch.cat(
        (-chi.conj(), torch.eye(m, dtype=rho.dtype, device=rho.device) - rho.conj()),
        dim=1,
    )
    return torch.cat((torch.cat((rho, chi), dim=1), lower), dim=0)


def _validate_inputs(
    q: torch.Tensor,
    gd: GeneralizedDensity,
    previous_mode: torch.Tensor,
    initial_mode: torch.Tensor | None,
    minimum_overlap: float | None,
    *,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    q = _require_complex128(q, "q")
    rho = _require_complex128(gd.rho, "gd.rho")
    chi = _require_complex128(gd.chi, "gd.chi")
    active_matrix = _require_complex128(gd.matrix, "gd.matrix")
    previous_mode = _require_complex128(previous_mode, "previous_mode")
    if initial_mode is not None:
        initial_mode = _require_complex128(initial_mode, "initial_mode")

    if q.ndim != 2 or q.shape[0] < q.shape[1] or q.shape[1] == 0:
        raise ValueError("q must be a nonempty tall or square matrix")
    dimension, active_dimension = q.shape
    if rho.shape != (active_dimension, active_dimension) or chi.shape != rho.shape:
        raise ValueError("gd.rho and gd.chi must match q's active dimension")
    if active_matrix.shape != (2 * active_dimension, 2 * active_dimension):
        raise ValueError("gd.matrix must be the active 2m-by-2m generalized density")
    if previous_mode.shape != (2 * dimension,):
        raise ValueError("previous_mode must be a primitive-space Nambu vector")
    if initial_mode is not None and initial_mode.shape != previous_mode.shape:
        raise ValueError("initial_mode must match previous_mode shape")
    tensors = (rho, chi, active_matrix, previous_mode)
    if any(value.device != q.device for value in tensors) or (
        initial_mode is not None and initial_mode.device != q.device
    ):
        raise ValueError("q, density, and mode tags must be on the same device")
    if minimum_overlap is not None and (
        not math.isfinite(float(minimum_overlap))
        or not 0.0 < float(minimum_overlap) <= 1.0
    ):
        raise ValueError("minimum_overlap must lie in (0, 1]")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    identity = torch.eye(active_dimension, dtype=q.dtype, device=q.device)
    if not torch.allclose(q.mH @ q, identity, atol=tolerance, rtol=0.0):
        raise ValueError("q columns must be orthonormal")
    if not torch.allclose(rho.mH, rho, atol=tolerance, rtol=0.0):
        raise ValueError("gd.rho must be Hermitian")
    if not torch.allclose(chi.mT, -chi, atol=tolerance, rtol=0.0):
        raise ValueError("gd.chi must be antisymmetric")
    reconstructed = _active_density_matrix(rho, chi)
    if not torch.allclose(active_matrix, reconstructed, atol=tolerance, rtol=0.0):
        raise ValueError("gd.matrix is inconsistent with gd.rho and gd.chi")
    if not torch.allclose(active_matrix.mH, active_matrix, atol=tolerance, rtol=0.0):
        raise ValueError("active generalized density must be Hermitian")
    if not torch.allclose(
        active_matrix @ active_matrix,
        active_matrix,
        atol=10.0 * tolerance,
        rtol=0.0,
    ):
        raise ValueError("active generalized density must be a pure projector")

    previous_norm = torch.linalg.vector_norm(previous_mode)
    if abs(float(previous_norm) - 1.0) > 10.0 * tolerance:
        raise ValueError("previous_mode must be normalized")
    if initial_mode is not None:
        initial_norm = torch.linalg.vector_norm(initial_mode)
        if abs(float(initial_norm) - 1.0) > 10.0 * tolerance:
            raise ValueError("initial_mode must be normalized")
    return q, rho, chi, previous_mode, initial_mode


def _r_action(
    q: torch.Tensor,
    rho: torch.Tensor,
    chi: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Apply the lifted generalized density without forming its 2D matrix."""

    dimension = q.shape[0]
    upper, lower = vector[:dimension], vector[dimension:]
    qh_upper = q.mH @ upper
    qt_lower = q.mT @ lower
    active_upper = rho @ qh_upper + chi @ qt_lower
    active_lower_correction = -chi.conj() @ qh_upper - rho.conj() @ qt_lower
    return torch.cat(
        (q @ active_upper, lower + q.conj() @ active_lower_correction)
    )


@torch.no_grad()
def transport_occupied_tag(
    q: torch.Tensor,
    gd: GeneralizedDensity,
    previous_mode: torch.Tensor,
    initial_mode: torch.Tensor | None = None,
    minimum_overlap: float | None = None,
) -> BlockingTransport:
    """Project an accepted Nambu tag into the current occupied plane.

    ``gd`` is the active-space ``GeneralizedDensity`` associated with the
    primitive lift ``q``.  The current lifted density acts on a primitive
    Nambu vector as

    ``[[Q rho Q†, Q chi Qᵀ], [-Q* chi* Q†, I-Q* rho* Qᵀ]]``.

    ``minimum_overlap`` is optional and diagnostic only.  It never rejects a
    physical trial.  ``initial_mode`` is also diagnostic: its overlap is
    returned but is never used as a rejection condition. The returned vector
    is phase-aligned so its overlap with ``previous_mode`` is real and
    nonnegative.

    This operation is intended for no-gradient branch diagnostics. It
    certifies membership in the supplied occupied plane only; it does not
    identify a Hamiltonian eigen-quasiparticle or certify physical state
    identity or ground-state status.
    """

    q, rho, chi, previous_mode, initial_mode = _validate_inputs(
        q, gd, previous_mode, initial_mode, minimum_overlap, tolerance=1.0e-9
    )
    projection = _r_action(q, rho, chi, previous_mode)
    projection_norm_tensor = torch.linalg.vector_norm(projection)
    projection_norm = float(projection_norm_tensor)
    if not math.isfinite(projection_norm):
        raise ValueError("occupied-plane projection norm is non-finite")
    if projection_norm > 1.0 + 1.0e-8:
        raise ValueError("occupied-plane projection norm exceeds one")
    # This is only a floating-point resolvability floor, not a physical
    # overlap condition.  A tag too small to normalize stably is reported as
    # unresolved while the caller remains free to accept the valid density.
    if projection_norm <= 1.0e-14:
        return BlockingTransport(
            mode=None,
            tag_resolved=False,
            local_overlap=float(torch.abs(torch.vdot(previous_mode, projection))),
            initial_overlap=None,
            projection_norm=projection_norm,
            particle_hole_error=None,
            physical_identity_certified=False,
            above_minimum_overlap=(
                None if minimum_overlap is None else projection_norm >= minimum_overlap
            ),
        )

    phase_overlap = torch.vdot(previous_mode, projection)
    phase_overlap_abs = torch.abs(phase_overlap)
    if float(phase_overlap_abs) <= torch.finfo(torch.float64).tiny:
        return BlockingTransport(
            mode=None,
            tag_resolved=False,
            local_overlap=float(phase_overlap_abs),
            initial_overlap=None,
            projection_norm=projection_norm,
            particle_hole_error=None,
            physical_identity_certified=False,
            above_minimum_overlap=(
                None if minimum_overlap is None else float(phase_overlap_abs) >= minimum_overlap
            ),
        )
    mode = projection / projection_norm_tensor
    mode = mode * phase_overlap.conj() / phase_overlap_abs
    local_overlap = float(torch.abs(torch.vdot(previous_mode, mode)))

    dimension = q.shape[0]
    ph_mode = _particle_hole(mode, dimension)
    ph_occupied_norm = float(torch.linalg.vector_norm(_r_action(q, rho, chi, ph_mode)))
    ph_orthogonality = float(torch.abs(torch.vdot(mode, ph_mode)))
    particle_hole_error = max(ph_occupied_norm, ph_orthogonality)
    if not math.isfinite(particle_hole_error) or particle_hole_error > 1.0e-7:
        raise ValueError(
            "transported tag violates particle-hole occupied/complement relation "
            f"(error={particle_hole_error:.6g})"
        )

    initial_overlap = (
        None if initial_mode is None else float(torch.abs(torch.vdot(initial_mode, mode)))
    )
    return BlockingTransport(
        mode=mode,
        tag_resolved=True,
        local_overlap=local_overlap,
        initial_overlap=initial_overlap,
        projection_norm=projection_norm,
        particle_hole_error=particle_hole_error,
        physical_identity_certified=False,
        above_minimum_overlap=(
            None if minimum_overlap is None else local_overlap >= minimum_overlap
        ),
    )
