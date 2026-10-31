"""Persist a number retraction inside a network's canonical reference frame.

The correction is a state buffer, not a trainable parameter.  It is composed
as a right Thouless transform of the fixed reference quasiparticle frame so
the next ordinary ``model.forward()`` returns the corrected U/V density while
the primitive Q basis, neural parameters, and optimizer parameter identities
remain unchanged.  No orbital refresh or eigensolve is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from .network import GeneralizedDensityNet
from .number_retraction import NumberRetraction, correct_mean_particle_number


@dataclass(frozen=True)
class NetworkNumberTransportReport:
    target: int
    converged: bool
    reason: str
    mean_number_before: float
    mean_number_after: float
    correction_errors: tuple[float, ...]
    accepted_corrections: int
    transport_updated: bool
    q_unchanged: bool
    corrected_state_error: float | None
    transport_before_sha256: str
    transport_after_sha256: str
    retraction: NumberRetraction


@torch.no_grad()
def apply_mean_number_correction(
    model: GeneralizedDensityNet,
    target: int,
    *,
    tolerance: float = 1.0e-10,
    max_corrections: int = 64,
    max_backtracks: int = 24,
    max_generator_norm: float = 0.5,
    verification_tolerance: float = 2.0e-9,
) -> NetworkNumberTransportReport:
    """Retract a network output to target mean number and persist the action.

    If the finite canonical correction converges, its relative frame transform
    is accumulated in ``model.canonical_number_transport``.  The next call to
    the model's normal forward interface then returns that corrected active
    density.  If number response is singular or correction does not converge,
    the model buffer is left unchanged; no pairing seed is inserted.

    The constructor reference is required to admit the network's analytic
    zero/one Slater frame.  Failure to represent the correction raises rather
    than diagonalizing or refreshing a basis.  This function does not change
    trainable tensors or optimizer state.
    """

    if not isinstance(model, GeneralizedDensityNet):
        raise TypeError("model must be a GeneralizedDensityNet")
    if not hasattr(model, "canonical_number_transport"):
        raise RuntimeError("network lacks the versioned canonical number-transport buffer")
    if verification_tolerance <= 0.0 or not torch.isfinite(
        torch.as_tensor(verification_tolerance, dtype=torch.float64)
    ):
        raise ValueError("verification_tolerance must be finite and positive")

    q_before, _density_before, state_before = model.forward_with_bogoliubov_frame()
    if state_before.physical_identity_certified:
        raise RuntimeError("network unexpectedly certified a blocked qp identity")
    transport_before = snapshot_number_transport(model)
    before_hash = _sha256(transport_before)
    mean_before = float(torch.trace(state_before.density.rho).real)
    retraction = correct_mean_particle_number(
        state_before,
        target,
        tolerance=tolerance,
        max_corrections=max_corrections,
        max_backtracks=max_backtracks,
        max_generator_norm=max_generator_norm,
    )
    mean_after = float(torch.trace(retraction.state.density.rho).real)

    if not retraction.converged:
        return NetworkNumberTransportReport(
            target=int(target),
            converged=False,
            reason=retraction.reason,
            mean_number_before=mean_before,
            mean_number_after=mean_after,
            correction_errors=retraction.errors,
            accepted_corrections=retraction.accepted_corrections,
            transport_updated=False,
            q_unchanged=True,
            corrected_state_error=None,
            transport_before_sha256=before_hash,
            transport_after_sha256=before_hash,
            retraction=retraction,
        )

    if retraction.accepted_corrections == 0:
        return NetworkNumberTransportReport(
            target=int(target),
            converged=True,
            reason=retraction.reason,
            mean_number_before=mean_before,
            mean_number_after=mean_after,
            correction_errors=retraction.errors,
            accepted_corrections=0,
            transport_updated=False,
            q_unchanged=True,
            corrected_state_error=0.0,
            transport_before_sha256=before_hash,
            transport_after_sha256=before_hash,
            retraction=retraction,
        )

    # W_new = W_old C_delta, so C_total <- C_total C_delta. The network's
    # effective left transform conjugates C_total by its fixed Slater frame.
    delta = state_before.frame.mH @ retraction.state.frame
    composed = transport_before @ delta
    model._validate_number_transport(composed)
    if composed.device != model.canonical_number_transport.device:
        raise ValueError("number transport and network buffers must share a device")

    model.canonical_number_transport.copy_(composed)
    model._number_transport_active = not torch.equal(
        model.canonical_number_transport,
        torch.eye(
            model.canonical_number_transport.shape[0],
            dtype=model.canonical_number_transport.dtype,
            device=model.canonical_number_transport.device,
        ),
    )
    try:
        q_after, density_after, state_after = model.forward_with_bogoliubov_frame()
        q_unchanged = torch.allclose(
            q_after, q_before, atol=verification_tolerance, rtol=0.0
        )
        state_error = max(
            float(torch.linalg.matrix_norm(state_after.frame - retraction.state.frame)),
            float(torch.linalg.matrix_norm(density_after.matrix - retraction.state.density.matrix)),
        )
        if not q_unchanged or state_error > verification_tolerance:
            raise RuntimeError(
                "network forward did not reproduce the number-retracted state "
                f"(Q unchanged={q_unchanged}, state error={state_error:.3e})"
            )
        if state_after.density.fermion_parity != state_before.density.fermion_parity:
            raise RuntimeError("number transport changed fermion-parity metadata")
        if state_after.physical_identity_certified:
            raise RuntimeError("number transport must not certify qp identity")
    except Exception:
        restore_number_transport(model, transport_before)
        raise

    after_hash = _sha256(model.canonical_number_transport)
    return NetworkNumberTransportReport(
        target=int(target),
        converged=True,
        reason=retraction.reason,
        mean_number_before=mean_before,
        mean_number_after=mean_after,
        correction_errors=retraction.errors,
        accepted_corrections=retraction.accepted_corrections,
        transport_updated=True,
        q_unchanged=True,
        corrected_state_error=state_error,
        transport_before_sha256=before_hash,
        transport_after_sha256=after_hash,
        retraction=retraction,
    )


@torch.no_grad()
def snapshot_number_transport(model: GeneralizedDensityNet) -> torch.Tensor:
    """Snapshot the non-parameter correction state for line-search rollback."""
    _check_model(model)
    return model.canonical_number_transport.detach().clone()


@torch.no_grad()
def restore_number_transport(model: GeneralizedDensityNet, snapshot: torch.Tensor) -> None:
    """Restore a previously captured transport buffer without touching weights."""
    _check_model(model)
    if not isinstance(snapshot, torch.Tensor):
        raise TypeError("number-transport snapshot must be a tensor")
    model._validate_number_transport(snapshot)
    if snapshot.device != model.canonical_number_transport.device:
        raise ValueError("snapshot and network buffer must share a device")
    model.canonical_number_transport.copy_(snapshot)
    model._number_transport_active = not torch.equal(
        model.canonical_number_transport,
        torch.eye(
            model.canonical_number_transport.shape[0],
            dtype=model.canonical_number_transport.dtype,
            device=model.canonical_number_transport.device,
        ),
    )


def _check_model(model: GeneralizedDensityNet) -> None:
    if not isinstance(model, GeneralizedDensityNet):
        raise TypeError("model must be a GeneralizedDensityNet")
    if not hasattr(model, "canonical_number_transport"):
        raise RuntimeError("network lacks the versioned canonical number-transport buffer")


def _sha256(value: torch.Tensor) -> str:
    data = value.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
