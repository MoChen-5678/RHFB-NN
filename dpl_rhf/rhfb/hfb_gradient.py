"""U/V quasiparticle gradients and number-tangent projection.

All formulas use ``W=[[U,V*],[V,U*]]``, ``rho=V*V.T`` and
``chi=V*U.T``.  The H20 and Thouless derivative formulas follow Robledo &
Bertsch, Phys. Rev. C 84, 014312 (2011), Eqs. (6), (8), (11)--(18).
They are finite-space variational identities; they do not certify a no-sea
branch, a stationary physical nucleus, or a global energy minimum.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .bogoliubov_blocking import BogoliubovState, state_from_frame
from .manifold import canonical_generator


def _complex128_finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.complex128:
        raise TypeError(f"{name} must be a complex128 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class NumberProjectedH20:
    """H20 after removing its number-constraint tangent component."""

    chemical_potential: torch.Tensor
    identifiable: bool
    h20: torch.Tensor
    n20: torch.Tensor
    constrained_h20: torch.Tensor
    gram: torch.Tensor
    numerator: torch.Tensor


def h20_matrix(
    u: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    delta: torch.Tensor,
    *,
    tolerance: float = 1.0e-9,
) -> torch.Tensor:
    r"""Return the skew two-quasiparticle field gradient.

    .. math::

       H^{20}=U^\dagger hV^*-V^\dagger h^T U^*
              -V^\dagger\Delta^*V^*+U^\dagger\Delta U^*.

    ``h`` is Hermitian and ``delta`` is transpose-antisymmetric. The result
    must be skew-symmetric under transpose.
    """
    u, v, h, delta = _validate_fields(u, v, h, delta)
    result = (
        u.mH @ h @ v.conj()
        - v.mH @ h.mT @ u.conj()
        - v.mH @ delta.conj() @ v.conj()
        + u.mH @ delta @ u.conj()
    )
    if not torch.allclose(result.mT, -result, atol=tolerance, rtol=0.0):
        raise ValueError("H20 is not skew-symmetric; U/V or field conventions disagree")
    return result


def n20_matrix(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return the number-operator 20 block in the same U/V convention."""
    u, v = _validate_uv(u, v)
    result = u.mH @ v.conj() - v.mH @ u.conj()
    if not torch.allclose(result.mT, -result, atol=1.0e-9, rtol=0.0):
        raise ValueError("N20 is not skew-symmetric for this canonical U/V frame")
    return result


def project_h20_number_tangent(
    h20: torch.Tensor,
    n20: torch.Tensor,
    *,
    fallback_mu: torch.Tensor | float = 0.0,
    numerical_gram_tolerance: float = 64.0 * torch.finfo(torch.float64).eps,
) -> NumberProjectedH20:
    """Remove the least-squares number-gradient component from H20.

    This is the one-constraint specialization of the Thouless Gram projection
    in Robledo & Bertsch Eq. (17): ``mu=Re sum(conj(N20)*H20)/||N20||^2``.
    A numerically singular number tangent is reported as unidentifiable and
    uses ``fallback_mu``; the tolerance is only a floating-point singularity
    test, not a pairing or physical acceptance threshold.
    """
    h20, n20 = _validate_20_pair(h20, n20)
    if (
        not math.isfinite(float(numerical_gram_tolerance))
        or numerical_gram_tolerance < 0.0
    ):
        raise ValueError("numerical Gram tolerance must be finite and nonnegative")
    fallback = torch.as_tensor(
        fallback_mu, dtype=torch.float64, device=h20.device
    )
    if fallback.shape != () or not bool(torch.isfinite(fallback)):
        raise ValueError("fallback_mu must be a finite scalar")
    gram = n20.abs().square().sum()
    numerator = (n20.conj() * h20).sum().real
    identifiable = bool(gram.detach() > numerical_gram_tolerance)
    mu = numerator / gram if identifiable else fallback
    if not bool(torch.isfinite(mu)):
        raise FloatingPointError("number-tangent projection became nonfinite")
    return NumberProjectedH20(
        chemical_potential=mu,
        identifiable=identifiable,
        h20=h20,
        n20=n20,
        constrained_h20=h20 - mu * n20,
        gram=gram,
        numerator=numerator,
    )


def project_fields_number_tangent(
    u: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    delta: torch.Tensor,
    *,
    fallback_mu: torch.Tensor | float = 0.0,
    numerical_gram_tolerance: float = 64.0 * torch.finfo(torch.float64).eps,
) -> NumberProjectedH20:
    """Build H20/N20 from fields and project the number direction."""
    h20 = h20_matrix(u, v, h, delta)
    n20 = n20_matrix(u, v)
    return project_h20_number_tangent(
        h20,
        n20,
        fallback_mu=fallback_mu,
        numerical_gram_tolerance=numerical_gram_tolerance,
    )


def apply_thouless_update(
    frame: torch.Tensor,
    z: torch.Tensor,
    *,
    step: float,
    fermion_parity: int,
    blocked_index: int | None = None,
) -> BogoliubovState:
    """Apply a finite canonical Thouless step and rebuild its U/V density.

    This supplies the variational manifold action ``W' = W exp(K_Z)``. It
    does not choose a step size, perform a line search, minimize the energy,
    or certify a physical branch. ``Z`` is the independent skew 20
    coordinate; caller-supplied ``step`` can be signed.
    """
    frame = _complex128_finite(frame, "frame")
    z = _complex128_finite(z, "Z")
    if frame.ndim != 2 or frame.shape[0] != frame.shape[1] or frame.shape[0] % 2:
        raise ValueError("frame must be a square (2M,2M) matrix")
    m = frame.shape[0] // 2
    if z.shape != (m, m) or z.device != frame.device:
        raise ValueError("Z must match the active frame dimension and device")
    if not math.isfinite(float(step)):
        raise ValueError("Thouless step must be finite")
    if not torch.allclose(z.mT, -z, atol=1.0e-10, rtol=0.0):
        raise ValueError("Thouless Z must be skew-symmetric under transpose")
    # Validate the input frame before transporting it on the canonical group.
    state_from_frame(frame, fermion_parity=fermion_parity)
    zero = torch.zeros_like(z)
    generator = canonical_generator(zero, z)
    updated = frame @ torch.matrix_exp(float(step) * generator)
    column = None if blocked_index is None else m + int(blocked_index)
    return state_from_frame(
        updated,
        fermion_parity=fermion_parity,
        blocked_index=blocked_index,
        blocked_mode_column=column,
    )


def thouless_density_derivative(
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return first-order ``(d rho, d chi)`` for an arbitrary skew Z.

    The generalized Thouless update is ``dU=V* Z*`` and ``dV=U* Z*``.
    This computes the density derivatives directly from those U/V variations,
    providing an independent route to the H20 contraction.
    """
    u, v = _validate_uv(u, v)
    z = _complex128_finite(z, "Z")
    if z.shape != u.shape or z.device != u.device:
        raise ValueError("Z must match the U/V active space and device")
    if not torch.allclose(z.mT, -z, atol=1.0e-10, rtol=0.0):
        raise ValueError("Thouless Z must be skew-symmetric under transpose")
    du = v.conj() @ z.conj()
    dv = u.conj() @ z.conj()
    drho = dv.conj() @ v.mT + v.conj() @ dv.mT
    dchi = dv.conj() @ u.mT + v.conj() @ du.mT
    if not torch.allclose(drho, drho.mH, atol=1.0e-9, rtol=0.0):
        raise ValueError("Thouless variation produced a non-Hermitian d rho")
    if not torch.allclose(dchi.mT, -dchi, atol=1.0e-9, rtol=0.0):
        raise ValueError("Thouless variation produced a non-skew d chi")
    return drho, dchi


def field_energy_directional_derivative(
    h: torch.Tensor,
    delta: torch.Tensor,
    drho: torch.Tensor,
    dchi: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate ``Tr(h d rho) - Re Tr(Delta* d chi)``.

    The factor convention is the functional derivative convention for HFB
    fields: if ``E=Tr(t rho)+1/2 Tr(Gamma rho)-1/2 Tr(Delta chi*)``, with
    self-consistent fields ``h=t+Gamma`` and ``Delta`` from the pairing
    functional, this expression is the full first variation. It includes the
    normal and anomalous double-counting closures.
    """
    h = _complex128_finite(h, "h")
    delta = _complex128_finite(delta, "Delta")
    drho = _complex128_finite(drho, "d rho")
    dchi = _complex128_finite(dchi, "d chi")
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be square")
    if any(value.shape != h.shape or value.device != h.device
           for value in (delta, drho, dchi)):
        raise ValueError("fields and density variations must share shape and device")
    if not torch.allclose(h, h.mH, atol=1.0e-9, rtol=0.0):
        raise ValueError("h must be Hermitian")
    if not torch.allclose(delta.mT, -delta, atol=1.0e-9, rtol=0.0):
        raise ValueError("Delta must be skew-symmetric under transpose")
    if not torch.allclose(drho, drho.mH, atol=1.0e-9, rtol=0.0):
        raise ValueError("d rho must be Hermitian")
    if not torch.allclose(dchi.mT, -dchi, atol=1.0e-9, rtol=0.0):
        raise ValueError("d chi must be skew-symmetric under transpose")
    normal = torch.trace(h @ drho)
    pairing = torch.trace(delta.conj() @ dchi)
    return normal.real - pairing.real


def thouless_energy_directional_derivative(
    u: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    delta: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the same first variation by the quasiparticle H20 gradient."""
    h20 = h20_matrix(u, v, h, delta)
    z = _complex128_finite(z, "Z")
    if z.shape != h20.shape or z.device != h20.device:
        raise ValueError("Z must match H20")
    if not torch.allclose(z.mT, -z, atol=1.0e-10, rtol=0.0):
        raise ValueError("Thouless Z must be skew-symmetric under transpose")
    # Robledo & Bertsch Eq. (14); Z's independent coordinates are skew.
    return -torch.trace(h20 @ z.conj()).real


def _validate_fields(
    u: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u, v = _validate_uv(u, v)
    h = _complex128_finite(h, "h")
    delta = _complex128_finite(delta, "Delta")
    if h.shape != u.shape or delta.shape != u.shape:
        raise ValueError("h and Delta must match the U/V active dimension")
    if h.device != u.device or delta.device != u.device:
        raise ValueError("U, V, h and Delta must share a device")
    if not torch.allclose(h, h.mH, atol=1.0e-10, rtol=0.0):
        raise ValueError("h must be Hermitian")
    if not torch.allclose(delta, -delta.mT, atol=1.0e-10, rtol=0.0):
        raise ValueError("Delta must be skew-symmetric under transpose")
    return u, v, h, delta


def _validate_uv(u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u = _complex128_finite(u, "U")
    v = _complex128_finite(v, "V")
    if u.ndim != 2 or u.shape[0] != u.shape[1] or v.shape != u.shape:
        raise ValueError("U and V must be square matrices of identical shape")
    if u.device != v.device:
        raise ValueError("U and V must be on the same device")
    identity = torch.eye(u.shape[0], dtype=u.dtype, device=u.device)
    if not torch.allclose(u.mH @ u + v.mH @ v, identity, atol=1.0e-9, rtol=0.0):
        raise ValueError("U/V columns do not satisfy U†U+V†V=I")
    if not torch.allclose(u.mT @ v + v.mT @ u, torch.zeros_like(u),
                          atol=1.0e-9, rtol=0.0):
        raise ValueError("U/V columns do not satisfy UᵀV+VᵀU=0")
    return u, v


def _validate_20_pair(h20: torch.Tensor, n20: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h20 = _complex128_finite(h20, "H20")
    n20 = _complex128_finite(n20, "N20")
    if h20.ndim != 2 or h20.shape[0] != h20.shape[1] or n20.shape != h20.shape:
        raise ValueError("H20 and N20 must be square matrices of identical shape")
    if h20.device != n20.device:
        raise ValueError("H20 and N20 must share a device")
    if not torch.allclose(h20.mT, -h20, atol=1.0e-9, rtol=0.0):
        raise ValueError("H20 must be skew-symmetric under transpose")
    if not torch.allclose(n20.mT, -n20, atol=1.0e-9, rtol=0.0):
        raise ValueError("N20 must be skew-symmetric under transpose")
    return h20, n20

