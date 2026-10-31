"""Canonical U/V frames and explicit one-quasiparticle blocking.

The convention matches :mod:`dpl_rhf.rhfb.manifold`:

``W = [[U, V*], [V, U*]]`` and ``R = W diag(0, I) W†``.

Consequently ``rho = V* V.T`` and ``chi = V* U.T``.  Blocking one
quasiparticle exchanges one particle-hole column pair of ``W`` and rebuilds
the density from the exchanged frame.  The selected column is a constructor
tag, not a Hamiltonian quasiparticle identity or a ground-state certificate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .manifold import GeneralizedDensity, generalized_density_matrix


@dataclass(frozen=True)
class BogoliubovState:
    """A canonical active frame and the density it defines.

    ``blocked_mode`` is an optional occupied Nambu column selected by the
    caller.  ``physical_identity_certified`` is always false: U/V column
    bookkeeping alone does not establish an instantaneous qp eigenmode.
    """

    u: torch.Tensor
    v: torch.Tensor
    frame: torch.Tensor
    density: GeneralizedDensity
    blocked_mode: torch.Tensor | None = None
    blocked_index: int | None = None
    physical_identity_certified: bool = False


def bogoliubov_frame(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Assemble the full ``2M x 2M`` canonical frame from U and V."""

    _validate_uv_shapes(u, v)
    return torch.cat(
        (
            torch.cat((u, v.conj()), dim=1),
            torch.cat((v, u.conj()), dim=1),
        ),
        dim=0,
    )


def slater_bogoliubov_frame(occupations: torch.Tensor) -> torch.Tensor:
    """Construct the canonical frame for zero/one Slater occupations.

    This is the reference-frame construction used by the neural canonical
    chart.  It does not perform a quasiparticle swap; supplied occupations
    already describe the reference state.
    """

    occupations = torch.as_tensor(occupations)
    if occupations.dtype != torch.float64:
        raise TypeError("occupations must use float64")
    if occupations.ndim != 1 or occupations.numel() == 0:
        raise ValueError("occupations must be a nonempty vector")
    if not bool(torch.isfinite(occupations).all()) or not bool(
        torch.all((occupations == 0.0) | (occupations == 1.0))
    ):
        raise ValueError("Slater occupations must be finite zero/one values")
    occupied = torch.diag(occupations.to(torch.complex128))
    empty = torch.eye(
        occupations.numel(), dtype=torch.complex128, device=occupations.device
    ) - occupied
    return bogoliubov_frame(empty, occupied)


def bogoliubov_density(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    fermion_parity: int,
) -> GeneralizedDensity:
    """Rebuild ``rho``, ``chi`` and ``R`` from the documented U/V convention."""

    _validate_uv_shapes(u, v)
    if fermion_parity not in {-1, 1}:
        raise ValueError("fermion_parity must be -1 or 1")
    rho = v.conj() @ v.mT
    chi = v.conj() @ u.mT
    matrix = generalized_density_matrix(rho, chi)
    return GeneralizedDensity(rho, chi, matrix, int(fermion_parity))


def state_from_frame(
    frame: torch.Tensor,
    *,
    fermion_parity: int,
    blocked_index: int | None = None,
    blocked_mode_column: int | None = None,
    tolerance: float = 1.0e-9,
) -> BogoliubovState:
    """Validate a full canonical frame and derive its density.

    The optional mode is selected by an explicit frame column.  It is
    normalized and occupied in the supplied generalized density, but is not
    thereby certified as an eigen-quasiparticle of any Hamiltonian.
    """

    frame = _complex128_finite(frame, "frame")
    if frame.ndim != 2 or frame.shape[0] != frame.shape[1] or frame.shape[0] % 2:
        raise ValueError("frame must be a square (2M,2M) matrix")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    m = frame.shape[0] // 2
    if fermion_parity not in {-1, 1}:
        raise ValueError("fermion_parity must be -1 or 1")
    if blocked_index is not None and not 0 <= int(blocked_index) < m:
        raise IndexError("blocked_index lies outside the active space")
    if blocked_mode_column is not None and not 0 <= int(blocked_mode_column) < 2 * m:
        raise IndexError("blocked_mode_column lies outside the Nambu frame")

    u = frame[:m, :m]
    v = frame[m:, :m]
    canonical = bogoliubov_frame(u, v)
    identity = torch.eye(2 * m, dtype=frame.dtype, device=frame.device)
    if not torch.allclose(frame, canonical, atol=tolerance, rtol=0.0):
        raise ValueError("frame does not have canonical particle-hole block structure")
    if not torch.allclose(frame.mH @ frame, identity, atol=tolerance, rtol=0.0):
        raise ValueError("Bogoliubov frame is not unitary/canonical")

    density = bogoliubov_density(u, v, fermion_parity=fermion_parity)
    if not torch.allclose(
        density.matrix @ density.matrix,
        density.matrix,
        atol=10.0 * tolerance,
        rtol=0.0,
    ):
        raise ValueError("Bogoliubov frame does not define a pure generalized density")
    mode = None
    if blocked_mode_column is not None:
        mode = frame[:, int(blocked_mode_column)]
        if not torch.allclose(density.matrix @ mode, mode, atol=10.0 * tolerance, rtol=0.0):
            raise ValueError("selected blocked-mode column is not occupied by the density")
    return BogoliubovState(
        u=u,
        v=v,
        frame=frame,
        density=density,
        blocked_mode=mode,
        blocked_index=None if blocked_index is None else int(blocked_index),
        physical_identity_certified=False,
    )


def swap_quasiparticle_column(
    frame: torch.Tensor,
    index: int,
    *,
    fermion_parity: int,
    tolerance: float = 1.0e-9,
) -> BogoliubovState:
    """Block one qp by exchanging frame columns ``index`` and ``M+index``.

    This is a pure-state one-qp blocking operation.  It exchanges the chosen
    particle/hole-conjugate columns exactly once, rebuilds U/V and the
    generalized density, and flips number parity metadata.  It makes no
    equal-filling, spherical-symmetry, or time-odd approximation.
    """

    frame = _complex128_finite(frame, "frame")
    if frame.ndim != 2 or frame.shape[0] != frame.shape[1] or frame.shape[0] % 2:
        raise ValueError("frame must be a square (2M,2M) matrix")
    if fermion_parity not in {-1, 1}:
        raise ValueError("fermion_parity must be -1 or 1")
    m = frame.shape[0] // 2
    if not 0 <= int(index) < m:
        raise IndexError("quasiparticle index lies outside the active space")
    # Validate the even/odd input before doing the explicit PH exchange.
    state_from_frame(frame, fermion_parity=fermion_parity, tolerance=tolerance)
    swapped = frame.clone()
    left = swapped[:, int(index)].clone()
    swapped[:, int(index)] = swapped[:, m + int(index)]
    swapped[:, m + int(index)] = left
    return state_from_frame(
        swapped,
        fermion_parity=-fermion_parity,
        blocked_index=int(index),
        blocked_mode_column=m + int(index),
        tolerance=tolerance,
    )


def _validate_uv_shapes(u: torch.Tensor, v: torch.Tensor) -> None:
    u = _complex128_finite(u, "U")
    v = _complex128_finite(v, "V")
    if u.ndim != 2 or u.shape[0] != u.shape[1] or v.shape != u.shape:
        raise ValueError("U and V must be square matrices of identical shape")
    if u.device != v.device:
        raise ValueError("U and V must be on the same device")


def _complex128_finite(value: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if result.dtype != torch.complex128:
        raise TypeError(f"{name} must use complex128")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result
