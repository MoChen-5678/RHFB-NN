"""Local covariant bilinears for arbitrary complex four-component spinors.

The matrices use the Dirac representation and metric ``(+---)``.  The active
one-body density is supplied explicitly, so blocked and time-reversal-breaking
states produce their spatial currents, axial densities, and tensor densities.
"""

from __future__ import annotations

import torch


def _gamma_matrices(*, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    i2 = torch.eye(2, dtype=dtype, device=device)
    z2 = torch.zeros((2, 2), dtype=dtype, device=device)
    sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    g0 = torch.block_diag(i2, -i2)
    gammas = [g0]
    for sigma in (sx, sy, sz):
        gammas.append(torch.cat((torch.cat((z2, sigma), 1), torch.cat((-sigma, z2), 1)), 0))
    gamma5 = 1j * gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
    return tuple(gammas), gamma5


def local_bilinears(spinors: torch.Tensor, rho_active: torch.Tensor) -> dict[str, torch.Tensor]:
    """Evaluate local Dirac bilinears from sampled spinors and active density.

    Args:
        spinors: Complex128 values ``Q[p,component,state]`` with four Dirac
            components ordered (upper-up, upper-down, lower-up, lower-down).
            No quadrature weights are included.
        rho_active: Complex128 Hermitian active one-body density, shape (M,M).

    Returns a dict of pointwise values:
        ``scalar``: ``bar(psi) psi``;
        ``baryon_current``: contravariant ``bar(psi) gamma^mu psi`` ordered
            (time,x,y,z);
        ``axial_current``: contravariant ``bar(psi) gamma^mu gamma5 psi``;
        ``tensor``: ``bar(psi) sigma^{mu nu} psi`` as a (4,4) antisymmetric
            array, where ``sigma^{mu nu}=i[gamma^mu,gamma^nu]/2``.

    Isospin projection and proton charge are intentionally combined by the
    caller from these per-species bilinears.
    """
    if spinors.ndim != 3 or spinors.shape[1] != 4:
        raise ValueError("spinors must have shape (points, 4 Dirac components, states)")
    npoint, _, nstate = spinors.shape
    if npoint == 0 or nstate == 0:
        raise ValueError("spinors must contain at least one point and one state")
    if rho_active.shape != (nstate, nstate):
        raise ValueError("rho_active must have shape (states, states) matching spinors")
    if spinors.dtype != torch.complex128 or rho_active.dtype != torch.complex128:
        raise TypeError("spinors and rho_active must use complex128")
    if spinors.device != rho_active.device:
        raise ValueError("spinors and rho_active must be on the same device")
    if not bool(torch.isfinite(spinors).all()) or not bool(torch.isfinite(rho_active).all()):
        raise ValueError("spinors and rho_active must be finite")
    hermitian_error = torch.max(torch.abs(rho_active - rho_active.mH)).detach().item()
    if hermitian_error > 1.0e-10:
        raise ValueError(f"rho_active must be Hermitian; max error={hermitian_error:g}")

    gamma, gamma5 = _gamma_matrices(device=spinors.device, dtype=spinors.dtype)
    # D[p,i,j] = sum_ab Q[p,i,a] rho[a,b] Q*[p,j,b].
    local_density = torch.einsum("ab,pia,pjb->pij", rho_active, spinors, spinors.conj())

    def bilinear(operator: torch.Tensor) -> torch.Tensor:
        # Tr(D O) equals sum occupied psi^dagger O psi.
        return torch.einsum("pij,ji->p", local_density, operator)

    scalar = bilinear(gamma[0])
    baryon = torch.stack(tuple(bilinear(gamma[0] @ gm) for gm in gamma), dim=-1).real
    axial = torch.stack(tuple(bilinear(gamma[0] @ gm @ gamma5) for gm in gamma), dim=-1).real
    sigma_up = []
    for mu in range(4):
        row = []
        for nu in range(4):
            sigma = 0.5j * (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu])
            row.append(bilinear(gamma[0] @ sigma))
        sigma_up.append(torch.stack(row, dim=-1))
    tensor = torch.stack(sigma_up, dim=-2)
    # Hermitian density with Hermitian observables gives real-valued bilinears;
    # discard only floating-point imaginary roundoff.
    scalar = scalar.real
    tensor = tensor.real
    return {
        "scalar": scalar,
        "baryon_current": baryon,
        "axial_current": axial,
        "tensor": tensor,
    }


def invariant_baryon_density(
    total_current: torch.Tensor,
    *,
    vacuum_tolerance: float = 1.0e-14,
) -> torch.Tensor:
    """Return ``sqrt(j_mu j^mu)`` for a future-timelike baryon four-current.

    ``total_current[..., :]`` is contravariant ``(j0,jx,jy,jz)`` and the metric
    is ``(+---)``.  Exact/controlled vacuum zeros return zero.  A materially
    spacelike or non-vacuum null current raises rather than being clamped into
    a fictitious scalar density.
    """
    if total_current.ndim < 1 or total_current.shape[-1] != 4:
        raise ValueError("total_current must end in four components (j0,jx,jy,jz)")
    if not total_current.is_floating_point() or total_current.dtype != torch.float64:
        raise TypeError("total_current must use float64")
    if not bool(torch.isfinite(total_current).all()):
        raise ValueError("total_current must be finite")
    if vacuum_tolerance < 0:
        raise ValueError("vacuum_tolerance must be nonnegative")
    j0 = total_current[..., 0]
    j2 = j0.square() - torch.sum(total_current[..., 1:].square(), dim=-1)
    vacuum = torch.max(torch.abs(total_current), dim=-1).values <= vacuum_tolerance
    invalid = ((j2 <= 0) | (j0 <= 0)) & ~vacuum
    if bool(invalid.any()):
        bad_min = torch.min(j2[invalid]).detach().item()
        raise ValueError(
            "baryon current must be future-timelike outside vacuum; "
            f"minimum j_mu j^mu={bad_min:g}"
        )
    safe_j2 = torch.where(vacuum, torch.ones_like(j2), j2)
    return torch.where(vacuum, torch.zeros_like(j2), torch.sqrt(safe_j2))
