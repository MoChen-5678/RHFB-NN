"""Finite-range D1S neutron/proton pairing for the blocked-RHFB prototype.

This module implements only the two-Gaussian central part of Gogny D1S in
separate nn and pp channels.  It deliberately has no J=0, spherical, or
time-reversal reduction.  Coordinates and quadrature weights are separate;
``spinors[p, dirac, state]`` is an unweighted sampled Dirac spinor.

The output convention is the standard antisymmetrized one

    E_pp = 1/4 sum_abcd chi*_ab Vbar_abcd chi_cd,
    Delta_ab = 1/2 sum_cd Vbar_abcd chi_cd.

The oracle below materializes Vbar only for tiny tests.  The production path
streams coordinate chunks and never constructs a four-state-index tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class GognyD1S:
    """D1S finite-range central-force parameters (MeV and fm)."""

    ranges_fm: tuple[float, float] = (0.7, 1.2)
    w_mev: tuple[float, float] = (-1720.30, 103.639)
    b_mev: tuple[float, float] = (1300.00, -163.483)
    h_mev: tuple[float, float] = (-1813.53, 162.812)
    # The published original D1S convention is commonly tabulated as -223.934.
    # CEA's online table gives -223.933; this 1-keV rounding discrepancy is
    # recorded rather than silently mixed.  We use the standard -223.934 value.
    m_mev: tuple[float, float] = (1397.60, -223.934)
    strength: float = 1.0


D1S = GognyD1S()
D1S_PROVENANCE = (
    "J.F. Berger, M. Girod, and D. Gogny, Comput. Phys. Commun. 63, 365 (1991), "
    "doi:10.1016/0010-4655(91)90263-K; parameter values cross-checked against "
    "the D1S table in CEA Gogny Force Characteristics. CEA lists M2=-223.933; "
    "the standard tabulation used here is M2=-223.934 MeV."
)


def _spin_exchange_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Input component pair selected by P_sigma for each output pair.

    Components are ordered (upper-up, upper-down, lower-up, lower-down).
    P_sigma exchanges Pauli spin between particles while preserving each
    particle's Dirac upper/lower block.
    """
    a = torch.arange(4, device=device)[:, None]
    b = torch.arange(4, device=device)[None, :]
    gamma = 2 * (a // 2) + (b % 2)
    delta = 2 * (b // 2) + (a % 2)
    return gamma.expand(4, 4), delta.expand(4, 4)


def _validate_inputs(
    spinors: torch.Tensor,
    chi: torch.Tensor,
    positions: torch.Tensor,
    pointweights: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
    if spinors.ndim != 3 or spinors.shape[1] != 4:
        raise ValueError("spinors must have shape (points, 4 Dirac components, states)")
    npoint, _, nstate = spinors.shape
    if npoint == 0 or nstate == 0:
        raise ValueError("spinors must contain at least one point and one state")
    if chi.shape != (nstate, nstate):
        raise ValueError("chi must have shape (states, states) matching spinors")
    if positions.shape != (npoint, 3):
        raise ValueError("positions must have shape (points, 3)")
    if pointweights.shape != (npoint,):
        raise ValueError("pointweights must have shape (points,)")
    if spinors.dtype != torch.complex128 or chi.dtype != torch.complex128:
        raise TypeError("spinors and chi must use complex128")
    if positions.dtype != torch.float64 or pointweights.dtype != torch.float64:
        raise TypeError("positions and pointweights must use float64")
    if any(t.device != spinors.device for t in (chi, positions, pointweights)):
        raise ValueError("all inputs must be on the same device")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not bool(torch.isfinite(spinors).all()) or not bool(torch.isfinite(chi).all()):
        raise ValueError("spinors and chi must be finite")
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(pointweights).all()):
        raise ValueError("positions and pointweights must be finite")
    skew_error = torch.max(torch.abs(chi + chi.T)).detach().item()
    if skew_error > 1.0e-10:
        raise ValueError(f"chi must be skew-symmetric; max |chi+chi.T|={skew_error:g}")
    return npoint, nstate


def pairing_gap(
    spinors: torch.Tensor,
    chi: torch.Tensor,
    positions: torch.Tensor,
    pointweights: torch.Tensor,
    *,
    species: str,
    chunk_size: int = 32,
    force: GognyD1S = D1S,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(E_pp, Delta)`` for one like-particle species.

    Args:
        spinors: Physical, unweighted samples ``Q[p,component,state]`` with
            components (upper-up, upper-down, lower-up, lower-down).
        chi: Complex skew anomalous density in the state basis, shape (M,M).
        positions: Coordinates of samples in fm, shape (P,3).
        pointweights: Quadrature weights in fm^3, shape (P,).  Weights are
            applied once at each of the two interaction endpoints.
        species: ``"neutron"`` or ``"proton"``.  No pn channel is provided.
        chunk_size: Number of first-endpoint coordinates streamed at once.
        force: D1S finite-range parameters; the standard strength is exactly 1.

    Notes:
        D1S is inserted into the relativistic pp channel with gamma-zero
        factors at both endpoints.  In Hamiltonian representation these give
        psi^dagger psi, so upper and lower components enter with the same sign.
        The spin-exchange operator exchanges Pauli-spin labels, preserving each
        particle's upper/lower Dirac block.  All angular-momentum channels are
        present; this routine performs no J=0 projection. For basis outputs
        too large to hold as a full Q array, use :func:`pairing_gap_from_sampler`.
    """
    if species not in ("neutron", "proton"):
        raise ValueError("species must be 'neutron' or 'proton'; pn pairing is not implemented")
    npoint, _ = _validate_inputs(spinors, chi, positions, pointweights, chunk_size)

    def sampler(start: int, stop: int):
        return spinors[start:stop], pointweights[start:stop], positions[start:stop]

    return pairing_gap_from_sampler(
        chi,
        npoint,
        sampler,
        species=species,
        chunk_size=chunk_size,
        force=force,
        checkpoint_blocks=False,
    )


def _validate_sample(
    sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    expected_count: int,
    nstate: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        raise TypeError("sampler(start, stop) must return (spinors, pointweights, positions)")
    q, weights, positions = sample
    if not all(isinstance(x, torch.Tensor) for x in sample):
        raise TypeError("sampler outputs must be torch tensors")
    if q.shape != (expected_count, 4, nstate):
        raise ValueError(
            f"sampler spinors must have shape ({expected_count}, 4, {nstate}); got {tuple(q.shape)}"
        )
    if weights.shape != (expected_count,) or positions.shape != (expected_count, 3):
        raise ValueError("sampler weights/positions must have shapes (count,) and (count,3)")
    if q.dtype != torch.complex128 or weights.dtype != torch.float64 or positions.dtype != torch.float64:
        raise TypeError("sampler must return complex128 spinors and float64 weights/positions")
    if any(t.device != device for t in sample):
        raise ValueError("sampler outputs and chi must be on the same device")
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("sampler spinors and weights must be finite")
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("sampler positions must be finite")
    return q, weights, positions


def _pair_delta_block(
    qp: torch.Tensor,
    qq: torch.Tensor,
    chi: torch.Tensor,
    xp: torch.Tensor,
    xq: torch.Tensor,
    wp: torch.Tensor,
    wq: torch.Tensor,
    *,
    force: GognyD1S,
    spin_a: torch.Tensor,
    spin_b: torch.Tensor,
) -> torch.Tensor:
    ranges = torch.as_tensor(force.ranges_fm, device=qp.device, dtype=qp.real.dtype)
    w = torch.as_tensor(force.w_mev, device=qp.device, dtype=qp.real.dtype)
    b = torch.as_tensor(force.b_mev, device=qp.device, dtype=qp.real.dtype)
    h = torch.as_tensor(force.h_mev, device=qp.device, dtype=qp.real.dtype)
    m = torch.as_tensor(force.m_mev, device=qp.device, dtype=qp.real.dtype)
    coeff_direct = (w - h) * force.strength
    coeff_spin = (b - m) * force.strength
    qamp = torch.einsum("pia,ab,qjb->pqij", qp, chi, qq)
    distance2 = torch.sum((xp[:, None, :] - xq[None, :, :]) ** 2, dim=-1)
    gaussian = torch.exp(-distance2[..., None] / (ranges * ranges))
    direct_kernel = torch.sum(gaussian * coeff_direct, dim=-1)
    spin_kernel = torch.sum(gaussian * coeff_spin, dim=-1)
    acted = direct_kernel[..., None, None] * qamp + spin_kernel[..., None, None] * qamp[..., spin_a, spin_b]
    weights_pq = (wp[:, None] * wq[None, :]).to(qp.dtype)
    return torch.einsum(
        "pia,qjb,pqij,pq->ab", qp.conj(), qq.conj(), acted, weights_pq
    )


def pairing_gap_from_sampler(
    chi: torch.Tensor,
    point_count: int,
    sampler,
    *,
    species: str,
    chunk_size: int = 32,
    force: GognyD1S = D1S,
    checkpoint_blocks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded-memory D1S contraction from a differentiable coordinate sampler.

    ``sampler(start, stop)`` must generate only that slice and return
    ``(physical_spinors, pointweights, positions)``.  Both coordinate endpoints
    are chunked, so no full pointwise Q array or P-by-P pair amplitude is
    formed by this API.  With ``checkpoint_blocks=True`` each first-endpoint
    block is exactly recomputed during backward; sampler parameters captured by
    the closure retain gradients, without saving all generated coordinate
    chunks in the autograd graph.  The sampler must be deterministic and pure
    for identical ``start, stop`` arguments.
    """
    if species not in ("neutron", "proton"):
        raise ValueError("species must be 'neutron' or 'proton'; pn pairing is not implemented")
    if chi.ndim != 2 or chi.shape[0] != chi.shape[1]:
        raise ValueError("chi must be a square (states, states) matrix")
    if chi.dtype != torch.complex128:
        raise TypeError("chi must use complex128")
    if not isinstance(point_count, int) or point_count < 1:
        raise ValueError("point_count must be a positive integer")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if force.strength != 1.0:
        raise ValueError("the blocked-RHFB prototype fixes unscaled D1S strength to 1")
    nstate = chi.shape[0]
    skew_error = torch.max(torch.abs(chi + chi.T)).detach().item()
    if skew_error > 1.0e-10:
        raise ValueError(f"chi must be skew-symmetric; max |chi+chi.T|={skew_error:g}")
    spin_a, spin_b = _spin_exchange_indices(chi.device)

    def accumulate_first_block(chi_arg: torch.Tensor, p0: int, p1: int) -> torch.Tensor:
        qp, wp, xp = _validate_sample(
            sampler(p0, p1), expected_count=p1 - p0, nstate=nstate, device=chi.device
        )
        partial = torch.zeros((nstate, nstate), dtype=chi.dtype, device=chi.device)
        for q0 in range(0, point_count, chunk_size):
            q1 = min(point_count, q0 + chunk_size)
            qq, wq, xq = _validate_sample(
                sampler(q0, q1), expected_count=q1 - q0, nstate=nstate, device=chi.device
            )
            partial = partial + _pair_delta_block(
                qp, qq, chi_arg, xp, xq, wp, wq,
                force=force, spin_a=spin_a, spin_b=spin_b,
            )
        return partial

    delta = torch.zeros((nstate, nstate), dtype=chi.dtype, device=chi.device)
    for p0 in range(0, point_count, chunk_size):
        p1 = min(point_count, p0 + chunk_size)
        if checkpoint_blocks and torch.is_grad_enabled():
            partial = checkpoint(
                lambda chi_arg, lo=p0, hi=p1: accumulate_first_block(chi_arg, lo, hi),
                chi,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            partial = accumulate_first_block(chi, p0, p1)
        delta = delta + partial

    delta = 0.5 * (delta - delta.T)
    energy = 0.5 * torch.sum(chi.conj() * delta).real
    return energy, delta


def pairing_energy_gap(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias with an explicit energy-and-gap name."""
    return pairing_gap(*args, **kwargs)


def pairing_gap_oracle(
    spinors: torch.Tensor,
    chi: torch.Tensor,
    positions: torch.Tensor,
    pointweights: torch.Tensor,
    *,
    species: str,
    force: GognyD1S = D1S,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny explicit coordinate-spin oracle, including full ``Vbar``.

    This deliberately constructs the M^4 antisymmetrized matrix element and
    is suitable only for small independent test cases.  Production code must
    use :func:`pairing_gap`.
    """
    import numpy as np

    if species not in ("neutron", "proton"):
        raise ValueError("species must be 'neutron' or 'proton'; pn pairing is not implemented")
    npoint, nstate = _validate_inputs(spinors, chi, positions, pointweights, max(1, spinors.shape[0]))
    q = spinors.detach().cpu().numpy()
    c = chi.detach().cpu().numpy()
    x = positions.detach().cpu().numpy()
    wt = pointweights.detach().cpu().numpy()
    direct = np.zeros((nstate, nstate, nstate, nstate), dtype=np.complex128)
    flip = np.empty((4, 4, 2), dtype=np.int64)
    for a in range(4):
        for bcomp in range(4):
            flip[a, bcomp] = (2 * (a // 2) + (bcomp % 2), 2 * (bcomp // 2) + (a % 2))
    for p in range(npoint):
        for r in range(npoint):
            d2 = float(np.dot(x[p] - x[r], x[p] - x[r]))
            c0 = 0.0
            cs = 0.0
            for i in range(2):
                g = np.exp(-d2 / force.ranges_fm[i] ** 2)
                c0 += g * (force.w_mev[i] - force.h_mev[i])
                cs += g * (force.b_mev[i] - force.m_mev[i])
            factor = force.strength * wt[p] * wt[r]
            for a in range(4):
                for bcomp in range(4):
                    for ccomp in range(4):
                        for dcomp in range(4):
                            spin_op = c0 * (a == ccomp and bcomp == dcomp)
                            gam, delt = flip[a, bcomp]
                            spin_op += cs * (ccomp == gam and dcomp == delt)
                            if spin_op == 0:
                                continue
                            direct += factor * spin_op * np.einsum(
                                "a,b,c,d->abcd",
                                q[p, a].conj(), q[r, bcomp].conj(),
                                q[p, ccomp], q[r, dcomp],
                            )
    vbar = direct - direct.transpose(0, 1, 3, 2)
    gap = 0.5 * np.einsum("abcd,cd->ab", vbar, c)
    energy = 0.25 * np.einsum("ab,ab->", c.conj(), np.einsum("abcd,cd->ab", vbar, c)).real
    return (
        torch.as_tensor(energy, dtype=torch.float64, device=spinors.device),
        torch.as_tensor(gap, dtype=torch.complex128, device=spinors.device),
    )
