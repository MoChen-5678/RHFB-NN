"""Independent all-J angular recoupling oracle for the relativistic D1S pp term.

This reference implements the multipole matrix elements in the uncoupled
spinor-spherical-harmonic basis using Clebsch-Gordan and Gaunt coefficients.
It does not sample coordinate angles and does not call the production Gaussian
Green/action or the direct coordinate-pair kernel.  It is intentionally small
space only; the radial grid is contracted with the analytic Gaussian multipole
coefficient while angular factors are evaluated independently.

The operator convention follows Geng & Long, PRC 105, 034329 (2022),
Eqs. (20d), (24)-(26), (30)-(32), and Appendix B, Eqs. (B1)-(B12): the two
gamma-zero factors cancel the Dirac adjoints at the two endpoints, D1S acts
with equal upper/lower Hamiltonian signs, and all J components are retained.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch
from scipy.special import ive
from sympy import Rational, S, sqrt as sympy_sqrt
from sympy.physics.wigner import clebsch_gordan, wigner_3j

from .pairing import D1S, GognyD1S


def _spinor_harmonic_terms(kappa: int, m2: int) -> tuple[tuple[int, int, float], ...]:
    """Return ``(m_l, spin2, coefficient)`` in Condon-Shortley convention."""
    ell = -kappa - 1 if kappa < 0 else kappa
    j = Rational(2 * abs(kappa) - 1, 2)
    m = Rational(m2, 2)
    terms = []
    for spin2 in (-1, 1):
        spin = Rational(spin2, 2)
        ml = m - spin
        if abs(ml) > ell or ml.q != 1:
            continue
        value = clebsch_gordan(S(ell), Rational(1, 2), j, ml, spin, m)
        if value != 0:
            terms.append((int(ml), spin2, float(value)))
    return tuple(terms)


@lru_cache(maxsize=None)
def _gaunt_y_conjugate(l_out: int, m_out: int, ell: int, q: int,
                       l_in: int, m_in: int) -> float:
    """Integral ``int Y_out* Y_(ell,q)* Y_in dOmega`` from Wigner 3j symbols."""
    if m_in - m_out - q != 0 or abs(l_out - l_in) > ell or ell > l_out + l_in:
        return 0.0
    if (l_out + ell + l_in) % 2:
        return 0.0
    prefactor = math.sqrt(
        (2*l_out + 1) * (2*ell + 1) * (2*l_in + 1) / (4.0 * math.pi)
    )
    g0 = float(wigner_3j(l_out, ell, l_in, 0, 0, 0))
    gm = float(wigner_3j(l_out, ell, l_in, -m_out, -q, m_in))
    return ((-1.0) ** (m_out + q)) * prefactor * g0 * gm


@lru_cache(maxsize=None)
def _gaunt_y(l_out: int, m_out: int, ell: int, q: int,
             l_in: int, m_in: int) -> float:
    """Integral ``int Y_out* Y_(ell,q) Y_in dOmega`` from Wigner 3j symbols."""
    if m_in - m_out + q != 0 or abs(l_out - l_in) > ell or ell > l_out + l_in:
        return 0.0
    if (l_out + ell + l_in) % 2:
        return 0.0
    prefactor = math.sqrt(
        (2*l_out + 1) * (2*ell + 1) * (2*l_in + 1) / (4.0 * math.pi)
    )
    g0 = float(wigner_3j(l_out, ell, l_in, 0, 0, 0))
    gm = float(wigner_3j(l_out, ell, l_in, -m_out, q, m_in))
    return ((-1.0) ** m_out) * prefactor * g0 * gm


def _multipole_transitions(basis, lmax: int) -> dict[str, np.ndarray]:
    """Build exact spin-resolved transition multipoles for each Dirac block."""
    nchannel = len(basis.channels)
    orders = [(ell, q) for ell in range(lmax + 1)
              for q in range(-ell, ell + 1)]
    result = {}
    for block in (0, 1):
        # [channel_out, channel_in, spin_out, spin_in, multipole_order]
        star_y = np.zeros((nchannel, nchannel, 2, 2, len(orders)), np.complex128)
        plain_y = np.zeros_like(star_y)
        for a, out in enumerate(basis.channels):
            kout = out.kappa if block == 0 else -out.kappa
            out_terms = _spinor_harmonic_terms(kout, out.m2)
            for c, inn in enumerate(basis.channels):
                kin = inn.kappa if block == 0 else -inn.kappa
                in_terms = _spinor_harmonic_terms(kin, inn.m2)
                for so in range(2):
                    # The sampled-component layout is (up, down), i.e.
                    # spin2=(+1,-1).
                    spin_out2 = 1 if so == 0 else -1
                    for si in range(2):
                        spin_in2 = 1 if si == 0 else -1
                        for idx, (ell, q) in enumerate(orders):
                            val_star = 0.0
                            val_plain = 0.0
                            for mo, mso, co in out_terms:
                                if mso != spin_out2:
                                    continue
                                for mi, msi, ci in in_terms:
                                    if msi != spin_in2:
                                        continue
                                    val_star += co * ci * _gaunt_y_conjugate(
                                        abs(kout) - 1 if kout < 0 else kout,
                                        mo, ell, q,
                                        abs(kin) - 1 if kin < 0 else kin,
                                        mi,
                                    )
                                    val_plain += co * ci * _gaunt_y(
                                        abs(kout) - 1 if kout < 0 else kout,
                                        mo, ell, q,
                                        abs(kin) - 1 if kin < 0 else kin,
                                        mi,
                                    )
                            star_y[a, c, so, si, idx] = val_star
                            plain_y[a, c, so, si, idx] = val_plain
        result[f"star_y_{block}"] = star_y
        result[f"plain_y_{block}"] = plain_y
    result["orders"] = np.asarray(orders, dtype=np.int64)
    return result


def _angular_pair_tensors(basis, lmax: int):
    """Return all-J direct/exchange factors for each endpoint block pair."""
    trans = _multipole_transitions(basis, lmax)
    orders = [tuple(map(int, item)) for item in trans["orders"]]
    nchannel = len(basis.channels)
    # [block_x, block_y, ell, out_x, in_x, out_y, in_y]
    direct = np.zeros((2, 2, lmax + 1, nchannel, nchannel,
                       nchannel, nchannel), np.complex128)
    exchange = np.zeros_like(direct)
    for bx in (0, 1):
        # exp(z cos gamma) = 4 pi sum_lm i_l(z) Y_lm(Omega_x)
        # Y_lm*(Omega_y); the x endpoint therefore uses the plain harmonic.
        tx = trans[f"plain_y_{bx}"]
        for by in (0, 1):
            uy = trans[f"star_y_{by}"]
            for order_index, (ell, q) in enumerate(orders):
                for a in range(nchannel):
                    for c in range(nchannel):
                        for b in range(nchannel):
                            for d in range(nchannel):
                                vd = 0.0j
                                vx = 0.0j
                                for sx in range(2):
                                    vd_x = tx[a, c, sx, sx, order_index]
                                    for sy in range(2):
                                        if vd_x != 0:
                                            vd += vd_x * uy[b, d, sy, sy, order_index]
                                        vx += (tx[a, c, sx, sy, order_index]
                                               * uy[b, d, sy, sx, order_index])
                                direct[bx, by, ell, a, c, b, d] += vd
                                exchange[bx, by, ell, a, c, b, d] += vx
    return direct, exchange


def _gaussian_radial_multipoles(radii: np.ndarray, width: float,
                                lmax: int) -> np.ndarray:
    """Radial coefficient of the Gaussian expansion, including its 4pi factor."""
    r = radii[:, None]
    rp = radii[None, :]
    z = 2.0 * r * rp / (width * width)
    base = np.exp(-((r - rp) / width) ** 2)
    radial = np.empty((lmax + 1, r.shape[0], rp.shape[1]), dtype=np.float64)
    for ell in range(lmax + 1):
        radial[ell] = 4.0 * math.pi * base * ive(ell + 0.5, z) * np.sqrt(
            math.pi / (2.0 * z)
        )
    return radial


def angular_pairing_reference(
    basis,
    q: torch.Tensor,
    chi: torch.Tensor,
    *,
    species: str,
    force: GognyD1S = D1S,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``(energy, active_gap, block_terms)`` by angular recoupling.

    ``block_terms`` contains unsymmetrized gap contributions for endpoint
    upper/lower pairs ``++``, ``+-``, ``-+``, ``--``.  Its sum is
    antisymmetrized at the same point as the production action; each pair is
    independently compared in tests before that final symmetrization.
    """
    if species not in {"n", "p", "neutron", "proton"}:
        raise ValueError("species must be n/neutron or p/proton")
    if q.dtype != torch.complex128 or chi.dtype != torch.complex128:
        raise TypeError("q and chi must use complex128")
    if q.shape[0] != basis.dimension or chi.shape != (q.shape[1], q.shape[1]):
        raise ValueError("q/chi do not match the primitive basis")
    if not torch.allclose(chi, -chi.T, atol=1e-10, rtol=0.0):
        raise ValueError("chi must be skew-symmetric")
    if force.strength != 1.0:
        raise ValueError("the blocked-RHFB reference fixes D1S strength to one")

    lmax = 2 * basis.kappa_max
    direct, exchange = _angular_pair_tensors(basis, lmax)
    nchannel = len(basis.channels)
    nr = basis.n_radial
    m = q.shape[1]
    coeff = q.reshape(nchannel, 2, nr, m)
    # transition[block, out_channel, in_channel, out_state, in_state, r]
    transition = torch.empty((2, nchannel, nchannel, m, m, nr),
                             dtype=q.dtype, device=q.device)
    # Out state is a, source state is c, and the radial factor is q_a^* q_c.
    for block in (0, 1):
        for p in range(nchannel):
            for t in range(nchannel):
                transition[block, p, t] = torch.einsum(
                    "ra,rc->acr", coeff[p, block].conj(), coeff[t, block]
                )

    radial_terms = []
    radii = basis.x.astype(np.float64)
    for width, w, b, h, mass_mev in zip(
        force.ranges_fm, force.w_mev, force.b_mev, force.h_mev, force.m_mev
    ):
        radial_terms.append((float(force.strength * (w-h)),
                             float(force.strength * (b-mass_mev)),
                             _gaussian_radial_multipoles(radii, width, lmax)))

    terms: dict[str, torch.Tensor] = {}
    zero = q.new_zeros((m, m))
    for bx, by, label in ((0, 0, "++"), (0, 1, "+-"),
                          (1, 0, "-+"), (1, 1, "--")):
        term = zero.clone()
        ang_dir = direct[bx, by]
        ang_ex = exchange[bx, by]
        for cdirect, cexchange, radial in radial_terms:
            for ell in range(lmax + 1):
                angular_d = torch.as_tensor(ang_dir[ell], dtype=q.dtype, device=q.device)
                angular_x = torch.as_tensor(ang_ex[ell], dtype=q.dtype, device=q.device)
                kr = torch.as_tensor(radial[ell], dtype=torch.float64, device=q.device)
                for r in range(nr):
                    for rp in range(nr):
                        weight = kr[r, rp]
                        if float(weight) == 0.0:
                            continue
                        x = transition[bx, :, :, :, :, r]
                        y = transition[by, :, :, :, :, rp]
                        factor = cdirect * angular_d + cexchange * angular_x
                        term = term + weight * torch.einsum(
                            "acug,gw,bdvw,acbd->uv", x, chi, y, factor
                        )
        terms[label] = term
    raw_gap = sum(terms.values(), zero.clone())
    gap = 0.5 * (raw_gap - raw_gap.T)
    energy = 0.5 * torch.sum(chi.conj() * gap).real
    return energy, gap, terms


def time_reversal_phase(kappa: int, m2: int) -> int:
    """Condon-Shortley phase ``(-1)^(j-m)`` for a Dirac spherical orbit."""
    exponent_numerator = 2 * abs(kappa) - 1 - m2
    if exponent_numerator % 2:
        raise ValueError("invalid half-integer m for kappa")
    return -1 if (exponent_numerator // 2) % 2 else 1


def time_reverse_coefficients(basis, q: torch.Tensor) -> torch.Tensor:
    """Apply ``T|kappa,m> = (-1)^(j-m)|kappa,-m>`` to radial factors."""
    if q.dtype != torch.complex128 or q.shape[0] != basis.dimension:
        raise ValueError("q must be complex128 and match the primitive basis")
    out = torch.zeros_like(q)
    for index, channel in enumerate(basis.channels):
        partner = basis.channel_index[(channel.kappa, -channel.m2)]
        # basis._spinor_harmonic uses a channel-dependent overall Condon
        # phase: i*sigma_y*K Omega_(kappa,m) = sign(kappa)*(-1)^(j-m)
        # Omega_(kappa,-m).  The same sign applies to lower components because
        # conjugating the explicit i in i*f*Omega_(-kappa) cancels the phase
        # reversal of the -kappa angular harmonic.
        phase = (1 if channel.kappa > 0 else -1) * time_reversal_phase(
            channel.kappa, channel.m2
        )
        src = q[channel.offset:channel.offset + 2*basis.n_radial]
        out_channel = basis.channels[partner]
        out[out_channel.offset:out_channel.offset + 2*basis.n_radial] = phase * src.conj()
    return out
