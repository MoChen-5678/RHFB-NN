"""Spherical J=0 D1S pairing against the independent all-J m-scheme oracle."""
import math

import torch

from dpl_rhf.rhf.pairing import (
    D1S_PAIRING,
    GognyD1SPairingForce,
    SphericalD1SPairing,
    angular_coefficients,
)
from dpl_rhf.rhfb.angular_pairing_reference import time_reverse_coefficients
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.pairing import GognyD1S, pairing_gap


# Longer ranges keep the m-scheme angular quadrature of the Gaussian accurate,
# so the comparison tests the angular reduction and normalization, not the
# finite angular mesh. The four exchange strengths are the D1S values.
_RANGES = (1.6, 2.4)


def _radial(basis, kappa, width, lower_scale):
    r = torch.as_tensor(basis.r, dtype=torch.float64)
    l_upper = -kappa - 1 if kappa < 0 else kappa
    upper = r.pow(l_upper + 1) * torch.exp(-r.square() / (2 * width**2))
    lower = lower_scale * r * upper
    upper, lower = upper[1:-1], lower[1:-1]
    norm = torch.sqrt(torch.sum(upper.square() + lower.square()))
    return torch.cat([upper, lower]) / norm


def _m_scheme_energy(basis, blocks):
    positive, weights = [], []
    for kappa, radial, uv in blocks:
        twice_j = 2 * abs(kappa) - 1
        for m2 in range(1, twice_j + 1, 2):
            channel = basis.channels[basis.channel_index[(kappa, m2)]]
            column = torch.zeros(basis.dimension, dtype=torch.complex128)
            column[channel.offset:channel.offset + 2 * basis.n_radial] = radial
            positive.append(column)
            weights.append(uv)
    q_plus = torch.stack(positive, dim=1)
    q_bar = time_reverse_coefficients(basis, q_plus)
    q = torch.cat([q_plus, q_bar], dim=1)
    pairs = torch.diag(torch.tensor(weights, dtype=torch.complex128))
    zero = torch.zeros_like(pairs)
    # BCS vacuum prod_m (u + v c+_m c+_mbar): chi_{m,mbar} = <c_mbar c_m> = uv.
    chi = torch.cat([torch.cat([zero, pairs], 1), torch.cat([-pairs, zero], 1)], 0)
    spinors, point_weights, positions = basis.sample(q)
    force = GognyD1S(ranges_fm=_RANGES)
    energy, _ = pairing_gap(spinors, chi, positions, point_weights,
                            species="neutron", chunk_size=8, force=force)
    return float(energy.real)


def _spherical_energy(basis, blocks, convention="time_reversal"):
    r = torch.as_tensor(basis.r, dtype=torch.float64)
    force = GognyD1SPairingForce(ranges_fm=_RANGES)
    # The m-scheme basis integrates radially with uniform weights h*x^2; use
    # the same radial rule so only the angular quadrature differs.
    uniform = torch.full((basis.n_radial,), basis.spacing_fm, dtype=torch.float64)
    pairing = SphericalD1SPairing(r, [k for k, _, _ in blocks], force=force,
                                  radial_weights=uniform, convention=convention)
    tensors = []
    for kappa, radial, uv in blocks:
        weights = torch.tensor([2 * abs(kappa) * uv], dtype=torch.float64)
        tensors.append((kappa, pairing.pairing_tensor(radial[:, None], weights, pairing.h)))
    return float(pairing.energy(tensors))


def _basis(n_theta, n_phi):
    return AngularRadialBasis(radius_fm=6.0, spacing_fm=0.4, kappa_max=2,
                              n_theta=n_theta, n_phi=n_phi)


def test_single_s_shell_matches_m_scheme_including_lower_components():
    basis = _basis(12, 24)
    blocks = [(-1, _radial(basis, -1, 1.4, 0.08), 0.43)]
    m_scheme, spherical = _m_scheme_energy(basis, blocks), _spherical_energy(basis, blocks)
    assert abs(m_scheme) > 1e-3
    assert math.isclose(spherical, m_scheme, rel_tol=1e-8)


def test_mixed_kappa_cross_terms_match_m_scheme_time_reversed_pairs():
    basis = _basis(12, 24)
    blocks = [
        (-1, _radial(basis, -1, 1.3, 0.05), 0.37),
        (-2, _radial(basis, -2, 1.5, 0.07), 0.21),
        (1, _radial(basis, 1, 1.6, 0.06), 0.29),
        (2, _radial(basis, 2, 1.7, 0.04), 0.18),
    ]
    m_scheme = _m_scheme_energy(basis, blocks)
    spherical = _spherical_energy(basis, blocks)
    assert math.isclose(spherical, m_scheme, rel_tol=1e-7)
    # The reference program's J=0 phase flips cross terms between j=1/2 and
    # j=3/2 blocks; the diagonal blocks are convention independent.
    reference = _spherical_energy(basis, blocks, convention="reference")
    assert abs(reference - m_scheme) > 1e-2


def test_coefficients_follow_reference_c3j_definition():
    # 1s1/2 x 1s1/2 at L=0: (0 0 0;000)^2 {0 0 0;1/2 1/2 1/2}^2 = 1 * 1/2.
    c3j, c3l, c3s = angular_coefficients(-1, -1, 0)
    assert math.isclose(c3j, 0.5, rel_tol=1e-14)
    assert math.isclose(c3l, 1.0, rel_tol=1e-14)
    # Lower component of s1/2 is l'=1: (1 1 0;000)^2 = 1/3.
    assert math.isclose(c3s, 1.0 / 3.0, rel_tol=1e-14)


def test_energy_is_quadratic_in_kappa_and_zero_without_pairs():
    r = torch.arange(121, dtype=torch.float64) * 0.1
    r[0] = 0.001
    pairing = SphericalD1SPairing(r, [-1, -3], force=D1S_PAIRING)
    torch.manual_seed(7)
    factors = torch.linalg.qr(torch.randn(2 * 119, 3, dtype=torch.float64))[0]
    weights = torch.tensor([0.3, 0.1, 0.2], dtype=torch.float64)
    tensor = pairing.pairing_tensor(factors, weights, pairing.h)
    e1 = pairing.energy([(-3, tensor)])
    e2 = pairing.energy([(-3, 2.0 * tensor)])
    assert math.isclose(float(e2), 4.0 * float(e1), rel_tol=1e-12)
    assert float(pairing.energy([(-3, 0.0 * tensor)])) == 0.0
    assert float(pairing.energy([])) == 0.0


def test_batched_pairing_energy_matches_reference_loop():
    import torch as _torch

    from dpl_rhf.rhf.pairing import SphericalD1SPairing

    r = _torch.arange(81, dtype=_torch.float64) * 0.1
    r[0] = 0.001
    kappas = [-1, 1, -2, 2, -3]
    pairing = SphericalD1SPairing(r, kappas)
    generator = _torch.Generator().manual_seed(11)
    size = r.numel() - 2
    tensors = []
    for kappa in (-1, 2, -3):
        a = _torch.randn(2 * size, 2 * size, dtype=_torch.float64, generator=generator)
        matrix = 0.5 * (a + a.T)
        tensors.append((kappa, _torch.stack([
            _torch.stack([matrix[:size, :size], matrix[:size, size:]]),
            _torch.stack([matrix[size:, :size], matrix[size:, size:]]),
        ]).requires_grad_(True)))
    batched = pairing.energy(tensors)
    reference = pairing.energy_reference(tensors)
    assert _torch.allclose(batched, reference, rtol=1.0e-12, atol=1.0e-10)
    g_batched = _torch.autograd.grad(batched, [t for _, t in tensors])
    g_reference = _torch.autograd.grad(reference, [t for _, t in tensors])
    for left, right in zip(g_batched, g_reference):
        assert _torch.allclose(left, right, rtol=1.0e-12, atol=1.0e-12)
