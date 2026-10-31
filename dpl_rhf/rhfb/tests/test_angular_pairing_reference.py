from __future__ import annotations

import torch

from dpl_rhf.rhfb.angular_pairing_reference import (
    _angular_pair_tensors,
    angular_pairing_reference,
    time_reverse_coefficients,
)
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.gaussian_pairing import GaussianPairingAction
from dpl_rhf.rhfb.pairing import D1S, _spin_exchange_indices, pairing_gap_from_sampler


def _small_basis():
    # Resolves the full L<=2 source bandwidth while keeping the explicit
    # coordinate-pair block comparison small.
    return AngularRadialBasis(
        radius_fm=1.75, spacing_fm=0.25, kappa_max=1,
        n_theta=6, n_phi=12,
    )


def _coordinate_block_pair(basis, q, chi, bx, by, chunk=32):
    """Direct sampled-coordinate contribution for a fixed endpoint block pair."""
    device = q.device
    values = basis._sample_range(q, 0, basis.point_count)
    xspin = values.clone()
    yspin = values.clone()
    xspin[:, 2:] = 0
    yspin[:, 2:] = 0
    if bx:
        xspin[:, :2] = 0
        xspin[:, 2:] = values[:, 2:]
    if by:
        yspin[:, :2] = 0
        yspin[:, 2:] = values[:, 2:]
    positions = torch.as_tensor(basis.positions, dtype=torch.float64, device=device)
    weights = torch.as_tensor(basis.weights, dtype=torch.float64, device=device)
    spin_a, spin_b = _spin_exchange_indices(device)
    ranges = torch.as_tensor(D1S.ranges_fm, dtype=torch.float64, device=device)
    direct_coeff = torch.as_tensor(
        [w-h for w, h in zip(D1S.w_mev, D1S.h_mev)], dtype=torch.float64, device=device
    )
    exchange_coeff = torch.as_tensor(
        [b-m for b, m in zip(D1S.b_mev, D1S.m_mev)], dtype=torch.float64, device=device
    )
    output = q.new_zeros((q.shape[1], q.shape[1]))
    for p0 in range(0, basis.point_count, chunk):
        p1 = min(p0+chunk, basis.point_count)
        qp = xspin[p0:p1]
        for r0 in range(0, basis.point_count, chunk):
            r1 = min(r0+chunk, basis.point_count)
            qr = yspin[r0:r1]
            qamp = torch.einsum("pia,ab,qjb->pqij", qp, chi, qr)
            d2 = (positions[p0:p1, None] - positions[None, r0:r1]).square().sum(-1)
            gauss = torch.exp(-d2[..., None]/ranges.square())
            kd = torch.sum(gauss*direct_coeff, dim=-1)
            kx = torch.sum(gauss*exchange_coeff, dim=-1)
            acted = kd[..., None, None]*qamp + kx[..., None, None]*qamp[..., spin_a, spin_b]
            wpq = (weights[p0:p1, None]*weights[None, r0:r1]).to(q.dtype)
            output += torch.einsum("pia,qjb,pqij,pq->ab", qp.conj(), qr.conj(), acted, wpq)
    return output


def test_appendix_b_angular_oracle_matches_each_dirac_endpoint_block():
    torch.manual_seed(418)
    basis = _small_basis()
    raw = torch.randn((basis.dimension, 3), dtype=torch.complex128)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    chi = torch.tensor(
        [[0, .18+.07j, -.03j], [-.18-.07j, 0, .11], [.03j, -.11, 0]],
        dtype=torch.complex128,
    )
    _, _, angular_terms = angular_pairing_reference(basis, q, chi, species="n")

    for bx, by, label in ((0, 0, "++"), (0, 1, "+-"),
                          (1, 0, "-+"), (1, 1, "--")):
        direct_term = _coordinate_block_pair(basis, q, chi, bx, by)
        assert torch.allclose(angular_terms[label], direct_term, atol=4e-5, rtol=4e-5), label


def test_all_j_reference_includes_a_pure_j1_pair_and_nonzero_higher_multipoles():
    basis = AngularRadialBasis(
        radius_fm=1.75, spacing_fm=0.25, kappa_max=1,
        n_theta=8, n_phi=16,
    )
    # Both particles have m=+1/2, so M=1 forces the antisymmetric pair into
    # total J=1; a J=0 projection would annihilate this anomalous component.
    q = torch.zeros((basis.dimension, 2), dtype=torch.complex128)
    for state, kappa in enumerate((-1, 1)):
        channel = basis.channels[basis.channel_index[(kappa, 1)]]
        q[channel.offset+2, state] = 1.0
    chi = torch.tensor([[0, .2], [-.2, 0]], dtype=torch.complex128)
    reference_energy, reference_gap, _ = angular_pairing_reference(
        basis, q, chi, species="n"
    )
    action = GaussianPairingAction(
        basis, q, chi, species="n", lmax=2, chunk_size=1,
        checkpoint_blocks=False,
    )
    action_energy, action_gap = action.energy_and_active_gap()
    assert abs(float(reference_energy)) > 1e-3
    assert torch.linalg.vector_norm(reference_gap) > 1e-3
    assert torch.allclose(action_energy, reference_energy, atol=2e-12, rtol=2e-12)
    assert torch.allclose(action_gap, reference_gap, atol=2e-12, rtol=2e-12)

    # The full kappa=1 space has upper s/p and lower p/s spinor harmonics.
    # B5-B10's sum over allowed angular ranks is visibly not lambda=0 only.
    direct, exchange = _angular_pair_tensors(basis, 2)
    assert float(abs(direct[:, :, 1]).max()) > 1e-2
    assert float(abs(exchange[:, :, 2]).max()) > 1e-2


def test_all_j_angular_oracle_matches_production_action_and_energy_normalization():
    torch.manual_seed(913)
    basis = _small_basis()
    raw = torch.randn((basis.dimension, 3), dtype=torch.complex128)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    chi = torch.tensor(
        [[0, .14+.05j, -.02], [-.14-.05j, 0, .09j], [.02, -.09j, 0]],
        dtype=torch.complex128,
    )
    reference_energy, reference_gap, _ = angular_pairing_reference(
        basis, q, chi, species="proton"
    )
    action = GaussianPairingAction(
        basis, q, chi, species="p", lmax=2, chunk_size=2,
        active_chunk_size=2, checkpoint_blocks=False,
    )
    action_energy, action_gap = action.energy_and_active_gap()

    def sampler(start, stop):
        return (
            basis._sample_range(q, start, stop),
            torch.as_tensor(basis.weights[start:stop], dtype=torch.float64),
            torch.as_tensor(basis.positions[start:stop], dtype=torch.float64),
        )

    point_energy, point_gap = pairing_gap_from_sampler(
        chi, basis.point_count, sampler, species="proton", chunk_size=32,
        checkpoint_blocks=False,
    )
    assert torch.allclose(action_gap, reference_gap, atol=2e-8, rtol=2e-8)
    assert torch.allclose(action_energy, reference_energy, atol=2e-8, rtol=2e-8)
    # The independent point-pair path has a finite angular quadrature error;
    # its 6x12 rule resolves all basis products and is a separate kernel path.
    assert torch.allclose(point_gap, reference_gap, atol=8e-4, rtol=8e-4)
    assert torch.allclose(point_energy, reference_energy, atol=8e-4, rtol=8e-4)
    assert torch.allclose(
        reference_energy,
        0.5*torch.sum(chi.conj()*reference_gap).real,
        atol=2e-12, rtol=2e-12,
    )

    # On the skew subspace, both occurrences of chi in
    # E=1/2 chi* Delta[chi] contribute; their equality leaves dE=Re(Delta* dchi).
    direction = torch.randn(chi.shape, dtype=torch.complex128)
    direction = direction-direction.T
    step = 2e-6
    plus = angular_pairing_reference(
        basis, q, chi+step*direction, species="proton"
    )[0]
    minus = angular_pairing_reference(
        basis, q, chi-step*direction, species="proton"
    )[0]
    finite_difference = (plus-minus)/(2*step)
    pairing_derivative = torch.sum(reference_gap.conj()*direction).real
    assert torch.allclose(finite_difference, pairing_derivative, atol=2e-8, rtol=2e-8)


def test_time_reversal_matches_i_sigma_y_conjugation_with_i_f_convention():
    torch.manual_seed(73)
    basis = _small_basis()
    q = torch.randn((basis.dimension, 2), dtype=torch.complex128)
    sampled = basis._sample_range(q, 0, basis.point_count)
    reversed_sampled = basis._sample_range(time_reverse_coefficients(basis, q), 0, basis.point_count)

    expected = torch.empty_like(reversed_sampled)
    # T=i sigma_y K in the code's (up,down) component ordering.
    for up, down in ((0, 1), (2, 3)):
        expected[:, up] = sampled[:, down].conj()
        expected[:, down] = -sampled[:, up].conj()
    assert torch.allclose(reversed_sampled, expected, atol=2e-12, rtol=2e-12)
    twice = time_reverse_coefficients(basis, time_reverse_coefficients(basis, q))
    assert torch.allclose(twice, -q, atol=2e-12, rtol=2e-12)

    # Under the antiunitary time reversal, kappa transforms as chi* in the
    # time-reversed orbital columns. The even D1S force preserves E_pair.
    raw = torch.randn((basis.dimension, 3), dtype=torch.complex128)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    chi = torch.tensor(
        [[0, .14+.03j, -.02], [-.14-.03j, 0, .09j], [.02, -.09j, 0]],
        dtype=torch.complex128,
    )
    energy, gap, _ = angular_pairing_reference(basis, q, chi, species="n")
    reversed_energy, reversed_gap, _ = angular_pairing_reference(
        basis, time_reverse_coefficients(basis, q), chi.conj(), species="n"
    )
    assert torch.allclose(reversed_energy, energy, atol=2e-12, rtol=2e-12)
    assert torch.allclose(reversed_gap, gap.conj(), atol=2e-12, rtol=2e-12)
