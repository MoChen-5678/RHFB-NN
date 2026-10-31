from __future__ import annotations

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.gaussian_pairing import GaussianGreen, GaussianPairingAction
from dpl_rhf.rhfb.pairing_action import D1SPairingAction
from dpl_rhf.rhfb.pairing import pairing_gap_from_sampler


def _basis():
    # Resolves all products through l=2 for kappa_max=1 and gives a dense
    # angular quadrature so the point-pair oracle tests angular convergence.
    return AngularRadialBasis(radius_fm=1.0, spacing_fm=0.125, kappa_max=1,
                              n_theta=10, n_phi=21)


def test_gaussian_multipoles_converge_to_direct_point_kernel():
    basis = _basis()
    green = GaussianGreen(basis, lmax=2)
    radius = torch.as_tensor(basis.x, dtype=torch.float64)
    angular = torch.as_tensor(basis.angular_weights, dtype=torch.float64)
    theta = torch.as_tensor(basis.theta, dtype=torch.float64)
    source = (radius[:, None] ** 2 * torch.cos(theta)[None, :] ** 2).reshape(-1)
    approximate = green.apply_gaussian(source, 0.7)

    positions = torch.as_tensor(basis.positions, dtype=torch.float64)
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    exact = torch.zeros_like(source)
    for start in range(0, basis.point_count, 64):
        stop = min(start + 64, basis.point_count)
        distance2 = (positions[start:stop, None] - positions[None, :]).square().sum(-1)
        kernel = torch.exp(-distance2 / 0.7**2)
        exact[start:stop] = kernel @ (source * weights)
    relative = torch.linalg.vector_norm(approximate - exact) / torch.linalg.vector_norm(exact)
    assert float(relative) < 2.0e-3


def test_full_space_gaussian_gap_action_matches_direct_two_gaussian_sum():
    torch.manual_seed(19)
    basis = _basis()
    qraw = torch.randn((basis.dimension, 2), dtype=torch.complex128)
    q, _ = torch.linalg.qr(qraw, mode="reduced")
    chi = torch.tensor([[0.0, 0.2 + 0.1j], [-0.2 - 0.1j, 0.0]], dtype=torch.complex128)
    vector = torch.randn(basis.dimension, dtype=torch.complex128)
    direct = D1SPairingAction(basis, q, chi, species="n", chunk_size=64,
                              checkpoint_blocks=False)
    multipole = GaussianPairingAction(basis, q, chi, species="n", chunk_size=1,
                                      lmax=2, checkpoint_blocks=False)
    expected = direct.apply(vector)
    actual = multipole.apply(vector)
    relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    assert float(relative) < 3.0e-3

    def sampler(start, stop):
        return (basis._sample_range(q, start, stop),
                torch.as_tensor(basis.weights[start:stop], dtype=torch.float64),
                torch.as_tensor(basis.positions[start:stop], dtype=torch.float64))

    expected_energy, expected_gap = pairing_gap_from_sampler(
        chi, basis.point_count, sampler, species="neutron", chunk_size=64,
        checkpoint_blocks=False,
    )
    actual_energy, actual_gap = multipole.energy_and_active_gap()
    assert torch.allclose(actual_gap, expected_gap, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(actual_energy, expected_energy, atol=2.0e-5, rtol=2.0e-5)


def test_gap_action_matches_direct_external_vector_gradients():
    torch.manual_seed(31)
    basis = AngularRadialBasis(radius_fm=1.0, spacing_fm=0.125, kappa_max=1,
                               n_theta=4, n_phi=8)
    qraw = torch.randn((basis.dimension, 2), dtype=torch.complex128, requires_grad=True)
    q, _ = torch.linalg.qr(qraw, mode="reduced")
    chi = torch.tensor([[0.0, 0.3 - 0.2j], [-0.3 + 0.2j, 0.0]],
                       dtype=torch.complex128, requires_grad=True)
    vector = torch.randn(basis.dimension, dtype=torch.complex128, requires_grad=True)
    direct = D1SPairingAction(basis, q, chi, species="p", chunk_size=64,
                              checkpoint_blocks=False)
    multipole = GaussianPairingAction(basis, q, chi, species="p", chunk_size=1,
                                      active_chunk_size=1, lmax=2,
                                      checkpoint_blocks=False)
    expected, actual = direct.apply(vector), multipole.apply(vector)
    relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    assert float(relative) < 1.0e-2
    probe = torch.randn(basis.dimension, dtype=torch.complex128)
    variables = (qraw, chi, vector)
    direct_grad = torch.autograd.grad(torch.vdot(probe, expected).real, variables,
                                      retain_graph=True)
    multipole_grad = torch.autograd.grad(torch.vdot(probe, actual).real, variables)
    for direct_component, multipole_component in zip(direct_grad, multipole_grad):
        grad_error = torch.linalg.vector_norm(direct_component - multipole_component)
        grad_scale = torch.linalg.vector_norm(direct_component).clamp_min(1.0e-12)
        assert float(grad_error / grad_scale) < 2.0e-2


def test_gaussian_green_rejects_underresolved_angular_cutoff():
    basis = AngularRadialBasis(radius_fm=1.0, spacing_fm=0.125, kappa_max=1,
                               n_theta=3, n_phi=6)
    try:
        GaussianGreen(basis, lmax=3)
    except ValueError as exc:
        assert "cannot resolve" in str(exc)
    else:
        raise AssertionError("underresolved Gaussian angular expansion was accepted")
