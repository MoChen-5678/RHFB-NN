"""Independent coordinate reflection and Fock-space checks of spatial parity."""
import numpy as np
import pytest
import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.manifold import slater_reference
from dpl_rhf.rhfb.operators import DensityFactors
from dpl_rhf.rhfb.spatial_parity import (ParityConservingDensityNet,
    primitive_spatial_parity, parity_adapted_density_diagnostic)


def ingredients():
    basis = AngularRadialBasis(radius_fm=8., spacing_fm=1., kappa_max=2,
                               derivative_order=5)
    q, labels = basis.build_initial_q(2, minimum=6, empty_states=0)
    return basis, q, labels


def test_dirac_beta_reflection_has_same_parity_on_both_radial_components():
    basis, _, _ = ingredients()
    rows = primitive_spatial_parity(basis)
    beta = torch.tensor([1., 1., -1., -1.], dtype=torch.float64)
    # Check actual complex coordinate spinors rather than duplicating the
    # kappa-to-upper-l formula that assigns primitive parity.
    for channel in basis.channels:
        q = torch.zeros((basis.dimension, 1), dtype=torch.complex128)
        q[channel.offset:channel.offset+2*basis.n_radial, 0] = torch.arange(
            1, 2*basis.n_radial+1, dtype=torch.float64)*(1+.3j)
        psi = basis.sample(q)[0].reshape(basis.n_radial, basis.n_theta, basis.n_phi, 4, 1)
        opposite_phi = (torch.arange(basis.n_phi)+basis.n_phi//2)%basis.n_phi
        reflected = psi.flip(1)[:, :, opposite_phi]*beta[None, None, None, :, None]
        torch.testing.assert_close(reflected, rows[channel.offset]*psi,
                                   atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize('blocked', [0, 2])
def test_spatial_parity_pfaffian_agrees_with_explicit_paired_fock_state(blocked):
    basis, q, _ = ingredients()
    m, dim = q.shape[1], 2**q.shape[1]
    annihilate = []
    for i in range(m):
        a = np.zeros((dim, dim), dtype=np.complex128)
        for mask in range(dim):
            if mask & (1 << i):
                sign = (-1)**((mask & ((1 << i)-1)).bit_count())
                a[mask ^ (1 << i), mask] = sign
        annihilate.append(a)
    # An explicitly odd Gaussian state with a negative-parity BCS pair.
    state = np.zeros(dim, dtype=np.complex128)
    state[1 << blocked] = .8
    state[(1 << blocked)|(1 << 3)|(1 << 4)] = .6*np.exp(.4j)
    rho = np.array([[np.vdot(annihilate[j]@state, annihilate[i]@state)
                     for j in range(m)] for i in range(m)])
    chi = np.array([[np.vdot(state, annihilate[j]@annihilate[i]@state)
                     for j in range(m)] for i in range(m)])
    factors = DensityFactors(q, torch.from_numpy(rho), torch.from_numpy(chi))
    # Shell construction here is s(+,+), p(-,-,-,-). This direct Fock
    # operator expectation is independent of the density Pfaffian formula.
    negative_mask = sum(1 << i for i in range(2, m))
    fock_parity = sum(abs(state[mask])**2*(-1)**((mask & negative_mask).bit_count())
                      for mask in range(dim))
    result = parity_adapted_density_diagnostic(basis, factors)
    assert result['many_body_spatial_parity'] == round(fock_parity)
    assert result['many_body_spatial_parity'] == (1 if blocked == 0 else -1)


def test_network_keeps_spatial_parity_and_learns_outside_initial_span():
    basis, q, labels = ingredients()
    occupations = torch.tensor([1., 1., 0., 0., 0., 0.], dtype=torch.float64)
    reference = slater_reference(occupations, blocked_index=2, blocking_operation='particle')
    net = ParityConservingDensityNet(basis, q, labels, reference,
        blocked_index=2, hidden=8, depth=1, radial_basis_size=4)
    with torch.no_grad():
        net.raw_a.normal_(0., .08)
        net.raw_b.normal_(0., .09)
        net.raw_coordinates.normal_(0., .04)
        net.output_head.weight.normal_(0., .02)
    actual_q, gd, mode = net.forward_with_blocked_mode()
    factors = DensityFactors(actual_q, gd.rho, gd.chi)
    diagnostic = parity_adapted_density_diagnostic(basis, factors)
    assert diagnostic['many_body_spatial_parity'] == -1
    assert max(diagnostic[k] for k in ('frame_parity_max_error',
        'normal_opposite_parity_max_error', 'anomalous_opposite_parity_max_error')) < 1e-12
    assert float(torch.linalg.vector_norm(mode).detach()) == pytest.approx(1., abs=2e-12)
    target = torch.randn_like(actual_q)
    target *= net.primitive_parities[:, None] == net.orbital_parities[None, :]
    external = target-q@(q.mH@target)
    loss = (actual_q.conj()*external).sum().real + gd.chi.abs().square().sum()
    loss.backward()
    gradient = net.raw_coordinates.grad
    assert torch.linalg.vector_norm(gradient-q@(q.mH@gradient)) > 1e-7
    assert torch.linalg.vector_norm(net.output_head.weight.grad) > 1e-7
    assert torch.linalg.vector_norm(net.radial_encoder[0].weight.grad) > 1e-9
    assert torch.count_nonzero(gradient[
        net.primitive_parities[:, None] != net.orbital_parities[None, :]]) == 0


def test_parity_checkpoint_cannot_be_loaded_as_unrestricted_network():
    from dpl_rhf.rhfb.network import GeneralizedDensityNet
    basis, q, labels = ingredients()
    reference = slater_reference(torch.tensor([1., 1., 0., 0., 0., 0.], dtype=torch.float64))
    kwargs = dict(hidden=8, depth=1, radial_basis_size=4)
    restricted = ParityConservingDensityNet(basis, q, labels, reference, **kwargs)
    general = GeneralizedDensityNet(basis, q, labels, reference, **kwargs)
    with pytest.raises(RuntimeError):
        general.load_state_dict(restricted.state_dict())
