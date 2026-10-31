import torch

from dpl_rhf.rhfb.fields import fit_number_tangent_chemical_potential, make_nambu_actions
from dpl_rhf.rhfb.operators import DensityFactors, NambuAction, stationarity_metrics
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.gaussian_pairing import GaussianPairingAction
from dpl_rhf.rhfb.factorized_commutator import FactorizedCommutator
from dpl_rhf.rhfb.manifold import slater_reference


def test_number_tangent_fit_recovers_chemical_potential_for_known_paired_stationary_state():
    torch.manual_seed(500)
    q, _ = torch.linalg.qr(torch.randn(7, 3, dtype=torch.complex128))
    rho = torch.diag(torch.tensor([1., .3, .3], dtype=torch.complex128))
    chi = torch.zeros_like(rho)
    chi[1, 2] = .21**.5
    chi[2, 1] = -chi[1, 2]
    state = DensityFactors(q, rho, chi)
    expected_mu, scale = -2.4, 1.8
    h = expected_mu*torch.eye(7, dtype=torch.complex128)+scale*(2*q@rho@q.mH-torch.eye(7, dtype=torch.complex128))
    delta = 2*scale*q@chi@q.mT
    action = NambuAction(7, lambda v: h@v, lambda v: delta@v)
    mu, identified = fit_number_tangent_chemical_potential(action, state)
    assert identified
    assert abs(float(mu)-expected_mu) < 1e-12
    cached = FactorizedCommutator(state, h@q, lambda v: delta@v)
    cached_mu, cached_identified = fit_number_tangent_chemical_potential(
        action, state, cached_commutator=cached)
    assert cached_identified
    torch.testing.assert_close(cached_mu, mu)
    action.chemical_potential = mu
    assert stationarity_metrics(action, state, chunk_size=3)['full_rms_mev'] < 1e-12


def test_zero_pairing_number_gauge_is_unidentifiable_and_full_actions_are_hermitian():
    basis = AngularRadialBasis(radius_fm=8., spacing_fm=1., kappa_max=1)
    q, _ = basis.build_initial_q(1, minimum=2, empty_states=0)
    gd = slater_reference(torch.tensor([1., 0.], dtype=torch.float64))
    state = DensityFactors(q, gd.rho, gd.chi)
    states = {'n': state, 'p': state}
    def energy(basis, current):
        return sum(torch.trace(s.rho_active@(s.q.mH@basis.apply_kinetic(s.q, species=sp))).real
                   for sp, s in current.items())
    actions = make_nambu_actions(basis, states, energy, chunk_size=42, vector_chunk=4,
                                 create_graph=False)
    action = actions['n']
    assert isinstance(action.delta_apply.__self__, GaussianPairingAction)
    mu, identified = fit_number_tangent_chemical_potential(action, state)
    assert not identified and mu == 0.
    torch.manual_seed(273)
    v = torch.randn(2*basis.dimension, 2, dtype=torch.complex128)
    w = torch.randn(2*basis.dimension, 2, dtype=torch.complex128)
    torch.testing.assert_close((v.conj()*action.apply(w)).sum(),
                               (action.apply(v).conj()*w).sum(), atol=2e-10, rtol=1e-12)
