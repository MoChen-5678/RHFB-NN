import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.manifold import slater_reference
from dpl_rhf.rhfb.normal_action import AutogradNormalAction
from dpl_rhf.rhfb.operators import DensityFactors


def example():
    basis = AngularRadialBasis(radius_fm=8., spacing_fm=1., kappa_max=1)
    q, _ = basis.build_initial_q(1, minimum=2, empty_states=0)
    reference = slater_reference(torch.tensor([1., 0.], dtype=torch.float64))
    state = DensityFactors(q, reference.rho, reference.chi)
    return basis, {'n': state, 'p': state}


def test_normal_action_matches_dense_independent_density_derivative_and_external_vectors(monkeypatch):
    basis, states = example()
    torch.manual_seed(902)
    d = basis.dimension
    h0 = torch.randn(d, d, dtype=torch.complex128)
    h0 = .5*(h0+h0.mH)
    strength = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
    def normal_energy(_basis, current):
        rhos = {s: f.q @ f.rho_active @ f.q.mH for s, f in current.items()}
        total = rhos['n']+rhos['p']
        return sum(torch.trace(h0 @ rho).real for rho in rhos.values()) + strength*torch.trace(total @ total).real/2
    rho = sum(f.q @ f.rho_active @ f.q.mH for f in states.values())
    expected_h = h0+strength*rho
    vectors = torch.randn(d, 5, dtype=torch.complex128)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('no training eigensolver may be used')
    for name in ('eig', 'eigh', 'eigvalsh'):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    action = AutogradNormalAction(basis, states, normal_energy, species='n', vector_chunk=2)
    actual = action.apply(vectors)
    torch.testing.assert_close(actual, expected_h @ vectors, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(action.apply(vectors[:, 0]), actual[:, 0], atol=1e-12, rtol=1e-12)
    assert torch.linalg.matrix_norm(actual-states['n'].q @ (states['n'].q.mH @ actual)) > 1.
    deriv, = torch.autograd.grad(actual.abs().square().sum(), strength)
    oracle, = torch.autograd.grad((expected_h @ vectors).abs().square().sum(), strength)
    torch.testing.assert_close(deriv, oracle, atol=1e-10, rtol=1e-12)


def test_normal_action_uses_full_local_density_dependent_rearrangement():
    basis, states = example()
    def density_energy(basis, current):
        result = torch.zeros((), dtype=torch.float64)
        for state in current.values():
            samples, weights, _ = basis.sample(state.q)
            rho = torch.einsum('ab,pia,pib->p', state.rho_active, samples, samples.conj()).real
            result = result+(weights*rho.pow(3)).sum()/3
        return result
    q, weights, _ = basis.sample(states['n'].q)
    rho = torch.einsum('ab,pia,pib->p', states['n'].rho_active, q, q.conj()).real
    torch.manual_seed(44)
    v = torch.randn(basis.dimension, 2, dtype=torch.complex128)
    samples, _, _ = basis.sample(v)
    oracle = basis.project_sample_range(rho.square()[:, None, None]*samples, 0, basis.point_count)
    action = AutogradNormalAction(basis, states, density_energy, species='n', vector_chunk=1)
    with torch.no_grad():
        actual = action.apply(v)
    torch.testing.assert_close(actual, oracle, atol=1e-12, rtol=1e-12)
