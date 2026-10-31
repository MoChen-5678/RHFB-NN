import torch

from dpl_rhf.rhfb.factorized_commutator import FactorizedCommutator
from dpl_rhf.rhfb.manifold import slater_reference, transform_reference
from dpl_rhf.rhfb.operators import (DensityFactors, NambuAction,
                                    commutator_action, commutator_metrics)


def _state(*, variables=False):
    torch.manual_seed(1049)
    d, m = 9, 4
    qraw = torch.randn(d, m, dtype=torch.complex128, requires_grad=variables)
    q, _ = torch.linalg.qr(qraw)
    a = torch.randn(m, m, dtype=torch.complex128, requires_grad=variables)
    b = torch.randn(m, m, dtype=torch.complex128, requires_grad=variables)
    reference = slater_reference(torch.tensor([1., 1., 0., 0.], dtype=torch.float64))
    gd = transform_reference(reference, .06*(a-a.mH), .09*(b-b.mT))
    state = DensityFactors(q, gd.rho, gd.chi)
    return (state, (qraw, a, b)) if variables else state


def test_cached_normal_field_gives_exact_commutator_on_full_primitive_space():
    state = _state()
    d = state.dimension
    h0 = torch.randn(d, d, dtype=torch.complex128)
    h = .5*(h0+h0.mH)
    delta0 = torch.randn(d, d, dtype=torch.complex128)
    delta = .5*(delta0-delta0.mT)
    mu = .37
    action = NambuAction(d, lambda v: h@v, lambda v: delta@v, mu)
    cached = FactorizedCommutator(state, h@state.q, lambda v: delta@v,
                                  chemical_potential=mu)
    unit = torch.eye(2*d, dtype=torch.complex128)
    actual = cached.apply(unit)
    expected = commutator_action(action, state, unit)
    torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(cached.apply(unit[:, 3]), expected[:, 3])
    metrics = commutator_metrics(cached.apply, state, chunk_size=3)
    torch.testing.assert_close(metrics['full_frobenius_mev'], torch.linalg.matrix_norm(expected))
    torch.testing.assert_close(metrics['full_element_rms_mev'], expected.abs().square().mean().sqrt())
    # Pairing entirely in the omitted space survives the cached algebra.
    omitted = torch.eye(d, dtype=state.q.dtype)-state.q@state.q.mH
    omitted_delta = omitted@delta@omitted.conj()
    outside = FactorizedCommutator(state, torch.zeros_like(state.q), lambda v: omitted_delta@v)
    assert torch.linalg.vector_norm(outside.apply(unit)) > .1


def test_cached_commutator_preserves_field_and_state_gradients():
    state, state_parameters = _state(variables=True)
    d = state.dimension
    h0 = torch.randn(d, d, dtype=torch.complex128, requires_grad=True)
    delta0 = torch.randn(d, d, dtype=torch.complex128, requires_grad=True)
    h, delta = .5*(h0+h0.mH), .5*(delta0-delta0.mT)
    vectors = torch.randn(2*d, 3, dtype=torch.complex128)
    action = NambuAction(d, lambda v: h@v, lambda v: delta@v, .21)
    cached = FactorizedCommutator(state, h@state.q, lambda v: delta@v,
                                  chemical_potential=.21)
    expected = commutator_action(action, state, vectors).abs().square().sum()
    actual = cached.apply(vectors).abs().square().sum()
    inputs = (h0, delta0) + state_parameters
    expected_grad = torch.autograd.grad(expected, inputs, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, inputs)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
