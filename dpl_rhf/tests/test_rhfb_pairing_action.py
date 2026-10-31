import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.pairing import pairing_gap
from dpl_rhf.rhfb.pairing_action import D1SPairingAction


def example():
    basis = AngularRadialBasis(radius_fm=8., spacing_fm=1., kappa_max=1)
    q, _ = basis.build_initial_q(1, minimum=2, empty_states=0)
    chi = torch.tensor([[0, .2+.08j], [-.2-.08j, 0]], dtype=torch.complex128)
    return basis, q, chi


def test_sampler_projection_is_weighted_adjoint_and_exact_on_primitive_space():
    basis, _, _ = example()
    torch.manual_seed(441)
    coefficients = torch.randn(basis.dimension, 3, dtype=torch.complex128)
    values = torch.randn(basis.point_count, 4, 3, dtype=torch.complex128)
    samples, weights, _ = basis.sample(coefficients)
    projected = basis.project_sample_range(values, 0, basis.point_count)
    torch.testing.assert_close((samples.conj()*values*weights[:, None, None]).sum(),
                               (coefficients.conj()*projected).sum(), atol=2e-12, rtol=2e-13)
    inverse = basis.project_sample_range(samples, 0, basis.point_count)
    torch.testing.assert_close(inverse, coefficients, atol=2e-12, rtol=2e-13)
    chunked = torch.zeros_like(coefficients)
    for start in range(0, basis.point_count, 11):
        stop = min(start+11, basis.point_count)
        chunked += basis.project_sample_range(values[start:stop], start, stop)
    torch.testing.assert_close(chunked, projected, atol=2e-12, rtol=2e-13)


def test_full_pairing_action_matches_active_gap_and_is_antisymmetric():
    basis, q, chi = example()
    spinors, weights, positions = basis.sample(q)
    _, gap = pairing_gap(spinors, chi, positions, weights, species='neutron', chunk_size=30)
    action = D1SPairingAction(basis, q, chi, species='n', chunk_size=30, checkpoint_blocks=False)
    output = action.apply(q.conj())
    torch.testing.assert_close(q.mH @ output, gap, atol=5e-12, rtol=3e-12)
    # A purely active gap would omit this external part.
    external = output-q @ (q.mH @ output)
    assert external.abs().max() > 1e-4
    delta = action.apply(torch.eye(basis.dimension, dtype=torch.complex128))
    torch.testing.assert_close(delta, -delta.mT, atol=5e-12, rtol=3e-12)
    torch.manual_seed(27)
    vector = torch.randn(basis.dimension, dtype=torch.complex128)
    torch.testing.assert_close(action.apply(vector), delta @ vector, atol=5e-12, rtol=3e-12)


def test_pairing_action_backward_recomputation_preserves_gradients():
    basis, q0, chi0 = example()
    torch.manual_seed(72)
    vector0 = torch.randn(basis.dimension, 2, dtype=torch.complex128)
    results = []
    for recompute in (False, True):
        q = q0.clone().requires_grad_()
        chi = chi0.clone().requires_grad_()
        vector = vector0.clone().requires_grad_()
        action = D1SPairingAction(basis, q, chi, species='p', chunk_size=30, checkpoint_blocks=recompute)
        output = action.apply(vector)
        grads = torch.autograd.grad(output.abs().square().sum(), (q, chi, vector))
        assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)
        results.append((output, grads))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-12, rtol=1e-12)
    for direct, recomputed in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(direct, recomputed, atol=1e-11, rtol=1e-12)
