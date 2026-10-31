"""Independent dense algebra oracle for a streamed no-spectrum loss."""
import pytest
import torch

from dpl_rhf.rhfb.factorized_commutator import FactorizedCommutator
from dpl_rhf.rhfb.manifold import slater_reference, transform_reference
from dpl_rhf.rhfb.operators import DensityFactors


def _complex_random(shape, generator):
    return torch.complex(torch.randn(shape, generator=generator, dtype=torch.float64),
                         torch.randn(shape, generator=generator, dtype=torch.float64))


def _case(parameter):
    rng = torch.Generator().manual_seed(3081)
    d, m = 8, 4
    q0 = torch.linalg.qr(_complex_random((d, m), rng)).Q
    raw_rotation = _complex_random((d, d), rng)
    q = torch.matrix_exp(.04*parameter*(raw_rotation-raw_rotation.mH)) @ q0
    raw_a = _complex_random((m, m), rng)
    raw_b = _complex_random((m, m), rng)
    reference = slater_reference(torch.tensor([1., 1., 0., 0.], dtype=torch.float64),
                                 blocked_index=2, blocking_operation="particle")
    active = transform_reference(reference, .05*parameter*(raw_a-raw_a.mH),
                                 .07*parameter*(raw_b-raw_b.mT))
    density = DensityFactors(q, active.rho, active.chi)
    rho = q @ active.rho @ q.mH
    chi = q @ active.chi @ q.mT
    raw_h = _complex_random((d, d), rng)
    raw_delta = _complex_random((d, d), rng)
    # Include field dependence as well as orbit/density dependence. This is
    # an algebra test, not a nuclear functional or physical prescription.
    h = .3*(raw_h+raw_h.mH) + .2*rho + .1*(chi@chi.mH)
    delta = .15*(raw_delta-raw_delta.mT) + .1*chi
    mu = .23*parameter
    return density, h, delta, mu, rho, chi


def _dense_oracle(h, delta, mu, rho, chi):
    d = h.shape[0]
    identity = torch.eye(d, dtype=h.dtype)
    normal = h-mu*identity
    big_h = torch.cat((torch.cat((normal, delta), dim=1),
                       torch.cat((-delta.conj(), -normal.conj()), dim=1)), dim=0)
    big_r = torch.cat((torch.cat((rho, chi), dim=1),
                       torch.cat((-chi.conj(), identity-rho.conj()), dim=1)), dim=0)
    comm = big_h@big_r-big_r@big_h
    return comm, comm.abs().square().sum()


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 20])
def test_streamed_full_loss_and_gradient_match_independent_dense_oracle(chunk_size):
    parameter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    density, h, delta, mu, rho, chi = _case(parameter)
    action = FactorizedCommutator(density, h@density.q, lambda x: delta@x,
                                 chemical_potential=mu)
    result = action.frobenius_squared(chunk_size=chunk_size)
    comm, expected = _dense_oracle(h, delta, mu, rho, chi)
    assert torch.allclose(result["full_squared_mev2"], expected, atol=2e-12, rtol=2e-13)
    assert torch.allclose(result["normal_squared_mev2"], comm[:8, :8].abs().square().sum(),
                          atol=1e-12, rtol=2e-13)
    assert torch.allclose(result["anomalous_squared_mev2"], comm[:8, 8:].abs().square().sum(),
                          atol=1e-12, rtol=2e-13)
    got_grad = torch.autograd.grad(result["full_squared_mev2"], parameter,
                                   retain_graph=True)[0]
    oracle_grad = torch.autograd.grad(expected, parameter)[0]
    assert torch.allclose(got_grad, oracle_grad, atol=2e-11, rtol=2e-12)
    step = 1e-5
    values = []
    for offset in (step, -step):
        other_density, other_h, other_delta, other_mu, other_rho, other_chi = _case(
            torch.tensor(.7+offset, dtype=torch.float64))
        values.append(_dense_oracle(other_h, other_delta, other_mu,
                                    other_rho, other_chi)[1].item())
    assert abs(got_grad.item()-(values[0]-values[1])/(2*step)) < 2e-8


def test_pairing_between_two_omitted_orbits_remains_in_full_loss():
    q = torch.eye(4, dtype=torch.complex128)[:, :2]
    rho_a = torch.diag(torch.tensor([1., 0.], dtype=torch.float64)).to(q.dtype)
    density = DensityFactors(q, rho_a, torch.zeros_like(rho_a))
    h = torch.diag(torch.tensor([-3., 2., 5., 7.], dtype=torch.float64)).to(q.dtype)
    strength = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    skew = q.new_zeros((4, 4))
    skew[2, 3], skew[3, 2] = 1., -1.
    delta = strength*skew
    result = FactorizedCommutator(density, h@q, lambda x: delta@x).frobenius_squared(
        chunk_size=3)
    assert result["normal_squared_mev2"].item() == 0.
    assert torch.allclose(result["full_squared_mev2"], 4*strength.square(), atol=1e-15)
    assert torch.allclose(torch.autograd.grad(result["full_squared_mev2"], strength)[0],
                          8*strength, atol=1e-15)


def test_reject_invalid_chunk_size():
    density, h, delta, mu, _, _ = _case(torch.tensor(.7, dtype=torch.float64))
    action = FactorizedCommutator(density, h@density.q, lambda x: delta@x,
                                 chemical_potential=mu)
    for chunk_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            action.frobenius_squared(chunk_size=chunk_size)
