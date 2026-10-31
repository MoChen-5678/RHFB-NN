"""Independent dense AD and finite differences for block-recomputed field VJP."""
import pytest
import torch

from dpl_rhf.rhfb.streamed_field_autograd import streamed_energy_and_fields


def _densities(x):
    an = torch.stack((torch.stack((x[0], x[1]+1j*x[2])),
                      torch.stack((x[1]-1j*x[2], x[3]))))
    ap = torch.stack((torch.stack((x[4], x[5]+1j*x[6])),
                      torch.stack((x[5]-1j*x[6], x[7]))))
    return an.to(torch.complex128), ap.to(torch.complex128)


def _factory(rhos):
    def local(n, p):
        # Nonlinear shared vertex gives cross-species Hessian terms.
        vertex = torch.exp(-0.13*(torch.trace(n+p).real))
        return {'dd_local': vertex*(n.square().sum()+p.square().sum()).real}
    def exchange(n, p):
        return {'cross': 0.07*torch.trace(n@p@n@p).real,
                'linear': 0.31*torch.trace(n).real-0.17*torch.trace(p).real}
    yield local
    yield exchange


def _ordinary(n, p):
    blocks = [block(n, p) for block in _factory({'n': n, 'p': p})]
    e = sum(sum(terms.values()) for terms in blocks)
    g = torch.autograd.grad(e, (n, p), create_graph=True)
    return e, *[0.5*(value+value.mH) for value in g]


def _loss(outputs, kind):
    e, hn, hp = outputs
    if kind == 'energy':
        return e
    if kind == 'neutron_field':
        return hn.abs().square().sum()
    return 0.4*e+0.7*hn.abs().square().sum()+0.2*hp.abs().square().sum()


@pytest.mark.parametrize('kind', ['energy', 'neutron_field', 'combined'])
def test_recomputed_gradient_includes_density_and_field_hessian(kind):
    x = torch.tensor([0.4, 0.1, -0.03, 0.6, 0.3, 0.04, 0.02, 0.5],
                     dtype=torch.float64, requires_grad=True)
    n, p = _densities(x)
    expected = _ordinary(n, p)
    got = streamed_energy_and_fields(n, p, _factory)
    for value, reference in zip(got, expected):
        torch.testing.assert_close(value, reference, atol=1e-12, rtol=1e-12)
    grad_expected, = torch.autograd.grad(_loss(expected, kind), x, retain_graph=True)
    grad_got, = torch.autograd.grad(_loss(got, kind), x)
    torch.testing.assert_close(grad_got, grad_expected, atol=2e-12, rtol=2e-12)
    direction = torch.tensor([0.2, -0.1, 0.3, 0.05, -0.2, 0.1, 0.04, -0.3],
                             dtype=torch.float64)
    step = 1e-5
    plus = _loss(streamed_energy_and_fields(*_densities(x.detach()+step*direction), _factory), kind)
    minus = _loss(streamed_energy_and_fields(*_densities(x.detach()-step*direction), _factory), kind)
    torch.testing.assert_close((plus-minus)/(2*step), grad_got@direction,
                               atol=2e-9, rtol=2e-9)
    if kind == 'neutron_field':
        assert grad_got[4:].abs().max() > 1e-3  # cross-species DD pullback retained
