"""Check the allowed first variations, including real unscaled D1S."""
import torch

from dpl_rhf.rhfb.branch import nambu_hamiltonian
from dpl_rhf.rhfb.branch_prototype import make_full_dirac_model
from dpl_rhf.rhfb.manifold import canonical_generator, slater_reference, transform_reference
from dpl_rhf.rhfb.no_sea_variation import (
    external_projector_response_source, orbital_first_variation_gradient,
)
from dpl_rhf.rhfb.pairing import pairing_gap


def test_d1s_orbital_response_and_nambu_first_variation():
    torch.manual_seed(20261006)
    model = make_full_dirac_model()
    d, m = model.dimension, 6
    raw = torch.randn(d, m, dtype=torch.complex128)
    b, _ = torch.linalg.qr(raw)
    a = .07 * torch.randn(m, m, dtype=torch.complex128)
    a = a - a.mH
    beta = .21 * torch.randn(m, m, dtype=torch.complex128)
    beta = beta - beta.mT
    reference = slater_reference(torch.tensor([1., 1., 0., 0., 0., 0.], dtype=torch.float64),
                                 blocked_index=2)
    active = transform_reference(reference, a, beta)
    h = model.h0
    mu = .45

    def evaluate(frame):
        rho = frame @ active.rho @ frame.mH
        chi = frame @ active.chi @ frame.mT
        pair, gap = pairing_gap(model.spinors, chi, model.positions, model.weights,
                                species="neutron", chunk_size=8)
        return (torch.trace((h - mu * torch.eye(d, dtype=h.dtype)) @ rho).real + pair,
                rho, chi, gap)

    _, rho, chi, delta = evaluate(b)
    g = orbital_first_variation_gradient(h, delta, b, active.rho, active.chi,
                                         chemical_potential=mu)
    # Infinitesimal rotations into the excluded particle complement.
    x = torch.randn_like(b)
    db = x - b @ (b.mH @ x)
    t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    numerical = torch.autograd.grad(evaluate(b + t * db)[0], t)[0]
    analytic = 2 * torch.trace(g.mH @ db).real
    torch.testing.assert_close(analytic, numerical, atol=2e-11, rtol=2e-11)

    drho = db @ active.rho @ b.mH + b @ active.rho @ db.mH
    dchi = db @ active.chi @ b.mT + b @ active.chi @ db.mT
    dr = torch.cat((torch.cat((drho, dchi), 1),
                    torch.cat((-dchi.conj(), -drho.conj()), 1)), 0)
    hb = nambu_hamiltonian(h, delta, chemical_potential=mu)
    torch.testing.assert_close(.5 * torch.trace(hb @ dr).real, numerical,
                               atol=2e-11, rtol=2e-11)
    dp = db @ b.mH + b @ db.mH
    source = external_projector_response_source(b, g)
    torch.testing.assert_close(torch.trace(source @ dp).real, numerical,
                               atol=2e-11, rtol=2e-11)
    assert float(torch.linalg.matrix_norm(active.chi)) > .1
    assert active.fermion_parity == -1


def test_fixed_space_canonical_variation_is_active_commutator():
    torch.manual_seed(6728)
    d = 6
    h = torch.randn(d, d, dtype=torch.complex128)
    h = h + h.mH
    delta = torch.randn(d, d, dtype=torch.complex128)
    delta = delta - delta.mT
    state = slater_reference(torch.tensor([1., 1., 0., 0., 0., 0.], dtype=torch.float64),
                              blocked_index=2)
    a = torch.randn(d, d, dtype=torch.complex128)
    a = a - a.mH
    b = torch.randn(d, d, dtype=torch.complex128)
    b = b - b.mT
    k = canonical_generator(a, b)
    hb = nambu_hamiltonian(h, delta, chemical_potential=.2)
    dr = k @ state.matrix - state.matrix @ k
    comm = state.matrix @ hb - hb @ state.matrix
    torch.testing.assert_close(.5 * torch.trace(hb @ dr).real,
                               .5 * torch.trace(comm @ k).real, atol=1e-12, rtol=1e-12)
