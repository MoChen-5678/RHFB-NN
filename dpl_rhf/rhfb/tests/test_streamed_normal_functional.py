from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.coordinate_normal_functional import CoordinateNormalFunctional
from dpl_rhf.rhfb.spatial import GreenOperator
from dpl_rhf.rhfb.streamed_normal_functional import StreamedPrimitiveNormalFunctional


def _coordinate_block_oracle(functional, primitive_rhos, c, weights, green,
                             kinetic_ops, start, stop, include_local):
    # Rebuild R for one scalar block only. The caller consumes its gradient
    # before requesting the next block, so no traversal graph is retained.
    rhos = {s: torch.einsum("pai,ij,qbj->paqb", c, primitive_rhos[s], c.conj())
            for s in ("n", "p")}
    _, values = functional._energy_from_coordinate_density(
        rhos, weights, green, pair_range=(start, stop), include_local=include_local)
    if include_local:
        values["kinetic"] = sum((torch.trace(primitive_rhos[s] @ kinetic_ops[s]).real
                                 for s in ("n", "p")), weights.new_zeros(()))
    total = sum(values.values(), weights.new_zeros(()))
    gradients = torch.autograd.grad(total, tuple(primitive_rhos[s] for s in ("n", "p")))
    result = (total.detach(), {k: v.detach() for k, v in values.items()},
              tuple((0.5*(g+g.mH)).detach() for g in gradients))
    del total, values, gradients, rhos
    return result


def _independent_coordinate_oracle(functional, states):
    device = states["n"].q.device
    c, weights = functional._weighted_basis(device)
    green = GreenOperator(functional.basis, lmax=functional.lmax, device=device)
    identity = torch.eye(functional.basis.dimension, dtype=torch.complex128, device=device)
    rho = {s: (states[s].q @ states[s].rho_active @ states[s].q.mH)
           .detach().requires_grad_(True) for s in ("n", "p")}
    kinetic_ops = {s: functional.basis.apply_kinetic(identity, species=s)
                   for s in ("n", "p")}
    terms = {}
    energy = weights.new_zeros(())
    fields = {s: torch.zeros_like(rho[s]) for s in ("n", "p")}
    blocks = [(0, 0, True)] + [(a, min(functional.basis.point_count,
                                        a + functional.pair_chunk), False)
                              for a in range(0, functional.basis.point_count,
                                             functional.pair_chunk)]
    for start, stop, include_local in blocks:
        e, block_terms, gradients = _coordinate_block_oracle(
            functional, rho, c, weights, green, kinetic_ops,
            start, stop, include_local)
        energy += e
        for name, value in block_terms.items():
            terms[name] = terms.get(name, weights.new_zeros(())) + value
        for s, grad in zip(("n", "p"), gradients):
            fields[s] += grad
    terms["d1s_pairing"] = weights.new_zeros(())
    terms["density_rearrangement"] = weights.new_zeros(())
    return energy, terms, fields


def _nonorthogonal_full_density_states(basis):
    torch.manual_seed(6721)
    states = {}
    for species in ("n", "p"):
        raw = torch.randn((basis.dimension, 3), dtype=torch.float64).to(torch.complex128)
        raw = raw + 1j * torch.randn((basis.dimension, 3), dtype=torch.float64)
        # Deliberately retain a non-orthogonal Q at moderate normalization:
        # the provider must contract Q rho Q† exactly without driving the
        # density-dependent vertices into an extreme-density regime.
        q = raw * (0.3 / (2*basis.dimension)**0.5)
        a = torch.randn((3, 3), dtype=torch.complex128)
        unitary, _ = torch.linalg.qr(a)
        occupations = torch.tensor([0.31, 0.22, 0.17], dtype=torch.float64)
        rho = unitary @ torch.diag(occupations.to(torch.complex128)) @ unitary.mH
        states[species] = SimpleNamespace(q=q, rho_active=rho, chi_active=None)
    return states


def test_streamed_pka1_energy_and_full_primitive_field_match_coordinate_oracle():
    # This numerical test is run only through scripts/run_rhfb_limited.py.
    basis = AngularRadialBasis(radius_fm=1.6, spacing_fm=0.2, kappa_max=1,
                               n_theta=3, n_phi=6, derivative_order=5)
    states = _nonorthogonal_full_density_states(basis)
    oracle = CoordinateNormalFunctional(
        basis, pair_chunk=5, chunk_size=8, lmax=0,
        checkpoint_fock_blocks=True)
    expected_e, expected_terms, expected_h = _independent_coordinate_oracle(oracle, states)
    actual = StreamedPrimitiveNormalFunctional(
        basis, tile_size=5, chunk_size=8, lmax=0).evaluate(states)

    assert actual.diagnostics["coordinate_R_four_axis_materialized"] is False
    assert actual.diagnostics["max_coordinate_density_tile_shape"] == (
        5, 4, basis.point_count, 4)
    assert actual.diagnostics["max_coordinate_density_tile_bytes_per_species"] == (
        5*4*basis.point_count*4*16)
    assert actual.energy.energy_terms["density_rearrangement"] == 0
    assert float(actual.energy.energy_terms["rho_tensor_contact"].abs()) > 0
    assert float(actual.energy.energy_terms["pion_pv_contact"].abs()) > 0
    torch.testing.assert_close(actual.energy.total_energy, expected_e,
                               atol=4e-7, rtol=3e-9)
    for name, value in expected_terms.items():
        torch.testing.assert_close(actual.energy.energy_terms[name], value,
                                   atol=4e-7, rtol=3e-9, msg=name)
    for species in ("n", "p"):
        torch.testing.assert_close(actual.h[species], expected_h[species],
                                   atol=2e-5, rtol=3e-7, msg=species)
