"""PKA1 field-loss pullback against independent dense-coordinate autograd."""
from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.coordinate_normal_functional import CoordinateNormalFunctional
from dpl_rhf.rhfb.spatial import GreenOperator
from dpl_rhf.rhfb.streamed_normal_functional import StreamedPrimitiveNormalFunctional


def _oracle_block(functional, rhos, c, weights, green, kinetic_ops,
                  start, stop, include_local, *, create_graph=False,
                  field_seed=None):
    coord = {s: torch.einsum("pai,ij,qbj->paqb", c, rhos[s], c.conj())
             for s in ("n", "p")}
    _, terms = functional._energy_from_coordinate_density(
        coord, weights, green, pair_range=(start, stop), include_local=include_local)
    if include_local:
        terms["kinetic"] = sum((torch.trace(rhos[s] @ kinetic_ops[s]).real
                                for s in ("n", "p")), weights.new_zeros(()))
    total = sum(terms.values(), weights.new_zeros(()))
    gradients = torch.autograd.grad(total, (rhos["n"], rhos["p"]),
                                    create_graph=create_graph)
    fields = tuple(0.5*(g+g.mH) for g in gradients)
    pullback = None
    if field_seed is not None:
        objective = 0.003*total + (field_seed.conj()*fields[0]).sum().real
        pullback = torch.autograd.grad(objective, (rhos["n"], rhos["p"]))
        pullback = tuple((0.5*(g+g.mH)).detach() for g in pullback)
    result = (total.detach(), tuple(v.detach() for v in fields),
              None if pullback is None else pullback)
    del total, terms, gradients, fields, coord
    return result


def _independent_coordinate_vjp(functional, rhos):
    device = rhos["n"].device
    c, weights = functional._weighted_basis(device)
    green = GreenOperator(functional.basis, lmax=functional.lmax, device=device)
    identity = torch.eye(functional.basis.dimension, dtype=torch.complex128, device=device)
    kinetic_ops = {s: functional.basis.apply_kinetic(identity, species=s)
                   for s in ("n", "p")}
    blocks = [(0, 0, True)] + [(a, min(functional.basis.point_count,
                                        a + functional.pair_chunk), False)
                              for a in range(0, functional.basis.point_count,
                                             functional.pair_chunk)]
    energy = weights.new_zeros(())
    fields = {s: torch.zeros_like(rhos[s]) for s in ("n", "p")}
    for start, stop, include_local in blocks:
        e, (hn, hp), _ = _oracle_block(
            functional, rhos, c, weights, green, kinetic_ops, start, stop,
            include_local)
        energy += e
        fields["n"] += hn
        fields["p"] += hp

    # The field loss depends on the sum of all block fields. Recompute every
    # block's Hessian-vector product against that detached global cotangent.
    seed = 2*fields["n"] / 1e6
    pullback = [torch.zeros_like(rhos[s]) for s in ("n", "p")]
    for start, stop, include_local in blocks:
        _e, _h, contribution = _oracle_block(
            functional, rhos, c, weights, green, kinetic_ops, start, stop,
            include_local, create_graph=True, field_seed=seed)
        for i, value in enumerate(contribution):
            pullback[i] += value
    return energy, fields, tuple(pullback)


def test_complete_pka1_field_vjp_matches_dense_energy_hessian():
    # The unchanged five-point radial implementation requires >=7 intervals;
    # the original 1.2/0.2 fixture had only six and never reached the EDF.
    basis = AngularRadialBasis(radius_fm=1.4, spacing_fm=0.2, kappa_max=1,
                               n_theta=3, n_phi=6, derivative_order=5)
    d = basis.dimension
    torch.manual_seed(71007)
    rhos, directions = {}, {}
    for species in ('n', 'p'):
        raw = torch.randn((d, 3), dtype=torch.complex128)
        rho = 0.003*torch.eye(d, dtype=torch.complex128)+0.001*(raw@raw.mH)/d
        rhos[species] = rho.requires_grad_(True)
        direction = torch.randn((d, d), dtype=torch.complex128)
        directions[species] = 0.5*(direction+direction.mH)/d
    eye = torch.eye(d, dtype=torch.complex128)
    states = {s: SimpleNamespace(q=eye, rho_active=rhos[s]) for s in ('n', 'p')}
    ordinary = CoordinateNormalFunctional(basis, pair_chunk=10, lmax=0,
                                          checkpoint_fock_blocks=False)
    expected_e, expected_h, expected_pullback = _independent_coordinate_vjp(
        ordinary, {"n": rhos["n"], "p": rhos["p"]})

    backend = StreamedPrimitiveNormalFunctional(basis, tile_size=8, chunk_size=32, lmax=0)
    actual_e, actual_h = backend.differentiable_energy_and_fields(states)
    torch.testing.assert_close(actual_e, expected_e, atol=1e-10, rtol=2e-11)
    for species in ('n', 'p'):
        reference = expected_h[species]
        torch.testing.assert_close(actual_h[species], reference, atol=3e-9, rtol=2e-11)
    actual_loss = 0.003*actual_e+actual_h['n'].abs().square().sum()/1e6
    actual_pullback = torch.autograd.grad(actual_loss, tuple(rhos.values()))
    # Both interfaces target the Hermitian density tangent. The dense scalar
    # also defines an extension away from that physical domain; compare its
    # Hermitian pullback rather than those arbitrary off-domain components.
    for got, reference in zip(actual_pullback, expected_pullback):
        torch.testing.assert_close(got, reference, atol=3e-8, rtol=3e-9)
    assert actual_pullback[1].abs().max() > 1e-5  # n-field reaches p via DD/shared sources
