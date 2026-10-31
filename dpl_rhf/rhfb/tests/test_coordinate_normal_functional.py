from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.coordinate_normal_functional import CoordinateNormalFunctional
from dpl_rhf.rhfb.functional import evaluate_particle_hole_energy
from dpl_rhf.rhfb.normal_action import AutogradNormalAction


def _fixture():
    torch.manual_seed(4451)
    basis = AngularRadialBasis(radius_fm=1.6, spacing_fm=0.2, kappa_max=1,
                               n_theta=3, n_phi=6, derivative_order=5)
    d, m = basis.dimension, 3
    states = {}
    for sp in ("n", "p"):
        raw = torch.randn((d, m), dtype=torch.float64).to(torch.complex128)
        raw = raw + 1j * torch.randn((d, m), dtype=torch.float64)
        q, _ = torch.linalg.qr(raw)
        u0 = torch.randn((m, m), dtype=torch.complex128)
        u, _ = torch.linalg.qr(u0)
        occupations = torch.tensor([0.16, 0.27, 0.34], dtype=torch.float64)
        rho = u @ torch.diag(occupations.to(torch.complex128)) @ u.mH
        states[sp] = SimpleNamespace(q=q, rho_active=rho, chi_active=None)
    return basis, states


def test_one_pass_coordinate_field_matches_factorized_energy_and_action():
    basis, states = _fixture()
    fast = CoordinateNormalFunctional(basis, pair_chunk=32, chunk_size=32, lmax=0,
                                     checkpoint_fock_blocks=True).evaluate(states)
    reference = evaluate_particle_hole_energy(
        basis, states, pair_chunk=8, chunk_size=32, lmax=0,
        _checkpoint_fock_blocks=False,
    )
    assert fast.diagnostics["energy_forward_count"] == 1
    assert fast.diagnostics["interaction_four_tensor_materialized"] is False
    assert set(fast.energy.energy_terms) == set(reference.energy_terms)
    for name, value in reference.energy_terms.items():
        torch.testing.assert_close(fast.energy.energy_terms[name], value,
                                   atol=2e-9, rtol=2e-11, msg=name)

    probes = torch.randn((basis.dimension, 2), dtype=torch.float64).to(torch.complex128)
    probes = probes + 1j * torch.randn((basis.dimension, 2), dtype=torch.float64)
    oracle = AutogradNormalAction(
        basis, states, lambda b, s: evaluate_particle_hole_energy(
            b, s, pair_chunk=8, chunk_size=32, lmax=0,
            _checkpoint_fock_blocks=False,
        ).total_energy,
        species="n", vector_chunk=1, create_graph=False,
    )
    torch.testing.assert_close(fast.h["n"] @ probes, oracle.apply(probes),
                               atol=3e-6, rtol=2e-7)
    for species in ("n", "p"):
        torch.testing.assert_close(fast.h[species], fast.h[species].mH,
                                   atol=2e-11, rtol=0)


def test_coordinate_field_matches_hermitian_density_directional_fd():
    basis, states = _fixture()
    result = CoordinateNormalFunctional(basis, pair_chunk=32, chunk_size=32, lmax=0,
                                        checkpoint_fock_blocks=False).evaluate(states)
    torch.manual_seed(4452)
    direction = {}
    plus, minus = {}, {}
    for sp in ("n", "p"):
        raw = torch.randn((3, 3), dtype=torch.float64).to(torch.complex128)
        raw = raw + 1j * torch.randn((3, 3), dtype=torch.float64)
        direction[sp] = (raw + raw.mH) / (2 * torch.linalg.matrix_norm(raw))
        plus[sp] = SimpleNamespace(q=states[sp].q,
                                    rho_active=states[sp].rho_active + 1e-4*direction[sp],
                                    chi_active=None)
        minus[sp] = SimpleNamespace(q=states[sp].q,
                                     rho_active=states[sp].rho_active - 1e-4*direction[sp],
                                     chi_active=None)
    functional = CoordinateNormalFunctional(basis, pair_chunk=32, chunk_size=32, lmax=0,
                                            checkpoint_fock_blocks=False)
    eplus, tplus = functional.energy_only(plus)
    eminus, tminus = functional.energy_only(minus)
    finite_difference = (eplus - eminus) / 2e-4
    # Independently vary the production factorized scalar at the same frozen
    # Q and Hermitian active-density direction. Compare every PH term's slope,
    # so a channel sign error cannot hide through cancellation in the total.
    pref_kwargs = dict(pair_chunk=8, chunk_size=32, lmax=0,
                       _checkpoint_fock_blocks=False)
    pref_plus = evaluate_particle_hole_energy(basis, plus, **pref_kwargs)
    pref_minus = evaluate_particle_hole_energy(basis, minus, **pref_kwargs)
    for name in result.energy.energy_terms:
        if name == "d1s_pairing":
            continue
        coord_slope = (tplus[name] - tminus[name]) / 2e-4
        prod_slope = (pref_plus.energy_terms[name] - pref_minus.energy_terms[name]) / 2e-4
        torch.testing.assert_close(coord_slope, prod_slope,
                                   atol=2e-6, rtol=3e-5, msg=f"directional {name}")
    analytic = torch.zeros((), dtype=torch.float64)
    for sp in ("n", "p"):
        q = states[sp].q
        drho_primitive = q @ direction[sp] @ q.mH
        analytic = analytic + torch.sum(result.h[sp].conj() * drho_primitive).real
    torch.testing.assert_close(analytic, finite_difference, atol=3e-7, rtol=3e-6)
