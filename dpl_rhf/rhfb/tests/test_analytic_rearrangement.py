"""Small-space checks for the independently contracted PKA1 rearrangement source."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.analytic_rearrangement import analytic_rearrangement_reference
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.currents import _gamma_matrices, local_bilinears
from dpl_rhf.rhfb.functional import evaluate_particle_hole_energy, evaluate_rhfb_energy
import dpl_rhf.rhfb.functional as functional_module
from dpl_rhf.rhf.parameters import pka1_parameters


def _states(basis):
    gen = torch.Generator().manual_seed(20261005)
    result = {}
    for species in ("n", "p"):
        q = torch.zeros((basis.dimension, 2), dtype=torch.complex128)
        for channel in basis.channels:
            upper = torch.randn((basis.n_radial, 2), generator=gen).to(torch.complex128)
            upper = upper + 0.17j * torch.randn(upper.shape, generator=gen)
            lower = torch.randn((basis.n_radial, 2), generator=gen).to(torch.complex128)
            lower = 0.025 * (lower + 0.7j * torch.randn(lower.shape, generator=gen))
            q[channel.offset:channel.offset + basis.n_radial] = 0.012 * upper
            q[channel.offset + basis.n_radial:channel.offset + 2 * basis.n_radial] = 0.012 * lower
        raw = torch.randn((2, 2), generator=gen).to(torch.complex128)
        rho = 0.42 * torch.eye(2, dtype=torch.complex128) + 0.012 * (raw + raw.mH) / 2
        result[species] = SimpleNamespace(q=q, rho_active=rho)
    return result


def _energy(basis, states, params):
    return evaluate_particle_hole_energy(
        basis, states, chunk_size=128, pair_chunk=2, params=params, lmax=2,
    ).total_energy


def test_analytic_endpoint_sources_match_independent_vertex_parameter_fd(record_property):
    basis = AngularRadialBasis(
        radius_fm=1.8, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    states = _states(basis)
    params = pka1_parameters()
    reference = analytic_rearrangement_reference(basis, states, params)
    parameter_to_vertices = {
        "g_sigma": ("gsig",), "g_omega": ("gome",),
        # rho_tensor_vertex is derived from g_rho * f_rho in Define.f90.
        "f_rho": ("grtn",), "g_rho": ("grho", "grtn"), "f_pi": ("fpi",),
    }
    eps = 2.0e-5
    for parameter, keys in parameter_to_vertices.items():
        value = getattr(params, parameter)
        plus_eval = evaluate_particle_hole_energy(
            basis, states, chunk_size=128, pair_chunk=2,
            params=replace(params, **{parameter: value * (1 + eps)}), lmax=2,
        )
        minus_eval = evaluate_particle_hole_energy(
            basis, states, chunk_size=128, pair_chunk=2,
            params=replace(params, **{parameter: value * (1 - eps)}), lmax=2,
        )
        plus, minus = plus_eval.total_energy, minus_eval.total_energy
        fd = (plus - minus) / (2 * eps)
        ad = torch.sum(
            torch.as_tensor(basis.weights, dtype=torch.float64)
            * sum(reference["dE_dvertex"][key] * reference["vertices"][key]
                  for key in keys)
        )
        record_property(f"endpoint_fd_{parameter}", float(fd.detach()))
        record_property(f"endpoint_analytic_{parameter}", float(ad.detach()))
        if parameter == "f_rho":
            for component, term in (("rho_tensor_direct", "rho_tensor_direct"),
                                    ("rho_tensor_fock", "rho_tensor_fock"),
                                    ("rho_tensor_contact", "rho_tensor_contact")):
                comp = reference["dE_dvertex_components"][term]
                comp_value = torch.sum(torch.as_tensor(basis.weights) * comp * reference["vertices"]["grtn"])
                component_fd = (plus_eval.energy_terms[term] - minus_eval.energy_terms[term]) / (2 * eps)
                record_property(f"component_fd_{component}", float(component_fd.detach()))
                record_property(f"component_analytic_{component}", float(comp_value.detach()))
                torch.testing.assert_close(component_fd, comp_value, atol=2e-10, rtol=2e-8)
        torch.testing.assert_close(fd, ad, atol=2e-10, rtol=2e-8)


def test_time_reversal_bilinears_and_every_ph_energy_channel(record_property):
    basis = AngularRadialBasis(
        radius_fm=1.8, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    states = _states(basis)
    gamma, _ = _gamma_matrices(device=torch.device("cpu"), dtype=torch.complex128)
    time_reversal = 1j * gamma[1] @ gamma[3]
    transformed = {}
    parity_current = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
    parity_tensor = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, -1.0, -1.0],
         [1.0, -1.0, -1.0, -1.0], [1.0, -1.0, -1.0, -1.0]],
        dtype=torch.float64,
    )
    for species, state in states.items():
        sample = basis._sample_range(state.q, 0, basis.point_count)
        q_tr_sample = torch.einsum("ij,pjm->pim", time_reversal, sample.conj())
        q_tr = basis.project_sample_range(q_tr_sample, 0, basis.point_count)
        transformed[species] = SimpleNamespace(
            q=q_tr, rho_active=state.rho_active.conj(),
        )
        before = local_bilinears(sample, state.rho_active)
        after = local_bilinears(q_tr_sample, state.rho_active.conj())
        torch.testing.assert_close(after["scalar"], before["scalar"], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(after["baryon_current"],
                                   before["baryon_current"] * parity_current,
                                   atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(after["axial_current"],
                                   before["axial_current"] * parity_current,
                                   atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(after["tensor"],
                                   before["tensor"] * parity_tensor,
                                   atol=2e-12, rtol=2e-12)

    normal = evaluate_particle_hole_energy(
        basis, states, chunk_size=128, pair_chunk=2, params=pka1_parameters(), lmax=2,
    )
    reversed_normal = evaluate_particle_hole_energy(
        basis, transformed, chunk_size=128, pair_chunk=2,
        params=pka1_parameters(), lmax=2,
    )
    assert normal.energy_terms.keys() == reversed_normal.energy_terms.keys()
    for name, value in normal.energy_terms.items():
        record_property(f"time_reversal_energy_abs_error_{name}",
                        float((value - reversed_normal.energy_terms[name]).abs().detach()))
        torch.testing.assert_close(value, reversed_normal.energy_terms[name],
                                   atol=1e-12, rtol=1e-10)

    # Also test the anomalous Gaussian pairing action under chi -> chi*.
    paired, paired_tr = {}, {}
    for species, state in states.items():
        chi = torch.tensor([[0.0, 0.015 + 0.009j], [-0.015 - 0.009j, 0.0]],
                           dtype=torch.complex128)
        paired[species] = SimpleNamespace(q=state.q, rho_active=state.rho_active,
                                          chi_active=chi)
        paired_tr[species] = SimpleNamespace(
            q=transformed[species].q,
            rho_active=transformed[species].rho_active,
            chi_active=chi.conj(),
        )
    full = evaluate_rhfb_energy(
        basis, paired, chunk_size=128, pair_chunk=2, params=pka1_parameters(), lmax=2,
    )
    full_tr = evaluate_rhfb_energy(
        basis, paired_tr, chunk_size=128, pair_chunk=2,
        params=pka1_parameters(), lmax=2,
    )
    for name, value in full.energy_terms.items():
        record_property(f"full_time_reversal_energy_abs_error_{name}",
                        float((value - full_tr.energy_terms[name]).abs().detach()))
        torch.testing.assert_close(value, full_tr.energy_terms[name],
                                   atol=1e-12, rtol=1e-10)


def test_nonuniform_signed_vertex_endpoint_directions(record_property, monkeypatch):
    basis = AngularRadialBasis(
        radius_fm=1.8, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    states = _states(basis)
    params = pka1_parameters()
    ref = analytic_rearrangement_reference(basis, states, params)
    baseline_couplings = functional_module._couplings(ref["rho_b"], params)
    positions = torch.as_tensor(basis.positions, dtype=torch.float64)
    radius = torch.linalg.vector_norm(positions, dim=1).clamp_min(1e-12)
    x, y, z = (positions[:, i] / radius for i in range(3))
    patterns = (
        0.31 * x - 0.17 * y + 0.09 * z,
        0.23 * torch.sin(2.1 * x) * torch.cos(1.4 * y) - 0.11 * torch.sin(1.7 * z),
    )
    # These perturbations are independent signed fields, not a common scaling
    # of a parameter.  Thus they test the pointwise endpoint density.
    key_to_functional_name = {
        "gsig": "gsig", "gome": "gome", "grho": "grho",
        "grtn": "grtn", "fpi": "fpio",
    }
    # The fixed-coupling EDF is at most quadratic in a single vertex field,
    # so the central difference has no cubic truncation term. A larger step
    # improves subtraction accuracy while preserving the exact target slope.
    eps = 1.0e-3
    for pattern_index, pattern in enumerate(patterns):
        for key, functional_name in key_to_functional_name.items():
            delta_effective = ref["vertices"][key] * pattern
            if key == "fpi":
                # functional._couplings returns f_pi before its hbar_c/m_pi
                # conversion; the endpoint reference uses the actual vertex.
                delta_functional = delta_effective * params.m_pi / params.hbar_c
            else:
                delta_functional = delta_effective
            def perturbed_couplings(rho_b, p, sign):
                values = dict(baseline_couplings)
                values[functional_name] = values[functional_name] + sign * eps * delta_functional
                return values

            def evaluate(sign):
                monkeypatch.setattr(
                    functional_module, "_couplings",
                    lambda rho_b, p, _sign=sign: perturbed_couplings(rho_b, p, _sign),
                )
                return evaluate_particle_hole_energy(
                    basis, states, chunk_size=128, pair_chunk=2,
                    params=params, lmax=2,
                ).total_energy

            plus, minus = evaluate(+1.0), evaluate(-1.0)
            fd = (plus - minus) / (2 * eps)
            analytic = torch.sum(
                torch.as_tensor(basis.weights, dtype=torch.float64)
                * ref["dE_dvertex"][key] * delta_effective
            )
            record_property(f"signed_endpoint_abs_error_{pattern_index}_{key}",
                            float((fd - analytic).abs().detach()))
            record_property(f"signed_endpoint_fd_{pattern_index}_{key}", float(fd.detach()))
            record_property(f"signed_endpoint_analytic_{pattern_index}_{key}",
                            float(analytic.detach()))
            torch.testing.assert_close(fd, analytic, atol=2e-11, rtol=2e-7)


def test_rearrangement_dirac_operator_equals_covariant_density_source(record_property):
    basis = AngularRadialBasis(
        radius_fm=1.8, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    states = _states(basis)
    params = pka1_parameters()
    gen = torch.Generator().manual_seed(20261006)
    directions = {}
    for species in ("n", "p"):
        raw = torch.randn((2, 2), generator=gen).to(torch.complex128)
        directions[species] = 7.0e-4 * (raw + raw.mH) / 2
    ref = analytic_rearrangement_reference(basis, states, params)
    sampled = {s: basis._sample_range(states[s].q, 0, basis.point_count)
               for s in ("n", "p")}
    delta_currents = {
        s: local_bilinears(sampled[s], directions[s])["baryon_current"]
        for s in ("n", "p")
    }
    delta_j = delta_currents["n"] + delta_currents["p"]
    j = ref["baryon_current"]
    rho_b = ref["rho_b"]
    delta_rho_b = (j[:, 0] * delta_j[:, 0]
                   - torch.sum(j[:, 1:] * delta_j[:, 1:], dim=1)) / rho_b
    weights = torch.as_tensor(basis.weights, dtype=torch.float64)
    covariant_source = torch.sum(weights * ref["S_R"] * delta_rho_b)

    gamma, _ = _gamma_matrices(device=torch.device("cpu"), dtype=torch.complex128)
    alpha = [gamma[0] @ gamma[i] for i in range(1, 4)]
    identity = torch.eye(4, dtype=torch.complex128)
    h_rearr = ref["S_R"][:, None, None] / rho_b[:, None, None] * (
        j[:, 0, None, None] * identity
        - sum(j[:, i + 1, None, None] * alpha[i] for i in range(3))
    )
    dirac_source = weights.new_zeros(())
    weighted_h = h_rearr * weights[:, None, None]
    for species in ("n", "p"):
        active_field = torch.einsum(
            "pia,pij,pjb->ab", sampled[species].conj(), weighted_h,
            sampled[species],
        )
        dirac_source = dirac_source + torch.trace(
            directions[species] @ active_field
        ).real
    torch.testing.assert_close(dirac_source, covariant_source, atol=1e-12, rtol=1e-10)

    eps = 1.0e-3
    def rearrangement_difference(sign):
        trial = {
            s: SimpleNamespace(
                q=states[s].q,
                rho_active=states[s].rho_active + sign * eps * directions[s],
            ) for s in ("n", "p")
        }
        complete = evaluate_particle_hole_energy(
            basis, trial, chunk_size=128, pair_chunk=2, params=params, lmax=2,
        )
        fixed = evaluate_particle_hole_energy(
            basis, trial, chunk_size=128, pair_chunk=2, params=params, lmax=2,
            _vertex_density_override=rho_b.detach(),
        )
        return complete.total_energy - fixed.total_energy

    fd = (rearrangement_difference(+1.0) - rearrangement_difference(-1.0)) / (2 * eps)
    record_property("dirac_covariant_source", float(dirac_source.detach()))
    record_property("invariant_density_direction_source", float(covariant_source.detach()))
    record_property("rearrangement_energy_fd", float(fd.detach()))
    record_property("rearrangement_energy_fd_abs_error", float((fd - covariant_source).abs().detach()))
    torch.testing.assert_close(fd, covariant_source, atol=2e-11, rtol=2e-5)
