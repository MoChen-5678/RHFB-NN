"""Independent density-direction closure for the experimental general-m EDF."""
from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.currents import invariant_baryon_density, local_bilinears
from dpl_rhf.rhfb.functional import evaluate_particle_hole_energy
from dpl_rhf.rhfb.normal_action import AutogradNormalAction
from dpl_rhf.rhf.parameters import pka1_parameters


def _hermitian(seed, n):
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn((n, n), generator=gen, dtype=torch.float64).to(torch.complex128)
    return (raw + raw.mH) / 2


def _states_and_directions(basis):
    gen = torch.Generator().manual_seed(20261005)
    states, directions = {}, {}
    for index, species in enumerate(("n", "p")):
        q = torch.zeros((basis.dimension, 2), dtype=torch.complex128)
        for channel in basis.channels:
            upper_raw = torch.randn((basis.n_radial, 2), generator=gen, dtype=torch.float64)
            upper = upper_raw.to(torch.complex128) + 0.17j * torch.randn(
                upper_raw.shape, generator=gen, dtype=torch.float64
            )
            lower_raw = torch.randn((basis.n_radial, 2), generator=gen, dtype=torch.float64)
            lower = 0.025 * (lower_raw.to(torch.complex128) + 0.7j * torch.randn(
                lower_raw.shape, generator=gen, dtype=torch.float64
            ))
            q[channel.offset:channel.offset + basis.n_radial] = 0.012 * upper
            q[channel.offset + basis.n_radial:channel.offset + 2 * basis.n_radial] = 0.012 * lower
        rho = 0.42 * torch.eye(2, dtype=torch.complex128) + 0.012 * _hermitian(81 + index, 2)
        direction = 0.004 * _hermitian(181 + index, 2)
        states[species] = SimpleNamespace(q=q, rho_active=rho.requires_grad_())
        directions[species] = direction
    return states, directions


def _evaluate(basis, states, *, freeze=False, vertex_density=None):
    return evaluate_particle_hole_energy(
        basis, states, chunk_size=128, pair_chunk=2,
        params=pka1_parameters(), lmax=2,
        _freeze_vertex_density=freeze,
        _vertex_density_override=vertex_density,
    )


def test_pka1_channel_directional_and_rearrangement_derivatives_close(record_property):
    basis = AngularRadialBasis(
        radius_fm=1.8, spacing_fm=0.2, kappa_max=1, n_theta=3, n_phi=6,
    )
    states, directions = _states_and_directions(basis)
    sampled = {s: basis._sample_range(states[s].q, 0, basis.point_count)
               for s in ("n", "p")}
    current = {s: local_bilinears(sampled[s], states[s].rho_active)["baryon_current"]
               for s in ("n", "p")}
    total_current = current["n"] + current["p"]
    assert float(torch.linalg.vector_norm(total_current[:, 1:], dim=1).max().detach()) > 0.0
    reference_vertex_density = invariant_baryon_density(total_current).detach()
    full = _evaluate(basis, states)
    fixed = _evaluate(basis, states, freeze=True)
    term_names = tuple(full.audit["included_energy_terms"])
    output_count = len(term_names)
    identity = torch.eye(output_count, dtype=torch.float64)
    input_densities = tuple(states[s].rho_active for s in ("n", "p"))
    full_jacobian = torch.autograd.grad(
        torch.stack([full.energy_terms[k] for k in term_names]), input_densities,
        grad_outputs=identity, is_grads_batched=True, retain_graph=True,
    )
    fixed_jacobian = torch.autograd.grad(
        torch.stack([fixed.energy_terms[k] for k in term_names]), input_densities,
        grad_outputs=identity, is_grads_batched=True,
    )
    directional_ad = torch.stack([
        sum(torch.sum(full_jacobian[i][term].conj() * directions[s]).real
            for i, s in enumerate(("n", "p")))
        for term in range(output_count)
    ])
    rearrangement_ad = torch.stack([
        sum(torch.sum((full_jacobian[i][term] - fixed_jacobian[i][term]).conj()
                      * directions[s]).real
            for i, s in enumerate(("n", "p")))
        for term in range(output_count)
    ])

    eps = 1.0e-5

    def energies(sign):
        trial = {
            s: SimpleNamespace(
                q=states[s].q,
                rho_active=(states[s].rho_active.detach() + sign * eps * directions[s]),
            )
            for s in ("n", "p")
        }
        complete = _evaluate(basis, trial)
        vertex_fixed = _evaluate(basis, trial, vertex_density=reference_vertex_density)
        return complete, vertex_fixed

    plus_full, plus_fixed = energies(+1.0)
    minus_full, minus_fixed = energies(-1.0)
    fd = torch.stack([(plus_full.energy_terms[k] - minus_full.energy_terms[k]) / (2 * eps)
                      for k in term_names])
    rearrangement_fd = torch.stack([
        ((plus_full.energy_terms[k] - plus_fixed.energy_terms[k])
         - (minus_full.energy_terms[k] - minus_fixed.energy_terms[k])) / (2 * eps)
        for k in term_names
    ])

    torch.testing.assert_close(fd, directional_ad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(rearrangement_fd, rearrangement_ad,
                               atol=3e-5, rtol=3e-5)
    for i, name in enumerate(term_names):
        record_property(f"fd_abs_error_{name}", float((fd[i] - directional_ad[i]).abs().detach()))
        record_property(f"rearrangement_fd_abs_error_{name}",
                        float((rearrangement_fd[i] - rearrangement_ad[i]).abs().detach()))
    record_property("density_direction_lmax", 2)

    # The matrix-free primitive normal field must give the same Rayleigh
    # derivative when an independent orbital projector is added to rho.
    normal_states = {
        s: SimpleNamespace(q=states[s].q, rho_active=states[s].rho_active,
                           chi_active=torch.zeros_like(states[s].rho_active))
        for s in ("n", "p")
    }
    def normal_energy(active_basis, tangent_states):
        return evaluate_particle_hole_energy(
            active_basis, tangent_states, chunk_size=128, pair_chunk=2,
            params=pka1_parameters(), lmax=2,
        ).total_energy

    field = AutogradNormalAction(
        basis, normal_states, normal_energy, species="n", vector_chunk=1,
    )
    vector = torch.zeros(basis.dimension, dtype=torch.complex128)
    vector[-1] = 0.8
    vector[0] = 0.6j
    rayleigh = torch.sum(vector.conj() * field.apply(vector)).real
    orbital_eps = 1.0e-5

    def add_orbital_occupation(occupation):
        trial = dict(normal_states)
        state = normal_states["n"]
        q_augmented = torch.cat((state.q, vector[:, None]), dim=1)
        rho_augmented = torch.zeros((3, 3), dtype=torch.complex128)
        rho_augmented[:2, :2] = state.rho_active.detach()
        rho_augmented[2, 2] = occupation
        trial["n"] = SimpleNamespace(q=q_augmented, rho_active=rho_augmented,
                                     chi_active=torch.zeros_like(rho_augmented))
        return normal_energy(basis, trial)

    rayleigh_fd = (add_orbital_occupation(orbital_eps)
                   - normal_energy(basis, normal_states)) / orbital_eps
    torch.testing.assert_close(rayleigh_fd, rayleigh, atol=2e-4, rtol=2e-4)
    assert torch.isfinite(full.total_energy)
    assert all(torch.isfinite(value) for value in full.energy_terms.values())
    required = {
        "sigma_direct", "sigma_fock", "omega_direct", "omega_fock",
        "rho_vector_direct", "rho_vector_fock", "rho_tensor_direct",
        "rho_tensor_fock", "rho_vt_direct", "rho_vt_fock", "rho_tv_direct",
        "rho_tv_fock", "pion_pv_direct", "pion_pv_fock", "coulomb_direct",
        "coulomb_fock", "rho_tensor_contact", "pion_pv_contact",
    }
    assert required.issubset(full.audit["included_energy_terms"])
    # The local derivative test uses lmax=2 for kappa_max=1. It is deliberately
    # not a higher-cutoff convergence study, physical-reference comparison,
    # odd-nucleus validation, or readiness gate.
