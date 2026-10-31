"""Fixed-density spherical-reduction check against the existing Python RHF code."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.functional import evaluate_particle_hole_energy
from dpl_rhf.rhf.density import nonlocal_density_from_block
from dpl_rhf.rhf.functional import SphericalRHFFunctional
from dpl_rhf.rhf.parameters import pka1_parameters


_GROUPS = {
    "sigma": (("sigma_direct", "sigma_fock"), ("E_sigma", "E_sigma_fock")),
    "omega": (("omega_direct", "omega_fock"), ("E_omega", "E_omega_fock")),
    "rho_vector": (("rho_vector_direct", "rho_vector_fock"), ("E_rho", "E_rho_fock")),
    # The m-scheme includes RHF-e contact compensation in these groups. The
    # spherical kernels use the equivalent trace-reduced convention inline.
    "rho_tensor": (("rho_tensor_direct", "rho_tensor_fock", "rho_tensor_contact"),
                   ("E_rho_tensor_direct", "E_rho_tensor_fock")),
    "rho_vector_tensor": (("rho_vt_direct", "rho_tv_direct", "rho_vt_fock", "rho_tv_fock"),
                          ("E_rho_vt_direct", "E_rho_vt_fock")),
    "pion_pv": (("pion_pv_direct", "pion_pv_fock", "pion_pv_contact"),
                ("E_pion_fock",)),
    "coulomb": (("coulomb_direct", "coulomb_fock"), ("E_coul", "E_coul_fock")),
}


def _directional_record(m_scheme_value, spherical_value):
    """Compare scalar tangent actions, using absolute error near zero."""
    scale = torch.maximum(m_scheme_value.abs(), spherical_value.abs())
    absolute_error = (m_scheme_value - spherical_value).abs()
    near_zero = float(scale.detach()) < 1e-8
    return {
        "m_scheme_dE_mev": float(m_scheme_value.detach()),
        "spherical_dE_mev": float(spherical_value.detach()),
        "absolute_error_mev": float(absolute_error.detach()),
        "relative_error": (None if near_zero
                           else float((absolute_error / scale).detach())),
        "near_zero_absolute_rule": near_zero,
    }


def _compare_same_0s_density(spacing, radius, *, check_channel_fields=False):
    basis = AngularRadialBasis(
        radius_fm=radius, spacing_fm=spacing, kappa_max=1,
        n_theta=3, n_phi=6,
    )
    r = torch.arange(round(radius / spacing) + 1, dtype=torch.float64) * spacing
    r[0] = 0.001
    upper = r * torch.exp(-r.square() / (2 * 1.3**2))
    lower = 0.035 * r * upper
    norm = torch.sqrt(torch.sum(upper[1:-1].square() + lower[1:-1].square()))
    # Keep the same fixed radial density as the existing energy-reduction
    # check, but make its components independent variables when auditing
    # first variations below.  These variations are radial matrix-element
    # probes; they are not a normalization-constrained HF solve.
    upper = (upper / norm).detach().requires_grad_(check_channel_fields)
    lower = (lower / norm).detach().requires_grad_(check_channel_fields)

    states, spherical_blocks, orbital_blocks = {}, [], []
    for species in ("n", "p"):
        q = torch.zeros((basis.dimension, 2), dtype=torch.complex128)
        for column, twice_m in enumerate((-1, 1)):
            channel = basis.channels[basis.channel_index[(-1, twice_m)]]
            q[channel.offset:channel.offset + basis.n_radial, column] = upper[1:-1]
            q[channel.offset + basis.n_radial:channel.offset + 2 * basis.n_radial,
              column] = lower[1:-1]
        states[species] = SimpleNamespace(
            q=q, rho_active=torch.eye(2, dtype=torch.complex128),
        )
        radial = torch.cat((upper[1:-1], lower[1:-1]))[:, None]
        block = {
            "factors": radial,
            "occupations": torch.ones(1, dtype=torch.float64),
            "degeneracy": 2.0,
            "species_sign": 1.0 if species == "n" else -1.0,
            "kappa": -1,
            "principals": (1,),
        }
        spherical_blocks.append(nonlocal_density_from_block(block, spacing))
        orbital_blocks.append({
            "free_hamiltonian": torch.zeros(
                (2 * basis.n_radial, 2 * basis.n_radial), dtype=torch.float64,
            ),
        })

    params = pka1_parameters()
    spherical = SphericalRHFFunctional(r, 2, 2, params).evaluate(
        orbital_blocks, spherical_blocks,
    )
    general_m = evaluate_particle_hole_energy(
        basis, states, params=params, lmax=2, chunk_size=1024, pair_chunk=2,
    )
    general_m_interaction = general_m.total_energy - general_m.energy_terms["kinetic"]
    relative_error = (general_m_interaction - spherical.total).abs() / spherical.total.abs()
    group_errors, group_energies, group_field_errors = {}, {}, {}
    group_tangent_field_errors = {}
    group_directional_field_errors = {}
    total_field_error = None
    total_tangent_field_error = None
    total_directional_field_errors = {}
    if check_channel_fields:
        m_total_grad = torch.autograd.grad(
            general_m_interaction, (upper, lower), retain_graph=True,
        )
        sph_total_grad = torch.autograd.grad(
            spherical.total, (upper, lower), retain_graph=True,
        )
        # The comparison state is a normalized occupied spinor.  Its physical
        # first variations obey q†δq + δq†q = 0.  A raw Euclidean gradient
        # includes the orbital-normal component, which is absorbed by the
        # particle-number/orbital multiplier and is not a variational field
        # comparison.  Restrict both gradients to the same normalized radial
        # tangent space before comparing their action.
        radial_state = torch.cat([upper[1:-1], lower[1:-1]])

        def radial_tangent(gradients):
            vector = torch.cat([g[1:-1].reshape(-1) for g in gradients])
            normal = radial_state * (
                torch.dot(radial_state, vector)
                / torch.dot(radial_state, radial_state).clamp_min(1e-30)
            )
            return vector - normal

        m_total_vector = torch.cat([g.reshape(-1) for g in m_total_grad])
        sph_total_vector = torch.cat([g.reshape(-1) for g in sph_total_grad])
        total_field_error = torch.linalg.vector_norm(
            m_total_vector - sph_total_vector
        ) / torch.maximum(
            torch.linalg.vector_norm(m_total_vector),
            torch.linalg.vector_norm(sph_total_vector),
        ).clamp_min(1e-10)
        m_total_tangent = radial_tangent(m_total_grad)
        sph_total_tangent = radial_tangent(sph_total_grad)
        total_tangent_field_error = torch.linalg.vector_norm(
            m_total_tangent - sph_total_tangent
        ) / torch.maximum(
            torch.linalg.vector_norm(m_total_tangent),
            torch.linalg.vector_norm(sph_total_tangent),
        ).clamp_min(1e-10)

        # Probe the variational derivative with common smooth directions in
        # the underlying reduced radial functions.  These scalar directional
        # derivatives compare the same δu in both discretizations; unlike a
        # norm of nodal covectors, they do not compare the two quadrature
        # weight patterns as though they were physical fields.
        def normalized_radial_directions():
            x = radial_state
            radius = r[1:-1]
            profile = (radius / 1.3).square()
            mean_profile = torch.sum(x.square() * torch.cat([profile, profile]))
            shape = torch.cat([
                upper[1:-1] * (profile - mean_profile),
                lower[1:-1] * (profile - mean_profile),
            ])
            # A smooth large/small-component rotation.  Its norm constraint
            # is satisfied pointwise before normalization: G*(F*s)+F*(-G*s)=0.
            mix_profile = radius / (1.3 + radius)
            mixing = torch.cat([
                lower[1:-1] * mix_profile,
                -upper[1:-1] * mix_profile,
            ])
            candidates = []
            for candidate in (shape, mixing):
                candidate = candidate - x * torch.dot(x, candidate)
                norm = torch.linalg.vector_norm(candidate)
                if float(norm.detach()) <= 1e-14:
                    raise AssertionError("radial tangent probe is degenerate")
                candidate = candidate / norm
                candidates.append((candidate[:radius.numel()],
                                   candidate[radius.numel():]))
            return {"smooth_shape": candidates[0], "large_small_mix": candidates[1]}

        def directional_value(gradients, direction):
            du, df = direction
            return torch.sum(gradients[0][1:-1] * du) + torch.sum(
                gradients[1][1:-1] * df
            )

        directions = normalized_radial_directions()

        total_directional_field_errors = {
            name: _directional_record(
                directional_value(m_total_grad, direction),
                directional_value(sph_total_grad, direction),
            )
            for name, direction in directions.items()
        }
    for name, (m_keys, spherical_keys) in _GROUPS.items():
        m_energy = sum((general_m.energy_terms[k] for k in m_keys), torch.zeros_like(spherical.total))
        sph_energy = sum((spherical.direct_terms.get(k, spherical.exchange_terms.get(k))
                          for k in spherical_keys), torch.zeros_like(spherical.total))
        channel_error = (m_energy - sph_energy).abs() / sph_energy.abs().clamp_min(1e-12)
        group_errors[name] = float(channel_error.detach())
        group_energies[name] = (float(m_energy.detach()), float(sph_energy.detach()))
        if check_channel_fields:
            # The gradient of each channel's energy along the same G/F
            # radial factors is its field action on this spherical state.
            # This catches a wrong sign, vertex placement, contact factor, or
            # rearrangement derivative that can survive a total-energy sum.
            m_grad = torch.autograd.grad(
                m_energy, (upper, lower), retain_graph=True, allow_unused=True,
            )
            sph_grad = torch.autograd.grad(
                sph_energy, (upper, lower), retain_graph=True, allow_unused=True,
            )
            mg = torch.cat([g.reshape(-1) for g in m_grad if g is not None])
            sg = torch.cat([g.reshape(-1) for g in sph_grad if g is not None])
            field_error = torch.linalg.vector_norm(mg - sg) / torch.maximum(
                torch.linalg.vector_norm(mg), torch.linalg.vector_norm(sg)
            ).clamp_min(1e-10)
            group_field_errors[name] = float(field_error.detach())
            mt = radial_tangent(m_grad)
            st = radial_tangent(sph_grad)
            tangent_error = torch.linalg.vector_norm(mt - st) / torch.maximum(
                torch.linalg.vector_norm(mt), torch.linalg.vector_norm(st)
            ).clamp_min(1e-10)
            group_tangent_field_errors[name] = float(tangent_error.detach())
            group_directional_field_errors[name] = {
                probe_name: _directional_record(
                    directional_value(m_grad, direction),
                    directional_value(sph_grad, direction),
                )
                for probe_name, direction in directions.items()
            }
    return {
        "spherical": float(spherical.total.detach()),
        "general_m": float(general_m_interaction.detach()),
        "relative_error": float(relative_error.detach()),
        "group_errors": group_errors,
        "group_energies": group_energies,
        "group_field_errors": group_field_errors,
        "group_tangent_field_errors": group_tangent_field_errors,
        "group_directional_field_errors": group_directional_field_errors,
        "total_field_error": (None if total_field_error is None
                              else float(total_field_error.detach())),
        "total_tangent_field_error": (
            None if total_tangent_field_error is None
            else float(total_tangent_field_error.detach())
        ),
        "total_directional_field_errors": total_directional_field_errors,
        "pion_hartree": float(general_m.energy_terms["pion_pv_direct"].detach()),
    }


def test_closed_0s_m_scheme_energy_converges_to_spherical_pka1_energy(record_property):
    # The same analytic radial 0s spinor, occupations, and box are used in
    # both codes.  A second box checks boundary sensitivity independently.
    coarse = _compare_same_0s_density(0.125, 6.5, check_channel_fields=True)
    larger_box = _compare_same_0s_density(0.125, 8.0)
    fine = _compare_same_0s_density(0.0625, 6.5, check_channel_fields=True)
    finer = _compare_same_0s_density(0.03125, 6.5, check_channel_fields=True)
    assert abs(coarse["general_m"] - larger_box["general_m"]) < 1e-7
    assert abs(coarse["spherical"] - larger_box["spherical"]) < 1e-7
    assert coarse["relative_error"] < 0.03
    assert fine["relative_error"] < 0.008
    assert max(coarse["group_errors"].values()) < 0.03
    assert max(fine["group_errors"].values()) < 0.007
    # Nodal-gradient norms are diagnostics only: they are covectors carrying
    # different radial quadrature weights in the two implementations.  The
    # smooth-direction scalar actions below are the meaningful finite-mesh
    # closure probes and are recorded across h for convergence review.
    for mesh_result in (coarse, fine, finer):
        assert all(
            math.isfinite(value)
            for direction in mesh_result["total_directional_field_errors"].values()
            for value in direction.values()
            if value is not None
        )
    assert abs(fine["pion_hartree"]) < 1e-12
    record_property("coarse_h0p125_total_relative_error", coarse["relative_error"])
    record_property("fine_h0p0625_total_relative_error", fine["relative_error"])
    record_property("coarse_h0p125_channel_relative_errors", str(coarse["group_errors"]))
    record_property("fine_channel_relative_errors", str(fine["group_errors"]))
    record_property("fine_channel_field_relative_errors", str(fine["group_field_errors"]))
    record_property("fine_channel_tangent_field_relative_errors",
                    str(fine["group_tangent_field_errors"]))
    record_property("fine_total_field_relative_error", fine["total_field_error"])
    record_property("fine_total_tangent_field_relative_error",
                    fine["total_tangent_field_error"])
    record_property("fine_channel_directional_field_relative_errors",
                    str(fine["group_directional_field_errors"]))
    record_property("fine_total_directional_field_relative_errors",
                    str(fine["total_directional_field_errors"]))
    record_property("finer_h0p03125_channel_directional_errors",
                    str(finer["group_directional_field_errors"]))
    record_property("finer_h0p03125_total_directional_errors",
                    str(finer["total_directional_field_errors"]))
    record_property("same_mesh_box_energy_change_mev",
                    abs(coarse["general_m"] - larger_box["general_m"]))
    record_property("pion_hartree_spherical_limit_mev", fine["pion_hartree"])

    mesh_results = {"0.125": coarse, "0.0625": fine, "0.03125": finer}
    channel_mesh_comparison = {}
    for channel in _GROUPS:
        channel_mesh_comparison[channel] = {}
        for probe in ("smooth_shape", "large_small_mix"):
            records = {
                spacing: result["group_directional_field_errors"][channel][probe]
                for spacing, result in mesh_results.items()
            }
            channel_mesh_comparison[channel][probe] = {
                "by_spacing": records,
                "fine_vs_coarse_absolute_error_ratio": (
                    records["0.0625"]["absolute_error_mev"]
                    / max(records["0.125"]["absolute_error_mev"], 1e-30)
                ),
                "finer_vs_coarse_absolute_error_ratio": (
                    records["0.03125"]["absolute_error_mev"]
                    / max(records["0.125"]["absolute_error_mev"], 1e-30)
                ),
            }
    accepted = all(
        (record["relative_error"] is not None and record["relative_error"] < 0.05)
        or (record["relative_error"] is None and record["absolute_error_mev"] < 1e-8)
        for channel in channel_mesh_comparison.values()
        for probe in channel.values()
        for record in [probe["by_spacing"]["0.03125"]]
    )
    # An accidental agreement on one mesh is insufficient. Every nonzero
    # probe must also approach the legacy continuum action as h is refined.
    convergence_passed = all(
        probe["finer_vs_coarse_absolute_error_ratio"] < 0.2
        or probe["by_spacing"]["0.125"]["absolute_error_mev"] < 1.e-8
        for channel in channel_mesh_comparison.values() for probe in channel.values()
    )
    omega_probe = fine["group_directional_field_errors"]["omega"]["smooth_shape"]
    diagnostic = {
        "status": "directional_mesh_closure_diagnostic",
        "directional_field_closure_passed": bool(accepted),
        "directional_mesh_convergence_passed": bool(convergence_passed),
        "physical_result_accepted": False,
        "radial_discretizations": {
            "m_scheme": "primitive coefficients use uniform dr norm; Green sources and volume actions use dr*r^2 angular weights",
            "spherical_reference": "Yukawa source folding uses the legacy six-point half mesh; exchange outer action uses simpson_weights; direct outer action uses dr*r^2",
        },
        "directions": {
            "smooth_shape": "delta(G,F) proportional to (G,F)*(r^2-<r^2>), projected to normalized-state tangent",
            "large_small_mix": "deltaG=F*s, deltaF=-G*s, s=r/(1.3 fm+r), then unit normalized",
            "normalization": "unit Euclidean norm in the primitive radial coefficient space at each h",
        },
        "acceptance_rule": (
            "at h=0.03125 fm, each channel and both shared unit-norm tangent "
            "directions have relative dE mismatch <5%; if both derivatives "
            "are below 1e-8 MeV, require absolute mismatch <1e-8 MeV"
        ),
        "mesh_results": mesh_results,
        "channel_directional_comparison": channel_mesh_comparison,
        "raw_and_projected_nodal_gradient_errors_are_diagnostics_only": True,
        "fault_injection_counterexample": {
            "correct_same_channel_actions": _directional_record(
                torch.tensor(omega_probe["m_scheme_dE_mev"], dtype=torch.float64),
                torch.tensor(omega_probe["spherical_dE_mev"], dtype=torch.float64),
            ),
            "deliberate_channel_sign_flip": _directional_record(
                torch.tensor(-omega_probe["m_scheme_dE_mev"], dtype=torch.float64),
                torch.tensor(omega_probe["spherical_dE_mev"], dtype=torch.float64),
            ),
        },
    }
    assert diagnostic["fault_injection_counterexample"]["correct_same_channel_actions"]["relative_error"] < 0.05
    assert diagnostic["fault_injection_counterexample"]["deliberate_channel_sign_flip"]["relative_error"] > 1.0
    json_path = os.environ.get("PKA1_SPHERICAL_CLOSURE_JSON")
    if json_path:
        destination = Path(json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    record_property("directional_field_closure_passed", accepted)
    record_property("directional_mesh_closure_json_path", json_path or "not requested")
    assert accepted, "directional PKA1 field closure did not reach the declared fine-mesh tolerance"
    assert convergence_passed, "directional fields did not approach the same limit under mesh refinement"
