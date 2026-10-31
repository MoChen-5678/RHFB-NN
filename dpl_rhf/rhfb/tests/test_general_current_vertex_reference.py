"""Independent Gamma-endpoint check of time-odd PKA1 derivative vertices."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.general_current_vertex_reference import general_current_vertex_reference


def _make_states(basis):
    generator = torch.Generator().manual_seed(20261006)
    states = {}
    for species_index, species in enumerate(("n", "p")):
        q = torch.zeros((basis.dimension, 2), dtype=torch.complex128)
        for channel in basis.channels:
            # Restrict to the complete kappa=-1 multiplet. Both Dirac
            # components then have the same total parity; time reversal is
            # broken by the complex active density, not by parity mixing.
            if channel.kappa != -1:
                continue
            for offset, scale in ((channel.offset, 1.0),
                                  (channel.offset + basis.n_radial, 0.035)):
                real = torch.randn((basis.n_radial, 2), generator=generator,
                                   dtype=torch.float64)
                imag = torch.randn((basis.n_radial, 2), generator=generator,
                                   dtype=torch.float64)
                q[offset:offset + basis.n_radial] = scale * (real + 0.6j * imag)
        q = q / torch.linalg.vector_norm(q, dim=0, keepdim=True)
        rho = 0.42 * torch.eye(2, dtype=torch.complex128)
        off = torch.tensor(0.013 + 0.009j * (species_index + 1),
                           dtype=torch.complex128)
        rho[0, 1], rho[1, 0] = off, off.conj()
        states[species] = SimpleNamespace(q=q, rho_active=rho)
    return states


def test_time_odd_derivative_vertices_against_explicit_gamma_endpoints(record_property):
    basis = AngularRadialBasis(
        radius_fm=1.8, spacing_fm=0.2, kappa_max=1,
        n_theta=3, n_phi=6, derivative_order=5,
    )
    states = _make_states(basis)
    check = general_current_vertex_reference(basis, states, lmax=2)
    ref, prod = check["reference"], check["production"]

    assert float(torch.max(torch.linalg.vector_norm(
        ref["spatial_baryon_current"], dim=1))) > 1.0e-8
    assert float(torch.max(torch.linalg.vector_norm(
        ref["spatial_axial_current"], dim=1))) > 1.0e-8
    assert torch.min(ref["rho_b"]) > 0.0

    reference_terms = {
        "rho_tensor_direct": ref["rho_tensor_direct_components"].sum(),
        "rho_vt_direct": ref["rho_vt_direct_components"].sum(),
        "rho_tv_direct": ref["rho_tv_direct_components"].sum(),
        "pion_pv_direct": ref["pion_pv_direct"],
        "rho_tensor_fock": ref["rho_tensor_fock"],
        "rho_vt_fock": ref["rho_vt_fock"],
        "rho_tv_fock": ref["rho_tv_fock"],
        "pion_pv_fock": ref["pion_pv_fock"],
    }
    errors = {}
    for name, expected in reference_terms.items():
        observed = prod[name].detach()
        errors[name] = float((observed - expected).abs())
        assert torch.allclose(observed, expected, atol=2e-9, rtol=2e-9), (
            f"{name}: production={float(observed):.15g}, "
            f"explicit endpoint={float(expected):.15g}, error={errors[name]:.3e}"
        )
        record_property(f"{name}_reference_mev", float(expected))
        record_property(f"{name}_production_mev", float(observed))
        record_property(f"{name}_abs_error_mev", errors[name])

    # Record raw Fock endpoint contractions by ordered species pair, Lorentz
    # component, and derivative axis as a localization oracle.
    for key in ("rho_tensor_fock_species_by_muij",
                "rho_vt_fock_species_by_mu_axis",
                "rho_tv_fock_species_by_mu_axis"):
        record_property(f"{key}_mev", ref[key].tolist())
    for key in ("rho_tensor_fock_by_muij", "rho_vt_fock_by_mu_axis",
                "rho_tv_fock_by_mu_axis", "pion_pv_fock_by_ij"):
        record_property(f"{key}_mev", ref[key].tolist())

    # Record rho-T by Lorentz component and both mixed direct orientations.
    for mu, value in enumerate(ref["rho_tensor_direct_components"]):
        record_property(f"rho_tensor_direct_mu{mu}_mev", float(value))
    for mu, value in enumerate(ref["rho_vt_direct_components"]):
        record_property(f"rho_vt_direct_mu{mu}_mev", float(value))
    for mu, value in enumerate(ref["rho_tv_direct_components"]):
        record_property(f"rho_tv_direct_mu{mu}_mev", float(value))
    record_property("max_channel_abs_error_mev", max(errors.values()))
    record_property("lmax", 2)
    record_property("radial_adf_points", 5)
    record_property("parity_support", "kappa=-1 s_1/2 multiplet only")

    artifact = Path(__file__).resolve().parents[3] / "outputs" / "validation" / (
        "pka1_current_vertex_explicit_gamma_20261006.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "status": "bounded explicit-endpoint algebra check; no physical readiness claim",
        "prior_runs": {
            "latest_case": "this parity-conserving kappa=-1 run, including corrected rho-T, VT/TV, and pion Fock endpoint traces",
            "preserved_general_case": "pka1_current_vertex_explicit_gamma_20261006_general_case.json",
            "preserved_parity_case_before_fock_extension": "pka1_current_vertex_explicit_gamma_20261006_parity_case.json",
            "preserved_mixed_threefold_reference_failure": "pka1_current_vertex_fock_threefold_reference_failure_20261006.json",
            "preserved_initial_endpoint_transpose_failure": "pka1_current_vertex_fock_initial_failure_20261006.json",
        },
        "scope": (
            "small complex time-odd normal density; explicit coordinate-pair Gamma contractions; "
            "rho-T all Lorentz components, both ordered rho-VT/rho-TV direct and Fock terms, and pion-PV"
        ),
        "cutoffs": {"kappa_max": 1, "lmax": 2, "radial_adf_points": 5,
                    "point_count": basis.point_count,
                    "parity_support": "kappa=-1 s_1/2 multiplet only"},
        "nonzero_sources": {
            "max_spatial_baryon_current": float(torch.max(torch.linalg.vector_norm(
                ref["spatial_baryon_current"], dim=1))),
            "max_spatial_axial_current": float(torch.max(torch.linalg.vector_norm(
                ref["spatial_axial_current"], dim=1))),
            "minimum_invariant_density_fm3": float(torch.min(ref["rho_b"])),
        },
        "rho_tensor_direct_by_mu_mev": [float(v) for v in ref["rho_tensor_direct_components"]],
        "rho_vt_direct_by_mu_mev": [float(v) for v in ref["rho_vt_direct_components"]],
        "rho_tv_direct_by_mu_mev": [float(v) for v in ref["rho_tv_direct_components"]],
        "fock_reference_mev": {
            "rho_tensor": float(ref["rho_tensor_fock"]),
            "rho_vt": float(ref["rho_vt_fock"]),
            "rho_tv": float(ref["rho_tv_fock"]),
            "pion_pv": float(ref["pion_pv_fock"]),
        },
        "fock_production_mev": {
            name: float(prod[name].detach())
            for name in ("rho_tensor_fock", "rho_vt_fock", "rho_tv_fock", "pion_pv_fock")
        },
        "fock_raw_by_species_mu_axis": {
            "species_order": ["n", "p"],
            "rho_tensor_mu_i_j_mev": ref["rho_tensor_fock_species_by_muij"].tolist(),
            "rho_vt_mu_i_mev": ref["rho_vt_fock_species_by_mu_axis"].tolist(),
            "rho_tv_mu_i_mev": ref["rho_tv_fock_species_by_mu_axis"].tolist(),
            "rho_vt_total_mu_i_mev": ref["rho_vt_fock_by_mu_axis"].tolist(),
            "rho_tv_total_mu_i_mev": ref["rho_tv_fock_by_mu_axis"].tolist(),
        },
        "reference_terms_mev": {k: float(v) for k, v in reference_terms.items()},
        "absolute_errors_mev": errors,
        "max_absolute_error_mev": max(errors.values()),
        "tolerances": {"atol_mev": 2e-9, "rtol": 2e-9},
        "limitations": [
            "the raw projected Green kernel is shared; this independently checks endpoint Gamma algebra and signs, not the propagator construction",
            "the bounded Fock comparison checks discrete endpoint contractions only; it does not certify continuum/contact closure",
            "lmax=2 and kappa_max=1 are a bounded algebra benchmark, not angular convergence",
            "no stationarity, solver readiness, or full nuclear physical closure is certified",
        ],
    }
    artifact.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
