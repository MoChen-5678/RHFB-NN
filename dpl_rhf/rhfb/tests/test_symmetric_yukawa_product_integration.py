"""Prototype test for symmetric P1 product integration of a bare l=0 kernel."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dpl_rhf.rhf.parameters import pka1_parameters
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.green_propagator_reference import (
    gaussian_yukawa_box_integral,
    gaussian_yukawa_integral,
)
from dpl_rhf.rhfb.spatial import GreenOperator
from dpl_rhf.rhfb.symmetric_yukawa_product_integration import (
    gaussian_reduced_radial_source,
    p1_source_norm,
    p1_yukawa_product_matrix,
)


def test_symmetric_product_integration_preserves_bare_energy_and_gradient():
    params = pka1_parameters()
    radius_fm, width_fm, particle_number = 12.0, 2.0, 1.0
    spacings = (0.4, 0.2, 0.1)
    channels = {
        "sigma": params.m_sigma,
        "omega": params.m_omega,
        "pion": params.m_pi,
        "coulomb": 0.0,
    }
    rows = {name: [] for name in channels}
    rng = np.random.default_rng(20261006)

    for h in spacings:
        basis = AngularRadialBasis(
            radius_fm=radius_fm, spacing_fm=h, kappa_max=1,
            n_theta=3, n_phi=6, derivative_order=5,
        )
        green = GreenOperator(basis, lmax=0, device="cpu")
        radii = torch.as_tensor(
            np.repeat(basis.x, basis.n_angular_points), dtype=torch.float64)
        source = particle_number / (np.pi**1.5*width_fm**3) * torch.exp(
            -(radii/width_fm)**2)
        spatial_weights = torch.as_tensor(basis.weights, dtype=torch.float64)
        old_integral = {
            name: float(torch.sum(spatial_weights * source * green.apply(source, mass)))
            for name, mass in channels.items()
        }

        radial_nodes, c = gaussian_reduced_radial_source(
            radius_fm, h, particle_number, width_fm)
        norm = p1_source_norm(radius_fm, h, c)
        for name, mass in channels.items():
            mu = mass/params.hbar_c
            matrix = p1_yukawa_product_matrix(radius_fm, h, mu, quadrature_order=8)
            asymmetry = float(np.max(np.abs(matrix-matrix.T)))
            product_integral = float(c @ matrix @ c)
            product_energy = 0.5*product_integral
            exact_box = gaussian_yukawa_box_integral(
                particle_number, width_fm, mu, radius_fm)
            infinite = gaussian_yukawa_integral(particle_number, width_fm, mu)

            direction = rng.normal(size=c.shape)
            direction /= np.linalg.norm(direction)
            gradient = matrix @ c
            eps = 2e-6
            e_plus = 0.5*(c+eps*direction) @ matrix @ (c+eps*direction)
            e_minus = 0.5*(c-eps*direction) @ matrix @ (c-eps*direction)
            fd = float((e_plus-e_minus)/(2*eps))
            analytic_directional = float(direction @ gradient)
            derivative_abs_error = abs(fd-analytic_directional)
            derivative_rel_error = derivative_abs_error/max(abs(analytic_directional), 1e-14)

            rows[name].append({
                "spacing_fm": h,
                "radial_intervals": int(round(radius_fm/h)),
                "p1_nodes_including_origin_and_box": int(c.size),
                "unrenormalized_p1_source_norm": norm,
                "source_norm_abs_error": abs(norm-particle_number),
                "production_green_integral_fm_inverse": old_integral[name],
                "p1_product_integral_fm_inverse": product_integral,
                "finite_box_exact_source_reference_fm_inverse": exact_box,
                "infinite_fourier_reference_fm_inverse": infinite,
                "production_abs_error_fm_inverse": abs(old_integral[name]-exact_box),
                "p1_product_abs_error_fm_inverse": abs(product_integral-exact_box),
                "production_relative_error": abs(old_integral[name]-exact_box)/abs(exact_box),
                "p1_product_relative_error": abs(product_integral-exact_box)/abs(exact_box),
                "product_error_improvement_factor": abs(old_integral[name]-exact_box)/
                    max(abs(product_integral-exact_box), 1e-30),
                "energy_fm_inverse": product_energy,
                "matrix_max_abs_asymmetry": asymmetry,
                "energy_directional_fd": fd,
                "energy_directional_exact": analytic_directional,
                "energy_directional_abs_error": derivative_abs_error,
                "energy_directional_rel_error": derivative_rel_error,
                "quadrature_order_per_cell_coordinate": 8,
            })
            assert asymmetry < 2e-15
            assert derivative_rel_error < 2e-8
            assert abs(norm-particle_number) < 1e-12

    for name in channels:
        product_errors = [row["p1_product_relative_error"] for row in rows[name]]
        production_errors = [row["production_relative_error"] for row in rows[name]]
        assert product_errors[2] < product_errors[1] < product_errors[0], (name, product_errors)
        assert production_errors[2] < production_errors[1] < production_errors[0], (
            name, production_errors)

    artifact = Path(__file__).resolve().parents[3] / "outputs" / "validation" / (
        "green_symmetric_product_integration_20261006.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "status": "experimental symmetric radial product-integration prototype; bare l=0 only",
        "literature_context": {
            "RHF_paper_kernel": "Geng-Long 2202.00482 Eq. 9: exp(-m_phi*r)/(4*pi*r), static/no-retardation; photon is Coulomb limit",
            "RHF_e_radial_kernel": "Chapter 3.1 Eqs. 3.15-3.16 spherical harmonic and modified-Bessel expansion",
            "documented_numerical_method": "Poeschl-Vretenar-Ring nucl-th/9706031 uses arbitrary-order finite elements on an adaptive radial mesh for coordinate-space DHB/Klein-Gordon equations; it does not specify this nonlocal Fock product rule",
            "RHF_e_quadrature_method": "Eqs. 3.400-3.407 describe six-point source interpolation with composite Simpson integration; the present symmetric P1 cell-pair integration is a separate derivation",
        },
        "prototype_design": {
            "reduced_radial_unknown": "c_i = r_i*rho(r_i), P1 finite-element nodal samples on [0,R]",
            "bare_l0_bilinear": "I=4*pi int dr dr' c(r)c(r') sinh(mu*r_<) exp(-mu*r_>)/mu; mu=0 kernel r_<",
            "matrix_energy": "I=c^T A c with cell-pair Gauss integration; diagonal cells split into both ordered triangles; off-diagonal blocks inserted with transpose",
            "energy_field_derivative": "for E=0.5*c^T A c, dE/dc=A c; no separately approximated field",
            "source_rescaling": "none",
            "quadrature_order_per_cell_coordinate": 8,
        },
        "source": {
            "N": particle_number,
            "b_fm": width_fm,
            "fixed_box_R_fm": radius_fm,
            "hbar_c_mev_fm": params.hbar_c,
            "meson_masses_mev": channels,
            "radial_spacings_fm": list(spacings),
            "derivative_order_metadata": 5,
            "green_lmax": 0,
        },
        "results_by_channel": rows,
        "scope_limits": [
            "a P1 approximation to r*rho is the only source approximation; no N-dependent or error-tuned renormalization",
            "bare constant-coupling l=0 Green kernel only",
            "does not establish accuracy for nonspherical multipoles, density-dependent vertices, derivative vertices, coincident Hessians, or contact terms",
            "this is a standalone prototype and does not change spatial.py or the production functional",
        ],
    }
    artifact.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
