"""Bounded normalization/mesh check for the projected bare Green operator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dpl_rhf.rhf.parameters import pka1_parameters
from dpl_rhf.rhfb.basis import AngularRadialBasis
from dpl_rhf.rhfb.green_propagator_reference import (
    gaussian_yukawa_box_integral,
    gaussian_yukawa_fourier_quad,
    gaussian_yukawa_integral,
)
from dpl_rhf.rhfb.spatial import GreenOperator


def test_bare_green_against_independent_gaussian_fourier_reference():
    params = pka1_parameters()
    radius_fm, width_fm, particle_number = 12.0, 2.0, 1.0
    spacings = (0.8, 0.4, 0.2, 0.1, 0.05)
    channels = {
        "sigma": params.m_sigma,
        "omega": params.m_omega,
        "pion": params.m_pi,
        "coulomb": 0.0,
    }
    inverse_ranges = {name: mass / params.hbar_c for name, mass in channels.items()}
    infinite = {name: gaussian_yukawa_integral(
        particle_number, width_fm, inverse_ranges[name]) for name in channels}
    fourier_quad = {name: gaussian_yukawa_fourier_quad(
        particle_number, width_fm, inverse_ranges[name]) for name in channels}
    box = {name: gaussian_yukawa_box_integral(
        particle_number, width_fm, inverse_ranges[name], radius_fm)
           for name in channels}

    rows = {name: [] for name in channels}
    for h in spacings:
        basis = AngularRadialBasis(
            radius_fm=radius_fm, spacing_fm=h, kappa_max=1,
            n_theta=3, n_phi=6, derivative_order=5,
        )
        green = GreenOperator(basis, lmax=0, device="cpu")
        radii = torch.as_tensor(
            np.repeat(basis.x, basis.n_angular_points), dtype=torch.float64)
        source = particle_number / (np.pi**1.5 * width_fm**3) * torch.exp(
            -(radii / width_fm)**2)
        weights = torch.as_tensor(basis.weights, dtype=torch.float64)
        mesh_particle_number = float(torch.sum(weights * source))
        for name, mass in channels.items():
            potential = green.apply(source, mass)
            mesh_integral = float(torch.sum(weights * source * potential))
            rows[name].append({
                "spacing_fm": h,
                "mesh_integral_fm_inverse": mesh_integral,
                "mesh_energy_fm_inverse": 0.5 * mesh_integral,
                "mesh_source_integral": mesh_particle_number,
                "abs_source_normalization_error": abs(mesh_particle_number-particle_number),
                "infinite_fourier_reference_fm_inverse": infinite[name],
                "finite_box_reference_fm_inverse": box[name],
                "abs_error_vs_box_fm_inverse": abs(mesh_integral - box[name]),
                "rel_error_vs_box": abs(mesh_integral - box[name]) / abs(box[name]),
                "abs_error_vs_infinite_fm_inverse": abs(mesh_integral - infinite[name]),
                "relative_box_truncation": abs(box[name] - infinite[name]) / abs(infinite[name]),
                "radial_points": basis.n_radial,
                "angular_points": basis.n_angular_points,
                "green_lmax": 0,
                "derivative_order_metadata": basis.derivative_order,
                "green_adf_stencil_points": 5,
            })

    for name in channels:
        assert abs(infinite[name] - fourier_quad[name]) < 2e-12
        assert abs(box[name] - infinite[name]) / abs(infinite[name]) < 2e-9
        errors = [row["abs_error_vs_box_fm_inverse"] for row in rows[name]]
        assert errors[-1] < errors[0], (name, errors)
        assert errors[-1] / abs(box[name]) < 1e-2, (name, errors[-1] / abs(box[name]))

    artifact = Path(__file__).resolve().parents[3] / "outputs" / "validation" / (
        "green_gaussian_propagator_20261006.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "status": "bounded bare-propagator normalization and radial-grid check; not derivative/contact closure",
        "reference": {
            "density": "N/(pi^(3/2)b^3) exp(-r^2/b^2)",
            "particle_number": particle_number,
            "width_fm": width_fm,
            "fixed_box_radius_fm": radius_fm,
            "kernel": "exp(-mu*r)/(4*pi*r); Coulomb at mu=0",
            "fourier_integral": "N^2/(2*pi^2) int_0^inf dq q^2 exp(-a q^2)/(q^2+mu^2), a=b^2/2",
            "analytic_integral_fm_inverse": infinite,
            "independent_fourier_quad_fm_inverse": fourier_quad,
            "independent_finite_box_radial_integral_fm_inverse": box,
        },
        "meson_masses_mev": channels,
        "hbar_c_mev_fm": params.hbar_c,
        "grid": {
            "radial_spacings_fm": list(spacings),
            "derivative_order_metadata": 5,
            "radial_green_derivative_stencil_points": 5,
            "green_lmax": 0,
            "angular_quadrature": "n_theta=3, n_phi=6; source is exactly spherical",
        },
        "results_by_channel": rows,
        "scope_limits": [
            "constant-coupling, spherical l=0 source only; no density-dependent vertex or rearrangement",
            "this tests bare sigma/omega/pion/Coulomb Green normalization and radial quadrature only",
            "no derivative Green, rho-T, mixed VT/TV, pion-PV derivative, contact, or Hessian closure is tested",
            "production spatial.py is only called for the comparison; the analytic and quadrature references are independent",
        ],
    }
    artifact.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
