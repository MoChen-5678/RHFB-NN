"""Radial-grid check of the projected Yukawa Green operator.

This diagnostic compares ``GreenOperator.apply`` on a spherical Gaussian
source with its infinite-space screened-Gaussian convolution.  It probes only
the propagator and radial quadrature; it does not evaluate a nuclear density,
PKA1 coupling, self-energy, or self-consistent state.  The motivation that a
coarse radial mesh could distort the relative short-range omega contribution
is a hypothesis to test, not a conclusion about the archived O17 result.

The real-space convention is K_mu(s)=exp(-mu*s)/(4*pi*s), mu=m/(hbar*c).
For the source rho(r)=exp(-r^2/a^2), its infinite-space convolution is

  Phi(r) = a^3 sqrt(pi)/(8r) exp((mu*a/2)^2) [
             exp(-mu*r) erfc(mu*a/2-r/a)
           - exp( mu*r) erfc(mu*a/2+r/a) ].

The implementation evaluates the equivalent erfcx expression to avoid
overflow/cancellation.  ``SphericalPropagators`` uses the radial coefficient
mu*i_l(mu*r_<)*k_l(mu*r_>), with k_0(x)=exp(-x)/x; together with the
orthonormal spherical harmonics this has the same 1/(4*pi) normalization.
See Geng and Long, arXiv:2202.00482, Eq. (9) and Appendix A for the Yukawa
propagator expansion used in the RHFB implementation:
https://arxiv.org/abs/2202.00482
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import erfcx

from .basis import AngularRadialBasis
from .execution_limits import require_bounded_cgroup
from .spatial import GreenOperator
from dpl_rhf.rhf.parameters import pka1_parameters


def screened_gaussian_potential(radius_fm: Any, *, width_fm: float,
                                inverse_range_fm: float) -> np.ndarray:
    """Infinite-space ``K_mu * exp(-r^2/a^2)`` for ``K=e^-mu*r/(4*pi*r)``."""
    r = np.asarray(radius_fm, dtype=np.float64)
    if not np.isfinite(r).all() or np.any(r <= 0.0):
        raise ValueError("radius_fm must contain finite positive radii")
    if not math.isfinite(width_fm) or width_fm <= 0.0:
        raise ValueError("width_fm must be finite and positive")
    if not math.isfinite(inverse_range_fm) or inverse_range_fm <= 0.0:
        raise ValueError("inverse_range_fm must be finite and positive")

    scaled_mass = 0.5 * inverse_range_fm * width_fm
    scaled_radius = r / width_fm
    # exp(t^2 +/- mu*r) erfc(t +/- r/a) simplifies to
    # exp(-r^2/a^2) erfcx(t +/- r/a), avoiding large exponentials.
    bracket = (erfcx(scaled_mass - scaled_radius)
               - erfcx(scaled_mass + scaled_radius))
    return (width_fm**3 * math.sqrt(math.pi) / (8.0 * r)
            * np.exp(-scaled_radius**2) * bracket)


def _gaussian_radial_measure_tail_fraction(radius_fm: float,
                                           width_fm: float) -> float:
    """Fraction of ``int r^2 exp(-r^2/a^2) dr`` beyond ``radius_fm``."""
    x = radius_fm / width_fm
    return float(math.erfc(x) + 2.0 * x / math.sqrt(math.pi) * math.exp(-x*x))


def run_yukawa_mesh_diagnostic(output_path: str | Path, *,
                               radius_fm: float = 8.0,
                               spacings_fm: tuple[float, ...] =
                               (0.8, 0.4, 0.2, 0.1, 0.05),
                               width_fm: float = 2.0,
                               common_radii_fm: tuple[float, ...] =
                               (0.8, 1.6, 2.4, 3.2, 4.0)) -> dict[str, Any]:
    """Run the guarded propagator-only mesh comparison and write JSON.

    The cgroup guard is deliberately the first runtime action, before basis,
    Green operator, or mesh-sized tensor construction.
    """
    limits = require_bounded_cgroup()
    started = time.monotonic()
    torch.set_num_threads(2)

    params = pka1_parameters()
    masses = {"sigma": params.m_sigma, "omega": params.m_omega}
    records: list[dict[str, Any]] = []
    mesh_records: list[dict[str, Any]] = []
    for spacing in spacings_fm:
        intervals = int(round(radius_fm / spacing))
        if not math.isclose(intervals * spacing, radius_fm,
                            rel_tol=0.0, abs_tol=1e-11):
            raise ValueError("every radial spacing must divide radius_fm")

        # K=1 supplies a valid basis object; lmax=0 is exact for this spherical
        # source, whose higher multipoles vanish analytically.
        basis = AngularRadialBasis(
            radius_fm=radius_fm,
            spacing_fm=spacing,
            kappa_max=1,
            n_theta=3,
            n_phi=6,
            derivative_order=5,
            hbar_c_mev_fm=params.hbar_c,
        )
        green = GreenOperator(basis, lmax=0, device="cpu")
        radial_source = np.exp(-basis.x**2 / width_fm**2)
        source = torch.as_tensor(
            np.repeat(radial_source, basis.n_angular_points), dtype=torch.float64
        )
        common_values: dict[str, dict[float, float]] = {
            name: {} for name in masses
        }
        angular_spread: dict[float, dict[str, float]] = {}

        for name, mass_mev in masses.items():
            potential = green.apply(source, mass_mev).reshape(
                basis.n_radial, basis.n_angular_points
            ).cpu().numpy()
            mu = mass_mev / params.hbar_c
            for requested_r in common_radii_fm:
                radial_index = int(round(requested_r / spacing)) - 1
                if radial_index < 0 or radial_index >= basis.n_radial:
                    raise ValueError(f"common radius {requested_r} is not on this mesh")
                actual_r = float(basis.x[radial_index])
                if not math.isclose(actual_r, requested_r,
                                    rel_tol=0.0, abs_tol=1e-10):
                    raise ValueError(f"common radius {requested_r} is not on this mesh")
                discrete = float(potential[radial_index, 0])
                analytic = float(screened_gaussian_potential(
                    actual_r, width_fm=width_fm, inverse_range_fm=mu
                ))
                common_values[name][requested_r] = discrete
                records.append({
                    "spacing_fm": spacing,
                    "radius_fm": requested_r,
                    "meson": name,
                    "mass_mev": mass_mev,
                    "inverse_range_fm": mu,
                    "discrete_green_convolution_fm2": discrete,
                    "infinite_space_analytic_fm2": analytic,
                    "relative_error": discrete / analytic - 1.0,
                    "angular_spread_max_abs_fm2": float(
                        np.max(np.abs(potential[radial_index]
                                      - potential[radial_index, 0]))
                    ),
                })
                angular_spread.setdefault(requested_r, {})[name] = float(
                    np.max(np.abs(potential[radial_index]
                                  - potential[radial_index, 0]))
                )

        for requested_r in common_radii_fm:
            discrete_ratio = (common_values["omega"][requested_r]
                              / common_values["sigma"][requested_r])
            analytic_sigma = float(screened_gaussian_potential(
                requested_r, width_fm=width_fm,
                inverse_range_fm=masses["sigma"] / params.hbar_c))
            analytic_omega = float(screened_gaussian_potential(
                requested_r, width_fm=width_fm,
                inverse_range_fm=masses["omega"] / params.hbar_c))
            analytic_ratio = analytic_omega / analytic_sigma
            records.append({
                "spacing_fm": spacing,
                "radius_fm": requested_r,
                "quantity": "omega_over_sigma_propagator_ratio",
                "discrete_ratio": discrete_ratio,
                "analytic_ratio": analytic_ratio,
                "relative_error": discrete_ratio / analytic_ratio - 1.0,
            })

        mesh_records.append({
            "spacing_fm": spacing,
            "radial_points_including_endpoints": intervals + 1,
            "interior_radial_points": basis.n_radial,
            "angular_points": basis.n_angular_points,
            "green_lmax": 0,
            "source_radial_measure_fraction_beyond_last_interior_point":
                _gaussian_radial_measure_tail_fraction(
                    float(basis.x[-1]), width_fm
                ),
            "common_radius_angular_spread_max_abs_fm2": angular_spread,
        })
        del green, basis, source

    report = {
        "status": "completed_propagator_only_diagnostic",
        "physical_result_accepted": False,
        "scope": (
            "spherical dimensionless Gaussian source convolved with sigma/omega "
            "Yukawa Green operators; no couplings, nuclear density, self-energy, "
            "pairing, training, or self-consistent fields"
        ),
        "hypothesis_under_test": (
            "a coarse radial mesh could distort the relative short-range omega "
            "propagator contribution; this is not established as an O17 field error"
        ),
        "kernel_convention": "exp(-mu*s)/(4*pi*s)",
        "analytic_reference": "infinite-space screened Gaussian, erfcx-stable form",
        "analytic_formula_source": "direct convolution of the declared real-space kernel and Gaussian source",
        "propagator_reference": (
            "Geng and Long, arXiv:2202.00482, Eq. (9), Appendix A; "
            "https://arxiv.org/abs/2202.00482"
        ),
        "kernel_normalization_check": (
            "SphericalPropagators returns mu*i_l(mu*r_<)*k_l(mu*r_>); "
            "with k_0(x)=exp(-x)/x and orthonormal Y_lm this expands "
            "exp(-mu*s)/(4*pi*s), without fitted normalization"
        ),
        "source": {"form": "exp(-r^2/a^2)", "width_fm": width_fm,
                   "dimension": "dimensionless"},
        "radius_fm": radius_fm,
        "common_radii_fm": list(common_radii_fm),
        "spacings_fm": list(spacings_fm),
        "green_multipole_cutoff": 0,
        "basis_angular_grid": {"kappa_max": 1, "n_theta": 3, "n_phi": 6},
        "radial_derivative_order": 5,
        "hbar_c_mev_fm": params.hbar_c,
        "meson_masses_mev": masses,
        "finite_box_note": (
            "The numerical source is sampled only on interior radii through R-h; "
            "the analytic reference includes the infinite Gaussian tail. The "
            "reported omitted radial source-measure fraction is not a field-error "
            "estimate. For r<=4 fm the massive Yukawa kernel further suppresses "
            "the outer source tail; the mismatch is nevertheless retained in "
            "the comparison rather than corrected away."
        ),
        "kernel_limits": limits,
        "mesh_records": mesh_records,
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="destination JSON report (must be a new/replaceable file)")
    args = parser.parse_args(argv)
    run_yukawa_mesh_diagnostic(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
