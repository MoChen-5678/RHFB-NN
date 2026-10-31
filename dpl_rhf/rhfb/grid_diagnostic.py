"""Radial-grid convergence diagnostic for the experimental PKA1+D1S energy.

This is an independent quadrature study, not a training or physical-acceptance
path. It evaluates the same deterministic O-17 oscillator initialization on
several radial meshes and records every functional term and integrated radius.
No Dirac/Bogoliubov eigensolve or Fortran backend is used.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .basis import AngularRadialBasis
from .config import BlockedRHFBConfig
from .currents import local_bilinears
from .functional import evaluate_rhfb_energy
from .operators import DensityFactors
from .trainer import prepare_networks
from dpl_rhf.rhf.parameters import pka1_parameters


def _initial_states(config: BlockedRHFBConfig, basis: AngularRadialBasis):
    """Recreate the O-17 spherical, paired start used by experimental_run."""
    torch.manual_seed(config.seed)
    networks, blocks = prepare_networks(config, basis, device="cpu")
    states = {}
    with torch.no_grad():
        for species, data in networks.items():
            q, gd, _ = data["network"].forward_with_blocked_mode()
            states[species] = DensityFactors(q, gd.rho, gd.chi)
    return states, blocks


def _number_and_radius(basis: AngularRadialBasis, states: dict[str, DensityFactors]):
    device = next(iter(states.values())).q.device
    weights = torch.as_tensor(basis.weights, dtype=torch.float64, device=device)
    radius2 = torch.as_tensor(basis.positions, dtype=torch.float64, device=device).square().sum(-1)
    values = {}
    for species, state in states.items():
        spinors = basis._sample_range(state.q, 0, basis.point_count)
        density = local_bilinears(spinors, state.rho_active)["baryon_current"][:, 0]
        number = torch.sum(weights * density)
        mean_square = torch.sum(weights * density * radius2) / number
        values[species] = {
            "particle_number": float(number),
            "rms_radius_fm": math.sqrt(max(float(mean_square), 0.0)),
        }
    matter_number = values["n"]["particle_number"] + values["p"]["particle_number"]
    # Species radii combine by their integrated particle-number weights.
    matter_r2 = sum(values[s]["particle_number"] * values[s]["rms_radius_fm"]**2
                    for s in ("n", "p")) / matter_number
    return {"species": values, "matter_rms_radius_fm": math.sqrt(max(matter_r2, 0.0))}


def _yukawa_radial_quadrature_probe(boxes, spacings):
    """Compare the actual radial mesh rule with a high-order integral oracle.

    The fixed analytic spherical source makes this probe independent of how
    oscillator orbitals change between meshes. It uses the same l=0 Yukawa
    propagator as GreenOperator, but integrates both radial variables with a
    600-point Gauss-Legendre rule for the reference value.
    """
    p = pka1_parameters()
    masses = {"sigma": p.m_sigma, "omega": p.m_omega,
              "rho": p.m_rho, "pion": p.m_pi}
    width_fm = 1.2
    report = {"source": "rho(r)=exp(-r^2/(2*(1.2 fm)^2))",
              "multipole": 0, "angular_factor": "omitted equally in all ratios",
              "high_order_reference": "600-point Gauss-Legendre on [0, box]",
              "integrals": {}}
    nodes, weights = np.polynomial.legendre.leggauss(600)
    for box in boxes:
        r_ref = 0.5 * box * (nodes + 1.0)
        w_ref = 0.5 * box * weights
        from dpl_rhf.rhf.kernels import SphericalPropagators
        ref_kernel = SphericalPropagators(
            torch.as_tensor(r_ref, dtype=torch.float64), p.hbar_c
        )
        rho_ref = np.exp(-r_ref**2 / (2.0 * width_fm**2))
        radial_measure_ref = w_ref * r_ref**2 * rho_ref
        reference_values = {}
        for name, mass in masses.items():
            kernel = ref_kernel.yukawa(0, mass).cpu().numpy()
            reference_values[name] = float(radial_measure_ref @ kernel @ radial_measure_ref)
        for spacing in spacings:
            radii = np.arange(spacing, box, spacing, dtype=np.float64)
            radial_measure = (spacing * radii**2
                              * np.exp(-radii**2 / (2.0 * width_fm**2)))
            propagator = SphericalPropagators(
                torch.as_tensor(radii, dtype=torch.float64), p.hbar_c
            )
            for name, mass in masses.items():
                kernel = propagator.yukawa(0, mass).cpu().numpy()
                value = float(radial_measure @ kernel @ radial_measure)
                key = f"box={box:g}_h={spacing:g}_{name}"
                report["integrals"][key] = {
                    "mass_mev": mass,
                    "inverse_range_fm": mass / p.hbar_c,
                    "range_fm": p.hbar_c / mass,
                    "discrete_radial_integral": value,
                    "gauss_legendre_reference": reference_values[name],
                    "relative_error": value / reference_values[name] - 1.0,
                }
    return report


def _evaluate(radius_fm: float, spacing_fm: float, *, seed: int,
              chunk_size: int, pair_chunk: int) -> dict:
    config = BlockedRHFBConfig(
        z=8, n=9, radius_fm=radius_fm, spacing_fm=spacing_fm,
        kappa_max=3, active_n=14, active_p=14, epochs=1, seed=seed,
        chunk_size=chunk_size, pair_chunk=pair_chunk, lmax=6,
    )
    basis = AngularRadialBasis(
        radius_fm=radius_fm, spacing_fm=spacing_fm, kappa_max=3,
        n_theta=7, n_phi=14, derivative_order=5,
    )
    states, blocks = _initial_states(config, basis)
    started = time.monotonic()
    with torch.no_grad():
        evaluation = evaluate_rhfb_energy(
            basis, states, chunk_size=chunk_size, pair_chunk=pair_chunk, lmax=6,
        )
        moments = _number_and_radius(basis, states)
    elapsed = time.monotonic() - started
    terms = {key: float(value.detach()) for key, value in evaluation.energy_terms.items()}
    return {
        "radius_fm": radius_fm,
        "spacing_fm": spacing_fm,
        "nucleus": "O17",
        "seed": seed,
        "kappa_max": basis.kappa_max,
        "active_n": 14,
        "active_p": 14,
        "lmax": 6,
        "n_theta": basis.n_theta,
        "n_phi": basis.n_phi,
        "n_radial": basis.n_radial,
        "spatial_points": basis.point_count,
        "primitive_dimension": basis.dimension,
        "runtime_seconds": elapsed,
        "total_energy_mev": float(evaluation.total_energy.detach()),
        "energy_terms_mev": terms,
        "moments": moments,
        "blocking": {key: (None if value is None else {
            "label": value.label, "index": value.index, "kappa": value.kappa,
            "m2": value.m2,
        }) for key, value in blocks.items()},
        "audit_status": "experimental_unvalidated",
        "calculation_stage": "pure_initial_state_before_optimizer_update",
        "optimizer_updates": 0,
        "physical_result_accepted": False,
        "backend": "Python/PyTorch CPU",
        "fortran_used": False,
        "eigensolver_used": False,
    }


def run(output: str | Path, *, boxes=(8.0, 12.0), spacings=(0.8, 0.4, 0.2),
        seed=20261002, chunk_size=128, pair_chunk=14) -> dict:
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for radius in boxes:
        for spacing in spacings:
            print(f"grid box={radius:g} h={spacing:g} fm: starting", flush=True)
            row = _evaluate(radius, spacing, seed=seed, chunk_size=chunk_size,
                            pair_chunk=pair_chunk)
            rows.append(row)
            print(f"grid box={radius:g} h={spacing:g} fm: "
                  f"E={row['total_energy_mev']:.9f} MeV "
                  f"points={row['spatial_points']} "
                  f"runtime={row['runtime_seconds']:.2f}s", flush=True)
            del row
            gc.collect()

    comparisons = {}
    for radius in boxes:
        group = sorted((row for row in rows if row["radius_fm"] == radius),
                       key=lambda row: row["spacing_fm"], reverse=True)
        finest = group[-1]
        term_deltas = {}
        for row in group[:-1]:
            key = f"h={row['spacing_fm']:g}_vs_h={finest['spacing_fm']:g}"
            deltas = {term: row["energy_terms_mev"][term]
                      - finest["energy_terms_mev"][term]
                      for term in finest["energy_terms_mev"]}
            term_deltas[key] = {
                "total_energy_delta_mev": row["total_energy_mev"] - finest["total_energy_mev"],
                "energy_term_delta_mev": deltas,
                "largest_absolute_term_delta": max(deltas,
                    key=lambda term: abs(deltas[term])),
                "sigma_omega_direct_fock_delta_l1_mev": sum(
                    abs(deltas[name]) for name in ("sigma_direct", "sigma_fock",
                                                    "omega_direct", "omega_fock")),
            }
        comparisons[str(radius)] = {
            "finest_spacing_used_as_reference_fm": finest["spacing_fm"],
            "coarse_to_fine": term_deltas,
        }
    payload = {
        "format": "o17-radial-grid-diagnostic-v1",
        "status": "quadrature_diagnostic_only",
        "calculation_stage": "pure_initial_state_before_optimizer_update",
        "optimizer_updates": 0,
        "functional": "experimental Python/PyTorch PKA1+D1S",
        "threads": torch.get_num_threads(),
        "configuration": {"boxes_fm": list(boxes), "spacings_fm": list(spacings),
                          "seed": seed, "chunk_size": chunk_size,
                          "pair_chunk": pair_chunk, "kappa_max": 3,
                          "active_n": 14, "active_p": 14, "lmax": 6,
                          "angular_grid_fixed": [7, 14]},
        "interpretation_limit": (
            "The finest radial mesh is a comparison reference, not an exact answer; "
            "all terms remain physically uncertified."),
        "comparisons_by_box_fm": comparisons,
        "fixed_source_yukawa_quadrature_probe": _yukawa_radial_quadrature_probe(
            boxes, spacings),
        "runs": rows,
    }
    (out / "grid_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = out / "grid_diagnostic.csv"
    columns = ["radius_fm", "spacing_fm", "n_radial", "spatial_points",
               "primitive_dimension", "runtime_seconds", "total_energy_mev",
               "neutron_number", "proton_number", "neutron_rms_radius_fm",
               "proton_rms_radius_fm", "matter_rms_radius_fm"]
    term_names = list(rows[0]["energy_terms_mev"]) if rows else []
    columns += [f"energy_{name}_mev" for name in term_names]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            moment = row["moments"]
            flat = {
                "radius_fm": row["radius_fm"], "spacing_fm": row["spacing_fm"],
                "n_radial": row["n_radial"], "spatial_points": row["spatial_points"],
                "primitive_dimension": row["primitive_dimension"],
                "runtime_seconds": row["runtime_seconds"],
                "total_energy_mev": row["total_energy_mev"],
                "neutron_number": moment["species"]["n"]["particle_number"],
                "proton_number": moment["species"]["p"]["particle_number"],
                "neutron_rms_radius_fm": moment["species"]["n"]["rms_radius_fm"],
                "proton_rms_radius_fm": moment["species"]["p"]["rms_radius_fm"],
                "matter_rms_radius_fm": moment["matter_rms_radius_fm"],
            }
            flat.update({f"energy_{name}_mev": value
                         for name, value in row["energy_terms_mev"].items()})
            writer.writerow(flat)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--boxes", type=float, nargs="+", default=(8.0, 12.0))
    parser.add_argument("--spacings", type=float, nargs="+", default=(0.8, 0.4, 0.2))
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--pair-chunk", type=int, default=14)
    args = parser.parse_args(argv)
    run(args.out, boxes=tuple(args.boxes), spacings=tuple(args.spacings),
        seed=args.seed, chunk_size=args.chunk_size, pair_chunk=args.pair_chunk)


if __name__ == "__main__":
    main()
