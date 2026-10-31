#!/usr/bin/env python3
"""Audit dE_RHF/dP against converged Core occupied PKA1 orbitals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.legacy.dpl_rmf_core import RHFCore
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-density PKA1 RHF Hamiltonian/eigen-equation closure audit"
    )
    parser.add_argument("--z", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a", type=int, default=16)
    parser.add_argument("--core-dir", default="../Core-1204")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--core-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--core-max-iterations", type=int, default=500)
    parser.add_argument("--energy-per-a-tolerance-mev", type=float, default=1.0e-4)
    parser.add_argument("--energy-component-tolerance-mev", type=float, default=1.0e-3)
    parser.add_argument("--level-tolerance-mev", type=float, default=1.0e-3)
    parser.add_argument("--local-potential-tolerance-mev", type=float, default=1.0e-3)
    parser.add_argument("--field-tolerance-fm-inverse", type=float, default=1.0e-6)
    parser.add_argument("--rearrangement-tolerance-fm-inverse", type=float, default=1.0e-7)
    parser.add_argument("--explicit-hd-tolerance-mev", type=float, default=1.0e-10)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _converge_core(args: argparse.Namespace) -> tuple[RHFCore, list[float]]:
    core = RHFCore(args.core_dir)
    core.init("PKA1", args.z, args.n, args.a)
    history = []
    for iteration in range(1, args.core_max_iterations + 1):
        residual = core.step()
        history.append(residual)
        if iteration >= 5 and abs(residual) <= args.core_tolerance:
            break
    return core, history


def _core_blocks(
    backend: DensityMatrixRHFBackend,
    waves: dict[str, np.ndarray],
) -> tuple[list[dict[str, torch.Tensor]], dict[tuple[str, int, int], float]]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    core_energies: dict[tuple[str, int, int], float] = {}
    principals: dict[int, int] = {}
    for index in range(len(waves["energy"])):
        if float(waves["occupancy"][index]) <= 1.0e-12:
            continue
        species = str(waves["species"][index])
        kappa = int(waves["kappa"][index])
        match = re.search(r"\.(\d+)[a-z]", str(waves["name"][index]).lower())
        principal = int(match.group(1)) if match else int(waves["n"][index]) + 1
        principals[index] = principal
        grouped[(species, kappa)].append(index)
        core_energies[(species, kappa, principal)] = float(waves["energy"][index])

    specs = {
        ("n" if spec.species_sign > 0.0 else "p", spec.kappa): spec
        for spec in backend.block_specs
    }
    blocks: list[dict[str, torch.Tensor]] = []
    h = backend.functional.h
    for key, indices in sorted(grouped.items()):
        if key not in specs:
            raise KeyError(f"Core occupied block {key} is absent from the neural model space")
        indices.sort(key=lambda index: principals[index])
        spec = specs[key]
        block_principals = tuple(principals[index] for index in indices)
        if block_principals != spec.principals:
            raise ValueError(
                f"principal-state mismatch for {key}: Core={block_principals}, model={spec.principals}"
            )
        factors = np.stack(
            [
                np.sqrt(h)
                * np.r_[waves["G"][index, 1:-1], waves["F"][index, 1:-1]]
                for index in indices
            ],
            axis=1,
        )
        blocks.append(
            {
                "factors": torch.as_tensor(
                    factors, dtype=torch.float64, device=backend.device
                ).requires_grad_(True),
                "occupations": torch.as_tensor(
                    waves["occupancy"][indices],
                    dtype=torch.float64,
                    device=backend.device,
                ),
                "degeneracy": torch.tensor(
                    float(waves["degeneracy"][indices[0]]),
                    dtype=torch.float64,
                    device=backend.device,
                ),
                "species_sign": torch.tensor(
                    1.0 if key[0] == "n" else -1.0,
                    dtype=torch.float64,
                    device=backend.device,
                ),
                "kappa": torch.tensor(key[1], device=backend.device),
                "principals": torch.as_tensor(
                    block_principals, dtype=torch.int64, device=backend.device
                ),
                "free_hamiltonian": spec.free_hamiltonian,
                "tail_energies_mev": torch.as_tensor(
                    [
                        core_energies[(key[0], key[1], principal)]
                        for principal in block_principals
                    ],
                    dtype=torch.float64,
                    device=backend.device,
                ),
            }
        )
    return blocks, core_energies


def main() -> None:
    args = _arguments()
    if args.a != args.z + args.n:
        raise ValueError("A must equal Z+N")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    core, core_history = _converge_core(args)
    core_observables = core.energy()
    waves = core.wavefunctions()
    backend = DensityMatrixRHFBackend(
        NucleusCase("PKA1", args.z, args.n, args.a), device=device
    )
    blocks, core_energies = _core_blocks(backend, waves)
    result = backend.evaluate(blocks, create_graph=False)

    rows = []
    residual_vectors = []
    for block in blocks:
        species = "n" if float(block["species_sign"]) > 0.0 else "p"
        kappa = int(block["kappa"])
        block_name = f"{species}_kappa_{kappa:+d}"
        hamiltonian = result.hamiltonians[block_name]
        kinetic_hamiltonian = result.kinetic_hamiltonians[block_name]
        direct_hamiltonian = result.direct_hamiltonians[block_name]
        fock_hamiltonian = result.fock_hamiltonians[block_name]
        for column, principal in enumerate(block["principals"].tolist()):
            q = block["factors"][:, column]
            hq = hamiltonian @ q
            norm = torch.dot(q, q)
            rayleigh = torch.dot(q, hq) / norm
            component_expectations = {
                "kinetic_expectation_mev": torch.dot(
                    q, kinetic_hamiltonian @ q
                ) / norm,
                "direct_expectation_mev": torch.dot(
                    q, direct_hamiltonian @ q
                ) / norm,
                "fock_expectation_mev": torch.dot(q, fock_hamiltonian @ q) / norm,
            }
            residual = hq - rayleigh * q
            residual_vectors.append(residual)
            core_level_energy = core_energies[(species, kappa, int(principal))]
            rows.append(
                {
                    "species": species,
                    "kappa": kappa,
                    "principal": int(principal),
                    "core_energy_mev": core_level_energy,
                    "autograd_rayleigh_mev": float(rayleigh.detach().cpu()),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in component_expectations.items()
                    },
                    "component_sum_error_mev": float(
                        (
                            sum(component_expectations.values()) - rayleigh
                        ).detach().cpu()
                    ),
                    "delta_energy_mev": float(rayleigh.detach().cpu()) - core_level_energy,
                    "equation_residual_rms_mev": float(
                        torch.sqrt(torch.mean(residual.square())).detach().cpu()
                    ),
                    "equation_residual_l2_mev": float(
                        torch.linalg.vector_norm(residual).detach().cpu()
                    ),
                    "interior_norm": float(norm.detach().cpu()),
                }
            )

    deltas = np.asarray([row["delta_energy_mev"] for row in rows])
    core_local = core.local_potentials()
    local_python = {
        "vps_n": result.fields["Sigma_plus_n_total_local"],
        "vms_n": result.fields["Sigma_minus_n_total_local"]
        - 2.0 * backend.params.mass_n / backend.params.hbar_c,
        "vtt_n": result.fields["Sigma_T_n_total_local"],
        "vps_p": result.fields["Sigma_plus_p_total_local"],
        "vms_p": result.fields["Sigma_minus_p_total_local"]
        - 2.0 * backend.params.mass_p / backend.params.hbar_c,
        "vtt_p": result.fields["Sigma_T_p_total_local"],
    }
    local_closure = {}
    for name, value in local_python.items():
        difference = value.detach().cpu().numpy()[1:-1] - core_local[name][1:-1]
        local_closure[name] = {
            "max_abs_fm_inverse": float(np.max(np.abs(difference))),
            "rms_fm_inverse": float(np.sqrt(np.mean(difference**2))),
            "max_abs_mev": float(
                backend.params.hbar_c * np.max(np.abs(difference))
            ),
        }

    explicit_hd_closure = {}
    for block in blocks:
        species = "n" if float(block["species_sign"]) > 0.0 else "p"
        kappa = int(block["kappa"])
        name = f"{species}_kappa_{kappa:+d}"
        sigma_plus = result.fields[f"Sigma_plus_{species}_total_local"][1:-1]
        sigma_minus = result.fields[f"Sigma_minus_{species}_total_local"][1:-1]
        sigma_tensor = result.fields[f"Sigma_T_{species}_total_local"][1:-1]
        upper = torch.diag(backend.params.hbar_c * sigma_plus)
        lower = torch.diag(backend.params.hbar_c * sigma_minus)
        tensor = torch.diag(backend.params.hbar_c * sigma_tensor)
        explicit = torch.cat(
            [torch.cat([upper, tensor], dim=1), torch.cat([tensor, lower], dim=1)],
            dim=0,
        )
        difference = result.direct_hamiltonians[name] - explicit
        maximum_flat = int(torch.argmax(torch.abs(difference)).detach().cpu())
        maximum_row, maximum_column = np.unravel_index(
            maximum_flat, tuple(difference.shape)
        )
        split = difference.shape[0] // 2
        bulk = torch.cat(
            [torch.arange(1, split - 1, device=difference.device),
             torch.arange(split + 1, 2 * split - 1, device=difference.device)]
        )
        bulk_difference = difference[bulk][:, bulk]
        explicit_hd_closure[name] = {
            "max_abs_mev": float(torch.max(torch.abs(difference)).detach().cpu()),
            "rms_mev": float(torch.sqrt(torch.mean(difference.square())).detach().cpu()),
            "max_location": [int(maximum_row), int(maximum_column)],
            "bulk_excluding_origin_and_tail_max_abs_mev": float(
                torch.max(torch.abs(bulk_difference)).detach().cpu()
            ),
            "bulk_excluding_origin_and_tail_rms_mev": float(
                torch.sqrt(torch.mean(bulk_difference.square())).detach().cpu()
            ),
        }

    core_fields = core.fields()
    field_closure = {}
    for name in ("sigma", "omega", "rho", "coul"):
        difference = (
            result.fields[name].detach().cpu().numpy()[1:-1]
            - core_fields[name][1:-1]
        )
        field_closure[name] = {
            "max_abs_fm_inverse": float(np.max(np.abs(difference))),
            "rms_fm_inverse": float(np.sqrt(np.mean(difference**2))),
        }

    core_sigr_buffer = (ctypes.c_double * 2000).in_dll(
        core.lib._lib, "__meanfield_MOD_sigr"
    )
    core_sigr = np.ctypeslib.as_array(core_sigr_buffer)[: core.npt].copy()
    rearrangement_closure = {}
    for name in ("sigma_rearr_direct", "sigma_rearr_fock", "sigma_rearr_total"):
        values = result.fields[name].detach().cpu().numpy()
        rearrangement_closure[name] = {}
        if name == "sigma_rearr_total":
            difference = values[1:-1] - core_sigr[1:-1]
            rearrangement_closure[name].update({
                "max_abs_difference_fm_inverse": float(np.max(np.abs(difference))),
                "rms_difference_fm_inverse": float(np.sqrt(np.mean(difference**2))),
            })
        rearrangement_closure[name]["min_max_fm_inverse"] = [
            float(np.min(values[1:-1])), float(np.max(values[1:-1]))
        ]
    rearrangement_closure["core_total"] = {
        "min_max_fm_inverse": [
            float(np.min(core_sigr[1:-1])), float(np.max(core_sigr[1:-1]))
        ]
    }
    direct_rearr = result.fields["sigma_rearr_direct"].detach().cpu().numpy()
    fock_rearr = result.fields["sigma_rearr_fock"].detach().cpu().numpy()
    expected_fock = core_sigr - direct_rearr
    interior_fock = fock_rearr[1:-1]
    interior_expected = expected_fock[1:-1]
    fitted_scale = float(
        np.dot(interior_fock, interior_expected)
        / max(np.dot(interior_fock, interior_fock), 1.0e-300)
    )
    rearrangement_closure["fock_against_core_minus_direct"] = {
        "least_squares_scale": fitted_scale,
        "unscaled_max_abs_difference_fm_inverse": float(
            np.max(np.abs(interior_fock - interior_expected))
        ),
        "scaled_max_abs_difference_fm_inverse": float(
            np.max(np.abs(fitted_scale * interior_fock - interior_expected))
        ),
        "expected_min_max_fm_inverse": [
            float(np.min(interior_expected)), float(np.max(interior_expected))
        ],
    }

    python_energy_components = {
        "kinetic_mev": float(result.action_terms["E_kinetic"].detach().cpu()),
        "direct_mev": float(result.action_terms["E_direct"].detach().cpu()),
        "fock_mev": float(result.action_terms["E_fock"].detach().cpu()),
    }
    core_energy_components = {
        "kinetic_mev": core_observables.e_kin,
        "direct_mev": core_observables.e_dir,
        "fock_mev": core_observables.e_exc,
    }
    energy_component_differences = {
        name: python_energy_components[name] - core_energy_components[name]
        for name in python_energy_components
    }
    maximum_local_error = max(
        values["max_abs_mev"] for values in local_closure.values()
    )
    maximum_field_error = max(
        values["max_abs_fm_inverse"] for values in field_closure.values()
    )
    maximum_hd_error = max(
        values["max_abs_mev"] for values in explicit_hd_closure.values()
    )
    acceptance = {
        "energy_per_a": {
            "value": abs(result.energy_per_a_no_com - core_observables.e_per_A_no_com),
            "tolerance": args.energy_per_a_tolerance_mev,
        },
        "energy_components": {
            "value": max(abs(value) for value in energy_component_differences.values()),
            "tolerance": args.energy_component_tolerance_mev,
        },
        "occupied_levels": {
            "value": float(np.max(np.abs(deltas))),
            "tolerance": args.level_tolerance_mev,
        },
        "local_potentials": {
            "value": maximum_local_error,
            "tolerance": args.local_potential_tolerance_mev,
        },
        "direct_fields": {
            "value": maximum_field_error,
            "tolerance": args.field_tolerance_fm_inverse,
        },
        "total_rearrangement": {
            "value": rearrangement_closure["sigma_rearr_total"][
                "max_abs_difference_fm_inverse"
            ],
            "tolerance": args.rearrangement_tolerance_fm_inverse,
        },
        "explicit_hD_assembly": {
            "value": maximum_hd_error,
            "tolerance": args.explicit_hd_tolerance_mev,
        },
    }
    for check in acceptance.values():
        check["passed"] = bool(check["value"] <= check["tolerance"])
    report = {
        "case": {"model": "PKA1", "z": args.z, "n": args.n, "a": args.a},
        "reference": "RHF-e Eqs. (3.384)-(3.394); Core used only as a read-only closure test",
        "core_iterations": len(core_history),
        "core_final_residual": core_history[-1],
        "python_energy_per_a_no_com_mev": result.energy_per_a_no_com,
        "core_energy_per_a_no_com_mev": core_observables.e_per_A_no_com,
        "python_energy_components": python_energy_components,
        "core_energy_components": core_energy_components,
        "energy_component_differences": energy_component_differences,
        "occupied_level_mae_mev": float(np.mean(np.abs(deltas))),
        "occupied_level_max_abs_mev": float(np.max(np.abs(deltas))),
        "occupied_equation_residual_rms_mev": float(
            torch.sqrt(torch.mean(torch.cat(residual_vectors).square())).detach().cpu()
        ),
        "autograd_projected_stationarity_rms_mev": result.diagnostics[
            "projected_stationarity_rms_mev"
        ],
        "local_direct_hamiltonian_closure": local_closure,
        "explicit_hD_matrix_closure": explicit_hd_closure,
        "direct_field_closure": field_closure,
        "rearrangement_closure": rearrangement_closure,
        "levels": rows,
        "solver_counts": {
            "initialization_block_diagonalizations": backend.initialization_block_diagonalizations,
            "refresh_block_diagonalizations": backend.refresh_block_diagonalizations,
            "training_dirac_solves": backend.training_dirac_solves,
        },
        "acceptance": acceptance,
        "accepted": all(check["passed"] for check in acceptance.values()),
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not report["accepted"]:
        failed = ", ".join(
            name for name, check in acceptance.items() if not check["passed"]
        )
        raise SystemExit(f"RHF Hamiltonian audit failed: {failed}")


if __name__ == "__main__":
    main()
