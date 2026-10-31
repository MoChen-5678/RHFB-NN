#!/usr/bin/env python3
"""Evaluate Python PKO2 RHF on the exact occupied density exported by Core-1204."""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dpl_rhf.rhf.density import nonlocal_density_from_block
from dpl_rhf.rhf.functional import PKO2RHFFunctional
from dpl_rhf.rhf.parameters import rhf_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", type=Path, default=Path("../Core-1204"))
    parser.add_argument("--z", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a", type=int, default=16)
    parser.add_argument("--model", choices=("PKO2", "PKA1"), default="PKO2")
    parser.add_argument(
        "--core-state",
        choices=("converged", "initial"),
        default="converged",
        help="Core state used for the read-only closure audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.core_dir.resolve()))
    import rhf_ctypes as core  # type: ignore[import-not-found]

    core_call = core.ddrhf_run if args.core_state == "converged" else core.ddrhf_init
    core_call(2 if args.model == "PKO2" else 0, args.z, args.n, args.a)
    reference = core.ddrhf_get_observables()
    groups: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for species in (1, 2):
        for index in range(1, core.ddrhf_get_norbits(species) + 1):
            g, f, name, kappa, _, principal, _, occupation, degeneracy, ok = (
                core.ddrhf_extract_wavefunction(index, species)
            )
            if ok and occupation > 1.0e-12:
                groups[(species, kappa)].append(
                    (principal, occupation, degeneracy, g[:201], f[:201], name)
                )

    radius = np.arange(201, dtype=np.float64) * 0.1
    radius[0] = 0.001
    functional = PKO2RHFFunctional(
        torch.as_tensor(radius), args.z, args.n, rhf_parameters(args.model)
    )
    blocks = []
    for (species, kappa), states in sorted(groups.items()):
        factors = [
            np.sqrt(functional.h) * np.r_[state[3][1:-1], state[4][1:-1]]
            for state in states
        ]
        factor_tensor = torch.as_tensor(np.stack(factors, axis=1)).requires_grad_()
        blocks.append(
            {
                "factors": factor_tensor,
                "occupations": torch.as_tensor([state[1] for state in states]),
                "degeneracy": torch.tensor(float(states[0][2])),
                "species_sign": torch.tensor(1.0 if species == 1 else -1.0),
                "kappa": torch.tensor(kappa),
                "principals": torch.as_tensor([state[0] for state in states]),
                "free_hamiltonian": torch.zeros((398, 398), dtype=torch.float64),
            }
        )

    densities = [nonlocal_density_from_block(block, functional.h) for block in blocks]
    result = functional.evaluate(blocks, densities)
    fock_rearrangement = result.fields["sigma_rearr_fock_explicit"]
    sigma_core, omega_core, rho_core, _ = core.ddrhf_extract_fields()
    couplings = functional.couplings(result.coupling_density)
    fields_on_core = dict(result.fields)
    fields_on_core.update(
        sigma=torch.as_tensor(sigma_core[:201]),
        omega=torch.as_tensor(omega_core[:201]),
        rho=torch.as_tensor(rho_core[:201]),
    )
    direct_rearrangement_on_core_fields = functional.direct.rearrangement(
        result.densities, fields_on_core, couplings
    )
    total_rearrangement = direct_rearrangement_on_core_fields + fock_rearrangement
    core_buffer = (ctypes.c_double * 2000).in_dll(
        core._lib, "__meanfield_MOD_sigr"
    )
    core_rearrangement = np.ctypeslib.as_array(core_buffer)[:201].copy()
    rearrangement_difference = (
        total_rearrangement.detach().numpy() - core_rearrangement
    )
    maximum_rearrangement_index = int(
        np.argmax(np.abs(rearrangement_difference[1:-1])) + 1
    )
    def scalar(value: torch.Tensor) -> float:
        return float(value.detach().cpu())

    report = {
        "model": args.model,
        "core_state": args.core_state,
        "nucleus": {"z": args.z, "n": args.n, "a": args.a},
        "python": {
            "E_direct": scalar(result.direct_terms["E_direct"]),
            "E_fock": scalar(result.exchange_terms["E_fock"]),
            **{
                name: scalar(value)
                for name, value in result.exchange_terms.items()
                if name != "E_fock"
            },
        },
        "core": {"E_direct": reference["e_dir"], "E_fock": reference["e_exc"]},
        "rearrangement": {
            "max_abs_difference_interior_fm_inv": float(
                np.max(np.abs(rearrangement_difference[1:-1]))
            ),
            "max_difference_index": maximum_rearrangement_index,
            "python_at_max_fm_inv": scalar(
                total_rearrangement[maximum_rearrangement_index]
            ),
            "core_at_max_fm_inv": float(
                core_rearrangement[maximum_rearrangement_index]
            ),
            "python_fock_min_max_fm_inv": [
                scalar(fock_rearrangement[1:-1].min()),
                scalar(fock_rearrangement[1:-1].max()),
            ],
            "components_at_max_fm_inv": {
                "direct": scalar(
                    direct_rearrangement_on_core_fields[maximum_rearrangement_index]
                ),
                "fock": scalar(fock_rearrangement[maximum_rearrangement_index]),
            },
        },
    }
    report["difference"] = {
        "E_direct": report["python"]["E_direct"] - report["core"]["E_direct"],
        "E_fock": report["python"]["E_fock"] - report["core"]["E_fock"],
    }
    print("RHF_SAME_DENSITY_AUDIT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
