#!/usr/bin/env python3
"""Audit PKA1 constants and density-dependent vertices against Core-1204."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpl_rhf.legacy.dpl_rmf_core import RHFCore
from dpl_rhf.rhf.direct import RHFDirectFunctional
from dpl_rhf.rhf.kernels import SphericalPropagators
from dpl_rhf.rhf.parameters import pka1_parameters


CORE_PKA1 = {
    "rho_sat": 0.159996,
    "mass_n": 938.9,
    "mass_p": 938.9,
    "m_sigma": 488.227904,
    "m_omega": 783.0,
    "m_rho": 769.0,
    "m_pi": 138.0,
    "g_sigma": 8.372672,
    "g_omega": 11.270457,
    "g_rho": 3.649857,
    "f_pi": 1.030722,
    "a_sigma": 1.103589,
    "b_sigma": 16.490109,
    "c_sigma": 18.278714,
    "d_sigma": 0.135041,
    "a_omega": 1.126166,
    "b_omega": 0.108010,
    "c_omega": 0.141251,
    "d_omega": 1.536183,
    "a_rho": 0.544017,
    "a_pi": 1.2,
    "f_rho": 3.199491,
    "a_rho_tensor": 0.820583,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", default="../Core-1204")
    parser.add_argument("--scalar-tolerance", type=float, default=0.0)
    parser.add_argument("--tensor-conversion-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--coupling-grid-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    torch.set_default_dtype(torch.float64)
    params = pka1_parameters()
    scalar = {}
    for name, reference in CORE_PKA1.items():
        value = float(getattr(params, name))
        scalar[name] = {
            "python": value,
            "core_define": reference,
            "absolute_difference": value - reference,
        }

    core = RHFCore(args.core_dir)
    core.init("PKA1", 8, 8, 16)
    history = []
    for _ in range(500):
        history.append(core.step())
        if len(history) >= 5 and abs(history[-1]) <= 1.0e-10:
            break
    rho_b = torch.as_tensor(core.densities()["rho_b"])
    radius = torch.as_tensor(core.r)
    direct = RHFDirectFunctional(
        radius,
        8,
        8,
        params,
        SphericalPropagators(radius, params.hbar_c),
    )
    python_couplings = direct.couplings(rho_b)

    # Define.f90 declares ten contiguous MSD arrays in this exact order.
    npt = int(core.npt)
    raw_type = ctypes.c_double * (10 * npt)
    raw = raw_type.in_dll(core.lib._lib, "__define_MOD_cct")
    core_couplings = np.ctypeslib.as_array(raw).reshape(10, npt)
    names = (
        "gsig", "gome", "grho", "fpio", "grtn",
        "dgsig", "dgome", "dgrho", "dfpio", "dgrtn",
    )
    grid = {}
    for index, name in enumerate(names):
        difference = python_couplings[name].detach().numpy() - core_couplings[index]
        grid[name] = {
            "max_abs_difference": float(np.max(np.abs(difference))),
            "rms_difference": float(np.sqrt(np.mean(difference**2))),
        }

    converted_tensor = params.g_rho * params.f_rho / (
        params.mass_n / params.hbar_c + params.mass_p / params.hbar_c
    )
    maximum_scalar_difference = max(
        abs(values["absolute_difference"]) for values in scalar.values()
    )
    maximum_coupling_difference = max(
        values["max_abs_difference"] for values in grid.values()
    )
    acceptance = {
        "scalar_parameters": {
            "value": maximum_scalar_difference,
            "tolerance": args.scalar_tolerance,
        },
        "rho_tensor_conversion": {
            "value": abs(params.rho_tensor_vertex - converted_tensor),
            "tolerance": args.tensor_conversion_tolerance,
        },
        "coupling_grid": {
            "value": maximum_coupling_difference,
            "tolerance": args.coupling_grid_tolerance,
        },
    }
    for check in acceptance.values():
        check["passed"] = bool(check["value"] <= check["tolerance"])
    report = {
        "reference": {
            "parameters": "Core-1204 Define.f90 lines 56-62",
            "density_dependence": "Core-1204 Density.f90 lines 118-147",
            "tensor_conversion": "Core-1204 DDRHF.f90 line 40",
        },
        "scalar_parameters": scalar,
        "rho_tensor_vertex_fm": {
            "python": params.rho_tensor_vertex,
            "core_formula": converted_tensor,
            "absolute_difference": params.rho_tensor_vertex - converted_tensor,
        },
        "coupling_grid_closure": grid,
        "core_iterations": len(history),
        "core_final_residual": history[-1],
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
        raise SystemExit(f"PKA1 parameter audit failed: {failed}")


if __name__ == "__main__":
    main()
