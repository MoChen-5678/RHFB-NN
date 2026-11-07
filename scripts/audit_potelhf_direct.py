#!/usr/bin/env python3
"""Post-training check of NN Hartree terms on saved S-RHFB-C densities.

This does not train a model or change the functional. Both the native NN
outer integration and the reference's Simpson outer integration are reported.
Run numerical evaluation through the host's RHFB resource launcher.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpl_rhf.rhf.direct import RHFDirectFunctional
from dpl_rhf.rhf.kernels import SphericalPropagators
from dpl_rhf.rhf.parameters import pka1_parameters
from dpl_rhf.rhf.quadrature import simpson_weights
from build_rhfb_observables import fortran_channels


CHANNELS = {
    "E_sigma": "Direct E-sigma (Scalar coupling)",
    "E_omega": "Direct E-omega (Vector coupling)",
    "E_rho": "Direct E-rho (Vector coupling)",
    "E_coul": "Direct Coulomb (Vector coupling)",
    "E_rho_tensor_direct": "Direct E-rho (Tensor coupling)",
    "E_rho_vt_direct": "Direct E-rho (Vector-Tensor coupling)",
}


def evaluate(run):
    parsed = json.loads((run / "parsed.json").read_text())
    nucleus = parsed["nucleus"]
    data = np.loadtxt(run / "DEN" / f"{nucleus}-den.PKA1", skiprows=1)
    r = torch.tensor(data[:, 0], dtype=torch.float64)
    density = {key: torch.tensor(data[:, col], dtype=torch.float64)
               for col, key in enumerate(("rho_s_n", "rho_s_p", "rho_v_n",
                                           "rho_v_p", "rho_t_n", "rho_t_p"), 1)}
    meta = parsed["input"]
    params = pka1_parameters()
    direct = RHFDirectFunctional(r, meta["Z"], meta["N"], params,
                                SphericalPropagators(r, params.hbar_c))
    couplings = direct.couplings(density["rho_v_n"] + density["rho_v_p"])
    fields = direct.fields(density, couplings)
    native = direct.energy_terms(density, fields, couplings)
    # A separate integration of the same fields isolates outer-quadrature error.
    reference_metric = RHFDirectFunctional(r, meta["Z"], meta["N"], params,
                                          direct.propagators)
    reference_metric.measure = 4 * torch.pi * r.square() * simpson_weights(
        len(r), direct.h, device=r.device, dtype=r.dtype)
    simpson = reference_metric.energy_terms(density, fields, couplings)
    table = fortran_channels(run, nucleus)
    channels = {}
    for key, label in CHANNELS.items():
        reference = table[label][2]
        a, b = float(native[key]), float(simpson[key])
        channels[key] = {"nn_native_mev": a, "nn_simpson_outer_mev": b,
                         "fortran_mev": reference,
                         "native_minus_fortran_mev": a - reference,
                         "simpson_minus_fortran_mev": b - reference}
    return {"nucleus": nucleus, "run": str(run.resolve()),
            "reference_converged": parsed["converged"], "channels": channels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-tolerance-mev", type=float, default=0.002)
    args = parser.parse_args()
    with torch.no_grad():
        rows = [evaluate(run) for run in args.runs]
    errors = [abs(row["channels"][key]["native_minus_fortran_mev"])
              for row in rows for key in ("E_rho_tensor_direct", "E_rho_vt_direct")]
    passed = all(row["reference_converged"] for row in rows) and all(
        np.isfinite(error) and error <= args.tensor_tolerance_mev for error in errors)
    report = {"passed": passed, "tensor_tolerance_mev": args.tensor_tolerance_mev,
              "max_tensor_native_abs_error_mev": max(errors), "results": rows,
              "scope": "Saved-density Hartree comparison; does not certify NN variational convergence."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
