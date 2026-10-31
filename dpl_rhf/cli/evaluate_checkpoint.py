from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.density_matrix_rmf import DensityMatrixRMFBackend
from dpl_rhf.io.outputs import ensure_out, write_json
from dpl_rhf.models.density_matrix_net import DensityMatrixNet


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.norm(actual - reference)
            / torch.linalg.norm(reference).clamp_min(1.0e-14)
        ).detach().cpu()
    )


def evaluate(checkpoint_path: Path, out_path: Path, device_name: str) -> dict:
    torch.set_default_dtype(torch.float64)
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    summary = checkpoint.get("summary")
    if not isinstance(summary, dict) or summary.get("variation_mode") != "density-matrix":
        raise ValueError("checkpoint is not a DPL-RMF density-matrix model")
    case_data = summary["case"]
    case = NucleusCase(
        str(case_data["model"]), int(case_data["z"]), int(case_data["n"]), int(case_data["a"])
    )
    backend = DensityMatrixRMFBackend(
        case,
        device=device,
        derivative_order=int(summary.get("derivative_order", 7)),
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r, dtype=torch.float64, device=device),
        backend.block_specs,
        hidden=int(summary["hidden"]),
        depth=int(summary["network_depth"]),
        radial_basis_size=int(summary["radial_basis_size"]),
        correction_scale=float(summary.get("density_correction_scale", 1.0e-2)),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError("checkpoint does not contain model_state_dict")
    incompatible = network.load_state_dict(state, strict=False)
    missing = [
        key for key in incompatible.missing_keys
        if not key.endswith(".tail_energies_mev")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    network.eval()

    with torch.no_grad():
        result = backend.evaluate(network.density_blocks())
        orbitals = backend.final_spectral_validation(result.hamiltonian)
        spectral_densities = backend.functional.densities_from_orbitals(orbitals)

    out = ensure_out(out_path)
    np.savez(
        out / "dpl_hamiltonian.npz",
        r=backend.r,
        channels=np.asarray(backend.functional_spec.channel_spec.names),
        stack=result.hamiltonian.detach().cpu().numpy(),
    )
    np.savez(
        out / "physical_components.npz",
        r=backend.r,
        components=np.asarray(("S", "V0", "V3", "VC")),
        stack=np.stack(
            [
                result.fields[name].detach().cpu().numpy()
                for name in ("component_S", "component_V0", "component_V3", "component_VC")
            ]
        ),
    )
    np.savez(
        out / "density_profiles.npz",
        r=backend.r,
        **{name: value.detach().cpu().numpy() for name, value in result.densities.items()},
    )
    orbital_data = {
        **{name: value.detach().cpu().numpy() for name, value in orbitals.items()},
        **backend.orbital_metadata,
    }
    np.savez(out / "final_wavefunctions.npz", r=backend.r, **orbital_data)

    density_errors = {
        name: _relative_l2(result.densities[name], spectral_densities[name])
        for name in result.densities
    }
    evaluation = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_format": checkpoint.get("checkpoint_format", "legacy-density-matrix"),
        "case": case_data,
        "device": str(device),
        "energy_per_a_no_com_mev": result.energy_per_a_no_com,
        "diagnostics": result.diagnostics,
        "spectral_density_relative_l2": density_errors,
        "max_spectral_density_relative_l2": max(density_errors.values()),
        "training_configuration": {
            key: summary.get(key)
            for key in (
                "activation", "hidden", "network_depth", "radial_basis_size",
                "density_correction_scale", "derivative_order", "seed",
            )
        },
    }
    write_json(out / "evaluation_summary.json", evaluation)
    return evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a DPL-RMF density-matrix checkpoint and recompute observables"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluated_checkpoint"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args.checkpoint, args.out, args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
