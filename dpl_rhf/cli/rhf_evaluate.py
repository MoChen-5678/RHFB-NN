from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend
from dpl_rhf.rhf.trainer import RHFTrainingConfig, build_network, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a neural PKO2 RHF checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default="outputs/rhf/evaluation")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    case = NucleusCase(**payload["case"])
    config = RHFTrainingConfig(**payload["config"])
    backend = DensityMatrixRHFBackend(case, device=device)
    network = build_network(backend, config)
    incompatible = network.load_state_dict(payload["model_state"], strict=False)
    missing = [
        key for key in incompatible.missing_keys
        if not key.endswith(".tail_energies_mev")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    result = backend.evaluate(network(), create_graph=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_result(out, backend, network, result, config, [])
    print(f"E/A(no CoM)={result.energy_per_a_no_com:.8f} MeV")


if __name__ == "__main__":
    main()
