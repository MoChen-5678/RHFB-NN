#!/usr/bin/env python3
"""Audit kinetic, direct, and Fock neural gradients without SCF supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend
from dpl_rhf.rhf.trainer import RHFTrainingConfig, build_network, _load_network_state


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("PKO2", "PKA1"), default="PKA1")
    parser.add_argument("--z", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def flatten_gradient(
    value: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        value, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def main() -> None:
    args = arguments()
    if args.a != args.z + args.n:
        raise ValueError("A must equal Z+N")
    device = torch.device(args.device)
    backend = DensityMatrixRHFBackend(
        NucleusCase(args.model, args.z, args.n, args.a), device=device
    )
    config = RHFTrainingConfig()
    network = build_network(backend, config)
    checkpoint_epoch = None
    if args.checkpoint is not None:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        _load_network_state(network, saved["model_state"])
        checkpoint_epoch = saved.get("next_epoch")
    result = backend.evaluate(network(), create_graph=True)
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    energies = {
        "kinetic": result.action_terms["E_kinetic"] / args.a,
        "direct": result.action_terms["E_direct"] / args.a,
        "fock": result.action_terms["E_fock"] / args.a,
    }
    gradients = {
        name: flatten_gradient(value, parameters, retain_graph=True)
        for name, value in energies.items()
    }
    gradients["total"] = sum(gradients.values(), torch.zeros_like(gradients["kinetic"]))
    norms = {
        name: float(torch.linalg.vector_norm(gradient).detach().cpu())
        for name, gradient in gradients.items()
    }
    cosines: dict[str, float] = {}
    names = ("kinetic", "direct", "fock")
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            denominator = (
                torch.linalg.vector_norm(gradients[first])
                * torch.linalg.vector_norm(gradients[second])
            ).clamp_min(1.0e-30)
            cosines[f"{first}_vs_{second}"] = float(
                (torch.dot(gradients[first], gradients[second]) / denominator)
                .detach()
                .cpu()
            )
    report = {
        "case": {"model": args.model, "z": args.z, "n": args.n, "a": args.a},
        "checkpoint_epoch": checkpoint_epoch,
        "energy_per_a_mev": {
            name: float(value.detach().cpu()) for name, value in energies.items()
        },
        "network_gradient_norms": norms,
        "network_gradient_cosines": cosines,
        "projected_hamiltonian_residuals_mev": {
            "total": float(result.stationarity_residual_tensor.detach().cpu()),
            **{
                name: float(value.detach().cpu())
                for name, value in result.stationarity_components.items()
            },
        },
        "component_residual_sum_error_mev": result.diagnostics[
            "component_residual_sum_error_mev"
        ],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
