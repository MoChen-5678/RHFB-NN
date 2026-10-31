"""Tiny SiLU neural parameterization of projected RHFB fixed-point equations.

This module solves physical state equations by differentiating those equations
through a neural output map. The objective is the supplied stationarity,
field-self-consistency, and particle-number residual; it is not an energy
minimization or a nuclear-results acceptance path.

The problem object is intentionally supplied by ``projected_rhfb_fixedpoint``
so this file does not own or duplicate the RHFB physics. Its required interface
is ``state_from_coordinates(c, eta_raw, mu)`` and either ``residual_vector``
or ``residual_blocks``. All network parameters are passed through
``torch.func.functional_call`` during the solve.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from .stationary_solver import (
    StationaryOptions,
    StationaryResult,
    real_residual,
    solve_stationarity,
)


class FixedPointSiLUNet(nn.Module):
    """Small MLP whose output is the projected RHFB coordinate vector.

    The final bias starts at the requested chart seed. Hidden width four keeps
    the dense exact-Jacobian method suitable for the tiny moving-space
    benchmark. The hidden feature extractor is trainable by default; freezing
    it is an explicit reduced-chart mode used to reproduce the bounded run.
    """

    def __init__(
        self,
        initial_output: torch.Tensor,
        *,
        hidden: int = 4,
        input_features: torch.Tensor | None = None,
        seed: int = 20261006,
        freeze_hidden: bool = False,
    ) -> None:
        super().__init__()
        if initial_output.dtype != torch.float64 or initial_output.ndim != 1:
            raise TypeError("initial_output must be a float64 vector")
        if initial_output.numel() < 3:
            raise ValueError("output must contain chart coordinates, eta_raw, and mu")
        if hidden < 2:
            raise ValueError("hidden width must be at least two")
        if input_features is None:
            input_features = torch.tensor([1.0, 0.25, -0.5, 0.75], dtype=torch.float64)
        if input_features.dtype != torch.float64 or input_features.ndim != 1:
            raise TypeError("input_features must be a float64 vector")
        self.register_buffer("features", input_features.reshape(1, -1).clone())
        self.hidden = nn.Linear(input_features.numel(), hidden, dtype=torch.float64)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden, initial_output.numel(), dtype=torch.float64)
        self.freeze_hidden = bool(freeze_hidden)
        self.hidden.requires_grad_(not self.freeze_hidden)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.hidden.weight.normal_(mean=0.0, std=0.08, generator=generator)
            self.hidden.bias.normal_(mean=0.0, std=0.04, generator=generator)
            self.output.weight.normal_(mean=0.0, std=2.0e-4, generator=generator)
            self.output.bias.copy_(initial_output)

    def forward(self, features: torch.Tensor | None = None) -> torch.Tensor:
        if features is None:
            features = self.features
        return self.output(self.activation(self.hidden(features))).squeeze(0)


@dataclass
class NeuralFixedPointResult:
    network: FixedPointSiLUNet
    state: Any
    output_coordinates: torch.Tensor
    residual_blocks: dict[str, torch.Tensor]
    residual: torch.Tensor
    solver: StationaryResult
    initial_parameters: torch.Tensor
    final_parameters: torch.Tensor
    parameter_change_l2: float
    per_layer_change_l2: dict[str, float]
    eigensolver_calls_observed: int = 0
    explicit_refresh_calls_observed: int = 0


def _pack_problem_residual(problem: Any, state: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if hasattr(problem, "residual_blocks"):
        value = problem.residual_blocks(state)
        if isinstance(value, dict):
            raw = value
        elif is_dataclass(value):
            # dataclasses.asdict deep-copies tensors and breaks non-leaf AD
            # graph nodes, so retain the tensor objects directly.
            raw = {item.name: getattr(value, item.name) for item in fields(value)}
        else:
            names = ("stationarity", "field_self_consistency", "number")
            raw = {name: getattr(value, name) for name in names if hasattr(value, name)}
            if not raw:
                raise TypeError("residual_blocks must be a mapping or expose physical block attributes")
        ordered = [raw[key] for key in ("stationarity", "field_self_consistency", "number") if key in raw]
        if not ordered:
            raise ValueError("problem.residual_blocks returned no physical residuals")
        packed = real_residual(*ordered)
        return packed, raw
    if hasattr(problem, "residual_vector"):
        packed = problem.residual_vector(state)
        if packed.dtype != torch.float64 or packed.ndim != 1:
            raise TypeError("problem.residual_vector must return a float64 vector")
        return packed, {"all_physical_equations": packed}
    raise TypeError("problem must expose residual_blocks or residual_vector")


def _parameter_views(module: nn.Module, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    views: dict[str, torch.Tensor] = {}
    offset = 0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            views[name] = parameter
            continue
        count = parameter.numel()
        views[name] = flat[offset : offset + count].reshape(parameter.shape)
        offset += count
    if offset != flat.numel():
        raise ValueError("flat parameter vector has the wrong size")
    return views


def solve_neural_fixedpoint(
    problem: Any,
    initial_coordinates: torch.Tensor,
    initial_eta_raw: torch.Tensor | float,
    initial_mu: torch.Tensor | float,
    *,
    options: StationaryOptions | None = None,
    hidden: int = 4,
    input_features: torch.Tensor | None = None,
    seed: int = 20261006,
    freeze_hidden: bool = False,
) -> NeuralFixedPointResult:
    """Find a projected RHFB root through actual SiLU network parameters.

    ``initial_coordinates`` is the canonical chart seed, not a fixed external
    pairing field. D1S and the density-dependent normal field are generated by
    ``problem.state_from_coordinates`` from each current neural output.
    """
    if initial_coordinates.dtype != torch.float64 or initial_coordinates.ndim != 1:
        raise TypeError("initial_coordinates must be a float64 vector")
    if not initial_coordinates.numel():
        raise ValueError("initial_coordinates cannot be empty")
    eta0 = torch.as_tensor(initial_eta_raw, dtype=torch.float64, device=initial_coordinates.device).reshape(())
    mu0 = torch.as_tensor(initial_mu, dtype=torch.float64, device=initial_coordinates.device).reshape(())
    initial_output = torch.cat((initial_coordinates, eta0.reshape(1), mu0.reshape(1)))
    network = FixedPointSiLUNet(
        initial_output,
        hidden=hidden,
        input_features=input_features,
        seed=seed,
        freeze_hidden=freeze_hidden,
    ).to(device=initial_coordinates.device)

    named = tuple((name, parameter) for name, parameter in network.named_parameters()
                  if parameter.requires_grad)
    initial_parameters = torch.cat([parameter.detach().reshape(-1) for _, parameter in named])
    parameter_slices: dict[str, slice] = {}
    offset = 0
    for name, parameter in named:
        parameter_slices[name] = slice(offset, offset + parameter.numel())
        offset += parameter.numel()

    def output_from_parameters(flat: torch.Tensor) -> torch.Tensor:
        params = _parameter_views(network, flat)
        return functional_call(network, params, (network.features,))

    def physical_state(flat: torch.Tensor):
        output = output_from_parameters(flat)
        c = output[:-2]
        eta_raw, mu = output[-2], output[-1]
        return problem.state_from_coordinates(c, eta_raw, mu), output

    def residual_equations(flat: torch.Tensor) -> torch.Tensor:
        state, _ = physical_state(flat)
        residual, _ = _pack_problem_residual(problem, state)
        if residual.dtype != torch.float64 or residual.ndim != 1:
            raise TypeError("packed physical residual must be a float64 vector")
        return residual

    solve_options = options or StationaryOptions(
        max_iterations=40,
        residual_tolerance=1.0e-7,
        initial_damping=1.0e-2,
        maximum_trial_steps=10,
        maximum_variables=512,
        maximum_residuals=2048,
    )
    solved = solve_stationarity(residual_equations, initial_parameters, options=solve_options)
    with torch.enable_grad():
        final_flat = solved.parameters.detach().requires_grad_(True)
        state, output = physical_state(final_flat)
        residual, blocks = _pack_problem_residual(problem, state)

    per_layer_change = {
        name: float(torch.linalg.vector_norm(
            solved.parameters[parameter_slices[name]]
            - initial_parameters[parameter_slices[name]]
        ))
        for name, _ in named
    }
    final_views = _parameter_views(network, solved.parameters.detach())
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            parameter.copy_(final_views[name])
    return NeuralFixedPointResult(
        network=network,
        state=state,
        output_coordinates=output.detach(),
        residual_blocks=blocks,
        residual=residual.detach(),
        solver=solved,
        initial_parameters=initial_parameters.detach(),
        final_parameters=solved.parameters.detach(),
        parameter_change_l2=float(torch.linalg.vector_norm(
            solved.parameters.detach() - initial_parameters
        )),
        per_layer_change_l2=per_layer_change,
    )


def result_metadata(result: NeuralFixedPointResult) -> dict[str, object]:
    """Small JSON-safe report; network state belongs in a separate torch file."""
    return {
        "method": "damped Gauss-Newton on actual FixedPointSiLUNet parameters via torch.func.functional_call",
        "target": "projected physical stationarity + field self-consistency + particle number residuals; not energy minimization",
        "solver_converged": bool(result.solver.converged),
        "stop_reason": result.solver.stop_reason,
        "residual_max": float(result.residual.abs().max()),
        "residual_l2": float(torch.linalg.vector_norm(result.residual)),
        "parameter_change_l2": result.parameter_change_l2,
        "per_layer_change_l2": result.per_layer_change_l2,
        "initial_network_parameter_count": int(result.initial_parameters.numel()),
        "observed_eigensolver_calls": result.eigensolver_calls_observed,
        "observed_explicit_refresh_calls": result.explicit_refresh_calls_observed,
        "refresh_evidence": "this candidate module contains no refresh or eigensolver call; the supplied projected toy adapter uses differentiable rational-sign P plus QR",
        "history": result.solver.history,
    }


def run_bounded_toy(*, max_iterations: int = 12, hidden: int = 2,
                    seed: int = 20261006) -> tuple[NeuralFixedPointResult, dict[str, object]]:
    """Run the bounded, non-nuclear D1S moving-space toy under an eig guard."""
    from .projected_rhfb_fixedpoint import (
        candidate_spectral_solver_guard,
        initial_pair_coordinates,
        make_projected_fixedpoint_problem,
    )

    problem = make_projected_fixedpoint_problem()
    initial = initial_pair_coordinates(problem)
    options = StationaryOptions(
        max_iterations=max_iterations,
        residual_tolerance=2.0e-6,
        initial_damping=1.0e-2,
        maximum_trial_steps=8,
        maximum_variables=512,
        maximum_residuals=512,
    )
    with candidate_spectral_solver_guard() as calls:
        result = solve_neural_fixedpoint(
            problem, initial, torch.tensor(0.02, dtype=torch.float64),
            torch.tensor(0.45, dtype=torch.float64), options=options,
            hidden=hidden, seed=seed, freeze_hidden=True,
        )
    result.eigensolver_calls_observed = sum(calls.values())
    norms = {
        name: float(torch.linalg.vector_norm(value.detach()))
        for name, value in result.residual_blocks.items()
    }
    metadata = result_metadata(result)
    initial_views = _parameter_views(result.network, result.initial_parameters)
    initial_output = functional_call(
        result.network, initial_views, (result.network.features,)
    ).detach()
    metadata.update({
        "scope": "8-state moving-Q odd-HFB toy using the saved full finite-range D1S action; not a PKA1 nuclear calculation",
        "problem": {
            "primitive_dimension": problem.toy.d,
            "active_rank": problem.toy.rank,
            "target_number": problem.target_number,
            "number_parity": result.state.active.fermion_parity,
            "blocked_index_in_active_basis": problem.reference.blocked_index,
            "blocking_operation": problem.reference.blocking_operation,
            "pairing_force": "Gogny D1S strength 1; delta recomputed from current state chi each evaluation",
            "no_external_pairing_field": True,
            "chi_frobenius": float(torch.linalg.matrix_norm(result.state.chi.detach())),
            "delta_frobenius_mev": float(torch.linalg.matrix_norm(result.state.delta.detach())),
            "pairing_energy_mev": float((result.state.energy - problem.toy.normal(result.state.rho)[0]).detach()),
            "field_self_consistency_abs": abs(float(result.residual_blocks["field_self_consistency"].detach())),
            "number_abs": abs(float(result.residual_blocks["number"].detach())),
            "stationarity_max": float(result.residual_blocks["stationarity"].detach().abs().max()),
        },
        "residual_block_l2": norms,
        "observed_eigensolver_calls": result.eigensolver_calls_observed,
        "observed_torch_linalg_eigensolver_calls_by_operation": calls,
        "observed_explicit_refresh_calls": 0,
        "refresh_accounting": "candidate module and projected state path contain no refresh call; only explicit active chart and differentiable rational-sign projector/QR are used",
        "numpy_scipy_eigensolver_evidence": "the candidate path imports/calls no NumPy or SciPy spectral solver",
        "fortran_calls": 0,
        "network_output": result.output_coordinates.detach().cpu().tolist(),
        "neural_architecture": {
            "input_features": result.network.features.detach().cpu().reshape(-1).tolist(),
            "hidden_width": hidden,
            "hidden_activation": "SiLU",
            "output_size": int(result.output_coordinates.numel()),
            "trainable_layers": [name for name, p in result.network.named_parameters() if p.requires_grad],
            "frozen_layers": [name for name, p in result.network.named_parameters() if not p.requires_grad],
            "trainable_parameter_count": int(result.initial_parameters.numel()),
        },
        "initial_network_output": initial_output.cpu().tolist(),
        "coordinate_output_change_l2": float(torch.linalg.vector_norm(
            result.output_coordinates[:-2] - initial_output[:-2]
        )),
        "coordinate_output_change_max": float((
            result.output_coordinates[:-2] - initial_output[:-2]
        ).abs().max()),
        "initial_network_parameter_vector": result.initial_parameters.cpu().tolist(),
        "final_network_parameter_vector": result.final_parameters.cpu().tolist(),
    })
    payload = {
        "network_state_dict": {key: value.detach().cpu() for key, value in result.network.state_dict().items()},
        "initial_parameter_vector": result.initial_parameters.cpu(),
        "final_parameter_vector": result.final_parameters.cpu(),
        "optimizer_state": {
            "method": "damped Gauss-Newton on network parameters",
            "options": {
                "max_iterations": max_iterations,
                "residual_tolerance": options.residual_tolerance,
                "initial_damping": options.initial_damping,
                "maximum_trial_steps": options.maximum_trial_steps,
                "maximum_variables": options.maximum_variables,
                "maximum_residuals": options.maximum_residuals,
            },
            "history": result.solver.history,
            "stop_reason": result.solver.stop_reason,
            "converged": result.solver.converged,
        },
    }
    return result, {"metadata": metadata, "checkpoint_payload": payload}


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result, bundle = run_bounded_toy(
        max_iterations=args.max_iterations, hidden=args.hidden, seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "neural_fixedpoint_toy.json").write_text(
        json.dumps(bundle["metadata"], indent=2) + "\n", encoding="utf-8"
    )
    torch.save(bundle["checkpoint_payload"], args.output_dir / "neural_fixedpoint_toy.pt")
    print(args.output_dir)


if __name__ == "__main__":
    main()
