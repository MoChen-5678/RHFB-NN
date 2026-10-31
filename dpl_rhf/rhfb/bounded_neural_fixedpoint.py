"""Locally bounded SiLU-network solve for projected RHFB fixed-point equations.

This remains an experimental MovingToy path. The physical state is still
generated from canonical odd-HFB coordinates and the actual D1S functional;
only the neural chart output and each Gauss--Newton parameter step are bounded.
No eigensolver, refresh, external pairing field, or energy-KKT equation is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.func import functional_call

from .neural_rhfb_fixedpoint import _pack_problem_residual, _parameter_views
from .stationary_solver import StationaryOptions, StationaryResult


class BoundedFixedPointSiLUNet(nn.Module):
    """SiLU MLP with a fixed-center, componentwise tanh output chart."""

    def __init__(self, center: torch.Tensor, radii: torch.Tensor, *,
                 hidden: int = 4, input_features: torch.Tensor | None = None,
                 seed: int = 20261006) -> None:
        super().__init__()
        if center.dtype != torch.float64 or center.ndim != 1 or center.numel() < 3:
            raise TypeError("center must be a float64 vector containing c, eta_raw, mu")
        if radii.dtype != torch.float64 or radii.shape != center.shape:
            raise TypeError("radii must be a float64 vector matching center")
        if bool((radii <= 0).any()):
            raise ValueError("all output radii must be positive")
        if hidden < 2:
            raise ValueError("hidden width must be at least two")
        if input_features is None:
            input_features = torch.tensor([1.0, .25, -.5, .75], dtype=torch.float64)
        if input_features.dtype != torch.float64 or input_features.ndim != 1:
            raise TypeError("input_features must be a float64 vector")
        self.register_buffer("features", input_features.reshape(1, -1).clone())
        self.register_buffer("center", center.detach().clone())
        self.register_buffer("radii", radii.detach().clone())
        self.hidden = nn.Linear(input_features.numel(), hidden, dtype=torch.float64)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden, center.numel(), dtype=torch.float64)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.hidden.weight.normal_(mean=0.0, std=.08, generator=generator)
            self.hidden.bias.normal_(mean=0.0, std=.04, generator=generator)
            self.output.weight.normal_(mean=0.0, std=2e-2, generator=generator)
            # Center the raw head exactly while retaining hidden-layer
            # derivatives through the nonzero output weights.
            hidden = self.activation(self.hidden(self.features))
            self.output.bias.copy_(-(hidden @ self.output.weight.T).squeeze(0))

    def raw_output(self, features: torch.Tensor | None = None) -> torch.Tensor:
        if features is None:
            features = self.features
        return self.output(self.activation(self.hidden(features))).squeeze(0)

    def forward(self, features: torch.Tensor | None = None) -> torch.Tensor:
        return self.center + self.radii * torch.tanh(self.raw_output(features))


@dataclass
class BoundedNeuralFixedPointResult:
    network: BoundedFixedPointSiLUNet
    state: Any
    output_coordinates: torch.Tensor
    raw_output: torch.Tensor
    residual_blocks: dict[str, torch.Tensor]
    residual: torch.Tensor
    solver: StationaryResult
    initial_parameters: torch.Tensor
    final_parameters: torch.Tensor
    initial_output: torch.Tensor
    parameter_change_l2: float
    per_layer_change_l2: dict[str, float]
    optimizer_state: dict[str, Any]
    eigensolver_counts: dict[str, int]
    output_radii: torch.Tensor
    at_bound: torch.Tensor


def _trust_region_gauss_newton(
    residual: Callable[[torch.Tensor], torch.Tensor],
    initial_parameters: torch.Tensor,
    *, options: StationaryOptions,
    maximum_step_l2: float,
) -> tuple[StationaryResult, dict[str, Any]]:
    """Damped exact-AD GN with a hard bound on every accepted weight step."""
    if maximum_step_l2 <= 0:
        raise ValueError("maximum_step_l2 must be positive")
    x = initial_parameters.detach().clone()
    if x.dtype != torch.float64 or x.ndim != 1 or not bool(torch.isfinite(x).all()):
        raise TypeError("initial parameters must be a finite float64 vector")
    if x.numel() > options.maximum_variables:
        raise ValueError("dense stationary solve exceeds maximum_variables")

    def checked(value: torch.Tensor) -> torch.Tensor:
        r = residual(value)
        if r.dtype != torch.float64 or r.ndim != 1 or not r.numel():
            raise TypeError("residual must be a nonempty float64 vector")
        if r.numel() > options.maximum_residuals or not bool(torch.isfinite(r).all()):
            raise ValueError("residual exceeds configured size or is nonfinite")
        return r

    damping = float(options.initial_damping)
    history: list[dict[str, Any]] = []
    reason = "iteration_limit"
    for iteration in range(options.max_iterations + 1):
        x = x.detach().requires_grad_(True)
        r = checked(x)
        loss = .5 * torch.dot(r, r)
        row: dict[str, Any] = {
            "iteration": iteration,
            "residual_max": float(r.detach().abs().max()),
            "residual_l2": float(torch.linalg.vector_norm(r.detach())),
            "damping": damping,
            "step_accepted": False,
        }
        history.append(row)
        if row["residual_max"] <= options.residual_tolerance:
            reason = "residual_tolerance"
            break
        if iteration == options.max_iterations:
            break
        jac = torch.autograd.functional.jacobian(
            checked, x, vectorize=options.vectorize_jacobian,
        ).detach()
        gradient = jac.T @ r.detach()
        row["gradient_max"] = float(gradient.abs().max())
        if not bool(torch.isfinite(jac).all()):
            reason = "nonfinite_jacobian"
            break
        if row["gradient_max"] <= 1e-16:
            reason = "nonzero_residual_stagnation"
            break
        normal = jac.T @ jac
        diagonal = torch.diagonal(normal)
        scale = diagonal.clamp_min(max(1e-14, float(diagonal.max()) * 1e-12))
        original_loss = float(loss.detach())
        accepted = False
        for trial in range(options.maximum_trial_steps):
            matrix = normal + damping * torch.diag(scale)
            factor, info = torch.linalg.cholesky_ex(matrix)
            if int(info) != 0:
                damping *= 10
                continue
            step = torch.cholesky_solve(-gradient[:, None], factor)[:, 0]
            step_norm = float(torch.linalg.vector_norm(step))
            if step_norm > maximum_step_l2:
                step = step * (maximum_step_l2 / step_norm)
                step_norm = maximum_step_l2
            proposal = x.detach() + step
            proposed_r = checked(proposal).detach()
            proposed_loss = .5 * float(torch.dot(proposed_r, proposed_r))
            predicted_drop = float(-torch.dot(gradient, step)
                                   - .5 * torch.dot(step, normal @ step))
            if proposed_loss < original_loss and predicted_drop > 0:
                ratio = (original_loss - proposed_loss) / predicted_drop
                damping *= max(1/3, 1 - (2 * ratio - 1) ** 3)
                damping = max(1e-14, damping)
                x = proposal
                row.update({"step_accepted": True, "trial_steps": trial + 1,
                            "step_l2": step_norm,
                            "actual_to_predicted_drop": ratio})
                accepted = True
                break
            damping *= 10
        if not accepted:
            reason = "no_residual_decrease"
            break
    final = checked(x.detach()).detach()
    converged = float(final.abs().max()) <= options.residual_tolerance
    solved = StationaryResult(x.detach(), final, converged, reason, history)
    return solved, {
        "method": "damped exact-AD Gauss-Newton",
        "final_damping": damping,
        "maximum_step_l2": maximum_step_l2,
        "accepted_step_l2_max": max(
            (float(item["step_l2"]) for item in history if item.get("step_accepted")),
            default=0.0,
        ),
        "history": history,
    }


@contextmanager
def spectral_api_guard():
    """Observe and fail on all common linalg eigen/spectrum APIs on candidate path."""
    names = ("eig", "eigvals", "eigh", "eigvalsh")
    originals = {name: getattr(torch.linalg, name) for name in names}
    counts = {f"torch.linalg.{name}": 0 for name in names}

    def forbidden(name: str):
        def call(*_args, **_kwargs):
            counts[f"torch.linalg.{name}"] += 1
            raise AssertionError(f"candidate called torch.linalg.{name}")
        return call

    try:
        for name in names:
            setattr(torch.linalg, name, forbidden(name))
        yield counts
    finally:
        for name, function in originals.items():
            setattr(torch.linalg, name, function)


def solve_bounded_neural_fixedpoint(
    problem: Any,
    initial_coordinates: torch.Tensor,
    initial_eta_raw: torch.Tensor | float,
    initial_mu: torch.Tensor | float,
    *,
    options: StationaryOptions | None = None,
    hidden: int = 4,
    seed: int = 20261006,
    canonical_radius: float = 1.25,
    eta_raw_radius: float = .5,
    mu_radius: float = 4.0,
    maximum_step_l2: float = 2.0,
) -> BoundedNeuralFixedPointResult:
    """Run all-trainable SiLU NN against canonical odd-RHFB equations.

    Bounds are local numerical safeguards around the declared initial chart,
    not proof that a physical root lies inside them. The primitive orbitals and
    blocked reference stay fixed; only active canonical coordinates vary.
    """
    if initial_coordinates.dtype != torch.float64 or initial_coordinates.ndim != 1:
        raise TypeError("initial_coordinates must be a float64 vector")
    device = initial_coordinates.device
    eta = torch.as_tensor(initial_eta_raw, dtype=torch.float64, device=device).reshape(())
    mu = torch.as_tensor(initial_mu, dtype=torch.float64, device=device).reshape(())
    center = torch.cat((initial_coordinates, eta.reshape(1), mu.reshape(1)))
    radii = torch.cat((torch.full_like(initial_coordinates, canonical_radius),
                       torch.tensor([eta_raw_radius, mu_radius],
                                    dtype=torch.float64, device=device)))
    network = BoundedFixedPointSiLUNet(center, radii, hidden=hidden, seed=seed).to(device)
    named = tuple(network.named_parameters())
    initial_parameters = torch.cat([p.detach().reshape(-1) for _, p in named])
    initial_output = center.detach().clone()

    def output_from_parameters(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = _parameter_views(network, flat)
        # Keep one forward graph and separately expose the pre-tanh head.
        raw = functional_call(_RawOutputView(network), params, (network.features,))
        output = network.center + network.radii * torch.tanh(raw)
        return output, raw

    def physical_state(flat: torch.Tensor):
        output, raw = output_from_parameters(flat)
        state = problem.state_from_coordinates(output[:-2], output[-2], output[-1])
        return state, output, raw

    def residual_equations(flat: torch.Tensor) -> torch.Tensor:
        state, _, _ = physical_state(flat)
        packed, _ = _pack_problem_residual(problem, state)
        return packed

    solve_options = options or StationaryOptions(
        max_iterations=12, residual_tolerance=1e-7, initial_damping=1e-2,
        maximum_trial_steps=10, maximum_variables=512, maximum_residuals=2048,
    )
    with spectral_api_guard() as counts:
        solved, optimizer_state = _trust_region_gauss_newton(
            residual_equations, initial_parameters, options=solve_options,
            maximum_step_l2=maximum_step_l2,
        )
        with torch.enable_grad():
            final = solved.parameters.detach().requires_grad_(True)
            state, output, raw = physical_state(final)
            residual, blocks = _pack_problem_residual(problem, state)

    per_layer = {}
    offset = 0
    for name, parameter in named:
        size = parameter.numel()
        per_layer[name] = float(torch.linalg.vector_norm(
            solved.parameters[offset:offset + size] - initial_parameters[offset:offset + size]
        ))
        offset += size
    with torch.no_grad():
        final_views = _parameter_views(network, solved.parameters)
        for name, parameter in network.named_parameters():
            parameter.copy_(final_views[name])
    saturation = raw.detach().abs() >= 4.0
    return BoundedNeuralFixedPointResult(
        network=network, state=state, output_coordinates=output.detach(),
        raw_output=raw.detach(), residual_blocks=blocks, residual=residual.detach(),
        solver=solved, initial_parameters=initial_parameters.detach(),
        final_parameters=solved.parameters.detach(), initial_output=initial_output,
        parameter_change_l2=float(torch.linalg.vector_norm(
            solved.parameters.detach() - initial_parameters)),
        per_layer_change_l2=per_layer, optimizer_state=optimizer_state,
        eigensolver_counts=dict(counts), output_radii=radii.detach(), at_bound=saturation,
    )


class _RawOutputView(nn.Module):
    """Module view exposing the same MLP's raw head to functional_call."""

    def __init__(self, network: BoundedFixedPointSiLUNet) -> None:
        super().__init__()
        self.hidden = network.hidden
        self.activation = network.activation
        self.output = network.output

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(features))).squeeze(0)
