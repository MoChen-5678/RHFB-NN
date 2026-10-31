"""Damped Gauss--Newton for small neural stationary-equation benchmarks.

This solves a declared real residual, not an energy-minimum problem. It does
not select a physical no-sea/blocked branch: the caller must supply that model
and independently validate the resulting root. No Hamiltonian eigensystem or
subspace refresh is used. Dense Jacobians here are limited to small method
benchmarks; this is not a production large-space RHFB optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class StationaryOptions:
    max_iterations: int = 100
    residual_tolerance: float = 1e-9
    initial_damping: float = 1e-3
    maximum_trial_steps: int = 12
    maximum_variables: int = 512
    maximum_residuals: int = 2048
    vectorize_jacobian: bool = True


@dataclass
class StationaryResult:
    parameters: torch.Tensor
    residual: torch.Tensor
    converged: bool
    stop_reason: str
    history: list[dict]


def real_residual(*values: torch.Tensor) -> torch.Tensor:
    """Pack real and complex equations without discarding imaginary parts."""
    packed = []
    for value in values:
        if value.dtype == torch.complex128:
            packed.extend((value.real.reshape(-1), value.imag.reshape(-1)))
        elif value.dtype == torch.float64:
            packed.append(value.reshape(-1))
        else:
            raise TypeError("stationary residuals require float64 or complex128")
    if not packed:
        raise ValueError("provide at least one stationary equation")
    return torch.cat(packed)


def solve_stationarity(
    residual: Callable[[torch.Tensor], torch.Tensor],
    initial_parameters: torch.Tensor,
    *,
    options: StationaryOptions = StationaryOptions(),
    accepted_callback: Callable[[torch.Tensor, dict], None] | None = None,
) -> StationaryResult:
    """Solve r(theta)=0 using exact AD Jacobians and damped linear solves.

    ``theta`` may encode neural weights or a complete neural variational chart.
    Parameter updates are accepted only when the supplied residual norm falls.
    Positive diagonal damping regularizes gauge/redundant neural directions;
    it does not impose a physical branch, sea penalty, or energy prescription.
    A small gradient with a nonzero residual is reported as stagnation.
    """
    x = initial_parameters.detach().clone()
    if x.dtype != torch.float64 or x.ndim != 1 or x.numel() == 0:
        raise TypeError("initial_parameters must be a nonempty float64 vector")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("initial parameters must be finite")
    if x.numel() > options.maximum_variables:
        raise ValueError("dense stationary solver is restricted to small benchmarks")
    if (options.max_iterations < 1 or options.maximum_trial_steps < 1
            or options.residual_tolerance <= 0 or options.initial_damping <= 0):
        raise ValueError("iteration counts and tolerances must be positive")

    def checked(value):
        r = residual(value)
        if r.dtype != torch.float64 or r.ndim != 1 or r.numel() == 0:
            raise TypeError("residual callback must return a nonempty real float64 vector")
        if r.numel() > options.maximum_residuals:
            raise ValueError("dense residual dimension exceeds benchmark limit")
        if r.device != x.device or not bool(torch.isfinite(r).all()):
            raise ValueError("residual must be finite and share the parameter device")
        return r

    damping = float(options.initial_damping)
    history = []
    reason = "iteration_limit"
    for iteration in range(options.max_iterations + 1):
        x = x.detach().requires_grad_(True)
        r = checked(x)
        loss = .5 * torch.dot(r, r)
        maximum = float(r.detach().abs().max())
        row = {
            "iteration": iteration,
            "residual_max": maximum,
            "residual_l2": float(torch.linalg.vector_norm(r.detach())),
            "residual_square_loss": float(loss.detach()),
            "damping": damping,
            "step_accepted": False,
        }
        history.append(row)
        if maximum <= options.residual_tolerance:
            reason = "residual_tolerance"
            break
        if iteration == options.max_iterations:
            break
        jac = torch.autograd.functional.jacobian(
            checked, x, vectorize=options.vectorize_jacobian
        ).detach()
        r_det = r.detach()
        gradient = jac.T @ r_det
        row["gradient_max"] = float(gradient.abs().max())
        if not bool(torch.isfinite(jac).all()):
            reason = "nonfinite_jacobian"
            break
        if float(gradient.abs().max()) <= 1e-16:
            reason = "nonzero_residual_stagnation"
            break
        normal = jac.T @ jac
        # Column scaling makes damping insensitive to the units of coordinates
        # and keeps dormant/gauge coordinates regularized instead of singular.
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
                            "step_l2": float(torch.linalg.vector_norm(step)),
                            "actual_to_predicted_drop": ratio})
                if accepted_callback is not None:
                    accepted_callback(x.detach(), dict(row))
                accepted = True
                break
            damping *= 10
        if not accepted:
            reason = "no_residual_decrease"
            break
    final = checked(x.detach()).detach()
    converged = float(final.abs().max()) <= options.residual_tolerance
    return StationaryResult(x.detach(), final, converged, reason, history)
