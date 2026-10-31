"""Correct mean particle number by canonical HFB variations, without spectra.

Robledo--Bertsch, arXiv:1104.5453, Eqs. (13), (20)--(22), separates
correction of an existing constraint error from projecting an energy gradient
onto its tangent. This module implements that missing correction action for
one species in a fixed particle space. It is not particle-number projection,
an energy minimizer, or a prescription for a field-dependent no-sea space.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .bogoliubov_blocking import BogoliubovState, state_from_frame
from .hfb_gradient import apply_thouless_update, n20_matrix


@dataclass(frozen=True)
class NumberRetraction:
    state: BogoliubovState
    converged: bool
    reason: str
    errors: tuple[float, ...]
    accepted_corrections: int


@torch.no_grad()
def correct_mean_particle_number(
    state: BogoliubovState,
    target: int,
    *,
    tolerance: float = 1.e-10,
    max_corrections: int = 64,
    max_backtracks: int = 24,
    max_generator_norm: float = 0.5,
) -> NumberRetraction:
    """Reduce Tr(rho)-target on the same connected canonical manifold.

    For this code's Z convention, dN=Re sum(conj(N20)*Z). Thus the
    linear correction is Z=-(N-target)*N20/||N20||². Finite canonical
    exponentiation and backtracking control its nonlinear error. Step-size
    bounds and iteration limits are numerical controls, not physical cuts.

    A vanishing N20 with a nonzero number error is reported as singular;
    pairing is never seeded to manufacture a number response. Returned
    states are detached solver actions: this is not a differentiable NN
    parameter layer and does not change any network's weights.
    """
    if not isinstance(state, BogoliubovState):
        raise TypeError("state must be a BogoliubovState")
    if isinstance(target, bool) or not isinstance(target, int):
        raise TypeError("target must be an integer particle number")
    if not 0 <= target <= state.u.shape[0]:
        raise ValueError("target must lie in the represented particle space")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(max_generator_norm) or max_generator_norm <= 0:
        raise ValueError("max_generator_norm must be finite and positive")
    if (isinstance(max_corrections, bool) or not isinstance(max_corrections, int)
            or max_corrections < 0 or isinstance(max_backtracks, bool)
            or not isinstance(max_backtracks, int) or max_backtracks < 1):
        raise ValueError("invalid iteration or backtracking limit")
    m = state.u.shape[0]
    column = None if state.blocked_index is None else m + state.blocked_index
    current = state_from_frame(
        state.frame.detach(), fermion_parity=state.density.fermion_parity,
        blocked_index=state.blocked_index, blocked_mode_column=column,
    )
    errors = [float(torch.trace(current.density.rho).real) - target]

    def finish(converged, reason):
        return NumberRetraction(current, converged, reason, tuple(errors), len(errors)-1)

    for _ in range(max_corrections):
        error = errors[-1]
        if abs(error) <= tolerance:
            return finish(True, "mean_number_converged")
        n20 = n20_matrix(current.u, current.v)
        gram = float(n20.abs().square().sum())
        # This is a floating-point response test, not a pairing cutoff.
        if gram <= 64 * torch.finfo(torch.float64).eps:
            return finish(False, "singular_number_response")
        direction = (-error / gram) * n20
        norm = float(torch.linalg.vector_norm(direction))
        if norm > max_generator_norm:
            direction = direction * (max_generator_norm / norm)
        accepted = False
        for backtrack in range(max_backtracks):
            trial = apply_thouless_update(
                current.frame, direction, step=0.5**backtrack,
                fermion_parity=current.density.fermion_parity,
                blocked_index=current.blocked_index,
            )
            trial_error = float(torch.trace(trial.density.rho).real) - target
            if abs(trial_error) < abs(error):
                current = trial
                errors.append(trial_error)
                accepted = True
                break
        if not accepted:
            return finish(False, "number_backtracking_exhausted")
    return finish(abs(errors[-1]) <= tolerance,
                  "mean_number_converged" if abs(errors[-1]) <= tolerance
                  else "number_iteration_limit")
