"""Small, auditable odd-HFB prototype; not a PKA1 production backend.

The prototype compares a differentiable moving sign-projector chart against a
conventional positive-space SCF oracle.  Dense diagonalization is confined to
the reference implementation and counted separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn

from dpl_rhf.rhfb.manifold import (
    GeneralizedDensity,
    canonical_generator,
    generalized_density_matrix,
    lift_active_density,
    orthonormalize_factors,
    slater_reference,
    transform_reference,
)
from dpl_rhf.rhfb.no_sea import RationalSignOperator


@dataclass(frozen=True)
class ToyDiracPairingModel:
    h0: torch.Tensor
    interaction_shape: torch.Tensor
    coupling: float
    pairing_strength: float
    pair_kernel_mixing: float
    shift: float
    target_number: float
    number_penalty: float
    chemical_potential: float
    physical_dimension: int

    @property
    def dimension(self) -> int:
        return int(self.h0.shape[0])

    @property
    def interaction_norm_bound(self) -> float:
        # Hermitian infinity norm bounds the spectral norm for this symmetric
        # toy interaction (maximum absolute row sum).
        return float(torch.sum(torch.abs(self.interaction_shape), dim=1).max())

    def hamiltonian(self, rho: torch.Tensor) -> torch.Tensor:
        scalar = torch.trace(self.interaction_shape @ rho).real
        return self.h0 + self.coupling * scalar * self.interaction_shape

    def pair_kernel(self) -> torch.Tensor:
        identity = torch.eye(self.dimension, dtype=torch.complex128, device=self.h0.device)
        return identity + self.pair_kernel_mixing * self.interaction_shape

    def pairing_field(self, chi: torch.Tensor) -> torch.Tensor:
        kernel = self.pair_kernel()
        return -self.pairing_strength * (kernel @ chi @ kernel.mT)

    def energy(
        self, rho: torch.Tensor, chi: torch.Tensor, *, number_multiplier: float = 0.0
    ) -> torch.Tensor:
        physical = self.physical_energy(rho, chi)
        number = torch.trace(rho).real
        number_delta = number - self.target_number
        number_constraint = (
            number_multiplier * number_delta
            + 0.5 * self.number_penalty * number_delta.square()
        )
        return physical + number_constraint

    def physical_energy(self, rho: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
        scalar = torch.trace(self.interaction_shape @ rho).real
        one_body = torch.trace(self.h0 @ rho).real
        mean_field = 0.5 * self.coupling * scalar.square()
        kernel = self.pair_kernel()
        paired_tensor = kernel @ chi @ kernel.mT
        pairing = -0.5 * self.pairing_strength * torch.sum(
            chi.conj() * paired_tensor
        ).real
        return one_body + mean_field + pairing


class _CanonicalCoordinates(nn.Module):
    def __init__(self, dimension: int, *, device: torch.device, seed_pairing: bool = True):
        super().__init__()
        self.raw_a = nn.Parameter(torch.zeros((dimension, dimension), dtype=torch.complex128, device=device))
        self.raw_b = nn.Parameter(torch.zeros((dimension, dimension), dtype=torch.complex128, device=device))
        with torch.no_grad():
            if seed_pairing and dimension >= 5:
                self.raw_b[0, 1] = 0.1 + 0.03j
                self.raw_b[3, 4] = -0.1 + 0.03j
            elif seed_pairing and dimension >= 3:
                self.raw_b[1, 2] = 0.025 + 0.01j

    def transform(self, reference: GeneralizedDensity) -> GeneralizedDensity:
        a = 0.5 * (self.raw_a - self.raw_a.mH)
        b = 0.5 * (self.raw_b - self.raw_b.mT)
        return transform_reference(reference, a, b)

    def gradient_norm(self) -> float:
        values = [parameter.grad for parameter in self.parameters() if parameter.grad is not None]
        if not values:
            return float("inf")
        return float(torch.sqrt(sum(torch.sum(torch.abs(value).square()) for value in values)).detach())


def _toy_model(
    *, coupling: float, device: torch.device, pair_kernel_mixing: float = 0.0
) -> ToyDiracPairingModel:
    # Energies are mass-subtracted MeV.  W=-4 MeV separates five particle
    # levels from five negative-energy levels in this deliberately tiny model.
    particle = [0.00, 0.22, 0.47, 0.69, 0.91]
    sea = [-8.00, -8.22, -8.47, -8.69, -8.91]
    h0 = torch.diag(torch.tensor(particle + sea, dtype=torch.float64, device=device)).to(torch.complex128)
    shape = torch.zeros((10, 10), dtype=torch.complex128, device=device)
    shape[0, 0] = 0.35
    shape[1, 1] = -0.15
    shape[0, 5] = 0.4
    shape[5, 0] = 0.4
    shape[1, 6] = 0.2j
    shape[6, 1] = -0.2j
    # Normalize once so the interaction norm is exactly covered by the
    # Gershgorin row-sum certificate used below.
    shape = shape / float(torch.sum(torch.abs(shape), dim=1).max())
    return ToyDiracPairingModel(
        h0=h0,
        interaction_shape=shape,
        coupling=coupling,
        pairing_strength=5.0,
        pair_kernel_mixing=pair_kernel_mixing,
        shift=-4.0,
        target_number=3.0,
        number_penalty=50.0,
        chemical_potential=0.0,
        physical_dimension=5,
    )


def _certified_bounds(model: ToyDiracPairingModel) -> tuple[float, float, str]:
    diagonal = torch.diagonal(model.h0).real
    distances = torch.abs(diagonal - model.shift)
    free_gap = float(distances.min())
    free_bound = float(distances.max())
    perturbation = abs(model.coupling) * model.dimension * model.interaction_norm_bound
    gap = free_gap - perturbation
    bound = free_bound + perturbation
    if gap <= 0.0:
        raise RuntimeError("analytic Weyl enclosure closes the no-sea gap")
    certificate = (
        "diagonal h0 has exact distance bounds; |Tr(S rho)|<=D for 0<=rho<=I; "
        "||S||_2<=max-row-sum(S); Weyl perturbation bound "
        f"|g|*D*||S||<={perturbation:.6g} MeV"
    )
    return gap, bound, certificate


def _operator(model: ToyDiracPairingModel, h: torch.Tensor, *, order: int = 121) -> RationalSignOperator:
    gap, bound, certificate = _certified_bounds(model)
    return RationalSignOperator(
        h,
        model.dimension,
        shift=model.shift,
        spectral_gap=gap,
        spectral_bound=bound,
        bound_certificate=certificate,
        quadrature_order=order,
        tolerance=2.0e-10,
        solve_tolerance=2.0e-10,
    )


def _seed_active_density(device: torch.device) -> tuple[GeneralizedDensity, _CanonicalCoordinates]:
    occupations = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    reference = slater_reference(occupations)
    return reference, _CanonicalCoordinates(5, device=device)


def _active_state(
    reference: GeneralizedDensity,
    coordinates: _CanonicalCoordinates,
    q: torch.Tensor,
) -> GeneralizedDensity:
    active = coordinates.transform(reference)
    return lift_active_density(
        q, active.rho, active.chi, fermion_parity=reference.fermion_parity
    )


def _fixed_point_state(
    model: ToyDiracPairingModel,
    reference: GeneralizedDensity,
    coordinates: _CanonicalCoordinates,
    seed: torch.Tensor,
    *,
    iterations: int = 5,
) -> tuple[GeneralizedDensity, torch.Tensor, float, RationalSignOperator]:
    q = seed
    state = _active_state(reference, coordinates, q)
    sign_operator = _operator(model, model.hamiltonian(state.rho))
    residual = float("inf")
    for _ in range(iterations):
        h = model.hamiltonian(state.rho)
        sign_operator = _operator(model, h)
        projected = sign_operator.project_factors(seed)
        q_new = orthonormalize_factors(projected)
        residual = float(torch.linalg.matrix_norm(q_new @ q_new.mH - q @ q.mH).detach())
        q = q_new
        state = _active_state(reference, coordinates, q)
    return state, q, residual, sign_operator


def _metrics(
    model: ToyDiracPairingModel,
    state: GeneralizedDensity,
    q: torch.Tensor,
    sign_operator: RationalSignOperator,
) -> dict[str, float]:
    rho, chi, r = state.rho, state.chi, state.matrix
    rho_projected, chi_projected = sign_operator.project_density(rho, chi)
    pair_field = model.pairing_field(chi)
    _zero_projected, pair_field_projected = sign_operator.project_density(
        torch.zeros_like(pair_field), pair_field
    )
    return {
        "hermiticity_residual": float(torch.linalg.matrix_norm(rho - rho.mH).detach()),
        "pair_antisymmetry_residual": float(torch.linalg.matrix_norm(chi + chi.mT).detach()),
        "generalized_idempotency_residual": float(torch.linalg.matrix_norm(r @ r - r).detach()),
        "particle_number_residual": float(torch.abs(torch.trace(rho).real - model.target_number).detach()),
        "sea_leakage_residual": float(torch.linalg.matrix_norm(rho - rho_projected).detach()),
        "pair_sea_leakage_residual": float(torch.linalg.matrix_norm(chi - chi_projected).detach()),
        "pair_field_cross_gap_norm": float(
            torch.linalg.matrix_norm(pair_field - pair_field_projected).detach()
        ),
        "pair_field_norm": float(torch.linalg.matrix_norm(pair_field).detach()),
        "pair_kernel_mixing": float(model.pair_kernel_mixing),
        "pairing_tensor_norm": float(torch.linalg.matrix_norm(chi).detach()),
        "energy": float(model.physical_energy(rho, chi).detach()),
        "constrained_objective_at_zero_multiplier": float(model.energy(rho, chi).detach()),
        "fermion_parity": float(state.fermion_parity),
    }


def _optimize_moving_projector(
    model: ToyDiracPairingModel,
    *,
    epochs: int,
    seed: int,
    fixed_h: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    device = model.h0.device
    torch.manual_seed(seed)
    reference, coordinates = _seed_active_density(device)
    seed_factors = torch.eye(model.dimension, dtype=torch.complex128, device=device)[:, : model.physical_dimension]
    optimizer = torch.optim.Adam(coordinates.parameters(), lr=0.025)
    stationarity = float("inf")
    q = seed_factors
    state = _active_state(reference, coordinates, q)
    fixed_h0 = model.hamiltonian(state.rho).detach() if fixed_h else None
    fp_residual = 0.0 if fixed_h else float("inf")
    sign_operator = _operator(model, model.hamiltonian(state.rho))
    number_multiplier = model.chemical_potential
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        if fixed_h:
            h = fixed_h0
            sign_operator = _operator(model, h)
            q = orthonormalize_factors(sign_operator.project_factors(seed_factors))
            state = _active_state(reference, coordinates, q)
        else:
            state, q, fp_residual, sign_operator = _fixed_point_state(
                model, reference, coordinates, seed_factors, iterations=5
            )
        loss = model.energy(state.rho, state.chi, number_multiplier=number_multiplier)
        loss.backward()
        stationarity = coordinates.gradient_norm()
        optimizer.step()

    # Constrained quasi-Newton polishing is reported separately from the
    # requested Adam epochs.  The outer multiplier updates target the actual
    # odd-state Tr(rho), rather than an even-core estimate.
    lbfgs_iterations = 0
    lbfgs_function_evaluations = 0
    number_delta = float("inf")
    stationarity = float("inf")
    max_outer_iterations = 12
    for outer_index in range(max_outer_iterations):
        # Curvature history belongs to one augmented-Lagrangian objective;
        # discard it after changing the number multiplier.
        polish = torch.optim.LBFGS(
            coordinates.parameters(),
            lr=0.8,
            max_iter=100,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            polish.zero_grad(set_to_none=True)
            if fixed_h:
                current_op = _operator(model, fixed_h0)
                current_q = orthonormalize_factors(current_op.project_factors(seed_factors))
                current_state = _active_state(reference, coordinates, current_q)
            else:
                current_state, current_q, _, _ = _fixed_point_state(
                    model, reference, coordinates, seed_factors, iterations=5
                )
            value = model.energy(
                current_state.rho,
                current_state.chi,
                number_multiplier=number_multiplier,
            )
            value.backward()
            return value

        polish.step(closure)
        optimizer_state = polish.state.get(next(iter(coordinates.parameters())), {})
        lbfgs_iterations += int(optimizer_state.get("n_iter", 0))
        lbfgs_function_evaluations += int(optimizer_state.get("func_evals", 0))
        if fixed_h:
            sign_operator = _operator(model, fixed_h0)
            q = orthonormalize_factors(sign_operator.project_factors(seed_factors))
            state = _active_state(reference, coordinates, q)
        else:
            state, q, fp_residual, sign_operator = _fixed_point_state(
                model, reference, coordinates, seed_factors, iterations=5
            )
        number_delta = float((torch.trace(state.rho).real - model.target_number).detach())
        loss = model.energy(state.rho, state.chi, number_multiplier=number_multiplier)
        loss.backward()
        stationarity = coordinates.gradient_norm()
        # Stop only when number and tangent stationarity hold at the same
        # multiplier.  Otherwise update the augmented-Lagrangian dual and
        # reoptimize before measuring the next residual.
        if abs(number_delta) <= 1.0e-8 and stationarity <= 1.0e-6:
            break
        if outer_index + 1 >= max_outer_iterations:
            break
        number_multiplier += model.number_penalty * number_delta
    optimizer.zero_grad(set_to_none=True)
    if fixed_h:
        sign_operator = _operator(model, fixed_h0)
        q = orthonormalize_factors(sign_operator.project_factors(seed_factors))
        state = _active_state(reference, coordinates, q)
    else:
        state, q, fp_residual, sign_operator = _fixed_point_state(
            model, reference, coordinates, seed_factors, iterations=5
        )
    final_loss = model.energy(state.rho, state.chi, number_multiplier=number_multiplier)
    final_loss.backward()
    stationarity = coordinates.gradient_norm()
    metrics = _metrics(model, state, q, sign_operator)
    metrics["canonical_parameter_stationarity"] = stationarity
    metrics["number_multiplier"] = number_multiplier
    metrics["adam_epochs"] = float(epochs)
    metrics["lbfgs_outer_iterations"] = float(outer_index + 1)
    metrics["lbfgs_max_iterations_per_outer"] = 100.0
    metrics["lbfgs_actual_iterations"] = float(lbfgs_iterations)
    metrics["lbfgs_actual_function_evaluations"] = float(lbfgs_function_evaluations)
    metrics["projector_fixed_point_residual"] = fp_residual
    metrics["sign_quadrature_bound"] = sign_operator.quadrature_error_bound
    metrics["sign_linear_solve_residual"] = float(sign_operator.last_max_residual.detach())
    return metrics, {
        "rho": state.rho.detach(),
        "chi": state.chi.detach(),
        "q": q.detach(),
        "energy": metrics["energy"],
        "coordinates": coordinates,
    }


def _positive_space_reference(
    model: ToyDiracPairingModel,
    *,
    epochs: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any], tuple[int, int]]:
    """Independent BdG/SCF reference for the invariant orbital-2 blocked branch.

    The toy normal field and seed pairing tensor leave active orbital 2 exactly
    decoupled.  We therefore solve the even N=2 BdG problem on [0,1,3,4] and
    occupy orbital 2 explicitly.  This avoids a quasiparticle branch switch at
    a zero-energy crossing while retaining a true self-consistent odd blocked
    reference for this deliberately restricted toy model.
    """

    del epochs, seed  # The reference converges by residual, not training epoch.
    device = model.h0.device
    d = model.dimension
    m = model.physical_dimension
    seed_factors = torch.eye(d, dtype=torch.complex128, device=device)[:, :m]
    rho = torch.zeros((d, d), dtype=torch.complex128, device=device)
    rho[0, 0] = 1.0
    rho[1, 1] = 1.0
    chi = torch.zeros_like(rho)
    chi[0, 1] = 0.02 + 0.01j
    chi[1, 0] = -chi[0, 1]
    chi[3, 4] = -0.02 + 0.01j
    chi[4, 3] = -chi[3, 4]
    blocked_index = 2
    unblocked = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=device)
    dirac_eigensolves = 0
    bogoliubov_eigensolves = 0
    density_residual = float("inf")
    projector_residual = float("inf")
    stationarity = float("inf")
    q = seed_factors
    reference, coordinates = _seed_active_density(device)
    state = _active_state(reference, coordinates, q)

    def diagonalize_bdg(h_active: torch.Tensor, delta_active: torch.Tensor, chemical: float):
        nonlocal bogoliubov_eigensolves
        hu = h_active.index_select(0, unblocked).index_select(1, unblocked)
        du = delta_active.index_select(0, unblocked).index_select(1, unblocked)
        nu = int(unblocked.numel())
        eye = torch.eye(nu, dtype=torch.complex128, device=device)
        top = torch.cat([hu - chemical * eye, du], dim=1)
        bottom = torch.cat([-du.conj(), -hu.conj() + chemical * eye], dim=1)
        bdg = torch.cat([top, bottom], dim=0)
        values, vectors = torch.linalg.eigh(bdg)
        bogoliubov_eigensolves += 1
        negative = torch.arange(values.numel(), device=device) < nu
        r_even = vectors[:, negative] @ vectors[:, negative].mH
        number_even = torch.trace(r_even[:nu, :nu]).real
        return r_even, number_even, values, bdg

    for _iteration in range(250):
        h = model.hamiltonian(rho)
        eval_h, evec_h = torch.linalg.eigh(h)
        dirac_eigensolves += 1
        positive = eval_h > model.shift
        if int(positive.sum()) != m:
            raise RuntimeError("reference normal h gap closed or positive-space rank changed")
        exact_projector = evec_h[:, positive] @ evec_h[:, positive].mH
        q_new = orthonormalize_factors(exact_projector @ seed_factors)
        projector_residual = float(torch.linalg.matrix_norm(q_new @ q_new.mH - q @ q.mH).detach())
        q = q_new
        h_active = q.mH @ h @ q
        chi_active = q.mH @ chi @ q.conj()
        kernel_active = q.mH @ model.pair_kernel() @ q
        delta_active = -model.pairing_strength * (
            kernel_active @ chi_active @ kernel_active.mT
        )
        delta_active = 0.5 * (delta_active - delta_active.mT)

        # Solve the unblocked even N=2 core; the invariant blocked particle
        # contributes exactly one to the total odd particle number.
        low, high = -5.0, 5.0
        low_state = diagonalize_bdg(h_active, delta_active, low)
        high_state = diagonalize_bdg(h_active, delta_active, high)
        n_low, n_high = float(low_state[1]), float(high_state[1])
        if n_low > n_high:
            low, high = high, low
            n_low, n_high = n_high, n_low
        even_target = model.target_number - 1.0
        if not (n_low <= even_target <= n_high):
            raise RuntimeError("reference chemical-potential bracket does not cover even-core target N")
        for _ in range(36):
            mid = 0.5 * (low + high)
            mid_state = diagonalize_bdg(h_active, delta_active, mid)
            if float(mid_state[1]) < even_target:
                low = mid
            else:
                high = mid
        chemical = 0.5 * (low + high)
        r_even, _n_even, _values, bdg = diagonalize_bdg(
            h_active, delta_active, chemical
        )
        nu = int(unblocked.numel())
        rho_even = r_even[:nu, :nu]
        chi_even = r_even[:nu, nu:]
        rho_active = torch.zeros((m, m), dtype=torch.complex128, device=device)
        chi_active_new = torch.zeros_like(rho_active)
        rho_active[unblocked[:, None], unblocked[None, :]] = rho_even
        chi_active_new[unblocked[:, None], unblocked[None, :]] = chi_even
        rho_active[blocked_index, blocked_index] = 1.0
        rho_new = q @ rho_active @ q.mH
        chi_new = q @ chi_active_new @ q.mT
        density_residual = float(
            torch.sqrt(
                torch.linalg.matrix_norm(rho_new - rho).square()
                + torch.linalg.matrix_norm(chi_new - chi).square()
            ).detach()
        )
        rho = 0.8 * rho_new + 0.2 * rho
        chi = 0.8 * chi_new + 0.2 * chi
        state = GeneralizedDensity(
            rho_new, chi_new, generalized_density_matrix(rho_new, chi_new), -1
        )
        # Full odd-state stationarity includes the decoupled occupied orbital.
        delta_full = -model.pairing_strength * (
            kernel_active @ chi_active_new @ kernel_active.mT
        )
        delta_full = 0.5 * (delta_full - delta_full.mT)
        full_h_active = q.mH @ h @ q
        eye_m = torch.eye(m, dtype=torch.complex128, device=device)
        full_hfb = torch.cat(
            [
                torch.cat([full_h_active - chemical * eye_m, delta_full], dim=1),
                torch.cat([-delta_full.conj(), -full_h_active.conj() + chemical * eye_m], dim=1),
            ],
            dim=0,
        )
        stationarity = float(
            torch.linalg.matrix_norm(
                full_hfb @ generalized_density_matrix(rho_active, chi_active_new)
                - generalized_density_matrix(rho_active, chi_active_new) @ full_hfb
            ).detach()
        )
        if density_residual < 1.0e-10 and projector_residual < 1.0e-10:
            break
    sign_operator = _operator(model, model.hamiltonian(state.rho))
    final = _metrics(model, state, q, sign_operator)
    final["canonical_parameter_stationarity"] = stationarity
    final["scf_density_residual"] = density_residual
    final["scf_projector_residual"] = projector_residual
    final["blocked_index"] = float(blocked_index)
    final["reference_chemical_potential"] = float(chemical)
    return final, {
        "rho": state.rho.detach(),
        "chi": state.chi.detach(),
        "q": q.detach(),
        "energy": final["energy"],
    }, (dirac_eigensolves, bogoliubov_eigensolves)


def _compare(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    chi_reference = reference["chi"]
    chi_candidate = candidate["chi"]
    overlap = torch.sum(chi_reference.conj() * chi_candidate)
    phase = overlap.conj() / torch.abs(overlap).clamp_min(1.0e-30)
    chi_global_phase = float(torch.linalg.matrix_norm(phase * chi_candidate - chi_reference))
    # The toy pair kernel connects 0/1 with sea partners 5/6; these form one
    # symmetry block, while 3/4 are a second disconnected block. Fix phase
    # across complete blocks and retain the global-only difference.
    chi_gauge_fixed = chi_candidate.clone()
    for block in ((0, 1, 5, 6), (3, 4)):
        rows = torch.tensor(block, dtype=torch.long, device=chi_candidate.device)
        ref_channel = chi_reference.index_select(0, rows).index_select(1, rows)
        cand_channel = chi_candidate.index_select(0, rows).index_select(1, rows)
        channel_overlap = torch.sum(ref_channel.conj() * cand_channel)
        channel_phase = channel_overlap.conj() / torch.abs(channel_overlap).clamp_min(1.0e-30)
        chi_gauge_fixed[rows[:, None], rows[None, :]] *= channel_phase
    chi_absolute = float(torch.linalg.matrix_norm(chi_gauge_fixed - chi_reference))
    chi_reference_norm = float(torch.linalg.matrix_norm(chi_reference))
    return {
        "energy_deviation": abs(candidate["energy"] - reference["energy"]),
        "rho_relative_frobenius": float(
            torch.linalg.matrix_norm(candidate["rho"] - reference["rho"])
            / torch.linalg.matrix_norm(reference["rho"]).clamp_min(1.0e-12)
        ),
        "chi_absolute_frobenius_phase_aligned": chi_absolute,
        "chi_absolute_frobenius_global_phase_only": chi_global_phase,
        "chi_relative_frobenius_phase_aligned": chi_absolute / max(chi_reference_norm, 1.0e-12),
        "candidate_pairing_tensor_norm": float(torch.linalg.matrix_norm(chi_candidate)),
        "reference_pairing_tensor_norm": chi_reference_norm,
        "positive_projector_deviation": float(
            torch.linalg.matrix_norm(
                candidate["q"] @ candidate["q"].mH
                - reference["q"] @ reference["q"].mH
            )
        ),
    }


def _field_dependent_candidate_gradient_check(
    model: ToyDiracPairingModel, *, seed: int
) -> dict[str, float]:
    """Check an actual candidate-energy derivative through the moving P+(h[rho])."""

    device = model.h0.device
    occupations = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    reference = slater_reference(occupations)
    seed_factors = torch.eye(model.dimension, dtype=torch.complex128, device=device)[:, : model.physical_dimension]
    theta = 0.0
    epsilon = 2.0e-5

    def evaluate(value: float, *, differentiable: bool) -> tuple[torch.Tensor, float | None]:
        coordinates = _CanonicalCoordinates(model.physical_dimension, device=device)
        with torch.no_grad():
            coordinates.raw_b[0, 1] = complex(0.1 + value, 0.03)
        state, _q, _residual, _op = _fixed_point_state(
            model, reference, coordinates, seed_factors, iterations=5
        )
        energy = model.energy(state.rho, state.chi)
        if not differentiable:
            return energy.detach(), None
        gradient = torch.autograd.grad(energy, coordinates.raw_b)[0]
        return energy.detach(), float(gradient[0, 1].real.detach())

    _base, automatic = evaluate(theta, differentiable=True)
    plus, _ = evaluate(theta + epsilon, differentiable=False)
    minus, _ = evaluate(theta - epsilon, differentiable=False)
    finite_difference = float(((plus - minus) / (2.0 * epsilon)).detach())
    absolute_error = abs(automatic - finite_difference)
    relative_error = absolute_error / max(1.0e-12, abs(automatic), abs(finite_difference))
    return {
        "automatic_gradient": automatic,
        "centered_finite_difference": finite_difference,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "epsilon": epsilon,
        "fixed_point_iterations": 5.0,
    }


def _reference_moving_projector_gradient(
    model: ToyDiracPairingModel,
    reference_state: dict[str, Any],
    chemical_potential: float,
) -> dict[str, float]:
    """Evaluate moving-projector canonical stationarity at the SCF reference."""

    q_reference = reference_state["q"]
    rho_active = q_reference.mH @ reference_state["rho"] @ q_reference
    chi_active = q_reference.mH @ reference_state["chi"] @ q_reference.conj()
    active_reference = GeneralizedDensity(
        rho_active,
        chi_active,
        generalized_density_matrix(rho_active, chi_active),
        -1,
    )
    seed_factors = torch.eye(
        model.dimension, dtype=torch.complex128, device=model.h0.device
    )[:, : model.physical_dimension]
    def derivatives(iterations: int):
        coordinates = _CanonicalCoordinates(
            model.physical_dimension, device=model.h0.device, seed_pairing=False
        )
        state, _q, _residual, _operator = _fixed_point_state(
            model, active_reference, coordinates, seed_factors, iterations=iterations
        )
        parameters = tuple(coordinates.parameters())
        energy_gradient = torch.autograd.grad(
            model.physical_energy(state.rho, state.chi),
            parameters,
            retain_graph=True,
        )
        number_gradient = torch.autograd.grad(torch.trace(state.rho).real, parameters)
        inner = sum(
            torch.sum(energy.conj() * number).real
            for energy, number in zip(energy_gradient, number_gradient, strict=True)
        )
        number_norm_sq = sum(torch.sum(torch.abs(number).square()) for number in number_gradient)
        multiplier = inner / number_norm_sq.clamp_min(1.0e-30)
        projected = tuple(
            energy - multiplier * number
            for energy, number in zip(energy_gradient, number_gradient, strict=True)
        )
        raw = torch.sqrt(sum(torch.sum(torch.abs(gradient).square()) for gradient in energy_gradient))
        constrained = torch.sqrt(sum(torch.sum(torch.abs(gradient).square()) for gradient in projected))
        return coordinates, energy_gradient, number_gradient, projected, raw, multiplier, constrained

    five = derivatives(5)
    eight = derivatives(8)
    coordinates, energy_gradient, number_gradient, projected, raw, multiplier, constrained = eight
    direction_norm = constrained.clamp_min(1.0e-30)
    direction = tuple(-gradient / direction_norm for gradient in projected)
    epsilon = 1.0e-4

    def evaluate_direction(sign: float) -> tuple[float, float]:
        trial = _CanonicalCoordinates(
            model.physical_dimension, device=model.h0.device, seed_pairing=False
        )
        with torch.no_grad():
            for parameter, tangent in zip(trial.parameters(), direction, strict=True):
                parameter.copy_(sign * epsilon * tangent)
        trial_state, _qt, _rt, _ot = _fixed_point_state(
            model, active_reference, trial, seed_factors, iterations=8
        )
        return (
            float(model.physical_energy(trial_state.rho, trial_state.chi).detach()),
            float(torch.trace(trial_state.rho).real.detach()),
        )

    energy_plus, number_plus = evaluate_direction(1.0)
    energy_minus, number_minus = evaluate_direction(-1.0)
    finite_energy_slope = (energy_plus - energy_minus) / (2.0 * epsilon)
    finite_number_slope = (number_plus - number_minus) / (2.0 * epsilon)
    predicted_energy_slope = sum(
        torch.sum(energy.conj() * tangent).real
        for energy, tangent in zip(energy_gradient, direction, strict=True)
    )
    predicted_number_slope = sum(
        torch.sum(number.conj() * tangent).real
        for number, tangent in zip(number_gradient, direction, strict=True)
    )
    directional_absolute_error = abs(finite_energy_slope - float(predicted_energy_slope.detach()))
    directional_relative_error = directional_absolute_error / max(
        1.0e-12, abs(finite_energy_slope), abs(float(predicted_energy_slope.detach()))
    )
    return {
        "unconstrained_energy_gradient_norm": float(raw.detach()),
        "best_fit_number_multiplier": float(multiplier.detach()),
        "number_tangent_projected_gradient_norm": float(constrained.detach()),
        "five_iteration_number_tangent_projected_gradient_norm": float(five[6].detach()),
        "eight_iteration_number_tangent_projected_gradient_norm": float(eight[6].detach()),
        "five_vs_eight_projected_gradient_difference": float(
            torch.abs(five[6] - eight[6]).detach()
        ),
        "number_orthogonal_directional_energy_slope_fd": finite_energy_slope,
        "number_orthogonal_directional_energy_slope_autograd": float(predicted_energy_slope.detach()),
        "number_orthogonal_directional_energy_slope_relative_error": directional_relative_error,
        "number_orthogonal_directional_energy_slope_absolute_error": directional_absolute_error,
        "number_orthogonal_directional_particle_slope_fd": finite_number_slope,
        "number_orthogonal_directional_particle_slope_autograd": float(predicted_number_slope.detach()),
        "directional_fd_step": epsilon,
    }


def _independent_moving_variational_oracle(
    model: ToyDiracPairingModel,
    initial_state: dict[str, Any],
    *,
    max_iterations: int = 80,
) -> dict[str, Any]:
    """Optimize the moving-projector functional without training autograd.

    This is a tiny reference-only oracle.  Every objective/constraint call
    obtains P+(h[rho]) from a fresh dense Hermitian eigensolve and converges
    the projector/density fixed point.  SciPy finite differences, rather
    than the neural/canonical optimizer's backpropagation through rational
    sign solves, provide its search directions.  Eigensolves here are
    explicitly reference-only; this oracle is not a production algorithm.

    The independent coordinates are the non-redundant odd Thouless chart:
    occupied-to-empty particle rotations, occupied-hole pair amplitudes,
    and empty-particle pair amplitudes.  This removes null symmetric raw-A/B
    coordinates from the oracle.
    """

    device = model.h0.device
    d, m = model.dimension, model.physical_dimension
    occupations = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    if occupations.numel() != m:
        raise ValueError("the independent toy oracle expects the five-mode odd chart")
    active_reference = slater_reference(occupations)
    seed = torch.eye(d, dtype=torch.complex128, device=device)[:, :m]
    occupied, empty = (0, 1, 2), (3, 4)
    a_pairs = tuple((i, j) for i in occupied for j in empty)
    b_pairs = ((0, 1), (0, 2), (1, 2), (3, 4))
    complex_slots = tuple(("a", pair) for pair in a_pairs) + tuple(
        ("b", pair) for pair in b_pairs
    )
    dirac_eigensolves = 0

    def unpack(vector: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.zeros((m, m), dtype=torch.complex128, device=device)
        b = torch.zeros_like(a)
        for k, (kind, (i, j)) in enumerate(complex_slots):
            value = complex(float(vector[2 * k]), float(vector[2 * k + 1]))
            if kind == "a":
                a[i, j] = value
                a[j, i] = -value.conjugate()
            else:
                b[i, j] = value
                b[j, i] = -value
        return a, b

    def pack_saved_coordinates() -> np.ndarray:
        coords = initial_state.get("coordinates")
        if coords is None:
            return np.zeros(2 * len(complex_slots), dtype=np.float64)
        values: list[float] = []
        with torch.no_grad():
            for kind, (i, j) in complex_slots:
                raw = coords.raw_a if kind == "a" else coords.raw_b
                # Match the candidate's projected anti-Hermitian/skew chart.
                z = 0.5 * (raw[i, j] - (raw[j, i].conj() if kind == "a" else raw[j, i]))
                values.extend((float(z.real), float(z.imag)))
        return np.asarray(values, dtype=np.float64)

    def build_state(vector: np.ndarray) -> tuple[GeneralizedDensity, torch.Tensor, float]:
        nonlocal dirac_eigensolves
        a, b = unpack(vector)
        active = transform_reference(active_reference, a, b)
        q = seed
        residual = float("inf")
        state = _active_state(active_reference, _CoordinatesAdapter(active), q)
        # Exact sign projector is reference-only.  QR of P seed fixes the
        # frame gauge without tracking or refreshing any training subspace.
        for _ in range(40):
            h = model.hamiltonian(state.rho)
            values, vectors = torch.linalg.eigh(h)
            dirac_eigensolves += 1
            positive = values > model.shift
            if int(positive.sum()) != m:
                raise RuntimeError("moving variational oracle lost the certified positive rank")
            projector = vectors[:, positive] @ vectors[:, positive].mH
            q_new = orthonormalize_factors(projector @ seed)
            residual = float(torch.linalg.matrix_norm(q_new @ q_new.mH - q @ q.mH))
            q = q_new
            state = lift_active_density(q, active.rho, active.chi, fermion_parity=-1)
            if residual <= 2.0e-13:
                break
        if residual > 1.0e-10:
            raise RuntimeError(f"moving variational oracle projector did not converge: {residual:.3e}")
        return state, q, residual

    # The small adapter only lets _active_state share the same lift path.
    class _CoordinatesAdapter:
        def __init__(self, state: GeneralizedDensity):
            self._state = state

        def transform(self, _reference: GeneralizedDensity) -> GeneralizedDensity:
            return self._state

    evaluations = 0

    def values(vector: np.ndarray) -> tuple[float, float, float]:
        nonlocal evaluations
        evaluations += 1
        with torch.no_grad():
            state, _q, fp = build_state(vector)
            energy = float(model.physical_energy(state.rho, state.chi))
            number = float(torch.trace(state.rho).real)
        return energy, number, fp

    def objective(vector: np.ndarray) -> float:
        return values(vector)[0]

    def number_constraint(vector: np.ndarray) -> float:
        return values(vector)[1] - model.target_number

    initial = pack_saved_coordinates()
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac="3-point",
        constraints=({"type": "eq", "fun": number_constraint},),
        options={"maxiter": int(max_iterations), "ftol": 2.0e-12, "disp": False},
    )
    final_energy, final_number, final_fp = values(np.asarray(result.x, dtype=np.float64))
    with torch.no_grad():
        final_state, final_q, _ = build_state(np.asarray(result.x, dtype=np.float64))
    # Independent centered finite differences certify number-constrained
    # tangent stationarity at the oracle solution.  The equality-constrained
    # gradient is projected orthogonal to grad(N) in this 20D chart.
    x = np.asarray(result.x, dtype=np.float64)
    step = 2.0e-5
    grad_e = np.empty_like(x)
    grad_n = np.empty_like(x)
    for index in range(x.size):
        direction = np.zeros_like(x)
        direction[index] = step
        ep, np_ = objective(x + direction), number_constraint(x + direction)
        em, nm = objective(x - direction), number_constraint(x - direction)
        grad_e[index] = (ep - em) / (2.0 * step)
        grad_n[index] = (np_ - nm) / (2.0 * step)
    multiplier = float(np.dot(grad_e, grad_n) / max(np.dot(grad_n, grad_n), 1.0e-30))
    projected_gradient = grad_e - multiplier * grad_n
    stationarity = float(np.linalg.norm(projected_gradient))
    report = {
        "optimizer_success": bool(result.success),
        "optimizer_status": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective_evaluations": evaluations,
        "reference_dirac_eigensolves": dirac_eigensolves,
        "energy": final_energy,
        "particle_number": final_number,
        "particle_number_residual": abs(final_number - model.target_number),
        "projector_fixed_point_residual": final_fp,
        "number_tangent_stationarity_fd": stationarity,
        "number_multiplier_fd": multiplier,
        "fermion_parity": float(final_state.fermion_parity),
        "pairing_tensor_norm": float(torch.linalg.matrix_norm(final_state.chi)),
        "sea_leakage_residual": float(
            torch.linalg.matrix_norm(
                final_state.rho
                - (final_q @ (final_q.mH @ final_state.rho @ final_q) @ final_q.mH)
            )
        ),
        "rho": final_state.rho.detach(),
        "chi": final_state.chi.detach(),
        "q": final_q.detach(),
    }
    return report


def reference_derivative_check_valid(result: dict[str, float]) -> bool:
    """Certify the reference tangent's derivative without a zero-slope ratio.

    A stationary control has no resolvable relative slope. Its absolute
    tolerance is 100 times smaller than the 1e-6 stationarity threshold.
    Nonstationary directions must instead meet the true relative FD tolerance.
    """
    near_stationary = result["number_tangent_projected_gradient_norm"] <= 1e-6
    finite_difference_valid = (
        result["number_orthogonal_directional_energy_slope_absolute_error"] <= 1e-8
        if near_stationary else
        result["number_orthogonal_directional_energy_slope_relative_error"] <= 1e-3
    )
    return bool(
        abs(result["five_vs_eight_projected_gradient_difference"]) <= 1e-7
        and finite_difference_valid
        and abs(result["number_orthogonal_directional_particle_slope_fd"]) <= 1e-6
    )


def run_prototype_validation(
    out: str | Path,
    *,
    epochs: int = 300,
    seed: int = 20260930,
) -> dict[str, Any]:
    """Run fixed-field and density-dependent no-sea gates and save a JSON report."""

    if epochs < 10:
        raise ValueError("epochs must be at least 10 for a meaningful stationarity check")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    # K=I+0.2S generates nonzero pairing matrix elements across the normal
    # positive/negative Dirac split through S's particle-sea couplings.
    model = _toy_model(
        coupling=0.0, device=torch.device("cpu"), pair_kernel_mixing=0.2
    )
    fixed_candidate, fixed_state = _optimize_moving_projector(
        model, epochs=epochs, seed=seed, fixed_h=True
    )
    fixed_reference, fixed_reference_state, fixed_reference_eigs = _positive_space_reference(
        model, epochs=epochs, seed=seed
    )
    fixed_deviation = _compare(fixed_state, fixed_reference_state)

    field_model = _toy_model(
        coupling=0.012, device=torch.device("cpu"), pair_kernel_mixing=0.2
    )
    field_candidate, field_state = _optimize_moving_projector(
        field_model, epochs=epochs, seed=seed + 1, fixed_h=False
    )
    field_reference, field_reference_state, field_reference_eigs = _positive_space_reference(
        field_model, epochs=epochs, seed=seed + 1
    )
    field_deviation = _compare(field_state, field_reference_state)
    fixed_reference_moving_gradient = _reference_moving_projector_gradient(
        model,
        fixed_reference_state,
        fixed_reference["reference_chemical_potential"],
    )
    field_reference_moving_gradient = _reference_moving_projector_gradient(
        field_model,
        field_reference_state,
        field_reference["reference_chemical_potential"],
    )
    field_gradient_check = _field_dependent_candidate_gradient_check(
        field_model, seed=seed + 17
    )
    field_variational_oracle = _independent_moving_variational_oracle(
        field_model, field_state, max_iterations=80
    )

    structural_keys = (
        "hermiticity_residual",
        "pair_antisymmetry_residual",
        "generalized_idempotency_residual",
        "sea_leakage_residual",
        "pair_sea_leakage_residual",
    )
    max_structural = max(
        fixed_candidate[key] for key in structural_keys
    )
    max_structural = max(max_structural, *(field_candidate[key] for key in structural_keys))
    max_particle = max(
        fixed_candidate["particle_number_residual"],
        field_candidate["particle_number_residual"],
    )
    max_sea = max(
        fixed_candidate["sea_leakage_residual"],
        fixed_candidate["pair_sea_leakage_residual"],
        field_candidate["sea_leakage_residual"],
        field_candidate["pair_sea_leakage_residual"],
    )
    # Explicit, conservative first-gate thresholds.  The reference deviations
    # are reported separately; field-dependent agreement is not presumed.
    fixed_reference_pass = (
        fixed_deviation["energy_deviation"] < 2.0e-3
        and fixed_deviation["rho_relative_frobenius"] < 1.0e-3
        and fixed_deviation["chi_absolute_frobenius_phase_aligned"] < 1.0e-3
        and fixed_deviation["positive_projector_deviation"] < 1.0e-8
    )
    structure_number_sea_valid = (
        max_structural < 1.0e-10
        and max_particle < 1.0e-6
        and max_sea < 1.0e-8
        and fixed_candidate["fermion_parity"] == -1.0
        and field_candidate["fermion_parity"] == -1.0
        and fixed_candidate["pairing_tensor_norm"] > 1.0e-3
        and field_candidate["pairing_tensor_norm"] > 1.0e-3
        and fixed_candidate["projector_fixed_point_residual"] < 1.0e-10
        and field_candidate["projector_fixed_point_residual"] < 1.0e-10
        and fixed_reference["pair_field_cross_gap_norm"] > 1.0e-6
        and field_reference["pair_field_cross_gap_norm"] > 1.0e-6
    )
    candidate_stationarity_valid = (
        fixed_candidate["canonical_parameter_stationarity"] <= 1.0e-6
        and field_candidate["canonical_parameter_stationarity"] <= 1.0e-6
    )
    field_gradient_valid = field_gradient_check["relative_error"] <= 1.0e-3
    fixed_hfb_converged = (
        fixed_reference["scf_density_residual"] < 1.0e-10
        and fixed_reference["scf_projector_residual"] < 1.0e-10
        and fixed_reference["canonical_parameter_stationarity"] < 1.0e-8
        and fixed_reference["particle_number_residual"] < 1.0e-6
    )
    field_hfb_converged = (
        field_reference["scf_density_residual"] < 1.0e-10
        and field_reference["scf_projector_residual"] < 1.0e-10
        and field_reference["canonical_parameter_stationarity"] < 1.0e-8
        and field_reference["particle_number_residual"] < 1.0e-6
    )
    reference_moving_stationary = (
        fixed_reference_moving_gradient["number_tangent_projected_gradient_norm"] <= 1.0e-6
        and field_reference_moving_gradient["number_tangent_projected_gradient_norm"] <= 1.0e-6
    )
    reference_gradient_checks_converged = all(map(
        reference_derivative_check_valid,
        (fixed_reference_moving_gradient, field_reference_moving_gradient),
    ))
    field_reference_pass = (
        field_deviation["energy_deviation"] < 2.0e-3
        and field_deviation["rho_relative_frobenius"] < 1.0e-3
        and field_deviation["chi_absolute_frobenius_phase_aligned"] < 1.0e-3
        and field_deviation["positive_projector_deviation"] < 1.0e-8
    )
    required_solver_converged = (
        fixed_hfb_converged
        and candidate_stationarity_valid
        and field_gradient_valid
        and field_variational_oracle["optimizer_success"]
        and field_variational_oracle["particle_number_residual"] <= 1.0e-6
        and field_variational_oracle["projector_fixed_point_residual"] <= 1.0e-10
        and field_variational_oracle["number_tangent_stationarity_fd"] <= 1.0e-6
    )
    if not required_solver_converged:
        status = "inconclusive"
    elif not (
        structure_number_sea_valid
        and fixed_reference_pass
        and abs(field_candidate["energy"] - field_variational_oracle["energy"]) < 2.0e-3
        and float(
            torch.linalg.matrix_norm(field_state["rho"] - field_variational_oracle["rho"])
            / torch.linalg.matrix_norm(field_variational_oracle["rho"]).clamp_min(1.0e-12)
        ) < 1.0e-3
        and field_variational_oracle["fermion_parity"] == -1.0
        and field_variational_oracle["pairing_tensor_norm"] > 1.0e-3
    ):
        status = "failed"
    else:
        status = "passed"
    report: dict[str, Any] = {
        "status": status,
        "production_readiness": "not_assessed_by_toy_prototype; standard no-sea RHFB remains unvalidated",
        "validation_tier": "cross_gap_pairing_decisive_research_gate",
        "status_scope": "finite_toy_variational_algorithm_gate_only",
        "baseline_scalar_pairing": "not_included_in_this_decisive_cross_gap_report",
        "scope": "finite-dimensional odd Dirac+separable-pairing no-sea prototype only; no PKA1 claim",
        "epochs": int(epochs),
        "seed": int(seed),
        "reference_derivative_acceptance": {
            "near_stationary_gradient_threshold": 1e-6,
            "near_stationary_absolute_slope_error": 1e-8,
            "nonstationary_relative_slope_error": 1e-3,
            "five_vs_eight_gradient_difference": 1e-7,
            "number_directional_slope": 1e-6,
        },
        "cases": {
            "fixed_h": {
                "candidate": fixed_candidate,
                "positive_space_reference": fixed_reference,
                "deviations": fixed_deviation,
                "pair_field_cross_gap_norm": fixed_reference["pair_field_cross_gap_norm"],
                "moving_projector_gradient_at_independent_reference": fixed_reference_moving_gradient,
                "comparison_passed": bool(fixed_reference_pass),
                "comparison_is_interpretable": bool(fixed_hfb_converged and candidate_stationarity_valid),
                "reference_dirac_eigensolves": fixed_reference_eigs[0],
                "reference_bogoliubov_eigensolves": fixed_reference_eigs[1],
            },
            "density_dependent_h": {
                "candidate": field_candidate,
                "positive_space_scf_reference": field_reference,
                "deviations": field_deviation,
                "moving_projector_full_variational_oracle": {
                    key: value for key, value in field_variational_oracle.items()
                    if key not in {"rho", "chi", "q"}
                },
                "moving_projector_variational_deviations": {
                    "physical_energy_absolute": abs(
                        field_candidate["energy"] - field_variational_oracle["energy"]
                    ),
                    "rho_relative_frobenius": float(
                        torch.linalg.matrix_norm(field_state["rho"] - field_variational_oracle["rho"])
                        / torch.linalg.matrix_norm(field_variational_oracle["rho"]).clamp_min(1.0e-12)
                    ),
                    "chi_gauge_invariant_norm_difference": abs(
                        field_candidate["pairing_tensor_norm"]
                        - field_variational_oracle["pairing_tensor_norm"]
                    ),
                },
                "pair_field_cross_gap_norm": field_reference["pair_field_cross_gap_norm"],
                "moving_projector_gradient_at_independent_reference": field_reference_moving_gradient,
                "comparison_is_interpretable": bool(field_hfb_converged and candidate_stationarity_valid),
                "field_dependent_candidate_gradient_check": field_gradient_check,
                "reference_dirac_eigensolves": field_reference_eigs[0],
                "reference_bogoliubov_eigensolves": field_reference_eigs[1],
                "comparison_note": (
                    "The positive-space SCF is retained as a distinct conventional restricted-HFB diagnostic. "
                    "The independent moving-projector oracle instead optimizes the same field-dependent "
                    "constraint as the candidate using reference-only exact projectors and finite differences."
                ),
            },
        },
        "maxima": {
            "structural_residual": max_structural,
            "particle_number_residual": max_particle,
            "sea_leakage_residual": max_sea,
        },
        "solver_counts": {
            "training_dirac_eigensolves": 0,
            "training_bogoliubov_eigensolves": 0,
            "initialization_dirac_eigensolves": 0,
            "initialization_bogoliubov_eigensolves": 0,
            "reference_only_dirac_eigensolves": fixed_reference_eigs[0] + field_reference_eigs[0],
            "reference_only_bogoliubov_eigensolves": fixed_reference_eigs[1] + field_reference_eigs[1],
            "reference_only_moving_variational_oracle_dirac_eigensolves": field_variational_oracle[
                "reference_dirac_eigensolves"
            ],
            "reference_only_moving_variational_oracle_bogoliubov_eigensolves": 0,
            "subspace_refreshes": 0,
        },
        "assumptions": [
            "toy h0 is exactly diagonal and its shifted spectral bounds are known analytically",
            "the interaction norm is bounded by the Hermitian row-sum norm and 0<=rho<=I",
            "pair kernel K=I+0.2S intentionally couples particle/sea sectors; 0.2 is a toy stress parameter, not D1S",
            "D1S and PKA1 are not included",
            "the number constraint is an expectation-value penalty in this toy",
            "reference moving-projector stationarity is tested after projecting out the particle-number tangent, so chemical-potential convention cannot cause a false mismatch",
            "positive-space reference diagonalizations are test/reference operations only",
            "the moving-projector candidate imposes a constrained no-pair manifold; this is not asserted to be the standard RHB no-sea prescription",
            "the moving-projector variational oracle uses dense eigensolves and SciPy finite differences for validation only; this does not satisfy a zero-eigensolve production solver requirement",
            "the conventional positive-space SCF mismatch is retained as a diagnostic for the difference between frozen-SCF and fully varied moving-projector constraints",
        ],
        "gates": {
            "fixed_field_sign_projector_vs_reference": bool(
                fixed_reference_pass
                and fixed_reference_moving_gradient["number_tangent_projected_gradient_norm"] <= 1.0e-6
            ),
            "candidate_structure_number_and_sea_support": bool(structure_number_sea_valid),
            "candidate_canonical_stationarity_le_1e-6": bool(candidate_stationarity_valid),
            "independent_reference_moving_stationarity_le_1e-6": bool(reference_moving_stationary),
            "independent_reference_gradient_stable_fd_checked": bool(reference_gradient_checks_converged),
            "independent_full_variational_moving_projector_oracle": bool(
                field_variational_oracle["optimizer_success"]
                and field_variational_oracle["particle_number_residual"] <= 1.0e-6
                and field_variational_oracle["projector_fixed_point_residual"] <= 1.0e-10
                and field_variational_oracle["number_tangent_stationarity_fd"] <= 1.0e-6
            ),
            "candidate_matches_full_variational_moving_projector_oracle": bool(
                abs(field_candidate["energy"] - field_variational_oracle["energy"]) < 2.0e-3
                and float(
                    torch.linalg.matrix_norm(field_state["rho"] - field_variational_oracle["rho"])
                    / torch.linalg.matrix_norm(field_variational_oracle["rho"]).clamp_min(1.0e-12)
                ) < 1.0e-3
            ),
            "field_dependent_candidate_gradient_fd_le_1e-3": bool(field_gradient_valid),
            "density_dependent_self_consistent_equivalence": bool(
                field_hfb_converged
                and field_reference_pass
                and field_reference_moving_gradient["number_tangent_projected_gradient_norm"] <= 1.0e-6
            ),
        },
    }
    report["gates"]["density_dependent_deviation"] = field_deviation
    report["gates"]["fixed_field_reference_converged"] = bool(fixed_hfb_converged)
    report["gates"]["density_dependent_reference_converged"] = bool(field_hfb_converged)
    report["gates"]["conventional_scf_is_stationary_on_moving_constraint"] = bool(
        field_reference_moving_gradient["number_tangent_projected_gradient_norm"] <= 1.0e-6
    )
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
