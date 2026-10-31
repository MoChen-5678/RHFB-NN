from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class InverseHamiltonianMetrics:
    loss: torch.Tensor
    tangent_residual: torch.Tensor
    minimum_branch_score: torch.Tensor
    mean_branch_score: torch.Tensor
    linear_solve_residual: torch.Tensor
    solve_count: int


def inverse_hamiltonian_constraint(
    blocks: Sequence[dict[str, torch.Tensor]],
    hamiltonians: Mapping[str, torch.Tensor],
    shifts: Mapping[str, float],
    *,
    branch_floor: float = 0.0,
    alignment_weight: float = 1.0,
    branch_weight: float = 1.0,
    inverse_variation_weight: float = 1.0e-2,
    differentiate_hamiltonian: bool = False,
) -> InverseHamiltonianMetrics:
    """Differentiable no-sea constraint in the complete Dirac space.

    For W in the gap between the Fermi level and the Dirac sea, occupied
    positive-energy states have positive inverse-Hamiltonian Rayleigh values.
    The tangent residual vanishes exactly when the occupied projector is an
    invariant subspace of H (and therefore of (H-W)^-1).
    """

    tangent_numerator = None
    tangent_denominator = None
    solve_numerator = None
    solve_denominator = None
    scores: list[torch.Tensor] = []
    solve_count = 0
    for block in blocks:
        species = "n" if float(block["species_sign"]) > 0.0 else "p"
        name = f"{species}_kappa_{int(block['kappa']):+d}"
        hamiltonian = hamiltonians[name]
        # In inverse-Hamiltonian evolution H[rho_k] is reconstructed from the
        # current density and then held fixed while the occupied factors are
        # moved. Differentiating this auxiliary no-sea step through dH/drho
        # introduces an unrelated second-order self-consistency feedback. The
        # exact RHF energy term retains its complete density dependence.
        inverse_hamiltonian = (
            hamiltonian if differentiate_hamiltonian else hamiltonian.detach()
        )
        q = block["factors"]
        shift = inverse_hamiltonian.new_tensor(float(shifts[species]))
        identity = torch.eye(
            inverse_hamiltonian.shape[0],
            dtype=inverse_hamiltonian.dtype,
            device=inverse_hamiltonian.device,
        )
        operator = inverse_hamiltonian - shift * identity
        inverse_q = torch.linalg.solve(operator, q)
        solve_count += 1

        tangent = inverse_q - q @ (q.T @ inverse_q)
        residual = operator @ inverse_q - q
        tangent_value = torch.sum(tangent.square())
        inverse_value = torch.sum(inverse_q.square())
        solve_value = torch.sum(residual.square())
        q_value = torch.sum(q.square())
        tangent_numerator = tangent_value if tangent_numerator is None else tangent_numerator + tangent_value
        tangent_denominator = inverse_value if tangent_denominator is None else tangent_denominator + inverse_value
        solve_numerator = solve_value if solve_numerator is None else solve_numerator + solve_value
        solve_denominator = q_value if solve_denominator is None else solve_denominator + q_value

        occupied = block["occupations"] > 0.0
        q_occupied = q[:, occupied]
        inverse_q_occupied = inverse_q[:, occupied]
        projected_inverse = q_occupied.T @ inverse_q_occupied
        projected_inverse = 0.5 * (projected_inverse + projected_inverse.T)
        # This is a matrix inequality on the occupied subspace. Its
        # eigenvalues, unlike diagonal Rayleigh quotients, are invariant under
        # arbitrary occupied-orbital rotations Q -> Q U.
        scores.append((-shift) * torch.linalg.eigvalsh(projected_inverse))

    if not scores:
        raise ValueError("inverse-Hamiltonian constraint requires occupied blocks")
    all_scores = torch.cat(scores)
    tangent_residual = torch.sqrt(
        tangent_numerator / tangent_denominator.clamp_min(1.0e-30)
    )
    linear_residual = torch.sqrt(
        solve_numerator / solve_denominator.clamp_min(1.0e-30)
    )
    branch_barrier = torch.relu(
        all_scores.new_tensor(float(branch_floor)) - all_scores
    ).square().mean()
    # The small inverse-variation term follows the inverse-Hamiltonian
    # variational principle and selects the lowest states within the positive
    # branch. The exact RMF/RHF energy remains the primary objective.
    loss = (
        float(alignment_weight) * tangent_residual.square()
        + float(branch_weight) * branch_barrier
        - float(inverse_variation_weight) * all_scores.mean()
    )
    return InverseHamiltonianMetrics(
        loss=loss,
        tangent_residual=tangent_residual,
        minimum_branch_score=all_scores.min(),
        mean_branch_score=all_scores.mean(),
        linear_solve_residual=linear_residual,
        solve_count=solve_count,
    )


@dataclass(frozen=True)
class PositiveEnergyProjectors:
    projectors: dict[str, torch.Tensor]
    max_newton_iterations: int
    max_trace_error: float
    max_idempotency_error: float


def positive_energy_projector(
    hamiltonian: torch.Tensor,
    shift: float,
    *,
    tolerance: float = 1.0e-13,
    max_iterations: int = 100,
) -> tuple[torch.Tensor, int]:
    """No-sea projector Lambda_+ = [1 + sign(H - W)] / 2 without eigensolves.

    The matrix sign function is obtained by the determinant-scaled Newton
    iteration X <- (mu X + (mu X)^-1) / 2, which only needs dense inverses.
    It neither computes eigenvectors nor replaces the variational space; it
    only removes the Dirac-sea part (spectrum below W) of the occupied factors.
    """

    with torch.no_grad():
        operator = hamiltonian.detach()
        operator = 0.5 * (operator + operator.T)
        size = operator.shape[0]
        identity = torch.eye(size, dtype=operator.dtype, device=operator.device)
        x = operator - float(shift) * identity
        iterations = 0
        scaled = True
        while iterations < max_iterations:
            inverse = torch.linalg.inv(x)
            if scaled:
                _, log_abs_det = torch.linalg.slogdet(x)
                mu = torch.exp(-log_abs_det / size)
                updated = 0.5 * (mu * x + inverse / mu)
            else:
                updated = 0.5 * (x + inverse)
            updated = 0.5 * (updated + updated.T)
            change = float(
                (torch.linalg.norm(updated - x) / torch.linalg.norm(updated)).cpu()
            )
            x = updated
            iterations += 1
            if scaled and change < 1.0e-3:
                # Scaling only accelerates the global phase; the final
                # unscaled steps converge quadratically to sign(H-W).
                scaled = False
            elif not scaled and change < tolerance:
                break
        else:
            raise RuntimeError(
                "Newton sign iteration did not converge; W is not inside the "
                "Dirac gap of the current Hamiltonian"
            )
        return 0.5 * (identity + x), iterations


def positive_energy_projectors(
    hamiltonians: Mapping[str, torch.Tensor],
    shifts: Mapping[str, float],
) -> PositiveEnergyProjectors:
    projectors: dict[str, torch.Tensor] = {}
    max_iterations = 0
    max_trace_error = 0.0
    max_idempotency_error = 0.0
    for name, hamiltonian in hamiltonians.items():
        projector, iterations = positive_energy_projector(
            hamiltonian, float(shifts[name[0]])
        )
        projectors[name] = projector
        max_iterations = max(max_iterations, iterations)
        # The discrete radial Dirac operator has equal numbers of particle
        # and sea modes, so tr(Lambda_+)=N/2 checks that W lies in the gap.
        max_trace_error = max(
            max_trace_error,
            abs(float(torch.trace(projector).cpu()) - 0.5 * projector.shape[0]),
        )
        max_idempotency_error = max(
            max_idempotency_error,
            float(torch.max(torch.abs(projector @ projector - projector)).cpu()),
        )
    return PositiveEnergyProjectors(
        projectors=projectors,
        max_newton_iterations=max_iterations,
        max_trace_error=max_trace_error,
        max_idempotency_error=max_idempotency_error,
    )


def dirac_sea_weight(
    blocks: Sequence[dict[str, torch.Tensor]],
    projectors: Mapping[str, torch.Tensor],
) -> float:
    """Largest Dirac-sea weight of any occupied direction, rotation invariant."""

    weights = []
    with torch.no_grad():
        for block in blocks:
            species = "n" if float(block["species_sign"]) > 0.0 else "p"
            name = f"{species}_kappa_{int(block['kappa']):+d}"
            q = block["factors"].detach()
            q = q[:, block["occupations"] > 0.0]
            positive = q.T @ projectors[name] @ q
            positive = 0.5 * (positive + positive.T)
            weights.append(1.0 - torch.linalg.eigvalsh(positive).min())
    return float(torch.stack(weights).max().cpu())


def rmf_block_hamiltonians(backend, result) -> dict[str, torch.Tensor]:
    matrices: dict[str, torch.Tensor] = {}
    for block in result.blocks:
        species = "n" if float(block["species_sign"]) > 0.0 else "p"
        name = f"{species}_kappa_{int(block['kappa']):+d}"
        matrices[name] = backend.matrix_solver.matrix(
            result.hamiltonian,
            float(block["species_sign"]),
            int(block["kappa"]),
        )
    return matrices
