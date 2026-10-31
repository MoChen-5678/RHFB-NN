"""Joint self-consistent RHFB stationarity loss with streamed fields.

This is an experimental variational-residual prototype.  It recomputes the
complete particle-hole functional on every call and keeps the D1S pairing
field as a full primitive-space action.  It does not select or certify a
no-sea branch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from .execution_limits import require_bounded_cgroup
from .factorized_commutator import FactorizedCommutator
from .gaussian_pairing import GaussianPairingAction
from .operators import DensityFactors
from .number_constraint import number_tangent_projection
from .streamed_normal_functional import StreamedPrimitiveNormalFunctional


@dataclass
class SelfConsistentStationarityEvaluation:
    """Differentiable values and fields from one coupled n/p evaluation."""

    loss: torch.Tensor
    total_energy: torch.Tensor
    particle_hole_energy: torch.Tensor
    pairing_energy: torch.Tensor
    normal_fields: dict[str, torch.Tensor]
    delta_actions: dict[str, Any]
    active_pairing_gaps: dict[str, torch.Tensor]
    species_diagnostics: dict[str, dict[str, torch.Tensor]]
    energy_channels: dict[str, torch.Tensor] | None = None
    physical_result_accepted: bool = False
    no_sea_branch_certified: bool = False
    no_sea_status: str = "unverified; physical branch selection has not been supplied"


def selfconsistent_stationarity_loss(
    basis,
    states: dict[str, DensityFactors],
    chemical_potentials: dict[str, torch.Tensor | float],
    number_targets: dict[str, int],
    *,
    normal_functional=None,
    pairing_action_factory: Callable[..., Any] = GaussianPairingAction,
    normal_tile_size: int = 8,
    normal_chunk_size: int = 64,
    residual_chunk_size: int = 8,
    pairing_chunk_size: int = 2,
    pairing_active_chunk_size: int = 1,
    lmax: int | None = None,
    number_weight: float = 1.0e4,
    chemical_potential_mode: str = "provided",
) -> SelfConsistentStationarityEvaluation:
    """Evaluate coupled PH+D1S fields and exact streamed full-space residual.

    ``states`` must contain neutron and proton ``DensityFactors`` from the
    same primitive basis and device.  The normal functional is evaluated
    jointly from both species on every call, retaining its custom HVP VJP.
    Each pairing action is rebuilt from the current Q and chi and acts on
    arbitrary primitive columns; Delta is never projected to Q.

    The full commutator Frobenius norm is streamed over primitive Nambu basis
    columns.  No dense Delta, D^4 interaction, P^2 chi, or large Nambu matrix
    is formed.  The returned loss is the sum over species of
    ``||[H_mu,R]||_F^2/(2D)^2`` plus the requested number penalties.

    This loss targets stationary equations, not an energy minimum. Excited
    stationary states can also attain zero loss at the correct particle number
    and parity. Ground-state selection and stability in the physically allowed
    no-sea tangent space require separate evidence.

    With ``chemical_potential_mode='number_tangent'`` the instantaneous
    number-gradient Gram equation supplies mu, retaining its field/density
    derivatives. This projects the canonical energy gradient off the number
    direction. It does not itself make Tr rho equal the target; the actual
    number error and its correction objective remain explicit. A restricted
    physical no-sea tangent would require its corresponding Gram equation.

    CUDA is used only when the supplied states already reside on CUDA.  This
    function never configures CUDA or falls back to another device.
    """
    # Enforce the actual process memory guard before constructing fields,
    # actions, or residual workspaces.
    require_bounded_cgroup()
    if set(states) != {"n", "p"}:
        raise ValueError("states must contain exactly neutron and proton")
    if set(chemical_potentials) != {"n", "p"} or set(number_targets) != {"n", "p"}:
        raise ValueError("chemical_potentials and number_targets must contain n and p")
    positive_integers = (normal_tile_size, normal_chunk_size,
                         residual_chunk_size, pairing_chunk_size,
                         pairing_active_chunk_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in positive_integers):
        raise ValueError("all tile and chunk sizes must be positive integers")
    if not math.isfinite(float(number_weight)) or number_weight < 0:
        raise ValueError("number_weight must be finite and nonnegative")
    if chemical_potential_mode not in {"provided", "number_tangent"}:
        raise ValueError("chemical_potential_mode must be provided or number_tangent")
    required_lmax = 2 * int(basis.kappa_max)
    effective_lmax = required_lmax if lmax is None else int(lmax)
    if effective_lmax < required_lmax:
        raise ValueError(f"lmax must include all allowed D1S multipoles through {required_lmax}")

    device = states["n"].q.device
    dimension = int(basis.dimension)
    for species in ("n", "p"):
        state = states[species]
        if not isinstance(state, DensityFactors):
            raise TypeError(f"{species} state must be DensityFactors")
        if state.dimension != dimension or state.q.device != device:
            raise ValueError("both species must match the basis dimension and share a device")
        if state.q.dtype != torch.complex128:
            raise TypeError("self-consistent stationarity requires complex128 factors")
        if isinstance(number_targets[species], bool) or not isinstance(number_targets[species], int):
            raise TypeError("particle-number targets must be integers")
        if not 0 <= number_targets[species] <= dimension:
            raise ValueError("particle-number targets must fit the primitive space")

    if normal_functional is None:
        normal_functional = StreamedPrimitiveNormalFunctional(
            basis, tile_size=normal_tile_size, chunk_size=normal_chunk_size,
            lmax=effective_lmax,
        )

    # This call rebuilds both species' full PH fields and returns a custom
    # autograd output whose backward recomputes the energy blocks and HVP.
    particle_hole_energy, normal_fields = normal_functional.differentiable_energy_and_fields(states)
    reported_terms = getattr(normal_functional, "last_energy_terms", None)
    energy_channels = (None if reported_terms is None else
                       {key: value.detach().clone() for key, value in reported_terms.items()})
    if set(normal_fields) != {"n", "p"}:
        raise ValueError("normal functional must return n and p primitive fields")

    pairing_energy = particle_hole_energy.new_zeros(())
    residual_loss = particle_hole_energy.new_zeros(())
    number_loss = particle_hole_energy.new_zeros(())
    actions: dict[str, Any] = {}
    active_gaps: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}

    for species in ("n", "p"):
        state = states[species]
        h = normal_fields[species]
        if h.shape != (dimension, dimension) or h.dtype != torch.complex128 or h.device != device:
            raise ValueError(f"{species} normal field must be a full complex128 primitive matrix")
        if not torch.isfinite(h).all() or not torch.allclose(h, h.mH, atol=1e-10, rtol=0):
            raise ValueError(f"{species} normal field must be finite and Hermitian")
        action = pairing_action_factory(
            basis, state.q, state.chi_active, species=species,
            chunk_size=pairing_chunk_size,
            active_chunk_size=pairing_active_chunk_size,
            lmax=effective_lmax, checkpoint_blocks=True,
        )
        actions[species] = action

        # Reuse this full-space Delta Q* in the residual's factorized terms.
        # Only D-by-M columns are returned; no primitive Delta matrix exists.
        delta_q_star = action.apply(state.q.conj())
        if delta_q_star.shape != state.q.shape or delta_q_star.dtype != torch.complex128:
            raise ValueError(f"{species} pairing action returned invalid Delta Q* columns")
        active_gap = state.q.mH @ delta_q_star
        active_gaps[species] = 0.5 * (active_gap - active_gap.T)
        species_pairing_energy = 0.5 * torch.sum(
            state.chi_active.conj() * active_gaps[species]
        ).real
        pairing_energy = pairing_energy + species_pairing_energy

        mu = torch.as_tensor(chemical_potentials[species], dtype=torch.float64, device=device)
        if mu.shape != () or not torch.isfinite(mu):
            raise ValueError(f"{species} chemical potential must be a finite scalar")
        hq = h @ state.q
        number_projection = number_tangent_projection(
            state, hq, delta_q_star, fallback_mu=mu)
        if chemical_potential_mode == "number_tangent":
            mu = number_projection.chemical_potential
        commutator = FactorizedCommutator(
            state, hq, action.apply,
            chemical_potential=mu, delta_q_star=delta_q_star,
        )
        terms = commutator.frobenius_squared(chunk_size=residual_chunk_size)
        scaled_residual = terms["full_squared_mev2"] / float((2 * dimension) ** 2)
        number_error = state.particle_number() - float(number_targets[species])
        number_penalty = float(number_weight) * number_error.square()
        residual_loss = residual_loss + scaled_residual
        number_loss = number_loss + number_penalty
        diagnostics[species] = {
            "full_commutator_frobenius_squared_mev2": terms["full_squared_mev2"],
            "normal_commutator_squared_mev2": terms["normal_squared_mev2"],
            "anomalous_commutator_squared_mev2": terms["anomalous_squared_mev2"],
            "scaled_residual_loss": scaled_residual,
            "particle_number": state.particle_number(),
            "number_error": number_error,
            "number_penalty": number_penalty,
            "pairing_energy": species_pairing_energy,
            "chemical_potential_mev": mu,
            "number_tangent_chemical_potential_mev": number_projection.chemical_potential,
            "number_tangent_identifiable": mu.new_tensor(float(number_projection.identifiable)),
            "number_variance": number_projection.number_variance,
            "number_variance_rho_identity_error": (
                number_projection.number_variance - number_projection.number_variance_from_rho).abs(),
            "number_constraint_gram": number_projection.constraint_gram,
            "number_tangent_inner_product": number_projection.tangent_inner_product_after_projection,
        }

    total_energy = particle_hole_energy + pairing_energy
    if energy_channels is not None:
        # D1S is evaluated separately from PH. Replace the PH zero placeholder
        # once, so the reported channels sum to E_PH + E_pair without duplicates.
        energy_channels["d1s_pairing"] = pairing_energy.detach().clone()
    loss = residual_loss + number_loss
    return SelfConsistentStationarityEvaluation(
        loss=loss, total_energy=total_energy,
        particle_hole_energy=particle_hole_energy,
        pairing_energy=pairing_energy,
        normal_fields=normal_fields,
        delta_actions=actions,
        active_pairing_gaps=active_gaps,
        species_diagnostics=diagnostics,
        energy_channels=energy_channels,
    )
