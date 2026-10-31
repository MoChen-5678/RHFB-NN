from __future__ import annotations

import torch


def bound_density_tail_parameters(
    epsilon_mev: torch.Tensor,
    mass_mev: float,
    hbar_c_mev_fm: float,
) -> dict[str, torch.Tensor]:
    """Derive bound-density tails from epsilon = E(total) - M.

    The positive-energy bound branch has ``-M < epsilon < 0``.  Returned
    ``decay_fm_inverse`` is the one-orbital decay constant; a local density
    decays with twice this value.
    """

    epsilon = torch.as_tensor(epsilon_mev)
    mass = epsilon.new_tensor(float(mass_mev))
    hbar_c = epsilon.new_tensor(float(hbar_c_mev_fm))
    if torch.any(epsilon >= 0.0) or torch.any(epsilon <= -mass):
        raise ValueError(
            "a no-sea bound nucleon requires -M < epsilon=E(total)-M < 0"
        )
    binding = -epsilon
    total_energy = mass + epsilon
    momentum = torch.sqrt(binding * (2.0 * mass - binding))
    decay = momentum / hbar_c
    beta = momentum / (2.0 * mass - binding)
    return {
        "epsilon_mev": epsilon,
        "total_energy_mev": total_energy,
        "binding_mev": binding,
        "decay_fm_inverse": decay,
        "lower_to_upper_ratio": -beta,
        "scalar_to_vector_ratio": total_energy / mass,
        "tensor_to_vector_ratio": -momentum / mass,
        "radial_nonlocal_robin_fm_inverse": decay,
        "radial_local_robin_fm_inverse": 2.0 * decay,
    }


def physical_local_density_robin_coefficient(
    radius_fm: torch.Tensor,
    decay_fm_inverse: torch.Tensor,
) -> torch.Tensor:
    """Coefficient in (d/dr + 2*lambda + 2/r) rho_X -> 0."""

    radius = torch.as_tensor(radius_fm)
    decay = torch.as_tensor(
        decay_fm_inverse, dtype=radius.dtype, device=radius.device
    )
    if torch.any(radius <= 0.0):
        raise ValueError("the asymptotic Robin coefficient requires r > 0")
    return 2.0 * decay + 2.0 / radius


def continue_bound_radial_density(
    upper_density_at_match: torch.Tensor,
    epsilon_mev: torch.Tensor,
    mass_mev: float,
    hbar_c_mev_fm: float,
    match_radius_fm: torch.Tensor,
    radius_fm: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Continue a bound radial density with the exact free Dirac tail.

    ``upper_density_at_match`` is ``G(match)^2`` for each occupied state.
    The continuation is imposed on density bilinears directly; no wavefunction
    equation is solved.  The returned exterior moments start at ``radius_fm``.
    """

    upper_match = torch.as_tensor(upper_density_at_match)
    epsilon = torch.as_tensor(
        epsilon_mev, dtype=upper_match.dtype, device=upper_match.device
    )
    tail = bound_density_tail_parameters(epsilon, mass_mev, hbar_c_mev_fm)
    match_radius = torch.as_tensor(
        match_radius_fm, dtype=upper_match.dtype, device=upper_match.device
    )
    radius = torch.as_tensor(
        radius_fm, dtype=upper_match.dtype, device=upper_match.device
    )
    if torch.any(radius < match_radius):
        raise ValueError("tail continuation radius must not precede its match point")
    decay = tail["decay_fm_inverse"]
    beta = -tail["lower_to_upper_ratio"]
    upper = upper_match * torch.exp(-2.0 * decay * (radius - match_radius))
    lower = beta.square() * upper
    mixed = -beta * upper
    rate = 2.0 * decay
    radial_vector = upper + lower
    exterior_number = radial_vector / rate
    exterior_r2_moment = radial_vector * (
        radius.square() / rate
        + 2.0 * radius / rate.square()
        + 2.0 / rate.pow(3)
    )
    return {
        **tail,
        "upper": upper,
        "lower": lower,
        "mixed": mixed,
        "scalar": upper - lower,
        "vector": radial_vector,
        "tensor": 2.0 * mixed,
        "exterior_number": exterior_number,
        "exterior_r2_moment": exterior_r2_moment,
    }


def occupied_block_exterior_moments(
    factors: torch.Tensor,
    occupations: torch.Tensor,
    degeneracy: float | torch.Tensor,
    epsilon_mev: torch.Tensor,
    mass_mev: float,
    hbar_c_mev_fm: float,
    h_fm: float,
    match_radius_fm: torch.Tensor,
    boundary_radius_fm: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Analytic ``R..infinity`` number and radius moments of one block."""

    split = factors.shape[0] // 2
    upper_match = factors[split - 1].square() * occupations / float(h_fm)
    tail = continue_bound_radial_density(
        upper_match,
        epsilon_mev,
        mass_mev,
        hbar_c_mev_fm,
        match_radius_fm,
        boundary_radius_fm,
    )
    factor = torch.as_tensor(
        degeneracy, dtype=factors.dtype, device=factors.device
    )
    return {
        "number": factor * torch.sum(tail["exterior_number"]),
        "r2_moment": factor * torch.sum(tail["exterior_r2_moment"]),
    }
