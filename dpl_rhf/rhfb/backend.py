"""Validated RHFB component evaluation; no substitute PKA1 energy is supplied.

Normal/anomalous factors, currents, kinetic energy and D1S pairing can be
evaluated in a general m-scheme. A full nuclear energy additionally requires
the general PKA1 Hartree/Fock/rearrangement action and its acceptance gates.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .basis import AngularRadialBasis
from .currents import invariant_baryon_density, local_bilinears
from .operators import DensityFactors
from .pairing import pairing_gap_from_sampler


@dataclass
class RHFBComponentEvaluation:
    kinetic_energy: torch.Tensor
    pairing_energy: torch.Tensor
    pairing_gaps: dict[str, torch.Tensor]
    particle_numbers: dict[str, torch.Tensor]
    currents: dict[str, torch.Tensor]
    species_currents: dict[str, dict[str, torch.Tensor]]
    status: str = "components-only"


def evaluate_components(basis: AngularRadialBasis,
                        states: dict[str, DensityFactors], *, chunk_size: int = 32,
                        checkpoint_pairing: bool = True) -> RHFBComponentEvaluation:
    if set(states) != {"n", "p"}:
        raise ValueError("supply independent neutron and proton factorized states")
    energies, pair_energies, gaps, numbers = [], [], {}, {}
    all_currents = {}
    for species, state in states.items():
        if state.dimension != basis.dimension:
            raise ValueError("density factor does not match primitive basis")
        hq = basis.apply_kinetic(state.q, species=species)
        energies.append(torch.trace(state.rho_active @ (state.q.mH @ hq)).real)
        numbers[species] = state.particle_number()

        def sampler(start, stop, state=state):
            # Physical values and volume weights are distinct at both endpoints.
            spinors = basis._sample_range(state.q, start, stop)
            weights = torch.as_tensor(basis.weights[start:stop], device=state.q.device,
                                      dtype=torch.float64)
            positions = torch.as_tensor(basis.positions[start:stop], device=state.q.device,
                                        dtype=torch.float64)
            return spinors, weights, positions

        energy, gap = pairing_gap_from_sampler(
            state.chi_active, basis.point_count, sampler,
            species="neutron" if species == "n" else "proton",
            chunk_size=chunk_size, checkpoint_blocks=checkpoint_pairing)
        pair_energies.append(energy)
        gaps[species] = gap
        chunks = [local_bilinears(samples, state.rho_active)
                  for _, samples, _, _ in basis.iter_samples(state.q, chunk_size=chunk_size)]
        all_currents[species] = {key: torch.cat([chunk[key] for chunk in chunks], dim=0)
                                for key in chunks[0]}
    combined = {
        "rho_s": all_currents["n"]["scalar"] + all_currents["p"]["scalar"],
        "j_baryon": all_currents["n"]["baryon_current"] + all_currents["p"]["baryon_current"],
        "j_isovector_3": all_currents["n"]["baryon_current"] - all_currents["p"]["baryon_current"],
        "j_charge": all_currents["p"]["baryon_current"],
        "axial_3": all_currents["n"]["axial_current"] - all_currents["p"]["axial_current"],
        "tensor_3": all_currents["n"]["tensor"] - all_currents["p"]["tensor"],
    }
    combined["rho_b_invariant"] = invariant_baryon_density(combined["j_baryon"])
    return RHFBComponentEvaluation(sum(energies), sum(pair_energies), gaps, numbers,
                                   combined, all_currents)
