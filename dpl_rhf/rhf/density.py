from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NonlocalDensityBlock:
    """Four radial components of Eq. (3.94) in RHF-e.

    ``matrix`` has shape ``(2, 2, nr, nr)``.  Component 0 is G (+), component
    1 is F (-).  At fixed species and kappa, occupied principal states are
    summed first; the magnetic degeneracy is then included exactly once:

    ``rho_kappa = (2j+1) sum_n v_n^2 |n,kappa><n,kappa|``.
    """

    species_sign: float
    kappa: int
    degeneracy: float
    principals: tuple[int, ...]
    occupations: torch.Tensor
    radial_factors: torch.Tensor
    matrix: torch.Tensor
    projector: torch.Tensor
    tail_energies_mev: torch.Tensor | None = None

    @property
    def suffix(self) -> str:
        return "n" if self.species_sign > 0.0 else "p"

    def state_matrix(self, state_index: int) -> torch.Tensor:
        """Return the occupied ``(n, kappa)`` contribution to Eq. (3.94)."""

        radial = self.radial_factors[:, state_index]
        projector = self.occupations[state_index] * torch.outer(radial, radial)
        split = radial.shape[0] // 2
        return self.degeneracy * torch.stack(
            [
                torch.stack([projector[:split, :split], projector[:split, split:]]),
                torch.stack([projector[split:, :split], projector[split:, split:]]),
            ]
        )


def kappa_quantum_numbers(kappa: int) -> tuple[int, int, float]:
    """Return upper l, lower l and j for the nuclear Dirac convention."""

    if kappa == 0:
        raise ValueError("Dirac kappa cannot be zero")
    if kappa < 0:
        l_upper = -kappa - 1
        l_lower = -kappa
    else:
        l_upper = kappa
        l_lower = kappa - 1
    return l_upper, l_lower, abs(kappa) - 0.5


def nonlocal_density_from_block(
    block: dict[str, torch.Tensor], h: float
) -> NonlocalDensityBlock:
    factors = block["factors"]
    occupations = block["occupations"]
    split = factors.shape[0] // 2
    if factors.shape[0] != 2 * split:
        raise ValueError("RHF factors must contain equally sized G and F parts")
    projector = (factors * occupations[None, :]) @ factors.T
    radial_projector = projector / h
    principals = tuple(int(value) for value in block["principals"])
    if len(principals) != factors.shape[1]:
        raise ValueError("each RHF density factor must have one principal quantum number")
    matrix = torch.stack(
        [
            torch.stack([radial_projector[:split, :split], radial_projector[:split, split:]]),
            torch.stack([radial_projector[split:, :split], radial_projector[split:, split:]]),
        ]
    )
    degeneracy = float(block["degeneracy"])
    return NonlocalDensityBlock(
        species_sign=float(block["species_sign"]),
        kappa=int(block["kappa"]),
        degeneracy=degeneracy,
        principals=principals,
        occupations=occupations,
        radial_factors=factors / h**0.5,
        matrix=degeneracy * matrix,
        projector=projector,
        tail_energies_mev=block.get("tail_energies_mev"),
    )


def local_source_densities(
    nonlocal_blocks: list[NonlocalDensityBlock],
    r: torch.Tensor,
    *,
    include_tensor: bool = False,
    mass_n_mev: float | None = None,
    mass_p_mev: float | None = None,
    hbar_c_mev_fm: float | None = None,
) -> dict[str, torch.Tensor]:
    """Build RHF local sources as diagonal contractions of the one-body density."""

    names = ["rho_s_n", "rho_s_p", "rho_v_n", "rho_v_p"]
    if include_tensor:
        names.extend(["rho_t_n", "rho_t_p"])
    channels = {name: torch.zeros_like(r) for name in names}
    denominator = 4.0 * torch.pi * r.square().clamp_min(1.0e-12)
    from dpl_rhf.rhf.asymptotics import continue_bound_radial_density
    for block in nonlocal_blocks:
        gg = torch.diagonal(block.matrix[0, 0])
        ff = torch.diagonal(block.matrix[1, 1])
        gf = torch.diagonal(block.matrix[0, 1])
        fg = torch.diagonal(block.matrix[1, 0])
        suffix = block.suffix
        l_upper, l_lower, _ = kappa_quantum_numbers(block.kappa)
        upper_origin = (
            3.0 * gg[0] / denominator[1]
            - 3.0 * gg[1] / denominator[2]
            + gg[2] / denominator[3]
            if l_upper == 0 else torch.zeros_like(gg[0])
        )
        lower_origin = (
            3.0 * ff[0] / denominator[1]
            - 3.0 * ff[1] / denominator[2]
            + ff[2] / denominator[3]
            if l_lower == 0 else torch.zeros_like(ff[0])
        )
        channels[f"rho_s_{suffix}"][0] += upper_origin - lower_origin
        channels[f"rho_v_{suffix}"][0] += upper_origin + lower_origin
        values_by_channel = [
            (f"rho_s_{suffix}", gg - ff),
            (f"rho_v_{suffix}", gg + ff),
        ]
        if include_tensor:
            values_by_channel.append((f"rho_t_{suffix}", gf + fg))
        for name, values in values_by_channel:
            channel = channels[name]
            channel[1:-1] = channel[1:-1] + values / denominator[1:-1]
        mass = mass_n_mev if block.species_sign > 0.0 else mass_p_mev
        if (
            block.tail_energies_mev is not None
            and mass is not None
            and hbar_c_mev_fm is not None
        ):
            split = block.radial_factors.shape[0] // 2
            upper_match = block.radial_factors[split - 1].square()
            tail = continue_bound_radial_density(
                upper_match, block.tail_energies_mev, mass,
                hbar_c_mev_fm, r[-2], r[-1],
            )
            factor = block.degeneracy / denominator[-1]
            channels[f"rho_s_{suffix}"][-1] += factor * torch.sum(
                block.occupations * tail["scalar"]
            )
            channels[f"rho_v_{suffix}"][-1] += factor * torch.sum(
                block.occupations * tail["vector"]
            )
            if include_tensor:
                channels[f"rho_t_{suffix}"][-1] += factor * torch.sum(
                    block.occupations * tail["tensor"]
                )
    for name, values in channels.items():
        # Core extrapolates the origin after constructing the density, then
        # treats that stored mesh value as independent in Rearrange.  Letting
        # autograd propagate the extrapolation back into points 1..3 creates
        # an unphysical extra rearrangement contribution at the first point.
        if name.startswith("rho_t_"):
            values[0] = 0.0
        else:
            values[0] = values[0].detach()
    return channels


def local_densities(
    nonlocal_blocks: list[NonlocalDensityBlock],
    r: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Backward-compatible name for the four local PKO2 sources."""

    return local_source_densities(nonlocal_blocks, r)


# Compatibility alias for checkpoints and external scripts from the first RHF
# implementation. Production RHF code uses ``local_source_densities``.
hartree_source_densities = local_source_densities
