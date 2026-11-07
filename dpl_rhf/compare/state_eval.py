"""Evaluate an explicit set of occupied Dirac states with the RHF functional.

Both the neural state and a reference (Fortran) state are passed through the
same PKA1 functional so their Hartree mean fields, Fock matrices and energy
decompositions are defined identically.  Used only after training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class OccupiedStates:
    species: str            # "n" or "p"
    kappa: int
    upper: np.ndarray       # G on the full mesh, shape (npt, nstates)
    lower: np.ndarray       # F on the full mesh, shape (npt, nstates)
    occupations: np.ndarray # v^2 per state
    principals: list[int]
    energies: np.ndarray | None = None  # single-particle energies (MeV) for the exterior tail


def _dummy_column(npt_interior: int, dtype, device) -> torch.Tensor:
    column = torch.zeros((2 * npt_interior, 1), dtype=dtype, device=device)
    column[0, 0] = 1.0
    return column


def blocks_from_states(backend, states: list[OccupiedStates],
                       kappas: dict[str, list[int]]) -> list[dict[str, torch.Tensor]]:
    """Block dictionaries in the network format for every requested (q, kappa).

    Blocks without occupied states get one zero-occupation column so that
    their single-particle Hamiltonian (including Fock) is still generated.
    """

    device = backend.device
    dtype = torch.float64
    h = float(backend.functional.h)
    npt_interior = len(backend.r) - 2
    by_key = {(s.species, s.kappa): s for s in states}
    blocks = []
    for species in ("n", "p"):
        sign = 1.0 if species == "n" else -1.0
        for kappa in kappas[species]:
            state = by_key.get((species, kappa))
            if state is not None and state.upper.shape[1] > 0:
                factors = np.sqrt(h) * np.concatenate(
                    [state.upper[1:-1], state.lower[1:-1]], axis=0)
                factors = torch.as_tensor(factors, dtype=dtype, device=device)
                occupations = torch.as_tensor(state.occupations, dtype=dtype, device=device)
                principals = list(state.principals)
                energies = (np.full(len(principals), -10.0) if state.energies is None
                            else np.minimum(np.asarray(state.energies, dtype=float), -0.5))
            else:
                factors = _dummy_column(npt_interior, dtype, device)
                occupations = torch.zeros(1, dtype=dtype, device=device)
                principals = [0]
                energies = np.full(1, -10.0)
            factors = factors.clone().requires_grad_(True)
            blocks.append({
                "species_sign": torch.tensor(sign, dtype=dtype, device=device),
                "kappa": torch.tensor(kappa, dtype=torch.int64, device=device),
                "degeneracy": torch.tensor(2.0 * abs(kappa), dtype=dtype, device=device),
                "principals": torch.tensor(principals, dtype=torch.int64, device=device),
                "occupations": occupations,
                "coefficients": factors,
                "factors": factors,
                "free_hamiltonian": backend.dirac_kinetic.matrix_fm_inverse(sign, kappa),
                # Exterior free-Dirac tail beyond the box; affects only r = r_max.
                "tail_energies_mev": torch.as_tensor(energies, dtype=dtype, device=device),
            })
    return blocks


def mean_fields_mev(backend, result) -> dict[str, np.ndarray]:
    """Local direct + rearrangement mean fields (MeV) on the full mesh."""

    hbar_c = backend.params.hbar_c
    fields = {}
    for q in ("n", "p"):
        plus = result.fields[f"Sigma_plus_{q}_total_local"].detach().cpu().numpy() * hbar_c
        minus = result.fields[f"Sigma_minus_{q}_total_local"].detach().cpu().numpy() * hbar_c
        tensor = result.fields[f"Sigma_T_{q}_total_local"].detach().cpu().numpy() * hbar_c
        fields[f"sigma_plus_{q}"] = plus
        fields[f"sigma_minus_{q}"] = minus
        fields[f"sigma_tensor_{q}"] = tensor
        fields[f"scalar_{q}"] = 0.5 * (plus - minus)
        fields[f"vector_{q}"] = 0.5 * (plus + minus)
    return fields


def block_matrices_mev(result) -> dict[str, dict[str, np.ndarray]]:
    return {
        kind: {name: value.detach().cpu().numpy() for name, value in matrices.items()}
        for kind, matrices in (
            ("fock", result.fock_hamiltonians),
            ("direct", result.direct_hamiltonians),
            ("kinetic", result.kinetic_hamiltonians),
            ("total", result.hamiltonians),
        )
    }
