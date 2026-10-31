from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.torch_rmf import TorchRMFBackend
from dpl_rhf.models.density_matrix_net import DensityBlockSpec
from dpl_rhf.rhf.asymptotics import (
    continue_bound_radial_density,
    occupied_block_exterior_moments,
)
from dpl_rhf.rhf.density import kappa_quantum_numbers


@dataclass
class DensityMatrixResult:
    energy_per_a_no_com: float
    energy_total_no_com: float
    energy_total_tensor: torch.Tensor
    stationarity_residual_tensor: torch.Tensor
    hamiltonian: torch.Tensor
    blocks: list[dict[str, torch.Tensor]]
    densities: dict[str, torch.Tensor]
    fields: dict[str, torch.Tensor]
    action_terms: dict[str, torch.Tensor]
    diagnostics: dict[str, float]
    functional_type: str = "rmf-density-matrix"


class DensityMatrixRMFBackend:
    """Orbital-free training graph using a no-sea one-body density projector."""

    def __init__(self, case: NucleusCase, *, device: torch.device, derivative_order: int = 7):
        self.base = TorchRMFBackend(case, device=device, derivative_order=derivative_order)
        self.case = case
        self.device = device
        self.r = self.base.r
        self.functional = self.base.functional
        self.functional_spec = self.base.functional_spec
        self.initial_hamiltonian_np = self.base.initial_hamiltonian_np
        self.matrix_solver = self.base.matrix_solver
        self.orbital_metadata = self.base.orbital_metadata
        self.initialization_block_diagonalizations = 0
        self.training_dirac_solves = 0
        self.refresh_phases = 0
        self.refresh_block_diagonalizations = 0
        self.final_validation_dirac_solves = 0
        self.block_specs = self._build_full_space_blocks()

    def _build_full_space_blocks(self) -> list[DensityBlockSpec]:
        """Build analytic occupied factors without a Dirac eigensolve."""
        data = self.matrix_solver.orbital_data
        groups: dict[tuple[float, int], list[int]] = {}
        for index, (species, kappa, occupation) in enumerate(
            zip(data["species_sign"], data["kappa"], data["occupancy"])
        ):
            if float(occupation) > 0.0:
                groups.setdefault((float(species), int(kappa)), []).append(index)

        stack = torch.as_tensor(
            self.initial_hamiltonian_np, dtype=torch.float64, device=self.device
        )
        zero_stack = torch.zeros_like(stack)
        zero_stack[1] = -2.0 * self.functional.params.mass_n / self.functional.params.hbar_c
        zero_stack[3] = -2.0 * self.functional.params.mass_p / self.functional.params.hbar_c
        metric_scale = float(np.sqrt(self.functional.h))
        epsilon_initial = -0.5 * 41.0 * self.case.a ** (-1.0 / 3.0)
        specs: list[DensityBlockSpec] = []
        with torch.no_grad():
            for (species, kappa), indices in sorted(groups.items()):
                ordered = sorted(indices, key=lambda i: int(data["principal"][i]))
                initial = torch.stack(
                    [
                        metric_scale
                        * torch.cat(
                            [data["G"][index, 1:-1], data["F"][index, 1:-1]]
                        )
                        for index in ordered
                    ],
                    dim=1,
                )
                initial, _ = torch.linalg.qr(initial, mode="reduced")
                degeneracies = {float(data["degeneracy"][index]) for index in ordered}
                if len(degeneracies) != 1:
                    raise ValueError(f"inconsistent degeneracy in species={species}, kappa={kappa}")
                specs.append(
                    DensityBlockSpec(
                        species_sign=species,
                        kappa=kappa,
                        degeneracy=degeneracies.pop(),
                        principals=tuple(int(data["principal"][index]) for index in ordered),
                        occupations=data["occupancy"][ordered].clone().to(self.device),
                        initial_factors=initial.to(self.device),
                        free_hamiltonian=self.matrix_solver.matrix(zero_stack, species, kappa),
                        tail_energies_mev=torch.full(
                            (len(ordered),), epsilon_initial,
                            dtype=torch.float64, device=self.device,
                        ),
                    )
                )
        return specs

    def refresh_positive_energy_spaces(
        self, hamiltonian: torch.Tensor
    ) -> list[DensityBlockSpec]:
        raise RuntimeError("subspace refresh is forbidden in the no-refresh solver")

    def densities_from_blocks(
        self, blocks: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        size = len(self.r)
        channels = {
            key: torch.zeros(size, dtype=torch.float64, device=self.device)
            for key in ("rho_s_n", "rho_s_p", "rho_v_n", "rho_v_p")
        }
        h = self.functional.h
        denominator = 4.0 * torch.pi * self.functional.r2.clamp_min(1.0e-12)
        for block in blocks:
            factors = block["factors"]
            occupations = block["occupations"]
            weighted = factors.square() * occupations[None, :]
            split = factors.shape[0] // 2
            upper_states = weighted[:split] / h
            lower_states = weighted[split:] / h
            upper = upper_states.sum(dim=1)
            lower = lower_states.sum(dim=1)
            scalar = torch.zeros(size, dtype=factors.dtype, device=factors.device)
            vector = torch.zeros_like(scalar)
            scalar[1:-1] = block["degeneracy"] * (upper - lower) / denominator[1:-1]
            vector[1:-1] = block["degeneracy"] * (upper + lower) / denominator[1:-1]
            l_upper, l_lower, _ = kappa_quantum_numbers(int(block["kappa"]))
            upper_origin = (
                3.0 * (upper_states[0] / denominator[1])
                - 3.0 * (upper_states[1] / denominator[2])
                + upper_states[2] / denominator[3]
                if l_upper == 0 else torch.zeros_like(upper[0])
            ).sum()
            lower_origin = (
                3.0 * (lower_states[0] / denominator[1])
                - 3.0 * (lower_states[1] / denominator[2])
                + lower_states[2] / denominator[3]
                if l_lower == 0 else torch.zeros_like(lower[0])
            ).sum()
            scalar[0] = block["degeneracy"] * (upper_origin - lower_origin)
            vector[0] = block["degeneracy"] * (upper_origin + lower_origin)
            mass = (
                self.functional.params.mass_n
                if float(block["species_sign"]) > 0.0
                else self.functional.params.mass_p
            )
            tail = continue_bound_radial_density(
                upper_states[-1], block["tail_energies_mev"], mass,
                self.functional.params.hbar_c, self.r[-2], self.r[-1],
            )
            scalar[-1] = block["degeneracy"] * tail["scalar"].sum() / denominator[-1]
            vector[-1] = block["degeneracy"] * tail["vector"].sum() / denominator[-1]
            suffix = "n" if float(block["species_sign"]) > 0.0 else "p"
            channels[f"rho_s_{suffix}"] = channels[f"rho_s_{suffix}"] + scalar
            channels[f"rho_v_{suffix}"] = channels[f"rho_v_{suffix}"] + vector
        return channels

    def kinetic_energy(self, blocks: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        for block in blocks:
            factors = block["factors"]
            expectations = torch.sum(factors * (block["free_hamiltonian"] @ factors), dim=0)
            total = total + (
                block["degeneracy"]
                * torch.sum(block["occupations"] * expectations)
                * self.functional.params.hbar_c
            )
        return total

    def evaluate(self, blocks: list[dict[str, torch.Tensor]]) -> DensityMatrixResult:
        densities = self.densities_from_blocks(blocks)
        rho_b = densities["rho_v_n"] + densities["rho_v_p"]
        couplings = self.functional.couplings(rho_b)
        rebuilt_fields = self.functional.reconstruct_fields(densities, couplings)
        state = {**densities, **rebuilt_fields}
        potentials = self.functional.potentials(**state)
        hamiltonian = torch.stack(
            [potentials[name] for name in ("vps_n", "vms_n", "vps_p", "vms_p")]
        )
        kinetic = self.kinetic_energy(blocks)
        direct = self.functional.direct_energy_terms(state)
        energy = kinetic + direct["E_direct"]
        observables = self.functional.observables(state)
        exterior = {
            "n_number": torch.zeros((), dtype=torch.float64, device=self.device),
            "n_r2": torch.zeros((), dtype=torch.float64, device=self.device),
            "p_number": torch.zeros((), dtype=torch.float64, device=self.device),
            "p_r2": torch.zeros((), dtype=torch.float64, device=self.device),
        }
        for block in blocks:
            suffix = "n" if float(block["species_sign"]) > 0.0 else "p"
            mass = (
                self.functional.params.mass_n if suffix == "n"
                else self.functional.params.mass_p
            )
            moments = occupied_block_exterior_moments(
                block["factors"], block["occupations"], block["degeneracy"],
                block["tail_energies_mev"], mass, self.functional.params.hbar_c,
                self.functional.h, self.r[-2], self.r[-1],
            )
            exterior[f"{suffix}_number"] += moments["number"]
            exterior[f"{suffix}_r2"] += moments["r2_moment"]
        field_residuals = self.functional.field_stationarity_residuals(state, couplings)

        orthogonality = []
        idempotency = []
        commutators = []
        stationarity_pieces = []
        for block in blocks:
            q = block["factors"]
            identity = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
            orthogonality_error = torch.max(torch.abs(q.T @ q - identity))
            orthogonality.append(orthogonality_error)
            occupations = block["occupations"]
            if bool(torch.all(occupations == 1.0)):
                # For P=QQ^T, idempotency follows directly from Q^TQ=I.
                idempotency.append(orthogonality_error)
            full_hamiltonian = self.matrix_solver.matrix(
                hamiltonian, float(block["species_sign"]), int(block["kappa"])
            )
            hq_weighted = (full_hamiltonian @ q) * occupations[None, :]
            tangent = q.T @ hq_weighted
            tangent_symmetric = 0.5 * (tangent + tangent.T)
            riemannian_gradient = hq_weighted - q @ tangent_symmetric
            stationarity_pieces.append(riemannian_gradient.reshape(-1))
            commutators.append(
                torch.sqrt(torch.mean(riemannian_gradient.square()))
                * self.functional.params.hbar_c
            )

        diagnostics = {
            "rms_n_no_com": float(observables["rms_n_no_com"].detach().cpu()),
            "rms_p_no_com": float(observables["rms_p_no_com"].detach().cpu()),
            "rms_matter_no_com": float(observables["rms_matter_no_com"].detach().cpu()),
            "charge_radius_no_com": float(observables["charge_radius_no_com"].detach().cpu()),
            "n_number_error": float(abs(sum(
                float(block["degeneracy"] * block["occupations"].sum())
                for block in blocks if float(block["species_sign"]) > 0.0
            ) - self.functional.n)),
            "z_number_error": float(abs(sum(
                float(block["degeneracy"] * block["occupations"].sum())
                for block in blocks if float(block["species_sign"]) < 0.0
            ) - self.functional.z)),
            "max_orthogonality_error": float(torch.stack(orthogonality).max().detach().cpu()),
            "max_idempotency_rms": float(torch.stack(idempotency).max().detach().cpu()) if idempotency else 0.0,
            "projected_commutator_rms_mev": float(torch.stack(commutators).max().detach().cpu()),
            "training_dirac_solves": self.training_dirac_solves,
            "initialization_block_diagonalizations": self.initialization_block_diagonalizations,
            "sparse_refresh_phases": self.refresh_phases,
            "refresh_block_diagonalizations": self.refresh_block_diagonalizations,
            "n_exterior_tail": float(exterior["n_number"].detach().cpu()),
            "z_exterior_tail": float(exterior["p_number"].detach().cpu()),
        }
        for name, residual in field_residuals.items():
            diagnostics[f"weak_el_{name}_rms"] = float(
                torch.sqrt(torch.mean(residual.square())).detach().cpu()
            )
        fields = {
            **rebuilt_fields,
            "sigma_rearr": potentials["sigma_rearr"],
            "component_S": potentials["scalar_self_energy"],
            "component_V0": potentials["vector_isoscalar"],
            "component_V3": potentials["vector_isovector"],
            "component_VC": potentials["vector_coulomb"],
        }
        action_terms = {**direct, "E_kinetic": kinetic, "rmf_action": energy}
        stationarity_residual = (
            torch.sqrt(torch.mean(torch.cat(stationarity_pieces).square()))
            * self.functional.params.hbar_c
        )
        return DensityMatrixResult(
            energy_per_a_no_com=float((energy / self.functional.a).detach().cpu()),
            energy_total_no_com=float(energy.detach().cpu()),
            energy_total_tensor=energy,
            stationarity_residual_tensor=stationarity_residual,
            hamiltonian=hamiltonian,
            blocks=blocks,
            densities=densities,
            fields=fields,
            action_terms=action_terms,
            diagnostics=diagnostics,
        )

    def final_spectral_validation(self, hamiltonian: torch.Tensor) -> dict[str, torch.Tensor]:
        self.final_validation_dirac_solves += 1
        return self.matrix_solver.solve(hamiltonian.detach())
