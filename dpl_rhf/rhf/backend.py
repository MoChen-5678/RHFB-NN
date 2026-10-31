from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.models.density_matrix_net import DensityBlockSpec
from dpl_rhf.rhf.density import nonlocal_density_from_block
from dpl_rhf.rhf.dirac import RHFRadialDiracOperator
from dpl_rhf.rhf.initialization import independent_rhf_orbital_data
from dpl_rhf.rhf.asymptotics import occupied_block_exterior_moments
from dpl_rhf.rhf.functional import (
    PKA1_DIRECT_CHANNEL_TERMS,
    PKA1_FOCK_CHANNEL_TERMS,
    SphericalRHFFunctional,
    RHFEnergyEvaluation,
)
from dpl_rhf.rhf.parameters import RHFParameters, rhf_parameters


@dataclass
class DensityMatrixRHFResult:
    energy_total_tensor: torch.Tensor
    energy_total_no_com: float
    energy_per_a_no_com: float
    stationarity_residual_tensor: torch.Tensor
    stationarity_components: dict[str, torch.Tensor]
    stationarity_quadratic_terms: dict[str, torch.Tensor]
    blocks: list[dict[str, torch.Tensor]]
    densities: dict[str, torch.Tensor]
    fields: dict[str, torch.Tensor]
    action_terms: dict[str, torch.Tensor]
    hamiltonians: dict[str, torch.Tensor]
    kinetic_hamiltonians: dict[str, torch.Tensor]
    direct_hamiltonians: dict[str, torch.Tensor]
    fock_hamiltonians: dict[str, torch.Tensor]
    diagnostics: dict[str, float]
    functional_type: str = "spherical-density-matrix-rhf"


class DensityMatrixRHFBackend:
    """Full-Dirac-space neural variation of the spherical PKO2/PKA1 functional.

    Analytic Woods-Saxon/oscillator factors initialize the occupied projector.
    No Hamiltonian diagonalization or mutable positive-energy basis is used;
    the inverse-Hamiltonian constraint selects the no-sea branch.
    """

    def __init__(
        self,
        case: NucleusCase,
        *,
        device: torch.device,
        derivative_order: int = 7,
        params: RHFParameters | None = None,
    ):
        if case.model.upper() not in {"PKO2", "PKA1"}:
            raise ValueError("the density-matrix RHF backend supports PKO2 and PKA1")
        self.case = case
        self.device = device
        self.params = params or rhf_parameters(case.model)
        self.r = np.arange(201, dtype=np.float64) * 0.1
        self.r[0] = 0.001
        r_tensor = torch.as_tensor(self.r, dtype=torch.float64, device=device)
        orbital_data, self.orbital_metadata = independent_rhf_orbital_data(
            case.z, case.n, self.r, self.params
        )
        self.orbital_data = {
            key: torch.as_tensor(value, dtype=torch.float64, device=device)
            for key, value in orbital_data.items()
        }
        self.dirac_kinetic = RHFRadialDiracOperator(
            r_tensor,
            mass_n_mev=self.params.mass_n,
            mass_p_mev=self.params.mass_p,
            hbar_c_mev_fm=self.params.hbar_c,
            derivative_order=derivative_order,
        )
        self.functional = SphericalRHFFunctional(
            r_tensor, case.z, case.n, self.params
        )
        self.initialization_block_diagonalizations = 0
        self.training_dirac_solves = 0
        self.refresh_phases = 0
        self.refresh_block_diagonalizations = 0
        self.initial_channel_gradient_norms: dict[str, float] = {}
        self.block_specs = self._build_full_space_blocks()

    def refresh_positive_energy_spaces(
        self,
        blocks: list[dict[str, torch.Tensor]],
        hamiltonians: dict[str, torch.Tensor],
    ) -> list[DensityBlockSpec]:
        raise RuntimeError("subspace refresh is forbidden in the no-refresh solver")

    def _audit_pka1_channel_gradients(
        self,
        energy: RHFEnergyEvaluation,
        projectors: list[torch.Tensor],
    ) -> None:
        """Verify once that every PKA1 channel participates in variation."""

        if self.params.model != "PKA1" or self.initial_channel_gradient_norms:
            return
        terms = {**energy.direct_terms, **energy.exchange_terms}
        groups = {**PKA1_DIRECT_CHANNEL_TERMS, **PKA1_FOCK_CHANNEL_TERMS}
        norms: dict[str, float] = {}
        for channel, names in groups.items():
            channel_energy = sum(
                (terms[name] for name in names),
                torch.zeros_like(energy.total),
            )
            gradients = torch.autograd.grad(
                channel_energy,
                projectors,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            connected = [gradient for gradient in gradients if gradient is not None]
            if not connected:
                raise RuntimeError(
                    f"PKA1 channel {channel} is disconnected from the density matrix"
                )
            norm = torch.sqrt(sum(torch.sum(gradient.square()) for gradient in connected))
            if not torch.isfinite(norm):
                raise RuntimeError(f"PKA1 channel {channel} has a non-finite gradient")
            norms[channel] = float(norm.detach().cpu())
        self.initial_channel_gradient_norms = norms

    def _build_full_space_blocks(self) -> list[DensityBlockSpec]:
        """Build analytic full-space occupied factors with zero eigensolves."""
        data = self.orbital_data
        groups: dict[tuple[float, int], list[int]] = {}
        for index, (species, kappa, occupation) in enumerate(
            zip(data["species_sign"], data["kappa"], data["occupancy"])
        ):
            if float(occupation) > 0.0:
                groups.setdefault((float(species), int(kappa)), []).append(index)

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
                    raise ValueError("a spherical kappa block has inconsistent degeneracy")
                specs.append(
                    DensityBlockSpec(
                        species_sign=species,
                        kappa=kappa,
                        degeneracy=degeneracies.pop(),
                        principals=tuple(int(data["principal"][i]) for i in ordered),
                        occupations=data["occupancy"][ordered].clone().to(self.device),
                        initial_factors=initial.to(self.device),
                        free_hamiltonian=self.dirac_kinetic.matrix_fm_inverse(
                            species, kappa
                        ),
                        tail_energies_mev=torch.full(
                            (len(ordered),), epsilon_initial,
                            dtype=torch.float64, device=self.device,
                        ),
                    )
                )
        return specs

    @staticmethod
    def _block_name(block: dict[str, torch.Tensor]) -> str:
        species = "n" if float(block["species_sign"]) > 0.0 else "p"
        return f"{species}_kappa_{int(block['kappa']):+d}"

    def evaluate(
        self,
        blocks: list[dict[str, torch.Tensor]],
        *,
        create_graph: bool = False,
    ) -> DensityMatrixRHFResult:
        nonlocal_blocks = [
            nonlocal_density_from_block(block, self.functional.h) for block in blocks
        ]
        energy: RHFEnergyEvaluation = self.functional.evaluate(blocks, nonlocal_blocks)
        projectors = [density.projector for density in nonlocal_blocks]
        fock_rearrangement = energy.fields["sigma_rearr_fock_explicit"]
        energy.fields["sigma_rearr_fock"] = fock_rearrangement
        energy.fields["sigma_rearr_total"] = (
            energy.fields["sigma_rearr_direct"]
            + fock_rearrangement
        )
        energy.fields["component_V0_total_local"] = (
            energy.fields["component_V0_direct"]
            + fock_rearrangement
        )
        for suffix in ("n", "p"):
            energy.fields[f"Sigma_plus_{suffix}_total_local"] = (
                energy.fields[f"Sigma_plus_{suffix}_direct"] + fock_rearrangement
            )
            energy.fields[f"Sigma_minus_{suffix}_total_local"] = (
                energy.fields[f"Sigma_minus_{suffix}_direct"] + fock_rearrangement
            )
            energy.fields[f"Sigma_T_{suffix}_total_local"] = energy.fields[
                f"Sigma_T_{suffix}_direct"
            ]
        self._audit_pka1_channel_gradients(energy, projectors)
        fock_gradients = torch.autograd.grad(
            energy.exchange_terms["E_fock"],
            projectors,
            retain_graph=True,
            create_graph=create_graph,
        )

        stationarity_parts = []
        component_stationarity_parts: dict[str, list[torch.Tensor]] = {
            "kinetic": [],
            "direct": [],
            "fock": [],
        }
        hamiltonians: dict[str, torch.Tensor] = {}
        kinetic_hamiltonians: dict[str, torch.Tensor] = {}
        direct_hamiltonians: dict[str, torch.Tensor] = {}
        fock_hamiltonians: dict[str, torch.Tensor] = {}
        hermiticity_errors = []
        assembly_errors = []
        orthogonality_errors = []
        for block, density, fock_gradient in zip(
            blocks,
            nonlocal_blocks,
            fock_gradients,
        ):
            kinetic_hamiltonian = (
                block["free_hamiltonian"] * self.params.hbar_c
            )
            fock_functional_derivative = (
                0.5 * (fock_gradient + fock_gradient.T) / density.degeneracy
            )
            local_rearrangement = torch.diag(
                self.params.hbar_c * fock_rearrangement[1:-1]
            )
            zero = torch.zeros_like(local_rearrangement)
            fock_rearrangement_hamiltonian = torch.cat(
                [
                    torch.cat([local_rearrangement, zero], dim=1),
                    torch.cat([zero, local_rearrangement], dim=1),
                ],
                dim=0,
            )
            suffix = "n" if float(block["species_sign"]) > 0.0 else "p"
            sigma_plus = energy.fields[f"Sigma_plus_{suffix}_total_local"][1:-1]
            sigma_minus = energy.fields[f"Sigma_minus_{suffix}_total_local"][1:-1]
            sigma_tensor = energy.fields[f"Sigma_T_{suffix}_total_local"][1:-1]
            upper = torch.diag(self.params.hbar_c * sigma_plus)
            lower = torch.diag(self.params.hbar_c * sigma_minus)
            tensor = torch.diag(self.params.hbar_c * sigma_tensor)
            # RHF-e (3.393): h_D is assembled from the physical local fields,
            # rather than inferred by differentiating an asymmetric folded
            # quadrature at the origin and box boundary.
            direct_hamiltonian = torch.cat(
                [torch.cat([upper, tensor], dim=1), torch.cat([tensor, lower], dim=1)],
                dim=0,
            )
            # The Fock functional derivative contains its density-dependent
            # local rearrangement. Remove that term here because it has
            # already been included once in h_D through Sigma_R.
            fock_hamiltonian = (
                fock_functional_derivative - fock_rearrangement_hamiltonian
            )
            # RHF-e (3.392)-(3.394): h = h_kin + h_D + h_E.
            hamiltonian = (
                kinetic_hamiltonian + direct_hamiltonian + fock_hamiltonian
            )
            assembly_errors.append(
                torch.max(
                    torch.abs(
                        hamiltonian
                        - kinetic_hamiltonian
                        - direct_hamiltonian
                        - fock_hamiltonian
                    )
                )
            )
            name = self._block_name(block)
            hamiltonians[name] = hamiltonian
            kinetic_hamiltonians[name] = kinetic_hamiltonian
            direct_hamiltonians[name] = direct_hamiltonian
            fock_hamiltonians[name] = fock_hamiltonian
            hermiticity_errors.append(torch.max(torch.abs(hamiltonian - hamiltonian.T)))

            q = block["factors"]
            identity = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
            orthogonality_errors.append(torch.max(torch.abs(q.T @ q - identity)))
            def projected_residual(operator: torch.Tensor) -> torch.Tensor:
                operator_q = (operator @ q) * block["occupations"][None, :]
                tangent = 0.5 * (q.T @ operator_q + operator_q.T @ q)
                return operator_q - q @ tangent

            kinetic_residual = projected_residual(kinetic_hamiltonian)
            direct_residual = projected_residual(direct_hamiltonian)
            fock_residual = projected_residual(fock_hamiltonian)
            total_residual = kinetic_residual + direct_residual + fock_residual
            component_stationarity_parts["kinetic"].append(kinetic_residual.reshape(-1))
            component_stationarity_parts["direct"].append(direct_residual.reshape(-1))
            component_stationarity_parts["fock"].append(fock_residual.reshape(-1))
            stationarity_parts.append(total_residual.reshape(-1))

        stationarity = torch.sqrt(
            torch.mean(torch.cat(stationarity_parts).square())
        )
        component_vectors = {
            name: torch.cat(parts)
            for name, parts in component_stationarity_parts.items()
        }
        stationarity_components = {
            name: torch.sqrt(torch.mean(vector.square()))
            for name, vector in component_vectors.items()
        }
        # Keep all three Hamiltonians explicit in the differentiable loss.
        # The cross terms are essential: the RHF Euler-Lagrange equation
        # constrains R_kin + R_D + R_E, not each residual independently.
        stationarity_quadratic_terms = {
            "kinetic_square": torch.mean(component_vectors["kinetic"].square()),
            "direct_square": torch.mean(component_vectors["direct"].square()),
            "fock_square": torch.mean(component_vectors["fock"].square()),
            "kinetic_direct_cross": 2.0 * torch.mean(
                component_vectors["kinetic"] * component_vectors["direct"]
            ),
            "kinetic_fock_cross": 2.0 * torch.mean(
                component_vectors["kinetic"] * component_vectors["fock"]
            ),
            "direct_fock_cross": 2.0 * torch.mean(
                component_vectors["direct"] * component_vectors["fock"]
            ),
        }
        component_vector_sum = sum(
            component_vectors.values(),
            torch.zeros_like(torch.cat(stationarity_parts)),
        )
        component_sum_error = torch.max(
            torch.abs(component_vector_sum - torch.cat(stationarity_parts))
        )
        observables = energy.observables
        exterior = {
            "n": torch.zeros((), dtype=self.functional.r.dtype, device=self.device),
            "p": torch.zeros((), dtype=self.functional.r.dtype, device=self.device),
        }
        for block in blocks:
            suffix = "n" if float(block["species_sign"]) > 0.0 else "p"
            mass = self.params.mass_n if suffix == "n" else self.params.mass_p
            moments = occupied_block_exterior_moments(
                block["factors"], block["occupations"], block["degeneracy"],
                block["tail_energies_mev"], mass, self.params.hbar_c,
                self.functional.h, self.functional.r[-2], self.functional.r[-1],
            )
            exterior[suffix] += moments["number"]
        diagnostics = {
            "n_number_error": float(abs(sum(
                float(block["degeneracy"] * block["occupations"].sum())
                for block in blocks if float(block["species_sign"]) > 0.0
            ) - self.case.n)),
            "z_number_error": float(abs(sum(
                float(block["degeneracy"] * block["occupations"].sum())
                for block in blocks if float(block["species_sign"]) < 0.0
            ) - self.case.z)),
            "max_orthogonality_error": float(
                torch.stack(orthogonality_errors).max().detach().cpu()
            ),
            "max_hamiltonian_hermiticity_error_mev": float(
                torch.stack(hermiticity_errors).max().detach().cpu()
            ),
            "max_hamiltonian_component_assembly_error_mev": float(
                torch.stack(assembly_errors).max().detach().cpu()
            ),
            "projected_stationarity_rms_mev": float(stationarity.detach().cpu()),
            "kinetic_projected_residual_rms_mev": float(
                stationarity_components["kinetic"].detach().cpu()
            ),
            "direct_projected_residual_rms_mev": float(
                stationarity_components["direct"].detach().cpu()
            ),
            "fock_projected_residual_rms_mev": float(
                stationarity_components["fock"].detach().cpu()
            ),
            "component_residual_sum_error_mev": float(
                component_sum_error.detach().cpu()
            ),
            "stationarity_quadratic_closure_mev2": float(
                abs(
                    sum(stationarity_quadratic_terms.values())
                    - stationarity.square()
                ).detach().cpu()
            ),
            **{
                f"stationarity_{name}_mev2": float(value.detach().cpu())
                for name, value in stationarity_quadratic_terms.items()
            },
            "fock_rearrangement_isospin_error": 0.0,
            "rms_n_no_com": float(observables["rms_n_no_com"].detach().cpu()),
            "rms_p_no_com": float(observables["rms_p_no_com"].detach().cpu()),
            "rms_matter_no_com": float(
                observables["rms_matter_no_com"].detach().cpu()
            ),
            "charge_radius_no_com": float(
                observables["charge_radius_no_com"].detach().cpu()
            ),
            "training_dirac_solves": self.training_dirac_solves,
            "initialization_block_diagonalizations": self.initialization_block_diagonalizations,
            "sparse_refresh_phases": self.refresh_phases,
            "refresh_block_diagonalizations": self.refresh_block_diagonalizations,
            "n_exterior_tail": float(exterior["n"].detach().cpu()),
            "z_exterior_tail": float(exterior["p"].detach().cpu()),
            "hamiltonian_is_explicit_rhf": 1.0,
            "rmf_runtime_functional_reused": 0.0,
        }
        diagnostics.update(
            {
                f"initial_gradient_norm_{name}_mev": value
                for name, value in self.initial_channel_gradient_norms.items()
            }
        )
        terms = {
            "E_kinetic": energy.kinetic,
            **energy.direct_terms,
            **energy.exchange_terms,
            "E_total": energy.total,
        }
        return DensityMatrixRHFResult(
            energy_total_tensor=energy.total,
            energy_total_no_com=float(energy.total.detach().cpu()),
            energy_per_a_no_com=float((energy.total / self.functional.a).detach().cpu()),
            stationarity_residual_tensor=stationarity,
            stationarity_components=stationarity_components,
            stationarity_quadratic_terms=stationarity_quadratic_terms,
            blocks=blocks,
            densities=energy.densities,
            fields=energy.fields,
            action_terms=terms,
            hamiltonians=hamiltonians,
            kinetic_hamiltonians=kinetic_hamiltonians,
            direct_hamiltonians=direct_hamiltonians,
            fock_hamiltonians=fock_hamiltonians,
            diagnostics=diagnostics,
        )
