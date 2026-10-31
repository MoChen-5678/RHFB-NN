from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.functionals.pkdd_rmf import PKDDRMFFunctionalSpec
from dpl_rhf.functionals.pkdd_action import (
    PKDDParameters,
    PKDDRMFFunctional,
    rmf_parameters,
)


SHELLS = [
    ("1s1/2", -1, 0, 1, 2), ("1p3/2", -2, 1, 1, 4), ("1p1/2", 1, 1, 1, 2),
    ("1d5/2", -3, 2, 1, 6), ("2s1/2", -1, 0, 2, 2), ("1d3/2", 2, 2, 1, 4),
    ("1f7/2", -4, 3, 1, 8), ("2p3/2", -2, 1, 2, 4), ("1f5/2", 3, 3, 1, 6),
    ("2p1/2", 1, 1, 2, 2), ("1g9/2", -5, 4, 1, 10), ("1g7/2", 4, 4, 1, 8),
    ("2d5/2", -3, 2, 2, 6), ("2d3/2", 2, 2, 2, 4), ("3s1/2", -1, 0, 3, 2),
    ("1h11/2", -6, 5, 1, 12), ("2f7/2", -4, 3, 2, 8), ("1h9/2", 5, 5, 1, 10),
    ("1i13/2", -7, 6, 1, 14), ("3p3/2", -2, 1, 3, 4), ("2f5/2", 3, 3, 2, 6),
    ("3p1/2", 1, 1, 3, 2),
]


def _spectroscopic_letter(angular: int) -> str:
    letters = "spdfghiklmnoqrtuvwxyz"
    if angular < len(letters):
        return letters[angular]
    return f"l{angular}"


def spherical_shells(target_capacity: int) -> list[tuple[str, int, int, int, int]]:
    """Return a deterministic spherical shell space with sufficient capacity.

    The established Core-compatible ordering is retained as an exact prefix.
    For nuclei beyond that table, additional ``(n, kappa)`` blocks are added
    major-shell by major-shell.  Their ordering only initializes the density;
    the variational functional and final filling determine the physical order.
    """

    if target_capacity < 0:
        raise ValueError("target shell capacity cannot be negative")
    shells = list(SHELLS)
    capacity = sum(shell[4] for shell in shells)
    seen = {(shell[2], shell[3], shell[1]) for shell in shells}
    major = 0
    while capacity < target_capacity:
        for angular in range(major, -1, -2):
            principal = (major - angular) // 2 + 1
            kappas = [-(angular + 1)]
            if angular > 0:
                kappas.append(angular)
            for kappa in kappas:
                key = (angular, principal, kappa)
                if key in seen:
                    continue
                seen.add(key)
                twice_j = 2 * abs(kappa) - 1
                name = (
                    f"{principal}{_spectroscopic_letter(angular)}"
                    f"{twice_j}/2"
                )
                degeneracy = 2 * abs(kappa)
                shells.append((name, kappa, angular, principal, degeneracy))
                capacity += degeneracy
        major += 1
    return shells


@dataclass
class TorchBackendResult:
    energy_per_a_no_com: float
    energy_total_no_com: float
    energy_per_a_tensor: torch.Tensor
    hamiltonian: torch.Tensor
    reconstructed_hamiltonian: torch.Tensor
    grad_energy_h: torch.Tensor
    orbitals: dict[str, torch.Tensor]
    densities: dict[str, torch.Tensor]
    fields: dict[str, torch.Tensor]
    action_terms: dict[str, torch.Tensor]
    diagnostics: dict[str, float]
    functional_type: str = "rmf"


def generalized_laguerre(order: int, alpha: float, x: np.ndarray) -> np.ndarray:
    if order == 0:
        return np.ones_like(x)
    previous = np.ones_like(x)
    current = 1.0 + alpha - x
    for k in range(2, order + 1):
        previous, current = current, ((2 * k - 1 + alpha - x) * current - (k - 1 + alpha) * previous) / k
    return current


def independent_orbital_data(
    z: int,
    n: int,
    r: np.ndarray,
    params: PKDDParameters | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Nucleus-only occupied shell metadata and textbook Woods-Saxon initial orbitals."""
    records: list[tuple[Any, ...]] = []
    a = z + n
    hbar_omega = 41.0 * a ** (-1.0 / 3.0)
    p = params or PKDDParameters()
    for species, number, sign in (("n", n, 1.0), ("p", z, -1.0)):
        bare_mass = p.mass_n if species == "n" else p.mass_p
        oscillator_length = math.sqrt(p.hbar_c**2 / (bare_mass * hbar_omega))
        remaining = int(number)
        represented = 0
        target_capacity = int(number) + 12
        for name, kappa, angular, principal, degeneracy in spherical_shells(target_capacity):
            if represented >= target_capacity and remaining <= 0:
                break
            particles = min(max(remaining, 0), degeneracy)
            occupation = particles / degeneracy
            q = (r / oscillator_length) ** 2
            radial = generalized_laguerre(principal - 1, angular + 0.5, q)
            large = r ** (angular + 1) * np.exp(-0.5 * q) * radial
            large /= np.sqrt(np.trapezoid(large * large, r))
            records.append((species, sign, name, kappa, angular, principal, occupation, degeneracy, large))
            remaining -= particles
            represented += degeneracy
        if remaining:
            raise ValueError(f"shell table does not cover {species}={number}")

    v0, av = -71.28, 11.1175
    radius_plus = 0.5 * (1.2334 + 1.2496) * a ** (1.0 / 3.0)
    radius_minus = 0.5 * (1.1443 + 1.1400) * a ** (1.0 / 3.0)
    diffuseness_plus = 0.5 * (0.6150 + 0.6124)
    diffuseness_minus = 0.5 * (0.6476 + 0.6469)
    sigma_plus = v0 / (1.0 + np.exp((r - radius_plus) / diffuseness_plus))
    sigma_minus = -v0 * av / (1.0 + np.exp((r - radius_minus) / diffuseness_minus))
    charge_radius = 1.2496 * a ** (1.0 / 3.0)
    alpha = 1.0 / p.alpha_inv
    initial_coul = np.where(
        r < charge_radius,
        alpha * z * (3.0 / (2.0 * charge_radius) - r * r / (2.0 * charge_radius**3)),
        alpha * z / r,
    )
    epsilon_initial = -0.5 * hbar_omega
    initial_large = np.stack([row[8] for row in records])
    initial_small = []
    for row, large in zip(records, initial_large):
        coulomb = initial_coul if row[0] == "p" else 0.0
        mass = p.mass_n if row[0] == "n" else p.mass_p
        vms = sigma_minus / p.hbar_c + coulomb - 2.0 * mass / p.hbar_c
        derivative = np.gradient(large, r, edge_order=2)
        small = (derivative + row[3] * large / r) / (epsilon_initial / p.hbar_c - vms)
        norm = np.sqrt(np.trapezoid(large * large + small * small, r))
        large /= norm
        initial_small.append(small / norm)

    data = {
        "G": initial_large,
        "F": np.stack(initial_small),
        "kappa": np.asarray([row[3] for row in records], dtype=np.int32),
        "principal": np.asarray([row[5] for row in records], dtype=np.int32),
        "occupancy": np.asarray([row[6] for row in records], dtype=np.float64),
        "degeneracy": np.asarray([row[7] for row in records], dtype=np.float64),
        "species_sign": np.asarray([row[1] for row in records], dtype=np.float64),
    }
    metadata = {
        "species": np.asarray([row[0] for row in records]),
        "it": np.asarray([1 if row[0] == "n" else 2 for row in records], dtype=np.int32),
        "index": np.arange(1, len(records) + 1, dtype=np.int32),
        "name": np.asarray([row[2] for row in records]),
        "l": np.asarray([row[4] for row in records], dtype=np.int32),
        "n": np.asarray([row[5] for row in records], dtype=np.int32),
    }
    return data, metadata


def _first_derivative_coefficients(offsets: list[int]) -> np.ndarray:
    matrix = np.array([[float(offset) ** power for offset in offsets] for power in range(len(offsets))])
    rhs = np.zeros(len(offsets))
    rhs[1] = 1.0
    return np.linalg.solve(matrix, rhs)


def noncentral_derivative_matrix(r: torch.Tensor, order: int = 7) -> torch.Tensor:
    x = r[1:-1]
    size = x.numel()
    h = float(r[2] - r[1])
    derivative = torch.zeros((size, size), dtype=r.dtype, device=r.device)
    if order == 1:
        derivative.diagonal().fill_(-1.0 / h)
        derivative.diagonal(offset=1).fill_(1.0 / h)
        return derivative
    if order in {4, 5, 6, 7}:
        points = 5 if order in {4, 5} else 7
        if size < points:
            raise ValueError(f"{points}-point asymmetric derivative needs at least {points} interior points")
        coefficients = torch.as_tensor(
            _first_derivative_coefficients(list(range(points))) / h,
            dtype=r.dtype,
            device=r.device,
        )
        # The paper imposes f=0 at and outside the radial box.  Keep one
        # forward ADF throughout the box and drop coefficients multiplying
        # those known exterior values.  Switching to a backward closure here
        # changes the discrete operator and reintroduces boundary states.
        for row in range(size):
            stop = min(size, row + points)
            derivative[row, row:stop] = coefficients[:stop - row]
        return derivative
    raise ValueError("use derivative order 1, 4, 5, 6, or 7")


class DifferentiableDiracMatrix:
    """Paper-faithful coordinate-space ADF matrix for the spherical Dirac equation."""

    def __init__(
        self,
        r: torch.Tensor,
        orbital_data: dict[str, np.ndarray],
        orbital_weights: torch.Tensor,
        derivative_order: int = 7,
        params: PKDDParameters | None = None,
    ):
        self.r = r
        self.x = r[1:-1]
        self.derivative_plus = noncentral_derivative_matrix(r, derivative_order)
        self.metric_weights = orbital_weights[1:-1]
        if torch.any(self.metric_weights <= 0.0):
            raise ValueError("Dirac quadrature weights must be positive on the interior mesh")
        # Zhang et al. use the ordinary coordinate-space matrix transpose:
        # -D_backward = D_forward^T.  Folding a nonuniform quadrature metric
        # into this relation changes the ADF coefficients and is not Eq. (15).
        self.derivative_minus = -self.derivative_plus.T
        self.stencil_points = 5 if derivative_order in {4, 5} else 7
        self.params = params or PKDDParameters()
        self.orbital_data = {
            key: torch.as_tensor(value, dtype=r.dtype, device=r.device)
            for key, value in orbital_data.items()
            if key in {"G", "F", "kappa", "species_sign", "occupancy", "degeneracy", "principal"}
        }

    @staticmethod
    def _component_angular_momenta(kappa: int) -> tuple[int, int]:
        """Return the upper/lower orbital angular momenta for a Dirac state."""
        if kappa < 0:
            return -kappa - 1, -kappa
        return kappa, kappa - 1

    def _regular_basis(self, angular: int) -> torch.Tensor:
        """Return coordinate samples; physical quadrature normalization follows."""
        size = self.x.numel()
        return torch.eye(size, dtype=self.r.dtype, device=self.r.device)

    def _block_system(
        self, stack: torch.Tensor, species_sign: float, kappa: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = 0 if species_sign > 0 else 2
        vps = stack[offset, 1:-1]
        vms = stack[offset + 1, 1:-1]
        l_upper, l_lower = self._component_angular_momenta(kappa)
        basis_g = self._regular_basis(l_upper)
        basis_f = self._regular_basis(l_lower)
        angular = torch.diag(torch.full_like(self.x, float(kappa)) / self.x)
        # Paper equations (11)-(15): construct the two off-diagonal blocks
        # independently.  The paper's parity refers to radial extension. Since
        # G(r) ~ r^(l_upper+1), even orbital l gives odd radial G and must use
        # forward ADF; odd orbital l gives even radial G and uses backward.
        radial_g_is_odd = (l_upper + 1) % 2 == 1
        if radial_g_is_odd:
            derivative_forward = self.derivative_plus
            derivative_backward = self.derivative_minus
        else:
            derivative_forward = self.derivative_minus
            derivative_backward = self.derivative_plus
        b1_physical = -derivative_backward + angular
        b2_physical = derivative_forward + angular
        upper = torch.cat([torch.diag(vps), b1_physical], dim=1)
        lower = torch.cat([b2_physical, torch.diag(vms)], dim=1)
        hamiltonian = torch.cat([upper, lower], dim=0)
        identity = torch.eye(hamiltonian.shape[0], dtype=self.r.dtype, device=self.r.device)
        return hamiltonian, identity, basis_g, basis_f

    def matrix(self, stack: torch.Tensor, species_sign: float, kappa: int) -> torch.Tensor:
        return self._block_system(stack, species_sign, kappa)[0]

    def _physical_candidates(
        self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, species: float
    ) -> torch.Tensor:
        p = self.params
        mass = p.mass_n if species > 0 else p.mass_p
        # With the rest-mass shift, epsilon > -M cleanly separates the physical
        # positive-energy branch from the Dirac sea.  An upper-component norm
        # threshold is an empirical constraint and is not part of no-sea RMF.
        return torch.where(eigenvalues * p.hbar_c > -mass)[0]

    def solve(self, stack: torch.Tensor) -> dict[str, torch.Tensor]:
        count = len(self.orbital_data["kappa"])
        G = torch.zeros((count, self.r.numel()), dtype=self.r.dtype, device=self.r.device)
        F = torch.zeros_like(G)
        energies = torch.zeros(count, dtype=self.r.dtype, device=self.r.device)
        selected_indices = torch.zeros(count, dtype=torch.int64, device=self.r.device)
        spectral_residuals = torch.zeros(count, dtype=self.r.dtype, device=self.r.device)
        kinetic_expectations = torch.zeros(count, dtype=self.r.dtype, device=self.r.device)
        node_counts = torch.zeros(count, dtype=torch.int64, device=self.r.device)
        expected_node_counts = torch.clamp(self.orbital_data["principal"].to(torch.int64) - 1, min=0)
        groups: dict[tuple[float, int], list[int]] = {}
        for i, (species, kappa) in enumerate(zip(self.orbital_data["species_sign"], self.orbital_data["kappa"])):
            groups.setdefault((float(species), int(kappa)), []).append(i)
        for (species, kappa), orbit_indices in groups.items():
            h_block, cholesky, basis_g, basis_f = self._block_system(stack, species, kappa)
            eigenvalues, eigenvectors = torch.linalg.eigh(h_block)
            candidates = self._physical_candidates(eigenvalues, eigenvectors, species)
            ordered_orbits = sorted(orbit_indices, key=lambda i: int(self.orbital_data["principal"][i]))
            if candidates.numel() < len(ordered_orbits):
                raise RuntimeError(f"not enough no-sea states for species={species}, kappa={kappa}")
            candidate_vectors = eigenvectors[:, candidates]
            candidate_coefficients = candidate_vectors
            block_size = basis_g.shape[1]
            candidate_g = basis_g @ candidate_coefficients[:block_size]
            candidate_f = basis_f @ candidate_coefficients[block_size:]
            candidate_norm = torch.sqrt(torch.sum(
                self.metric_weights[:, None]
                * (candidate_g.square() + candidate_f.square()),
                dim=0,
            )).clamp_min(1.0e-14)
            candidate_g = candidate_g / candidate_norm
            candidate_f = candidate_f / candidate_norm
            candidate_nodes = []
            for column in range(candidate_g.shape[1]):
                values = candidate_g[:, column]
                grid_index = torch.arange(values.numel(), device=values.device)
                closure = self.stencil_points
                active = (
                    (values.abs() > values.abs().max() * 1.0e-3)
                    & (grid_index >= closure)
                    & (grid_index < values.numel() - closure)
                )
                filtered = values[active]
                nodes = int(torch.sum(filtered[1:] * filtered[:-1] < 0.0).detach().cpu()) if filtered.numel() > 1 else 0
                candidate_nodes.append(nodes)
            used: set[int] = set()
            for orbit_index in ordered_orbits:
                reference_g = self.orbital_data["G"][orbit_index, 1:-1]
                reference_f = self.orbital_data["F"][orbit_index, 1:-1]
                overlaps = torch.abs(
                    torch.sum(self.metric_weights[:, None] * reference_g[:, None] * candidate_g, dim=0)
                    + torch.sum(self.metric_weights[:, None] * reference_f[:, None] * candidate_f, dim=0)
                )
                principal = int(self.orbital_data["principal"][orbit_index])
                expected_nodes = principal - 1
                # Analytic nucleus-only orbitals identify the physical branch;
                # node count remains an independent gate.  Rewarding a node
                # match can select a deeply bound boundary mode over the smooth
                # physical state.
                ranking = torch.argsort(overlaps.detach(), descending=True)
                matching = [
                    int(position) for position in ranking
                    if int(position) not in used and candidate_nodes[int(position)] == expected_nodes
                ]
                if not matching:
                    raise RuntimeError(
                        f"no physical node-matched state for species={species}, "
                        f"kappa={kappa}, principal={principal}"
                    )
                candidate_position = matching[0]
                used.add(candidate_position)
                selected = candidates[candidate_position]
                vector = eigenvectors[:, selected]
                physical_g = candidate_g[:, candidate_position]
                physical_f = candidate_f[:, candidate_position]
                upper = torch.cat([vector.new_zeros(1), physical_g, vector.new_zeros(1)])
                lower = torch.cat([vector.new_zeros(1), physical_f, vector.new_zeros(1)])
                G[orbit_index] = upper
                F[orbit_index] = lower
                energies[orbit_index] = eigenvalues[selected]
                selected_indices[orbit_index] = selected
                node_counts[orbit_index] = candidate_nodes[candidate_position]
                residual = h_block @ vector - eigenvalues[selected] * vector
                spectral_residuals[orbit_index] = torch.linalg.norm(residual)
                zero_stack = torch.zeros_like(stack)
                zero_stack[1] = -2.0 * self.params.mass_n / self.params.hbar_c
                zero_stack[3] = -2.0 * self.params.mass_p / self.params.hbar_c
                free_block = self.matrix(zero_stack, species, kappa)
                kinetic_expectations[orbit_index] = (
                    vector @ free_block @ vector * self.params.hbar_c
                )
        # Nuclear ground-state occupation: globally fill the lowest physical
        # states for each species. Nominal shell occupations only define N/Z.
        dynamic_occupancy = torch.zeros_like(self.orbital_data["occupancy"])
        nominal_particles = self.orbital_data["occupancy"] * self.orbital_data["degeneracy"]
        for species in (1.0, -1.0):
            indices = torch.where(self.orbital_data["species_sign"] == species)[0]
            target = float(nominal_particles[indices].sum().detach().cpu())
            remaining = target
            for index in indices[torch.argsort(energies[indices].detach())]:
                degeneracy = float(self.orbital_data["degeneracy"][index].detach().cpu())
                particles = min(max(remaining, 0.0), degeneracy)
                dynamic_occupancy[index] = particles / degeneracy
                remaining -= particles
            if remaining > 1.0e-8:
                raise RuntimeError(f"candidate shell space cannot hold species={species} target={target}")
        return {
            "G": G,
            "F": F,
            "epsilon": energies * self.params.hbar_c,
            "eigen_index": selected_indices,
            "spectral_residual": spectral_residuals,
            "kinetic_expectation_mev": kinetic_expectations,
            "node_count": node_counts,
            "expected_node_count": expected_node_counts,
            **{key: self.orbital_data[key] for key in ("kappa", "species_sign", "occupancy", "degeneracy")},
            "occupancy": dynamic_occupancy,
        }

    def hermiticity_error(self, stack: torch.Tensor) -> float:
        errors = []
        for species, kappa in {(float(s), int(k)) for s, k in zip(self.orbital_data["species_sign"], self.orbital_data["kappa"])}:
            h = self.matrix(stack, species, kappa)
            errors.append(torch.max(torch.abs(h - h.T)).detach())
        return float(torch.stack(errors).max().cpu())


class TorchRMFBackend:
    """Differentiable density-dependent RMF backend implementing PRL E[H]."""

    def __init__(self, case: NucleusCase, *, device: torch.device, derivative_order: int = 7):
        self.case = case
        self.device = device
        self.derivative_order = derivative_order
        # Core-1204/Define.f90 and Inout.f90 define R=20 fm, h=0.1 fm,
        # npt=201 and replace the origin by h*1e-2.  Reproduce that mesh
        # directly: constructing the differentiable backend must never invoke
        # the Fortran initializer or an SCF calculation.
        self.r = np.arange(201, dtype=np.float64) * 0.1
        self.r[0] = 0.001
        self.params = rmf_parameters(case.model)
        self.functional_spec = PKDDRMFFunctionalSpec(
            case.z, case.n, self.r, params=self.params
        )
        self.initial_hamiltonian_np = self.functional_spec.initial_hamiltonian()
        r_tensor = torch.as_tensor(self.r, dtype=torch.float64, device=device)
        self.functional = PKDDRMFFunctional(
            r_tensor, case.z, case.n, params=self.params
        )
        self.orbital_data, self.orbital_metadata = independent_orbital_data(
            case.z, case.n, self.r, params=self.params
        )
        self.matrix_solver = DifferentiableDiracMatrix(
            r_tensor,
            self.orbital_data,
            self.functional.orbital_w,
            derivative_order,
            params=self.params,
        )

    def evaluate_tensor(self, hamiltonian: torch.Tensor) -> TorchBackendResult:
        if hamiltonian.shape != (4, self.r.size):
            raise ValueError(f"expected Hamiltonian shape (4, {self.r.size}), got {tuple(hamiltonian.shape)}")
        if not hamiltonian.requires_grad:
            hamiltonian.requires_grad_(True)

        orbitals = self.matrix_solver.solve(hamiltonian)
        densities = self.functional.densities_from_orbitals(orbitals)
        coup = self.functional.couplings(densities["rho_v_n"] + densities["rho_v_p"])
        rebuilt_fields = self.functional.reconstruct_fields(densities, coup)
        rebuilt_state = {**densities, **rebuilt_fields}
        potentials = self.functional.potentials(**rebuilt_state)
        reconstructed = torch.stack([potentials[name] for name in ("vps_n", "vms_n", "vps_p", "vms_p")])
        action_terms = self.functional.off_shell_rmf_action(orbitals, rebuilt_state)
        energy_total = action_terms["rmf_action"]
        energy_per_a = energy_total / self.functional.a
        # PRL varies the total functional E[H], not E/A.  The tensor is the
        # derivative with respect to the internal fm^-1 Hamiltonian.
        grad_energy = torch.autograd.grad(energy_total, hamiltonian, retain_graph=True, create_graph=False)[0]
        obs = self.functional.observables(rebuilt_state)
        el_residuals = self.functional.euler_lagrange_residuals(orbitals, rebuilt_state)
        weak_field_residuals = self.functional.field_stationarity_residuals(rebuilt_state, coup)
        interior = slice(3, -3)
        reduced_energy = self.functional.exact_kinetic_energy(orbitals) + self.functional.direct_energy_terms(rebuilt_state)["E_direct"]
        reconstruction_rmse = torch.sqrt(torch.mean((hamiltonian.detach() - reconstructed.detach()).square()))
        local_bilinear_errors = []
        metric_scale = torch.sqrt(hamiltonian.new_tensor(self.functional.h))
        for index in range(orbitals["G"].shape[0]):
            offset = 0 if float(orbitals["species_sign"][index]) > 0.0 else 2
            g = orbitals["G"][index, 1:-1]
            f = orbitals["F"][index, 1:-1]
            coefficients = metric_scale * torch.cat([g, f])
            block_size = g.numel()
            matrix_value = (
                coefficients[:block_size].square() @ hamiltonian[offset, 1:-1]
                + coefficients[block_size:].square() @ hamiltonian[offset + 1, 1:-1]
            )
            density_value = self.functional.h * torch.sum(
                g.square() * hamiltonian[offset, 1:-1]
                + f.square() * hamiltonian[offset + 1, 1:-1]
            )
            local_bilinear_errors.append(torch.abs(matrix_value - density_value))
        diagnostics = {
            "e_per_a_no_com": float(energy_per_a.detach().cpu()),
            "rms_n_no_com": float(obs["rms_n_no_com"].detach().cpu()),
            "rms_p_no_com": float(obs["rms_p_no_com"].detach().cpu()),
            "rms_matter_no_com": float(obs["rms_matter_no_com"].detach().cpu()),
            "charge_radius_no_com": float(obs["charge_radius_no_com"].detach().cpu()),
            "total_energy_gradient_h_fm_norm": float(torch.linalg.norm(grad_energy).detach().cpu()),
            "gradient_norm": float(torch.linalg.norm(grad_energy).detach().cpu()),
            "reconstruction_rmse": float(reconstruction_rmse.cpu()),
            "hermiticity_error": self.matrix_solver.hermiticity_error(hamiltonian.detach()),
            "complete_action_per_a": float((action_terms["rmf_action"] / self.functional.a).detach().cpu()),
            "reduced_energy_per_a": float((reduced_energy / self.functional.a).detach().cpu()),
            "action_reduction_error_mev": float((action_terms["rmf_action"] - reduced_energy).detach().cpu()),
            "max_orbital_norm_residual": float(action_terms["max_orbital_norm_residual"].detach().cpu()),
            "max_spectral_residual": float(orbitals["spectral_residual"].max().detach().cpu()),
            "n_number_error": float(torch.abs(obs["n_number"] - self.functional.n).detach().cpu()),
            "z_number_error": float(torch.abs(obs["z_number"] - self.functional.z).detach().cpu()),
            "max_occupied_node_mismatch": float(torch.max(
                torch.abs(orbitals["node_count"] - orbitals["expected_node_count"])
                * (orbitals["occupancy"] > 0.0).to(torch.int64)
            ).detach().cpu()),
            "uniform_metric_max_error": float(
                torch.max(torch.abs(self.matrix_solver.metric_weights - self.functional.h)).detach().cpu()
            ),
            "max_local_bilinear_error": float(torch.stack(local_bilinear_errors).max().detach().cpu()),
        }
        for name, residual in weak_field_residuals.items():
            diagnostics[f"weak_el_{name}_rms"] = float(torch.sqrt(torch.mean(residual.square())).detach().cpu())
        for name, residual in el_residuals.items():
            values = residual[..., interior]
            # This is an independent collocation truncation diagnostic.  The
            # actual field constraint is weak_el_* from the discrete action.
            diagnostics[f"collocation_el_{name}_rms"] = float(
                torch.sqrt(torch.mean(values.square())).detach().cpu()
            )
        scalar_self_energy = self.functional.couplings(densities["rho_v_n"] + densities["rho_v_p"])["gsig"] * rebuilt_fields["sigma"]
        fields = {
            **rebuilt_fields,
            "sigma_rearr": potentials["sigma_rearr"],
            "component_S": potentials["scalar_self_energy"],
            "component_V0": potentials["vector_isoscalar"],
            "component_V3": potentials["vector_isovector"],
            "component_VC": potentials["vector_coulomb"],
            "scalar_self_energy_mev": scalar_self_energy * self.functional.params.hbar_c,
            "effective_mass_n_mev": self.functional.params.mass_n + scalar_self_energy * self.functional.params.hbar_c,
            "effective_mass_p_mev": self.functional.params.mass_p + scalar_self_energy * self.functional.params.hbar_c,
            "vector_isoscalar_mev": potentials["vector_isoscalar"] * self.functional.params.hbar_c,
            "vector_isovector_mev": potentials["vector_isovector"] * self.functional.params.hbar_c,
            "vector_coulomb_mev": potentials["vector_coulomb"] * self.functional.params.hbar_c,
        }
        return TorchBackendResult(
            energy_per_a_no_com=float(energy_per_a.detach().cpu()),
            energy_total_no_com=float(energy_total.detach().cpu()),
            energy_per_a_tensor=energy_per_a,
            hamiltonian=hamiltonian,
            reconstructed_hamiltonian=reconstructed,
            grad_energy_h=grad_energy,
            orbitals=orbitals,
            densities=densities,
            fields=fields,
            action_terms=action_terms,
            diagnostics=diagnostics,
        )

    def gradient_check(self, hamiltonian: torch.Tensor, *, epsilon: float = 1.0e-3, seed: int = 17) -> dict[str, float]:
        generator = torch.Generator(device=hamiltonian.device)
        generator.manual_seed(seed)
        direction = torch.randn(hamiltonian.shape, dtype=hamiltonian.dtype, device=hamiltonian.device, generator=generator)
        direction = direction / direction.norm().clamp(min=1.0e-14)
        base = hamiltonian.detach()
        plus = self.evaluate_energy_only(base + epsilon * direction)
        minus = self.evaluate_energy_only(base - epsilon * direction)
        fd = (plus - minus) / (2.0 * epsilon)
        checked = base.clone().requires_grad_(True)
        result = self.evaluate_tensor(checked)
        ad = torch.sum(result.grad_energy_h * direction).detach()
        abs_error = torch.abs(ad - fd)
        rel_error = abs_error / torch.abs(fd).clamp(min=1.0e-12)
        return {
            "finite_difference": float(fd.cpu()),
            "autograd": float(ad.cpu()),
            "abs_error": float(abs_error.cpu()),
            "rel_error": float(rel_error.cpu()),
            "epsilon": epsilon,
        }

    def evaluate_energy_only(self, hamiltonian: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            h = hamiltonian.detach().clone().requires_grad_(True)
            orbitals = self.matrix_solver.solve(h)
            densities = self.functional.densities_from_orbitals(orbitals)
            coup = self.functional.couplings(densities["rho_v_n"] + densities["rho_v_p"])
            fields = self.functional.reconstruct_fields(densities, coup)
            state = {**densities, **fields}
            return self.functional.off_shell_rmf_action(orbitals, state)["rmf_action"]
