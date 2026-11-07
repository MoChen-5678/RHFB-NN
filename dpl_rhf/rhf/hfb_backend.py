"""Spherical no-refresh relativistic Hartree-Fock-Bogoliubov (PKA1 + D1S).

The quasiparticle vacuum is parameterized in its canonical (Bloch-Messiah)
form.  For each (species, kappa) block the network supplies orthonormal
canonical radial factors Q in the complete fixed Dirac grid space, and one
logit t_n per canonical state gives v_n^2 = sigmoid(t_n), u_n v_n =
1/(2 cosh(t_n/2)).  Then

    rho_kappa   = (2j+1) sum_n o_n |n><n|,      o_n = v_n^2,
    kappa_kappa = (2j+1) sum_n p_n |n><n|,      p_n = u_n v_n,

with the spherical equal-filling rule of the reference RHFB program for an
odd species (blocked canonical level b):  o_b = (1 + (2j-1) v_b^2)/(2j+1) and
p_b = (2j-1)/(2j+1) u_b v_b.  Particle numbers are exact at every step: a
common logit shift per species is solved by bisection and attached with its
implicit derivative.  No Hamiltonian, HFB, or BCS diagonalization is used;
the Dirac sea is excluded by a positive-energy projector built with matrix
products only (``dpl_rhf.training.sea_projector``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.models.density_matrix_net import DensityBlockSpec
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend, DensityMatrixRHFResult
from dpl_rhf.rhf.initialization import independent_rhf_orbital_data
from dpl_rhf.rhf.pairing import D1S_PAIRING, GognyD1SPairingForce, SphericalD1SPairing
from dpl_rhf.rhf.parameters import RHFParameters
from dpl_rhf.training.sea_projector import positive_energy_projector


@dataclass
class OccupationState:
    occupations: list[torch.Tensor]
    pair_amplitudes: list[torch.Tensor]
    d_occupations: list[torch.Tensor]
    d_pairs: list[torch.Tensor]
    v_squared: list[torch.Tensor]
    shifts: dict[str, float]


@dataclass
class SphericalRHFBResult:
    rhf: DensityMatrixRHFResult
    energy_total_tensor: torch.Tensor
    pairing_energy_tensors: dict[str, torch.Tensor]
    stationarity_square_tensor: torch.Tensor
    orbital_residual_rms: float
    occupation_residual_rms: float
    chemical_potentials: dict[str, float]
    particle_numbers: dict[str, float]
    average_gaps: dict[str, float]
    canonical: list[dict] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)
    delta_diagonals: list[torch.Tensor] = field(default_factory=list)
    pair_amplitudes: list[torch.Tensor] = field(default_factory=list)

    @property
    def energy_total(self) -> float:
        return float(self.energy_total_tensor.detach().cpu())


class SphericalRHFBBackend(DensityMatrixRHFBackend):
    """Canonical-basis spherical RHFB on top of the strict PKA1 RHF backend."""

    def __init__(
        self,
        case: NucleusCase,
        *,
        device: torch.device,
        pairing: bool = True,
        extra_capacity: int = 20,
        kappa_limits: tuple[int, int] = (5, 4),
        blocking: dict[str, int | None] | None = None,
        derivative_order: int = 7,
        params: RHFParameters | None = None,
        force: GognyD1SPairingForce = D1S_PAIRING,
        capacities: dict[str, int] | None = None,
        min_states_per_kappa: int = 0,
        pair_sign_init: str = "phase",
    ):
        if pair_sign_init not in {"phase", "ones"}:
            raise ValueError("pair_sign_init must be 'phase' or 'ones'")
        self.extra_capacity = int(extra_capacity)
        self.min_states_per_kappa = int(min_states_per_kappa)
        self.pair_sign_init = pair_sign_init
        self.capacities = dict(capacities) if capacities else None
        self.kappa_limits = tuple(int(v) for v in kappa_limits)
        self.blocking = {"n": None, "p": None, **(blocking or {})}
        for species, number in (("n", case.n), ("p", case.z)):
            blocked = self.blocking[species]
            if (number % 2 == 1) != (blocked is not None):
                raise ValueError(
                    f"{species}: odd particle number requires exactly one blocked kappa "
                    "(and even numbers none)"
                )
        super().__init__(case, device=device, derivative_order=derivative_order,
                         params=params)
        self.pairing_enabled = bool(pairing)
        self.pairing = (
            SphericalD1SPairing(self.functional.r, [s.kappa for s in self.block_specs],
                                force=force)
            if pairing else None
        )
        self.shifts = {"n": -self.params.mass_n, "p": -self.params.mass_p}
        self.sign_iterations = 0
        # Sign of u_n v_n for time-reversed pairs.  The energy is not even in
        # these signs once several canonical levels pair, so a sign is flipped
        # (with the same one-step lag as the no-sea projector) whenever
        # p_n Delta_nn > 0, i.e. when flipping lowers the energy to first order.
        # The time-reversal kernel carries the phase (-1)^((|k1|-1)+(|k2|-1));
        # starting from u v ~ (-1)^(|k|-1) cancels it, i.e. all pairs attract
        # coherently (the Fortran reference solution has this pattern).  Equal
        # signs instead make even- and odd-|k| blocks cancel, which weakens the
        # pairing field and lets the optimizer fall into the unpaired minimum.
        self.pair_signs = [
            torch.full((len(s.principals),),
                       float((-1) ** (abs(s.kappa) - 1)) if pair_sign_init == "phase" else 1.0,
                       dtype=torch.float64, device=self.device)
            for s in self.block_specs]
        self.pair_sign_flips = 0

    # ------------------------------------------------------------------ blocks
    def _build_full_space_blocks(self) -> list[DensityBlockSpec]:
        data, metadata = independent_rhf_orbital_data(
            self.case.z, self.case.n, self.r, self.params,
            extra_capacity=self.extra_capacity, kappa_limits=self.kappa_limits,
            capacities=self.capacities, min_states_per_kappa=self.min_states_per_kappa,
        )
        data = {key: torch.as_tensor(value, dtype=torch.float64, device=self.device)
                for key, value in data.items()}
        self.orbital_data, self.orbital_metadata = data, metadata
        groups: dict[tuple[float, int], list[int]] = {}
        for index, (species, kappa) in enumerate(zip(data["species_sign"], data["kappa"])):
            groups.setdefault((float(species), int(kappa)), []).append(index)
        metric_scale = float(self.functional.h) ** 0.5
        epsilon_initial = -0.5 * 41.0 * self.case.a ** (-1.0 / 3.0)
        specs: list[DensityBlockSpec] = []
        self.blocked_columns: list[int | None] = []
        with torch.no_grad():
            for (species, kappa), indices in sorted(groups.items()):
                ordered = sorted(indices, key=lambda i: int(data["principal"][i]))
                initial = torch.stack(
                    [metric_scale * torch.cat([data["G"][i, 1:-1], data["F"][i, 1:-1]])
                     for i in ordered], dim=1)
                initial, _ = torch.linalg.qr(initial, mode="reduced")
                name = "n" if species > 0 else "p"
                specs.append(DensityBlockSpec(
                    species_sign=species, kappa=kappa,
                    degeneracy=float(data["degeneracy"][ordered[0]]),
                    principals=tuple(int(data["principal"][i]) for i in ordered),
                    occupations=data["occupancy"][ordered].clone(),
                    initial_factors=initial,
                    free_hamiltonian=self.dirac_kinetic.matrix_fm_inverse(species, kappa),
                    tail_energies_mev=torch.full((len(ordered),), epsilon_initial,
                                                 dtype=torch.float64, device=self.device),
                ))
                # Equal filling blocks the lowest canonical level of the kappa block,
                # as the reference program blocks the n=1 quasiparticle of that block.
                self.blocked_columns.append(0 if self.blocking[name] == kappa else None)
        for name in ("n", "p"):
            if self.blocking[name] is not None and not any(
                column is not None and (s.species_sign > 0) == (name == "n")
                for s, column in zip(specs, self.blocked_columns)
            ):
                raise ValueError(f"{name}: blocked kappa {self.blocking[name]} is not in the space")
        return specs

    def initial_logits(self) -> list[torch.Tensor]:
        logits = []
        for spec in self.block_specs:
            s = spec.occupations.clamp(0.05, 0.95)
            logits.append(torch.log(s / (1.0 - s)))
        return logits

    # ------------------------------------------------------------- occupations
    def _occupation_terms(self, t: torch.Tensor, index: int):
        spec = self.block_specs[index]
        s = torch.sigmoid(t)
        uv = 0.5 / torch.cosh(0.5 * t)
        scale = torch.ones_like(t)
        offset = torch.zeros_like(t)
        column = self.blocked_columns[index]
        if column is not None:
            mu = spec.degeneracy
            scale = scale.clone()
            offset = offset.clone()
            scale[column] = (mu - 2.0) / mu
            offset[column] = 1.0 / mu
        occupation = offset + scale * s
        d_occupation = scale * uv.square()
        sign = self.pair_signs[index] if hasattr(self, "pair_signs") else 1.0
        pair = sign * scale * uv
        d_pair = -0.5 * sign * scale * torch.tanh(0.5 * t) * uv
        return occupation, pair, d_occupation, d_pair, s

    def occupation_state(self, logits: list[torch.Tensor]) -> OccupationState:
        count = len(self.block_specs)
        occupations: list = [None] * count
        pairs: list = [None] * count
        d_occ: list = [None] * count
        d_pair: list = [None] * count
        v2: list = [None] * count
        shifts = {}
        for name, target in (("n", self.case.n), ("p", self.case.z)):
            indices = [i for i, spec in enumerate(self.block_specs)
                       if (spec.species_sign > 0) == (name == "n")]

            def number(shift, detach=True):
                total = 0.0
                for i in indices:
                    t = (logits[i].detach() if detach else logits[i]) + shift
                    total = total + self.block_specs[i].degeneracy * self._occupation_terms(t, i)[0].sum()
                return total

            # The Fermi-shift bisection only involves a few dozen numbers per
            # species; doing it on the host avoids hundreds of tiny device
            # kernels and synchronizations per epoch (same algorithm and
            # tolerance; the Newton step below still carries the gradient).
            t_host, weight_host, scale_host, offset_host = [], [], [], []
            for i in indices:
                values = logits[i].detach().cpu().numpy()
                mu = self.block_specs[i].degeneracy
                scale_i = np.ones_like(values)
                offset_i = np.zeros_like(values)
                column = self.blocked_columns[i]
                if column is not None:
                    scale_i[column] = (mu - 2.0) / mu
                    offset_i[column] = 1.0 / mu
                t_host.append(values); weight_host.append(np.full_like(values, mu))
                scale_host.append(scale_i); offset_host.append(offset_i)
            t_host = np.concatenate(t_host); weight_host = np.concatenate(weight_host)
            scale_host = np.concatenate(scale_host); offset_host = np.concatenate(offset_host)

            def number_host(shift):
                occupied = offset_host + scale_host / (1.0 + np.exp(-(t_host + shift)))
                return float(np.sum(weight_host * occupied))

            low, high = -80.0, 80.0
            if not number_host(low) < target < number_host(high):
                raise ValueError(f"{name}: canonical space cannot hold {target} particles")
            for _ in range(200):
                middle = 0.5 * (low + high)
                if number_host(middle) < target:
                    low = middle
                else:
                    high = middle
                if high - low < 1e-14:
                    break
            shift0 = 0.5 * (low + high)
            half = 0.5 * (t_host + shift0)
            slope = torch.tensor(float(np.sum(weight_host * scale_host * (0.5 / np.cosh(half)) ** 2)),
                                 dtype=torch.float64, device=self.device)
            # One Newton step attaches the implicit derivative d(shift)/d(logits).
            shift = shift0 + (target - number(shift0, detach=False)) / slope.clamp_min(1e-300)
            shifts[name] = float(shift0)
            for i in indices:
                o, p, do, dp, s = self._occupation_terms(logits[i] + shift, i)
                occupations[i], pairs[i], d_occ[i], d_pair[i], v2[i] = o, p, do, dp, s
        return OccupationState(occupations, pairs, d_occ, d_pair, v2, shifts)

    def update_pair_signs(self, result: "SphericalRHFBResult") -> int:
        flips = 0
        for i, (pair, delta) in enumerate(zip(result.pair_amplitudes, result.delta_diagonals)):
            wrong = (pair * delta > 0.0) & (pair.abs() > 1e-8)
            if bool(wrong.any()):
                self.pair_signs[i] = torch.where(wrong, -self.pair_signs[i], self.pair_signs[i])
                flips += int(wrong.sum())
        self.pair_sign_flips += flips
        return flips

    # --------------------------------------------------------------- no sea
    def positive_projectors(self, hamiltonians: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        projectors = []
        for spec in self.block_specs:
            name = f"{'n' if spec.species_sign > 0 else 'p'}_kappa_{spec.kappa:+d}"
            species = "n" if spec.species_sign > 0 else "p"
            result = positive_energy_projector(hamiltonians[name], self.shifts[species])
            self.sign_iterations += result.iterations
            projectors.append(result.projector)
        return projectors

    def kinetic_hamiltonians(self) -> dict[str, torch.Tensor]:
        """Free Dirac blocks, used only to build the first no-sea projector."""
        return {
            f"{'n' if s.species_sign > 0 else 'p'}_kappa_{s.kappa:+d}":
                s.free_hamiltonian * self.params.hbar_c
            for s in self.block_specs
        }

    # ------------------------------------------------------------- evaluate
    def evaluate_hfb(
        self,
        network_blocks: list[dict[str, torch.Tensor]],
        logits: list[torch.Tensor],
        *,
        create_graph: bool = False,
    ) -> SphericalRHFBResult:
        state = self.occupation_state(logits)
        blocks = []
        for block, occupation in zip(network_blocks, state.occupations):
            updated = dict(block)
            updated["occupations"] = occupation
            blocks.append(updated)
        rhf = super().evaluate(blocks, create_graph=create_graph)

        h = self.functional.h
        pairing_energy = {"n": torch.zeros((), dtype=torch.float64, device=self.device),
                          "p": torch.zeros((), dtype=torch.float64, device=self.device)}
        deltas: list[torch.Tensor | None] = [None] * len(blocks)
        if self.pairing is not None:
            projectors = []
            per_species: dict[str, list] = {"n": [], "p": []}
            for block, pair in zip(blocks, state.pair_amplitudes):
                q = block["factors"]
                projector = (q * pair[None, :]) @ q.T
                projectors.append(projector)
                mu = float(block["degeneracy"])
                split = q.shape[0] // 2
                matrix = mu * projector / h
                tensor = torch.stack([
                    torch.stack([matrix[:split, :split], matrix[:split, split:]]),
                    torch.stack([matrix[split:, :split], matrix[split:, split:]]),
                ])
                name = "n" if float(block["species_sign"]) > 0 else "p"
                per_species[name].append((int(block["kappa"]), tensor))
            for name in ("n", "p"):
                pairing_energy[name] = self.pairing.energy(per_species[name])
            total_pairing = pairing_energy["n"] + pairing_energy["p"]
            if total_pairing.requires_grad:
                gradients = torch.autograd.grad(
                    total_pairing, projectors, retain_graph=True,
                    create_graph=create_graph, allow_unused=True)
            else:
                gradients = [None] * len(projectors)
            for i, (block, gradient) in enumerate(zip(blocks, gradients)):
                if gradient is not None:
                    deltas[i] = 0.5 * (gradient + gradient.T) / float(block["degeneracy"])

        orbital_parts, occupation_parts = [], []
        chemical_potentials, numbers, gaps = {}, {}, {}
        canonical = []
        diag_terms = []
        for i, block in enumerate(blocks):
            name = self._block_name(block)
            q = block["factors"]
            hq = rhf.hamiltonians[name] @ q
            action = hq * state.occupations[i][None, :]
            h_diagonal = torch.sum(q * hq, dim=0)
            if deltas[i] is not None:
                dq = deltas[i] @ q
                action = action + dq * state.pair_amplitudes[i][None, :]
                delta_diagonal = torch.sum(q * dq, dim=0)
            else:
                delta_diagonal = torch.zeros_like(h_diagonal)
            tangent = 0.5 * (q.T @ action + action.T @ q)
            orbital_parts.append((action - q @ tangent).reshape(-1))
            diag_terms.append((h_diagonal, delta_diagonal))

        for species in ("n", "p"):
            indices = [i for i, b in enumerate(blocks)
                       if (float(b["species_sign"]) > 0) == (species == "n")]
            numerator = denominator = 0.0
            for i in indices:
                mu = float(blocks[i]["degeneracy"])
                h_d, delta_d = diag_terms[i]
                a = h_d * state.d_occupations[i] + delta_d * state.d_pairs[i]
                b = state.d_occupations[i]
                numerator = numerator + mu * torch.sum(a * b)
                denominator = denominator + mu * torch.sum(b * b)
            lam = (numerator / denominator.clamp_min(1e-300)).detach()
            chemical_potentials[species] = float(lam.cpu())
            number = 0.0
            weighted_gap = weight = 0.0
            for i in indices:
                mu = float(blocks[i]["degeneracy"])
                h_d, delta_d = diag_terms[i]
                residual = (h_d - lam) * state.d_occupations[i] + delta_d * state.d_pairs[i]
                occupation_parts.append(residual)
                number = number + mu * float(state.occupations[i].sum().detach().cpu())
                pair = state.pair_amplitudes[i].detach()
                weighted_gap = weighted_gap + mu * float(torch.sum(pair * delta_d.detach()).cpu())
                weight = weight + mu * float(pair.sum().cpu())
                for column in range(h_d.numel()):
                    e = float(h_d[column].detach().cpu())
                    d = float(delta_d[column].detach().cpu())
                    canonical.append({
                        "species": species, "kappa": int(blocks[i]["kappa"]),
                        "principal": int(blocks[i]["principals"][column]),
                        "degeneracy": mu,
                        "blocked": self.blocked_columns[i] == column,
                        "v2": float(state.v_squared[i][column].detach().cpu()),
                        "occupation": float(state.occupations[i][column].detach().cpu()),
                        "pair_amplitude": float(state.pair_amplitudes[i][column].detach().cpu()),
                        "h_nn_mev": e, "delta_nn_mev": d,
                        "quasiparticle_energy_mev": ((e - float(lam)) ** 2 + d * d) ** 0.5,
                    })
            numbers[species] = number
            gaps[species] = weighted_gap / weight if weight > 1e-14 else 0.0

        orbital_vector = torch.cat(orbital_parts)
        occupation_vector = torch.cat(occupation_parts)
        stationarity_square = (torch.mean(orbital_vector.square())
                               + torch.mean(occupation_vector.square()))
        energy_total = rhf.energy_total_tensor + pairing_energy["n"] + pairing_energy["p"]
        diagnostics = dict(rhf.diagnostics)
        diagnostics.update({
            "n_number_error": abs(numbers["n"] - self.case.n),
            "z_number_error": abs(numbers["p"] - self.case.z),
            "pairing_energy_n_mev": float(pairing_energy["n"].detach().cpu()),
            "pairing_energy_p_mev": float(pairing_energy["p"].detach().cpu()),
            "fermi_logit_shift_n": state.shifts["n"],
            "fermi_logit_shift_p": state.shifts["p"],
            "no_sea_sign_iterations": self.sign_iterations,
            "pair_sign_flips": self.pair_sign_flips,
        })
        return SphericalRHFBResult(
            rhf=rhf,
            energy_total_tensor=energy_total,
            pairing_energy_tensors=pairing_energy,
            stationarity_square_tensor=stationarity_square,
            orbital_residual_rms=float(torch.sqrt(torch.mean(orbital_vector.square())).detach().cpu()),
            occupation_residual_rms=float(torch.sqrt(torch.mean(occupation_vector.square())).detach().cpu()),
            chemical_potentials=chemical_potentials,
            particle_numbers=numbers,
            average_gaps=gaps,
            canonical=canonical,
            diagnostics=diagnostics,
            delta_diagonals=[d.detach() for _, d in diag_terms],
            pair_amplitudes=[p.detach() for p in state.pair_amplitudes],
        )
