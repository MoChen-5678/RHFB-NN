from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from fieldspace_model import FieldSpaceRMFNet
from rmf_functional import PKDDParameters, PKDDRMFFunctional
from variational_rmf import (
    compare_single_particle,
    evaluate_dpl_with_fortran,
    independent_orbital_data,
    run_scf,
)


def woods_saxon_local_stack(z: int, n: int, r: np.ndarray) -> np.ndarray:
    """Textbook Eqs. (3.75)-(3.77), in the Core binding-energy convention."""
    p = PKDDParameters()
    a = z + n
    v0, a0 = -71.28, 0.4616
    table = {
        "n": {"tau": 1.0, "av": 11.1175, "rplus": 1.2334, "rminus": 1.1443, "aplus": 0.6150, "aminus": 0.6476},
        "p": {"tau": -1.0, "av": 8.9698, "rplus": 1.2496, "rminus": 1.1400, "aplus": 0.6124, "aminus": 0.6469},
    }
    charge_radius = table["p"]["rplus"] * a ** (1.0 / 3.0)
    alpha = 1.0 / p.alpha_inv
    coul = np.where(
        r < charge_radius,
        alpha * z * (3.0 / (2.0 * charge_radius) - r * r / (2.0 * charge_radius**3)),
        alpha * z / r,
    )
    channels = []
    for species in ("n", "p"):
        q = table[species]
        factor = 1.0 - a0 * (n - z) * q["tau"] / a
        plus = v0 * factor / (1.0 + np.exp((r - q["rplus"] * a ** (1.0 / 3.0)) / q["aplus"]))
        minus = -v0 * q["av"] * factor / (1.0 + np.exp((r - q["rminus"] * a ** (1.0 / 3.0)) / q["aminus"]))
        if species == "p":
            plus = plus + p.hbar_c * coul
            minus = minus + p.hbar_c * coul
        mass = p.mass_n if species == "n" else p.mass_p
        channels.extend([plus / p.hbar_c, minus / p.hbar_c - 2.0 * mass / p.hbar_c])
    return np.stack(channels)


def _first_derivative_coefficients(offsets: list[int]) -> np.ndarray:
    matrix = np.array([[float(offset) ** power for offset in offsets] for power in range(len(offsets))])
    rhs = np.zeros(len(offsets))
    rhs[1] = 1.0
    return np.linalg.solve(matrix, rhs)


def noncentral_derivative_matrix(r: torch.Tensor, order: int = 1) -> torch.Tensor:
    """One-sided radial derivative on the interior mesh.

    The Dirac Hamiltonian uses this matrix in one off-diagonal block and its
    transpose in the other, so any real stencil here still gives an exactly
    Hermitian discrete Hamiltonian.  The last row uses the known outer Dirichlet
    boundary value f(Rmax)=0 instead of wrapping or central differencing.
    """
    x = r[1:-1]
    size = x.numel()
    h = float(r[2] - r[1])
    derivative = torch.zeros((size, size), dtype=r.dtype, device=r.device)
    if order == 1:
        derivative.diagonal().fill_(-1.0 / h)
        derivative.diagonal(offset=1).fill_(1.0 / h)
        return derivative
    if order == 2:
        if size < 3:
            raise ValueError("second-order noncentral derivative needs at least three interior points")
        rows = torch.arange(size - 2, device=r.device)
        derivative[rows, rows] = -3.0 / (2.0 * h)
        derivative[rows, rows + 1] = 4.0 / (2.0 * h)
        derivative[rows, rows + 2] = -1.0 / (2.0 * h)
        derivative[size - 2, size - 2] = -1.0 / h
        derivative[size - 2, size - 1] = 1.0 / h
        derivative[size - 1, size - 1] = -1.0 / h
        return derivative
    if order in {4, 5, 6, 7}:
        points = 5 if order in {4, 5} else 7
        if size < points:
            raise ValueError(f"{points}-point asymmetric derivative needs at least {points} interior points")
        for row in range(size):
            if row + points <= size:
                columns = list(range(row, row + points))
            else:
                columns = list(range(row - points + 1, row + 1))
            offsets = [column - row for column in columns]
            coefficients = _first_derivative_coefficients(offsets)
            derivative[row, columns] = torch.as_tensor(coefficients / h, dtype=r.dtype, device=r.device)
        return derivative
    raise ValueError(f"unsupported derivative order {order}; use 1, 2, 4, 5, 6, or 7")


class DifferentiableDiracMatrix:
    """Hermitian radial MatrixBackend with textbook no-sea occupied branches."""

    def __init__(self, r: torch.Tensor, orbital_data: dict, initial_stack: torch.Tensor, derivative_order: int = 1):
        self.r = r
        self.x = r[1:-1]
        # A non-central derivative removes the central-difference fermion
        # doubler. Pairing D+ with D+^T keeps the complete Hamiltonian exactly
        # Hermitian, which is the strict constraint needed by the variational
        # solve.
        self.derivative_order = derivative_order
        self.derivative_plus = noncentral_derivative_matrix(r, derivative_order)
        self.orbital_data = {
            key: torch.as_tensor(value, dtype=r.dtype, device=r.device)
            for key, value in orbital_data.items()
            if key in {"G", "F", "kappa", "species_sign", "occupancy", "degeneracy", "principal"}
        }
        self.selection = self._select_physical_branches(initial_stack)

    def matrix(self, stack: torch.Tensor, species_sign: float, kappa: int) -> torch.Tensor:
        offset = 0 if species_sign > 0 else 2
        vps = stack[offset, 1:-1]
        vms = stack[offset + 1, 1:-1]
        angular = torch.diag(torch.full_like(self.x, float(kappa)) / self.x)
        lower_operator = self.derivative_plus + angular
        upper = torch.cat([torch.diag(vps), lower_operator.T], dim=1)
        lower = torch.cat([lower_operator, torch.diag(vms)], dim=1)
        return torch.cat([upper, lower], dim=0)

    def _select_physical_branches(self, initial_stack: torch.Tensor) -> list[int]:
        selections = [-1] * len(self.orbital_data["kappa"])
        groups: dict[tuple[float, int], list[int]] = {}
        for i, (species, kappa) in enumerate(zip(self.orbital_data["species_sign"], self.orbital_data["kappa"])):
            groups.setdefault((float(species), int(kappa)), []).append(i)
        for (species, kappa), orbit_indices in groups.items():
            eigenvalues, eigenvectors = torch.linalg.eigh(self.matrix(initial_stack, species, kappa))
            upper_norm = eigenvectors[: self.x.numel()].square().sum(dim=0)
            candidates = torch.where(
                (eigenvalues * PKDDParameters().hbar_c > -200.0)
                & (eigenvalues * PKDDParameters().hbar_c < 50.0)
                & (upper_norm > 0.5)
            )[0]
            used = set()
            for orbit_index in sorted(orbit_indices, key=lambda i: int(self.orbital_data["principal"][i])):
                reference = torch.cat(
                    [self.orbital_data["G"][orbit_index, 1:-1], self.orbital_data["F"][orbit_index, 1:-1]]
                )
                reference = reference / reference.norm().clamp(min=1.0e-14)
                overlaps = torch.abs(eigenvectors[:, candidates].T @ reference)
                for candidate_position in torch.argsort(overlaps, descending=True):
                    index = int(candidates[candidate_position])
                    if index not in used:
                        selections[orbit_index] = index
                        used.add(index)
                        break
        if any(index < 0 for index in selections):
            raise RuntimeError("failed to identify every occupied Woods-Saxon branch")
        return selections

    def solve(self, stack: torch.Tensor) -> dict[str, torch.Tensor]:
        count = len(self.selection)
        G = torch.zeros((count, self.r.numel()), dtype=self.r.dtype, device=self.r.device)
        F = torch.zeros_like(G)
        energies = torch.zeros(count, dtype=self.r.dtype, device=self.r.device)
        groups: dict[tuple[float, int], list[int]] = {}
        for i, (species, kappa) in enumerate(zip(self.orbital_data["species_sign"], self.orbital_data["kappa"])):
            groups.setdefault((float(species), int(kappa)), []).append(i)
        for (species, kappa), orbit_indices in groups.items():
            eigenvalues, eigenvectors = torch.linalg.eigh(self.matrix(stack, species, kappa))
            for orbit_index in orbit_indices:
                vector = eigenvectors[:, self.selection[orbit_index]]
                G[orbit_index, 1:-1] = vector[: self.x.numel()]
                F[orbit_index, 1:-1] = vector[self.x.numel() :]
                energies[orbit_index] = eigenvalues[self.selection[orbit_index]]
        return {
            "G": G, "F": F, "epsilon": energies * PKDDParameters().hbar_c,
            **{key: self.orbital_data[key] for key in ("kappa", "species_sign", "occupancy", "degeneracy")},
        }

    def dirac_residual(self, stack: torch.Tensor, orbitals: dict[str, torch.Tensor]) -> torch.Tensor:
        residuals = []
        for i, (species, kappa) in enumerate(zip(orbitals["species_sign"], orbitals["kappa"])):
            vector = torch.cat([orbitals["G"][i, 1:-1], orbitals["F"][i, 1:-1]])
            eigenvalue = orbitals["epsilon"][i] / PKDDParameters().hbar_c
            residual = self.matrix(stack, float(species), int(kappa)) @ vector - eigenvalue * vector
            residuals.append(residual.square().mean() / vector.square().mean().clamp(min=1.0e-30))
        return torch.stack(residuals).mean()


def build_reconstructed_state(functional: PKDDRMFFunctional, orbitals: dict[str, torch.Tensor]):
    norm = torch.sum(functional.orbital_w[None, :] * (orbitals["G"].square() + orbitals["F"].square()), dim=1).sqrt()
    orbitals = dict(orbitals)
    orbitals["G"] = orbitals["G"] / norm[:, None]
    orbitals["F"] = orbitals["F"] / norm[:, None]
    densities = functional.densities_from_orbitals(orbitals)
    rho_b = densities["rho_v_n"] + densities["rho_v_p"]
    couplings = functional.couplings(rho_b)
    seed = {**densities, **{key: torch.zeros_like(rho_b) for key in ("sigma", "omega", "rho", "coul")}}
    fields = functional.reconstruct_fields_green(seed, couplings)
    state = {**densities, **fields}
    potentials = functional.potentials(**state)
    stack = torch.stack([potentials["vps_n"], potentials["vms_n"], potentials["vps_p"], potentials["vms_p"]])
    return orbitals, state, potentials, stack


def optimize(args: argparse.Namespace) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    r_np = np.concatenate(
        ([0.001], np.arange(args.mesh_step, args.r_max + 0.5 * args.mesh_step, args.mesh_step))
    )
    r = torch.as_tensor(r_np, device=device)
    initial_np = woods_saxon_local_stack(args.z, args.n, r_np)
    initial = torch.as_tensor(initial_np, device=device)
    orbital_data, metadata = independent_orbital_data(args.z, args.n, r_np)
    functional = PKDDRMFFunctional(r, args.z, args.n)
    matrix_solver = DifferentiableDiracMatrix(r, orbital_data, initial, args.derivative_order)
    network = FieldSpaceRMFNet(r_np, args.z, args.n, initial_np, args.hidden).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    scales = torch.tensor([0.15, 0.30, 0.15, 0.30], device=device)[:, None]

    def forward_objective():
        trial = network()
        occupied = matrix_solver.solve(trial)
        occupied, current_state, current_potentials, rebuilt = build_reconstructed_state(functional, occupied)
        kinetic_energy = functional.exact_kinetic_energy(occupied)
        direct_energy = functional.direct_energy_terms(current_state)["E_direct"]
        e_per_a = (kinetic_energy + direct_energy) / (args.z + args.n)
        mismatch = ((trial - rebuilt) / scales).square().mean()
        dirac_residual = matrix_solver.dirac_residual(rebuilt, occupied)
        norm = torch.sum(functional.orbital_w[None, :] * (occupied["G"].square() + occupied["F"].square()), dim=1)
        norm_residual = (norm - 1.0).square().mean()
        boundary_residual = (
            occupied["G"][:, [0, -1]].square().mean()
            + occupied["F"][:, [0, -1]].square().mean()
        )
        loss = (
            e_per_a / 50.0
            + args.lambda_reconstruct * mismatch
            + args.lambda_dirac * dirac_residual
            + args.lambda_norm * norm_residual
            + args.lambda_boundary * boundary_residual
        )
        constraints = {
            "potential_reconstruction": mismatch,
            "dirac_residual": dirac_residual,
            "normalization_residual": norm_residual,
            "boundary_residual": boundary_residual,
        }
        return loss, e_per_a, constraints, trial, occupied, current_state, current_potentials, rebuilt

    def record(epoch, phase, objective, energy_per_a, constraints, state, gradient_norm):
        nonlocal best
        value = float(objective.detach())
        if value < best["loss"]:
            best = {"loss": value, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}}
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs + args.lbfgs_steps:
            row = {
                "epoch": epoch, "phase": phase, "objective": value,
                "energy_per_a": float(energy_per_a.detach()),
                "reconstruction": float(constraints["potential_reconstruction"].detach()),
                "dirac_residual": float(constraints["dirac_residual"].detach()),
                "normalization_residual": float(constraints["normalization_residual"].detach()),
                "boundary_residual": float(constraints["boundary_residual"].detach()),
                "gradient_norm": float(gradient_norm),
                "rms_matter": float(functional.observables(state)["rms_matter_no_com"].detach()),
            }
            rows.append(row)
            print(
                f"{epoch:5d}/{args.epochs + args.lbfgs_steps} {phase:6s} L={value:.6e} "
                f"E/A={row['energy_per_a']:.6f} recon={row['reconstruction']:.6e} "
                f"dirac={row['dirac_residual']:.3e} "
                f"|g|={row['gradient_norm']:.3e} R={row['rms_matter']:.4f}"
            )

    for epoch in range(1, args.epochs + 1):
        objective, energy_per_a, constraints, _, _, state, _, _ = forward_objective()
        optimizer.zero_grad()
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
        record(epoch, "adam", objective, energy_per_a, constraints, state, gradient_norm)
        optimizer.step()

    if args.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            network.parameters(), lr=0.5, max_iter=1, history_size=20,
            tolerance_grad=1.0e-12, tolerance_change=1.0e-14, line_search_fn="strong_wolfe",
        )
        for step in range(1, args.lbfgs_steps + 1):
            cache = {}

            def closure():
                lbfgs.zero_grad()
                values = forward_objective()
                values[0].backward()
                cache["values"] = values
                return values[0]

            lbfgs.step(closure)
            objective, energy_per_a, constraints, _, _, state, _, _ = forward_objective()
            gradients = torch.autograd.grad(objective, tuple(network.parameters()), allow_unused=True)
            gradient_norm = torch.sqrt(sum(g.square().sum() for g in gradients if g is not None))
            record(args.epochs + step, "lbfgs", objective, energy_per_a, constraints, state, gradient_norm)

    network.load_state_dict(best["state"])
    _, energy_per_a, constraints, trial_stack, orbitals, state, potentials, reconstructed_stack = forward_objective()
    observables = functional.observables(state)
    np.savez(out / "dpl_local_stack.npz", r=r_np, stack=trial_stack.detach().cpu().numpy())
    np.savez(out / "reconstructed_local_stack.npz", r=r_np, stack=reconstructed_stack.detach().cpu().numpy())
    np.savez(out / "dpl_orbitals.npz", r=r_np, **{k: v.detach().cpu().numpy() for k, v in orbitals.items()}, **metadata)
    np.savez(out / "dpl_densities.npz", r=r_np, **{k: v.detach().cpu().numpy() for k, v in state.items()})
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "mode": "fieldspace-ai2dft-generalized-functional", "model": "PKDD", "z": args.z, "n": args.n,
        "adam_epochs": args.epochs, "lbfgs_steps": args.lbfgs_steps, "lambda_reconstruct": args.lambda_reconstruct,
        "lambda_dirac": args.lambda_dirac, "lambda_norm": args.lambda_norm, "lambda_boundary": args.lambda_boundary,
        "seed": args.seed,
        "mesh_step": args.mesh_step, "r_max": args.r_max, "derivative_order": args.derivative_order,
        "best_epoch": best["epoch"], "best_objective": best["loss"],
        "energy_per_a_no_com": float(energy_per_a.detach()),
        "physics_constraints": {key: float(value.detach()) for key, value in constraints.items()},
        "reconstruction_loss": float(constraints["potential_reconstruction"].detach()),
        "observables_no_com": {key: float(value.detach()) for key, value in observables.items()},
        "selected_eigen_indices": matrix_solver.selection,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"state_dict": network.state_dict(), "summary": summary}, out / "model.pt")
    if args.skip_compare:
        return {"fieldspace": summary}
    fixed_potential = evaluate_dpl_with_fortran(
        "PKDD", args.z, args.n, args.z + args.n, trial_stack.detach().cpu().numpy(), out
    )
    scf = run_scf("PKDD", args.z, args.n, args.z + args.n, out)
    single_particle = compare_single_particle(out)
    comparison = {
        "fieldspace": summary,
        "fortran_fixed_potential_diagnostic": fixed_potential,
        "scf": scf,
        "single_particle": single_particle,
    }
    (out / "compare_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="AI2DFT-style PKDD field-space variational solver")
    parser.add_argument("--z", type=int, default=8); parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-3); parser.add_argument("--lambda-reconstruct", type=float, required=True)
    parser.add_argument("--lambda-dirac", type=float, default=1.0e-3)
    parser.add_argument("--lambda-norm", type=float, default=1.0)
    parser.add_argument("--lambda-boundary", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=10.0); parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20240623)
    parser.add_argument("--mesh-step", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=20.0)
    parser.add_argument("--derivative-order", type=int, choices=[1, 2, 4, 5, 6, 7], default=7)
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--out", default="outputs/fieldspace_rmf")
    optimize(parser.parse_args())


if __name__ == "__main__":
    main()
