from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.density_matrix_rmf import DensityMatrixRMFBackend
from dpl_rhf.functionals.pkdd_action import canonical_rmf_model
from dpl_rhf.io.outputs import ensure_out, write_history, write_json
from dpl_rhf.models.density_matrix_net import DensityMatrixNet
from dpl_rhf import __release_name__, __version__
from dpl_rhf.training.inverse_hamiltonian import (
    inverse_hamiltonian_constraint,
    rmf_block_hamiltonians,
)


def _numpy_dict(values: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy() for name, value in values.items()}


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    delta = actual.detach() - reference.detach()
    return float(
        (torch.linalg.norm(delta) / torch.linalg.norm(reference.detach()).clamp_min(1.0e-14))
        .cpu()
    )


def train_density_matrix(args) -> dict:
    if args.backend != "torch-rmf":
        raise ValueError("density-matrix mode requires --backend torch-rmf")
    if args.density_refresh_interval != 0:
        raise ValueError("the no-refresh solver requires --density-refresh-interval 0")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    case = NucleusCase(
        model=canonical_rmf_model(args.model),
        z=args.z,
        n=args.n,
        a=args.a or args.z + args.n,
    )
    out = ensure_out(args.out)
    initialization_started = time.perf_counter()
    backend = DensityMatrixRMFBackend(
        case, device=device, derivative_order=args.derivative_order
    )
    network = DensityMatrixNet(
        torch.as_tensor(backend.r, dtype=torch.float64, device=device),
        backend.block_specs,
        hidden=args.hidden,
        depth=args.network_depth,
        radial_basis_size=args.radial_basis_size,
        correction_scale=args.density_correction_scale,
    ).to(device)
    if args.initial_checkpoint:
        checkpoint = torch.load(
            args.initial_checkpoint, map_location=device, weights_only=False
        )
        state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
        if state is None:
            raise ValueError("initial checkpoint has no density-matrix model state")
        network.load_state_dict(state)
    initialization_seconds = time.perf_counter() - initialization_started
    learning_rate = float(args.lr if args.lr is not None else 1.0e-3)
    def make_optimizer(lr: float):
        return torch.optim.Adam(network.parameters(), lr=lr)

    def make_scheduler(current_optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            current_optimizer,
            mode="min",
            factor=0.5,
            patience=args.scheduler_patience,
            cooldown=args.scheduler_cooldown,
            threshold=1.0e-3,
        )

    optimizer = make_optimizer(learning_rate)
    scheduler = make_scheduler(optimizer)
    history = []
    best = {"monitor": float("inf"), "epoch": 0, "state": None}
    commutator_reference = None
    effective_stationarity_weight = args.density_stationarity_weight
    inverse_linear_solves = 0

    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        result = backend.evaluate(network.density_blocks())
        inverse = inverse_hamiltonian_constraint(
            result.blocks,
            rmf_block_hamiltonians(backend, result),
            {
                "n": -backend.functional.params.mass_n / backend.functional.params.hbar_c,
                "p": -backend.functional.params.mass_p / backend.functional.params.hbar_c,
            },
            branch_floor=args.density_branch_floor,
        )
        inverse_linear_solves += inverse.solve_count
        parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
        stationarity_square = result.stationarity_residual_tensor.square()
        if effective_stationarity_weight is None:
            if args.density_energy_weight == 0.0:
                effective_stationarity_weight = 1.0
            else:
                energy_gradients = torch.autograd.grad(
                    result.energy_total_tensor, parameters, retain_graph=True, allow_unused=True
                )
                stationarity_gradients = torch.autograd.grad(
                    stationarity_square, parameters, retain_graph=True, allow_unused=True
                )
                energy_norm = torch.sqrt(sum(
                    torch.sum(gradient.square()) for gradient in energy_gradients if gradient is not None
                ))
                stationarity_norm = torch.sqrt(sum(
                    torch.sum(gradient.square()) for gradient in stationarity_gradients if gradient is not None
                ))
                effective_stationarity_weight = float(
                    (
                        args.density_energy_weight
                        * energy_norm
                        / stationarity_norm.clamp_min(1.0e-14)
                    ).detach().cpu()
                )
        loss = (
            args.density_energy_weight * result.energy_total_tensor
            + effective_stationarity_weight * stationarity_square
            + args.density_inverse_weight * inverse.loss
        )
        loss.backward()
        gradient_norm = float(
            torch.sqrt(
                sum(
                    torch.sum(parameter.grad.square())
                    for parameter in network.parameters()
                    if parameter.grad is not None
                )
            ).detach().cpu()
        )
        commutator = float(result.stationarity_residual_tensor.detach().cpu())
        if commutator_reference is None:
            commutator_reference = max(commutator, 1.0e-14)
        normalized_commutator = commutator / commutator_reference
        monitor = normalized_commutator
        scheduler.step(monitor)
        row = {
            "epoch": epoch,
            "energy_total_no_com_mev": result.energy_total_no_com,
            "energy_per_a_no_com_mev": result.energy_per_a_no_com,
            "projected_commutator_rms_mev": commutator,
            "normalized_commutator": normalized_commutator,
            "normalized_variational_gradient": monitor,
            "parameter_gradient_norm": gradient_norm,
            "inverse_tangent_residual": float(inverse.tangent_residual.detach().cpu()),
            "minimum_branch_score": float(inverse.minimum_branch_score.detach().cpu()),
            "mean_branch_score": float(inverse.mean_branch_score.detach().cpu()),
            "linear_solve_residual": float(inverse.linear_solve_residual.detach().cpu()),
            "stationarity_weight_mev_inverse": effective_stationarity_weight,
            "energy_weight": args.density_energy_weight,
            "n_number_error": result.diagnostics["n_number_error"],
            "z_number_error": result.diagnostics["z_number_error"],
            "max_orthogonality_error": result.diagnostics["max_orthogonality_error"],
            "max_idempotency_rms": result.diagnostics["max_idempotency_rms"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training_dirac_solves": backend.training_dirac_solves,
            "eigensolve_count": (
                backend.initialization_block_diagonalizations
                + backend.refresh_block_diagonalizations
            ),
            "refresh_count": backend.refresh_phases,
            "spectral_refresh_phases": backend.refresh_phases,
            "refreshed_after_epoch": False,
        }
        history.append(row)
        if monitor < best["monitor"]:
            best = {
                "monitor": monitor,
                "epoch": epoch,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                },
            }
        torch.nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
        optimizer.step()
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"{epoch:5d}/{args.epochs} E/A={result.energy_per_a_no_com:.8f} "
                f"comm={commutator:.3e} monitor={monitor:.3e} "
                f"grad={gradient_norm:.3e}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started

    if args.checkpoint_policy == "physics" and best["state"] is not None:
        network.load_state_dict(best["state"])
    final_result = backend.evaluate(network.density_blocks())
    final_hamiltonian = final_result.hamiltonian.detach()
    final_orbitals = None
    final_validation_seconds = 0.0
    density_relative_l2: dict[str, float] = {}
    max_density_relative_l2 = 0.0
    spectral_validation: dict[str, object] = {"enabled": False}
    if args.final_spectral_validation:
        validation_started = time.perf_counter()
        final_orbitals = backend.final_spectral_validation(final_hamiltonian)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        final_validation_seconds = time.perf_counter() - validation_started
        spectral_densities = backend.functional.densities_from_orbitals(final_orbitals)
        density_relative_l2 = {
            name: _relative_l2(final_result.densities[name], spectral_densities[name])
            for name in final_result.densities
        }
        max_density_relative_l2 = max(density_relative_l2.values())
        spectral_validation = {
            "enabled": True,
            "density_relative_l2": density_relative_l2,
            "max_density_relative_l2": max_density_relative_l2,
            "max_spectral_residual": float(
                final_orbitals["spectral_residual"].max().detach().cpu()
            ),
        }

    np.savez(
        out / "dpl_hamiltonian.npz",
        r=backend.r,
        channels=np.asarray(backend.functional_spec.channel_spec.names),
        stack=final_hamiltonian.cpu().numpy(),
    )
    np.savez(
        out / "physical_components.npz",
        r=backend.r,
        components=np.asarray(("S", "V0", "V3", "VC")),
        stack=np.stack(
            [
                final_result.fields[name].detach().cpu().numpy()
                for name in ("component_S", "component_V0", "component_V3", "component_VC")
            ]
        ),
        role=np.asarray("RMF fields reconstructed directly from the neural density matrix"),
    )
    np.savez(out / "density_profiles.npz", r=backend.r, **_numpy_dict(final_result.densities))
    np.savez(out / "density_diagnostics.npz", r=backend.r, **_numpy_dict(final_result.densities))
    np.savez(out / "field_diagnostics.npz", r=backend.r, **_numpy_dict(final_result.fields))
    np.savez(out / "action_diagnostics.npz", **_numpy_dict(final_result.action_terms))
    if final_orbitals is not None:
        orbital_data = {**_numpy_dict(final_orbitals), **backend.orbital_metadata}
        np.savez(out / "final_wavefunctions.npz", r=backend.r, **orbital_data)
        np.savez(out / "dpl_orbitals.npz", r=backend.r, **orbital_data)
    factor_archive: dict[str, np.ndarray] = {"r": backend.r}
    for index, block in enumerate(final_result.blocks):
        prefix = f"block_{index:02d}"
        factor_archive[f"{prefix}_species_sign"] = np.asarray(float(block["species_sign"]))
        factor_archive[f"{prefix}_kappa"] = np.asarray(int(block["kappa"]))
        factor_archive[f"{prefix}_degeneracy"] = np.asarray(float(block["degeneracy"]))
        factor_archive[f"{prefix}_principals"] = block["principals"].detach().cpu().numpy()
        factor_archive[f"{prefix}_occupations"] = block["occupations"].detach().cpu().numpy()
        factor_archive[f"{prefix}_coefficients"] = block["coefficients"].detach().cpu().numpy()
        factor_archive[f"{prefix}_factors"] = block["factors"].detach().cpu().numpy()
    np.savez(out / "density_matrix_factors.npz", **factor_archive)

    solver_counts = {
        "initialization_phases": 1,
        "initialization_block_diagonalizations": backend.initialization_block_diagonalizations,
        "training_dirac_solves": backend.training_dirac_solves,
        "sparse_refresh_phases": backend.refresh_phases,
        "refresh_block_diagonalizations": backend.refresh_block_diagonalizations,
        "inverse_hamiltonian_linear_solves": inverse_linear_solves,
        "final_validation_phases": backend.final_validation_dirac_solves,
        "initialization_seconds": initialization_seconds,
        "training_seconds": training_seconds,
        "seconds_per_training_epoch": training_seconds / max(args.epochs, 1),
        "final_validation_seconds": final_validation_seconds,
        "note": "Training contains zero eigensolvers and zero subspace refreshes; no-sea is imposed by differentiable inverse-Hamiltonian solves.",
    }
    formula_invariants = {
        "training_dirac_solves": backend.training_dirac_solves,
        "max_orthogonality_error": final_result.diagnostics["max_orthogonality_error"],
        "max_idempotency_rms": final_result.diagnostics["max_idempotency_rms"],
        "n_number_error": final_result.diagnostics["n_number_error"],
        "z_number_error": final_result.diagnostics["z_number_error"],
    }
    formula_invariants["passed"] = bool(
        formula_invariants["training_dirac_solves"] == 0
        and formula_invariants["max_orthogonality_error"] < 1.0e-10
        and formula_invariants["max_idempotency_rms"] < 1.0e-10
        and formula_invariants["n_number_error"] < 1.0e-10
        and formula_invariants["z_number_error"] < 1.0e-10
    )
    final_normalized_commutator = (
        float(final_result.stationarity_residual_tensor.detach().cpu())
        / max(float(commutator_reference), 1.0e-14)
    )
    internal_gate = bool(
        best["monitor"] <= args.density_stationarity_gate
        and final_normalized_commutator <= args.density_stationarity_gate
        and formula_invariants["passed"]
        and (
            not args.final_spectral_validation
            or max_density_relative_l2 <= args.density_spectral_gate
        )
    )
    summary = {
        "release_name": __release_name__,
        "version": __version__,
        "mode": "prl-density-matrix-rmf",
        "variation_mode": "density-matrix",
        "backend": "torch-rmf-density-matrix",
        "functional": args.functional,
        "functional_type": final_result.functional_type,
        "case": asdict(case),
        "epochs": args.epochs,
        "optimizer": "Adam",
        "activation": "silu",
        "hidden": args.hidden,
        "network_depth": args.network_depth,
        "radial_basis_size": args.radial_basis_size,
        "density_correction_scale": args.density_correction_scale,
        "trainable_parameters": sum(p.numel() for p in network.parameters()),
        "lr": learning_rate,
        "final_lr": optimizer.param_groups[0]["lr"],
        "derivative_order": args.derivative_order,
        "seed": args.seed,
        "initial_checkpoint": args.initial_checkpoint,
        "best_epoch": best["epoch"],
        "best_monitor": best["monitor"],
        "final_normalized_commutator": final_normalized_commutator,
        "effective_stationarity_weight_mev_inverse": effective_stationarity_weight,
        "density_energy_weight": args.density_energy_weight,
        "density_inverse_weight": args.density_inverse_weight,
        "density_branch_floor": args.density_branch_floor,
        "density_refresh_interval": args.density_refresh_interval,
        "density_stationarity_gate": args.density_stationarity_gate,
        "density_spectral_gate": args.density_spectral_gate,
        "checkpoint_policy": args.checkpoint_policy,
        "internal_gate_passed": internal_gate,
        "internal_physics_gate_passed": internal_gate,
        "physics_gate_passed": internal_gate,
        "external_scf_validation_passed": None,
        "energy_per_a_no_com": final_result.energy_per_a_no_com,
        "diagnostics": final_result.diagnostics,
        "solver_counts": solver_counts,
        "spectral_validation": spectral_validation,
        "formula_invariants": formula_invariants,
        "notes": [
            "The training loop varies a hard-constrained one-body density projector.",
            "Training does not solve or diagonalize Dirac and never refreshes a subspace.",
            "The positive-energy branch is constrained by differentiable inverse-Hamiltonian linear solves.",
            "The exact free Dirac kinetic energy is evaluated as Tr(T0 P); no TF/ETF approximation is used.",
            "SCF data are not used by the loss.",
        ],
    }
    write_history(out / "history.csv", history)
    write_history(out / "density_matrix_history.csv", history)
    write_json(out / "training_solver_counts.json", solver_counts)
    write_json(out / "final_spectral_validation.json", spectral_validation)
    write_json(out / "density_matrix_diagnostics.json", final_result.diagnostics)
    write_json(out / "formula_invariants.json", formula_invariants)
    write_json(out / "summary.json", summary)
    torch.save(
        {
            "checkpoint_format": "dpl-rmf-fast-1d-density-matrix-v2",
            "dpl_rmf_version": __version__,
            "release_name": __release_name__,
            "model_state_dict": network.state_dict(),
            "summary": summary,
        },
        out / "model.pt",
    )
    return summary
