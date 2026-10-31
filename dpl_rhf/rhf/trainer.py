from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from contextlib import nullcontext
from pathlib import Path
import time
import fcntl

import numpy as np
import torch

from dpl_rhf import __release_name__, __version__
from dpl_rhf.io.outputs import ensure_out, write_history, write_json
from dpl_rhf.models.density_matrix_net import DensityMatrixNet
from dpl_rhf.rhf.backend import DensityMatrixRHFBackend, DensityMatrixRHFResult
from dpl_rhf.rhf.disk_offload import SavedTensorDiskOffload, remove_stale_graph_directories
from dpl_rhf.training.inverse_hamiltonian import (
    dirac_sea_weight,
    inverse_hamiltonian_constraint,
    positive_energy_projectors,
)


@dataclass(frozen=True)
class RHFTrainingConfig:
    epochs: int = 500
    hidden: int = 64
    depth: int = 3
    radial_basis_size: int = 16
    correction_scale: float = 1.0e-2
    learning_rate: float = 1.0e-3
    energy_weight: float = 1.0e-2
    stationarity_weight: float | None = 1.0
    grad_clip: float = 10.0
    print_every: int = 10
    seed: int = 20260809
    refresh_interval: int = 0
    inverse_weight: float = 1.0
    branch_floor: float = 1.0
    channel_balance_weight: float = 0.0
    channel_balance_epochs: int = 0
    checkpoint_every: int = 10
    resume: bool = False
    graph_offload: str = "none"
    graph_offload_dir: str = "/tmp/dpl-rhf-graph-cache"
    graph_offload_min_bytes: int = 1 << 20
    initial_checkpoint: str | None = None
    initial_dilation_factor: float = 1.0
    initial_uniform_dilation: float | None = None
    no_sea_projection: bool = True


def build_network(
    backend: DensityMatrixRHFBackend, config: RHFTrainingConfig
) -> DensityMatrixNet:
    return DensityMatrixNet(
        torch.as_tensor(backend.r, dtype=torch.float64, device=backend.device),
        backend.block_specs,
        hidden=config.hidden,
        depth=config.depth,
        radial_basis_size=config.radial_basis_size,
        correction_scale=config.correction_scale,
    ).to(backend.device)


def _load_network_state(
    network: DensityMatrixNet, state: dict[str, torch.Tensor]
) -> None:
    incompatible = network.load_state_dict(state, strict=False)
    allowed_missing = {
        name
        for index in range(len(network.blocks))
        for name in (
            f"blocks.{index}.raw_dilation",
            f"blocks.{index}.dilation_tangent",
        )
    }
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys).difference(allowed_missing)
    if unexpected or missing:
        raise RuntimeError(
            f"incompatible RHF checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _scalar_terms(result: DensityMatrixRHFResult) -> dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in result.action_terms.items()
        if value.numel() == 1
    }


def save_result(
    out: Path,
    backend: DensityMatrixRHFBackend,
    network: DensityMatrixNet,
    result: DensityMatrixRHFResult,
    config: RHFTrainingConfig,
    history: list[dict[str, float]],
) -> None:
    arrays: dict[str, np.ndarray] = {"r": backend.r}
    arrays.update(
        {name: value.detach().cpu().numpy() for name, value in result.densities.items()}
    )
    arrays.update(
        {name: value.detach().cpu().numpy() for name, value in result.fields.items()}
    )
    np.savez_compressed(out / "local_observables.npz", **arrays)
    np.savez(
        out / "nonlocal_hamiltonians.npz",
        **{
            f"h_total_{name}": value.detach().cpu().numpy()
            for name, value in result.hamiltonians.items()
        },
        **{
            f"h_kinetic_{name}": value.detach().cpu().numpy()
            for name, value in result.kinetic_hamiltonians.items()
        },
        **{
            f"h_direct_{name}": value.detach().cpu().numpy()
            for name, value in result.direct_hamiltonians.items()
        },
        **{
            f"h_fock_{name}": value.detach().cpu().numpy()
            for name, value in result.fock_hamiltonians.items()
        },
    )
    factor_arrays: dict[str, np.ndarray] = {"r": backend.r}
    for index, block in enumerate(result.blocks):
        prefix = f"block_{index:02d}"
        factor_arrays[f"{prefix}_species_sign"] = np.asarray(
            float(block["species_sign"])
        )
        factor_arrays[f"{prefix}_kappa"] = np.asarray(int(block["kappa"]))
        factor_arrays[f"{prefix}_degeneracy"] = np.asarray(
            float(block["degeneracy"])
        )
        for name in ("principals", "occupations", "factors"):
            factor_arrays[f"{prefix}_{name}"] = block[name].detach().cpu().numpy()
    np.savez_compressed(out / "density_matrix_factors.npz", **factor_arrays)
    summary = {
        "release_name": __release_name__,
        "version": __version__,
        "solver": "density-matrix-neural-rhf",
        "functional": backend.params.model,
        "nucleus": {"z": backend.case.z, "n": backend.case.n, "a": backend.case.a},
        "energy_total_no_com_mev": result.energy_total_no_com,
        "energy_per_a_no_com_mev": result.energy_per_a_no_com,
        "energy_terms_mev": _scalar_terms(result),
        "diagnostics": result.diagnostics,
        "solver_counts": {
            "eigensolve_count": (
                backend.initialization_block_diagonalizations
                + backend.refresh_block_diagonalizations
            ),
            "refresh_count": backend.refresh_phases,
            "training_dirac_solves": backend.training_dirac_solves,
            "inverse_hamiltonian_linear_solves": int(
                history[-1].get("inverse_hamiltonian_linear_solves", 0)
                if history else 0
            ),
        },
        "training_config": asdict(config),
        "physics": {
            "direct_channels": (
                ["sigma", "omega", "rho-vector", "rho-tensor", "rho-vector-tensor", "coulomb"]
                if backend.params.model == "PKA1"
                else ["sigma", "omega", "rho-vector", "coulomb"]
            ),
            "fock_channels": (
                ["sigma", "omega-time-space", "rho-time-space", "rho-tensor-time-space", "rho-vector-tensor", "pion-pv", "coulomb-time-space"]
                if backend.params.model == "PKA1"
                else ["sigma", "omega-time-space", "rho-time-space", "coulomb-time-space"]
            ),
            "density_dependence_in_autograd": True,
            "fock_rearrangement_definition": (
                "explicit channel contractions from RHF-e and Core Meanfield:Rearrange"
            ),
            "hamiltonian_definition": (
                "RHF-e Eqs. (3.392)-(3.394): h_kin + h_D(local, including "
                "density-dependent rearrangement) + h_E(nonlocal exchange)"
            ),
            "first_forward_complete_channel_gradient_audit": (
                backend.initial_channel_gradient_norms
                if backend.params.model == "PKA1"
                else {}
            ),
            "training_dirac_solves": 0,
            "initialization_block_diagonalizations": backend.initialization_block_diagonalizations,
            "spectral_refresh_phases": backend.refresh_phases,
            "refresh_block_diagonalizations": backend.refresh_block_diagonalizations,
            "center_of_mass_correction": False,
            "energy_weight": config.energy_weight,
            "stationarity_weight": config.stationarity_weight,
            "inverse_hamiltonian_weight": config.inverse_weight,
            "initial_checkpoint": config.initial_checkpoint,
            "initial_dilation_factor": config.initial_dilation_factor,
            "initial_uniform_dilation": config.initial_uniform_dilation,
            "no_sea_projection": config.no_sea_projection,
        },
    }
    write_json(out / "summary.json", summary)
    write_history(out / "history.csv", history)
    torch.save(
        {
            "model_state": network.state_dict(),
            "positive_projectors": network.positive_projectors,
            "config": asdict(config),
            "case": {
                "model": backend.case.model,
                "z": backend.case.z,
                "n": backend.case.n,
                "a": backend.case.a,
            },
        },
        out / "best_checkpoint.pt",
    )


def _no_sea_shifts(backend: DensityMatrixRHFBackend) -> dict[str, float]:
    # The Hamiltonian already subtracts the rest mass, so W=-M lies between
    # the bound positive-energy states and the Dirac sea.
    return {"n": -backend.params.mass_n, "p": -backend.params.mass_p}


def _current_positive_projectors(
    backend: DensityMatrixRHFBackend, network: DensityMatrixNet, offload: bool
) -> dict[str, torch.Tensor]:
    context = (
        torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext()
    )
    with context:
        result = backend.evaluate(network(), create_graph=False)
    return positive_energy_projectors(
        result.hamiltonians, _no_sea_shifts(backend)
    ).projectors


def train_rhf(
    backend: DensityMatrixRHFBackend,
    config: RHFTrainingConfig,
    out_dir: str | Path,
) -> tuple[DensityMatrixNet, DensityMatrixRHFResult]:
    if config.epochs < 0:
        raise ValueError("epochs must be non-negative")
    if config.refresh_interval != 0:
        raise ValueError("the no-refresh solver requires refresh_interval=0")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    out = ensure_out(out_dir)
    lock_file = (out / ".training.lock").open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another RHF training process is writing {out}") from exc
    network = build_network(backend, config)
    if config.initial_checkpoint is not None and not config.resume:
        initial = torch.load(
            config.initial_checkpoint, map_location=backend.device, weights_only=False
        )
        _load_network_state(network, initial["model_state"])
        with torch.no_grad():
            for block in network.blocks:
                block.raw_dilation.mul_(config.initial_dilation_factor)
    if config.initial_uniform_dilation is not None and not config.resume:
        with torch.no_grad():
            for block in network.blocks:
                block.raw_dilation.fill_(config.initial_uniform_dilation)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, threshold=1.0e-3
    )
    history: list[dict[str, float]] = []
    best_monitor = float("inf")
    best_state = copy.deepcopy(network.state_dict())
    best_projectors: dict[str, torch.Tensor] | None = None
    stationarity_reference: float | None = None
    energy_reference: float | None = None
    effective_stationarity_weight = config.stationarity_weight
    start = time.time()
    start_epoch = 0
    inverse_linear_solves = 0
    training_checkpoint = out / "training_checkpoint.pt"
    backend.functional.recompute_fock = config.graph_offload == "recompute"
    if config.graph_offload == "disk":
        removed = remove_stale_graph_directories(config.graph_offload_dir)
        if removed:
            print(f"removed {removed} stale RHF graph-cache directories", flush=True)
    if config.resume and training_checkpoint.exists():
        saved = torch.load(
            training_checkpoint, map_location=backend.device, weights_only=False
        )
        _load_network_state(network, saved["model_state"])
        try:
            optimizer.load_state_dict(saved["optimizer_state"])
            scheduler.load_state_dict(saved["scheduler_state"])
        except ValueError:
            # Checkpoints before the analytic dilation direction have fewer
            # parameters. The physical state is migrated exactly; only Adam's
            # historical moments are restarted for the enlarged model.
            pass
        history = saved["history"]
        best_monitor = float(saved["best_monitor"])
        best_state = saved["best_state"]
        stationarity_reference = saved["stationarity_reference"]
        energy_reference = saved.get("energy_reference")
        effective_stationarity_weight = saved["effective_stationarity_weight"]
        start_epoch = int(saved["next_epoch"])
        if config.no_sea_projection and saved.get("positive_projectors") is not None:
            network.set_positive_projectors(saved["positive_projectors"])
            best_projectors = saved.get("best_projectors")
        backend.refresh_phases = int(saved.get("refresh_phases", 0))
        backend.refresh_block_diagonalizations = int(
            saved.get("refresh_block_diagonalizations", 0)
        )
        previous_objective = saved.get("objective_weights")
        current_objective = {
            "energy_weight": config.energy_weight,
            "stationarity_weight": config.stationarity_weight,
            "inverse_weight": config.inverse_weight,
            "branch_floor": config.branch_floor,
            "channel_balance_weight": config.channel_balance_weight,
            "channel_balance_epochs": config.channel_balance_epochs,
            "no_sea_projection": config.no_sea_projection,
        }
        if previous_objective != current_objective:
            # Selection objectives from different continuation stages are not
            # numerically comparable. Continue from the latest state, but
            # establish a fresh variational-best checkpoint for this stage.
            best_monitor = float("inf")
            best_state = copy.deepcopy(network.state_dict())
            best_projectors = network.positive_projectors
            stationarity_reference = None
            energy_reference = None
        print(f"resuming RHF training at epoch {start_epoch}", flush=True)

    if config.no_sea_projection and network.positive_projectors is None:
        # The first projector comes from the analytic initial density; each
        # later epoch uses the Hamiltonian of the preceding density, as in
        # the no-sea approximation of a self-consistent iteration.
        network.set_positive_projectors(
            _current_positive_projectors(
                backend,
                network,
                backend.device.type == "cuda"
                and backend.case.a >= 48
                and config.graph_offload in {"memory", "cpu"},
            )
        )
    if best_projectors is None:
        best_projectors = network.positive_projectors

    for epoch in range(start_epoch, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_projectors = network.positive_projectors
        # The batched Fock graph of Pb208 fits on one device.  Offloading
        # (save_on_cpu/disk) only changes storage placement, never the exact
        # gradient, and remains available for small-memory cards.  Do not pin
        # this graph: heavy PKA1 can lock enough host pages to stall a server.
        offload = (
            backend.device.type == "cuda"
            and backend.case.a >= 48
            and config.graph_offload != "none"
        )
        disk_offload: SavedTensorDiskOffload | None = None
        if offload and config.graph_offload == "disk":
            disk_offload = SavedTensorDiskOffload(
                config.graph_offload_dir,
                min_bytes=config.graph_offload_min_bytes,
            )
            saved_tensor_context = disk_offload
        elif offload and config.graph_offload in {"memory", "cpu"}:
            saved_tensor_context = torch.autograd.graph.save_on_cpu(pin_memory=False)
        else:
            saved_tensor_context = nullcontext()
        try:
            with saved_tensor_context:
                result = backend.evaluate(
                    network(),
                    create_graph=epoch < config.epochs or effective_stationarity_weight is None,
                )
                inverse = inverse_hamiltonian_constraint(
                    result.blocks,
                    result.hamiltonians,
                    _no_sea_shifts(backend),
                    branch_floor=config.branch_floor,
                )
                inverse_linear_solves += inverse.solve_count
                energy_per_a = result.energy_total_tensor / backend.case.a
                # Expanded exactly as ||R_kin + R_D + R_E||^2 so the kinetic,
                # Hartree and Fock Hamiltonians all remain explicit in the
                # backward graph, including their physical interference.
                stationarity_square = sum(
                    result.stationarity_quadratic_terms.values(),
                    torch.zeros_like(result.stationarity_residual_tensor),
                )
            parameters = [
                parameter for parameter in network.parameters() if parameter.requires_grad
            ]
            channel_energies = {
                "kinetic": result.action_terms["E_kinetic"] / backend.case.a,
                "direct": result.action_terms["E_direct"] / backend.case.a,
                "fock": result.action_terms["E_fock"] / backend.case.a,
            }
            stage_epoch = epoch - start_epoch
            if config.channel_balance_epochs > 0:
                channel_balance_scale = config.channel_balance_weight * max(
                    0.0, 1.0 - stage_epoch / config.channel_balance_epochs
                )
            else:
                channel_balance_scale = 0.0
            channel_gradient_norms: dict[str, torch.Tensor] = {}
            balanced_channel_energy = torch.zeros_like(energy_per_a)
            if channel_balance_scale > 0.0:
                for name, channel_energy in channel_energies.items():
                    gradients = torch.autograd.grad(
                        channel_energy,
                        parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    norm = torch.sqrt(
                        sum(
                            torch.sum(gradient.square())
                            for gradient in gradients
                            if gradient is not None
                        )
                    ).clamp_min(1.0e-14)
                    channel_gradient_norms[name] = norm.detach()
                    balanced_channel_energy = (
                        balanced_channel_energy + channel_energy / norm.detach()
                    )
            if effective_stationarity_weight is None:
                energy_gradients = torch.autograd.grad(
                    energy_per_a, parameters, retain_graph=True, allow_unused=True
                )
                stationarity_gradients = torch.autograd.grad(
                    stationarity_square, parameters, retain_graph=True, allow_unused=True
                )
                energy_norm = torch.sqrt(
                    sum(torch.sum(g.square()) for g in energy_gradients if g is not None)
                )
                stationarity_norm = torch.sqrt(
                    sum(torch.sum(g.square()) for g in stationarity_gradients if g is not None)
                )
                effective_stationarity_weight = float(
                    (
                        config.energy_weight
                        * energy_norm
                        / stationarity_norm.clamp_min(1.0e-14)
                    ).detach().cpu()
                )
            loss = (
                config.energy_weight * energy_per_a
                + effective_stationarity_weight * stationarity_square
                + config.inverse_weight * inverse.loss
                + channel_balance_scale * balanced_channel_energy
            )
            no_sea = positive_energy_projectors(
                result.hamiltonians, _no_sea_shifts(backend)
            )
            # Sea weight of the occupied space measured against the mean field
            # it generates; nonzero only through the one-epoch projector lag.
            sea_weight = dirac_sea_weight(result.blocks, no_sea.projectors)
            stationarity = float(result.stationarity_residual_tensor.detach().cpu())
            if stationarity_reference is None:
                stationarity_reference = max(stationarity, 1.0e-14)
            if energy_reference is None:
                energy_reference = result.energy_per_a_no_com
            normalized_stationarity = stationarity / stationarity_reference
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "energy_total_no_com_mev": result.energy_total_no_com,
                "energy_per_a_no_com_mev": result.energy_per_a_no_com,
                "stationarity_rms_mev": stationarity,
                "kinetic_residual_rms_mev": float(
                    result.stationarity_components["kinetic"].detach().cpu()
                ),
                "direct_residual_rms_mev": float(
                    result.stationarity_components["direct"].detach().cpu()
                ),
                "fock_residual_rms_mev": float(
                    result.stationarity_components["fock"].detach().cpu()
                ),
                **{
                    f"stationarity_{name}_mev2": float(value.detach().cpu())
                    for name, value in result.stationarity_quadratic_terms.items()
                },
                "normalized_stationarity": normalized_stationarity,
                "stationarity_weight_mev_inverse": effective_stationarity_weight,
                "energy_weight": config.energy_weight,
                "channel_balance_scale": channel_balance_scale,
                **{
                    f"{name}_network_gradient_norm": float(norm.cpu())
                    for name, norm in channel_gradient_norms.items()
                },
                "inverse_tangent_residual": float(inverse.tangent_residual.detach().cpu()),
                "minimum_branch_score": float(inverse.minimum_branch_score.detach().cpu()),
                "mean_branch_score": float(inverse.mean_branch_score.detach().cpu()),
                "linear_solve_residual": float(inverse.linear_solve_residual.detach().cpu()),
                "dirac_sea_weight": sea_weight,
                "no_sea_projection": float(epoch_projectors is not None),
                "no_sea_newton_iterations": no_sea.max_newton_iterations,
                "no_sea_trace_error": no_sea.max_trace_error,
                "inverse_hamiltonian_linear_solves": inverse_linear_solves,
                "eigensolve_count": (
                    backend.initialization_block_diagonalizations
                    + backend.refresh_block_diagonalizations
                ),
                "refresh_count": backend.refresh_phases,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - start,
            }
            branch_violation = max(
                0.0, config.branch_floor - row["minimum_branch_score"]
            )
            monitor = (
                normalized_stationarity
                + config.energy_weight
                * (row["energy_per_a_no_com_mev"] - energy_reference)
                + 1.0e3 * branch_violation**2
                + 1.0e3 * sea_weight
            )
            row["selection_objective"] = monitor
            row["energy_reference_mev_per_a"] = energy_reference
            history.append(row)
            if monitor < best_monitor and np.isfinite(monitor):
                best_monitor = monitor
                best_state = copy.deepcopy(network.state_dict())
                best_projectors = epoch_projectors
            if epoch % max(config.print_every, 1) == 0 or epoch == config.epochs:
                print(
                    f"epoch={epoch:5d} loss={row['loss']: .8f} "
                    f"E/A={row['energy_per_a_no_com_mev']: .8f} "
                    f"stationarity={row['stationarity_rms_mev']:.3e} MeV "
                    f"w_H={row['stationarity_weight_mev_inverse']:.3e} "
                    f"select={row['selection_objective']:.3e}",
                    flush=True,
                )
                write_history(out / "history.csv", history)
            if epoch == config.epochs:
                break
            if backend.device.type == "cuda":
                torch.cuda.empty_cache()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
            optimizer.step()
            if config.no_sea_projection:
                network.set_positive_projectors(no_sea.projectors)
        finally:
            if disk_offload is not None:
                written_gib = disk_offload.bytes_written / 1024**3
                read_gib = disk_offload.bytes_read / 1024**3
                disk_offload.cleanup()
                if epoch % max(config.print_every, 1) == 0:
                    print(
                        f"graph_disk_cache epoch={epoch} wrote={written_gib:.2f} GiB "
                        f"read={read_gib:.2f} GiB",
                        flush=True,
                    )
        scheduler.step(monitor)
        if config.checkpoint_every > 0 and (epoch + 1) % config.checkpoint_every == 0:
            payload = {
                "next_epoch": epoch + 1,
                "model_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_monitor": best_monitor,
                "best_state": best_state,
                "stationarity_reference": stationarity_reference,
                "energy_reference": energy_reference,
                "effective_stationarity_weight": effective_stationarity_weight,
                "refresh_phases": backend.refresh_phases,
                "refresh_block_diagonalizations": backend.refresh_block_diagonalizations,
                "objective_weights": {
                    "energy_weight": config.energy_weight,
                    "stationarity_weight": config.stationarity_weight,
                    "inverse_weight": config.inverse_weight,
                    "branch_floor": config.branch_floor,
                    "channel_balance_weight": config.channel_balance_weight,
                    "channel_balance_epochs": config.channel_balance_epochs,
                    "no_sea_projection": config.no_sea_projection,
                },
                "positive_projectors": network.positive_projectors,
                "best_projectors": best_projectors,
            }
            temporary = training_checkpoint.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(training_checkpoint)

    _load_network_state(network, best_state)
    network.set_positive_projectors(best_projectors)
    final_result = backend.evaluate(network(), create_graph=False)
    save_result(out, backend, network, final_result, config, history)
    return network, final_result
