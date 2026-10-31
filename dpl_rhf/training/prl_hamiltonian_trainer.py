from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.fortran_rmf import FortranRMFBackend
from dpl_rhf.backends.torch_rmf import TorchRMFBackend
from dpl_rhf.functionals.pkdd_rmf import PKDDRMFFunctionalSpec
from dpl_rhf.functionals.pkdd_action import canonical_rmf_model
from dpl_rhf.io.outputs import ensure_out, write_history, write_json
from dpl_rhf.models.hamiltonian_net import (
    DirectHamiltonianParameterization,
    LocalHamiltonianNet,
    PHYSICAL_COMPONENT_NAMES,
    compose_hamiltonian,
    physical_component_gradient,
)
from dpl_rhf.training.surrogate_gradient import generalized_prl_gradient, surrogate_loss


def _tensor_dict_to_numpy(data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in data.items()}


def _compose_backend_hamiltonian(
    components: torch.Tensor, backend: TorchRMFBackend
) -> torch.Tensor:
    """Assemble channels with the same parameter set used by the RMF graph."""
    return compose_hamiltonian(components, backend.functional.params)


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().square())).cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    denominator = torch.linalg.norm(left_flat) * torch.linalg.norm(right_flat)
    if float(denominator.cpu()) <= 1.0e-30:
        return float("nan")
    return float(torch.dot(left_flat, right_flat).div(denominator).cpu())


def _parameter_gradient_vector(
    scalar: torch.Tensor, parameters: list[torch.nn.Parameter], *, retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        scalar, parameters, retain_graph=retain_graph, allow_unused=True
    )
    pieces = [
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces)


def _error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference_array), 1.0e-14)
        ),
    }


def _compare_scf_profiles(out: Path, hamiltonian: np.ndarray, result: object) -> dict:
    """Compare radial observables only after training; never feed them to loss."""
    scf_potentials = np.load(out / "scf_potentials.npz")
    scf_densities = np.load(out / "scf_densities.npz")
    channels = ("vps_n", "vms_n", "vps_p", "vms_p")
    potential_metrics = {
        name: _error_metrics(hamiltonian[i], scf_potentials[name]) for i, name in enumerate(channels)
    }
    fields = _tensor_dict_to_numpy(result.fields)
    field_metrics = {
        name: _error_metrics(fields[name], scf_potentials[name]) for name in ("sigma", "omega", "rho", "coul")
    }
    densities = _tensor_dict_to_numpy(result.densities)
    density_profiles = {
        "rho_s": densities["rho_s_n"] + densities["rho_s_p"],
        "rho_b": densities["rho_v_n"] + densities["rho_v_p"],
        "rho_b3": densities["rho_v_n"] - densities["rho_v_p"],
    }
    density_metrics = {
        name: _error_metrics(profile, scf_densities[name]) for name, profile in density_profiles.items()
    }
    comparison = {
        "role": "post-training external diagnostic; SCF profiles are not training labels",
        "potentials_fm_inverse": potential_metrics,
        "fields_fm_inverse": field_metrics,
        "densities_fm_inverse_cubed": density_metrics,
    }
    write_json(out / "profile_compare.json", comparison)
    return comparison


def _canonical_orbital_name(name: object) -> str:
    value = str(name).strip()
    if value.startswith(("N.", "P.")):
        value = value[2:]
    return value.replace(".", "")


def _validate_adf_against_shooting(out: Path, backend: TorchRMFBackend) -> dict:
    """Validate ADF eigenpairs on a fixed SCF potential against Core shooting."""
    potential_data = np.load(out / "scf_potentials.npz")
    shooting = np.load(out / "scf_wavefunctions.npz")
    stack = np.stack([
        potential_data[name] for name in ("vps_n", "vms_n", "vps_p", "vms_p")
    ])
    with torch.no_grad():
        adf = backend.matrix_solver.solve(
            torch.as_tensor(stack, dtype=torch.float64, device=backend.device)
        )

    shooting_map = {}
    for index in range(len(shooting["energy"])):
        if float(shooting["occupancy"][index]) <= 0.0:
            continue
        key = (
            str(shooting["species"][index]),
            _canonical_orbital_name(shooting["name"][index]),
            int(shooting["kappa"][index]),
        )
        shooting_map[key] = index

    rows = []
    for index, occupancy in enumerate(adf["occupancy"]):
        if float(occupancy.detach().cpu()) <= 0.0:
            continue
        key = (
            str(backend.orbital_metadata["species"][index]),
            _canonical_orbital_name(backend.orbital_metadata["name"][index]),
            int(adf["kappa"][index].detach().cpu()),
        )
        if key not in shooting_map:
            continue
        shooting_index = shooting_map[key]
        adf_energy = float(adf["epsilon"][index].detach().cpu())
        shooting_energy = float(shooting["energy"][shooting_index])
        adf_g = adf["G"][index].detach().cpu().numpy()
        adf_f = adf["F"][index].detach().cpu().numpy()
        shooting_g = np.asarray(shooting["G"][shooting_index])
        shooting_f = np.asarray(shooting["F"][shooting_index])
        overlap = np.trapezoid(adf_g * shooting_g + adf_f * shooting_f, backend.r)
        rows.append({
            "species": key[0],
            "name": key[1],
            "kappa": key[2],
            "adf_energy_mev": adf_energy,
            "shooting_energy_mev": shooting_energy,
            "delta_mev": adf_energy - shooting_energy,
            "wavefunction_overlap_abs": float(abs(overlap)),
            "node_count": int(adf["node_count"][index].detach().cpu()),
            "expected_node_count": int(adf["expected_node_count"][index].detach().cpu()),
        })
    write_history(out / "adf_shooting_orbitals.csv", rows)
    differences = np.asarray([row["delta_mev"] for row in rows], dtype=np.float64)
    validation = {
        "role": "post-training discretization validation on the same fixed SCF potential",
        "occupied_matched": len(rows),
        "occupied_expected": int(sum(np.asarray(shooting["occupancy"]) > 0.0)),
        "energy_mae_mev": float(np.mean(np.abs(differences))) if len(rows) else None,
        "energy_max_abs_mev": float(np.max(np.abs(differences))) if len(rows) else None,
        "minimum_wavefunction_overlap_abs": (
            float(min(row["wavefunction_overlap_abs"] for row in rows)) if rows else None
        ),
        "max_node_mismatch": (
            max(abs(row["node_count"] - row["expected_node_count"]) for row in rows)
            if rows else None
        ),
    }
    write_json(out / "adf_shooting_validation.json", validation)
    return validation


def train_prl_hamiltonian(args) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    out = ensure_out(args.out)
    case = NucleusCase(model=args.model, z=args.z, n=args.n, a=args.a or args.z + args.n)
    model = canonical_rmf_model(case.model)
    case = NucleusCase(model=model, z=case.z, n=case.n, a=case.a)
    if args.functional not in {"rmf-dd", "rmf-pkdd"}:
        raise ValueError("the differentiable functional must be rmf-dd")
    if args.functional == "rmf-pkdd" and model != "PKDD":
        raise ValueError("rmf-pkdd compatibility mode only accepts PKDD")
    if args.backend == "torch-rmf":
        backend = TorchRMFBackend(case, device=device, derivative_order=args.derivative_order)
        functional = backend.functional_spec
        initial = backend.initial_hamiltonian_np
    elif args.backend == "fortran-fixed":
        backend = FortranRMFBackend(case)
        functional = PKDDRMFFunctionalSpec(
            case.z, case.n, backend.r, params=backend.params
        )
        initial = functional.initial_hamiltonian()
    else:
        raise ValueError(f"unsupported backend: {args.backend}")
    if args.mode == "network":
        if args.epochs > 200 and not args.allow_unvalidated_network:
            if not args.direct_gate:
                raise ValueError("network runs over 200 epochs require --direct-gate or --allow-unvalidated-network")
            direct_gate = __import__("json").loads(Path(args.direct_gate).read_text())
            if (
                direct_gate.get("variation_mode") != "direct"
                or not direct_gate.get(
                    "internal_physics_gate_passed",
                    direct_gate.get("physics_gate_passed", False),
                )
            ):
                raise ValueError("the supplied direct variation gate has not passed")
        network = LocalHamiltonianNet(
            backend.r,
            case.z,
            case.n,
            initial,
            hidden=args.hidden,
            depth=args.network_depth,
            radial_basis_size=args.radial_basis_size,
            message_neighbors=args.message_neighbors,
            params=backend.functional.params,
        ).to(device)
    else:
        network = DirectHamiltonianParameterization(
            backend.r,
            case.z,
            case.n,
            order=args.direct_order,
            params=backend.functional.params,
        ).to(device)
    initial_lr = float(
        args.lr if args.lr is not None
        else 6.0e-5 if args.mode == "network"
        else 5.0e-4
    )
    current_lr = initial_lr

    def make_optimizer():
        if args.mode == "network":
            return torch.optim.Adam(network.parameters(), lr=current_lr)
        return torch.optim.AdamW(
            network.parameters(), lr=current_lr, weight_decay=args.weight_decay
        )

    optimizer = make_optimizer()
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.scheduler_patience,
            cooldown=args.scheduler_cooldown,
            threshold=0.05,
        )
        if args.mode == "network"
        else None
    )
    rows: list[dict] = []
    physics_rows: list[dict] = []
    gradient_rows: list[dict] = []
    occupation_rows: list[dict] = []
    component_rows: list[dict] = []
    channel_gradient_rows: list[dict] = []
    best = {"monitor": float("inf"), "epoch": 0, "state": None}
    residual_reference = None
    backtrack_count = 0
    lambda_mode = (
        "balanced" if args.lambda_mode == "auto" and args.mode == "network"
        else "fixed" if args.lambda_mode == "auto"
        else args.lambda_mode
    )
    effective_lambda_reconstruct = None

    for epoch in range(1, args.epochs + 1):
        components = network.components()
        hamiltonian = _compose_backend_hamiltonian(components, backend)
        if args.backend == "torch-rmf":
            result = backend.evaluate_tensor(hamiltonian)
            reconstructed = result.reconstructed_hamiltonian
            hbar_c = backend.functional.params.hbar_c
            hamiltonian_mev = hamiltonian * hbar_c
            reconstructed_mev = reconstructed * hbar_c
            # dE/dH_MeV = dE/dH_fm / (hbar*c).
            grad_energy_mev = result.grad_energy_h / hbar_c
        else:
            raise ValueError("strict PRL training requires the differentiable torch-rmf backend")
        reconstructed_components = torch.stack([
            result.fields[name] for name in ("component_S", "component_V0", "component_V3", "component_VC")
        ]) if args.backend == "torch-rmf" else None
        if reconstructed_components is None:
            raise ValueError("physical-component PRL training requires the differentiable torch-rmf backend")
        hamiltonian_residual = hamiltonian_mev.detach() - reconstructed_mev.detach()
        if effective_lambda_reconstruct is None:
            if lambda_mode == "balanced":
                residual_rms = _rms(hamiltonian_residual)
                weighted_energy_rms = _rms(
                    float(args.energy_gradient_weight) * grad_energy_mev
                )
                effective_lambda_reconstruct = (
                    weighted_energy_rms / max(residual_rms, 1.0e-30)
                )
            else:
                effective_lambda_reconstruct = float(args.lambda_reconstruct)
        reconstruction_gradient = effective_lambda_reconstruct * hamiltonian_residual
        grad_h = generalized_prl_gradient(
            hamiltonian_mev,
            reconstructed_mev,
            grad_energy_mev,
            effective_lambda_reconstruct,
            args.energy_gradient_weight,
        )
        generalized_gradient_rms = _rms(grad_h)
        energy_gradient_rms = _rms(grad_energy_mev)
        reconstruction_scaled_rms = _rms(hamiltonian_mev - reconstructed_mev)
        field_gradient_cosine = _cosine(grad_energy_mev, reconstruction_gradient)

        record_gradient_alignment = epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs
        parameter_gradient_cosine = float("nan")
        if record_gradient_alignment:
            parameters = list(network.parameters())
            energy_surrogate = surrogate_loss(
                hamiltonian_mev,
                float(args.energy_gradient_weight) * grad_energy_mev,
            )
            reconstruction_surrogate = surrogate_loss(hamiltonian_mev, reconstruction_gradient)
            energy_parameter_gradient = _parameter_gradient_vector(
                energy_surrogate, parameters, retain_graph=True
            )
            reconstruction_parameter_gradient = _parameter_gradient_vector(
                reconstruction_surrogate, parameters, retain_graph=True
            )
            parameter_gradient_cosine = _cosine(
                energy_parameter_gradient, reconstruction_parameter_gradient
            )
            gradient_rows.append({
                "epoch": epoch,
                "energy_gradient_h_mev_rms": energy_gradient_rms,
                "hamiltonian_residual_mev_rms": reconstruction_scaled_rms,
                "reconstruction_gradient_rms": _rms(reconstruction_gradient),
                "generalized_gradient_rms": generalized_gradient_rms,
                "hamiltonian_gradient_cosine": field_gradient_cosine,
                "energy_parameter_gradient_norm": float(torch.linalg.norm(energy_parameter_gradient).detach().cpu()),
                "reconstruction_parameter_gradient_norm": float(torch.linalg.norm(reconstruction_parameter_gradient).detach().cpu()),
                "parameter_gradient_cosine": parameter_gradient_cosine,
            })
            energy_component_gradient = physical_component_gradient(
                float(args.energy_gradient_weight) * grad_energy_mev
            )
            reconstruction_component_gradient = physical_component_gradient(reconstruction_gradient)
            component_values_mev = components * hbar_c
            for component_index, component_name in enumerate(PHYSICAL_COMPONENT_NAMES):
                energy_component_surrogate = torch.sum(
                    component_values_mev[component_index]
                    * energy_component_gradient[component_index].detach()
                )
                reconstruction_component_surrogate = torch.sum(
                    component_values_mev[component_index]
                    * reconstruction_component_gradient[component_index].detach()
                )
                energy_parameter_component = _parameter_gradient_vector(
                    energy_component_surrogate, parameters, retain_graph=True
                )
                reconstruction_parameter_component = _parameter_gradient_vector(
                    reconstruction_component_surrogate, parameters, retain_graph=True
                )
                channel_gradient_rows.append({
                    "epoch": epoch,
                    "component": component_name,
                    "energy_h_gradient_rms": _rms(energy_component_gradient[component_index]),
                    "reconstruction_h_gradient_rms": _rms(reconstruction_component_gradient[component_index]),
                    "hamiltonian_gradient_cosine": _cosine(
                        energy_component_gradient[component_index],
                        reconstruction_component_gradient[component_index],
                    ),
                    "energy_parameter_gradient_norm": float(
                        torch.linalg.norm(energy_parameter_component).detach().cpu()
                    ),
                    "reconstruction_parameter_gradient_norm": float(
                        torch.linalg.norm(reconstruction_parameter_component).detach().cpu()
                    ),
                    "parameter_gradient_cosine": _cosine(
                        energy_parameter_component, reconstruction_parameter_component
                    ),
                })
        component_row = {"epoch": epoch}
        for index, name in enumerate(PHYSICAL_COMPONENT_NAMES):
            delta = (components[index].detach() - reconstructed_components[index].detach()) * backend.functional.params.hbar_c
            component_row[f"{name}_rmse_mev"] = float(torch.sqrt(torch.mean(delta.square())).cpu())
            component_row[f"{name}_max_abs_mev"] = float(delta.abs().max().cpu())
        component_rows.append(component_row)
        if args.backend == "torch-rmf":
            reconstruction_rmse = _rms(hamiltonian - reconstructed)
        else:
            reconstruction_rmse = float(np.sqrt(np.mean((result.hamiltonian - result.reconstructed_hamiltonian) ** 2)))
        radius = float(result.diagnostics["rms_matter_no_com"])
        if residual_reference is None:
            residual_reference = {
                "energy": max(energy_gradient_rms, 1.0e-14),
                "reconstruction": max(reconstruction_scaled_rms, 1.0e-14),
            }
        normalized_energy = energy_gradient_rms / residual_reference["energy"]
        normalized_reconstruction = reconstruction_scaled_rms / residual_reference["reconstruction"]
        # A summed generalized gradient can be small by cancellation. A valid
        # checkpoint must reduce both independent PRL stationarity conditions.
        monitor = max(normalized_energy, normalized_reconstruction)
        physics_row = {
            "epoch": epoch,
            "energy_gradient_scaled_rms": energy_gradient_rms,
            "reconstruction_scaled_rms": reconstruction_scaled_rms,
            "energy_gradient_h_mev_rms": energy_gradient_rms,
            "hamiltonian_residual_mev_rms": reconstruction_scaled_rms,
            "hamiltonian_gradient_cosine": field_gradient_cosine,
            "parameter_gradient_cosine": parameter_gradient_cosine,
            "normalized_energy_residual": normalized_energy,
            "normalized_reconstruction_residual": normalized_reconstruction,
            "joint_monitor": monitor,
            "max_spectral_residual": float(result.diagnostics.get("max_spectral_residual", float("nan"))),
            "max_field_stationarity_rms": max(
                (float(value) for key, value in result.diagnostics.items() if key.startswith("weak_el_") and key.endswith("_rms")),
                default=float("nan"),
            ),
            "n_number_error": float(result.diagnostics.get("n_number_error", float("nan"))),
            "z_number_error": float(result.diagnostics.get("z_number_error", float("nan"))),
            "learning_rate": current_lr,
            "backtracked": False,
        }
        physics_rows.append(physics_row)
        diverged = (
            args.mode == "direct"
            and
            best["state"] is not None
            and monitor > float(args.backtrack_threshold) * best["monitor"]
        )
        if diverged:
            if backtrack_count >= args.max_backtracks:
                raise RuntimeError(
                    f"physical residual diverged after {backtrack_count} backtracks: "
                    f"monitor={monitor:.6e}, best={best['monitor']:.6e}"
                )
            network.load_state_dict(best["state"])
            current_lr *= float(args.lr_decay)
            optimizer = make_optimizer()
            backtrack_count += 1
            physics_row["backtracked"] = True
            print(
                f"{epoch:5d}/{args.epochs} BACKTRACK monitor={monitor:.3e} "
                f"best={best['monitor']:.3e} lr={current_lr:.3e}"
            )
            continue
        if monitor < best["monitor"]:
            best = {
                "monitor": monitor,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in network.state_dict().items()},
            }
        loss = surrogate_loss(hamiltonian_mev, grad_h)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step(monitor)
            current_lr = float(optimizer.param_groups[0]["lr"])
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            row = {
                "epoch": epoch,
                "surrogate_loss": float(loss.detach().cpu()),
                "monitor": float(monitor),
                "energy_per_a_no_com": float(result.energy_per_a_no_com),
                "reconstruction_rmse": reconstruction_rmse,
                "generalized_gradient_rms": generalized_gradient_rms,
                "energy_gradient_scaled_rms": energy_gradient_rms,
                "reconstruction_scaled_rms": reconstruction_scaled_rms,
                "backend_gradient_norm": float(result.diagnostics["gradient_norm"]),
                "parameter_gradient_norm": float(grad_norm),
                "rms_matter_no_com": radius,
                "learning_rate": current_lr,
                "backtrack_count": backtrack_count,
            }
            rows.append(row)
            if args.backend == "torch-rmf":
                for orbital_index, (energy, eigen_index) in enumerate(zip(result.orbitals["epsilon"], result.orbitals["eigen_index"])):
                    occupation_rows.append({
                        "epoch": epoch,
                        "orbital_index": orbital_index + 1,
                        "species_sign": float(result.orbitals["species_sign"][orbital_index].detach().cpu()),
                        "kappa": int(result.orbitals["kappa"][orbital_index].detach().cpu()),
                        "eigen_index": int(eigen_index.detach().cpu()),
                        "energy_mev": float(energy.detach().cpu()),
                        "occupancy": float(result.orbitals["occupancy"][orbital_index].detach().cpu()),
                        "node_count": int(result.orbitals["node_count"][orbital_index].detach().cpu()),
                        "expected_node_count": int(result.orbitals["expected_node_count"][orbital_index].detach().cpu()),
                    })
            print(
                f"{epoch:5d}/{args.epochs} Ls={row['surrogate_loss']:.6e} "
                f"E/A={row['energy_per_a_no_com']:.6f} "
                f"rmse={row['reconstruction_rmse']:.6e} "
                f"gE={row['energy_gradient_scaled_rms']:.3e} "
                f"rH={row['reconstruction_scaled_rms']:.3e} "
                f"|gE|={row['backend_gradient_norm']:.3e} R={row['rms_matter_no_com']:.4f}"
            )

    if best["state"] is not None and args.checkpoint_policy == "physics":
        network.load_state_dict(best["state"])
    final_components = network.components()
    final_tensor = _compose_backend_hamiltonian(final_components, backend)
    if args.backend == "torch-rmf":
        final_result = backend.evaluate_tensor(final_tensor)
        final_hamiltonian = final_result.hamiltonian.detach().cpu().numpy()
        reconstructed = final_result.reconstructed_hamiltonian.detach().cpu().numpy()
        grad_energy_h = (
            final_result.grad_energy_h / backend.functional.params.hbar_c
        ).detach().cpu().numpy()
    else:
        final_hamiltonian = final_tensor.detach().cpu().numpy()
        final_result = backend.evaluate(final_hamiltonian)
        reconstructed = final_result.reconstructed_hamiltonian
        grad_energy_h = final_result.grad_energy_h
    np.savez(
        out / "dpl_hamiltonian.npz",
        r=backend.r,
        channels=np.asarray(functional.channel_spec.names),
        stack=final_hamiltonian,
    )
    variational_components_np = final_components.detach().cpu().numpy()
    np.savez(
        out / "network_variational_components.npz",
        r=backend.r,
        components=np.asarray(PHYSICAL_COMPONENT_NAMES),
        stack=variational_components_np,
        role=np.asarray("internal coordinates used to assemble H_theta"),
    )
    np.savez(
        out / "reconstructed_hamiltonian.npz",
        r=backend.r,
        channels=np.asarray(functional.channel_spec.names),
        stack=reconstructed,
    )
    np.savez(
        out / "autograd_gradient.npz",
        r=backend.r,
        channels=np.asarray(functional.channel_spec.names),
        stack=grad_energy_h,
        energy_definition=np.asarray("total_no_com_mev"),
        hamiltonian_unit=np.asarray("MeV"),
        gradient_unit=np.asarray("dimensionless"),
    )
    if args.backend == "torch-rmf":
        reconstructed_components_np = np.stack([
            final_result.fields[name].detach().cpu().numpy()
            for name in ("component_S", "component_V0", "component_V3", "component_VC")
        ])
        np.savez(
            out / "physical_components.npz",
            r=backend.r,
            components=np.asarray(PHYSICAL_COMPONENT_NAMES),
            stack=reconstructed_components_np,
            role=np.asarray("RMF fields reconstructed from the final density"),
        )
        np.savez(out / "density_diagnostics.npz", r=backend.r, **_tensor_dict_to_numpy(final_result.densities))
        np.savez(out / "field_diagnostics.npz", r=backend.r, **_tensor_dict_to_numpy(final_result.fields))
        np.savez(out / "action_diagnostics.npz", **_tensor_dict_to_numpy(final_result.action_terms))
        np.savez(
            out / "dpl_orbitals.npz",
            r=backend.r,
            **_tensor_dict_to_numpy(final_result.orbitals),
            **backend.orbital_metadata,
        )
        gradient_check = backend.gradient_check(final_tensor.detach())
        write_json(out / "gradient_check.json", gradient_check)
        write_json(out / "network_physical_constraints.json", network.constraint_diagnostics(final_components))
        write_json(
            out / "discrete_metric_check.json",
            {
                "metric": "uniform-coordinate-adf",
                "radial_step_fm": backend.functional.h,
                "uniform_metric_max_error": final_result.diagnostics["uniform_metric_max_error"],
                "max_local_bilinear_error": final_result.diagnostics["max_local_bilinear_error"],
                "identity": "<psi|V|psi>_matrix = integral rho(r)V(r)d3r",
            },
        )
        write_json(
            out / "spectral_diagnostics.json",
            {
                "max_spectral_residual": final_result.diagnostics["max_spectral_residual"],
                "max_orbital_norm_residual": final_result.diagnostics["max_orbital_norm_residual"],
                "n_number_error": final_result.diagnostics["n_number_error"],
                "z_number_error": final_result.diagnostics["z_number_error"],
            },
        )
        write_json(
            out / "action_identity.json",
            {
                "complete_action_mev": float(final_result.action_terms["rmf_action"].detach().cpu()),
                "action_reduction_error_mev": final_result.diagnostics["action_reduction_error_mev"],
                "field_stationarity_rms": {
                    key.removeprefix("weak_el_").removesuffix("_rms"): value
                    for key, value in final_result.diagnostics.items()
                    if key.startswith("weak_el_") and key.endswith("_rms")
                },
            },
        )
    else:
        np.savez(out / "density_diagnostics.npz", r=backend.r, **final_result.densities)
        np.savez(out / "field_diagnostics.npz", r=backend.r, **final_result.fields)
        gradient_check = None
    write_history(out / "history.csv", rows)
    write_history(out / "physics_residuals.csv", physics_rows)
    write_history(out / "gradient_alignment.csv", gradient_rows)
    write_history(out / "channel_gradient_projection.csv", channel_gradient_rows)
    write_history(out / "component_residuals.csv", component_rows)
    if occupation_rows:
        write_history(out / "occupation_history.csv", occupation_rows)
        write_history(out / "occupation_projector_history.csv", occupation_rows)
    internal_gate_passed = bool(
        best["monitor"] <= args.gate_residual
        and final_result.diagnostics.get("max_spectral_residual", 1.0) < 1.0e-9
        and final_result.diagnostics.get("max_orbital_norm_residual", 1.0) < 1.0e-10
        and final_result.diagnostics.get("max_occupied_node_mismatch", 1.0) == 0.0
    )
    formula_invariants = {
        "hermiticity_error": final_result.diagnostics.get("hermiticity_error"),
        "action_reduction_error_mev": final_result.diagnostics.get("action_reduction_error_mev"),
        "max_orbital_norm_residual": final_result.diagnostics.get("max_orbital_norm_residual"),
        "max_spectral_residual": final_result.diagnostics.get("max_spectral_residual"),
        "max_occupied_node_mismatch": final_result.diagnostics.get("max_occupied_node_mismatch"),
        "uniform_metric_max_error": final_result.diagnostics.get("uniform_metric_max_error"),
        "max_local_bilinear_error": final_result.diagnostics.get("max_local_bilinear_error"),
        "gradient_check_relative_error": gradient_check["rel_error"] if gradient_check else None,
        "gradient_check_absolute_error": gradient_check["abs_error"] if gradient_check else None,
    }
    formula_invariants["passed"] = bool(
        formula_invariants["hermiticity_error"] < 1.0e-12
        and abs(formula_invariants["action_reduction_error_mev"]) < 1.0e-8
        and formula_invariants["max_orbital_norm_residual"] < 1.0e-10
        and formula_invariants["max_spectral_residual"] < 1.0e-9
        and formula_invariants["max_occupied_node_mismatch"] == 0.0
        and formula_invariants["uniform_metric_max_error"] < 1.0e-14
        and formula_invariants["max_local_bilinear_error"] < 1.0e-12
        and formula_invariants["gradient_check_relative_error"] < 1.0e-5
        and formula_invariants["gradient_check_absolute_error"] < 1.0e-7
    )
    write_json(out / "formula_invariants.json", formula_invariants)
    summary = {
        "mode": f"prl-ai2dft-hamiltonian-{args.mode}",
        "backend": args.backend,
        "functional": args.functional,
        "case": asdict(case),
        "nucleon_masses_mev": {
            "neutron": backend.functional.params.mass_n,
            "proton": backend.functional.params.mass_p,
        },
        "functional_type": final_result.functional_type,
        "channels": list(functional.channel_spec.names),
        "epochs": args.epochs,
        "optimizer": "Adam" if args.mode == "network" else "AdamW",
        "variation_mode": args.mode,
        "activation": args.activation,
        "hidden": args.hidden if args.mode == "network" else None,
        "network_depth": args.network_depth if args.mode == "network" else None,
        "radial_basis_size": args.radial_basis_size if args.mode == "network" else None,
        "message_neighbors": args.message_neighbors if args.mode == "network" else None,
        "trainable_parameters": sum(parameter.numel() for parameter in network.parameters()),
        "lr": initial_lr,
        "final_lr": current_lr,
        "scheduler": (
            {
                "name": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": args.scheduler_patience,
                "cooldown": args.scheduler_cooldown,
                "threshold": 0.05,
            }
            if args.mode == "network" else None
        ),
        "backtrack_count": backtrack_count,
        "lambda_reconstruct": args.lambda_reconstruct,
        "lambda_mode": lambda_mode,
        "effective_lambda_reconstruct": effective_lambda_reconstruct,
        "lambda_reconstruct_unit": "MeV^-1",
        "gradient_energy_definition": "total_no_com_mev",
        "gradient_hamiltonian_unit": "MeV",
        "energy_gradient_weight": args.energy_gradient_weight,
        "derivative_order": args.derivative_order,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "best_monitor": best["monitor"],
        "checkpoint_policy": args.checkpoint_policy,
        "selected_epoch": best["epoch"] if args.checkpoint_policy == "physics" else args.epochs,
        "internal_gate_passed": internal_gate_passed,
        "internal_physics_gate_passed": internal_gate_passed,
        "physics_gate_passed": internal_gate_passed,
        "external_scf_validation_passed": None,
        "energy_per_a_no_com": final_result.energy_per_a_no_com,
        "diagnostics": final_result.diagnostics,
        "complete_action_terms_mev": (
            {key: float(value.detach().cpu()) for key, value in final_result.action_terms.items()}
            if args.backend == "torch-rmf" else None
        ),
        "gradient_check": gradient_check,
        "formula_invariants": formula_invariants,
        "notes": [
            "The torch-rmf backend computes E[H] and grad_E_H inside one PyTorch differentiable RMF graph.",
            "Fortran is used only for optional fixed-potential and SCF comparisons.",
            "SCF data are not used as training labels.",
            "The PRL gradient is evaluated in MeV Hamiltonian coordinates from the total no-CoM energy.",
            "Reported physical S/V0/V3/VC fields are reconstructed from the final density; internal variational coordinates are saved separately.",
            "Checkpoint selection minimizes the maximum of separately normalized energy and reconstruction residuals.",
        ],
    }
    if args.compare_scf:
        raise ValueError(
            "--compare-scf is no longer performed inside training; run "
            "dpl-rmf-compare after training so SCF remains an external validation step"
        )
    summary["external_validation"] = {
        "passed": False,
        "reason": "run dpl-rmf-compare as an independent post-training validation",
    }
    write_json(out / "summary.json", summary)
    if args.mode == "direct":
        write_json(out / "direct_variation_summary.json", summary)
    torch.save({"state_dict": network.state_dict(), "summary": summary}, out / "model.pt")
    return summary
