"""Factorized full-primitive RHFB residual on an enlarged canonical NN chart.

This module provides an initialization/gradient audit only. It does not run an
optimizer, choose a no-sea branch, refresh a basis, or claim a nuclear solution.
The active chart is enlarged from an existing neural state by one QR completion
of its analytic primitive seed frame; empty added orbitals initially leave rho
and chi unchanged. The full D1S Delta action remains unprojected.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .basis import AngularRadialBasis
from .factorized_commutator import FactorizedCommutator
from .manifold import GeneralizedDensity, blocking_overlap, generalized_density_matrix
from .operators import DensityFactors
from .execution_limits import require_bounded_cgroup
from .parity_diagnostic import number_parity_from_density
from .spatial_parity import ParityConservingDensityNet, primitive_spatial_parity


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _forbid_spectra() -> tuple[dict, dict[str, int]]:
    names = ("eig", "eigh", "eigvals", "eigvalsh")
    originals, counts = {}, {}
    for module_name, module in (("torch", torch.linalg), ("numpy", np.linalg)):
        for name in names:
            key = f"{module_name}.linalg.{name}"
            originals[key] = getattr(module, name)
            counts[key] = 0
            def forbidden(*_args, _key=key, **_kwargs):
                counts[_key] += 1
                raise RuntimeError(f"spectral routine {_key} is forbidden in this prototype")
            setattr(module, name, forbidden)
    return originals, counts


def _restore_spectra(originals: dict) -> None:
    for key, value in originals.items():
        module_name, function = key.split(".linalg.", 1)
        setattr(torch.linalg if module_name == "torch" else np.linalg, function, value)


def _expanded_reference(state: dict[str, torch.Tensor], m: int,
                        dtype: torch.dtype, parity: int) -> GeneralizedDensity:
    old = state["reference_rho"].shape[0]
    rho = torch.zeros((m, m), dtype=dtype)
    chi = torch.zeros_like(rho)
    rho[:old, :old] = state["reference_rho"]
    chi[:old, :old] = state["reference_chi"]
    return GeneralizedDensity(rho, chi, generalized_density_matrix(rho, chi), parity)


def _expand_checkpoint_state(old_state: dict[str, Any], model: ParityConservingDensityNet,
                             q_seed: torch.Tensor, labels: tuple[str, ...],
                             orbital_parities: torch.Tensor) -> dict[str, Any]:
    """Embed the old canonical state in a larger chart with exact empty additions."""
    expanded = copy.deepcopy(model.state_dict())
    old_m = int(old_state["raw_a"].shape[0])
    m = int(q_seed.shape[1])
    if m <= old_m:
        raise ValueError("expanded active dimension must exceed the checkpoint dimension")
    for name, value in old_state.items():
        if name not in expanded:
            continue
        target = expanded[name]
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        if target.shape == value.shape:
            expanded[name] = value.detach().clone().to(dtype=target.dtype, device=target.device)
        elif name == "canonical_number_transport":
            # Preserve particle/hole block ordering when adding empty modes.
            target[:old_m, :old_m] = value[:old_m, :old_m]
            target[:old_m, m:m+old_m] = value[:old_m, old_m:]
            target[m:m+old_m, :old_m] = value[old_m:, :old_m]
            target[m:m+old_m, m:m+old_m] = value[old_m:, old_m:]
            expanded[name] = target
        elif name == "raw_coordinates" and value.shape[0] == target.shape[0]:
            target[:, :old_m] = value
            expanded[name] = target
        elif name in {"raw_a", "raw_b"}:
            target[:old_m, :old_m] = value
            expanded[name] = target
        elif name in {"reference_rho", "reference_chi"}:
            target[:old_m, :old_m] = value
            expanded[name] = target
        elif name == "reference_matrix":
            block = generalized_density_matrix(expanded["reference_rho"], expanded["reference_chi"])
            expanded[name] = block
        elif name == "initial_q":
            expanded[name] = q_seed.detach().clone()
        elif name == "orbital_parities":
            expanded[name] = orbital_parities.to(dtype=target.dtype, device=target.device)
        elif name == "state_embedding.weight":
            target[:old_m] = value
            expanded[name] = target
    # This non-tensor metadata is part of the new chart and must describe M.
    extra = model.checkpoint_metadata()
    extra["labels"] = list(labels)
    expanded["_extra_state"] = extra
    return expanded


def build_enlarged_o17_model(
    checkpoint_path: str | Path,
    density_path: str | Path,
    *,
    species: str,
    active_dimension: int = 42,
    seed: int = 20261006,
    device: str | torch.device = 'cpu',
) -> tuple[AngularRadialBasis, ParityConservingDensityNet, dict[str, Any]]:
    """Load one saved O17 NN species and extend it without a spectral seed."""
    require_bounded_cgroup()
    checkpoint_path, density_path = Path(checkpoint_path), Path(density_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("physical_resume_compatible"):
        raise ValueError("expected the archived experimental state, not a certified continuation")
    config = payload["config"]
    if species not in {"n", "p"}:
        raise ValueError("species must be 'n' or 'p'")
    target_number = int(config['n' if species == 'n' else 'z'])
    minimum_active = max(40, target_number + 32)
    if active_dimension < minimum_active:
        raise ValueError(f'{species} active_dimension={active_dimension} is below the required {minimum_active}')
    old_state = payload["network_state"][species]
    old_meta = old_state["_extra_state"]
    old_m = int(old_meta["active_dimension"])
    d = int(old_state["initial_q"].shape[0])
    if active_dimension <= old_m or active_dimension > d:
        raise ValueError("active_dimension must be between old M and primitive D")
    old_parity = int(old_meta["fermion_parity"])
    blocked_index = old_meta.get("blocked_index")
    if species == "n" and (old_parity != -1 or blocked_index != 13):
        raise ValueError("unexpected saved O17 odd-neutron block metadata")
    if species == "p" and (old_parity != 1 or blocked_index is not None):
        raise ValueError("unexpected saved O17 proton parity/block metadata")

    basis = AngularRadialBasis(radius_fm=float(config["radius_fm"]),
                               spacing_fm=float(config["spacing_fm"]),
                               kappa_max=int(config["kappa_max"]),
                               derivative_order=5)
    if basis.dimension != d:
        raise ValueError("checkpoint primitive dimension disagrees with reconstructed grid")
    labels_old = tuple(old_meta["labels"])
    parity = primitive_spatial_parity(basis)
    q_old_initial = old_state["initial_q"].to(torch.complex128)
    columns = (q_old_initial.conj() * (parity[:, None] * q_old_initial)).sum(0).real
    signs_old = torch.where(columns >= 0, 1., -1.).to(torch.float64)
    old_model = ParityConservingDensityNet(
        basis, q_old_initial, labels_old, GeneralizedDensity(
            old_state["reference_rho"], old_state["reference_chi"],
            old_state["reference_matrix"], old_parity),
        blocked_index=blocked_index, hidden=int(old_meta["hidden"]),
        depth=int(old_meta["depth"]), radial_basis_size=int(old_meta["radial_basis_size"]),
        correction_scale=float(old_meta["correction_scale"]),
        pairing_seed=float(old_meta["pairing_seed"]),
        channel_chunk_size=int(old_meta["channel_chunk_size"]),
    ).to(device="cpu")
    seed_state_upgraded = "canonical_number_transport" not in old_state
    if seed_state_upgraded:
        from .network import upgrade_legacy_seed_state
        # This function is explicitly an archived NN seed conversion, never
        # an optimizer/checkpoint resume. Keep the archived file untouched.
        old_state = upgrade_legacy_seed_state(old_model, old_state)
    old_model.load_state_dict(old_state, strict=True)
    old_model.eval()
    with torch.no_grad():
        q_old, gd_old = old_model()
        t_old = old_model._canonical_transform()

    # One analytic QR completion of the saved non-spectral seed frame.
    # Complete each parity sector separately so Qperp is definite-parity.
    # Never replace/phase-rotate any of the checkpoint's active Q columns.
    remaining = active_dimension - old_m
    selected_perp = []
    add_counts = {}
    for sign in (1, -1):
        sector_rows = torch.nonzero(parity == sign).flatten()
        seed_cols = torch.nonzero(signs_old == sign).flatten()
        needed = remaining // 2 + (1 if sign == 1 and remaining % 2 else 0)
        add_counts[sign] = needed
        if not needed:
            selected_perp.append(q_old_initial.new_zeros((d, 0)))
            continue
        if sector_rows.numel() <= seed_cols.numel() + needed:
            raise ValueError(f"parity={sign:+d} primitive sector is too small for {needed} empty columns")
        q_sector = q_old_initial[sector_rows[:, None], seed_cols[None, :]]
        q_sector_complete, _ = torch.linalg.qr(q_sector, mode="complete")
        q_sector_perp = q_sector_complete[:, seed_cols.numel():seed_cols.numel()+needed]
        full_sector = q_old_initial.new_zeros((d, needed))
        full_sector[sector_rows] = q_sector_perp
        selected_perp.append(full_sector)
    q_seed = torch.cat((q_old_initial, *selected_perp), dim=1).contiguous()
    q_seed_parity = (q_seed.conj() * (parity[:, None] * q_seed)).sum(0).real
    signs = torch.where(q_seed_parity >= 0, 1., -1.).to(torch.float64)
    parity_error = float((parity[:, None] * q_seed - q_seed * signs[None, :]).abs().max())
    if parity_error > 2.0e-10:
        raise ValueError("analytic QR completion did not preserve definite spatial parity")
    labels = labels_old + tuple(f"analytic_qr_empty_{j:02d}_parity_{int(signs[j]):+d}"
                                for j in range(old_m, active_dimension))
    ref = _expanded_reference(old_state, active_dimension, torch.complex128, old_parity)

    # New random embeddings are deterministic. The old chart and its actual NN
    # weights are copied exactly; the appended orbitals begin empty and uncoupled.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ParityConservingDensityNet(
            basis, q_seed, labels, ref, blocked_index=blocked_index,
            hidden=int(old_meta["hidden"]), depth=int(old_meta["depth"]),
            radial_basis_size=int(old_meta["radial_basis_size"]),
            correction_scale=float(old_meta["correction_scale"]), pairing_seed=0.0,
            channel_chunk_size=int(old_meta["channel_chunk_size"]),
        ).to(device="cpu")
    state = _expand_checkpoint_state(old_state, model, q_seed, labels, signs)
    model.load_state_dict(state, strict=True)
    model.train()

    with torch.no_grad():
        q_new, gd_new = model()
        t_new = model._canonical_transform()
        _, _, mode_old = old_model.forward_with_blocked_mode()
        _, _, mode_new = model.forward_with_blocked_mode()
        rho_old_primitive = q_old @ gd_old.rho @ q_old.mH
        chi_old_primitive = q_old @ gd_old.chi @ q_old.mT
        rho_new_primitive = q_new @ gd_new.rho @ q_new.mH
        chi_new_primitive = q_new @ gd_new.chi @ q_new.mT
    rho_difference = float(torch.linalg.matrix_norm(rho_new_primitive-rho_old_primitive))
    chi_difference = float(torch.linalg.matrix_norm(chi_new_primitive-chi_old_primitive))
    if max(rho_difference, chi_difference) > 5.0e-9:
        raise RuntimeError(f"M{active_dimension} empty-orbital expansion changed the archived primitive density")
    old_nambu_indices = torch.cat((torch.arange(old_m),
                                   torch.arange(active_dimension, active_dimension + old_m)))
    canonical_difference = float(torch.linalg.matrix_norm(
        t_new[old_nambu_indices[:, None], old_nambu_indices[None, :]] - t_old))
    if canonical_difference > 2.0e-9:
        raise RuntimeError(f"M{active_dimension} chart does not retain the archived active canonical transform: {canonical_difference:.6e}")
    mode_overlap = None
    if blocked_index is not None:
        mode_overlap = float(blocking_overlap(mode_old, mode_new))
        if mode_overlap < 1.0 - 2.0e-8:
            raise RuntimeError(f"expanded chart did not preserve the seed blocked mode: overlap={mode_overlap:.12g}")

    arrays = np.load(density_path)
    rho_source = torch.as_tensor(arrays[f"{species}_Q"] @ arrays[f"{species}_rho_active"]
                                 @ arrays[f"{species}_Q"].conj().T)
    chi_source = torch.as_tensor(arrays[f"{species}_Q"] @ arrays[f"{species}_chi_active"]
                                 @ arrays[f"{species}_Q"].T)
    source_rho_diff = float(torch.linalg.matrix_norm(rho_old_primitive-rho_source))
    source_chi_diff = float(torch.linalg.matrix_norm(chi_old_primitive-chi_source))
    if max(source_rho_diff, source_chi_diff) > 5.0e-8:
        raise RuntimeError(f"checkpoint forward does not reproduce the frozen O17 density source: rho={source_rho_diff:.6e}, chi={source_chi_diff:.6e}")

    metadata = {
        "primitive_dimension": d, "checkpoint_active_dimension": old_m,
        "explicit_legacy_seed_transport_upgrade": seed_state_upgraded,
        "prototype_active_dimension": active_dimension,
        "species": species,
        "blocked_index": blocked_index, "fermion_parity": old_parity,
        "seed_source": f"archived NN experimental_network.pt {species} active Q/canonical coordinates; no spectral teacher",
        "completion": "one parity-sector torch.linalg.qr(mode='complete') for each required sector; retain original active columns exactly, append analytic empty complement columns; no refresh thereafter",
        "completion_parity_counts": {"positive": int((signs > 0).sum()),
                                    "negative": int((signs < 0).sum()),
                                    "added_positive": add_counts[1],
                                    "added_negative": add_counts[-1]},
        "completion_parity_max_error": parity_error,
        "primitive_rho_expansion_difference_frobenius": rho_difference,
        "primitive_chi_expansion_difference_frobenius": chi_difference,
        "canonical_T14_reproduction_difference_frobenius": canonical_difference,
        "initial_blocked_mode_overlap_checkpoint_to_expanded": mode_overlap,
        "checkpoint_vs_fixed_density_rho_difference_frobenius": source_rho_diff,
        "checkpoint_vs_fixed_density_chi_difference_frobenius": source_chi_diff,
        "initial_active_density_rank_limit": f"M={active_dimension} chart can only represent rank-{active_dimension} rho/chi factors at each iterate; full primitive D1S Delta still acts on all D rows/columns",
        "basis_completion_calls_per_species": sum(count > 0 for count in add_counts.values()),
        "labels": list(labels),
    }
    target_device = torch.device(device)
    if target_device.type not in {'cpu', 'cuda'}:
        raise ValueError('only CPU or CUDA devices are supported')
    if target_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device is unavailable; CPU fallback is refused')
    model = model.to(device=target_device)
    metadata['initialization_validation_device'] = 'cpu'
    metadata['optimization_device'] = str(target_device)
    return basis, model, metadata


def initial_gradient_audit(
    basis: AngularRadialBasis,
    model: ParityConservingDensityNet,
    h: torch.Tensor,
    delta: torch.Tensor,
    particle_number: int,
    *,
    chunk_size: int = 32,
    number_weight: float = 1.0e4,
) -> dict[str, Any]:
    """Run forward/backward once to verify real NN gradients; take no optimizer step."""
    d = basis.dimension
    if h.shape != (d, d) or delta.shape != (d, d):
        raise ValueError("frozen O17 h and Delta must be full D by D arrays")
    if not torch.allclose(h.mH, h, atol=1e-8, rtol=0):
        raise ValueError("normal h is not Hermitian")
    if not torch.allclose(delta.mT, -delta, atol=1e-8, rtol=0):
        raise ValueError("D1S Delta is not antisymmetric")
    model.zero_grad(set_to_none=True)
    q, gd = model()
    factors = DensityFactors(q, gd.rho, gd.chi)
    hq = h @ q
    action = FactorizedCommutator(
        factors, hq, lambda values: delta @ values,
        chemical_potential=torch.zeros((), dtype=torch.float64, device=h.device),
    )
    terms = action.frobenius_squared(chunk_size=chunk_size)
    residual = terms["full_squared_mev2"] / float((2*d)**2)
    number_error = factors.particle_number() - float(particle_number)
    number_term = number_weight * number_error.square()
    objective = residual + number_term
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("nonfinite factorized full-primitive residual objective")
    objective.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradients[name] = float(torch.linalg.vector_norm(parameter.grad.detach()))
    groups: dict[str, float] = {}
    for name, norm in gradients.items():
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0.0) + norm*norm
    groups = {name: value**0.5 for name, value in groups.items()}
    parity = number_parity_from_density(gd.rho.detach().cpu().numpy(),
                                        gd.chi.detach().cpu().numpy())
    return {
        "objective_mev2_scaled": float(objective.detach()),
        "full_commutator_frobenius_squared_mev2": float(terms["full_squared_mev2"].detach()),
        "normal_commutator_squared_mev2": float(terms["normal_squared_mev2"].detach()),
        "anomalous_commutator_squared_mev2": float(terms["anomalous_squared_mev2"].detach()),
        "element_mean_square_residual_mev2": float(residual.detach()),
        "number": float(factors.particle_number().detach()),
        "particle_number_target": int(particle_number),
        "number_error": float(number_error.detach()),
        "number_penalty_weight": number_weight,
        "number_penalty_term": float(number_term.detach()),
        "pfaffian_parity": int(parity["number_parity"]),
        "gradient_norm_by_layer": gradients,
        "gradient_norm_by_group": groups,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_updates": 0,
        "active_dimension": int(q.shape[1]),
        "factorized_full_primitive_residual_chunk_size": chunk_size,
        "objective_scope": "fixed h and full unprojected Delta; raw full commutator residual and number penalty logged separately; no energy minimization",
    }


def audit_o17(
    *,
    checkpoint: str | Path,
    density: str | Path,
    field_dir: str | Path,
    pairing_field: str | Path,
    output_dir: str | Path,
    active_dimension: int = 42,
    chunk_size: int = 32,
) -> dict[str, Any]:
    """Save reproducible O17 M42 n/p states plus full-residual AD audits."""
    require_bounded_cgroup()
    torch.set_num_threads(2)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snap = output / "run_start_sources"
    snap.mkdir(exist_ok=True)
    field_dir, pairing_field = Path(field_dir), Path(pairing_field)
    normal_fields = {s: field_dir / f"h_{s}_columns.npy" for s in ("n", "p")}
    source_files = [Path(__file__).resolve(), Path(__file__).with_name("factorized_commutator.py"),
                    Path(__file__).with_name("operators.py"), Path(__file__).with_name("network.py"),
                    Path(__file__).with_name("manifold.py"), Path(__file__).with_name("spatial_parity.py"),
                    Path(__file__).with_name("basis.py"), Path(__file__).with_name("parity_diagnostic.py"),
                    Path(checkpoint), Path(density), *normal_fields.values(), pairing_field]
    source_hashes = {}
    for path in source_files:
        target = snap / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        source_hashes[str(path)] = _sha256(target)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    deltas = np.load(pairing_field)
    species_results = {}
    spectral_attempts_all = {}
    for species in ("n", "p"):
        basis, model, init_metadata = build_enlarged_o17_model(
            checkpoint, density, species=species, active_dimension=active_dimension)
        h = torch.as_tensor(np.array(np.load(normal_fields[species], mmap_mode="r"), copy=True),
                            dtype=torch.complex128)
        delta = torch.as_tensor(deltas[f"{species}_delta"], dtype=torch.complex128)
        originals, spectral_attempts = _forbid_spectra()
        try:
            gradient = initial_gradient_audit(
                basis, model, h, delta,
                int(payload["config"]["n" if species == "n" else "z"]),
                chunk_size=chunk_size)
        finally:
            _restore_spectra(originals)
        spectral_attempts_all[species] = spectral_attempts
        checkpoint_out = output / f"{species}_m{active_dimension}_initial_neural_state.pt"
        torch.save({"model": model.state_dict(), "metadata": init_metadata,
                    "optimizer_updates": 0, "optimizer_created": False}, checkpoint_out)
        species_results[species] = {
            "initialization": init_metadata,
            "gradient_audit": gradient,
            "checkpoint_file": checkpoint_out.name,
            "checkpoint_file_sha256": _sha256(checkpoint_out),
        }
    result = {
        "status": f"o17_m{active_dimension}_full_branch_initial_gradient_audit_only",
        "physical_result_accepted": False,
        "branch_certification_status": "pending_no_independent_same_branch_reference",
        "branch_certification_pass": False,
        "scope": f"actual archived O17 neutron/proton NN states, full frozen PKA1 h and full D1S Delta, M{active_dimension} factorized canonical charts; no optimizer update",
        "source_hashes": source_hashes,
        "source_density_sha256": _sha256(Path(density)),
        "normal_field_sha256": {s: _sha256(path) for s, path in normal_fields.items()},
        "pairing_field_sha256": _sha256(pairing_field),
        "checkpoint_sha256": _sha256(Path(checkpoint)),
        "rational_or_full_matrix_eigensolver_attempts": spectral_attempts_all,
        "spectral_teacher_or_initialization": False,
        "basis_refreshes_after_initial_qr": 0,
        "basis_completion_calls": sum(info['initialization']['basis_completion_calls_per_species'] for info in species_results.values()),
        "qperp_source": "analytic seed-frame complete QR; no spectral information",
        "production_physics_changed": False,
        "fortran_used": False,
        "field_freezing": {"h_frozen": True, "delta_frozen": True,
                           "pairing_delta_projected_to_Q": False,
                           "delta_action_uses_full_D_by_D_matrix": True},
        "species": species_results,
        "reference_limit": f"M{active_dimension} chart may not contain all directions induced by the saved full D1S Delta; report residual honestly, no exact small-residual claim without rank/projection bound and branch reference",
    }
    (output / "initial_gradient_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--field-dir", required=True)
    parser.add_argument("--pairing-field", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--active-dimension", type=int, default=42)
    args = parser.parse_args()
    report = audit_o17(checkpoint=args.checkpoint, density=args.density,
                               field_dir=args.field_dir, pairing_field=args.pairing_field,
                               output_dir=args.output_dir, active_dimension=args.active_dimension)
    print(json.dumps({"status": report["status"], "species": report["species"]}, indent=2))
