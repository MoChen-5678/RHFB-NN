"""Protected full-space O17 N=9,Z=8 self-consistent NN run driver.

The saved experimental network is only an initialization.  Every optimizer
evaluation is delegated to ``train_joint`` and rebuilds the full PKA1 and D1S
fields.  This driver records solver outputs; it does not identify a ground
state or certify a physical no-sea branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import torch

from .network import GeneralizedDensityNet


HERE = Path(__file__).resolve().parent
VARIANT_ROOT = HERE.parents[1]
DEFAULT_CHECKPOINT = (VARIANT_ROOT / "outputs/demo/O17_neural_parity_candidate_20261006"
                      / "experimental_network.pt")
DEFAULT_DENSITY = (VARIANT_ROOT / "outputs/demo/O17_neural_parity_candidate_20261006"
                   / "density_factors.npz")
DEFAULT_REFERENCE_ROOT = (VARIANT_ROOT.parents[2] / "dpl-param/reference_fortran/user-rhfb-20261015")


def _source_parameter_audits(reference_source_dirs=None):
    """Compare reference parameter tables, without inheriting their symmetry.

    Radial blocking indices, source N/Z and DWS energy windows are provenance,
    not constraints on the unrestricted NN. An unspecified interactive source
    multiplier is reported as unknown; it never overwrites the declared D1S=1.
    """
    from .source_template import audit_payload
    if reference_source_dirs is None:
        reference_source_dirs = [DEFAULT_REFERENCE_ROOT / relative for relative in
                                 ("RHFB-BI", "RHFB-BI/Broyden", "S-RHFB-C")
                                 if (DEFAULT_REFERENCE_ROOT / relative).is_dir()]
    records = []
    for directory in reference_source_dirs:
        record = audit_payload(directory)
        mapping = record["python_parameter_mapping"]
        if not mapping["pka1_matches"]:
            raise ValueError(f"reference PKA1 constants disagree with the NN: {directory}")
        source_pair = mapping["d1s_active_source_to_python"]
        python_pair = mapping["d1s_python"]
        for key, value in source_pair.items():
            if key == "strength" and value is None:
                continue
            if value != python_pair[key]:
                raise ValueError(f"reference central D1S {key} disagrees with the NN: {directory}")
        record["scope"] = "parameter-table comparison only; source symmetry, occupations and no-sea algorithm are not inherited"
        records.append(record)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value)
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _ph_progress_callback(output: Path):
    """Write PH traversal progress independently of trainer progress.json."""
    history_path = output / "ph_progress.jsonl"
    latest_path = output / "latestPHprogress.json"

    def report(row):
        payload = {**_jsonable(row), "updated_unix_time": time.time()}
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_json(latest_path, payload)

    return report


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".npz",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        import numpy as np
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_snapshot(output: Path, checkpoint: Path, density: Path) -> dict[str, str]:
    """Copy/hash code and seed inputs so a run is tied to exact source bytes."""
    relative_sources = (
        "dpl_rhf/rhfb/o17_selfconsistent_neural.py",
        "dpl_rhf/rhfb/full_branch_neural_prototype.py",
        "dpl_rhf/rhfb/selfconsistent_neural_training.py",
        "dpl_rhf/rhfb/selfconsistent_stationarity.py",
        "dpl_rhf/rhfb/number_constraint.py",
        "dpl_rhf/rhfb/number_retraction.py",
        "dpl_rhf/rhfb/network_number_transport.py",
        "dpl_rhf/rhfb/mixed_nucleon_sea_branch.py",
        "dpl_rhf/rhfb/full_branch_stationarity.py",
        "dpl_rhf/rhfb/bdg_branch_filter.py",
        "dpl_rhf/rhfb/no_sea.py",
        "dpl_rhf/rhfb/center_of_mass.py",
        "dpl_rhf/rhfb/cm_momentum_adf.py",
        "dpl_rhf/rhfb/prescription.py",
        "dpl_rhf/rhfb/source_template.py",
        "dpl_rhf/rhfb/bogoliubov_blocking.py",
        "dpl_rhf/rhfb/hfb_gradient.py",
        "dpl_rhf/rhfb/streamed_normal_functional.py",
        "dpl_rhf/rhfb/factorized_commutator.py",
        "dpl_rhf/rhfb/blocking_tracking.py",
        "dpl_rhf/rhfb/manifold.py",
        "dpl_rhf/rhfb/operators.py",
        "dpl_rhf/rhfb/spatial.py",
        "dpl_rhf/rhfb/contact.py",
        "dpl_rhf/rhfb/functional.py",
        "dpl_rhf/rhfb/streamed_field_autograd.py",
        "dpl_rhf/rhfb/gaussian_pairing.py",
        "dpl_rhf/rhfb/pairing_action.py",
        "dpl_rhf/rhfb/pairing.py",
        "dpl_rhf/rhfb/currents.py",
        "dpl_rhf/rhfb/basis.py",
        "dpl_rhf/rhfb/network.py",
        "dpl_rhf/rhfb/spatial_parity.py",
        "dpl_rhf/rhfb/parity_diagnostic.py",
        "dpl_rhf/rhfb/cuda_resources.py",
        "dpl_rhf/rhfb/execution_limits.py",
        "dpl_rhf/rhf/parameters.py",
        "dpl_rhf/rhf/dirac.py",
    )
    snapshot = output / "run_start_sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    files = [VARIANT_ROOT / item for item in relative_sources] + [checkpoint, density]
    result = {}
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(f"required run source/input is missing: {source}")
        if source.is_relative_to(VARIANT_ROOT):
            target = snapshot / source.relative_to(VARIANT_ROOT)
        else:
            target = snapshot / "seed_inputs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        result[str(source)] = _sha256(target)
    return result


def _scalar(value) -> float:
    return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)


class UnrestrictedO17DensityNet(GeneralizedDensityNet):
    """GeneralizedDensityNet with no spatial-parity mask."""

    def _unrestricted_orbitals(self):
        correction = self.raw_coordinates + self._network_correction()
        candidate = (self.initial_q
                     + self.correction_scale * self.envelope[:, None] * correction)
        q, r = torch.linalg.qr(candidate, mode="reduced")
        diagonal = torch.diagonal(r)
        magnitude = diagonal.abs()
        phase = torch.where(magnitude > torch.finfo(magnitude.dtype).tiny,
                            diagonal / magnitude.clamp_min(torch.finfo(magnitude.dtype).tiny),
                            torch.ones_like(diagonal))
        return q * phase[None, :]

    def forward(self):
        q = self._unrestricted_orbitals()
        return q, self._canonical_density()

    def forward_with_blocked_mode(self):
        q, gd, state = self.forward_with_bogoliubov_frame()
        mode = None
        if state.blocked_mode is not None:
            m = self.active_dimension
            active_mode = state.blocked_mode
            mode = torch.cat((q @ active_mode[:m], q.conj() @ active_mode[m:]))
        return q, gd, mode

    def checkpoint_metadata(self):
        return {
            **super().checkpoint_metadata(),
            "format": "dpl-rhfb-unrestricted-o17-density-net-v2",
            "spatial_symmetry_constraints": [],
            "retained_structure": [
                "fermion-number parity from the pure HFB reference",
                "finite-box and component-origin envelopes of the primitive basis",
            ],
        }


def _convert_parity_seed_to_unrestricted(basis, seed_model, *, device: str):
    """Reparameterize one seed state without retaining its spatial mask."""
    import torch
    from .manifold import GeneralizedDensity

    with torch.no_grad():
        q_seed, gd_seed, mode_seed = seed_model.forward_with_blocked_mode()
        q_seed = q_seed.detach().clone()
        reference = GeneralizedDensity(
            seed_model.reference_rho.detach().clone(),
            seed_model.reference_chi.detach().clone(),
            seed_model.reference_matrix.detach().clone(),
            int(seed_model.reference_parity),
        )
        masked_a = seed_model.raw_a.detach() * seed_model._active_parity_mask()
        masked_b = seed_model.raw_b.detach() * seed_model._active_parity_mask()

    model = UnrestrictedO17DensityNet(
        basis, q_seed, seed_model.initial_q_labels, reference,
        blocked_index=seed_model.blocked_index,
        hidden=seed_model.hidden_size, depth=seed_model.network_depth,
        radial_basis_size=seed_model.radial_basis_size,
        correction_scale=seed_model.correction_scale,
        pairing_seed=seed_model.pairing_seed,
        channel_chunk_size=seed_model.channel_chunk_size,
    ).to(device=device)

    # Preserve the same NN feature map while zeroing the orbital correction
    # head.  The seed's Q is the new reference frame, so zero correction makes
    # its initial physical density exactly reproducible in unrestricted form.
    with torch.no_grad():
        for name in ("radial_encoder", "kappa_embedding", "m_embedding",
                     "component_embedding", "state_embedding"):
            getattr(model, name).load_state_dict(getattr(seed_model, name).state_dict())
        model.output_head.weight.zero_()
        model.output_head.bias.zero_()
        model.raw_coordinates.zero_()
        model.raw_a.copy_(masked_a.to(device=model.raw_a.device))
        model.raw_b.copy_(masked_b.to(device=model.raw_b.device))
        from .network_number_transport import restore_number_transport
        restore_number_transport(model, seed_model.canonical_number_transport.to(device=device))

        q_new, gd_new, mode_new = model.forward_with_blocked_mode()
        q_seed_on_device = q_seed.to(device=q_new.device)
        rho_seed = (q_seed_on_device @ gd_seed.rho.to(device=q_new.device)
                    @ q_seed_on_device.mH)
        chi_seed = (q_seed_on_device @ gd_seed.chi.to(device=q_new.device)
                    @ q_seed_on_device.mT)
        rho_new = q_new @ gd_new.rho @ q_new.mH
        chi_new = q_new @ gd_new.chi @ q_new.mT
        rho_error = float(torch.linalg.matrix_norm(rho_new - rho_seed))
        chi_error = float(torch.linalg.matrix_norm(chi_new - chi_seed))
        q_error = float(torch.linalg.matrix_norm(q_new - q_seed_on_device))
        seed_transform = seed_model._canonical_transform().to(device=q_new.device)
        canonical_error = float(torch.linalg.matrix_norm(
            model._canonical_transform() - seed_transform))
        mode_overlap = (None if mode_seed is None else
                        float(torch.abs(torch.vdot(mode_seed.to(device=mode_new.device), mode_new))))
    if max(rho_error, chi_error, q_error, canonical_error) > 1.0e-9:
        raise RuntimeError(
            "unrestricted reparameterization changed the NN seed state: "
            f"Q={q_error:.3e}, rho={rho_error:.3e}, chi={chi_error:.3e}, "
            f"canonical_transform={canonical_error:.3e}")
    if mode_seed is not None and (mode_overlap is None or mode_overlap < 1.0 - 1.0e-9):
        raise RuntimeError(f"unrestricted reparameterization changed blocked mode: overlap={mode_overlap}")
    model._run_initial_blocked_mode = None if mode_new is None else mode_new.detach().clone()
    return model, {
        "conversion": "parity-conserving seed decoded, then copied to unrestricted GeneralizedDensityNet subclass",
        "spatial_parity_constraint_retained": False,
        "initial_q_difference_frobenius": q_error,
        "primitive_rho_difference_frobenius": rho_error,
        "primitive_chi_difference_frobenius": chi_error,
        "canonical_transform_difference_frobenius": canonical_error,
        "blocked_mode_overlap_seed_to_unrestricted": mode_overlap,
        "validation_scope": "same archived NN state reparameterized; not a Hamiltonian solve, teacher, or basis refresh",
    }


def _callback_factory(basis, models, output: Path, *, sample_chunk_size: int = 64):
    """Create exports from the trainer's already computed accepted evaluation."""
    import numpy as np
    import torch
    from .currents import local_bilinears
    from .manifold import blocking_overlap
    from .hfb_gradient import project_fields_number_tangent

    exports = output / "accepted_evaluations"
    exports.mkdir(parents=True, exist_ok=True)

    def callback(step, result, states, outputs, info):
        event = str(info.get("event", "accepted_evaluation"))
        record: dict[str, Any] = {
            "step": int(step), "event": event, "phase": info.get("phase"),
            "physical_result_accepted": False,
            "no_sea_branch_certified": False,
            "ground_state_claim": False,
            "chemical_potentials_mev": info.get("chemical_potentials_mev"),
            "number_constraint_duals": info.get("number_constraint_duals"),
            "number_constraint_dual_units": "MeV^2; residual-objective dual, not chemical potential",
            "blocked_metrics": info.get("blocked_metrics"),
            "energy_mev": {
                "total": _scalar(result.total_energy),
                "particle_hole": _scalar(result.particle_hole_energy),
                "d1s_pairing": _scalar(result.pairing_energy),
                "channels": _jsonable(result.energy_channels),
            },
            "species": {},
            "scope": "stationarity optimization output; not a selected energy minimum",
        }
        arrays: dict[str, Any] = {}
        weights = torch.as_tensor(basis.weights, dtype=torch.float64)
        positions = torch.as_tensor(basis.positions, dtype=torch.float64)
        radii = torch.linalg.vector_norm(positions, dim=1)
        q20_kernel = 2.0 * positions[:, 2].square() - positions[:, 0].square() - positions[:, 1].square()

        for species in ("n", "p"):
            q, gd = outputs[species]
            state = states[species]
            q_cpu = q.detach().cpu().numpy()
            rho_cpu = gd.rho.detach().cpu().numpy()
            chi_cpu = gd.chi.detach().cpu().numpy()
            rho_primitive = q @ gd.rho @ q.mH
            chi_primitive = q @ gd.chi @ q.mT
            h = result.normal_fields[species]
            gap = result.active_pairing_gaps[species]
            delta_action = result.delta_actions[species]
            with torch.no_grad():
                frame_q, frame_density, uv_state = models[species].forward_with_bogoliubov_frame()
                if (not torch.allclose(frame_q, q, atol=1.e-10, rtol=0.)
                        or not torch.allclose(frame_density.matrix, gd.matrix, atol=1.e-10, rtol=0.)):
                    raise RuntimeError("exported U/V frame differs from the accepted NN density")
                active_gradient = project_fields_number_tangent(
                    uv_state.u, uv_state.v, q.mH @ h @ q, gap,
                    fallback_mu=result.species_diagnostics[species]["chemical_potential_mev"],
                )

            # Three deterministic columns probe Delta outside the active Q
            # chart without materializing the full D x D pairing matrix.
            d = basis.dimension
            probes = torch.zeros((d, 3), dtype=torch.complex128, device=q.device)
            probes[0, 0] = 1.0
            probes[d // 2, 1] = 1.0j
            indices = torch.arange(d, dtype=torch.float64, device=q.device)
            probes[:, 2] = torch.complex(torch.sin(0.017 * (indices + 1)),
                                         torch.cos(0.013 * (indices + 2)))
            probes[:, 2] /= torch.linalg.vector_norm(probes[:, 2]).clamp_min(1e-30)
            delta_probe = delta_action.apply(probes)

            # Stream local bilinears to bound peak sample storage.  The local
            # current is kept at quadrature points for transparent reanalysis.
            current_parts, scalar_parts, axial_parts, tensor_parts = [], [], [], []
            number = radius2 = q20 = 0.0
            for p0 in range(0, basis.point_count, sample_chunk_size):
                p1 = min(basis.point_count, p0 + sample_chunk_size)
                sampled = basis._sample_range(q, p0, p1)
                bilinears = local_bilinears(sampled, gd.rho)
                baryon = bilinears["baryon_current"].detach().cpu()
                scalar = bilinears["scalar"].detach().cpu()
                axial = bilinears["axial_current"].detach().cpu()
                tensor = bilinears["tensor"].detach().cpu()
                current_parts.append(baryon.numpy())
                scalar_parts.append(scalar.numpy())
                axial_parts.append(axial.numpy())
                tensor_parts.append(tensor.numpy())
                w = weights[p0:p1]
                rr = radii[p0:p1]
                qq = q20_kernel[p0:p1]
                number += float(torch.sum(w * baryon[:, 0]))
                radius2 += float(torch.sum(w * baryon[:, 0] * rr.square()))
                q20 += float(torch.sum(w * baryon[:, 0] * qq))

            prefix = species + "_"
            arrays[prefix + "Q"] = q_cpu
            arrays[prefix + "rho_active"] = rho_cpu
            arrays[prefix + "chi_active"] = chi_cpu
            arrays[prefix + "rho_primitive"] = rho_primitive.detach().cpu().numpy()
            arrays[prefix + "chi_primitive"] = chi_primitive.detach().cpu().numpy()
            arrays[prefix + "h_primitive"] = h.detach().cpu().numpy()
            arrays[prefix + "active_pairing_gap"] = gap.detach().cpu().numpy()
            arrays[prefix + "bogoliubov_U_active"] = uv_state.u.detach().cpu().numpy()
            arrays[prefix + "bogoliubov_V_active"] = uv_state.v.detach().cpu().numpy()
            arrays[prefix + "H20_active"] = active_gradient.h20.detach().cpu().numpy()
            arrays[prefix + "N20_active"] = active_gradient.n20.detach().cpu().numpy()
            arrays[prefix + "constrained_H20_active"] = active_gradient.constrained_h20.detach().cpu().numpy()
            arrays[prefix + "delta_action_probe_vectors"] = probes.detach().cpu().numpy()
            arrays[prefix + "delta_action_probe_results"] = delta_probe.detach().cpu().numpy()
            arrays[prefix + "baryon_current_points"] = np.concatenate(current_parts, axis=0)
            arrays[prefix + "scalar_density_points"] = np.concatenate(scalar_parts, axis=0)
            arrays[prefix + "axial_current_points"] = np.concatenate(axial_parts, axis=0)
            arrays[prefix + "tensor_density_points"] = np.concatenate(tensor_parts, axis=0)

            diagnostic = result.species_diagnostics[species]
            parity = int(models[species].reference_parity)
            blocked_index = getattr(models[species], "blocked_index", None)
            mode_overlap = None
            mode = None
            if blocked_index is not None:
                with torch.no_grad():
                    _, _, mode = models[species].forward_with_blocked_mode()
                mode_cpu = mode.detach().cpu()
                arrays[prefix + "blocked_mode"] = mode_cpu.numpy()
                initial_mode = getattr(models[species], "_run_initial_blocked_mode", None)
                if initial_mode is not None:
                    mode_overlap = float(blocking_overlap(initial_mode, mode).detach().cpu())
            record["species"][species] = {
                "target_number": 9 if species == "n" else 8,
                "particle_number_from_density": _scalar(torch.trace(gd.rho).real),
                "particle_number_from_coordinate_current": number,
                "number_error": _scalar(diagnostic["number_error"]),
                "chemical_potential_mev": _scalar(diagnostic["chemical_potential_mev"]),
                "number_tangent_identifiable": bool(diagnostic["number_tangent_identifiable"]),
                "number_variance": _scalar(diagnostic["number_variance"]),
                "number_variance_rho_identity_error": _scalar(diagnostic["number_variance_rho_identity_error"]),
                "active_H20_frobenius_mev": _scalar(torch.linalg.vector_norm(active_gradient.h20)),
                "active_constrained_H20_frobenius_mev": _scalar(torch.linalg.vector_norm(active_gradient.constrained_h20)),
                "active_H20_scope": "active-space energy gradient; full primitive stationarity still uses the untruncated commutator",
                "number_parity": parity,
                "blocked_index": blocked_index,
                "parameterization_column_overlap_to_seed_diagnostic": mode_overlap,
                "physical_identity_certified": False,
                "blocked_identity_diagnostic_scope": (
                    "parameterization overlap only; does not certify physical blocked identity"
                    if blocked_index is not None else None),
                "rms_radius_fm": (radius2 / number) ** 0.5 if number > 0 else None,
                "Q20_fm2": q20,
                "full_commutator_frobenius_squared_mev2": _scalar(
                    diagnostic["full_commutator_frobenius_squared_mev2"]),
                "normal_commutator_squared_mev2": _scalar(
                    diagnostic["normal_commutator_squared_mev2"]),
                "anomalous_commutator_squared_mev2": _scalar(
                    diagnostic["anomalous_commutator_squared_mev2"]),
                "scaled_stationarity_residual": _scalar(diagnostic["scaled_residual_loss"]),
            }

        stem = f"step_{int(step):04d}"
        _atomic_npz(exports / f"{stem}.npz", **arrays)
        record["export_file"] = str((exports / f"{stem}.npz").relative_to(output))
        record["export_sha256"] = _sha256(exports / f"{stem}.npz")
        _atomic_json(exports / f"{stem}.json", record)

    return callback


def run_o17_selfconsistent(
    *, output_dir: str | Path, updates: int = 2, device: str = "cpu",
    resume_from: str | Path | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    density: str | Path = DEFAULT_DENSITY,
    active_dimension: int = 42, cuda_budget_mib: int | None = None,
    normal_tile_size: int = 8, normal_chunk_size: int = 64,
    residual_chunk_size: int = 32, pairing_chunk_size: int = 16,
    pairing_active_chunk_size: int = 4,
    reference_source_dirs=None,
    branch_fixed_point_objective=None,
    post_variation_observer=None,
) -> dict[str, Any]:
    """Run the declared full PKA1+D1S O17 joint self-consistent objective."""
    # Guard first: this driver is a numerical launch entry point, not an
    # alternate path around the protected RHFB launcher.
    from .execution_limits import require_bounded_cgroup
    resource_limits = require_bounded_cgroup()
    reference_audits = _source_parameter_audits(reference_source_dirs)
    if active_dimension != 42:
        raise ValueError("the O17 run contract fixes the active chart at M=42; do not reduce it")

    checkpoint, density = Path(checkpoint).resolve(), Path(density).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and resume_from is None:
        raise FileExistsError(f"output directory exists; use --resume-from for continuation: {output}")
    output.mkdir(parents=True, exist_ok=resume_from is not None)

    import torch
    torch.set_num_threads(2)
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("device must be exactly cpu or cuda:0")
    cuda_evidence = None
    if device == "cuda:0":
        from .cuda_resources import configure_cuda
        budget = cuda_budget_mib
        if budget is None:
            raw_budget = os.environ.get("RHFB_CUDA_BUDGET_MIB")
            if raw_budget is None:
                raise RuntimeError("CUDA requires --cuda-budget-mib and protected launcher environment")
            budget = int(raw_budget)
        cuda_evidence = configure_cuda("cuda:0", budget_mib=budget)

    from .full_branch_neural_prototype import build_enlarged_o17_model
    from .selfconsistent_neural_training import train_joint
    from .streamed_normal_functional import StreamedPrimitiveNormalFunctional

    common = {"checkpoint_path": checkpoint, "density_path": density,
              "active_dimension": active_dimension, "seed": 20261006, "device": "cpu"}
    basis_n, parity_seed_n, init_n = build_enlarged_o17_model(species="n", **common)
    basis_p, parity_seed_p, init_p = build_enlarged_o17_model(species="p", **common)
    model_n, unrestricted_init_n = _convert_parity_seed_to_unrestricted(
        basis_n, parity_seed_n, device=device)
    model_p, unrestricted_init_p = _convert_parity_seed_to_unrestricted(
        basis_p, parity_seed_p, device=device)
    expected_basis = {"dimension": 432, "point_count": 882, "radius_fm": 8.0,
                      "spacing_fm": 0.8, "kappa_max": 3, "n_theta": 7,
                      "n_phi": 14, "derivative_order": 5}
    actual_basis = {
        "dimension": basis_n.dimension, "point_count": basis_n.point_count,
        "radius_fm": basis_n.radius_fm, "spacing_fm": basis_n.spacing_fm,
        "kappa_max": basis_n.kappa_max, "n_theta": basis_n.n_theta,
        "n_phi": basis_n.n_phi, "derivative_order": basis_n.derivative_order,
    }
    if basis_p.dimension != basis_n.dimension or any(
            actual_basis[key] != value for key, value in expected_basis.items() if value is not None):
        raise RuntimeError(f"O17 basis contract mismatch: {actual_basis}")
    if model_n.active_dimension != active_dimension or model_p.active_dimension != active_dimension:
        raise RuntimeError("both neutron and proton charts must retain the requested full M=42 space")

    snapshot_hashes = _source_snapshot(output, checkpoint, density)
    run_configuration = {
        "case": "O17", "N": 9, "Z": 8, "blocked_species": "n",
        "blocked_index": 13, "active_dimension": active_dimension,
        "primitive_dimension": basis_n.dimension, "basis": actual_basis,
        "functional": "complete PKA1 particle-hole plus full-space D1S pairing action",
        "angular_multipoles": "lmax=2K=6; PH angular convergence not certified",
        "density_prescription": "sqrt(j0^2-|j_space|^2) as configured in current implementation; odd-state response not externally benchmarked",
        "pairing": "D1S, full primitive-space action; nn and pp channels; no pn channel",
        "chemical_potential_mode": "number_tangent",
        "number_constraint_method": "field-dependent tangent Gram multiplier plus canonical mean-number retraction persisted in the network; not particle-number projection",
        "branch_fixed_point_objective": (None if branch_fixed_point_objective is None else
                                        branch_fixed_point_objective.checkpoint_metadata()),
        "post_variation_observer": (None if post_variation_observer is None else
                                    post_variation_observer.checkpoint_metadata()),
        "updates_requested": updates, "device": device,
        "chunking": {"normal_tile_size": normal_tile_size,
                     "normal_chunk_size": normal_chunk_size,
                     "residual_chunk_size": residual_chunk_size,
                     "pairing_chunk_size": pairing_chunk_size,
                     "pairing_active_chunk_size": pairing_active_chunk_size},
        "resource_limits": resource_limits, "cuda_policy": cuda_evidence,
        "initialization": {"checkpoint": str(checkpoint), "density_archive": str(density),
                           "neutron": init_n, "proton": init_p,
                           "unrestricted_neutron": unrestricted_init_n,
                           "unrestricted_proton": unrestricted_init_p,
                           "resume_is_separate": True},
        "source_sha256": snapshot_hashes,
        "source_parameter_audits": reference_audits,
        "physical_result_accepted": False,
        "ground_state_certified": False,
        "no_sea_branch_certified": False,
    }
    _atomic_json(output / "run_configuration.json", run_configuration)

    callback = _callback_factory(basis_n, {"n": model_n, "p": model_p}, output)
    normal_functional = StreamedPrimitiveNormalFunctional(
        basis_n, tile_size=normal_tile_size, chunk_size=normal_chunk_size,
        lmax=2 * basis_n.kappa_max,
        progress_callback=_ph_progress_callback(output),
    )
    started = time.time()
    report = train_joint(
        {"n": model_n, "p": model_p}, basis_n, {"n": 9, "p": 8}, updates,
        output / "trainer", chemical_potential_initial={"n": 0.0, "p": 0.0},
        train_chemical_potentials=False, chemical_potential_mode="number_tangent",
        normal_tile_size=normal_tile_size,
        normal_chunk_size=normal_chunk_size, residual_chunk_size=residual_chunk_size,
        pairing_chunk_size=pairing_chunk_size,
        pairing_active_chunk_size=pairing_active_chunk_size,
        normal_functional=normal_functional,
        resume_from=resume_from, progress_path=output / "progress.json",
        checkpoint_dir=output / "checkpoints", number_augmented_lagrangian=False,
        correct_mean_numbers=True,
        branch_fixed_point_objective=branch_fixed_point_objective,
        post_variation_observer=post_variation_observer,
        accepted_step_callback=callback,
    )
    summary = {
        "status": "algorithm_completed" if report.get("status") == "complete" else "algorithm_incomplete",
        "elapsed_seconds": time.time() - started,
        "training_report": report,
        "training_report_file": "trainer/training_report.json",
        "accepted_exports_dir": "accepted_evaluations",
        "checkpoint_last": (str(output / "checkpoints" / report["checkpoint_last"])
                             if report.get("checkpoint_last") else None),
        "resource_limits": resource_limits,
        "cuda_policy": cuda_evidence,
        "physical_result_accepted": False,
        "ground_state_certified": False,
        "no_sea_branch_certified": False,
        "limits_and_open_issues": [
            "Stationarity optimization does not establish an energy minimum or ground state.",
            "The physical no-sea variational branch is not supplied or certified.",
            "The PKA1 odd, time-reversal-breaking response has no external O17 benchmark here.",
            "The chosen invariant density prescription is an explicit covariant extension; its odd-state response is unbenchmarked.",
            "The configured PH angular cutoff is finite and has not been convergence-tested.",
            "The NN chart is M=42 inside D=432; the full D1S Delta action is retained, but densities remain chart-restricted.",
            "Neutron-neutron and proton-proton D1S pairing are used; proton-neutron pairing is absent by the selected like-particle implementation.",
        ],
    }
    _atomic_json(output / "run_report.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--resume-from", help="joint-selfconsistent training checkpoint only")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--density", default=str(DEFAULT_DENSITY))
    parser.add_argument("--active-dimension", type=int, default=42)
    parser.add_argument("--cuda-budget-mib", type=int)
    parser.add_argument("--reference-source", action="append",
                        help="read-only RHFB source directory to compare parameter tables; repeatable")
    parser.add_argument("--normal-tile-size", type=int, default=8)
    parser.add_argument("--normal-chunk-size", type=int, default=64)
    parser.add_argument("--residual-chunk-size", type=int, default=32)
    parser.add_argument("--pairing-chunk-size", type=int, default=16)
    parser.add_argument("--pairing-active-chunk-size", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_o17_selfconsistent(
        output_dir=args.output_dir, updates=args.updates, device=args.device,
        resume_from=args.resume_from, checkpoint=args.checkpoint, density=args.density,
        active_dimension=args.active_dimension, cuda_budget_mib=args.cuda_budget_mib,
        reference_source_dirs=args.reference_source,
        normal_tile_size=args.normal_tile_size, normal_chunk_size=args.normal_chunk_size,
        residual_chunk_size=args.residual_chunk_size,
        pairing_chunk_size=args.pairing_chunk_size,
        pairing_active_chunk_size=args.pairing_active_chunk_size,
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0 if report["status"] == "algorithm_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
