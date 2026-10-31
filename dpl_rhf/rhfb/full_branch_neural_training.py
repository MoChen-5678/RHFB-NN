"""Short frozen-field NN optimization audit for the O17 M42 chart.

This is a variational-residual descent experiment only.  It starts from the
archived NN state plus the existing analytic empty-column QR completion, keeps
the full frozen PKA1 ``h`` and full primitive D1S ``Delta`` actions, and never
updates fields, selects a no-sea branch, calls an eigensolver, or refreshes Q.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import resource
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .cuda_resources import configure_cuda, cuda_snapshot
from .execution_limits import require_bounded_cgroup
from .factorized_commutator import FactorizedCommutator
from .full_branch_neural_prototype import build_enlarged_o17_model
from .manifold import GeneralizedDensity, blocking_overlap
from .operators import DensityFactors
from .parity_diagnostic import number_parity_from_density
from .runtime_resources import resource_snapshot
from .spatial_parity import primitive_spatial_parity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _forbid_spectra():
    originals, counts = {}, {}
    for module_name, module in (("torch", torch.linalg), ("numpy", np.linalg)):
        for name in ("eig", "eigh", "eigvals", "eigvalsh"):
            key = f"{module_name}.linalg.{name}"
            originals[key] = getattr(module, name)
            counts[key] = 0

            def forbidden(*_args, _key=key, **_kwargs):
                counts[_key] += 1
                raise RuntimeError(f"spectral routine {_key} is forbidden in this prototype")

            setattr(module, name, forbidden)
    return originals, counts


def _restore_spectra(originals: dict[str, Any]) -> None:
    for key, value in originals.items():
        module_name, function = key.split(".linalg.", 1)
        setattr(torch.linalg if module_name == "torch" else np.linalg, function, value)


def _prepare_species_models(checkpoint: Path, density: Path, active_dimension: int):
    """Construct both analytic QR seeds before guarding optimization spectra.

    Basis construction creates the fixed Gauss-Legendre angular grid, whose
    NumPy implementation uses an eigensolver for a Jacobi matrix.  That setup
    is distinct from a Hamiltonian/BdG spectral solve and must precede the
    optimization-only spectrum guard.
    """
    prepared = {}
    for species in ("n", "p"):
        basis, model, init = build_enlarged_o17_model(
            checkpoint, density, species=species,
            active_dimension=active_dimension, device="cpu")
        prepared[species] = {"basis": basis, "model": model, "init": init}
    return prepared


def _validate_settings(active_dimension: int, updates: int, learning_rate: float,
                       chunk_size: int) -> None:
    if isinstance(active_dimension, bool) or active_dimension < 41:
        raise ValueError("active_dimension must be at least 41")
    if isinstance(updates, bool) or not 20 <= updates <= 50:
        raise ValueError("updates must be between 20 and 50")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA")
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {requested} is unavailable; CPU fallback is forbidden")
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device index {index} is unavailable")
        return torch.device("cuda", index)
    return requested


def _cuda_evidence(device: torch.device, *, reset_peak: bool = False):
    if device.type != "cuda":
        return {"cuda_used": False, "device": str(device)}
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    return cuda_snapshot(device)


def _assert_model_device_and_precision(model, device: torch.device) -> None:
    """Refuse mixed-device or reduced-precision NN state before building AD tape."""
    misplaced = [name for name, value in model.named_parameters()
                 if value.device != device]
    misplaced.extend(name for name, value in model.named_buffers()
                     if value.device != device)
    if misplaced:
        raise RuntimeError(f"model tensors are not all on {device}: {misplaced[:8]}")
    reduced = [name for name, value in model.named_parameters()
               if value.dtype not in (torch.float64, torch.complex128)]
    if reduced:
        raise TypeError(f"model parameters were reduced from float64/complex128: {reduced[:8]}")
    if model.initial_q.dtype != torch.complex128 or model.initial_q.device != device:
        raise TypeError("model initial_q must remain complex128 on the requested device")


class _RecomputeCounter:
    def __init__(self, counts: dict[str, int]):
        self.counts = counts

    def __enter__(self):
        self.counts["backward_recompute_calls"] += 1

    def __exit__(self, exc_type, exc, traceback):
        return False


def _model_forward(model, *, checkpoint_recompute: bool,
                   recompute_counts: dict[str, int] | None = None):
    """Run the complete tensor-valued model forward with optional AD rematerialization."""
    if not checkpoint_recompute or not torch.is_grad_enabled():
        return model()
    counts = recompute_counts if recompute_counts is not None else {
        "checkpointed_model_forward_executions": 0,
        "backward_recompute_calls": 0}

    def run(_token):
        counts["checkpointed_model_forward_executions"] += 1
        q, gd = model()
        return q, gd.rho, gd.chi, gd.matrix

    token = torch.ones((), dtype=torch.float64, device=next(model.parameters()).device)
    q, rho, chi, matrix = checkpoint(
        run, token, use_reentrant=False,
        context_fn=lambda: (nullcontext(), _RecomputeCounter(counts)),
    )
    return q, GeneralizedDensity(rho, chi, matrix, int(model.reference_parity))


def _objective(model, h: torch.Tensor, delta: torch.Tensor, mu: torch.Tensor,
               target_number: int, *, chunk_size: int, number_weight: float,
               checkpoint_recompute: bool = False,
               recompute_counts: dict[str, int] | None = None):
    q, gd = _model_forward(model, checkpoint_recompute=checkpoint_recompute,
                           recompute_counts=recompute_counts)
    factors = DensityFactors(q, gd.rho, gd.chi)
    action = FactorizedCommutator(
        factors, h @ q, lambda values: delta @ values,
        chemical_potential=mu,
    )
    terms = action.frobenius_squared(chunk_size=chunk_size)
    full = terms["full_squared_mev2"]
    residual = full / float((2 * h.shape[0]) ** 2)
    number_error = factors.particle_number() - float(target_number)
    number_penalty = float(number_weight) * number_error.square()
    return residual + number_penalty, {
        "full_squared_mev2": full,
        "normal_squared_mev2": terms["normal_squared_mev2"],
        "anomalous_squared_mev2": terms["anomalous_squared_mev2"],
        "element_mean_square_residual_mev2": residual,
        "number": factors.particle_number(),
        "number_error": number_error,
        "number_penalty": number_penalty,
        "q": q,
        "gd": gd,
    }


def _scalar_metrics(model, h, delta, mu, target_number, *, chunk_size, number_weight,
                    q_initial, blocked_initial, checkpoint_recompute=False,
                    recompute_counts=None):
    with torch.no_grad():
        objective, parts = _objective(model, h, delta, mu, target_number,
                                      chunk_size=chunk_size, number_weight=number_weight,
                                      checkpoint_recompute=checkpoint_recompute,
                                      recompute_counts=recompute_counts)
        q, gd = parts["q"], parts["gd"]
        if q_initial is None:
            q_motion = 0.0
        else:
            # ||QQ†-Q0Q0†||_F² = 2M - 2||Q0†Q||_F²; avoid a D×D projector.
            overlap = q_initial.mH @ q
            q_motion = float((2 * q.shape[1] - 2 * overlap.abs().square().sum()).clamp_min(0).sqrt())
        parity_rows = primitive_spatial_parity(model.basis, device=q.device)
        column_parity = model.orbital_parities.to(device=q.device)
        parity_error = float((parity_rows[:, None] * q - q * column_parity[None, :]).abs().max())
        density_parity = number_parity_from_density(
            gd.rho.detach().cpu().numpy(), gd.chi.detach().cpu().numpy())
        block_overlap = None
        if blocked_initial is not None:
            _, _, blocked_now = model.forward_with_blocked_mode()
            block_overlap = float(blocking_overlap(blocked_initial, blocked_now))
        return {
            "objective_mev2_scaled": float(objective),
            "full_commutator_frobenius_squared_mev2": float(parts["full_squared_mev2"]),
            "normal_commutator_squared_mev2": float(parts["normal_squared_mev2"]),
            "anomalous_commutator_squared_mev2": float(parts["anomalous_squared_mev2"]),
            "element_mean_square_residual_mev2": float(parts["element_mean_square_residual_mev2"]),
            "particle_number": float(parts["number"]),
            "number_error": float(parts["number_error"]),
            "number_penalty_term": float(parts["number_penalty"]),
            "number_parity": int(density_parity["number_parity"]),
            "purity_error": float(density_parity["purity_error"]),
            "q_projector_motion_frobenius": q_motion,
            "q_spatial_parity_max_error": parity_error,
            "blocked_mode_overlap_to_seed": block_overlap,
        }


def _train_species(model, h, delta, *, species: str, target_number: int,
                   updates: int, learning_rate: float, mu_initial: float,
                   train_mu: bool, number_weight: float, chunk_size: int,
                   max_trials_per_update: int, on_progress=None,
                   minimum_block_overlap: float = 0.90,
                   checkpoint_recompute: bool = True):
    if h.device != delta.device or h.device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"{species}: frozen h and Delta must share a CPU or CUDA device")
    if h.dtype != torch.complex128 or delta.dtype != torch.complex128:
        raise TypeError(f"{species}: frozen h and Delta must retain complex128 precision")
    _assert_model_device_and_precision(model, h.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    mu = torch.nn.Parameter(torch.tensor(float(mu_initial), dtype=torch.float64, device=h.device),
                            requires_grad=bool(train_mu))
    optimizer_parameters = parameters + ([mu] if train_mu else [])
    optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    with torch.no_grad():
        q_initial, _, blocked_initial = model.forward_with_blocked_mode()
        if q_initial.device != h.device or q_initial.dtype != torch.complex128:
            raise TypeError(f"{species}: initial Q must be complex128 on {h.device}")
        q_initial = q_initial.detach().clone()
        blocked_initial = None if blocked_initial is None else blocked_initial.detach().clone()
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()
                     if isinstance(value, torch.Tensor)}
    initial_neural = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                      if name.startswith(("radial_encoder.", "kappa_embedding.", "m_embedding.",
                                          "component_embedding.", "state_embedding.", "output_head."))}
    if on_progress is not None:
        on_progress(0, [], optimizer, mu, model)
    history = [_scalar_metrics(model, h, delta, mu, target_number,
                               chunk_size=chunk_size, number_weight=number_weight,
                               q_initial=q_initial, blocked_initial=blocked_initial)]
    history[0]["chemical_potential_mev"] = float(mu.detach())
    if on_progress is not None:
        on_progress(0, history, optimizer, mu, model)
    accepted = 0
    attempts = 0
    while accepted < updates:
        if attempts >= updates * max_trials_per_update:
            raise RuntimeError(f"{species}: line search could not accept {updates} updates")
        attempts += 1
        optimizer.zero_grad(set_to_none=True)
        recompute_counts = {"checkpointed_model_forward_executions": 0,
                            "backward_recompute_calls": 0}
        old_objective, _ = _objective(
            model, h, delta, mu, target_number, chunk_size=chunk_size,
            number_weight=number_weight, checkpoint_recompute=checkpoint_recompute,
            recompute_counts=recompute_counts)
        if not bool(torch.isfinite(old_objective)):
            raise FloatingPointError(f"{species}: nonfinite objective before optimizer step")
        old_objective.backward()
        torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=10.0)
        old_params = [p.detach().clone() for p in optimizer_parameters]
        old_opt = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        proposed = [p.detach().clone() for p in optimizer_parameters]
        proposed_opt = copy.deepcopy(optimizer.state_dict())
        accepted_trial = False
        accepted_metrics = None
        for backtrack in range(12):
            alpha = 0.5 ** backtrack
            with torch.no_grad():
                for parameter, old, new in zip(optimizer_parameters, old_params, proposed):
                    parameter.copy_(old + alpha * (new - old))
                trial_objective, _ = _objective(model, h, delta, mu, target_number,
                                                chunk_size=chunk_size, number_weight=number_weight)
            if (bool(torch.isfinite(trial_objective))
                    and float(trial_objective.detach()) < float(old_objective.detach())):
                trial_metrics = _scalar_metrics(
                    model, h, delta, mu, target_number, chunk_size=chunk_size,
                    number_weight=number_weight, q_initial=q_initial,
                    blocked_initial=blocked_initial)
                parity_ok = trial_metrics["number_parity"] == int(model.reference_parity)
                spatial_parity_ok = trial_metrics["q_spatial_parity_max_error"] <= 1.0e-9
                block_overlap = trial_metrics["blocked_mode_overlap_to_seed"]
                blocked_ok = block_overlap is None or block_overlap >= minimum_block_overlap
                if parity_ok and spatial_parity_ok and blocked_ok:
                    optimizer.load_state_dict(proposed_opt)
                    accepted_trial = True
                    accepted_metrics = trial_metrics
                    break
        if not accepted_trial:
            with torch.no_grad():
                for parameter, old in zip(optimizer_parameters, old_params):
                    parameter.copy_(old)
            optimizer.load_state_dict(old_opt)
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            continue
        accepted += 1
        row = accepted_metrics
        current_neural = {name: parameter.detach()
                          for name, parameter in model.named_parameters()
                          if name in initial_neural}
        current_changes = {
            name: float(torch.linalg.vector_norm(current_neural[name] - before).detach())
            for name, before in initial_neural.items()}
        encoder_change = sum(value * value for name, value in current_changes.items()
                             if not name.startswith("output_head.")) ** 0.5
        head_change = sum(value * value for name, value in current_changes.items()
                          if name.startswith("output_head.")) ** 0.5
        row.update({"accepted_update": accepted, "attempt": attempts,
                    "line_search_alpha": alpha, "learning_rate": optimizer.param_groups[0]["lr"],
                    "chemical_potential_mev": float(mu.detach()),
                    "neural_encoder_change_l2": encoder_change,
                    "neural_head_change_l2": head_change,
                    "neural_encoder_changed": encoder_change > 0.0,
                    "neural_head_changed": head_change > 0.0,
                    "checkpoint_recompute_counts": dict(recompute_counts)})
        history.append(row)
        if on_progress is not None:
            on_progress(accepted, history, optimizer, mu, model)
    final_neural = {name: parameter.detach() for name, parameter in model.named_parameters()
                    if name in initial_neural}
    neural_changes = {name: float(torch.linalg.vector_norm(final_neural[name] - before).detach())
                      for name, before in initial_neural.items()}
    encoder_change = sum(value * value for name, value in neural_changes.items()
                         if not name.startswith("output_head.")) ** 0.5
    head_change = sum(value * value for name, value in neural_changes.items()
                      if name.startswith("output_head.")) ** 0.5
    if encoder_change <= 0.0 or head_change <= 0.0:
        raise RuntimeError(
            f"{species}: accepted updates did not change both neural encoder and output head")
    return {
        "model": model,
        "optimizer": optimizer,
        "chemical_potential": mu,
        "history": history,
        "optimizer_updates": accepted,
        "optimizer_attempts": attempts,
        "neural_parameter_change_l2": neural_changes,
        "neural_encoder_change_l2": encoder_change,
        "neural_head_change_l2": head_change,
        "neural_encoder_changed": encoder_change > 0.0,
        "neural_head_changed": head_change > 0.0,
        "neural_encoder_and_head_changed": encoder_change > 0.0 and head_change > 0.0,
        "optimizer_state": optimizer.state_dict(),
        "initial_model_state": initial_state,
    }


def train_o17_short_frozen_field(
    *, checkpoint: str | Path, density: str | Path, field_dir: str | Path,
    pairing_field: str | Path, output_dir: str | Path, active_dimension: int = 42,
    updates: int = 20, learning_rate: float = 2.0e-4, mu_initial: float = 0.0,
    train_mu: bool = True, number_weight: float = 1.0e4, chunk_size: int = 32,
    max_trials_per_update: int = 8, device: str | torch.device = "cuda:0",
    cuda_budget_mib: int = 8192, checkpoint_recompute: bool = True,
) -> dict[str, Any]:
    """Train n then p sequentially against the same saved frozen O17 fields."""
    _validate_settings(active_dimension, updates, learning_rate, chunk_size)
    if not np.isfinite(mu_initial) or not np.isfinite(number_weight) or number_weight <= 0:
        raise ValueError("mu_initial and positive number_weight must be finite")
    if not 1 <= max_trials_per_update <= 16:
        raise ValueError("max_trials_per_update must be between 1 and 16")
    device = _resolve_device(device)
    if isinstance(cuda_budget_mib, bool) or not 256 <= cuda_budget_mib <= 8192:
        raise ValueError("cuda_budget_mib must be between 256 and 8192")
    if not isinstance(checkpoint_recompute, bool):
        raise ValueError("checkpoint_recompute must be boolean")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "history.jsonl"
    history_path.touch()
    start = time.time()
    limits = None
    results: dict[str, Any] = {}
    primitive_dimension = None
    active_checkpoint = None
    device_evidence: dict[str, Any] = {"requested_device": str(device),
                                       "cuda_used": device.type == "cuda"}
    initial_cuda_memory: dict[str, Any] | None = None

    def write_report(data):
        temporary = output / "training_report.json.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(output / "training_report.json")

    def base_report(status):
        return {
            "status": status,
            "physical_result_accepted": False,
            "no_sea_branch_certified": False,
            "scope": "actual saved O17 NN n/p seeds; same frozen full PKA1 h and full primitive D1S Delta; sequential optimizer updates; no field self-consistency",
            "active_dimension": int(active_dimension),
            "device": str(device),
            "cuda_budget_mib": int(cuda_budget_mib),
            "checkpoint_recompute": checkpoint_recompute,
            "field_recomputation_count": 0,
            "field_recomputation_policy": "h and Delta are fixed frozen inputs loaded once; no self-consistent field update exists in this prototype",
            "updates_requested_per_species": int(updates),
            "chemical_potential_initial_mev": float(mu_initial),
            "chemical_potential_trainable": bool(train_mu),
            "chemical_potential_source": f"explicit caller-specified non-spectral mu_initial={float(mu_initial):.17g}; not calibrated or physically accepted",
            "spectral_teacher_or_initialization": False,
            "spectrum_attempts_scope": "optimization only; fixed Gauss-Legendre basis-grid preprocessing occurs before the guard and is not a Hamiltonian/BdG spectral solve",
            "basis_refreshes_after_analytic_qr": 0,
            "fortran_used": False,
            "kernel_resource_limits": limits,
            "species": results,
            "started_unix": start,
            "elapsed_seconds": time.time() - start,
        }

    write_report(base_report("starting"))
    checkpoint, density, field_dir, pairing_field = map(
        Path, (checkpoint, density, field_dir, pairing_field))
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("full_branch_neural_prototype.py"),
        Path(__file__).with_name("factorized_commutator.py"),
        Path(__file__).with_name("network.py"),
        Path(__file__).with_name("manifold.py"),
        Path(__file__).with_name("spatial_parity.py"),
        Path(__file__).with_name("operators.py"),
        Path(__file__).with_name("basis.py"),
        Path(__file__).with_name("parity_diagnostic.py"),
        Path(__file__).with_name("execution_limits.py"),
        Path(__file__).with_name("runtime_resources.py"),
        Path(__file__).with_name("cuda_resources.py"),
        checkpoint, density, field_dir / "h_n_columns.npy",
        field_dir / "h_p_columns.npy", pairing_field,
    ]
    snapshot = output / "run_start_sources"
    source_hashes = {}
    originals = None
    spectrum_counts = {}
    try:
        # Fail closed before loading full O17 arrays or constructing the AD graph.
        limits = require_bounded_cgroup()
        limits.update({"no_swap": limits["memory_swap_max_bytes"] == 0,
                       "maximum_allowed_bytes": 4 * 1024**3})
        torch.set_num_threads(2)
        if device.type == "cuda":
            device_evidence = configure_cuda(device=str(device), budget_mib=cuda_budget_mib)
            torch.cuda.set_device(device)
            initial_cuda_memory = _cuda_evidence(device, reset_peak=True)
        else:
            device_evidence = {"cuda_used": False, "device": str(device),
                               "cpu_explicitly_selected": True,
                               "physical_space_reduced_for_cpu": False}
            initial_cuda_memory = _cuda_evidence(device)
        snapshot.mkdir()
        for source in source_paths:
            copied = snapshot / source.name
            copied.write_bytes(source.read_bytes())
            source_hashes[str(source)] = _sha256(copied)
        write_report({**base_report("running"), "source_sha256": source_hashes,
                      "run_start_source_snapshot": str(snapshot),
                      "device_evidence": device_evidence,
                      "cuda_memory": _cuda_evidence(device)})

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pairing = np.load(pairing_field)
        prepared_models = _prepare_species_models(checkpoint, density, active_dimension)
        originals, spectrum_counts = _forbid_spectra()
        for species in ("n", "p"):
            prepared = prepared_models.pop(species)
            basis, model, init = prepared["basis"], prepared["model"], prepared["init"]
            model = model.to(device=device)
            _assert_model_device_and_precision(model, device)
            primitive_dimension = basis.dimension
            h = torch.as_tensor(np.array(np.load(field_dir / f"h_{species}_columns.npy",
                                                 mmap_mode="r"), copy=True),
                                dtype=torch.complex128, device=device)
            delta = torch.as_tensor(pairing[f"{species}_delta"], dtype=torch.complex128,
                                    device=device)
            if any(value.device != device for value in (h, delta)):
                raise RuntimeError(f"{species}: h and Delta did not move to requested {device}")
            if h.dtype != torch.complex128 or delta.dtype != torch.complex128:
                raise TypeError(f"{species}: loaded frozen fields must be complex128")
            if h.shape != (basis.dimension, basis.dimension) or delta.shape != h.shape:
                raise ValueError(f"{species}: frozen field dimensions disagree with the O17 basis")
            if not torch.allclose(h, h.mH, atol=1e-8, rtol=0):
                raise ValueError(f"{species}: frozen normal field is not Hermitian")
            if not torch.allclose(delta, -delta.mT, atol=1e-8, rtol=0):
                raise ValueError(f"{species}: frozen D1S pairing field is not skew")
            target = int(payload["config"]["n" if species == "n" else "z"])

            active_checkpoint = output / f"{species}_m{active_dimension}_partial.pt"

            def progress(step, rows, optimizer, mu_parameter, current_model,
                         *, current_species=species, partial_path=active_checkpoint):
                row = ({"accepted_update": 0,
                        "chemical_potential_mev": float(mu_parameter.detach()),
                        "event": ("initialized_partial_state" if not rows
                                  else "initial_diagnostic_complete")} if step == 0
                       else {**rows[-1], "event": "accepted_update"})
                row["cuda_memory"] = _cuda_evidence(device)
                with history_path.open("a") as stream:
                    stream.write(json.dumps({"species": current_species, **row}) + "\n")
                    stream.flush()
                _atomic_torch_save({"model": current_model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "chemical_potential_mev": float(mu_parameter.detach()),
                                    "accepted_updates": int(step), "history": rows,
                                    "checkpoint_recompute": checkpoint_recompute,
                                    "field_recomputation_count": 0,
                                    "field_recomputation_policy": "frozen h and Delta are fixed inputs; no field updates",
                                    "no_sea_branch_certified": False,
                                    "physical_result_accepted": False}, partial_path)
                results[current_species] = {
                    "status": "training", "accepted_updates": int(step),
                    "partial_checkpoint": partial_path.name,
                    "checkpoint_recompute_counts": {
                        key: sum(int(row.get("checkpoint_recompute_counts", {}).get(key, 0))
                                 for row in rows)
                        for key in ("checkpointed_model_forward_executions",
                                    "backward_recompute_calls")},
                    "field_recomputation_count": 0,
                }
                write_report({**base_report("running"), "source_sha256": source_hashes,
                              "run_start_source_snapshot": str(snapshot),
                              "active_species": current_species,
                              "spectrum_attempts": spectrum_counts,
                              "spectrum_attempts_scope": "optimization only; fixed Gauss-Legendre basis-grid preprocessing occurs before the guard and is not a Hamiltonian/BdG spectral solve",
                              "runtime_resource_snapshot": resource_snapshot(),
                              "minimum_block_overlap_gate": 0.90,
                              "device_evidence": device_evidence,
                              "cuda_memory": _cuda_evidence(device)})

            trained = _train_species(
                model, h, delta, species=species, target_number=target,
                updates=updates, learning_rate=learning_rate, mu_initial=mu_initial,
                train_mu=train_mu, number_weight=number_weight, chunk_size=chunk_size,
                max_trials_per_update=max_trials_per_update, on_progress=progress,
                checkpoint_recompute=checkpoint_recompute)
            result_path = output / f"{species}_m{active_dimension}_frozen_training.pt"
            _atomic_torch_save({"model": trained["model"].state_dict(),
                                "optimizer": trained["optimizer_state"],
                                "chemical_potential_mev": float(trained["chemical_potential"].detach()),
                                "history": trained["history"], "initialization": init,
                                "optimizer_updates": trained["optimizer_updates"],
                "checkpoint_recompute": checkpoint_recompute,
                "field_recomputation_count": 0,
                "field_recomputation_policy": "frozen h and Delta are fixed inputs; no field updates",
                "no_sea_branch_certified": False,
                                "physical_result_accepted": False}, result_path)
            results[species] = {
                "initialization": init,
                "target_number": target,
                "mu_initial_mev": float(mu_initial),
                "mu_initialization_source": f"explicit caller-specified non-spectral prototype value {float(mu_initial):.17g}; no spectral values or bracket used",
                "chemical_potential_trainable": bool(train_mu),
                "chemical_potential_final_mev": float(trained["chemical_potential"].detach()),
                "number_weight": float(number_weight),
                "history": trained["history"],
                "optimizer_updates": trained["optimizer_updates"],
                "optimizer_attempts": trained["optimizer_attempts"],
                "neural_parameter_change_l2": trained["neural_parameter_change_l2"],
                "neural_encoder_change_l2": trained["neural_encoder_change_l2"],
                "neural_head_change_l2": trained["neural_head_change_l2"],
                "neural_encoder_changed": trained["neural_encoder_changed"],
                "neural_head_changed": trained["neural_head_changed"],
                "neural_encoder_and_head_changed": trained["neural_encoder_and_head_changed"],
                "minimum_block_overlap_gate": 0.90,
                "optimizer_state_saved": True,
                "checkpoint_file": result_path.name,
                "checkpoint_sha256": _sha256(result_path),
                "basis_refreshes": 0,
                "field_updates": 0,
                "field_recomputation_count": 0,
            }
            active_checkpoint = result_path
            write_report({**base_report("running"), "source_sha256": source_hashes,
                              "run_start_source_snapshot": str(snapshot),
                              "spectrum_attempts": spectrum_counts,
                              "spectrum_attempts_scope": "optimization only; fixed Gauss-Legendre basis-grid preprocessing occurs before the guard and is not a Hamiltonian/BdG spectral solve",
                              "device_evidence": device_evidence,
                              "cuda_memory": _cuda_evidence(device)})
            del h, delta, model, basis, trained
    except BaseException as exc:
        failure = {**base_report("failed"), "error_type": type(exc).__name__,
                   "error": str(exc), "source_sha256": source_hashes,
                   "run_start_source_snapshot": str(snapshot) if snapshot.exists() else None,
                   "spectrum_attempts": spectrum_counts,
                   "spectrum_attempts_scope": "optimization only; fixed Gauss-Legendre basis-grid preprocessing occurs before the guard and is not a Hamiltonian/BdG spectral solve",
                   "partial_checkpoint": (active_checkpoint.name
                                          if active_checkpoint is not None and active_checkpoint.exists()
                                          else None)}
        try:
            if limits is not None:
                failure["runtime_resource_snapshot"] = resource_snapshot()
                failure["cuda_memory"] = _cuda_evidence(device)
                failure["device_evidence"] = device_evidence
        except Exception as resource_error:
            failure["runtime_resource_error"] = f"{type(resource_error).__name__}: {resource_error}"
        write_report(failure)
        raise
    finally:
        if originals is not None:
            _restore_spectra(originals)
    try:
        runtime_limits = resource_snapshot()
    except BaseException as exc:
        write_report({**base_report("failed"), "error_type": type(exc).__name__,
                      "error": str(exc), "source_sha256": source_hashes,
                      "run_start_source_snapshot": str(snapshot),
                      "spectrum_attempts": spectrum_counts,
                      "spectrum_attempts_scope": "optimization only; fixed Gauss-Legendre basis-grid preprocessing occurs before the guard and is not a Hamiltonian/BdG spectral solve",
                      "partial_checkpoint": (active_checkpoint.name
                                             if active_checkpoint is not None and active_checkpoint.exists()
                                             else None)})
        raise
    report = {
        "status": f"o17_m{active_dimension}_frozen_field_nn_residual_descent_prototype",
        "physical_result_accepted": False,
        "no_sea_branch_certified": False,
        "scope": "actual saved O17 NN n/p seeds; same frozen full PKA1 h and full primitive D1S Delta; sequential Adam updates; no field self-consistency",
        "active_dimension": int(active_dimension),
        "device": str(device),
        "cuda_budget_mib": int(cuda_budget_mib),
        "checkpoint_recompute": checkpoint_recompute,
        "checkpoint_recompute_counts": {
            species: {"checkpointed_model_forward_executions": sum(int(row.get("checkpoint_recompute_counts", {}).get("checkpointed_model_forward_executions", 0))
                                            for row in entry.get("history", [])),
                      "backward_recompute_calls": sum(int(row.get("checkpoint_recompute_counts", {}).get("backward_recompute_calls", 0))
                                                       for row in entry.get("history", []))}
            for species, entry in results.items()
        },
        "primitive_dimension": int(primitive_dimension),
        "updates_requested_per_species": int(updates),
        "objective": "exact streamed full primitive ||[H_mu,R]||_F^2/(2D)^2 plus particle-number quadratic constraint; no energy minimization",
        "chemical_potential_source": f"explicit caller-specified non-spectral mu_initial={float(mu_initial):.17g}; trainable scalar when enabled; not calibrated or physically accepted",
        "spectral_teacher_or_initialization": False,
        "spectrum_attempts": spectrum_counts,
        "spectrum_attempts_scope": "optimization only; fixed Gauss-Legendre basis-grid preprocessing occurs before the guard and is not a Hamiltonian/BdG spectral solve",
        "basis_refreshes_after_analytic_qr": 0,
        "fortran_used": False,
        "field_freezing": {"h_frozen": True, "delta_frozen": True,
                           "pairing_delta_projected_to_Q": False,
                           "delta_action_uses_full_D_by_D_matrix": True,
                           "field_recomputation_count": 0,
                           "field_recomputation_policy": "fixed h and Delta are loaded once and never updated in this prototype"},
        "kernel_resource_limits": limits,
        "runtime_resource_snapshot": runtime_limits,
        "device_evidence": device_evidence,
        "initial_cuda_memory": initial_cuda_memory,
        "cuda_memory": _cuda_evidence(device),
        "observed_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "elapsed_seconds": time.time() - start,
        "source_sha256": source_hashes,
        "run_start_source_snapshot": str(snapshot),
        "history_jsonl": history_path.name,
        "species": results,
    }
    write_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--field-dir", required=True)
    parser.add_argument("--pairing-field", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--active-dimension", type=int, default=42)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--mu-initial", type=float, default=0.0)
    parser.add_argument("--freeze-mu", action="store_true")
    parser.add_argument("--device", default="cuda:0",
                        help="requested device; CPU must be selected explicitly; no automatic fallback")
    parser.add_argument("--cuda-budget-mib", type=int, default=8192)
    parser.add_argument("--no-checkpoint-recompute", action="store_true")
    args = parser.parse_args()
    report = train_o17_short_frozen_field(
        checkpoint=args.checkpoint, density=args.density, field_dir=args.field_dir,
        pairing_field=args.pairing_field, output_dir=args.output_dir,
        active_dimension=args.active_dimension, updates=args.updates,
        learning_rate=args.learning_rate, mu_initial=args.mu_initial,
        train_mu=not args.freeze_mu, device=args.device,
        cuda_budget_mib=args.cuda_budget_mib,
        checkpoint_recompute=not args.no_checkpoint_recompute)
    print(json.dumps({"status": report["status"], "species": {
        key: {"updates": value["optimizer_updates"],
              "initial_residual": value["history"][0]["element_mean_square_residual_mev2"],
              "final_residual": value["history"][-1]["element_mean_square_residual_mev2"]}
        for key, value in report["species"].items()}}, indent=2))


if __name__ == "__main__":
    main()
