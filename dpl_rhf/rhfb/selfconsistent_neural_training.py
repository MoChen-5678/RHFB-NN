"""Joint Adam training against freshly recomputed full PH+D1S fields.

This is an experimental optimizer driver, not a nuclear solver or a no-sea
branch selector. It optimizes two GeneralizedDensityNet instances together
against :func:`selfconsistent_stationarity_loss`; every line-search trial
rebuilds both species' PH fields and full-space pairing actions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import resource
import time
from dataclasses import asdict, is_dataclass
from contextlib import ContextDecorator, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .manifold import GeneralizedDensity
from .operators import DensityFactors
from .parity_diagnostic import number_parity_from_density
from .execution_limits import require_bounded_cgroup
from .network import GeneralizedDensityNet
from .selfconsistent_stationarity import selfconsistent_stationarity_loss
from .streamed_normal_functional import StreamedPrimitiveNormalFunctional
from .blocking_tracking import transport_occupied_tag
from .prescription import (current_neural_prescription, checkpoint_prescription_fields,
                           validate_checkpoint_prescription)


class _TrialRejected(RuntimeError):
    """A finite trial violates a required density or particle-number structure."""


class _NonfiniteLoss(FloatingPointError):
    """The evaluated joint loss is NaN/Inf; fatal at a base state, rejectable for a trial."""


class _NumberCorrectionFailed(RuntimeError):
    """Mean-number transport did not converge; fatal initially, rejectable for a trial."""


class _BackwardRecompute:
    def __init__(self, counts):
        self.counts = counts

    def __enter__(self):
        self.counts["backward_recompute_calls"] += 1

    def __exit__(self, *_args):
        return False


def _model_forward(model, counts=None):
    """Checkpoint the complete tensor-valued GeneralizedDensityNet forward."""
    if not torch.is_grad_enabled():
        return model()
    if counts is None:
        counts = {"checkpointed_model_forward_executions": 0,
                  "backward_recompute_calls": 0}

    def run(_token):
        counts["checkpointed_model_forward_executions"] += 1
        q, density = model()
        return q, density.rho, density.chi, density.matrix

    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("each model must have trainable parameters")
    token = torch.ones((), dtype=torch.float64, device=parameter.device)
    q, rho, chi, matrix = checkpoint(
        run, token, use_reentrant=False,
        context_fn=lambda: (nullcontext(), _BackwardRecompute(counts)),
    )
    return q, GeneralizedDensity(rho, chi, matrix, int(model.reference_parity))


def _basis_metadata(basis):
    names = ("radius_fm", "spacing_fm", "kappa_max", "n_theta", "n_phi",
             "derivative_order", "dimension", "point_count",
             "mass_n_mev", "mass_p_mev", "hbar_c_mev_fm")
    metadata = {}
    for name in names:
        value = getattr(basis, name, None)
        if value is not None:
            metadata[name] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return metadata


def _model_metadata(model):
    callback = getattr(model, "checkpoint_metadata", None)
    if callable(callback):
        return callback()
    return {"class": f"{type(model).__module__}.{type(model).__qualname__}"}


def _model_state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(name.encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _model_reference_fingerprint(model):
    """Fingerprint fixed reference/basis buffers, excluding trainable weights."""
    digest = hashlib.sha256()
    selected = {"initial_q", "reference_rho", "reference_chi", "reference_matrix"}
    for name, value in sorted(model.named_buffers()):
        if name.rsplit(".", 1)[-1] not in selected:
            continue
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _atomic_torch_save(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _rng_state():
    state = {"torch_cpu": torch.get_rng_state(),
             "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not isinstance(state, dict) or "torch_cpu" not in state:
        raise ValueError("checkpoint has no complete RNG state")
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "numpy" not in state or "python" not in state:
        raise ValueError("checkpoint RNG state predates complete-resume format")
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            raise ValueError("checkpoint has CUDA RNG state but caller did not configure CUDA")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _resource_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    snapshot = {"pid": os.getpid(), "max_rss_kib": int(usage.ru_maxrss)}
    for name, path in (("memory_current_bytes", "/sys/fs/cgroup/memory.current"),
                       ("memory_peak_bytes", "/sys/fs/cgroup/memory.peak")):
        try:
            snapshot[name] = int(Path(path).read_text().strip())
        except (OSError, ValueError):
            snapshot[name] = None
    return snapshot


def _resume_compatibility(metadata):
    models = {}
    for species in ("n", "p"):
        info = metadata["models"][species]
        models[species] = {
            "architecture": {k: v for k, v in info.items()
                             if k != "state_fingerprint"},
            "reference_fingerprint": metadata["model_reference_fingerprints"][species],
        }
    return {
        "format": "joint-selfconsistent-neural-training-v2",
        "physics": metadata["physics"],
        "production_physics_asserted": metadata["production_physics_asserted"],
        "optimization_target": metadata["optimization_target"],
        "physical_prescription": metadata["physical_prescription"],
        "physical_prescription_versioned": metadata.get("physical_prescription_versioned"),
        "particle_hole_parameters": metadata["particle_hole_parameters"],
        "pairing_parameters": metadata["pairing_parameters"],
        "normal_functional": metadata["normal_functional"],
        "stationarity_evaluator": metadata["stationarity_evaluator"],
        "branch_fixed_point_objective": metadata.get("branch_fixed_point_objective"),
        "post_variation_observer": metadata.get("post_variation_observer"),
        "basis": metadata["basis"],
        "number_targets": metadata["number_targets"],
        "models": models,
        "device": metadata["device"],
        # Continuation length and the unused fresh-run chemical-potential
        # initializer may change; all objective, optimizer, and diagnostic
        # settings remain exact-match requirements.
        "options": {k: v for k, v in metadata["options"].items()
                    if k not in {"updates_requested", "chemical_potential_initial"}},
    }


def _probe_vectors(dimension, device):
    rows = torch.arange(1, dimension + 1, dtype=torch.float64, device=device)[:, None]
    cols = torch.arange(1, 4, dtype=torch.float64, device=device)[None, :]
    real = torch.sin(rows * cols * 0.731)
    imag = torch.cos(rows * cols * 0.419)
    values = torch.complex(real, imag)
    return values / torch.linalg.vector_norm(values, dim=0, keepdim=True).clamp_min(1e-30)


def _field_signature(evaluation, dimension):
    """Hash h, active gaps, and deterministic full-Delta action probes."""
    digest = hashlib.sha256()
    vectors = _probe_vectors(dimension, evaluation.normal_fields["n"].device)
    chunks = []
    for species in ("n", "p"):
        h = evaluation.normal_fields[species].detach()
        gap = evaluation.active_pairing_gaps[species].detach()
        with torch.no_grad():
            delta_probe = evaluation.delta_actions[species].apply(vectors).detach()
        for value in (h, gap, delta_probe):
            array = value.to(device="cpu").contiguous().numpy()
            digest.update(species.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
            chunks.append(value.reshape(-1))
    signature = torch.cat(chunks).detach()
    return digest.hexdigest(), signature


def _json_energy_channels(evaluation):
    values = getattr(evaluation, "energy_channels", None)
    if values is None:
        values = getattr(evaluation, "energy_terms", None)
    if not isinstance(values, dict):
        return None
    return {str(name): float(torch.as_tensor(value).detach().cpu())
            for name, value in values.items()}


def _effective_chemical_potentials(evaluation, fallback):
    diagnostics = getattr(evaluation, "species_diagnostics", {})
    return {s: float(torch.as_tensor(diagnostics.get(s, {}).get(
        "chemical_potential_mev", fallback[s])).detach().cpu()) for s in ("n", "p")}


def _make_states(models, *, counts=None):
    states, outputs = {}, {}
    for species in ("n", "p"):
        q, gd = _model_forward(models[species], counts=counts)
        states[species] = DensityFactors(q, gd.rho, gd.chi)
        outputs[species] = (q, gd)
    return states, outputs


@contextmanager
def _forbid_spectral_calls():
    originals = []
    for module, names in ((torch.linalg, ("eig", "eigh", "eigvals", "eigvalsh")),
                          (np.linalg, ("eig", "eigh", "eigvals", "eigvalsh")),
                          (torch, ("eig", "symeig"))):
        for name in names:
            if hasattr(module, name):
                originals.append((module, name, getattr(module, name)))
                setattr(module, name, lambda *_a, _name=name, **_kw: (_ for _ in ()).throw(
                    RuntimeError(f"spectral routine {_name} is forbidden during joint training")))
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


class _NoSpectrum(ContextDecorator):
    """Keep eigensolver entry points disabled through forward and backward."""
    def __enter__(self):
        self._patch = _forbid_spectral_calls()
        self._patch.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._patch.__exit__(exc_type, exc, traceback)


def _evaluate(models, basis, targets, mus, evaluator, evaluator_options,
              counts=None):
    with _forbid_spectral_calls():
        states, outputs = _make_states(models, counts=counts)
        result = evaluator(basis, states, mus, targets, **evaluator_options)
    return result, states, outputs


def _branch_metrics(models, outputs, targets, initial_modes, previous_tags,
                    minimum_overlap):
    metrics, candidate_tags = {}, {}
    for species in ("n", "p"):
        q, density = outputs[species]
        model = models[species]
        parity = number_parity_from_density(
            density.rho.detach().cpu().numpy(), density.chi.detach().cpu().numpy())
        target_parity = -1 if targets[species] % 2 else 1
        if parity["number_parity"] != target_parity:
            raise _TrialRejected(f"{species}: trial changed particle-number parity")
        if int(model.reference_parity) != target_parity:
            raise ValueError(f"{species}: model reference parity disagrees with target number")
        blocked_index = getattr(model, "blocked_index", None)
        if blocked_index is not None:
            if initial_modes[species] is None or previous_tags[species] is None:
                raise RuntimeError(f"{species}: occupied blocked tag is unavailable")
            try:
                transport = transport_occupied_tag(
                    q, density, previous_tags[species],
                    initial_mode=initial_modes[species],
                    minimum_overlap=minimum_overlap,
                )
            except ValueError as error:
                raise _TrialRejected(f"{species}: invalid occupied-tag/density structure: {error}") from error
            tag_resolved = bool(transport.tag_resolved)
            candidate_tags[species] = (
                transport.mode.detach().clone() if tag_resolved
                else previous_tags[species].detach().clone()
            )
            local_overlap = transport.local_overlap
            initial_overlap = transport.initial_overlap
            projection_norm = transport.projection_norm
            particle_hole_error = transport.particle_hole_error
            above_minimum_overlap = getattr(transport, "above_minimum_overlap", None)
            identity_certified = False
        elif target_parity < 0:
            raise ValueError(f"{species}: odd target requires a blocked-index model")
        else:
            candidate_tags[species] = None
            local_overlap = initial_overlap = None
            projection_norm = particle_hole_error = above_minimum_overlap = None
            tag_resolved = None
            identity_certified = False
        metrics[species] = {
            "particle_number": float(torch.trace(density.rho).real.detach()),
            "number_parity": int(parity["number_parity"]),
            "purity_error": float(parity["purity_error"]),
            "blocked_mode_local_overlap": local_overlap,
            "blocked_mode_initial_overlap": initial_overlap,
            "blocked_tag_projection_norm": projection_norm,
            "blocked_tag_particle_hole_error": particle_hole_error,
            "blocked_tag_above_minimum_overlap_diagnostic": above_minimum_overlap,
            "blocked_tag_resolved": tag_resolved,
            "physical_identity_certified": identity_certified,
        }
    return metrics, candidate_tags


def _parameter_groups(models, mus):
    parameters, seen = [], set()
    for species in ("n", "p"):
        for parameter in models[species].parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError("n and p models must not share trainable parameters")
            seen.add(id(parameter))
            parameters.append(parameter)
    parameters.extend(mus[s] for s in ("n", "p") if mus[s].requires_grad)
    if not parameters:
        raise ValueError("joint training has no trainable parameters")
    return parameters


@_NoSpectrum()
def train_joint(
    models: dict[str, torch.nn.Module], basis, number_targets: dict[str, int],
    updates: int, output_dir: str | Path, *,
    chemical_potential_initial: dict[str, float] | None = None,
    train_chemical_potentials: bool = True,
    chemical_potential_mode: str = "provided",
    learning_rate: float = 1.0e-4,
    number_weight: float = 1.0e4,
    max_trials_per_update: int = 8,
    minimum_block_overlap: float | None = None,
    normal_tile_size: int = 8,
    normal_chunk_size: int = 64,
    residual_chunk_size: int = 8,
    pairing_chunk_size: int = 2,
    pairing_active_chunk_size: int = 1,
    lmax: int | None = None,
    normal_functional=None,
    stationarity_evaluator: Callable = selfconsistent_stationarity_loss,
    resume_from: str | Path | None = None,
    progress_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    number_augmented_lagrangian: bool = False,
    number_augmented_lagrangian_beta: float | None = None,
    accepted_step_callback: Callable | None = None,
    correct_mean_numbers: bool = False,
    number_correction_tolerance: float = 1.0e-10,
    branch_fixed_point_objective=None,
    post_variation_observer=None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Jointly optimize n/p networks with full recomputation and rollback.

    This API accepts externally built models/basis and writes one atomic,
    physically versioned checkpoint after initialization and every accepted
    update. Resume requires matching physics, basis, architecture, and blocked
    reference. It never calls an eigensolver, Fortran backend, moving-space
    refresh, or branch selector. A small stationarity residual does not certify
    an energy minimum or a physical no-sea branch.
    """
    limits = require_bounded_cgroup()
    if not isinstance(correct_mean_numbers, bool):
        raise TypeError("correct_mean_numbers must be boolean")
    if not math.isfinite(number_correction_tolerance) or number_correction_tolerance <= 0.:
        raise ValueError("number_correction_tolerance must be finite and positive")
    if correct_mean_numbers and any(not hasattr(models[s], "canonical_number_transport")
                                   for s in ("n", "p") if s in models):
        raise TypeError("mean-number correction requires network canonical transport buffers")
    for name, component in (("branch_fixed_point_objective", branch_fixed_point_objective),
                            ("post_variation_observer", post_variation_observer)):
        if component is not None and (not callable(getattr(component, "checkpoint_metadata", None))
                                      or not callable(getattr(component, "evaluate", None))):
            raise TypeError(f"{name} requires checkpoint_metadata() and evaluate()")
    if branch_fixed_point_objective is not None and number_augmented_lagrangian:
        raise ValueError("branch fixed-point and residual-objective number duals cannot be combined")
    if set(models) != {"n", "p"} or set(number_targets) != {"n", "p"}:
        raise ValueError("models and number_targets must contain exactly n and p")
    if branch_fixed_point_objective is not None:
        declared = branch_fixed_point_objective.checkpoint_metadata().get("blocked_species")
        expected = {s for s in ("n", "p") if number_targets[s] % 2}
        if not isinstance(declared, (list, tuple)) or set(declared) != expected:
            raise ValueError("branch policy blocking must match neutron/proton target number parity")
    if chemical_potential_mode not in {"provided", "number_tangent"}:
        raise ValueError("chemical_potential_mode must be provided or number_tangent")
    if chemical_potential_mode == "number_tangent" and train_chemical_potentials:
        raise ValueError("number-tangent multipliers are solved from the fields; disable train_chemical_potentials")
    if max_updates is not None:
        if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 0:
            raise ValueError("max_updates must be a nonnegative integer")
        updates = max_updates
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("updates must be a nonnegative integer")
    if (not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(number_weight) or number_weight < 0):
        raise ValueError("learning_rate must be positive and number_weight nonnegative")
    if isinstance(max_trials_per_update, bool) or not 1 <= max_trials_per_update <= 32:
        raise ValueError("max_trials_per_update must be in [1,32]")
    if minimum_block_overlap is not None and (
            not math.isfinite(float(minimum_block_overlap))
            or not 0.0 < float(minimum_block_overlap) <= 1.0):
        raise ValueError("minimum_block_overlap diagnostic must lie in (0,1]")
    if not isinstance(number_augmented_lagrangian, bool):
        raise TypeError("number_augmented_lagrangian must be boolean")
    al_beta = float(number_weight if number_augmented_lagrangian_beta is None
                    else number_augmented_lagrangian_beta)
    if not math.isfinite(al_beta) or al_beta <= 0.0:
        if number_augmented_lagrangian:
            raise ValueError("augmented-Lagrangian beta must be finite and positive")
    if accepted_step_callback is not None and not callable(accepted_step_callback):
        raise TypeError("accepted_step_callback must be callable")
    if any(isinstance(number_targets[s], bool) or not isinstance(number_targets[s], int)
           or number_targets[s] < 0 for s in ("n", "p")):
        raise ValueError("particle targets must be nonnegative integers")
    if post_variation_observer is not None:
        observer_metadata = post_variation_observer.checkpoint_metadata()
        if observer_metadata.get("mass_number") != sum(number_targets.values()):
            raise ValueError("post-variation mass number differs from neutron plus proton targets")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=resume_from is not None)
    history_path = output / "history.json"
    progress_file = Path(progress_path) if progress_path is not None else output / "progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(checkpoint_dir) if checkpoint_dir is not None else output
    checkpoints.mkdir(parents=True, exist_ok=True)
    initial_mu = chemical_potential_initial or {"n": 0.0, "p": 0.0}
    if set(initial_mu) != {"n", "p"}:
        raise ValueError("chemical_potential_initial must contain n and p")
    if any(not math.isfinite(float(initial_mu[s])) for s in ("n", "p")):
        raise ValueError("initial chemical potentials must be finite")
    production = stationarity_evaluator is selfconsistent_stationarity_loss
    if normal_functional is None and production:
        normal_functional = StreamedPrimitiveNormalFunctional(
            basis, tile_size=normal_tile_size, chunk_size=normal_chunk_size,
            lmax=lmax)
    if production and not isinstance(normal_functional, StreamedPrimitiveNormalFunctional):
        raise TypeError("production training requires the complete PKA1 streamed normal functional")
    basis_info = _basis_metadata(basis)
    model_info = {s: _model_metadata(models[s]) for s in ("n", "p")}
    model_info["state_fingerprints"] = {
        s: _model_state_fingerprint(models[s]) for s in ("n", "p")}
    functional_params = getattr(normal_functional, "params", None)
    if functional_params is None:
        from dpl_rhf.rhf.parameters import pka1_parameters
        functional_params = pka1_parameters()
    if production:
        from dpl_rhf.rhf.parameters import pka1_parameters
        if functional_params != pka1_parameters():
            raise ValueError("production training requires the fixed PKA1 parameter values")
        for name, expected in (("mass_n_mev", functional_params.mass_n),
                               ("mass_p_mev", functional_params.mass_p),
                               ("hbar_c_mev_fm", functional_params.hbar_c)):
            actual = getattr(basis, name, None)
            if actual is not None and not math.isclose(
                    float(actual), float(expected), rel_tol=0., abs_tol=1.e-12):
                raise ValueError(f"primitive kinetic {name} differs from the fixed PKA1 functional")
        if _basis_metadata(normal_functional.basis) != basis_info:
            raise ValueError("normal functional and pairing must use the same primitive and angular basis")
        minimum_lmax = 2 * int(basis.kappa_max)
        if lmax is not None and (isinstance(lmax, bool) or not isinstance(lmax, int)):
            raise ValueError("lmax must be an integer or None")
        effective_lmax = minimum_lmax if lmax is None else lmax
        if effective_lmax < minimum_lmax:
            raise ValueError("lmax is below the declared representation cutoff")
        provider_lmax = (minimum_lmax if normal_functional.lmax is None
                         else normal_functional.lmax)
        if provider_lmax != effective_lmax:
            raise ValueError("normal functional lmax differs from the requested cutoff")
    ph_physics = asdict(functional_params) if is_dataclass(functional_params) else repr(functional_params)
    from .pairing import D1S
    pairing_physics = asdict(D1S)
    normal_functional_info = {
        "class": f"{type(normal_functional).__module__}.{type(normal_functional).__qualname__}",
        "lmax": getattr(normal_functional, "lmax", lmax),
        "tile_size": getattr(normal_functional, "tile_size", normal_tile_size),
        "chunk_size": getattr(normal_functional, "chunk_size", normal_chunk_size),
    }
    metadata = {
        "format": "joint-selfconsistent-neural-training-v2",
        "physics": ("complete PKA1 particle-hole plus full-space D1S pairing action"
                    if production else "injected evaluator smoke/custom objective; production physics not asserted"),
        "production_physics_asserted": bool(production),
        # This selects an implementation, not a physically validated minimum.
        "optimization_target": ("squared_full_stationarity_residual_plus_number_augmented_lagrangian"
                                if production and number_augmented_lagrangian else
                                "squared_full_stationarity_residual_plus_number_penalty"
                                if production else "injected_custom_objective"),
        "objective_limit": "stationarity objective; does not certify an energy minimum",
        "energy_minimum_certified": False,
        "number_constraint_dual_units": "MeV^2" if production else "injected objective units",
        "number_constraint_dual_scope": "numerical dual of residual-squared objective; distinct from physical chemical potential",
        "physical_identity_certified": False,
        "physical_prescription": ({
            "density_argument": "sqrt((j0)^2-|j_space|^2)",
            "rearrangement": "variation of the same density-dependent functional",
            "pairing": "finite-range central Gogny D1S; strength=1; nn and pp; full G/F and represented J",
            "no_sea_constraint": "not supplied or certified by this optimizer",
            "blocking_tracking": "R-only occupied-plane tag transport; overlap diagnostics only; physical identity uncertified",
            "identity_tracking": "diagnostic_only",
            "PH_multipole_cutoff": "configured truncation; angular convergence not certified",
        } if production else None),
        "particle_hole_parameters": ph_physics,
        "pairing_parameters": pairing_physics,
        "normal_functional": normal_functional_info,
        "stationarity_evaluator": (
            f"{getattr(stationarity_evaluator, '__module__', type(stationarity_evaluator).__module__)}."
            f"{getattr(stationarity_evaluator, '__qualname__', type(stationarity_evaluator).__qualname__)}"),
        "branch_fixed_point_objective": (None if branch_fixed_point_objective is None else
                                        branch_fixed_point_objective.checkpoint_metadata()),
        "post_variation_observer": (None if post_variation_observer is None else
                                    post_variation_observer.checkpoint_metadata()),
        "evaluation_mode": ("production_selfconsistent"
                            if stationarity_evaluator is selfconsistent_stationarity_loss
                            else "injected_evaluator_smoke_or_custom"),
        "basis": basis_info,
        "number_targets": {s: int(number_targets[s]) for s in ("n", "p")},
        "models": model_info,
        "options": {
            "train_chemical_potentials": bool(train_chemical_potentials),
            "chemical_potential_mode": chemical_potential_mode,
            "correct_mean_numbers": correct_mean_numbers,
            "number_correction_tolerance": number_correction_tolerance,
            "number_weight": float(number_weight), "lmax": lmax,
            "normal_tile_size": normal_tile_size, "normal_chunk_size": normal_chunk_size,
            "residual_chunk_size": residual_chunk_size,
            "pairing_chunk_size": pairing_chunk_size,
            "pairing_active_chunk_size": pairing_active_chunk_size,
            "learning_rate": float(learning_rate),
            "updates_requested": int(updates),
            "max_trials_per_update": int(max_trials_per_update),
            "minimum_block_overlap_diagnostic": (None if minimum_block_overlap is None
                                                  else float(minimum_block_overlap)),
            "chemical_potential_initial": {s: float(initial_mu[s]) for s in ("n", "p")},
            "number_augmented_lagrangian": bool(number_augmented_lagrangian),
            "number_augmented_lagrangian_beta": al_beta,
        },
        "fortran_used": False, "eigensolver_used": False,
        "basis_refreshes": 0, "physical_result_accepted": False,
        "no_sea_branch_certified": False,
        "resume_supported": True,
        "resource_limits": limits,
    }
    if branch_fixed_point_objective is not None:
        metadata["optimization_target"] = "full_nambu_mixed_branch_fixed_point_plus_number_penalty"
        metadata["objective_limit"] = "conditional finite-space branch witness; not a ground-state certificate"
        metadata["number_constraint_dual_units"] = "dimensionless fixed-point objective units"
        if metadata["physical_prescription"] is not None:
            metadata["physical_prescription"]["no_sea_constraint"] = (
                "conditional full-Nambu branch target with per-field witness validation; "
                "no physical certification of an external witness is inferred")
    physics_prescription = (current_neural_prescription(
        basis_spec=json.dumps(basis_info, sort_keys=True, separators=(",", ":")),
        blocking="pure odd/even Slater reference followed by connected canonical variation; "
                 "explicit active U/V frame available; physical qp identity not certified",
    ) if production else None)
    metadata["physical_prescription_versioned"] = (
        physics_prescription.to_payload() if physics_prescription is not None else None)
    metadata["compatibility_fingerprint"] = _fingerprint(metadata)
    first_parameter = next(models["n"].parameters(), None)
    if first_parameter is None:
        raise ValueError("models must contain parameters")
    device = first_parameter.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("joint training supports only caller-selected CPU or CUDA tensors")
    metadata["device"] = str(device)
    metadata["model_reference_fingerprints"] = {
        s: _model_reference_fingerprint(models[s]) for s in ("n", "p")}
    metadata["resume_compatibility_fingerprint"] = _fingerprint(
        _resume_compatibility(metadata))
    metadata.pop("compatibility_fingerprint", None)
    metadata["compatibility_fingerprint"] = _fingerprint(metadata)
    mus = {s: torch.nn.Parameter(torch.tensor(float(initial_mu[s]), dtype=torch.float64,
                                               device=device),
                                 requires_grad=bool(train_chemical_potentials))
           for s in ("n", "p")}
    for species in ("n", "p"):
        model = models[species]
        params = list(model.parameters())
        if not params or any(p.device != device for p in params):
            raise ValueError("both models must have parameters on one shared device")
        if any(p.dtype not in (torch.float64, torch.complex128) for p in params):
            raise TypeError("models must retain float64/complex128 parameters")
        misplaced_buffers = [name for name, value in model.named_buffers()
                             if value.device != device]
        if misplaced_buffers:
            raise ValueError(f"{species} model buffers are not on {device}: {misplaced_buffers[:8]}")
        if int(getattr(model, "reference_parity", 0)) != (-1 if number_targets[species] % 2 else 1):
            raise ValueError(f"{species}: target parity differs from model reference")
        if number_targets[species] > int(basis.dimension):
            raise ValueError(f"{species}: target number exceeds primitive dimension")
        if production:
            if not isinstance(model, GeneralizedDensityNet):
                raise TypeError("production training requires GeneralizedDensityNet for both species")
            minimum_active = max(40, number_targets[species] + 32)
            if int(model.active_dimension) < minimum_active:
                raise ValueError(f"{species}: production active dimension must be >= {minimum_active}")
    evaluator_options = {
        "normal_functional": normal_functional,
        "normal_tile_size": normal_tile_size, "normal_chunk_size": normal_chunk_size,
        "residual_chunk_size": residual_chunk_size,
        "pairing_chunk_size": pairing_chunk_size,
        "pairing_active_chunk_size": pairing_active_chunk_size, "lmax": lmax,
        "number_weight": 0.0 if number_augmented_lagrangian else number_weight,
    }
    if chemical_potential_mode != "provided":
        evaluator_options["chemical_potential_mode"] = chemical_potential_mode
    optimizer_parameters = _parameter_groups(models, mus)
    optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    initial_modes: dict[str, torch.Tensor | None] = {s: None for s in ("n", "p")}
    previous_tags: dict[str, torch.Tensor | None] = {s: None for s in ("n", "p")}
    number_multipliers = {s: 0.0 for s in ("n", "p")}
    latest_number_corrections = {}
    rollback_transport_state = None
    rollback_number_corrections = {}

    def snapshot_transports():
        if not correct_mean_numbers:
            return None
        return {s: models[s].canonical_number_transport.detach().clone() for s in ("n", "p")}

    def restore_transports(snapshot):
        if snapshot is not None:
            from .network_number_transport import restore_number_transport
            for s in ("n", "p"):
                restore_number_transport(models[s], snapshot[s])

    def correct_network_numbers():
        nonlocal latest_number_corrections
        if not correct_mean_numbers:
            return
        from .network_number_transport import apply_mean_number_correction
        saved = snapshot_transports()
        reports = {}
        try:
            for s in ("n", "p"):
                correction = apply_mean_number_correction(
                    models[s], number_targets[s], tolerance=number_correction_tolerance)
                if not correction.converged:
                    raise _NumberCorrectionFailed(f"{s}: mean-number correction failed: {correction.reason}")
                # Never serialize the entire U/V state stored in retraction.
                reports[s] = {k: getattr(correction, k) for k in (
                    "target", "converged", "reason", "mean_number_before", "mean_number_after",
                    "correction_errors", "accepted_corrections", "transport_updated", "q_unchanged",
                    "corrected_state_error", "transport_before_sha256", "transport_after_sha256")}
        except Exception:
            restore_transports(saved)
            raise
        latest_number_corrections = reports

    history = []
    field_updates = 0
    previous_field_signature = None
    previous_field_vector = None
    model_forward_counts = {"checkpointed_model_forward_executions": 0,
                            "backward_recompute_calls": 0}
    accepted = 0
    attempts = 0
    started_at = time.monotonic()
    phase_started_at = started_at
    current_phase = "init"

    def write_progress(phase, *, status="running", detail=None):
        nonlocal current_phase, phase_started_at
        now = time.monotonic()
        if phase != current_phase:
            current_phase = phase
            phase_started_at = now
        payload = {
            "status": status,
            "current_phase": current_phase,
            "phase_elapsed_seconds": max(0.0, now-phase_started_at),
            "elapsed_seconds": max(0.0, now-started_at),
            "accepted_updates": int(accepted), "attempts": int(attempts),
            "updates_requested": int(updates),
            "checkpoint_last": (str(checkpoints / f"joint_step_{accepted:04d}.pt")
                                if accepted >= 0 and (checkpoints / f"joint_step_{accepted:04d}.pt").is_file()
                                else None),
            "resource_snapshot": _resource_snapshot(),
            "physical_result_accepted": False,
            "no_sea_branch_certified": False,
            "physical_identity_certified": False,
        }
        if detail is not None:
            payload["detail"] = str(detail)
        _atomic_json(progress_file, payload)

    def evaluate_current(*, track_grad):
        nonlocal field_updates
        write_progress("fields")
        context = nullcontext() if track_grad else torch.no_grad()
        with context:
            result, states, outputs = _evaluate(
                models, basis, number_targets, mus, stationarity_evaluator,
                evaluator_options, counts=model_forward_counts if track_grad else None)
            if branch_fixed_point_objective is not None:
                branch_loss, diagnostics = branch_fixed_point_objective.evaluate(
                    states, result, {
                        s: getattr(result, "species_diagnostics", {}).get(s, {}).get(
                            "chemical_potential_mev", mus[s]) for s in ("n", "p")})
                if not isinstance(branch_loss, torch.Tensor) or branch_loss.shape != ():
                    raise TypeError("branch objective must return a scalar tensor")
                if track_grad and not branch_loss.requires_grad:
                    raise RuntimeError("branch objective detached its network/field response")
                result.loss = branch_loss + number_weight * sum(
                    (states[s].particle_number() - float(number_targets[s])).square()
                    for s in ("n", "p"))
                result.branch_diagnostics = diagnostics
            if number_augmented_lagrangian:
                augmented = result.loss
                for species in ("n", "p"):
                    error = states[species].particle_number() - float(number_targets[species])
                    augmented = augmented + number_multipliers[species] * error
                    augmented = augmented + 0.5 * al_beta * error.square()
                result.loss = augmented
        field_updates += 1
        if not bool(torch.isfinite(result.loss.detach())):
            raise _NonfiniteLoss("joint stationarity loss became non-finite")
        return result, states, outputs

    def metrics_for(result, outputs, states, update, attempt, alpha):
        branch, candidate_tags = _branch_metrics(
            models, outputs, number_targets, initial_modes, previous_tags,
            minimum_block_overlap)
        signature, vector = _field_signature(result, int(basis.dimension))
        if previous_field_vector is None:
            change = 0.0
        else:
            denom = torch.linalg.vector_norm(previous_field_vector).clamp_min(1e-30)
            change = float(torch.linalg.vector_norm(vector - previous_field_vector) / denom)
        channels = _json_energy_channels(result)
        species_diagnostics = {
            s: {key: float(value.detach().cpu()) for key, value in result.species_diagnostics[s].items()
                if isinstance(value, torch.Tensor) and value.numel() == 1}
            for s in ("n", "p") if hasattr(result, "species_diagnostics")
        }
        changes = {}
        encoder_sq, head_sq = 0.0, 0.0
        for species in ("n", "p"):
            for name, parameter in models[species].named_parameters():
                before = initial_parameters[(species, name)]
                amount = float(torch.linalg.vector_norm(parameter.detach() - before))
                changes[f"{species}.{name}"] = amount
                if name.startswith(("radial_encoder.", "kappa_embedding.", "m_embedding.",
                                    "component_embedding.", "state_embedding.")):
                    encoder_sq += amount * amount
                elif name.startswith("output_head."):
                    head_sq += amount * amount
        row = {
            "accepted_update": int(update), "attempt": int(attempt),
            "line_search_alpha": float(alpha),
            "loss": float(result.loss.detach()),
            "total_energy_mev": float(result.total_energy.detach()),
            "particle_hole_energy_mev": float(result.particle_hole_energy.detach()),
            "pairing_energy_mev": float(result.pairing_energy.detach()),
            "energy_channels_mev": channels,
            "normal_fields_fingerprint": signature,
            "field_change_relative": change,
            "field_change_metric": "relative norm of h, active Delta gap, and deterministic full-Delta action probes",
            "field_updates": int(field_updates),
            "chemical_potentials_mev": _effective_chemical_potentials(result, mus),
            "chemical_potential_mode": chemical_potential_mode,
            "mean_number_corrections": copy.deepcopy(latest_number_corrections),
            "branch_diagnostics": getattr(result, "branch_diagnostics", None),
            "species": branch,
            "number_constraint_duals": dict(number_multipliers),
            "number_augmented_lagrangian": bool(number_augmented_lagrangian),
            "number_augmented_lagrangian_beta": al_beta,
            "species_diagnostics": species_diagnostics,
            "nn_parameter_change_l2": changes,
            "nn_encoder_change_l2": math.sqrt(encoder_sq),
            "nn_head_change_l2": math.sqrt(head_sq),
            "model_forward_counts": dict(model_forward_counts),
            "physical_result_accepted": False,
            "no_sea_branch_certified": False,
            "physical_identity_certified": False,
            "resume_supported": True,
        }
        if post_variation_observer is not None:
            with torch.no_grad():
                row["post_variation_observables"] = post_variation_observer.evaluate(states)
        return row, candidate_tags, signature, vector

    def checkpoint_payload(step):
        payload = {
            "checkpoint_format": "joint-selfconsistent-neural-training-v2",
            "metadata": metadata,
            "model_states": {s: models[s].state_dict() for s in ("n", "p")},
            "optimizer_state": optimizer.state_dict(),
            "chemical_potentials": {s: mus[s].detach().cpu() for s in ("n", "p")},
            "accepted_updates": int(step), "history": copy.deepcopy(history),
            "rng_state": _rng_state(), "field_updates": int(field_updates),
            "attempts": int(attempts),
            "initial_blocked_tags": {s: (None if initial_modes[s] is None
                                          else initial_modes[s].detach().cpu())
                                      for s in ("n", "p")},
            "previous_blocked_tags": {s: (None if previous_tags[s] is None
                                           else previous_tags[s].detach().cpu())
                                       for s in ("n", "p")},
            "number_constraint_duals": dict(number_multipliers),
            "mean_number_corrections": copy.deepcopy(latest_number_corrections),
            "physical_result_accepted": False, "no_sea_branch_certified": False,
            "physical_identity_certified": False,
            "fortran_used": False, "eigensolver_used": False, "basis_refreshes": 0,
        }
        if physics_prescription is not None:
            payload.update(checkpoint_prescription_fields(physics_prescription))
        return payload

    def incomplete_report(error):
        try:
            write_progress("incomplete", status="incomplete", detail=error)
        except Exception:
            pass
        last_checkpoint = checkpoints / f"joint_step_{accepted:04d}.pt"
        report = {
            **metadata, "status": "incomplete", "updates_requested": int(updates),
            "accepted_updates": int(accepted), "attempts": int(attempts),
            "field_updates": int(field_updates), "history_file": history_path.name,
            "progress_file": str(progress_file),
            "checkpoint_last": (str(last_checkpoint) if last_checkpoint.is_file() else None),
            "error": f"{type(error).__name__}: {error}",
            "physical_result_accepted": False, "no_sea_branch_certified": False,
            "physical_identity_certified": False, "resume_supported": True,
        }
        _atomic_json(output / "training_report.json", report)
        return report

    def restore_parameters(values):
        nonlocal latest_number_corrections
        try:
            restore_transports(rollback_transport_state)
            latest_number_corrections = copy.deepcopy(rollback_number_corrections)
            with torch.no_grad():
                for parameter, value in zip(optimizer_parameters, values):
                    parameter.copy_(value)
        except Exception as error:
            return error
        return None

    def restore_transaction(values, optimizer_state):
        error = restore_parameters(values)
        if error is not None:
            return error
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as error:
            return error
        return None

    if resume_from is not None:
        source_checkpoint = Path(resume_from)
        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {source_checkpoint}")
        payload = torch.load(source_checkpoint, map_location=device, weights_only=False)
        checkpoint_format = payload.get("checkpoint_format")
        saved_metadata = payload.get("metadata", {})
        if checkpoint_format != "joint-selfconsistent-neural-training-v2":
            raise ValueError(
                "checkpoint is a legacy/frozen format and cannot resume this self-consistent run"
            )
        if physics_prescription is not None:
            validate_checkpoint_prescription(payload, physics_prescription)
        saved_fingerprint = saved_metadata.get("resume_compatibility_fingerprint")
        if saved_fingerprint != metadata["resume_compatibility_fingerprint"]:
            raise ValueError(
                "resume checkpoint is incompatible with the requested physics, basis, "
                "model architecture/reference, device, or optimizer options"
            )
        if set(payload.get("model_states", {})) != {"n", "p"}:
            raise ValueError("resume checkpoint is missing n/p network states")
        for species in ("n", "p"):
            models[species].load_state_dict(payload["model_states"][species], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        for species in ("n", "p"):
            with torch.no_grad():
                mus[species].copy_(payload["chemical_potentials"][species].to(device))
        tags = payload.get("initial_blocked_tags")
        prior = payload.get("previous_blocked_tags")
        if not isinstance(tags, dict) or not isinstance(prior, dict):
            raise ValueError("resume checkpoint has no blocked-tag state")
        for species in ("n", "p"):
            initial_modes[species] = (None if tags[species] is None
                                      else tags[species].to(device=device, dtype=torch.complex128))
            previous_tags[species] = (None if prior[species] is None
                                      else prior[species].to(device=device, dtype=torch.complex128))
            if number_targets[species] % 2 and (
                    initial_modes[species] is None or previous_tags[species] is None):
                raise ValueError(f"resume checkpoint has no odd {species} occupied tag")
        multipliers = payload.get("number_constraint_duals")
        if not isinstance(multipliers, dict) or set(multipliers) != {"n", "p"}:
            raise ValueError("resume checkpoint has no number multiplier state")
        number_multipliers.update({s: float(multipliers[s]) for s in ("n", "p")})
        _restore_rng_state(payload.get("rng_state"))
        accepted = int(payload.get("accepted_updates", 0))
        attempts = int(payload.get("attempts", 0))
        field_updates = int(payload.get("field_updates", 0))
        history = copy.deepcopy(payload.get("history", []))
        latest_number_corrections = copy.deepcopy(payload.get("mean_number_corrections", {}))
        if accepted < 0 or accepted > updates or not isinstance(history, list):
            raise ValueError("resume checkpoint has invalid accepted-step/history state")
    else:
        # Project each raw blocked vector into its occupied generalized-density
        # plane once. Subsequent identity is transported from accepted tag to
        # trial plane; trial tags are not committed unless that trial is accepted.
        try:
            correct_network_numbers()
            for species in ("n", "p"):
                model = models[species]
                blocked_index = getattr(model, "blocked_index", None)
                if blocked_index is None:
                    if number_targets[species] % 2:
                        raise ValueError(f"{species}: odd target requires an occupied blocked mode")
                    continue
                if not hasattr(model, "forward_with_blocked_mode"):
                    raise ValueError(f"{species}: blocked model has no raw blocked-mode output")
                with torch.no_grad(), _forbid_spectral_calls():
                    _q0, _gd0, raw_mode = model.forward_with_blocked_mode()
                if raw_mode is None:
                    raise ValueError(f"{species}: raw blocked mode is unavailable")
                initial_modes[species] = raw_mode.detach().clone()
                previous_tags[species] = raw_mode.detach().clone()
        except Exception as error:
            return incomplete_report(error)

    initial_parameters = {
        (s, name): value.detach().clone()
        for s in ("n", "p") for name, value in models[s].named_parameters()
    }

    try:
        if correct_mean_numbers and resume_from is not None:
            with torch.no_grad():
                for s in ("n", "p"):
                    _, gd = models[s]()
                    if abs(float(torch.trace(gd.rho).real)-number_targets[s]) > number_correction_tolerance:
                        raise ValueError("resume state violates its mean-number correction contract")
        write_progress("init", detail="evaluating accepted checkpoint state")
        result, _states, outputs = evaluate_current(track_grad=False)
        initial_row, candidate_tags, initial_signature, initial_vector = metrics_for(
            result, outputs, _states, accepted, attempts, 0.0)
    except Exception as error:
        return incomplete_report(error)
    previous_tags.update(candidate_tags)
    previous_field_signature, previous_field_vector = initial_signature, initial_vector
    if resume_from is None:
        initial_row["event"] = "initial_state"
        history.append(initial_row)
    else:
        initial_row["event"] = "resume_state"
        if not history:
            history.append(initial_row)
    _atomic_json(history_path, history)
    checkpoint_file = checkpoints / f"joint_step_{accepted:04d}.pt"
    _atomic_torch_save(checkpoint_file, checkpoint_payload(accepted))
    write_progress("accepted", detail="accepted state checkpoint is current")
    if accepted_step_callback is not None:
        try:
            accepted_step_callback(accepted, result, _states, outputs, {
                "phase": "accepted", "event": "initial_state" if resume_from is None else "resume_state",
                "chemical_potentials_mev": _effective_chemical_potentials(result, mus),
                "number_constraint_duals": dict(number_multipliers),
                "blocked_metrics": initial_row["species"],
                "physical_identity_certified": False,
            })
        except Exception as error:
            return incomplete_report(error)
    del result, _states, outputs

    # A fully rejected line search restores the exact base parameters,
    # transports and optimizer moments, so its loss and clipped gradient are
    # reused instead of repeating the full PH+D1S forward/backward pass.
    cached_base = None
    while accepted < updates:
        if attempts >= updates * max_trials_per_update:
            break
        attempts += 1
        optimizer.zero_grad(set_to_none=True)
        if cached_base is not None:
            old_loss_value, cached_gradients = cached_base
            for parameter, gradient in zip(optimizer_parameters, cached_gradients):
                parameter.grad = None if gradient is None else gradient.clone()
            write_progress("backward", detail="reusing unchanged base gradient after rejected line search")
        else:
            try:
                old_result, _states, _outputs = evaluate_current(track_grad=True)
            except Exception as error:
                return incomplete_report(error)
            old_loss = old_result.loss
            if not bool(torch.isfinite(old_loss.detach())):
                return incomplete_report(FloatingPointError(
                    "joint loss is non-finite before Adam update"))
            try:
                write_progress("backward")
                old_loss.backward()
                bad_gradients = [index for index, parameter in enumerate(optimizer_parameters)
                                 if parameter.grad is not None
                                 and not bool(torch.isfinite(parameter.grad).all())]
                if bad_gradients:
                    del old_result, _states, _outputs, old_loss
                    optimizer.zero_grad(set_to_none=True)
                    return incomplete_report(FloatingPointError(
                        f"nonfinite optimizer gradients at parameter indices {bad_gradients[:8]}"))
                torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=10.0)
            except Exception as error:
                del old_result, _states, _outputs, old_loss
                return incomplete_report(error)
            old_loss_value = float(old_loss.detach())
            cached_base = (old_loss_value, [None if p.grad is None else p.grad.detach().clone()
                                            for p in optimizer_parameters])
            del old_result, _states, _outputs, old_loss
        old_values = [p.detach().clone() for p in optimizer_parameters]
        rollback_transport_state = snapshot_transports()
        rollback_number_corrections = copy.deepcopy(latest_number_corrections)
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        try:
            optimizer.step()
        except Exception as error:
            rollback_error = restore_transaction(old_values, old_optimizer)
            if rollback_error is not None:
                error = RuntimeError(f"optimizer step failed ({error}); rollback failed ({rollback_error})")
            return incomplete_report(error)
        proposed = [p.detach().clone() for p in optimizer_parameters]
        proposed_optimizer = copy.deepcopy(optimizer.state_dict())
        accepted_trial = False
        trial_record = None
        accepted_candidate_tags = None
        accepted_trial_data = None
        for backtrack in range(12):
            alpha = 0.5 ** backtrack
            restore_transports(rollback_transport_state)
            with torch.no_grad():
                for parameter, old, new in zip(optimizer_parameters, old_values, proposed):
                    parameter.copy_(old + alpha * (new - old))
            try:
                correct_network_numbers()
                write_progress("trial", detail=f"attempt={attempts}, backtrack={backtrack}")
                trial, _trial_states, trial_outputs = evaluate_current(track_grad=False)
            except (_NonfiniteLoss, _NumberCorrectionFailed):
                # Reject this step length only; a shorter step may be valid.
                rollback_error = restore_parameters(old_values)
                if rollback_error is not None:
                    return incomplete_report(RuntimeError(
                        f"rejected-trial rollback failed ({rollback_error})"))
                continue
            except Exception as error:
                rollback_error = restore_transaction(old_values, old_optimizer)
                if rollback_error is not None:
                    error = RuntimeError(f"trial evaluation failed ({error}); rollback failed ({rollback_error})")
                return incomplete_report(error)
            if (not bool(torch.isfinite(trial.loss.detach()))
                    or float(trial.loss.detach()) >= old_loss_value):
                rollback_error = restore_parameters(old_values)
                if rollback_error is not None:
                    return incomplete_report(RuntimeError(
                        f"nonfinite/nondecreasing trial rollback failed ({rollback_error})"))
                del trial, _trial_states, trial_outputs
                continue
            try:
                trial_record, accepted_candidate_tags, trial_signature, trial_vector = metrics_for(
                    trial, trial_outputs, _trial_states, accepted + 1, attempts, alpha)
            except _TrialRejected:
                rollback_error = restore_parameters(old_values)
                if rollback_error is not None:
                    return incomplete_report(RuntimeError(
                        f"invalid-state trial rollback failed ({rollback_error})"))
                del trial, _trial_states, trial_outputs
                continue
            except Exception as error:
                rollback_error = restore_transaction(old_values, old_optimizer)
                if rollback_error is not None:
                    error = RuntimeError(f"trial metrics failed ({error}); rollback failed ({rollback_error})")
                return incomplete_report(error)
            optimizer.load_state_dict(proposed_optimizer)
            accepted_trial_data = (trial, _trial_states, trial_outputs)
            accepted_trial = True
            break
        if not accepted_trial:
            rollback_error = restore_transaction(old_values, old_optimizer)
            if rollback_error is not None:
                return incomplete_report(RuntimeError(f"line-search rollback failed ({rollback_error})"))
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            write_progress("accepted", detail="all trials rejected; optimizer restored and learning rate reduced")
            continue
        accepted += 1
        cached_base = None
        previous_tags.update(accepted_candidate_tags)
        previous_field_signature, previous_field_vector = trial_signature, trial_vector
        if number_augmented_lagrangian:
            trial_state = accepted_trial_data[1]
            applied_multipliers = dict(number_multipliers)
            for species in ("n", "p"):
                error = float((trial_state[species].particle_number()
                               - float(number_targets[species])).detach())
                number_multipliers[species] += al_beta * error
            trial_record["number_constraint_duals_applied"] = applied_multipliers
            trial_record["number_constraint_duals_next"] = dict(number_multipliers)
        trial_record["event"] = "accepted_update"
        history.append(trial_record)
        _atomic_json(history_path, history)
        _atomic_torch_save(checkpoints / f"joint_step_{accepted:04d}.pt",
                           checkpoint_payload(accepted))
        write_progress("accepted", detail=f"accepted update {accepted} checkpointed")
        if accepted_step_callback is not None:
            try:
                trial, trial_states, trial_outputs = accepted_trial_data
                accepted_step_callback(accepted, trial, trial_states, trial_outputs, {
                    "phase": "accepted", "event": "accepted_update",
                    "chemical_potentials_mev": _effective_chemical_potentials(trial, mus),
                    "number_constraint_duals": dict(number_multipliers),
                    "blocked_metrics": trial_record["species"],
                    "physical_identity_certified": False,
                })
            except Exception as error:
                return incomplete_report(error)
        del accepted_trial_data

    status = "complete" if accepted == updates else "line_search_exhausted"
    last_checkpoint = checkpoints / f"joint_step_{accepted:04d}.pt"
    report = {
        **metadata, "status": status, "updates_requested": int(updates),
        "accepted_updates": int(accepted), "attempts": int(attempts),
        "field_updates": int(field_updates), "history_file": history_path.name,
        "checkpoint_last": str(last_checkpoint),
        "chemical_potentials_mev": history[-1]["chemical_potentials_mev"],
        "physical_result_accepted": False, "no_sea_branch_certified": False,
        "resume_supported": True,
        "fortran_used": False, "eigensolver_used": False, "basis_refreshes": 0,
    }
    write_progress("accepted", status=status, detail="training call finished")
    _atomic_json(output / "training_report.json", report)
    return report
