"""Variational training driver for explicit, number-parity blocked RHFB states.

The optimizer consumes only a complete, differentiable RHFB functional.  The
component backend and toy prototype are useful diagnostics, but cannot enter a
physical minimization through this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .basis import AngularRadialBasis, _shells_to_capacity
from .checkpoint import load_checkpoint, save_checkpoint
from .config import BlockedRHFBConfig
from .manifold import slater_reference
from .manifold import blocking_overlap
from .network import GeneralizedDensityNet
from .operators import DensityFactors
from .result import append_jsonl, atomic_json, json_value, save_observables
from .prescription import current_neural_prescription


REQUIRED_PKA1_CHANNELS = (
    "kinetic", "d1s_pairing",
    "sigma_direct", "sigma_fock", "omega_direct", "omega_fock",
    "rho_vector_direct", "rho_vector_fock",
    "rho_tensor_direct", "rho_tensor_fock",
    "rho_vt_direct", "rho_vt_fock", "rho_tv_direct", "rho_tv_fock",
    "pion_pv_direct", "pion_pv_fock", "coulomb_direct", "coulomb_fock",
    "rho_tensor_contact", "pion_pv_contact", "density_rearrangement",
)
REQUIRED_NO_SEA_ALGORITHM_GATES = (
    "candidate_canonical_stationarity_le_1e-6",
    "candidate_matches_full_variational_moving_projector_oracle",
    "candidate_structure_number_and_sea_support",
    "field_dependent_candidate_gradient_fd_le_1e-3",
    "fixed_field_sign_projector_vs_reference",
    "independent_full_variational_moving_projector_oracle",
    "independent_reference_gradient_stable_fd_checked",
)


class PhysicalGateError(RuntimeError):
    def __init__(self, failures: list[str], audit: dict[str, Any] | None = None):
        self.failures = failures
        self.audit = audit or {}
        super().__init__("; ".join(failures))


class _TrainingEigenCounter(TorchDispatchMode):
    """Count dense torch eigensolver calls made by the training path.

    The network and physical functional are deliberately matrix-free with
    respect to eigensolvers. Keeping this dispatch observer around the actual
    state/evaluation calls makes the saved zero count evidence from execution,
    rather than a constant copied into the result schema.
    """

    _EIGEN_OPS = ("linalg_eigh", "linalg_eigvalsh", "linalg_eig", "linalg_eigvals",
                  "linalg_eigvals", "symeig", "eig")

    def __init__(self, primitive_dimension: int, active_dimensions: Mapping[str, int]):
        super().__init__()
        self.primitive_dimension = int(primitive_dimension)
        self.nambu_dimensions = {2 * int(value) for value in active_dimensions.values()}
        self.counts = {"training_dirac_eigensolves": 0,
                       "training_bogoliubov_eigensolves": 0,
                       "training_unclassified_eigensolves": 0}
        self.operations: list[dict[str, Any]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        name = str(func)
        if any(op in name for op in self._EIGEN_OPS):
            tensor = next((item for item in args if isinstance(item, torch.Tensor)), None)
            dimension = int(tensor.shape[-1]) if tensor is not None and tensor.ndim >= 2 else None
            if dimension == self.primitive_dimension:
                category = "training_dirac_eigensolves"
            elif dimension in self.nambu_dimensions:
                category = "training_bogoliubov_eigensolves"
            else:
                category = "training_unclassified_eigensolves"
            self.counts[category] += 1
            self.operations.append({"operation": name, "dimension": dimension,
                                    "category": category})
        return output


def _require_no_training_eigensolves(counter: _TrainingEigenCounter) -> None:
    if sum(counter.counts.values()):
        raise PhysicalGateError(
            ["a torch eigensolver ran on the no-refresh training path"],
            {"solver_counts": {**counter.counts,
                               "instrumentation": "torch_dispatch_eigensolver_ops",
                               "eigensolver_operations": counter.operations}},
        )


@dataclass(frozen=True)
class BlockingSpec:
    label: str
    index: int
    kappa: int
    m2: int
    operation: str = "particle"


def _parse_selector(selector: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*kappa\s*=\s*(-?\d+)\s*,\s*m2\s*=\s*(-?\d+)\s*", selector)
    return (int(match.group(1)), int(match.group(2))) if match else None


def resolve_blocking(labels: tuple[str, ...], particle_number: int, selector: str,
                     *, species: str, z: int, n: int) -> BlockingSpec | None:
    """Resolve one explicit blocked basis mode from stable spectroscopic labels.

    `auto` chooses the next particle in the ordered spherical filling.  O-17
    pins the requested 1d5/2, m=+5/2 neutron mode rather than relying on label
    ordering within the degenerate multiplet.
    """
    if particle_number % 2 == 0:
        if selector not in {"auto", "none"}:
            raise ValueError(f"even {species} number cannot use one-mode blocking")
        return None
    requested = _parse_selector(selector)
    if selector == "none":
        raise ValueError(f"odd {species} number requires a blocked mode")
    if requested is None and selector != "auto":
        # Accept an exact basis label as well as the compact kappa/m2 syntax.
        matches = [i for i, label in enumerate(labels) if label == selector]
        if not matches:
            raise ValueError(f"unknown {species} blocked mode selector {selector!r}")
        index = matches[0]
    else:
        if requested is None and z == 8 and n == 9 and species == "n":
            requested = (-3, 5)
        if requested is None:
            index = particle_number - 1
        else:
            kappa, m2 = requested
            matches = [i for i, label in enumerate(labels)
                       if f"kappa={kappa}:m2={m2}" in label]
            if not matches:
                raise ValueError(
                    f"blocked mode kappa={kappa}, m2={m2} is outside the active space"
                )
            index = matches[0]
    if index >= len(labels):
        raise ValueError("active space does not contain the auto-selected blocked mode")
    label = labels[index]
    match = re.search(r"kappa=(-?\d+):m2=(-?\d+)", label)
    if not match:
        raise ValueError(f"cannot recover kappa/m2 from active label {label!r}")
    return BlockingSpec(label, index, int(match.group(1)), int(match.group(2)))


def reference_for_species(labels: tuple[str, ...], number: int,
                          block: BlockingSpec | None):
    """Make a pure Slater reference with an explicit particle/hole toggle."""
    occupations = torch.zeros(len(labels), dtype=torch.float64)
    if block is None:
        occupations[:number] = 1.0
        return slater_reference(occupations), None
    # Build the even core by filling the lowest available modes, leaving the
    # selected odd mode empty; slater_reference then explicitly adds it.
    filled = 0
    for index in range(len(labels)):
        if index == block.index:
            continue
        if filled < number - 1:
            occupations[index] = 1.0
            filled += 1
    if filled != number - 1:
        raise ValueError("active space is too small for the blocked odd reference")
    return slater_reference(occupations, blocked_index=block.index,
                            blocking_operation="particle"), block.index


def update_blocked_mode_tracker(previous: dict[str, torch.Tensor | None],
                                current: dict[str, torch.Tensor | None],
                                *, minimum_overlap: float = 0.5) -> dict[str, torch.Tensor | None]:
    """Require adjacent-iterate continuity of the full Nambu blocked mode."""
    overlaps: dict[str, torch.Tensor | None] = {}
    for species in ("n", "p"):
        before, now = previous.get(species), current.get(species)
        if before is None and now is None:
            overlaps[species] = None
            continue
        if before is None and now is not None:
            # First iterate establishes the adjacent-overlap reference.
            overlaps[species] = None
            continue
        if now is None:
            raise RuntimeError(f"{species} blocked Nambu mode disappeared during training")
        if now.ndim != 1 or before.shape != now.shape or now.dtype != torch.complex128:
            raise ValueError(f"{species} blocked modes must be equal-length complex128 vectors")
        now_norm = torch.linalg.vector_norm(now)
        before_norm = torch.linalg.vector_norm(before)
        if (not bool(torch.isfinite(now).all()) or not bool(torch.isfinite(before).all())
                or abs(float(now_norm.detach()) - 1.0) > 1.0e-8
                or abs(float(before_norm.detach()) - 1.0) > 1.0e-8):
            raise ValueError(f"{species} blocked Nambu mode must be finite and normalized")
        overlap = blocking_overlap(before, now)
        if float(overlap.detach()) < minimum_overlap:
            raise RuntimeError(
                f"{species} blocked Nambu-mode overlap {float(overlap.detach()):.6g} "
                f"fell below {minimum_overlap:.6g}; refusing an untracked blocking switch"
            )
        overlaps[species] = overlap
    previous.update({species: (None if vector is None else vector.detach().clone())
                     for species, vector in current.items()})
    return overlaps


def _basis_spec(basis: AngularRadialBasis) -> dict[str, Any]:
    return {"radius_fm": basis.radius_fm, "spacing_fm": basis.spacing_fm,
            "kappa_max": basis.kappa_max, "dimension": basis.dimension,
            "n_radial": basis.n_radial, "channels": len(basis.channels)}


def _blocking_spec(blocks: dict[str, BlockingSpec | None]) -> dict[str, Any]:
    return {key: (None if block is None else {
        "label": block.label, "index": block.index, "kappa": block.kappa,
        "m2": block.m2, "operation": block.operation,
    }) for key, block in blocks.items()}


def _pack_mode_vectors(vectors: dict[str, torch.Tensor | None]) -> dict[str, Any]:
    return {key: (None if value is None else {
        "real": value.real.cpu().tolist(), "imag": value.imag.cpu().tolist(),
    }) for key, value in vectors.items()}


def _unpack_mode_vectors(payload: Mapping[str, Any], *, device: torch.device) -> dict[str, torch.Tensor | None]:
    result = {}
    for species in ("n", "p"):
        packed = payload.get(species)
        result[species] = (None if packed is None else torch.complex(
            torch.as_tensor(packed["real"], dtype=torch.float64, device=device),
            torch.as_tensor(packed["imag"], dtype=torch.float64, device=device),
        ))
    return result


def functional_from_module() -> Callable[..., Any] | None:
    try:
        module = importlib.import_module("dpl_rhf.rhfb.functional")
    except ImportError:
        return None
    function = getattr(module, "evaluate_rhfb_energy", None)
    return function if callable(function) else None


def functional_preflight_from_module() -> tuple[bool, dict[str, Any]]:
    """Read a cheap module-level closure manifest before allocating a basis.

    `evaluate_rhfb_energy` being importable is insufficient: an incomplete
    implementation could trigger an enormous first evaluation before the
    trainer sees its audit. The readiness manifest is a lightweight statement
    of implementation/closure coverage; each training evaluation still has to
    pass the stricter numerical audit below.
    """
    try:
        module = importlib.import_module("dpl_rhf.rhfb.functional")
    except ImportError as exc:
        return False, {"status": "unavailable", "error": str(exc)}
    readiness = getattr(module, "functional_readiness", None)
    if not callable(readiness):
        return False, {"status": "manifest_missing",
                       "required_export": "functional_readiness"}
    try:
        manifest = readiness()
    except Exception as exc:
        return False, {"status": "manifest_error", "error": str(exc)}
    if not isinstance(manifest, Mapping):
        return False, {"status": "manifest_invalid", "manifest_type": type(manifest).__name__}
    channel_info = manifest.get("channels")
    failures = []
    if manifest.get("ready") is not True:
        failures.append("functional readiness manifest is not ready")
    if not isinstance(channel_info, Mapping):
        failures.append("functional readiness manifest has no channel map")
        channel_info = {}
    for channel in REQUIRED_PKA1_CHANNELS:
        evidence = channel_info.get(channel)
        if not isinstance(evidence, Mapping):
            failures.append(f"functional readiness lacks channel {channel}")
            continue
        if evidence.get("implemented") is not True:
            failures.append(f"functional readiness marks {channel} unimplemented")
        if evidence.get("closure_status") != "passed":
            failures.append(f"functional readiness closure for {channel} is not passed")
    return not failures, {"status": "ready" if not failures else "blocked",
                          "manifest": dict(manifest), "unresolved": failures}


def active_space_failures(config: BlockedRHFBConfig) -> list[str]:
    failures = []
    for species, number, configured in (("neutron", config.n, config.active_n),
                                        ("proton", config.z, config.active_p)):
        target = configured if configured is not None else max(40, number + 32)
        required = max(abs(int(shell[1])) for shell in _shells_to_capacity(target))
        if required > config.kappa_max:
            failures.append(
                f"{species} active target {target} needs kappa_max>={required}, "
                f"got {config.kappa_max}"
            )
    return failures


def validate_functional_evaluation(evaluation: Any) -> list[str]:
    """Check traceable per-channel closures and accounting before optimization."""
    failures: list[str] = []
    if not isinstance(evaluation, Mapping):
        # Dataclass/named tuple objects are normalized without trusting a
        # `complete=True` field as the sole evidence.
        if hasattr(evaluation, "__dict__"):
            record = vars(evaluation)
        else:
            failures.append("functional did not return a structured evaluation")
            return failures
    else:
        record = evaluation
    energy = record.get("total_energy")
    if not isinstance(energy, torch.Tensor) or energy.numel() != 1:
        failures.append("functional total_energy must be a differentiable scalar tensor")
    elif not energy.requires_grad or not bool(torch.isfinite(energy.detach())):
        failures.append("functional total_energy is detached or non-finite")
    terms = record.get("energy_terms")
    if not isinstance(terms, Mapping) or not terms:
        failures.append("functional energy_terms are missing")
    elif any(not isinstance(value, torch.Tensor) or value.numel() != 1
             or not bool(torch.isfinite(value.detach())) for value in terms.values()):
        failures.append("every functional energy term must be a finite scalar tensor")
    audit = record.get("audit")
    if not isinstance(audit, Mapping):
        failures.append("functional channel audit is missing")
        return failures
    if audit.get("complete") is not True:
        failures.append("functional has not passed its complete-channel acceptance audit")
    channels = audit.get("channels")
    if not isinstance(channels, Mapping):
        failures.append("functional audit has no per-channel evidence")
        return failures
    for channel in REQUIRED_PKA1_CHANNELS:
        evidence = channels.get(channel)
        if not isinstance(evidence, Mapping):
            failures.append(f"missing auditable PKA1 channel: {channel}")
            continue
        for key in ("implemented", "energy_identity_residual", "identity_tolerance",
                    "directional_fd_residual", "directional_fd_tolerance", "gradient_finite"):
            if key not in evidence:
                failures.append(f"{channel} evidence missing {key}")
        if evidence.get("implemented") is not True or evidence.get("gradient_finite") is not True:
            failures.append(f"{channel} implementation/gradient evidence did not pass")
        for residual_key, tolerance_key in (("energy_identity_residual", "identity_tolerance"),
                                            ("directional_fd_residual", "directional_fd_tolerance")):
            try:
                residual = float(evidence[residual_key])
                tolerance = float(evidence[tolerance_key])
                valid = math.isfinite(residual) and math.isfinite(tolerance) and residual >= 0 and tolerance > 0
            except (TypeError, KeyError, ValueError):
                valid = False
                residual, tolerance = math.inf, 0.0
            if not valid or residual > tolerance:
                failures.append(f"{channel} {residual_key} exceeds or lacks its recorded tolerance")
    for residual_key, tolerance_key in (
        ("total_energy_identity_residual", "total_energy_identity_tolerance"),
    ):
        try:
            residual = float(audit[residual_key])
            tolerance = float(audit[tolerance_key])
            valid = math.isfinite(residual) and math.isfinite(tolerance) and residual >= 0 and tolerance > 0
        except (KeyError, TypeError, ValueError):
            valid = False
            residual, tolerance = math.inf, 0.0
        if not valid or residual > tolerance:
            failures.append("total energy accounting identity exceeds or lacks its recorded tolerance")
    if audit.get("no_sea_status") not in {"passed", "experimental"}:
        failures.append("no-sea branch algorithm is not cleared for physical-functional training")
    for key, description in (
        ("normal_branch_status", "normal Dirac branch certificate"),
        ("anomalous_sea_branch_status", "anomalous Dirac-sea branch certificate"),
    ):
        if audit.get(key) != "passed":
            failures.append(f"{description} has not passed")
    if audit.get("independent_physical_reference_status") not in {
        "passed", "unverified", "pending"
    }:
        failures.append("independent physical-reference result is marked failed")
    if audit.get("rearrangement_status") != "passed":
        failures.append("density-dependent rearrangement closure has not passed")
    if not callable(record.get("stationarity")):
        failures.append("functional exposes no full-space stationarity callback")
    return failures


def validate_no_sea_report(path: str | Path) -> tuple[bool, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return False, {"status": "missing", "path": str(path)}
    import json
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, {"status": "invalid", "path": str(path), "error": str(exc)}
    gates = report.get("gates", {})

    algorithm_gates_pass = isinstance(gates, Mapping) and all(
        gates.get(key) is True for key in REQUIRED_NO_SEA_ALGORITHM_GATES
    )
    equivalence = gates.get("density_dependent_self_consistent_equivalence") if isinstance(gates, Mapping) else None
    return algorithm_gates_pass, {
        "status": report.get("status", "unknown"),
        "prevalidation_algorithm_gates": "passed" if algorithm_gates_pass else "failed",
        "moving_projector_physical_equivalence": (
            "passed" if equivalence is True else "failed" if equivalence is False else "unverified"
        ),
        "path": str(path), "gates": gates,
    }


def prepare_networks(config: BlockedRHFBConfig, basis: AngularRadialBasis,
                     *, device: torch.device | str = "cpu", network_class=None):
    """Build O-17 neutron/proton network factors and explicit odd reference."""
    network_class = GeneralizedDensityNet if network_class is None else network_class
    network_data = {}
    blocks = {}
    for species, number, active, selector in (
        ("n", config.n, config.active_n, config.block_n),
        ("p", config.z, config.active_p, config.block_p),
    ):
        target = active if active is not None else max(40, number + 32)
        q, labels = basis.build_initial_q(
            number, species=species, nucleus_a=config.a, proton_number=config.z,
            minimum=target, empty_states=0, device=device,
        )
        block = resolve_blocking(labels, number, selector, species=species,
                                 z=config.z, n=config.n)
        reference, blocked_index = reference_for_species(labels, number, block)
        net = network_class(
            basis, q, labels, reference, blocked_index=blocked_index,
        ).to(device)
        _initialize_candidate_shape(net, basis, config.candidate_shape,
                                    config.candidate_seed_amplitude)
        network_data[species] = {"network": net, "labels": labels,
                                 "initial_q": q, "reference": reference}
        blocks[species] = block
    return network_data, blocks


def _initialize_candidate_shape(network, basis, shape: str, amplitude: float) -> None:
    """Seed a spherical, axial, or triaxial m-scheme variational candidate.

    The shape perturbation adds small, symmetry-allowed channel admixtures to
    the trainable primitive Q coordinates.  It does not constrain later
    optimization: all candidate seeds retain the same unrestricted network.
    Axial seeds conserve doubled m and couple upper-component orbital angular
    momenta differing by two; triaxial seeds additionally couple delta-m=2.
    """
    if shape == "sphere" or amplitude == 0:
        return
    if shape not in {"prolate", "oblate", "triaxial"}:
        raise ValueError(f"unknown candidate shape {shape!r}")
    channels = tuple(basis.channels)
    channel_by_quantum_numbers = {(ch.kappa, ch.m2): ch for ch in channels}
    perturbation = torch.zeros_like(network.raw_coordinates)
    sign = -1.0 if shape == "oblate" else 1.0

    def upper_l(kappa: int) -> int:
        return -kappa - 1 if kappa < 0 else kappa

    for column, label in enumerate(network.initial_q_labels):
        match = re.search(r"kappa=(-?\d+):m2=(-?\d+)", label)
        if not match:
            continue
        source_kappa, source_m2 = int(match.group(1)), int(match.group(2))
        source = channel_by_quantum_numbers.get((source_kappa, source_m2))
        if source is None:
            continue
        source_start = source.offset
        source_stop = source_start + 2 * basis.n_radial
        source_values = network.initial_q[source_start:source_stop, column]
        for target in channels:
            delta_m2 = target.m2 - source_m2
            delta_l = abs(upper_l(target.kappa) - upper_l(source_kappa))
            if target.kappa == source_kappa or delta_l not in {0, 2}:
                continue
            if shape != "triaxial" and delta_m2 != 0:
                continue
            if shape == "triaxial" and delta_m2 not in {-4, 0, 4}:
                continue
            if delta_m2 == 0 and delta_l != 2:
                continue
            if delta_m2 != 0 and (delta_l == 0 or delta_l == 2):
                # A real tesseral quadrupole component couples m to m +/- 2.
                # Alternating signs distinguish the two principal directions.
                coefficient = sign * (1.0 if delta_m2 > 0 else -1.0)
            else:
                coefficient = sign
            target_start = target.offset
            target_stop = target_start + 2 * basis.n_radial
            perturbation[target_start:target_stop, column] += (
                float(amplitude * coefficient) * source_values
            )
    with torch.no_grad():
        network.raw_coordinates.copy_(perturbation)
    network.initial_candidate_shape = shape


def train_rhfb(config: BlockedRHFBConfig, basis: AngularRadialBasis,
               networks: dict[str, dict[str, Any]], blocks: dict[str, BlockingSpec | None],
               evaluate: Callable[..., Any], *, output_dir: str | Path,
               learning_rate: float = 2.0e-3, number_penalty: float = 50.0,
               checkpoint_every: int = 25, chunk_size: int | None = None,
               resume_from: str | Path | None = None,
               export_observables: bool = True) -> dict[str, Any]:
    """Minimize a certified complete functional on the complex blocked manifold."""
    from .execution_limits import require_bounded_cgroup
    require_bounded_cgroup()
    if config.refresh_interval != 0 or config.dtype != "complex128":
        raise ValueError("physical RHFB training requires complex128 with no refreshes")
    if set(networks) != {"n", "p"}:
        raise ValueError("both neutron and proton networks are required")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history_path = out / "history.jsonl"
    if resume_from is None:
        history_path.unlink(missing_ok=True)
    modules = {key: value["network"] for key, value in networks.items()}
    basis_spec = _basis_spec(basis)
    blocked_reference = _blocking_spec(blocks)
    physics_prescription = current_neural_prescription(
        basis_spec=json.dumps(basis_spec, sort_keys=True, separators=(",", ":")),
        blocking=json.dumps(blocked_reference, sort_keys=True, separators=(",", ":")),
    )
    params = [p for module in modules.values() for p in module.parameters()]
    eigen_counter = _TrainingEigenCounter(
        basis.dimension,
        {species: int(getattr(module, "active_dimension", 0)) for species, module in modules.items()},
    )
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    targets = {"n": config.n, "p": config.z}
    chunk_size = config.chunk_size if chunk_size is None else int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    evaluate_parameters = inspect.signature(evaluate).parameters
    evaluation_options = {"chunk_size": chunk_size,
                          "pair_chunk": config.pair_chunk,
                          "lmax": config.lmax}
    evaluation_options = {key: value for key, value in evaluation_options.items()
                          if key in evaluate_parameters or any(
                              parameter.kind == inspect.Parameter.VAR_KEYWORD
                              for parameter in evaluate_parameters.values())}

    def evaluate_state(states):
        return evaluate(basis, states, **evaluation_options)
    multipliers = {"n": 0.0, "p": 0.0}
    penalty = float(number_penalty)
    initial_penalty = float(number_penalty)
    start_epoch = 1

    previous_blocked: dict[str, torch.Tensor | None] = {"n": None, "p": None}

    if resume_from is not None:
        device = params[0].device
        payload = load_checkpoint(
            resume_from, config, basis_spec=basis_spec,
            blocked_reference=blocked_reference,
            physics_prescription=physics_prescription, map_location=device,
        )
        if set(payload["network_state"]) != {"n", "p"}:
            raise ValueError("checkpoint does not contain both species networks")
        for species in ("n", "p"):
            modules[species].load_state_dict(payload["network_state"][species], strict=True)
        if payload.get("optimizer_state") is None:
            raise ValueError("checkpoint lacks optimizer state and cannot resume training")
        optimizer.load_state_dict(payload["optimizer_state"])
        trainer_state = payload.get("trainer_state")
        if not isinstance(trainer_state, Mapping) or trainer_state.get("format_version") != 2:
            raise ValueError("checkpoint lacks compatible RHFB trainer state")
        if not math.isclose(float(trainer_state.get("learning_rate", math.nan)),
                            float(learning_rate), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("checkpoint learning rate differs from requested resume settings")
        if not math.isclose(float(trainer_state.get("initial_number_penalty", math.nan)),
                            initial_penalty, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("checkpoint initial number penalty differs from requested resume settings")
        completed = int(trainer_state.get("epochs_completed", 0))
        if completed < 1 or completed >= config.epochs:
            raise ValueError("checkpoint epoch must be positive and below requested total epochs")
        start_epoch = completed + 1
        multipliers = {key: float(value) for key, value in trainer_state["number_multipliers"].items()}
        penalty = float(trainer_state["number_penalty"])
        previous_blocked = _unpack_mode_vectors(
            trainer_state["previous_blocked_mode"], device=device
        )
        if not history_path.exists():
            source_history = Path(resume_from).parent / "history.jsonl"
            if source_history.is_file():
                history_path.write_bytes(source_history.read_bytes())

    def current_state():
        states = {}
        block_vectors = {}
        for species, module in modules.items():
            if not hasattr(module, "forward_with_blocked_mode"):
                raise RuntimeError("network must expose full Nambu blocked-mode tracking")
            q, gd, blocked_mode = module.forward_with_blocked_mode()
            states[species] = DensityFactors(q, gd.rho, gd.chi)
            block = blocks[species]
            if (block is None) != (blocked_mode is None):
                raise RuntimeError(f"{species} network returned an inconsistent blocked mode")
            block_vectors[species] = blocked_mode
        overlaps = update_blocked_mode_tracker(previous_blocked, block_vectors)
        return states, overlaps

    # Fail before the first update. An incomplete functional never contributes
    # an optimizer step, even if it happens to return a scalar component sum.
    with eigen_counter:
        states, _ = current_state()
        evaluation = evaluate_state(states)
    _require_no_training_eigensolves(eigen_counter)
    failures = validate_functional_evaluation(evaluation)
    if failures:
        raise PhysicalGateError(failures, getattr(evaluation, "audit", None))
    audit = evaluation.audit if hasattr(evaluation, "audit") else evaluation["audit"]
    for epoch in range(start_epoch, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        with eigen_counter:
            states, blocking_overlaps = current_state()
            evaluation = evaluate_state(states)
        _require_no_training_eigensolves(eigen_counter)
        failures = validate_functional_evaluation(evaluation)
        if failures:
            raise PhysicalGateError(failures, getattr(evaluation, "audit", None))
        energy = evaluation.total_energy if hasattr(evaluation, "total_energy") else evaluation["total_energy"]
        energy_terms = (evaluation.energy_terms if hasattr(evaluation, "energy_terms")
                        else evaluation["energy_terms"])
        numbers = {key: states[key].particle_number() for key in ("n", "p")}
        residuals = {key: numbers[key] - targets[key] for key in ("n", "p")}
        loss = energy
        for species in ("n", "p"):
            delta = residuals[species]
            loss = loss + multipliers[species] * delta + 0.5 * penalty * delta.square()
        with eigen_counter:
            loss.backward()
        grad_norm_sq = sum(float(p.grad.detach().abs().square().sum())
                           for p in params if p.grad is not None)
        grad_norm = math.sqrt(grad_norm_sq)
        if not math.isfinite(grad_norm):
            raise FloatingPointError("network gradient became non-finite")
        with eigen_counter:
            optimizer.step()
        multipliers_used = multipliers.copy()
        record = {
            "epoch": epoch,
            "energy": energy.detach(),
            "loss": loss.detach(),
            "energy_terms": energy_terms,
            "particle_numbers": numbers,
            "number_residuals": residuals,
            "number_multipliers": multipliers_used,
            "number_penalty": penalty,
            "network_gradient_norm": grad_norm,
            "blocking_overlaps": blocking_overlaps,
            "blocking": {key: (None if block is None else block.label)
                         for key, block in blocks.items()},
        }
        append_jsonl(history_path, record)
        if epoch % 25 == 0:
            for species in ("n", "p"):
                multipliers[species] += penalty * float(residuals[species].detach())
            if max(abs(float(residuals[s].detach())) for s in ("n", "p")) > 1.0e-3:
                penalty = min(penalty * 2.0, 1.0e8)
        if epoch % checkpoint_every == 0 or epoch == config.epochs:
            save_checkpoint(
                out / "checkpoint.pt", config, basis_spec=basis_spec,
                blocked_reference=blocked_reference,
                physics_prescription=physics_prescription,
                network_state={key: module.state_dict() for key, module in modules.items()},
                optimizer_state=optimizer.state_dict(),
                trainer_state={
                    "format_version": 2,
                    "epochs_completed": epoch,
                    "learning_rate": float(learning_rate),
                    "initial_number_penalty": initial_penalty,
                    "number_multipliers": multipliers.copy(),
                    "number_penalty": penalty,
                    "previous_blocked_mode": _pack_mode_vectors(previous_blocked),
                },
            )

    with eigen_counter:
        final_states, final_blocking_overlaps = current_state()
        final_eval = evaluate_state(final_states)
    _require_no_training_eigensolves(eigen_counter)
    final_numbers = {key: float(final_states[key].particle_number().detach()) for key in ("n", "p")}
    final_energy = float((final_eval.total_energy if hasattr(final_eval, "total_energy")
                          else final_eval["total_energy"]).detach())
    stationarity_fn = (final_eval.stationarity if hasattr(final_eval, "stationarity")
                       else final_eval.get("stationarity"))
    stationarity_parameters = inspect.signature(stationarity_fn).parameters
    stationarity_kwargs = {"chunk_size": chunk_size} if (
        "chunk_size" in stationarity_parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in stationarity_parameters.values())
    ) else {}
    stationarity = stationarity_fn(final_states, **stationarity_kwargs)
    final_audit = final_eval.audit if hasattr(final_eval, "audit") else final_eval["audit"]
    observables = None
    if export_observables:
        with torch.no_grad():
            observables = save_observables(
                basis, final_states, out,
                targets={"n": config.n, "p": config.z},
            )
    final = {
        "format_version": 2, "solver": "blocked-rhfb",
        "status": "trained_unverified",
        "nucleus": {"z": config.z, "n": config.n, "a": config.a},
        "config": config.__dict__, "last_energy_estimate_mev": final_energy,
        "initial_candidate_shape": config.candidate_shape,
        "candidate_seed_amplitude": config.candidate_seed_amplitude,
        "device": str(params[0].device) if params else "cpu",
        "functional_chunks": {"spatial": chunk_size,
                              "active_pair": config.pair_chunk,
                              "multipole_lmax": config.lmax},
        "physical_result": None,
        "validated": False,
        "validation_status": "pre-validation training; independent odd-nucleus reference and space convergence remain open",
        "final_acceptance": {
            "independent_physical_reference": final_audit.get(
                "independent_physical_reference_status", "unverified"
            ),
            "space_convergence": final_audit.get("space_convergence_status", "unverified"),
            "moving_projector_equivalence": final_audit.get(
                "moving_projector_equivalence_status", "unverified"
            ),
        },
        "particle_numbers": final_numbers,
        "number_residuals": {"n": final_numbers["n"] - config.n,
                             "p": final_numbers["p"] - config.z},
        "blocking": {key: (None if block is None else {
            "label": block.label, "index": block.index, "kappa": block.kappa,
            "m2": block.m2, "operation": block.operation,
        }) for key, block in blocks.items()},
        "blocking_overlaps": final_blocking_overlaps,
        "stationarity_diagnostics": stationarity,
        "observables": observables,
        "functional_audit": final_audit,
        "epochs_completed": config.epochs, "refresh_count": 0,
        "refresh_tracking": {
            "count": 0,
            "basis_refresh_operation": "none in the no-refresh trainer",
            "configuration_gate": "refresh_interval=0 enforced by config and train_rhfb",
            "scope": "active-subspace refreshes; eigensolver observations are separate",
        },
        "solver_counts": {
            **eigen_counter.counts,
            "instrumentation": "torch_dispatch_eigensolver_ops",
            "eigensolver_operations": eigen_counter.operations,
        },
        "checkpoint": str(out / "checkpoint.pt"), "history": str(history_path),
    }
    atomic_json(out / "summary.json", final)
    return json_value(final)
