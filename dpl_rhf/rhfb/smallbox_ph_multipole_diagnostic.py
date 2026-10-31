"""Single-case angular-cutoff diagnostic on the trusted M40/D48 toy fixture.

This is a numerical diagnostic helper, not a production solver or an O17
calculation.  It restores a saved joint neural state and evaluates its complete
streamed PKA1 particle-hole fields once at one requested (L, angular grid).
Run repeated cases serially under the main agent's resource guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from .execution_limits import require_bounded_cgroup
from .runtime_resources import resource_snapshot


FIXTURE_DIR = Path("outputs/validation/joint_real_nn_m40_channels_20261007")
EXPECTED_CHECKPOINT = FIXTURE_DIR / "training" / "joint_step_0002.pt"
EXPECTED_BASIS = {
    "radius_fm": 1.4,
    "spacing_fm": 0.2,
    "kappa_max": 1,
    "n_theta": 3,
    "n_phi": 6,
    "derivative_order": 5,
    "dimension": 48,
    "point_count": 108,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str((tuple(array.shape), str(array.dtype))).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def _same_number(actual, expected) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12
    )


def _validate_checkpoint_metadata(metadata: dict[str, Any]) -> None:
    declared_fingerprint = metadata.get("compatibility_fingerprint")
    fingerprint_payload = dict(metadata)
    fingerprint_payload.pop("compatibility_fingerprint", None)
    encoded = json.dumps(fingerprint_payload, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    if not isinstance(declared_fingerprint, str) or hashlib.sha256(encoded).hexdigest() != declared_fingerprint:
        raise ValueError("checkpoint metadata compatibility fingerprint mismatch")
    if metadata.get("format") != "joint-selfconsistent-neural-training-v1":
        raise ValueError("unsupported checkpoint format")
    if metadata.get("production_physics_asserted") is not True:
        raise ValueError("checkpoint does not assert the production joint physics")
    physics = str(metadata.get("physics", ""))
    if "complete PKA1 particle-hole" not in physics or "full-space D1S pairing" not in physics:
        raise ValueError("checkpoint is not declared complete PKA1 plus D1S")
    if metadata.get("fortran_used") is not False or metadata.get("eigensolver_used") is not False:
        raise ValueError("checkpoint violates the no-Fortran/no-eigensolver fixture contract")
    if int(metadata.get("basis", {}).get("dimension", -1)) != 48:
        raise ValueError("checkpoint primitive dimension is not D=48")
    for key, expected in EXPECTED_BASIS.items():
        if not _same_number(metadata.get("basis", {}).get(key), expected):
            raise ValueError(f"checkpoint basis mismatch for {key}")
    if metadata.get("number_targets") != {"n": 3, "p": 2}:
        raise ValueError("checkpoint targets do not match odd-N=3, even-Z=2 fixture")
    models = metadata.get("models", {})
    for species in ("n", "p"):
        model = models.get(species, {})
        if (model.get("format") != "dpl-rhfb-generalized-density-net-v1"
                or int(model.get("active_dimension", -1)) != 40
                or int(model.get("primitive_dimension", -1)) != 48):
            raise ValueError(f"checkpoint {species} model metadata mismatch")
    if int(metadata.get("options", {}).get("updates_requested", -1)) < 2:
        raise ValueError("checkpoint was not produced by a >=2-update joint fixture")
    if metadata.get("basis_refreshes") != 0 or metadata.get("no_sea_branch_certified") is not False:
        raise ValueError("checkpoint metadata contains an invalid fixture/acceptance flag")
    ph_parameters = metadata.get("particle_hole_parameters", {})
    from dataclasses import asdict
    from dpl_rhf.rhf.parameters import pka1_parameters
    if ph_parameters != asdict(pka1_parameters()):
        raise ValueError("checkpoint particle-hole parameters differ from PKA1")
    from .pairing import D1S
    if metadata.get("pairing_parameters") != asdict(D1S):
        raise ValueError("checkpoint D1S pairing metadata differs from this implementation")
    if models["n"].get("blocked_index") != 2 or models["n"].get("fermion_parity") != -1:
        raise ValueError("neutron state is not the fixture's odd particle-addition block")
    if models["p"].get("blocked_index") is not None or models["p"].get("fermion_parity") != 1:
        raise ValueError("proton state is not the fixture's even reference")


def run_case(*, checkpoint: Path, output: Path, lmax: int, n_theta: int,
             n_phi: int, tile_size: int = 4) -> dict[str, Any]:
    """Restore one trusted fixture and run exactly one full-PH cutoff case."""
    # Enforce memory before resolving/loading the checkpoint or allocating model/grid state.
    guard = require_bounded_cgroup()
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    expected = (Path(__file__).resolve().parents[2] / EXPECTED_CHECKPOINT).resolve()
    if checkpoint != expected or checkpoint.name != "joint_step_0002.pt":
        raise ValueError(f"checkpoint must be the trusted fixture artifact at {expected}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(output)
    if lmax < 0 or n_theta < lmax + 1 or n_phi < 2 * lmax + 1:
        raise ValueError("require L>=0, n_theta>=L+1 and n_phi>=2L+1")
    if not 1 <= tile_size <= 8:
        raise ValueError("tile_size must be between 1 and 8 in this guarded diagnostic")

    # The checkpoint is read only after guard and CLI/path admission checks.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("checkpoint payload is malformed")
    metadata = payload["metadata"]
    _validate_checkpoint_metadata(metadata)
    if int(payload.get("accepted_updates", -1)) != 2:
        raise ValueError("checkpoint is not joint_step_0002")
    if payload.get("physical_result_accepted") is not False or payload.get("no_sea_branch_certified") is not False:
        raise ValueError("checkpoint acceptance flags are not fail-closed")
    model_states = payload.get("model_states")
    if not isinstance(model_states, dict) or set(model_states) != {"n", "p"}:
        raise ValueError("checkpoint must contain both neutron and proton model states")

    from .basis import AngularRadialBasis
    from .operators import DensityFactors
    from .selfconsistent_neural_training import _forbid_spectral_calls
    from .selfconsistent_neural_smoke import FIXTURE, _make_models
    from .streamed_normal_functional import StreamedPrimitiveNormalFunctional

    torch.set_num_threads(1)
    torch.manual_seed(int(FIXTURE["network"]["torch_seed"]))
    device = torch.device("cpu")
    # Reconstruct the models on the original training angular grid.  The
    # evaluation basis changes only angular quadrature; primitive Q is kept.
    training_basis = AngularRadialBasis(
        radius_fm=1.4, spacing_fm=0.2, kappa_max=1,
        n_theta=3, n_phi=6, derivative_order=5,
    )
    if training_basis.dimension != 48 or training_basis.point_count != 108:
        raise RuntimeError("original M40/D48 training basis contract changed")
    model_cfg = FIXTURE["network"]
    models = _make_models(
        training_basis,
        hidden=int(model_cfg["hidden"]), depth=int(model_cfg["depth"]),
        radial_basis_size=int(model_cfg["radial_basis_size"]),
        channel_chunk_size=int(model_cfg["channel_chunk_size"]), device=device,
    )
    for species in ("n", "p"):
        models[species].load_state_dict(model_states[species], strict=True)
        if models[species].active_dimension != 40:
            raise RuntimeError("restored model is not M=40")
    del model_states, payload

    with torch.no_grad(), _forbid_spectral_calls():
        states = {}
        q_provenance = {}
        for species in ("n", "p"):
            q, gd = models[species]()
            states[species] = DensityFactors(q.detach(), gd.rho.detach(), gd.chi.detach())
            q_provenance[species] = {
                "q_sha256": _tensor_hash(q),
                "rho_active_sha256": _tensor_hash(gd.rho),
                "chi_active_sha256": _tensor_hash(gd.chi),
                "active_dimension": int(q.shape[1]),
                "primitive_dimension": int(q.shape[0]),
                "fermion_parity": int(gd.fermion_parity),
                "blocked_index": models[species].blocked_index,
                "q_origin": "restored GeneralizedDensityNet forward on original theta=3,phi=6 fixture basis",
            }
    del models, training_basis

    from .basis import AngularRadialBasis
    evaluation_basis = AngularRadialBasis(
        radius_fm=1.4, spacing_fm=0.2, kappa_max=1,
        n_theta=int(n_theta), n_phi=int(n_phi), derivative_order=5,
    )
    if evaluation_basis.dimension != 48 or evaluation_basis.n_radial != 6:
        raise RuntimeError("angular evaluation basis changed primitive channels or radial mesh")
    for state in states.values():
        if state.dimension != evaluation_basis.dimension or state.q.shape[1] != 40:
            raise RuntimeError("restored primitive Q does not match evaluation basis")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [
        Path(__file__).resolve(), Path(__file__).with_name("basis.py"),
        Path(__file__).with_name("contact.py"),
        Path(__file__).with_name("coordinate_normal_functional.py"),
        Path(__file__).with_name("currents.py"),
        Path(__file__).with_name("fock_checkpoint.py"),
        Path(__file__).with_name("functional.py"),
        Path(__file__).with_name("network.py"),
        Path(__file__).with_name("operators.py"),
        Path(__file__).with_name("pairing.py"),
        Path(__file__).with_name("selfconsistent_neural_smoke.py"),
        Path(__file__).with_name("selfconsistent_neural_training.py"),
        Path(__file__).with_name("streamed_normal_functional.py"),
        Path(__file__).with_name("streamed_field_autograd.py"),
        Path(__file__).with_name("spatial.py"),
        Path(__file__).with_name("execution_limits.py"),
        Path(__file__).with_name("runtime_resources.py"),
        Path(__file__).parent.parent / "rhf" / "parameters.py",
        Path(__file__).parent.parent / "rhf" / "dirac.py",
    ]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for index, raw in enumerate(source_paths):
        path = raw.resolve()
        target = snapshot / f"{index:02d}_{path.name}"
        shutil.copy2(path, target)
        source_hashes[str(path)] = _sha256(path)

    report: dict[str, Any] = {
        "status": "running",
        "scope": "M40,D48,R=1.4 fm,h=0.2 fm,K=1,N=3,Z=2 algebraic small-box joint-NN fixture; not O17",
        "basis": {
            "radius_fm": 1.4, "spacing_fm": 0.2, "kappa_max": 1,
            "dimension": 48, "n_radial": evaluation_basis.n_radial,
            "training_n_theta": 3, "training_n_phi": 6,
            "evaluation_n_theta": n_theta, "evaluation_n_phi": n_phi,
            "derivative_order": 5,
        },
        "active_dimension": 40,
        "targets": {"n": 3, "p": 2},
        "functional": "complete streamed PKA1 particle-hole fields",
        "physics_source_checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata_fingerprint": metadata.get("compatibility_fingerprint"),
        "one_case": {"lmax": int(lmax), "n_theta": int(n_theta),
                     "n_phi": int(n_phi), "tile_size": int(tile_size)},
        "density_provenance": q_provenance,
        "source_snapshot": str(snapshot),
        "source_sha256": source_hashes,
        "resource_guard": guard,
        "fortran_used": False,
        "eigensolver_used": False,
        "basis_refreshes": 0,
        "physical_result_accepted": False,
        "no_sea_branch_certified": False,
        "angular_converged": False,
        "direct_kernel_independently_validated": False,
        "limitations": [
            "one requested L/quadrature point only; compare separate serial launches",
            "no independent direct-kernel oracle and no singular-point guesses",
            "small-box fixture only; not an O17 or continuum result",
        ],
        "started_unix": time.time(),
    }
    _write_json(output / "diagnostic.json", report)

    progress_path = output / "progress.jsonl"

    def on_progress(row: dict[str, Any]) -> None:
        _append_jsonl(progress_path, {
            "case": report["one_case"],
            **row,
            "runtime_resource_snapshot": resource_snapshot(),
        })

    try:
        evaluator = StreamedPrimitiveNormalFunctional(
            evaluation_basis, tile_size=tile_size, chunk_size=8,
            lmax=int(lmax),
        )
        with _forbid_spectral_calls():
            result = evaluator.evaluate(states, progress=on_progress)
        channels = {
            key: float(value.detach().cpu())
            for key, value in result.energy.energy_terms.items()
        }
        report.update({
            "status": "complete",
            "energy_channels_mev": channels,
            "total_particle_hole_energy_mev": float(result.energy.total_energy.detach().cpu()),
            "normal_density_sha256": {s: _tensor_hash(result.rho_primitive[s]) for s in ("n", "p")},
            "normal_field_sha256": {s: _tensor_hash(result.h[s]) for s in ("n", "p")},
            "evaluation_audit": result.diagnostics,
            "runtime_resource_snapshot": resource_snapshot(),
            "elapsed_seconds": time.time() - report["started_unix"],
        })
        import numpy as np
        artifact_hashes = {}
        for species in ("n", "p"):
            field_path = output / f"h_{species}.npy"
            np.save(field_path, result.h[species].detach().cpu().numpy(), allow_pickle=False)
            artifact_hashes[field_path.name] = _sha256(field_path)
        density_path = output / "density_factors.npz"
        density_arrays = {}
        for species, state in states.items():
            density_arrays[f"{species}_q"] = state.q.detach().cpu().numpy()
            density_arrays[f"{species}_rho_active"] = state.rho_active.detach().cpu().numpy()
            density_arrays[f"{species}_chi_active"] = state.chi_active.detach().cpu().numpy()
        np.savez_compressed(density_path, **density_arrays)
        artifact_hashes[density_path.name] = _sha256(density_path)
        with np.load(density_path, allow_pickle=False) as saved_factors:
            for species in ("n", "p"):
                for suffix, hash_key in (("q", "q_sha256"),
                                         ("rho_active", "rho_active_sha256"),
                                         ("chi_active", "chi_active_sha256")):
                    loaded = torch.from_numpy(saved_factors[f"{species}_{suffix}"].copy())
                    if _tensor_hash(loaded) != q_provenance[species][hash_key]:
                        raise RuntimeError(f"saved {species}_{suffix} does not match forward output")
        report["saved_artifact_sha256"] = artifact_hashes
        report["saved_artifact_hashes_verified"] = all(
            _sha256(output / name) == digest for name, digest in artifact_hashes.items()
        )
        if not report["saved_artifact_hashes_verified"]:
            raise RuntimeError("saved diagnostic artifact hash verification failed")
    except Exception as exc:
        report.update({
            "status": "incomplete",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_resource_snapshot": resource_snapshot(),
            "elapsed_seconds": time.time() - report["started_unix"],
        })
        _write_json(output / "diagnostic.json", report)
        raise
    _write_json(output / "diagnostic.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lmax", type=int, required=True)
    parser.add_argument("--n-theta", type=int, default=7)
    parser.add_argument("--n-phi", type=int, default=14)
    parser.add_argument("--tile-size", type=int, default=4)
    args = parser.parse_args(argv)
    run_case(
        checkpoint=args.checkpoint, output=args.output, lmax=args.lmax,
        n_theta=args.n_theta, n_phi=args.n_phi, tile_size=args.tile_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
