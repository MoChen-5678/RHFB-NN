"""Bounded frozen-density O17 diagnostic for CoordinateNormalFunctional.

Run from the blocked-rhfb checkout. This compares one coordinate-kernel
normal-field evaluation against saved factorized PKA1 columns and an
independent scalar evaluation. It does not train, diagonalize, or update the
state.
"""
from __future__ import annotations

import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .basis import AngularRadialBasis
from .coordinate_normal_functional import (CoordinateNormalFunctional,
                                           estimate_coordinate_normal_memory)
from .functional import evaluate_particle_hole_energy
from .operators import DensityFactors
from .execution_limits import require_bounded_cgroup, require_small_coordinate_space


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run(source: Path, output: Path, *, wall_limit_s: float = 1800.0):
    # Fail before loading density arrays or building the unbounded AD graph.
    require_small_coordinate_space(type('RequestedSpace', (), {
        'point_count': 882, 'dimension': 432})())
    require_bounded_cgroup()
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snap = output / "source_begin"
    snap.mkdir()
    module_dir = Path(__file__).resolve().parent
    source_files = [module_dir / n for n in (
        "o17_coordinate_normal_diagnostic.py", "coordinate_normal_functional.py",
        "basis.py", "functional.py", "currents.py", "spatial.py", "operators.py",
    )] + [module_dir.parent / "rhf" / "parameters.py"]
    for path in source_files:
        shutil.copy2(path, snap / path.name)

    density = source / "density_seed.npz"
    hn_path, hp_path = source / "h_n_columns.npy", source / "h_p_columns.npy"
    report = {
        "status": "running", "scope": "frozen saved O17 density; full normal PH field/action diagnostic",
        "physical_result_accepted": False, "fortran_used": False, "neural_updates": 0,
        "source_density": str(density), "density_sha256": _sha(density),
        "reference_h": {"n": str(hn_path), "p": str(hp_path),
                        "n_sha256": _sha(hn_path), "p_sha256": _sha(hp_path)},
        "source_hashes": {p.name: _sha(p) for p in source_files},
        "source_snapshot": str(snap),
        "basis": {"radius_fm": 8.0, "spacing_fm": 0.8, "kappa_max": 3,
                  "n_theta": 7, "n_phi": 14, "lmax": 6,
                  "derivative_order": 5, "adf_points": 5},
        "algorithm": {"coordinate_R_materialized": True,
                      "interaction_four_tensor_materialized": False,
                      "pair_tile": 32, "checkpoint_fock_blocks": True,
                      "joint_species_gradient": True, "density_compression": False,
                      "spectrum_or_training": False},
        "limits": {"wall_seconds": wall_limit_s, "peak_rss_gib": 24.0},
        "started_unix": time.time(),
    }
    report_path = output / "diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    basis = AngularRadialBasis(radius_fm=8.0, spacing_fm=0.8, kappa_max=3,
                               n_theta=7, n_phi=14, derivative_order=5)
    with np.load(density) as z:
        states = {s: DensityFactors(*[
            torch.from_numpy(z[f"{s}_{k}"].copy())
            for k in ("Q", "rho_active", "chi_active")
        ]) for s in ("n", "p")}
    report["state_dimensions"] = {
        s: {"Q": list(st.q.shape), "rho_active": list(st.rho_active.shape),
            "chi_active": list(st.chi_active.shape),
            "particle_number": float(st.particle_number())}
        for s, st in states.items()
    }
    report["point_count"] = basis.point_count
    report["coordinate_memory_estimate_bytes"] = estimate_coordinate_normal_memory(
        basis, active_rank=14, pair_chunk=32
    )
    started = time.monotonic()
    torch.manual_seed(17062026)
    probes = torch.randn((basis.dimension, 2), dtype=torch.float64).to(torch.complex128)
    probes += 1j * torch.randn((basis.dimension, 2), dtype=torch.float64)
    try:
        coordinate = CoordinateNormalFunctional(
            basis, pair_chunk=32, chunk_size=32, lmax=6,
            checkpoint_fock_blocks=True,
        ).evaluate(states)
        report["coordinate_elapsed_seconds"] = time.monotonic() - started
        report["coordinate_peak_rss_gib"] = _rss_gib()
        report["coordinate_diagnostics"] = coordinate.diagnostics
        report["energy_terms_coordinate_mev"] = {
            k: float(v.detach()) for k, v in coordinate.energy.energy_terms.items()
        }
        refs = {"n": np.load(hn_path, mmap_mode="r"),
                "p": np.load(hp_path, mmap_mode="r")}
        closure = {}
        for s in ("n", "p"):
            ref_action = torch.from_numpy(np.asarray(refs[s])).to(torch.complex128) @ probes
            got_action = coordinate.h[s] @ probes
            diff = got_action - ref_action
            closure[s] = {
                "reference_random_action_max_abs_mev": float(ref_action.abs().max()),
                "coordinate_random_action_max_abs_mev": float(got_action.abs().max()),
                "max_abs_difference_mev": float(diff.abs().max()),
                "relative_frobenius": float(torch.linalg.matrix_norm(diff)
                                              / torch.linalg.matrix_norm(ref_action)),
                "hermiticity_max_abs_mev": float((coordinate.h[s]-coordinate.h[s].mH).abs().max()),
            }
        report["random_action_closure"] = closure
        # Evaluate factorized production scalar once at the exact same state,
        # preserving all channel energy terms as a second oracle.
        reference = evaluate_particle_hole_energy(
            basis, states, pair_chunk=32, chunk_size=128, lmax=6,
            _checkpoint_fock_blocks=False,
        )
        report["reference_energy_terms_mev"] = {
            k: float(v.detach()) for k, v in reference.energy_terms.items()
        }
        report["energy_term_differences_mev"] = {
            k: float(coordinate.energy.energy_terms[k].detach()
                     - reference.energy_terms[k].detach())
            for k in reference.energy_terms
        }
        report["energy_max_abs_term_difference_mev"] = max(
            abs(v) for v in report["energy_term_differences_mev"].values()
        )
        report["status"] = "completed_coordinate_O17_frozen_field_comparison"
    except BaseException as exc:
        report["status"] = "failed_coordinate_O17_frozen_field_comparison"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_gib"] = _rss_gib()
        report["within_wall_limit"] = report["elapsed_seconds"] <= wall_limit_s
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    source_default = here / "outputs/validation/O17_full_fields_physics_20261006"
    output_default = here / "outputs/validation/O17_coordinate_normal_kernel_20261006"
    run(source_default, output_default)
