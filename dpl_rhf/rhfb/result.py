"""Stable JSON diagnostics for blocked-RHFB optimization runs.

This module deliberately treats a gated run as a useful result: it records
what was checked and why training did not start, without inventing a nuclear
energy or a convergence claim.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from .currents import local_bilinears
from .operators import DensityFactors


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        return float(value.detach().cpu().item())
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TypeError(f"cannot encode {type(value).__name__} in RHFB result")


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_value(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_blocked_result(path: str | Path, *, config: Any, reason: str,
                         gates: dict[str, Any], initialization: dict[str, Any] | None = None,
                         run_id: str | None = None) -> dict[str, Any]:
    payload = {
        "format_version": 2,
        "solver": "blocked-rhfb",
        "status": "blocked",
        "run_id": run_id,
        "config": asdict(config) if is_dataclass(config) else config,
        "nucleus": {"z": config.z, "n": config.n, "a": config.z + config.n},
        "reason": reason,
        "gates": gates,
        "initialization": initialization or {},
        "training": {"started": False, "epochs_completed": 0},
        "physical_result": None,
        "solver_counts": {"training_dirac_eigensolves": 0,
                          "training_bogoliubov_eigensolves": 0,
                          "refresh_count": 0},
    }
    atomic_json(path, payload)
    return json_value(payload)


def save_observables(basis, states: dict[str, DensityFactors], output_dir: str | Path,
                     *, targets: dict[str, int] | None = None,
                     chunk_size: int = 1024) -> dict[str, Any]:
    """Stream pointwise neutron/proton observables to CSV and compressed NPZ.

    Densities are Dirac baryon densities in fm^-3. `Q20` uses
    sqrt(5/(16 pi))*integral[rho(2z^2-x^2-y^2)] and `current_l1` is
    integral[|j|] over volume. The full arrays remain available in the NPZ.
    """
    if set(states) != {"n", "p"}:
        raise ValueError("observable export requires neutron and proton states")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "positions_fm", "weights_fm3", "rho_n_fm3", "rho_p_fm3",
            "rho_scalar_n_fm3", "rho_scalar_p_fm3", "current_n_fm3",
            "current_p_fm3",
        )
    }
    csv_path = out / "observables.csv"
    descriptor, temporary = tempfile.mkstemp(prefix=csv_path.name + ".", dir=out)
    scalar_header = (
        "x_fm,y_fm,z_fm,weight_fm3,rho_n_fm3,rho_p_fm3,rho_matter_fm3,"
        "rho_scalar_n_fm3,rho_scalar_p_fm3,j_n_x_fm3,j_n_y_fm3,j_n_z_fm3,"
        "j_p_x_fm3,j_p_y_fm3,j_p_z_fm3,j_total_x_fm3,j_total_y_fm3,j_total_z_fm3"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(scalar_header + "\n")
            for start in range(0, basis.point_count, chunk_size):
                stop = min(start + chunk_size, basis.point_count)
                per_species = {}
                for species in ("n", "p"):
                    state = states[species]
                    spinors = basis._sample_range(state.q, start, stop)
                    bilinear = local_bilinears(spinors, state.rho_active)
                    per_species[species] = {
                        "rho": bilinear["baryon_current"][:, 0],
                        "scalar": bilinear["scalar"],
                        "current": bilinear["baryon_current"][:, 1:4],
                    }
                positions = np.asarray(basis.positions[start:stop], dtype=np.float64)
                weights = np.asarray(basis.weights[start:stop], dtype=np.float64)
                rho_n = per_species["n"]["rho"].detach().cpu().numpy()
                rho_p = per_species["p"]["rho"].detach().cpu().numpy()
                scalar_n = per_species["n"]["scalar"].detach().cpu().numpy()
                scalar_p = per_species["p"]["scalar"].detach().cpu().numpy()
                current_n = per_species["n"]["current"].detach().cpu().numpy()
                current_p = per_species["p"]["current"].detach().cpu().numpy()
                if not all(np.isfinite(array).all() for array in
                           (positions, weights, rho_n, rho_p, scalar_n, scalar_p,
                            current_n, current_p)):
                    raise FloatingPointError("non-finite pointwise RHFB observable")
                chunks["positions_fm"].append(positions)
                chunks["weights_fm3"].append(weights)
                chunks["rho_n_fm3"].append(rho_n)
                chunks["rho_p_fm3"].append(rho_p)
                chunks["rho_scalar_n_fm3"].append(scalar_n)
                chunks["rho_scalar_p_fm3"].append(scalar_p)
                chunks["current_n_fm3"].append(current_n)
                chunks["current_p_fm3"].append(current_p)
                current_total = current_n + current_p
                columns = np.column_stack((
                    positions, weights, rho_n, rho_p, rho_n + rho_p,
                    scalar_n, scalar_p, current_n, current_p, current_total,
                ))
                np.savetxt(stream, columns, delimiter=",", fmt="%.17e")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    rho_n, rho_p = arrays["rho_n_fm3"], arrays["rho_p_fm3"]
    rho_matter = rho_n + rho_p
    positions, weights = arrays["positions_fm"], arrays["weights_fm3"]
    radius2 = np.einsum("pi,pi->p", positions, positions)
    quadrupole_kernel = (2.0 * positions[:, 2] ** 2
                         - positions[:, 0] ** 2 - positions[:, 1] ** 2)
    current_n, current_p = arrays["current_n_fm3"], arrays["current_p_fm3"]
    current_total = current_n + current_p
    current_magnitude = np.linalg.norm(current_total, axis=1)

    def species_moments(density):
        number = float(np.dot(weights, density))
        r2_mean = float(np.dot(weights, density * radius2) / max(number, 1.0e-300))
        q0 = float(np.dot(weights, density * quadrupole_kernel))
        return number, float(np.sqrt(max(r2_mean, 0.0))), q0

    number_n, rms_n, q0_n = species_moments(rho_n)
    number_p, rms_p, q0_p = species_moments(rho_p)
    number_m, rms_m, q0_m = species_moments(rho_matter)
    current_l1 = float(np.dot(weights, current_magnitude))
    current_l2 = float(np.sqrt(np.dot(weights, current_magnitude**2)))
    q20_scale = np.sqrt(5.0 / (16.0 * np.pi))
    summary = {
        "density_units": "fm^-3",
        "integrated_particle_numbers": {"n": number_n, "p": number_p,
                                         "matter": number_m},
        "target_particle_numbers": targets,
        "rms_radii_fm": {"neutron": rms_n, "proton": rms_p, "matter": rms_m},
        "quadrupole_moments_fm2": {
            "Q0_neutron": q0_n, "Q0_proton": q0_p, "Q0_matter": q0_m,
            "Q20_neutron": q20_scale * q0_n,
            "Q20_proton": q20_scale * q0_p,
            "Q20_matter": q20_scale * q0_m,
            "Q20_convention": "sqrt(5/(16*pi))*integral rho*(2z^2-x^2-y^2) d^3r",
        },
        "spatial_baryon_current": {
            "current_l1_integral": current_l1,
            "current_l1_units": "dimensionless",
            "current_l2_integral": current_l2,
            "current_l2_units": "fm^-3/2",
        },
        "files": {"csv": str(csv_path), "npz": str(out / "observables.npz")},
        "grid": {"points": int(basis.point_count), "chunk_size": chunk_size,
                 "radius_fm": basis.radius_fm, "spacing_fm": basis.spacing_fm,
                 "kappa_max": basis.kappa_max},
    }
    npz_path = out / "observables.npz"
    descriptor, temporary = tempfile.mkstemp(prefix=npz_path.name + ".", dir=out)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream, **arrays, rho_matter_fm3=rho_matter,
                current_total_fm3=current_total,
                current_magnitude_fm3=current_magnitude,
                metadata_json=np.asarray(json.dumps(json_value(summary), sort_keys=True)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, npz_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return json_value(summary)
