#!/usr/bin/env python3
"""Render and audit saved O17 self-consistent evaluations without rerunning PH.

Run this only after the training process has ended and under the repository's
bounded-memory launcher.  The script reads accepted-evaluation JSON/NPZ files,
reconstructs the last-step full D1S gap action in column chunks, and writes
postprocessing artifacts.  It does not train or solve a spectrum.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="O17 run directory containing run_configuration.json")
    parser.add_argument("--device", choices=("cpu",), default="cpu",
                        help="Postprocessing is deliberately CPU-only")
    parser.add_argument("--column-chunk", type=int, default=16)
    parser.add_argument("--active-chunk", type=int, default=4)
    return parser


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_step_records(run_dir: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
    config_path = run_dir / "run_configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records_dir = run_dir / "accepted_evaluations"
    json_paths = sorted(records_dir.glob("step_*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no saved evaluations in {records_dir}")
    records = []
    for path in json_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append((record, path))
    initial = [item for item in records if int(item[0].get("step", -1)) == 0]
    if not initial:
        raise ValueError("saved evaluations do not contain step 0")
    accepted = [item for item in records
                if item[0].get("event") == "accepted_update"
                and int(item[0].get("step", 0)) > 0]
    if not accepted:
        # Some trainer versions label the initial evaluation separately and
        # use a generic accepted-evaluation event for accepted updates.
        accepted = [item for item in records if int(item[0].get("step", 0)) > 0]
    selected = [initial[0]]
    if accepted:
        selected.append(max(accepted, key=lambda item: int(item[0]["step"])))
    return config, selected


def _matrix_norm(np, value) -> float:
    return float(np.linalg.norm(value.reshape(-1)))


def _analyze_saved_state(np, record: dict[str, Any], arrays, basis_config: dict[str, Any]):
    """Return structural checks, angular averages, and integrated densities."""
    d = int(basis_config["dimension"])
    m = int(basis_config.get("active_dimension", 42))
    nr = int(round(float(basis_config["radius_fm"]) /
                   float(basis_config["spacing_fm"]))) - 1
    nt, nphi = int(basis_config["n_theta"]), int(basis_config["n_phi"])
    spacing = float(basis_config["spacing_fm"])
    radii = spacing * np.arange(1, nr + 1, dtype=np.float64)
    if d != 432 or m != 42 or nr != 9 or nt != 7 or nphi != 14:
        raise ValueError(f"unexpected O17 postprocessing basis: D={d}, M={m}, "
                         f"Nr={nr}, ntheta={nt}, nphi={nphi}")
    mu, wmu = np.polynomial.legendre.leggauss(nt)
    angular_weights = np.repeat(wmu, nphi) * (2.0 * np.pi / nphi)
    point_weights = (spacing * radii[:, None] ** 2
                     * angular_weights[None, :]).reshape(-1)
    radial: dict[str, Any] = {"r_fm": radii}
    structural_rows = []
    energy = record["energy_mev"]
    computed_numbers = {}

    for species in ("n", "p"):
        prefix = species + "_"
        q = np.asarray(arrays[prefix + "Q"], dtype=np.complex128)
        rho = np.asarray(arrays[prefix + "rho_active"], dtype=np.complex128)
        chi = np.asarray(arrays[prefix + "chi_active"], dtype=np.complex128)
        rho_primitive = np.asarray(arrays[prefix + "rho_primitive"], dtype=np.complex128)
        chi_primitive = np.asarray(arrays[prefix + "chi_primitive"], dtype=np.complex128)
        current = np.asarray(arrays[prefix + "baryon_current_points"], dtype=np.float64)
        scalar = np.asarray(arrays[prefix + "scalar_density_points"], dtype=np.float64)
        if q.shape != (d, m) or rho.shape != (m, m) or chi.shape != (m, m):
            raise ValueError(f"unexpected {species} factor shapes: {q.shape}, {rho.shape}, {chi.shape}")
        if current.shape != (nr * nt * nphi, 4) or scalar.shape != (current.shape[0],):
            raise ValueError(f"unexpected {species} current/scalar grid shape")

        identity = np.eye(m, dtype=np.complex128)
        generalized = np.block([[rho, chi], [-chi.conj(), identity - rho.conj()]])
        derived_rho = q @ rho @ q.conj().T
        derived_chi = q @ chi @ q.T
        q_error = _matrix_norm(np, q.conj().T @ q - identity)
        rho_hermiticity = _matrix_norm(np, rho - rho.conj().T)
        chi_skew = _matrix_norm(np, chi + chi.T)
        generalized_hermiticity = _matrix_norm(np, generalized - generalized.conj().T)
        generalized_purity = _matrix_norm(np, generalized @ generalized - generalized)
        primitive_rho_error = _matrix_norm(np, derived_rho - rho_primitive)
        primitive_chi_error = _matrix_norm(np, derived_chi - chi_primitive)
        lower_rows = np.arange(d) % (2*nr) >= nr
        diagonal_density = np.diag(rho_primitive).real
        lower_number = float(diagonal_density[lower_rows].sum())
        upper_number = float(diagonal_density[~lower_rows].sum())
        # Dirac-component probabilities are basis diagnostics, not a
        # projector onto occupied antiparticle/Dirac-sea branches.
        component_diagnostics = {
            "large_component_number": upper_number,
            "small_component_number": lower_number,
            "small_component_fraction": lower_number / (upper_number+lower_number),
        }
        for a_name, a_mask in (("G", ~lower_rows), ("F", lower_rows)):
            for b_name, b_mask in (("G", ~lower_rows), ("F", lower_rows)):
                component_diagnostics[f"chi_{a_name}{b_name}_frobenius"] = _matrix_norm(
                    np, chi_primitive[np.ix_(a_mask, b_mask)])

        # Stored order is radial-major, then Gauss-Legendre theta, with phi
        # varying fastest.  The weights implement integral(dOmega)/(4*pi).
        current_grid = current.reshape(nr, nt, nphi, 4)
        scalar_grid = scalar.reshape(nr, nt, nphi)
        weights_grid = (wmu[:, None] * (2.0 * np.pi / nphi))
        angular_mean_current = np.sum(current_grid * weights_grid[None, :, :, None],
                                      axis=(1, 2)) / (4.0 * np.pi)
        angular_mean_scalar = np.sum(scalar_grid * weights_grid[None, :, :],
                                     axis=(1, 2)) / (4.0 * np.pi)
        radial[f"{species}_baryon_j0_fm3"] = angular_mean_current[:, 0]
        radial[f"{species}_scalar_density_fm3"] = angular_mean_scalar
        radial[f"{species}_current_x_fm3"] = angular_mean_current[:, 1]
        radial[f"{species}_current_y_fm3"] = angular_mean_current[:, 2]
        radial[f"{species}_current_z_fm3"] = angular_mean_current[:, 3]
        number_integral = float(np.sum(point_weights * current[:, 0]))
        computed_numbers[species] = number_integral
        species_record = record["species"][species]
        structural_rows.append({
            "step": int(record["step"]), "event": record.get("event"), "species": species,
            "Q_orthonormal_frobenius": q_error,
            "rho_hermiticity_frobenius": rho_hermiticity,
            "chi_skew_symmetry_frobenius": chi_skew,
            "generalized_density_hermiticity_frobenius": generalized_hermiticity,
            "generalized_density_idempotency_frobenius": generalized_purity,
            "primitive_rho_reconstruction_frobenius": primitive_rho_error,
            "primitive_chi_reconstruction_frobenius": primitive_chi_error,
            "coordinate_current_number_integral": number_integral,
            "coordinate_current_number_metadata_abs_error": abs(
                number_integral - float(species_record["particle_number_from_coordinate_current"])),
            "density_trace": float(np.trace(rho).real),
            "density_trace_metadata_abs_error": abs(
                float(np.trace(rho).real) - float(species_record["particle_number_from_density"])),
            "reported_full_commutator_residual_mev": float(np.sqrt(max(
                0.0, float(species_record["full_commutator_frobenius_squared_mev2"])))),
            "reported_normal_commutator_squared_mev2": float(
                species_record["normal_commutator_squared_mev2"]),
            "reported_anomalous_commutator_squared_mev2": float(
                species_record["anomalous_commutator_squared_mev2"]),
            "scaled_stationarity_residual": float(species_record["scaled_stationarity_residual"]),
            "number_error_reported": float(species_record["number_error"]),
            "physical_identity_certified": bool(species_record.get("physical_identity_certified", False)),
            **component_diagnostics,
        })

    channels = energy.get("channels", {})
    channel_values = {str(key): float(value) for key, value in channels.items()}
    channel_sum = sum(channel_values.values())
    total = float(energy["total"])
    energy_row: dict[str, Any] = {
        "step": int(record["step"]), "event": record.get("event"),
        "total_energy_mev": total,
        "particle_hole_energy_mev": float(energy["particle_hole"]),
        "d1s_pairing_energy_mev": float(energy["d1s_pairing"]),
        "energy_channel_sum_mev": channel_sum,
        "energy_channel_sum_minus_total_mev": channel_sum - total,
        "energy_channel_sum_minus_particle_hole_mev": channel_sum - float(energy["particle_hole"]),
        "energy_channel_count": len(channel_values),
    }
    energy_row.update({f"energy_channel_{key}_mev": value
                       for key, value in channel_values.items()})
    for species in ("n", "p"):
        values = record["species"][species]
        for key, value in values.items():
            if isinstance(value, (int, float, bool)) or value is None:
                energy_row[f"{species}_{key}"] = value
    return structural_rows, energy_row, radial, computed_numbers


def _write_density_plots(np, run_dir: Path, radial_csv: Path, radial: dict[str, Any],
                         selected: list[tuple[dict[str, Any], Path]], energy_rows: list[dict[str, Any]],
                         structural_rows: list[dict[str, Any]], radial_series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in energy_rows]
    energies = [float(row["total_energy_mev"]) for row in energy_rows]
    particle_hole = [float(row["particle_hole_energy_mev"]) for row in energy_rows]
    pairing = [float(row["d1s_pairing_energy_mev"]) for row in energy_rows]
    residual_by_step: dict[int, float] = {}
    for row in structural_rows:
        step = int(row["step"])
        residual_by_step[step] = max(residual_by_step.get(step, 0.0),
                                     float(row["reported_full_commutator_residual_mev"]))
    residuals = [residual_by_step.get(step, float("nan")) for step in steps]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax_e, ax_r = axes[0]
    ax_np, ax_scalar = axes[1]
    ax_e.plot(steps, np.asarray(energies)-energies[0], marker="o", color="#1f4e79")
    ax_e.set(xlabel="saved update step", ylabel="EDF energy change (MeV)",
             title=f"Energy relative to initial {energies[0]:+.6f} MeV")
    ax_e.grid(True, alpha=0.25)
    for species, color in (("n", "#1f4e79"), ("p", "#ad6a26")):
        values = [row["reported_full_commutator_residual_mev"] for row in structural_rows
                  if row["species"] == species]
        ax_r.plot(steps, values, marker="o", color=color, label=species)
    ax_r.set(xlabel="saved update step", ylabel="residual (MeV)",
             title="Full commutator residual")
    ax_r.grid(True, alpha=0.25)
    ax_r.legend(frameon=False)
    for index, (step, profile) in enumerate(radial_series):
        style = "--" if index == 0 and len(radial_series) > 1 else "-"
        for species, color in (("n", "#1f4e79"), ("p", "#ad6a26")):
            ax_np.plot(profile["r_fm"], profile[f"{species}_baryon_j0_fm3"],
                       linestyle=style, color=color, label=f"{species}, step {step}")
            ax_scalar.plot(profile["r_fm"], profile[f"{species}_scalar_density_fm3"],
                           linestyle=style, color=color, label=f"{species}, step {step}")
    ax_np.set(xlabel="r (fm)", ylabel=r"angular mean $j^0$ (fm$^{-3}$)",
              title="Baryon density from stored currents")
    ax_np.grid(True, alpha=0.25)
    ax_np.legend(frameon=False)
    ax_scalar.set(xlabel="r (fm)", ylabel=r"angular mean scalar density (fm$^{-3}$)",
                  title="Scalar density from stored bilinears")
    ax_scalar.grid(True, alpha=0.25)
    ax_scalar.legend(frameon=False)
    last_step = max(steps)
    fig.suptitle(f"O17 saved self-consistent evaluations — UNCONVERGED (latest step {last_step})")
    output = radial_csv.parent
    fig.savefig(output / "o17_selfconsistent_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "o17_selfconsistent_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def _reconstruct_pairing(np, torch, basis, record, arrays, out_path: Path,
                         column_chunk: int, active_chunk: int) -> dict[str, Any]:
    from dpl_rhf.rhfb.gaussian_pairing import GaussianPairingAction

    d = basis.dimension
    matrices: dict[str, Any] = {}
    diagnostics = []
    with torch.no_grad():
        for species in ("n", "p"):
            prefix = species + "_"
            q = torch.as_tensor(np.asarray(arrays[prefix + "Q"]),
                                dtype=torch.complex128, device="cpu")
            chi = torch.as_tensor(np.asarray(arrays[prefix + "chi_active"]),
                                  dtype=torch.complex128, device="cpu")
            action = GaussianPairingAction(
                basis, q, chi, species=species, chunk_size=column_chunk,
                active_chunk_size=active_chunk, lmax=6, checkpoint_blocks=False)
            delta = np.empty((d, d), dtype=np.complex128)
            for start in range(0, d, column_chunk):
                stop = min(d, start + column_chunk)
                units = torch.zeros((d, stop - start), dtype=torch.complex128)
                units[torch.arange(start, stop), torch.arange(stop - start)] = 1.0
                delta[:, start:stop] = action.apply(units).cpu().numpy()

            probe = np.asarray(arrays[prefix + "delta_action_probe_vectors"], dtype=np.complex128)
            expected = np.asarray(arrays[prefix + "delta_action_probe_results"], dtype=np.complex128)
            reconstructed_probe = delta @ probe
            difference = reconstructed_probe - expected
            absolute_probe_error = float(np.linalg.norm(difference))
            expected_norm = float(np.linalg.norm(expected))
            relative_probe_error = absolute_probe_error / max(expected_norm, np.finfo(float).tiny)
            skew_error = float(np.linalg.norm(delta + delta.T))
            relative_skew_error = skew_error / max(float(np.linalg.norm(delta)), np.finfo(float).tiny)
            matrices[f"Delta_{species}"] = delta
            diagnostics.append({
                "step": int(record["step"]), "species": species,
                "matrix_shape": list(delta.shape),
                "matrix_frobenius_norm": float(np.linalg.norm(delta)),
                "skew_symmetry_frobenius_error": skew_error,
                "relative_skew_symmetry_error": relative_skew_error,
                "probe_frobenius_error": absolute_probe_error,
                "relative_probe_frobenius_error": relative_probe_error,
                "probe_max_abs_error": float(np.max(np.abs(difference))),
                "probe_count": int(probe.shape[1]),
                "lmax": 6,
                "column_chunk": column_chunk,
                "active_chunk": active_chunk,
            })
    np.savez_compressed(out_path, **matrices)
    return {"matrix_file": out_path.name, "species": diagnostics,
            "construction": "full primitive D1S action applied to chunked identity columns; no Nambu or D^4 tensor"}


def main() -> int:
    args = _parser().parse_args()
    if args.column_chunk < 1 or args.active_chunk < 1:
        raise ValueError("column and active chunk sizes must be positive")

    # This must precede reading any run data or importing numerical packages.
    from dpl_rhf.rhfb.execution_limits import require_bounded_cgroup
    resource_limits = require_bounded_cgroup()

    import numpy as np
    import torch
    from dpl_rhf.rhfb.basis import AngularRadialBasis

    run_dir = args.run_dir.resolve()
    config, selected = _load_step_records(run_dir)
    basis_config = dict(config["basis"])
    basis_config["active_dimension"] = int(config.get("active_dimension", 42))
    out_dir = run_dir / "postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    structural_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    latest_radial = None
    latest_arrays = None
    latest_record = None
    radial_series = []
    for record, json_path in selected:
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.is_file():
            raise FileNotFoundError(f"missing matching evaluation arrays: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        structural, energy, radial, _numbers = _analyze_saved_state(
            np, record, arrays, basis_config)
        structural_rows.extend(structural)
        energy_rows.append(energy)
        radial_series.append((int(record["step"]), radial))
        if latest_record is None or int(record["step"]) >= int(latest_record["step"]):
            latest_record, latest_arrays, latest_radial = record, arrays, radial

    energy_csv = out_dir / "energy_observables.csv"
    structural_csv = out_dir / "structural_checks.csv"
    radial_csv = out_dir / "radial_density.csv"
    _write_csv(energy_csv, energy_rows)
    _write_csv(structural_csv, structural_rows)
    _write_csv(radial_csv, [{"step": step, **{key: float(value[index])
                             for key, value in profile.items()}}
                           for step, profile in radial_series
                           for index in range(len(profile["r_fm"]))])
    _write_density_plots(np, run_dir, radial_csv, latest_radial, selected,
                         energy_rows, structural_rows, radial_series)

    basis = AngularRadialBasis(
        radius_fm=float(basis_config["radius_fm"]),
        spacing_fm=float(basis_config["spacing_fm"]),
        kappa_max=int(basis_config["kappa_max"]),
        n_theta=int(basis_config["n_theta"]),
        n_phi=int(basis_config["n_phi"]),
        derivative_order=int(basis_config["derivative_order"]),
    )
    pairing = _reconstruct_pairing(
        np, torch, basis, latest_record, latest_arrays,
        out_dir / "full_primitive_pairing_gaps.npz",
        args.column_chunk, args.active_chunk)
    _write_csv(out_dir / "pairing_delta_checks.csv", pairing["species"])
    summary = {
        "status": "postprocessed_saved_evaluations",
        "run_dir": str(run_dir),
        "selected_steps": [{"step": int(record["step"]), "event": record.get("event"),
                            "json": str(path.relative_to(run_dir))}
                           for record, path in selected],
        "latest_saved_step": int(latest_record["step"]),
        "converged": False,
        "physical_result_accepted": False,
        "resource_limits": resource_limits,
        "basis": basis_config,
        "outputs": {
            "energy_observables_csv": energy_csv.name,
            "structural_checks_csv": structural_csv.name,
            "radial_density_csv": radial_csv.name,
            "figure_png": "o17_selfconsistent_summary.png",
            "figure_pdf": "o17_selfconsistent_summary.pdf",
            "pairing_delta_checks_csv": "pairing_delta_checks.csv",
            "pairing_delta_npz": pairing["matrix_file"],
        },
        "pairing_reconstruction": pairing,
        "energy_channel_closure": [
            {"step": row["step"], "sum_minus_total_mev": row["energy_channel_sum_minus_total_mev"],
             "sum_minus_particle_hole_mev": row["energy_channel_sum_minus_particle_hole_mev"]}
            for row in energy_rows],
        "scope": "saved-state audit and plotting only; no PH rebuild, spectrum solve, or training",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
