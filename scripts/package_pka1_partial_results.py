from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch


COMPLETED_NUCLEI = ("O16", "Ca40", "Ca48")
COMPLETED_FILES = (
    "summary.json",
    "history.csv",
    "best_checkpoint.pt",
    "density_matrix_factors.npz",
    "local_observables.npz",
    "nonlocal_hamiltonians.npz",
    "train.log",
    "compare.log",
)
CORE_FILES = (
    "detailed_comparison.json",
    "core1204_scf_profiles.npz",
    "local_s_v_comparison.npz",
    "s_plus_minus_v_local_dpl_core.png",
    "single_particle_levels.csv",
    "nonlocal_density_R_comparison.npz",
    "nonlocal_density_R_metrics.csv",
    "nonlocal_density_R_error_overview.png",
)
CORE_DIRECTORIES = ("nonlocal_density_R_blocks",)
SNAPSHOT_NUCLEUS = "Sn132"
SNAPSHOT_FILES = ("training_checkpoint.pt", "history.csv", "train.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package completed PKA1 comparisons and an incomplete Sn132 checkpoint"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/pkdd_pka1_five_nuclei_refresh4_500/PKA1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/PKA1_DPL_Core_completed3_Sn132_checkpoint_20260816.zip"),
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required result file is missing: {path}")


def copy_file(source: Path, destination: Path) -> None:
    require_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def comparison_row(nucleus: str, case_dir: Path) -> dict[str, float | str]:
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads(
        (case_dir / "core_comparison" / "detailed_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    dpl = comparison["dpl"]
    core = comparison["core"]
    differences = comparison["differences"]
    dpl_ea = float(dpl["energy_per_a_no_com_mev"])
    core_ea = float(core["e_per_A_no_com"])
    return {
        "nucleus": nucleus,
        "dpl_energy_total_no_com_mev": float(dpl["energy_total_no_com_mev"]),
        "core_energy_total_no_com_mev": float(core["e_total_no_com"]),
        "delta_energy_total_no_com_mev": float(
            differences["energy_total_no_com_mev"]
        ),
        "dpl_energy_per_a_no_com_mev": dpl_ea,
        "core_energy_per_a_no_com_mev": core_ea,
        "delta_energy_per_a_no_com_mev": float(
            differences["energy_per_a_no_com_mev"]
        ),
        "relative_energy_error_percent": 100.0 * abs(dpl_ea - core_ea) / abs(core_ea),
        "dpl_matter_radius_no_com_fm": float(dpl["rms_matter_no_com"]),
        "core_matter_radius_no_com_fm": float(core["rms_matter_no_com"]),
        "delta_matter_radius_no_com_fm": float(
            differences["rms_matter_no_com_fm"]
        ),
        "dpl_charge_radius_no_com_fm": float(dpl["charge_radius_no_com"]),
        "core_charge_radius_no_com_fm": float(core["charge_radius_no_com"]),
        "delta_charge_radius_no_com_fm": float(
            differences["charge_radius_no_com_fm"]
        ),
        "occupied_level_mae_mev": float(differences["occupied_level_mae_mev"]),
        "occupied_level_rmse_mev": float(differences["occupied_level_rmse_mev"]),
        "nonlocal_density_R_relative_l2_percent": float(
            differences["nonlocal_density_R"]["aggregate_relative_l2_percent"]
        ),
        "nonlocal_density_R_rmse_fm_inverse": float(
            differences["nonlocal_density_R"]["aggregate_rmse_fm_inverse"]
        ),
        "projected_stationarity_rms_mev": float(
            summary["diagnostics"]["projected_stationarity_rms_mev"]
        ),
        "training_dirac_solves": int(summary["diagnostics"]["training_dirac_solves"]),
    }


def write_summaries(root: Path, rows: list[dict[str, float | str]]) -> None:
    (root / "COMPARISON_SUMMARY.json").write_text(
        json.dumps({"energy_convention": "no center-of-mass correction", "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with (root / "COMPARISON_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PKA1 DPL-Core completed-result comparison",
        "",
        "All energies and radii use the no-center-of-mass-correction convention.",
        "",
        "| Nucleus | DPL E/A (MeV) | Core E/A (MeV) | Delta E/A (MeV) | Relative error | Delta Rm (fm) | Delta Rc (fm) | Level MAE (MeV) | R relative L2 | Stationarity RMS (MeV) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {nucleus} | {dpl_energy_per_a_no_com_mev:.8f} | "
            "{core_energy_per_a_no_com_mev:.8f} | {delta_energy_per_a_no_com_mev:+.8f} | "
            "{relative_energy_error_percent:.5f}% | {delta_matter_radius_no_com_fm:+.6f} | "
            "{delta_charge_radius_no_com_fm:+.6f} | {occupied_level_mae_mev:.5f} | "
            "{nonlocal_density_R_relative_l2_percent:.5f}% | "
            "{projected_stationarity_rms_mev:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The plotted S+V and S-V curves contain local Hartree plus rearrangement terms only. "
            "PKA1 Fock exchange is tested through per-species/per-kappa GG/GF/FG/FF "
            "nonlocal-density R(r,r') comparisons against Core-1204.",
        ]
    )
    (root / "COMPARISON_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(root: Path, sn_next_epoch: int, snapshot_time: str) -> None:
    text = f"""# PKA1 DPL/Core result snapshot

Snapshot time: {snapshot_time}

## Completed results

O16, Ca40 and Ca48 completed 500 neural variational epochs. Their directories
contain the final neural checkpoint, density-matrix factors, local observables,
complete nonlocal total/Fock Hamiltonians, training history, Core-1204
comparison data, occupied levels and local S+/-V plots.

All primary comparisons omit the center-of-mass correction. Core-1204 is used
only after neural training for external validation and is not part of the loss.
Ordinary neural epochs perform zero Dirac solves.

## Incomplete result

Sn132 is a resumable training snapshot at next_epoch={sn_next_epoch}. It has no
final summary or Core comparison and must not be cited as a converged physical
result. Pb208 had no valid training result and is intentionally excluded.

## Fock interpretation

PKA1 Fock exchange is nonlocal. The S+V and S-V figures compare only local
Hartree plus rearrangement channels. Fock input is compared through the four
GG/GF/FG/FF components of R_kappa(r,r') for every species/kappa block. The
package contains DPL/Core/difference heatmaps, numerical metrics and full NPZ
arrays. Full Fock operators remain stored in nonlocal_hamiltonians.npz.

MANIFEST.sha256 covers every packaged file except the manifest itself.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256")
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    package_name = output.stem
    snapshot_time = datetime.now().astimezone().isoformat(timespec="seconds")

    with tempfile.TemporaryDirectory(prefix="pka1-package-", dir=output.parent) as temporary:
        package_root = Path(temporary) / package_name
        rows = []
        for nucleus in COMPLETED_NUCLEI:
            case_source = source / nucleus
            case_destination = package_root / "completed" / nucleus
            for relative in COMPLETED_FILES:
                copy_file(case_source / relative, case_destination / relative)
            for relative in CORE_FILES:
                copy_file(
                    case_source / "core_comparison" / relative,
                    case_destination / "core_comparison" / relative,
                )
            for relative in CORE_DIRECTORIES:
                source_directory = case_source / "core_comparison" / relative
                if not source_directory.is_dir():
                    raise FileNotFoundError(f"required result directory is missing: {source_directory}")
                shutil.copytree(
                    source_directory,
                    case_destination / "core_comparison" / relative,
                )
            rows.append(comparison_row(nucleus, case_source))

        sn_source = source / SNAPSHOT_NUCLEUS
        sn_destination = package_root / "incomplete" / SNAPSHOT_NUCLEUS
        # The trainer atomically replaces checkpoints, so this copy always
        # snapshots one complete inode even while training continues.
        for relative in SNAPSHOT_FILES:
            copy_file(sn_source / relative, sn_destination / relative)
        checkpoint = torch.load(
            sn_destination / "training_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        next_epoch = int(checkpoint["next_epoch"])
        (sn_destination / "STATUS.md").write_text(
            f"# Sn132 incomplete checkpoint\n\n"
            f"Snapshot time: {snapshot_time}\n\n"
            f"Resume position: `next_epoch={next_epoch}`.\n\n"
            "This directory is not a converged result and has no Core comparison.\n",
            encoding="utf-8",
        )

        write_summaries(package_root, rows)
        write_readme(package_root, next_epoch, snapshot_time)
        write_manifest(package_root)

        temporary_zip = output.with_suffix(output.suffix + ".tmp")
        if temporary_zip.exists():
            temporary_zip.unlink()
        with zipfile.ZipFile(
            temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package_root.parent))
        temporary_zip.replace(output)

    archive_digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{archive_digest}  {output.name}\n", encoding="ascii"
    )
    print(output)
    print(f"sha256={archive_digest}")


if __name__ == "__main__":
    main()
