from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def single_particle_metrics(dpl_wf: Path, scf_wf: Path, out_csv: Path) -> dict:
    dpl = np.load(dpl_wf)
    scf = np.load(scf_wf)
    scf_map = {}
    for i in range(len(scf["energy"])):
        key = (str(scf["species"][i]), str(scf["name"][i]).strip(), int(scf["kappa"][i]))
        scf_map[key] = {
            "energy": float(scf["energy"][i]),
            "occupancy": float(scf["occupancy"][i]),
        }

    rows = []
    occupied_diffs = []
    for i in range(len(dpl["energy"])):
        key = (str(dpl["species"][i]), str(dpl["name"][i]).strip(), int(dpl["kappa"][i]))
        if key not in scf_map:
            continue
        occ = float(dpl["occupancy"][i])
        eps_dpl = float(dpl["energy"][i])
        eps_scf = scf_map[key]["energy"]
        delta = eps_dpl - eps_scf
        rows.append({
            "species": key[0],
            "name": key[1],
            "kappa": key[2],
            "occupancy_dpl": occ,
            "occupancy_scf": scf_map[key]["occupancy"],
            "epsilon_dpl": eps_dpl,
            "epsilon_scf": eps_scf,
            "delta": delta,
        })
        if occ > 1.0e-8 or scf_map[key]["occupancy"] > 1.0e-8:
            occupied_diffs.append(delta)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys()) if rows else [
            "species", "name", "kappa", "occupancy_dpl", "occupancy_scf",
            "epsilon_dpl", "epsilon_scf", "delta",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    arr = np.asarray(occupied_diffs, dtype=np.float64)
    return {
        "matched_orbits": len(rows),
        "occupied_orbits": int(arr.size),
        "occupied_mae": float(np.mean(np.abs(arr))) if arr.size else None,
        "occupied_rmse": float(np.sqrt(np.mean(arr * arr))) if arr.size else None,
    }


def summarize_case(name: str, dpl_dir: Path, scf_dir: Path, out_dir: Path) -> dict:
    dpl_summary = load_json(dpl_dir / "summary.json")
    scf_summary = load_json(scf_dir / "summary.json")
    dpl_e = dpl_summary["energy"]
    scf_e = scf_summary["energy"]
    sp = single_particle_metrics(
        dpl_dir / "final_wavefunctions.npz",
        scf_dir / "final_wavefunctions.npz",
        out_dir / name / "single_particle_compare.csv",
    )
    row = {
        "nucleus": name,
        "dpl_epochs": dpl_summary.get("epochs"),
        "dpl_best_epoch": dpl_summary.get("best_epoch"),
        "dpl_best_loss": dpl_summary.get("best_loss"),
        "dpl_E_A_no_com": dpl_e.get("e_per_A_no_com", dpl_e.get("e_per_A")),
        "scf_E_A_no_com": scf_e.get("e_per_A_no_com", scf_e.get("e_per_A")),
        "delta_E_A_no_com": dpl_e.get("e_per_A_no_com", dpl_e.get("e_per_A"))
        - scf_e.get("e_per_A_no_com", scf_e.get("e_per_A")),
        "dpl_E_A_with_com": dpl_e.get("e_per_A_with_com"),
        "scf_E_A_with_com": scf_e.get("e_per_A_with_com"),
        "dpl_Rch_no_com": dpl_e.get("charge_radius_no_com"),
        "scf_Rch_no_com": scf_e.get("charge_radius_no_com"),
        "sp_occ_mae": sp["occupied_mae"],
        "sp_occ_rmse": sp["occupied_rmse"],
    }
    (out_dir / name).mkdir(parents=True, exist_ok=True)
    (out_dir / name / "compare_summary.json").write_text(
        json.dumps({"summary": row, "single_particle": sp}, indent=2) + "\n",
        encoding="utf-8",
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare old DPL output against SCF baselines")
    parser.add_argument("--dpl-root", required=True)
    parser.add_argument("--scf-root", default="outputs/pkdd_compare_scf")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dpl_root = Path(args.dpl_root)
    scf_root = Path(args.scf_root)
    out = Path(args.out)
    rows = []
    for dpl_dir in sorted(p for p in dpl_root.iterdir() if p.is_dir()):
        scf_dir = scf_root / dpl_dir.name
        if not (dpl_dir / "summary.json").exists() or not (scf_dir / "summary.json").exists():
            continue
        rows.append(summarize_case(dpl_dir.name, dpl_dir, scf_dir, out))

    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(out / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
