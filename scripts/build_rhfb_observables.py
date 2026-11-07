#!/usr/bin/env python3
"""Collect neural RHFB and Fortran S-RHFB-C observables into CSV tables.

Reads only saved files.  For every nucleus in ``<nn-root>/{production,dripline}/``
(evaluation ``summary.json``) the Fortran reference is taken from
``--fortran-dripline/<nucleus>`` for drip-line runs and from the first
``--fortran-production`` root that has it otherwise.  All energies and radii
are without the centre-of-mass correction.

Outputs in --out: observables.csv, single_particle_levels.csv,
channel_energies.csv, plus nn/ and fortran/ copies of the raw files.
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

L = "spdfghijklmno"


def label(kappa, principal):
    l = -kappa - 1 if kappa < 0 else kappa
    return f"{principal}{L[l]}{2 * abs(kappa) - 1}/2"


def fortran_channels(run_dir, nucleus):
    """Last EXPECT table of the Fortran main output: {row: (n, p, total)}."""
    text = (run_dir / f"{nucleus}.PKA1").read_text(errors="replace")
    block = text.rsplit("BEGIN EXPECT", 1)[1].split("END EXPECT", 1)[0]
    rows = {}
    for line in block.splitlines():
        m = re.match(r"^\s(\S.*?\S)\s{2,}(-?[\d.]+(?:E[-+]\d+)?)(?:\s+(-?[\d.]+))?(?:\s+(-?[\d.]+))?\s*$", line)
        if not m:
            continue
        values = [float(v) for v in m.groups()[1:] if v is not None]
        if len(values) == 3:
            rows[m.group(1).strip()] = values
        elif len(values) == 1:
            rows[m.group(1).strip()] = [None, None, values[0]]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn-root", type=Path, required=True,
                        help="evaluation summaries: <root>/{production,dripline}/<nucleus>/summary.json")
    parser.add_argument("--training-root", type=Path, default=None,
                        help="optional training summaries with the convergence record")
    parser.add_argument("--fortran-dripline", type=Path, required=True)
    parser.add_argument("--fortran-production", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out

    def fortran_dir(set_name, nucleus):
        roots = [args.fortran_dripline] if set_name == "dripline" else args.fortran_production
        for root in roots:
            if (root / nucleus / "parsed.json").exists():
                return root / nucleus
        return None

    obs_rows, level_rows, channel_rows, missing = [], [], [], []
    for set_name in ("production", "dripline"):
        for summary_path in sorted((args.nn_root / set_name).glob("*/summary.json")):
            nucleus = summary_path.parent.name
            s = json.loads(summary_path.read_text())
            terms = s["energy_terms_mev"]
            nn = {
                "E_total": s["energy_total_no_com_mev"], "E_per_A": s["energy_per_a_no_com_mev"],
                "E_pair_n": terms["E_pair_n"], "E_pair_p": terms["E_pair_p"], "E_pair": terms["E_pair"],
                "r_charge": s["radii_fm"]["charge_radius_no_com"], "r_matter": s["radii_fm"]["rms_matter_no_com"],
                "r_n": s["radii_fm"]["rms_n_no_com"], "r_p": s["radii_fm"]["rms_p_no_com"],
                "lambda_n": s["chemical_potentials_mev"]["n"], "lambda_p": s["chemical_potentials_mev"]["p"],
            }
            fdir = fortran_dir(set_name, nucleus)
            ft = {}
            if fdir is not None:
                p = json.loads((fdir / "parsed.json").read_text())
                e = p["energy"]
                ft = {
                    "E_total": e["E_total_no_cm_MeV"], "E_per_A": e["E_per_A_no_cm_MeV"],
                    "E_pair_n": e["E_pair_MeV"]["n"], "E_pair_p": e["E_pair_MeV"]["p"],
                    "E_pair": e["E_pair_MeV"]["total"],
                    "r_charge": p["radii_fm"]["charge_without_cm"],
                    "r_matter": p["radii_fm"]["rms_without_cm"]["total"],
                    "r_n": p["radii_fm"]["rms_without_cm"]["n"], "r_p": p["radii_fm"]["rms_without_cm"]["p"],
                    "lambda_n": p["lambda_MeV"]["n"], "lambda_p": p["lambda_MeV"]["p"],
                }
            else:
                missing.append(f"{set_name}/{nucleus}")
            row = {"set": set_name, "nucleus": nucleus,
                   "Z": s["nucleus"]["z"], "N": s["nucleus"]["n"], "A": s["nucleus"]["a"],
                   "fortran_reference": f"{fdir.parent.name}/{fdir.name}" if fdir else ""}
            if args.training_root is not None:
                training = args.training_root / set_name / nucleus / "summary.json"
                conv = (json.loads(training.read_text()).get("convergence") or {}) if training.exists() else {}
                row.update({"converged": conv.get("converged"), "epochs": conv.get("epochs_completed"),
                            "final_sigma_change_kev": conv.get("final_change_kev"),
                            "final_learning_rate": conv.get("final_learning_rate")})
            for key, value in nn.items():
                row[f"{key}_nn"] = value
                row[f"{key}_fortran"] = ft.get(key)
                row[f"{key}_diff"] = value - ft[key] if key in ft else None
            obs_rows.append(row)

            for name, value in terms.items():
                channel_rows.append({"set": set_name, "nucleus": nucleus, "source": "NN", "channel": name,
                                     "neutron": None, "proton": None, "total": value})
            if fdir is not None and (fdir / f"{nucleus}.PKA1").exists():
                for name, (vn, vp, vt) in fortran_channels(fdir, nucleus).items():
                    channel_rows.append({"set": set_name, "nucleus": nucleus, "source": "Fortran",
                                         "channel": name, "neutron": vn, "proton": vp, "total": vt})

            fl = {}
            if ft:
                for species in ("n", "p"):
                    for r in p["levels"][species]["canonical"]:
                        m = re.match(r"^[NP]\.?\s*(\d+)([a-z])\.?(\d+)/2$", r["state"].strip())
                        if not m:
                            continue
                        l = L.index(m.group(2))
                        kappa = -(l + 1) if int(m.group(3)) == 2 * l + 1 else l
                        fl[(species, kappa, int(m.group(1)))] = r
            for lv in s["canonical_levels"]:
                r = fl.get((lv["species"], lv["kappa"], lv["principal"]))
                level_rows.append({
                    "set": set_name, "nucleus": nucleus, "species": lv["species"], "kappa": lv["kappa"],
                    "principal": lv["principal"], "state": label(lv["kappa"], lv["principal"]),
                    "degeneracy": lv["degeneracy"], "blocked": lv["blocked"],
                    "v2_nn": lv["v2"], "e_nn_mev": lv["h_nn_mev"], "delta_nn_mev": lv["delta_nn_mev"],
                    "v2_fortran": r["v2"] if r else None, "e_fortran_mev": r["e_canonical"] if r else None,
                    "delta_fortran_mev": r["delta_canonical"] if r else None,
                    "e_diff_kev": 1000.0 * (lv["h_nn_mev"] - r["e_canonical"]) if r else None,
                    "nn_energy_definition": "orbital_expectation_h_nunu_not_hamiltonian_eigenvalue",
                    "state_matching": "species_kappa_saved_principal_not_wavefunction_overlap",
                })

            dst = out / "nn" / set_name / nucleus
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_path, dst / "summary.json")
            if fdir is not None:
                fdst = out / "fortran" / set_name / nucleus
                fdst.mkdir(parents=True, exist_ok=True)
                for item in ("parsed.json", f"{nucleus}.PKA1", "LEV", "RHF"):
                    src = fdir / item
                    if src.is_dir():
                        shutil.copytree(src, fdst / item, dirs_exist_ok=True)
                    elif src.exists():
                        shutil.copy2(src, fdst / item)

    for name, rows in (("observables.csv", obs_rows), ("single_particle_levels.csv", level_rows),
                       ("channel_energies.csv", channel_rows)):
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(len(obs_rows), "nuclei;", len(level_rows), "levels;", len(channel_rows),
          "channel rows; missing Fortran:", missing)


if __name__ == "__main__":
    main()
