#!/usr/bin/env python3
"""Compare spherical neural RHFB summaries with the Fortran S-RHFB-C reference
(Core-1204 RHF where the Fortran reference cannot hold the kappa range).

Reads only saved files: ``<nn_root>/<nucleus>/summary.json`` and
``outputs/fortran_reference/runs/<nucleus>/parsed.json``.  All energies are
without the centre-of-mass correction; radii are without c.m. as well.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {-1: "s1/2", 1: "p1/2", -2: "p3/2", 2: "d3/2", -3: "d5/2", 3: "f5/2",
          -4: "f7/2", 4: "g7/2", -5: "g9/2"}


CORE_ROOT = ROOT.parents[1] / "outputs/no_refresh/PKA1"
CORE_RUNS = {"O16": "O16_expanded_strict", "Ca40": "Ca40_strict", "Ca48": "Ca48_strict",
             "Sn132": "Sn132_strict_1ep", "Pb208": "Pb208_strict_1ep"}


def _fortran_reference(path):
    """Fortran S-RHFB-C (same PKA1+D1S prescription) in normalized form."""
    ft = json.loads(path.read_text())
    e = ft["energy"]
    levels = {}
    for species in ("n", "p"):
        for row in ft.get("levels", {}).get(species, {}).get("canonical", []):
            # Fixed-width labels: "N.1s.1/2", or "N10s.1/2" for two-digit principals.
            match = re.match(r"^[NP]\.?\s*(\d+)([a-z])\.?(\d+/2)$", row["state"].strip())
            if match is None:
                continue
            label = "".join(match.groups())
            levels[(species, label)] = (row["e_canonical"], row["v2"], abs(row["delta_canonical"]))
    return {
        "source": "Fortran S-RHFB-C (PKA1+D1S)",
        "E_total": e["E_total_no_cm_MeV"], "E_per_A": e["E_per_A_no_cm_MeV"],
        "E_pair_n": e["E_pair_MeV"]["n"], "E_pair_p": e["E_pair_MeV"]["p"],
        "lambda_n": ft["lambda_MeV"]["n"], "lambda_p": ft["lambda_MeV"]["p"],
        "gap_n": ft["pairing_gap_average_MeV"]["n"],
        "r_n": ft["radii_fm"]["rms_without_cm"]["n"], "r_p": ft["radii_fm"]["rms_without_cm"]["p"],
        "r_ch": ft["radii_fm"]["charge_without_cm"], "E_cm": e["E_cm_MeV"], "levels": levels,
    }


def _core_reference(nucleus):
    """Core-1204 PKA1 RHF (no pairing) in normalized form, or None."""
    run = CORE_RUNS.get(nucleus)
    path = CORE_ROOT / str(run) / "core_comparison/detailed_comparison.json"
    if run is None or not path.exists():
        return None
    core = json.loads(path.read_text())["core"]
    levels = {}
    level_file = path.with_name("single_particle_levels.csv")
    if level_file.exists():
        import csv
        for row in csv.DictReader(level_file.open()):
            label = f"{row['principal']}{LABELS.get(int(row['kappa']), row['kappa'])}"
            levels[(row["species"], label)] = (float(row["core_energy_mev"]),
                                               float(row["occupation"]), 0.0)
    return {
        "source": "Core-1204 PKA1 RHF (no pairing)",
        "E_total": core["e_total_no_com"], "E_per_A": core["e_per_A_no_com"],
        "E_pair_n": 0.0, "E_pair_p": 0.0, "lambda_n": None, "lambda_p": None, "gap_n": None,
        "r_n": core["rms_n_no_com"], "r_p": core["rms_p_no_com"],
        "r_ch": core["charge_radius_no_com"], "E_cm": core.get("e_cm"), "levels": levels,
    }


def compare(nucleus, nn_dir, reference, secondary=None):
    nn = json.loads((nn_dir / "summary.json").read_text())
    terms = nn["energy_terms_mev"]
    ref = reference
    row = {
        "nucleus": nucleus, "reference_source": ref["source"],
        "nn_E_total": nn["energy_total_no_com_mev"], "ft_E_total": ref["E_total"],
        "nn_E_per_A": nn["energy_per_a_no_com_mev"], "ft_E_per_A": ref["E_per_A"],
        "nn_E_pair_n": terms["E_pair_n"], "ft_E_pair_n": ref["E_pair_n"],
        "nn_E_pair_p": terms["E_pair_p"], "ft_E_pair_p": ref["E_pair_p"],
        "nn_lambda_n": nn["chemical_potentials_mev"]["n"], "ft_lambda_n": ref["lambda_n"],
        "nn_lambda_p": nn["chemical_potentials_mev"]["p"], "ft_lambda_p": ref["lambda_p"],
        "nn_gap_n": nn.get("pairing_gap_reference_definition_mev", {}).get("n"),
        "ft_gap_n": ref["gap_n"],
        "nn_r_n": nn["radii_fm"]["rms_n_no_com"], "ft_r_n": ref["r_n"],
        "nn_r_p": nn["radii_fm"]["rms_p_no_com"], "ft_r_p": ref["r_p"],
        "nn_r_ch": nn["radii_fm"]["charge_radius_no_com"], "ft_r_ch": ref["r_ch"],
        "nn_stationarity_orbital_mev": nn["stationarity"]["orbital_residual_rms_mev"],
        "nn_stationarity_occupation_mev": nn["stationarity"]["occupation_residual_rms_mev"],
        "nn_epochs": nn.get("epochs_completed"),
        "nn_eigensolves": nn["counts"]["eigensolve_count"],
        "ft_cm_energy": ref["E_cm"],
    }
    row["dE_per_A"] = row["nn_E_per_A"] - row["ft_E_per_A"]
    if secondary is not None:
        row["secondary_source"] = secondary["source"]
        row["secondary_E_per_A"] = secondary["E_per_A"]
        row["secondary_r_ch"] = secondary["r_ch"]
    levels = []
    for level in nn["canonical_levels"]:
        label = f"{level['principal']}{LABELS.get(level['kappa'], level['kappa'])}"
        match = ref["levels"].get((level["species"], label))
        if match is None or max(level["occupation"], match[1]) < 1e-3:
            continue
        levels.append({
            "species": level["species"], "level": label,
            "nn_e": level["h_nn_mev"], "ft_e": match[0],
            "nn_v2": level["occupation"], "ft_v2": match[1],
            "nn_abs_delta": abs(level["delta_nn_mev"]), "ft_abs_delta": match[2],
        })
    row["levels"] = levels
    # Canonical spinors with equal occupations (e.g. several v^2=1 levels of one
    # kappa block) are defined only up to a rotation within that subspace, so
    # their individual h_nn are not invariant; the block trace is.  Fully
    # occupied levels are therefore compared through their per-block sum, and
    # partially occupied levels (unique canonical basis) one by one.
    groups, deviations, count = {}, [], 0
    for level in levels:
        if max(level["nn_v2"], level["ft_v2"]) <= 0.05:
            continue
        if min(level["nn_v2"], level["ft_v2"]) > 0.99:
            key = (level["species"], level["level"][1:])  # same (species, l, j) block
            nn_sum, ft_sum, n = groups.get(key, (0.0, 0.0, 0))
            groups[key] = (nn_sum + level["nn_e"], ft_sum + level["ft_e"], n + 1)
        else:
            deviations.append(abs(level["nn_e"] - level["ft_e"]))
            count += 1
    for nn_sum, ft_sum, n in groups.values():
        deviations.append(abs(nn_sum - ft_sum))
        count += n
    row["level_mae_mev"] = sum(deviations) / count if count else None
    row["level_mae_definition"] = ("per occupied level: |block-trace difference| for fully "
                                   "occupied levels of a kappa block, |h_nn difference| otherwise")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nn-root", default=str(ROOT / "outputs/rhfb_spherical/production"))
    parser.add_argument("--fortran-root", default=str(ROOT / "outputs/fortran_reference/runs"))
    parser.add_argument("--nuclei", nargs="+", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs/rhfb_spherical/comparison.json"))
    args = parser.parse_args()
    rows = []
    for nucleus in args.nuclei:
        nn_dir, ft_path = Path(args.nn_root) / nucleus, Path(args.fortran_root) / nucleus / "parsed.json"
        if not ft_path.exists():
            # Heavy nuclei: KMX=(8,8) build on the remote host (O16/Ca40/Ca48 agree
            # with the local KMX=(5,4) build to 1e-6 MeV per nucleon).
            ft_path = ROOT / "outputs/fortran_reference/runs_kmx88_remote" / nucleus / "parsed.json"
        core = _core_reference(nucleus)
        if not (nn_dir / "summary.json").exists():
            print(f"skip {nucleus}: missing neural summary")
            continue
        if ft_path.exists():
            rows.append(compare(nucleus, nn_dir, _fortran_reference(ft_path), core))
        elif core is not None:
            rows.append(compare(nucleus, nn_dir, core))
        else:
            print(f"skip {nucleus}: no reference")
    Path(args.out).write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['nucleus']:5s} E/A nn={r['nn_E_per_A']:.5f} ft={r['ft_E_per_A']:.5f} "
              f"d={r['dE_per_A']:+.5f} | Epair_n nn={r['nn_E_pair_n']:.3f} ft={r['ft_E_pair_n']:.3f} "
              f"| lam_n nn={r['nn_lambda_n']:.3f} ref={r['ft_lambda_n']} "
              f"| r_ch nn={r['nn_r_ch']:.4f} ft={r['ft_r_ch']:.4f} | levelMAE={r['level_mae_mev']}")


if __name__ == "__main__":
    main()
