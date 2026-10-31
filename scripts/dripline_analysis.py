#!/usr/bin/env python3
"""Two-neutron separation energies and drip lines along O, Sn and Pb chains.

S2n(Z,N) = E(Z,N-2) - E(Z,N) with E the total energy without centre-of-mass
correction (the correction largely cancels in the difference).  The neutron
drip line is the last even-N nucleus before S2n first becomes negative.
Within each chain all NN states share one canonical space.  Reads only saved
summaries; no calculation is run.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/rhfb_spherical"
FORTRAN = [ROOT / "outputs/fortran_reference/runs_kmx88_remote", ROOT / "outputs/fortran_reference/runs"]
CHAINS = {"O": (8, range(8, 24, 2)), "Sn": (50, range(82, 132, 2)), "Pb": (82, range(126, 190, 2))}
# Chain members computed outside the drip-line runs with the same canonical space.
BASE = {"O16": R / "final/O16", "O18": R / "final/O18"}
NN, REF = "#2a78d6", "#eb6834"


def nn_energy(name):
    for directory in (R / "dripline_eval" / name, BASE.get(name)):
        if directory is not None and (directory / "summary.json").exists():
            s = json.loads((directory / "summary.json").read_text())
            return {"E": s["energy_total_no_com_mev"], "lambda_n": s["chemical_potentials_mev"]["n"],
                    "E_pair_n": s["energy_terms_mev"]["E_pair_n"],
                    "r_ch": s["radii_fm"]["charge_radius_no_com"],
                    "leak": s["no_sea"].get("sea_leakage_self_consistent")}
    return None


def ref_energy(name):
    for root in FORTRAN:
        path = root / name / "parsed.json"
        if path.exists():
            p = json.loads(path.read_text())
            if not p.get("converged", True):
                return None
            return {"E": p["energy"]["E_total_no_cm_MeV"], "lambda_n": p["lambda_MeV"]["n"],
                    "E_pair_n": p["energy"]["E_pair_MeV"]["n"]}
    return None


def chain_table(element):
    z, ns = CHAINS[element]
    rows = []
    for n in ns:
        name = f"{element}{z + n}"
        rows.append({"name": name, "Z": z, "N": n, "nn": nn_energy(name), "ref": ref_energy(name)})
    for i, row in enumerate(rows):
        for key in ("nn", "ref"):
            prev = rows[i - 1][key] if i > 0 else None
            row[f"S2n_{key}"] = (prev["E"] - row[key]["E"]) if (prev and row[key]) else None
    return rows


def drip(rows, key):
    """Last bound even-N nucleus, or None if S2n never turns negative in the data."""
    last = None
    for row in rows:
        s2n = row[f"S2n_{key}"]
        if s2n is None:
            continue
        if s2n < 0:
            return last, row["name"]
        last = row["name"]
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(R / "dripline_analysis.json"))
    parser.add_argument("--figure", default=str(ROOT / "paper/figures/s2n.pdf"))
    args = parser.parse_args()
    result = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex="col")
    for col, element in enumerate(CHAINS):
        rows = chain_table(element)
        result[element] = {"rows": rows, "drip_nn": drip(rows, "nn"), "drip_ref": drip(rows, "ref")}
        for key, color, marker, label in (("ref", REF, "s", "Reference RHFB"), ("nn", NN, "o", "NN-RHFB")):
            pts = [(r["N"], r[f"S2n_{key}"]) for r in rows if r[f"S2n_{key}"] is not None]
            if pts:
                axes[0, col].plot(*zip(*pts), marker=marker, ms=4 if key == "nn" else 5,
                                  mfc=color if key == "nn" else "none", color=color, lw=1.2, label=label)
            lam = [(r["N"], r[key]["lambda_n"]) for r in rows if r[key] is not None]
            if lam:
                axes[1, col].plot(*zip(*lam), marker=marker, ms=4 if key == "nn" else 5,
                                  mfc=color if key == "nn" else "none", color=color, lw=1.2)
        for ax in axes[:, col]:
            ax.axhline(0.0, color="#6b6a64", lw=0.8)
            ax.grid(True, color="#e4e3de", lw=0.5)
        axes[0, col].set_title(element, fontsize=8)
        axes[1, col].set_xlabel("N")
    axes[0, 0].set_ylabel(r"$S_{2n}$ (MeV)")
    axes[1, 0].set_ylabel(r"$\lambda_n$ (MeV)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.figure)
    fig.savefig(Path(args.figure).with_suffix(".png"), dpi=200)
    Path(args.out).write_text(json.dumps(result, indent=2))
    for element, data in result.items():
        done = sum(1 for r in data["rows"] if r["nn"])
        print(f"{element}: NN nuclei {done}/{len(data['rows'])}, "
              f"drip NN {data['drip_nn']}, drip ref {data['drip_ref']}")


if __name__ == "__main__":
    main()
