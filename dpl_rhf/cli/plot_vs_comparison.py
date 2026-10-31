from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from dpl_rhf.functionals.pkdd_action import PKDDParameters, rmf_parameters


def recover_vector_scalar(
    stack: np.ndarray, params: PKDDParameters | None = None
) -> dict[str, np.ndarray]:
    p = params or PKDDParameters()
    result = {}
    for species, offset, mass in (("n", 0, p.mass_n), ("p", 2, p.mass_p)):
        vps = stack[offset]
        vms = stack[offset + 1]
        result[f"V_{species}"] = 0.5 * p.hbar_c * (vps + vms) + mass
        result[f"S_{species}"] = 0.5 * p.hbar_c * (vps - vms) - mass
    return result


def load_hamiltonian_stack(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    if "stack" in archive.files:
        return np.asarray(archive["stack"])
    channels = ("vps_n", "vms_n", "vps_p", "vms_p")
    if all(name in archive.files for name in channels):
        return np.stack([np.asarray(archive[name]) for name in channels])
    raise ValueError(f"archive must contain stack or named channels {channels}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DPL and Core-1204 scalar/vector RMF potentials")
    parser.add_argument("--dpl", required=True, help="DPL dpl_hamiltonian.npz")
    parser.add_argument("--scf", required=True, help="Core-1204 profile npz containing stack")
    parser.add_argument("--out", required=True)
    parser.add_argument("--r-max", type=float, default=10.0)
    parser.add_argument("--model", default="PKDD")
    parser.add_argument("--nucleus", default="O16")
    args = parser.parse_args()

    dpl_archive = np.load(args.dpl)
    scf_archive = np.load(args.scf)
    r = np.asarray(dpl_archive["r"])
    if not np.allclose(r, scf_archive["r"]):
        raise ValueError("DPL and SCF radial grids differ")
    params = rmf_parameters(args.model)
    dpl = recover_vector_scalar(load_hamiltonian_stack(dpl_archive), params)
    scf = recover_vector_scalar(load_hamiltonian_stack(scf_archive), params)
    mask = (r >= 0.1) & (r <= args.r_max)

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True, constrained_layout=True)
    colors = {"V": "#1769aa", "S": "#c43d3d"}
    for column, species in enumerate(("n", "p")):
        ax = axes[0, column]
        for field in ("V", "S"):
            key = f"{field}_{species}"
            ax.plot(r[mask], dpl[key][mask], color=colors[field], linewidth=2.1, label=f"DPL {field}")
            ax.plot(r[mask], scf[key][mask], color=colors[field], linewidth=1.8, linestyle="--", label=f"SCF {field}")
        ax.axhline(0.0, color="#666666", linewidth=0.7)
        ax.set_title("Neutron potentials" if species == "n" else "Proton potentials")
        ax.set_ylabel("Potential (MeV)")
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.legend(frameon=False, ncol=2, fontsize=9)

        delta_ax = axes[1, column]
        for field in ("V", "S"):
            key = f"{field}_{species}"
            delta_ax.plot(r[mask], (dpl[key] - scf[key])[mask], color=colors[field], linewidth=2.0, label=f"Delta {field}")
        delta_ax.axhline(0.0, color="#333333", linewidth=0.8)
        delta_ax.set_xlabel("r (fm)")
        delta_ax.set_ylabel("DPL - SCF (MeV)")
        delta_ax.grid(True, color="#dddddd", linewidth=0.6)
        delta_ax.legend(frameon=False, ncol=2, fontsize=9)

    fig.suptitle(
        f"{params.model} {args.nucleus}: scalar S and vector V potentials",
        fontsize=14,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    rows = {"r": r}
    for key in dpl:
        rows[f"dpl_{key}"] = dpl[key]
        rows[f"scf_{key}"] = scf[key]
        rows[f"delta_{key}"] = dpl[key] - scf[key]
    np.savez(output.with_name(output.stem + "_data.npz"), **rows)


if __name__ == "__main__":
    main()
