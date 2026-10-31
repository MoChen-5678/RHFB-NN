from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PKO2 RHF local and nonlocal outputs")
    parser.add_argument("result_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    output = Path(args.out) if args.out else result_dir / "rhf_diagnostics.png"
    local = np.load(result_dir / "local_observables.npz")
    nonlocal_data = np.load(result_dir / "nonlocal_hamiltonians.npz")
    r = local["r"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(r, local["component_S_direct"] * 197.328284, label="S direct")
    axes[0, 0].plot(r, local["component_V0_direct"] * 197.328284, label="V0 direct")
    axes[0, 0].set(xlabel="r (fm)", ylabel="Self-energy (MeV)", title="Local Hartree fields")
    axes[0, 0].legend()
    axes[0, 1].plot(r, local["rho_v_n"], label="neutron")
    axes[0, 1].plot(r, local["rho_v_p"], label="proton")
    axes[0, 1].set(xlabel="r (fm)", ylabel=r"$\rho_v$ (fm$^{-3}$)", title="Vector densities")
    axes[0, 1].legend()
    fock_keys = sorted(key for key in nonlocal_data.files if key.startswith("h_fock_"))
    key = fock_keys[0]
    matrix = nonlocal_data[key]
    image = axes[1, 0].imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm")
    axes[1, 0].set_title(key.replace("h_fock_", "Fock h: "))
    fig.colorbar(image, ax=axes[1, 0], label="MeV")
    diagonal = np.diag(matrix)
    half = diagonal.size // 2
    axes[1, 1].plot(r[1:-1], diagonal[:half], label="GG diagonal")
    axes[1, 1].plot(r[1:-1], diagonal[half:], label="FF diagonal")
    axes[1, 1].set(xlabel="r (fm)", ylabel="MeV", title="State-block Fock diagonal")
    axes[1, 1].legend()
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
